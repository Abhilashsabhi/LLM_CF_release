"""
finetune_qwenvl_v2.py
======================
LoRA adaptation of Qwen2.5-VL to the RELEASED question--answer pairs.

Differences from finetune_qwenvl.py
-----------------------------------
*Prompts come from the data, not from this file.* The earlier trainer hard-coded
three prompt strings copied from the old benchmark. Those were neither the
released supervision nor what the current benchmark asks, so the adaptation
study measured a task appearing nowhere in the release. Here the question and
answer are taken verbatim from the sample record, which is itself a verbatim
copy of the released pair. There is no wording for a copy to drift from.

*Multi-image examples.* Five of the eight released types are clip-level and
carry several frames, so the collator emits one image placeholder per frame and
passes the whole list to the processor. Single-image examples are the same code
path with a list of length one.

The vision tower stays frozen and only the language tower is adapted, so the
comparison isolates whether the required information survives the visual
encoder and merely needs to be read out, rather than confounding that with
re-learning perception.

    python finetune_qwenvl_v2.py --data <ft_v2 dir> --out <dir> --epochs 1
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("qwenft2")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] qwenft2: %(message)s")

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


class Collator:
    """One training example: N images + the released question, loss on answer.

    The label mask is placed by tokenising the answer suffix and masking
    everything before it. We assert on the first example that the unmasked span
    decodes back to exactly the intended answer: a silently misplaced mask
    trains the model on its own prompt and presents as a bad learning rate.
    """

    def __init__(self, processor, images: np.ndarray, checked: List[bool]):
        self.p = processor
        self.images = images
        self.checked = checked

    def __call__(self, samples: List[Dict]):
        s = samples[0]                      # batch size 1; accumulate instead
        idxs = s["img"] if isinstance(s["img"], list) else [s["img"]]
        imgs = [Image.fromarray(np.asarray(self.images[i])) for i in idxs]
        content = [{"type": "image"} for _ in imgs]
        content.append({"type": "text", "text": s["question"]})
        msg = [{"role": "user", "content": content}]
        head = self.p.apply_chat_template(msg, tokenize=False,
                                          add_generation_prompt=True)
        ans = str(s["answer"]) + "<|im_end|>"
        batch = self.p(text=[head + ans], images=imgs, return_tensors="pt")

        ans_ids = self.p.tokenizer(ans, add_special_tokens=False)["input_ids"]
        labels = batch["input_ids"].clone()
        labels[:, :-len(ans_ids)] = -100
        if not self.checked[0]:
            got = self.p.tokenizer.decode(batch["input_ids"][0, -len(ans_ids):])
            assert got == ans, f"label mask misaligned: {got!r} != {ans!r}"
            logger.info("label mask verified on first example: %r", got)
            self.checked[0] = True
        batch["labels"] = labels
        return batch


def load_split(data: Path, tag: str):
    meta = json.load(open(data / f"{tag}_samples.json"))
    imgs = np.load(data / f"{tag}_images.npy", mmap_mode="r")
    return meta["samples"], imgs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--val-max", type=int, default=400)
    p.add_argument("--max-pixels", type=int, default=640 * 360,
                   help="per-image visual token budget, held identical to "
                        "the evaluation setting")
    p.add_argument("--ckpt-epochs", type=float, nargs="*",
                   default=[0.25, 0.5, 1.0])
    args = p.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                              get_cosine_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model

    proc = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=128 * 28 * 28,
                                         max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model.config.use_cache = False

    for n, prm in model.named_parameters():       # freeze the vision tower
        if "visual" in n:
            prm.requires_grad = False
    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        exclude_modules=r".*visual.*")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    samples, images = load_split(data, "train")
    val_samples, val_images = load_split(data, "val")
    if len(val_samples) > args.val_max:
        val_samples = random.Random(args.seed).sample(val_samples, args.val_max)
    coll = Collator(proc, images, [False])
    val_coll = Collator(proc, val_images, [True])

    n_steps = int(len(samples) * args.epochs) // args.accum
    if args.max_steps > 0:
        n_steps = min(n_steps, args.max_steps)
    logger.info("%d train samples | %d val | %d optimiser steps (accum %d)",
                len(samples), len(val_samples), n_steps, args.accum)

    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * n_steps), n_steps)

    steps_per_epoch = max(1, len(samples) // args.accum)
    ckpts = {min(n_steps, max(1, int(round(e * steps_per_epoch)))): e
             for e in args.ckpt_epochs if e * steps_per_epoch <= n_steps + 1}
    logger.info("checkpoints at steps %s", sorted(ckpts))

    @torch.no_grad()
    def val_loss() -> float:
        model.eval()
        tot = 0.0
        for vs in val_samples:
            b = val_coll([vs])
            b = {k: (v.to("cuda") if hasattr(v, "to") else v)
                 for k, v in b.items()}
            tot += float(model(**b).loss)
        model.train()
        return tot / len(val_samples)

    order = list(range(len(samples)))
    random.Random(args.seed).shuffle(order)
    it = iter(order)
    hist, t0, run = [], time.time(), 0.0
    model.train()
    for step in range(1, n_steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            try:
                i = next(it)
            except StopIteration:
                random.Random(args.seed + step).shuffle(order)
                it = iter(order); i = next(it)
            batch = coll([samples[i]])
            batch = {k: (v.to("cuda") if hasattr(v, "to") else v)
                     for k, v in batch.items()}
            loss = model(**batch).loss / args.accum
            loss.backward()
            run += loss.item()
        torch.nn.utils.clip_grad_norm_(
            [q for q in model.parameters() if q.requires_grad], 1.0)
        opt.step(); sched.step()
        if step in ckpts:
            ep = ckpts[step]
            vl = val_loss()
            tag = f"epoch{ep:g}".replace(".", "p")
            model.save_pretrained(out / f"adapter_{tag}")
            hist.append({"step": step, "epoch": ep, "val_loss": vl,
                         "checkpoint": tag})
            logger.info("CHECKPOINT %s at step %d: val loss %.4f", tag, step, vl)
            json.dump({"history": hist, "args": vars(args)},
                      open(out / "train_log.json", "w"), indent=2)
        if step % args.log_every == 0:
            el = time.time() - t0
            logger.info("step %4d/%d  loss %.4f  %.2f s/step  eta %.1f min  "
                        "peak %.1f GB", step, n_steps, run / args.log_every,
                        el / step, (n_steps - step) * el / step / 60,
                        torch.cuda.max_memory_allocated() / 1e9)
            run = 0.0

    model.save_pretrained(out / "adapter_final")
    json.dump({"history": hist, "args": vars(args)},
              open(out / "train_log.json", "w"), indent=2)
    logger.info("done in %.1f min -> %s", (time.time() - t0) / 60, out)


if __name__ == "__main__":
    main()
