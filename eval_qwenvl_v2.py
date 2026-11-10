"""
eval_qwenvl_v2.py
==================
Score Qwen2.5-VL on the released question--answer pairs, with or without the
LoRA adapter, using the release's own prompts and the benchmark's metrics.

Running the *unadapted* model here rather than reusing the Ollama row of the
zero-shot table is what makes the before/after comparison honest: Ollama serves
a 4-bit quantised build while fine-tuning requires bf16, and attributing a
precision difference to adaptation would overstate the result. Precision,
resolution, prompts, sampling and answer extraction are held identical across
both conditions; the adapter is the only thing that changes.

Evaluation items are drawn from the held-out test split with the same
class-balanced procedure the benchmark uses, so chance is exactly 1/k.

    python eval_qwenvl_v2.py --qa <vqa_v2> --root <dataset root> --out b.json
    python eval_qwenvl_v2.py --qa ... --adapter <dir> --out a.json
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from eval_vlm_v2 import (LABELS, NUMERIC, parse_label, pick_frames,
                         load_split, train_median_spacing, sample)
from eval_vlm_zeroshot import extract_number

logger = logging.getLogger("qweneval2")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] qweneval2: %(message)s")

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
W, H = 640, 360


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qa", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--raw", default=None)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--clip-images", type=int, default=4)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--max-pixels", type=int, default=W * H)
    p.add_argument("--split", default="test")
    p.add_argument("--types", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=128 * 28 * 28,
                                         max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        logger.info("loaded adapter %s", args.adapter)
    model.eval()

    qa_dir, root = Path(args.qa), Path(args.root)
    by = load_split(qa_dir, args.split)
    wanted = ([t.strip() for t in args.types.split(",")] if args.types
              else sorted(by))
    median = train_median_spacing(qa_dir)
    rng = random.Random(args.seed)
    raw = open(args.raw, "a") if args.raw else None

    out = {"model": MODEL_ID, "adapter": args.adapter, "split": args.split,
           "n_per_type": args.n, "clip_images": args.clip_images,
           "precision": "bfloat16", "resolution": [W, H],
           "train_median_spacing_m": median,
           "scores_released_pairs_verbatim": True, "types": []}

    @torch.no_grad()
    def generate(question: str, imgs: List[Image.Image]) -> str:
        content = [{"type": "image"} for _ in imgs]
        content.append({"type": "text", "text": question})
        text = proc.apply_chat_template([{"role": "user", "content": content}],
                                        tokenize=False,
                                        add_generation_prompt=True)
        batch = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
        gen = model.generate(**batch, max_new_tokens=args.max_new,
                             do_sample=False)
        return proc.tokenizer.decode(gen[0, batch["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

    def images_for(rec: Dict) -> List[Image.Image]:
        return [Image.open(root / p).convert("RGB").resize((W, H),
                                                           Image.BILINEAR)
                for p in pick_frames(rec["images"], args.clip_images)]

    for t in wanted:
        if t not in by:
            continue
        sel = sample(by[t], t, args.n, rng)
        t0 = time.time()

        if t in NUMERIC:
            preds, gts, unparsed = [], [], 0
            for r in sel:
                try:
                    imgs = images_for(r)
                except (FileNotFoundError, OSError):
                    continue
                txt = generate(r["question"], imgs)
                v = extract_number(txt)
                if raw:
                    raw.write(json.dumps({"qa_type": t, "gt": r["answer_value"],
                                          "parsed": v, "text": txt}) + "\n")
                if v is None or not (0.0 <= v <= 300.0):
                    unparsed += 1
                    continue
                preds.append(v); gts.append(float(r["answer_value"]))
            if not preds:
                out["types"].append({"qa_type": t, "n": 0,
                                     "n_unparsed": unparsed})
                continue
            pa, ga = np.array(preds), np.array(gts)
            e = {"qa_type": t, "kind": "numeric", "n": len(preds),
                 "n_unparsed": unparsed,
                 "mae_m": round(float(np.abs(pa - ga).mean()), 3),
                 "within_5m": round(float((np.abs(pa - ga) <= 5).mean()), 3),
                 "seconds": round(time.time() - t0, 1)}
            if median is not None:
                e["baseline_median_mae_m"] = round(
                    float(np.abs(median - ga).mean()), 3)
            out["types"].append(e)
            logger.info("%s -> %s", t, e)
            json.dump(out, open(args.out, "w"), indent=2)
            continue

        labels = LABELS[t]
        ok = tot = unparsed = 0
        preds_c: collections.Counter = collections.Counter()
        conf: Dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter)
        for r in sel:
            try:
                imgs = images_for(r)
            except (FileNotFoundError, OSError):
                continue
            txt = generate(r["question"], imgs)
            g = parse_label(txt, labels)
            if raw:
                raw.write(json.dumps({"qa_type": t, "gt": r["answer"],
                                      "parsed": g, "text": txt}) + "\n")
            if g is None:
                unparsed += 1
                continue
            tot += 1; preds_c[g] += 1; conf[r["answer"]][g] += 1
            ok += int(g == r["answer"])
        if not tot:
            out["types"].append({"qa_type": t, "n": 0, "n_unparsed": unparsed})
            continue
        recalls = [conf[c][c] / sum(conf[c].values()) for c in conf
                   if sum(conf[c].values())]
        top, topn = preds_c.most_common(1)[0]
        e = {"qa_type": t, "kind": "categorical", "n": tot,
             "n_unparsed": unparsed, "n_classes": len(labels),
             "accuracy": round(ok / tot, 3),
             "balanced_accuracy": round(float(np.mean(recalls)), 3),
             "chance": round(1.0 / len(labels), 3),
             "modal_pred": top, "modal_pred_share": round(topn / tot, 3),
             "pred_distribution": dict(preds_c.most_common()),
             "seconds": round(time.time() - t0, 1)}
        out["types"].append(e)
        logger.info("%s -> %s", t, e)
        json.dump(out, open(args.out, "w"), indent=2)

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n=== Qwen2.5-VL {'+ adapter' if args.adapter else '(base)'} ===")
    for e in out["types"]:
        if not e.get("n"):
            print(f"{e['qa_type']:<22} no usable answers")
        elif e["kind"] == "numeric":
            print(f"{e['qa_type']:<22} MAE {e['mae_m']:>7.2f} m  "
                  f"base {e.get('baseline_median_mae_m', float('nan')):.2f} m")
        else:
            print(f"{e['qa_type']:<22} bal.acc {e['balanced_accuracy']:.3f}  "
                  f"chance {e['chance']:.3f}  modal {e['modal_pred']}"
                  f"({e['modal_pred_share']:.2f})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
