"""
build_vlm_ft_data_v2.py
========================
Assemble the supervised corpus for adapting a vision--language model to the
RELEASED question--answer pairs.

Why this replaces build_vlm_ft_data.py
--------------------------------------
The earlier corpus was built from the old benchmark's probe prompts, which were
neither the released supervision nor what the current benchmark asks. With the
benchmark now scoring the released pairs verbatim, adaptation must train on
those same pairs, or the study measures a third task that appears nowhere in
the release. Prompts and answers here are taken from qa_train.jsonl unchanged;
nothing is re-worded.

Sampling
--------
Categorical types are drawn class-balanced. Training on the natural
distribution would teach the model to answer \"SAFE\" unconditionally -- which
is precisely the failure the zero-shot models already exhibit -- so balance is
imposed at the source rather than corrected at evaluation time.

Image cache
-----------
Images are deduplicated by path before caching. The three frame-level question
types ask different things about the *same* frame, and the five clip-level
types share the same clip, so a naive cache would store each image three to
five times; deduplication reduces the cache by roughly a factor of three.
Clip questions are cached at the same `--clip-images` evenly spaced frames the
benchmark presents, so training and evaluation see an identical view.

    python build_vlm_ft_data_v2.py --qa <vqa_v2 dir> --root <dataset root> \\
        --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from eval_vlm_v2 import pick_frames, LABELS, NUMERIC

logger = logging.getLogger("ftdata2")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] ftdata2: %(message)s")

W, H = 640, 360

FRAME_TYPES = ["spacing_estimation", "lead_presence", "visual_conditions"]
CLIP_TYPES = ["gap_trend", "relative_motion", "safety_ttc", "ego_action",
              "scenario_temporal"]


def load(qa_dir: Path, split: str) -> Dict[str, List[Dict]]:
    by: Dict[str, List[Dict]] = collections.defaultdict(list)
    with open(qa_dir / f"qa_{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            by[r["qa_type"]].append(r)
    return by


def balanced(recs: List[Dict], qa_type: str, n: int,
             rng: random.Random) -> List[Dict]:
    """
    Exactly equal counts per class, not merely a per-class cap.

    Taking min(budget, available) independently per class leaves the result as
    skewed as the source wherever a class is starved: only 2.0 % of training
    clips are UNSAFE and 4.6 % of frames lack a lead vehicle, so a per-class
    cap would still hand the model a corpus it can fit by answering SAFE and
    YES unconditionally -- the exact zero-shot failure this study is meant to
    correct. Equalising costs sample count on those types (safety falls to
    twice its 251 available UNSAFE clips) and that trade is deliberate: a
    smaller balanced set teaches a decision, a larger skewed one teaches a
    prior. Achieved counts are reported so the cost is visible.
    """
    if qa_type in NUMERIC:
        return rng.sample(recs, min(n, len(recs)))
    by: Dict[str, List[Dict]] = collections.defaultdict(list)
    for r in recs:
        by[r["answer"]].append(r)
    per = max(1, min(n // max(len(by), 1), min(len(v) for v in by.values())))
    sel = [x for v in by.values() for x in rng.sample(v, per)]
    rng.shuffle(sel)
    return sel


def cache_images(paths: List[str], root: Path, out: Path, tag: str,
                 workers: int) -> Tuple[np.ndarray, Dict[str, int]]:
    """Cache each unique path once; return the array and a path->index map."""
    index = {p: i for i, p in enumerate(paths)}
    arr = np.lib.format.open_memmap(out / f"{tag}_images.npy", mode="w+",
                                    dtype=np.uint8,
                                    shape=(len(paths), H, W, 3))
    bad = 0

    def work(item: Tuple[int, str]):
        i, rel = item
        try:
            im = Image.open(root / rel).convert("RGB").resize(
                (W, H), Image.BILINEAR)
        except (FileNotFoundError, OSError):
            return i, None
        return i, np.asarray(im, dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, (i, a) in enumerate(ex.map(work, list(enumerate(paths)),
                                          chunksize=32)):
            if a is None:
                bad += 1
                continue
            arr[i] = a
            if (k + 1) % 2000 == 0:
                logger.info("  cached %d/%d", k + 1, len(paths))
    arr.flush()
    logger.info("%s: %d images cached (%d unreadable)", tag, len(paths), bad)
    return arr, index


def build(by: Dict[str, List[Dict]], n_frame: int, n_clip: int,
          clip_images: int, rng: random.Random) -> Tuple[List[Dict], List[str]]:
    samples: List[Dict] = []
    paths: List[str] = []
    seen: Dict[str, int] = {}

    def idx_of(rel: str) -> int:
        if rel not in seen:
            seen[rel] = len(paths)
            paths.append(rel)
        return seen[rel]

    for t in FRAME_TYPES + CLIP_TYPES:
        if t not in by:
            continue
        n = n_frame if t in FRAME_TYPES else n_clip
        for r in balanced(by[t], t, n, rng):
            imgs = pick_frames(r["images"], clip_images)
            samples.append({
                "qa_type": t,
                "question": r["question"],
                "answer": r["answer"],
                "answer_value": r.get("answer_value"),
                "img": [idx_of(p) for p in imgs],
                "scene_id": r.get("scene_id"),
            })
    rng.shuffle(samples)
    return samples, paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qa", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--clip-images", type=int, default=4)
    p.add_argument("--n-frame", type=int, default=5000,
                   help="samples per frame-level type")
    p.add_argument("--n-clip", type=int, default=2500,
                   help="samples per clip-level type")
    p.add_argument("--n-val-frame", type=int, default=300)
    p.add_argument("--n-val-clip", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    qa, root, out = Path(args.qa), Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for tag, split, nf, nc in (("train", "train", args.n_frame, args.n_clip),
                               ("val", "val", args.n_val_frame,
                                args.n_val_clip)):
        by = load(qa, split)
        samples, paths = build(by, nf, nc, args.clip_images, rng)
        logger.info("%s: %d samples, %d unique images (dedup factor %.2fx)",
                    tag, len(samples), len(paths),
                    sum(len(s["img"]) for s in samples) / max(len(paths), 1))
        cache_images(paths, root, out, tag, args.workers)
        json.dump({"samples": samples, "paths": paths,
                   "clip_images": args.clip_images,
                   "resolution": [W, H], "seed": args.seed,
                   "scores_released_pairs_verbatim": True},
                  open(out / f"{tag}_samples.json", "w"))

        counts = collections.Counter(s["qa_type"] for s in samples)
        for k, v in sorted(counts.items()):
            print(f"  {tag:<6}{k:<22}{v:>7}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
