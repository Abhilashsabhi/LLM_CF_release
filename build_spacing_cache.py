"""
build_spacing_cache.py
=======================
Decode, resize and cache front-camera frames for the supervised spacing
regressor.

Why a cache. The released imagery is 1920x1080 JPEG on an external volume and
decodes at roughly 31 frames per second per process, so a single training epoch
over the frames used here would spend ~27 minutes on I/O and repeat that cost
every epoch. One parallel pass writes a uint8 memmap that subsequent epochs
read at memory speed. The cache is derived data and is not part of the release.

Frame subsampling. Consecutive frames at 10 Hz are nearly identical, so using
all 300 frames of a scene inflates the apparent sample size without adding
information. We take every `stride`-th frame. Splits are inherited unchanged
from the dataset's scene-level splits, so no scene contributes to more than one
split and the comparison against the zero-shot benchmark stays like-for-like.

    python build_spacing_cache.py --dataset <root> --out <cache dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("cache")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] cache: %(message)s")

W, H = 320, 180          # 16:9, preserves the aspect ratio of the source


def _load(args: Tuple[int, str]):
    """Decode one frame, or report it missing.

    A small number of scenes are missing individual camera frames that the
    index nevertheless lists (see the integrity note in the manuscript). We
    skip those frames and count them rather than aborting, but we never
    substitute a placeholder image: a black frame paired with a real gap label
    would be training on a lie.
    """
    i, path = args
    try:
        im = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
    except (FileNotFoundError, OSError):
        return i, None
    return i, np.asarray(im, dtype=np.uint8)


def build(records: List[dict], out: Path, tag: str, workers: int) -> None:
    """Write images.npy (N,H,W,3 uint8), spacing.npy (N,) and meta.json."""
    n = len(records)
    tmp = np.lib.format.open_memmap(out / f"{tag}_tmp.npy", mode="w+",
                                    dtype=np.uint8, shape=(n, H, W, 3))
    jobs = [(i, str(r["image"])) for i, r in enumerate(records)]
    ok = np.zeros(n, dtype=bool)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, arr in ex.map(_load, jobs, chunksize=64):
            if arr is not None:
                tmp[i] = arr
                ok[i] = True
            done += 1
            if done % 5000 == 0:
                logger.info("  %s %d/%d (%d missing)", tag, done, n,
                            done - int(ok[:done].sum()))
    tmp.flush()

    keep = np.flatnonzero(ok)
    imgs = np.lib.format.open_memmap(out / f"{tag}_images.npy", mode="w+",
                                     dtype=np.uint8, shape=(len(keep), H, W, 3))
    for j, i in enumerate(keep):
        imgs[j] = tmp[i]
    imgs.flush()
    del tmp, imgs
    (out / f"{tag}_tmp.npy").unlink()

    np.save(out / f"{tag}_spacing.npy",
            np.array([records[i]["spacing"] for i in keep], dtype=np.float32))
    missing = [str(records[i]["image"]) for i in np.flatnonzero(~ok)]
    with open(out / f"{tag}_meta.json", "w") as f:
        json.dump({"n": len(keep), "n_requested": n, "n_missing": len(missing),
                   "h": H, "w": W, "missing": missing[:200],
                   "scenes": sorted({records[i]["scene_id"] for i in keep})}, f)
    logger.info("%s: %d frames kept, %d missing, %d scenes", tag, len(keep),
                len(missing), len({records[i]["scene_id"] for i in keep}))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--train-stride", type=int, default=5)
    p.add_argument("--val-stride", type=int, default=15)
    p.add_argument("--test-stride", type=int, default=5)
    p.add_argument("--probe-n", type=int, default=400)
    p.add_argument("--probe-seed", type=int, default=42)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eval_vlm_zeroshot as V

    dataset, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for split, stride in (("train", args.train_stride),
                          ("val", args.val_stride),
                          ("test", args.test_stride)):
        pool = V.load_pool(dataset, split)
        sub = [r for r in pool if r["frame_idx"] % stride == 0]
        logger.info("%s: %d of %d frames (stride %d)", split, len(sub), len(pool), stride)
        build(sub, out, split, args.workers)

        if split == "test":
            # The EXACT frames the zero-shot models were scored on, drawn the
            # same way eval_vlm_zeroshot draws them (same pool order, same
            # seed, same first call on that generator). Caching them lets the
            # supervised model be reported on an identical sample, so the two
            # benchmarks differ only in the model.
            probes = random.Random(args.probe_seed).sample(
                pool, min(args.probe_n, len(pool)))
            build(probes, out, "probes", args.workers)

    logger.info("cache written to %s", out)


if __name__ == "__main__":
    main()
