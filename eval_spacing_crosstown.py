"""
eval_spacing_crosstown.py
==========================
Train monocular spacing on some CARLA towns, test on towns never seen.

The released splits are scene-level, which stops identical frames crossing
splits but leaves every one of the eight towns present in all three. A model
can therefore learn recurring road geometry, building facades and lane
markings, and a scene-level split will not detect it. This retrains the same
ResNet-18 on a subset of towns and evaluates on the held-out ones, so the
transfer is genuinely out-of-town.

Two conditions are reported for each holdout so that the comparison is like
for like: the cross-town model, and a within-town model trained on the same
number of frames drawn from the training towns and evaluated on frames of
those same towns. Comparing a cross-town result against the paper's main
number would confound the town shift with the change in training-set size.

Intervals are bootstrapped over SCENES. Frames within a scene are
near-duplicates and a frame-level bootstrap would understate them.

    python eval_spacing_crosstown.py --cache <dir> --dataset <root> --out <json>
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger("crosstown")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] crosstown: %(message)s")

# Held out together: one urban pair and one motorway town, so the transfer
# covers both regimes rather than only the easier one.
HOLDOUTS = {
    "urban_T01_T02": ["Town01", "Town02"],
    "motorway_T06": ["Town06"],
}


def frame_scene_ids(dataset: Path, split: str, stride: int,
                    missing: List[str]) -> List[str]:
    """Scene id of every cached frame, reconstructed in the builder's order."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eval_vlm_zeroshot as V
    pool = V.load_pool(dataset, split)
    sub = [r for r in pool if r["frame_idx"] % stride == 0]
    # The cache recorded absolute paths under the dataset's former directory
    # name, so matching on the path string fails after the rename. Match on
    # (scene, frame) parsed from the basename instead, which is stable.
    miss = set()
    for m in missing:
        stem = Path(m).stem                      # <scene>-frame-<idx>-<sensor>
        parts = stem.split("-frame-")
        if len(parts) == 2:
            miss.add((parts[0], int(parts[1].split("-")[0])))
    return [r["scene_id"] for r in sub
            if (r["scene_id"], r["frame_idx"]) not in miss]


def cluster_boot(err: np.ndarray, scene: np.ndarray, n: int = 3000,
                 seed: int = 0) -> Tuple[float, float]:
    scenes = np.unique(scene)
    if len(scenes) < 3:
        return (float("nan"), float("nan"))
    by = {s: err[scene == s] for s in scenes}
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        pick = rng.choice(scenes, size=len(scenes), replace=True)
        out[i] = np.concatenate([by[s] for s in pick]).mean()
    return (round(float(np.percentile(out, 2.5)), 3),
            round(float(np.percentile(out, 97.5)), 3))


def train_eval(Xtr, Ytr, Xte, Yte, ste, epochs: int, seed: int,
               batch: int = 128, lr: float = 3e-4) -> Dict:
    import torch
    from torch import nn, optim
    from torchvision.models import resnet18

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    m = m.to(dev)
    opt = optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, len(Xtr) // batch))
    lossf = nn.L1Loss()

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def batches(X, Y, shuffle):
        idx = np.arange(len(X))
        if shuffle:
            np.random.shuffle(idx)
        for k in range(0, len(idx) - (batch if shuffle else 0), batch):
            b = np.sort(idx[k:k + batch])
            im = (np.asarray(X[b], np.float32) / 255.0 - MEAN) / STD
            yield (torch.from_numpy(im.transpose(0, 3, 1, 2)).to(dev),
                   torch.from_numpy(Y[b]).to(dev))

    for ep in range(epochs):
        m.train()
        for xb, yb in batches(Xtr, Ytr, True):
            if np.random.rand() < 0.5:                  # label-preserving flip
                xb = torch.flip(xb, dims=[3])
            loss = lossf(m(xb).squeeze(-1), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            try:
                sched.step()
            except ValueError:
                pass
        logger.info("    epoch %d/%d", ep + 1, epochs)

    m.eval()
    preds = []
    with torch.no_grad():
        for k in range(0, len(Xte), batch):
            im = (np.asarray(Xte[k:k + batch], np.float32) / 255.0 - MEAN) / STD
            t = torch.from_numpy(im.transpose(0, 3, 1, 2)).to(dev)
            preds.append(m(t).squeeze(-1).cpu().numpy())
    pr = np.concatenate(preds)
    err = np.abs(pr - Yte)
    return {"n_test_frames": int(len(err)),
            "n_test_scenes": int(len(np.unique(ste))),
            "mae": round(float(err.mean()), 3),
            "within_5m": round(float((err <= 5).mean()), 3),
            "ci95_scene_cluster": cluster_boot(err, ste)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train", type=int, default=24000)
    args = p.parse_args()

    cache, root = Path(args.cache), Path(args.dataset)
    meta = json.load(open(cache / "train_meta.json"))
    X = np.load(cache / "train_images.npy", mmap_mode="r")
    Y = np.load(cache / "train_spacing.npy").astype(np.float32)
    sid = np.array(frame_scene_ids(root, "train", 5, meta.get("missing", [])))
    assert len(sid) == len(Y) == len(X), (len(sid), len(Y), len(X))
    logger.info("%d cached training frames, %d scenes", len(Y), len(set(sid)))

    town_of = {}
    for sc in json.load(open(root / "infos" / "cf_infos_train.json"))["data"]:
        town_of[sc["scene_id"]] = sc["scene_info"].get("carla_map")
    town = np.array([town_of.get(s, "?") for s in sid])

    results: Dict = {"epochs": args.epochs, "seed": args.seed, "holdouts": {}}
    rng = np.random.default_rng(args.seed)

    for name, held in HOLDOUTS.items():
        te = np.isin(town, held)
        tr = ~te
        if te.sum() < 200 or tr.sum() < 200:
            continue
        # Match training-set size across the two conditions so the comparison
        # isolates the town shift rather than the amount of data.
        n_tr = min(args.max_train, int(tr.sum()))
        tr_idx = np.sort(rng.choice(np.flatnonzero(tr), n_tr, replace=False))

        # Within-town control: hold out scenes (not towns) from the SAME towns
        # the cross-town model trains on, with the same training size.
        pool = np.flatnonzero(tr)
        ctrl_scenes = np.unique(sid[pool])
        rng.shuffle(ctrl_scenes)
        k = max(1, int(0.18 * len(ctrl_scenes)))
        ctrl_te_scenes = set(ctrl_scenes[:k].tolist())
        ctrl_te = np.array([s in ctrl_te_scenes for s in sid]) & tr
        ctrl_tr_pool = np.flatnonzero(tr & ~ctrl_te)
        ctrl_tr = np.sort(rng.choice(ctrl_tr_pool,
                                     min(n_tr, len(ctrl_tr_pool)), replace=False))

        logger.info("%s: cross-town train %d / test %d (%d towns held out)",
                    name, len(tr_idx), int(te.sum()), len(held))
        cross = train_eval(X[tr_idx], Y[tr_idx], X[te], Y[te], sid[te],
                           args.epochs, args.seed)
        logger.info("  cross-town MAE %.3f %s", cross["mae"],
                    cross["ci95_scene_cluster"])

        logger.info("%s: within-town control train %d / test %d",
                    name, len(ctrl_tr), int(ctrl_te.sum()))
        within = train_eval(X[ctrl_tr], Y[ctrl_tr], X[ctrl_te], Y[ctrl_te],
                            sid[ctrl_te], args.epochs, args.seed)
        logger.info("  within-town MAE %.3f %s", within["mae"],
                    within["ci95_scene_cluster"])

        results["holdouts"][name] = {
            "held_out_towns": held, "n_train_frames": int(len(tr_idx)),
            "cross_town": cross, "within_town_control": within,
            "degradation_m": round(cross["mae"] - within["mae"], 3)}
        json.dump(results, open(args.out, "w"), indent=2)

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n{'holdout':<18}{'within-town':>14}{'cross-town':>14}{'degradation':>14}")
    print("-" * 60)
    for k, v in results["holdouts"].items():
        print(f"{k:<18}{v['within_town_control']['mae']:>13.2f}m"
              f"{v['cross_town']['mae']:>13.2f}m{v['degradation_m']:>13.2f}m")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
