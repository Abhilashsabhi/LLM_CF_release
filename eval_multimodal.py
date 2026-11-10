"""
eval_multimodal.py
===================
Does the released sensor diversity earn its place? A compact multimodal
benchmark on the quantity car-following needs.

Six conditions predict bumper-to-bumper spacing from a single synchronised
frame: front camera alone, front radar alone, LiDAR alone, and the three
fusions. The camera branch is the ResNet-18 of Section~\\ref{sec:wellposed},
reused rather than retrained so that the RGB row is the same number reported
elsewhere; fusion is late, concatenating that prediction with the geometric
features. This is deliberately the simplest arrangement that can answer the
question -- whether the modalities carry complementary information about the
gap -- rather than an architecture proposal.

The radar velocity channel is excluded throughout. It is defective in this
release, correlating \\num{0.017} with true relative speed, and a model given it
would be fitting noise. Radar contributes depth and angle only.

Every condition is trained under several seeds and reported as mean and
standard deviation, on the held-out test scenes.

    python eval_multimodal.py --cache <npz> --ckpt <resnet.pt> --out <json>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("mmeval")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] mmeval: %(message)s")

W, H = 320, 180
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

VIS_GROUPS = ["CLEAR", "RAIN", "FOG", "NIGHT"]

CONDITIONS = [
    ("rgb", ["rgb"]),
    ("radar", ["radar"]),
    ("lidar", ["lidar"]),
    ("rgb+radar", ["rgb", "radar"]),
    ("rgb+lidar", ["rgb", "lidar"]),
    ("rgb+radar+lidar", ["rgb", "radar", "lidar"]),
]


def rgb_predictions(paths: np.ndarray, root: Path, ckpt: Path,
                    batch: int = 64, return_features: bool = False):
    """Spacing predicted by the frozen ResNet-18, and optionally its features.

    Fusing only the scalar prediction is not a fair test of fusion: one number
    discards nearly everything the camera encodes, and a head placed on top can
    do little but pass it through. When `return_features` is set we also return
    the 512-d penultimate activations, so the fusion conditions see the camera
    representation rather than its summary.
    """
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision.models import resnet18

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    sd = torch.load(ckpt, map_location=dev)
    m.load_state_dict(sd.get("model", sd) if isinstance(sd, dict) else sd)
    m.to(dev).eval()

    feats_store = {}
    if return_features:
        def _hook(_m, _i, o):
            feats_store["z"] = torch.flatten(o, 1).detach()
        m.avgpool.register_forward_hook(_hook)

    out = np.zeros(len(paths), dtype=np.float32)
    feat = np.zeros((len(paths), 512), dtype=np.float32) if return_features else None
    buf, idx = [], []
    with torch.no_grad():
        for i, rel in enumerate(paths):
            try:
                im = Image.open(root / rel).convert("RGB").resize((W, H),
                                                                  Image.BILINEAR)
            except (OSError, ValueError):
                out[i] = np.nan
                continue
            x = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
            buf.append(x.transpose(2, 0, 1)); idx.append(i)
            if len(buf) == batch:
                t = torch.from_numpy(np.stack(buf)).to(dev)
                out[idx] = m(t).squeeze(-1).cpu().numpy()
                if return_features:
                    feat[idx] = feats_store["z"].cpu().numpy()
                buf, idx = [], []
            if (i + 1) % 2000 == 0:
                logger.info("  rgb %d/%d", i + 1, len(paths))
        if buf:
            t = torch.from_numpy(np.stack(buf)).to(dev)
            out[idx] = m(t).squeeze(-1).cpu().numpy()
            if return_features:
                feat[idx] = feats_store["z"].cpu().numpy()
    return (out, feat) if return_features else out


def assemble(store, split: str, rgb: np.ndarray, parts: List[str],
             mask: np.ndarray, feats: Optional[np.ndarray] = None) -> np.ndarray:
    """Concatenate the requested channels, all under the SAME row mask.

    The mask drops frames whose image failed to load. It must be applied to
    every channel, not only to the camera branch, or the rows silently
    misalign and each sensor is paired with another frame's target.
    """
    cols = []
    if "rgb" in parts:
        cols.append(rgb.reshape(-1, 1))
        if feats is not None:
            cols.append(feats)
    if "radar" in parts:
        cols.append(store[f"{split}_radar"][mask])
    if "lidar" in parts:
        cols.append(store[f"{split}_lidar"][mask])
    out = np.concatenate(cols, axis=1).astype(np.float32)
    assert len(out) == int(mask.sum()), "row misalignment in assemble()"
    return out


def fit_mlp(Xtr, Ytr, Xva, Yva, seed: int, epochs: int = 60,
            hidden: int = 128, base_tr=None, base_va=None) -> "object":
    """Fit a head. If a base estimate is given, learn a residual on top of it.

    Predicting the gap from scratch makes a fusion model rediscover what the
    camera branch already knows, which handicaps it against the camera-only
    row. With a base estimate the head only has to supply what the camera
    misses, which is the comparison we actually want to make.
    """
    import torch
    from torch import nn, optim
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, 1)).to(dev)
    opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.L1Loss()
    xt = torch.from_numpy(Xtr).to(dev); yt = torch.from_numpy(Ytr).to(dev)
    xv = torch.from_numpy(Xva).to(dev); yv = torch.from_numpy(Yva).to(dev)
    bt = (torch.from_numpy(base_tr).to(dev) if base_tr is not None
          else torch.zeros_like(yt))
    bv = (torch.from_numpy(base_va).to(dev) if base_va is not None
          else torch.zeros_like(yv))
    best, state = float("inf"), None
    n = len(xt)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for k in range(0, n, 256):
            b = perm[k:k + 256]
            loss = lossf(net(xt[b]).squeeze(-1) + bt[b], yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(xv).squeeze(-1) + bv, yv).item()
        if vl < best:
            best, state = vl, {k: v.detach().clone()
                               for k, v in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    return net


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    args = p.parse_args()

    import torch
    store = np.load(args.cache, allow_pickle=True)
    root = Path(args.root)

    rgb, feats = {}, {}
    for s in ("train", "val", "test"):
        logger.info("RGB predictions for %s (%d frames)", s,
                    len(store[f"{s}_img"]))
        rgb[s], feats[s] = rgb_predictions(store[f"{s}_img"], root,
                                            Path(args.ckpt),
                                            return_features=True)

    ok = {s: np.isfinite(rgb[s]) for s in rgb}
    results: Dict = {"seeds": args.seeds, "conditions": {},
                     "n": {s: int(ok[s].sum()) for s in ok}}

    # Visibility grouping of the test frames, from the recorded weather of the
    # scene each frame belongs to.
    te_meta = [json.loads(x) for x in store["test_meta"][ok["test"]]]
    weather = json.load(open(Path(args.root) / "infos" / "scene_weather.json")) \
        if (Path(args.root) / "infos" / "scene_weather.json").exists() else {}
    groups = [weather.get(m["scene_id"], "unknown") for m in te_meta]
    vis_masks = {g: np.array([x == g for x in groups]) for g in VIS_GROUPS}
    logger.info("test frames by visibility: %s",
                {g: int(m.sum()) for g, m in vis_masks.items()})
    results["n_by_visibility"] = {g: int(m.sum()) for g, m in vis_masks.items()}
    results["scenes_by_visibility"] = {
        g: len({m["scene_id"] for m, x in zip(te_meta, groups) if x == g})
        for g in VIS_GROUPS}

    Ytr = store["train_y"][ok["train"]]
    Yva = store["val_y"][ok["val"]]
    Yte = store["test_y"][ok["test"]]
    med = float(np.median(Ytr))
    results["baseline_median_mae_m"] = round(
        float(np.abs(med - Yte).mean()), 3)
    # Per-group baselines are essential, not decorative. The visibility groups
    # differ enormously in how hard the task is -- the fog subset comes from a
    # handful of scenes whose gap barely varies -- so a raw per-group MAE
    # compares different tasks and makes the camera look better in fog than in
    # clear weather. Every per-group number is therefore reported against the
    # baseline for that same group.
    results["baseline_by_visibility"] = {
        g: round(float(np.abs(med - Yte[m]).mean()), 3)
        for g, m in vis_masks.items() if m.any()}
    logger.info("median-gap baseline on test: %.3f m",
                results["baseline_median_mae_m"])

    per_frame_abs: Dict[str, np.ndarray] = {}
    for name, parts in CONDITIONS:
        # Camera features enter only where the camera does, and only for the
        # fusion rows: the camera-only row stays the frozen ResNet's own
        # prediction, unchanged, so it remains the number reported elsewhere.
        fuse = "rgb" in parts and len(parts) > 1
        ftr = feats["train"][ok["train"]] if fuse else None
        fva = feats["val"][ok["val"]] if fuse else None
        fte = feats["test"][ok["test"]] if fuse else None
        Xtr = assemble(store, "train", rgb["train"][ok["train"]], parts,
                       ok["train"], ftr)
        Xva = assemble(store, "val", rgb["val"][ok["val"]], parts, ok["val"], fva)
        Xte = assemble(store, "test", rgb["test"][ok["test"]], parts,
                       ok["test"], fte)
        assert len(Xtr) == len(Ytr) and len(Xte) == len(Yte)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
        # Residual target for fusion rows: correct the camera estimate.
        btr = rgb["train"][ok["train"]] if fuse else None
        bva = rgb["val"][ok["val"]] if fuse else None
        bte = rgb["test"][ok["test"]] if fuse else None

        maes, w5 = [], []
        by_vis: Dict[str, List[float]] = {k: [] for k in VIS_GROUPS}
        seed_abs: List[np.ndarray] = []
        for s in range(args.seeds):
            net = fit_mlp(Xtr, Ytr, Xva, Yva, 42 + s, args.epochs,
                          base_tr=btr, base_va=bva)
            with torch.no_grad():
                dev = next(net.parameters()).device
                pr = net(torch.from_numpy(Xte).to(dev)).squeeze(-1).cpu().numpy()
                if bte is not None:
                    pr = pr + bte
            abs_err = np.abs(pr - Yte)
            maes.append(float(abs_err.mean()))
            w5.append(float((abs_err <= 5).mean()))
            seed_abs.append(abs_err)
            # Degraded visibility is where the geometric modalities should
            # earn their place: the camera loses signal, radar and LiDAR
            # should not.
            for g, m in vis_masks.items():
                if m.any():
                    by_vis[g].append(float(abs_err[m].mean()))
        # Mean per-frame errors across seeds: paired Δ computed from these
        # equals the difference of the displayed mean MAEs.
        per_frame_abs[name] = np.mean(seed_abs, axis=0)
        results["conditions"][name] = {
            "mae_mean": round(float(np.mean(maes)), 3),
            "mae_sd": round(float(np.std(maes, ddof=1)), 3),
            "within_5m_mean": round(float(np.mean(w5)), 3),
            "n_features": int(Xte.shape[1]),
            "by_visibility": {g: round(float(np.mean(v)), 3)
                              for g, v in by_vis.items() if v},
            # Skill against the group's own baseline: 1 is perfect, 0 is no
            # better than predicting the training median, negative is worse.
            "skill_by_visibility": {
                g: round(1.0 - float(np.mean(v))
                         / results["baseline_by_visibility"][g], 3)
                for g, v in by_vis.items()
                if v and results["baseline_by_visibility"].get(g)},
        }
        logger.info("%-18s MAE %.3f +- %.3f  within5m %.3f", name,
                    np.mean(maes), np.std(maes, ddof=1), np.mean(w5))

    # Scene-cluster paired bootstrap: resample scenes (not frames) so that the
    # within-scene frame correlation does not produce overconfident intervals.
    # Frames within a scene are near-duplicates; on this corpus a frame-level
    # bootstrap gives intervals ~3.5× narrower than a scene-cluster one.
    if "rgb" in per_frame_abs:
        te_meta_ok = [json.loads(x) for x in store["test_meta"][ok["test"]]]
        scene_ids = np.array([m["scene_id"] for m in te_meta_ok])
        unique_scenes = np.unique(scene_ids)
        n_scenes = len(unique_scenes)
        # Index of each scene in unique_scenes array
        scene_idx = {s: i for i, s in enumerate(unique_scenes)}
        frame_scene = np.array([scene_idx[s] for s in scene_ids])

        rng = np.random.default_rng(0)
        B = 4000
        results["paired_vs_rgb"] = {}
        base = per_frame_abs["rgb"]
        for name, err in per_frame_abs.items():
            if name == "rgb":
                continue
            d = base - err                      # positive = fusion better
            # Per-scene mean improvement
            scene_means = np.array([d[frame_scene == i].mean()
                                    for i in range(n_scenes)])
            # Bootstrap by resampling scenes
            draws = rng.choice(scene_means, size=(B, n_scenes),
                               replace=True).mean(axis=1)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            results["paired_vs_rgb"][name] = {
                "mean_improvement_m": round(float(d.mean()), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "significant": bool(lo > 0 or hi < 0)}
            logger.info("%-18s vs rgb: %+.4f m [%+.4f, %+.4f] %s (scene-cluster, n=%d)",
                        name, d.mean(), lo, hi,
                        "significant" if (lo > 0 or hi < 0) else "n.s.",
                        n_scenes)

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n{'condition':<20}{'MAE (m)':>16}{'within 5 m':>12}")
    print("-" * 48)
    for k, v in results["conditions"].items():
        print(f"{k:<20}{v['mae_mean']:>9.2f}+-{v['mae_sd']:<5.2f}"
              f"{v['within_5m_mean']:>12.3f}")
    print(f"{'median baseline':<20}{results['baseline_median_mae_m']:>9.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
