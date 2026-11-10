"""
train_spacing_cnn.py
=====================
Supervised monocular spacing regression: front camera -> bumper-to-bumper gap.

Purpose
-------
The zero-shot vision--language benchmark found that no evaluated model beats a
constant median-gap predictor. On its own that result is ambiguous: it is
consistent with the models being weak, but equally consistent with the task
being ill-posed -- monocular metric distance from a single frame might simply
not be recoverable from this imagery. This experiment separates the two. It
trains a small supervised regressor on the dataset's own training split and
evaluates it on the identical 400 test frames the zero-shot models were scored
on. If it beats the baseline, the information is demonstrably present and
learnable, and the zero-shot result is a gap in adaptation rather than an
ill-posed task. If it does not, the framing itself needs revisiting.

Protocol
--------
Splits are the dataset's scene-level splits, inherited unchanged, so no scene
contributes to both training and evaluation. The reported metrics and the
median baseline use the same definitions as the zero-shot benchmark, so the
rows are directly comparable.

    python train_spacing_cnn.py --cache <cache dir> --out <results dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("spacing")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] spacing: %(message)s")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class SpacingSet(Dataset):
    """Cached frames with their exact gap label.

    Horizontal flip is the only geometric augmentation: mirroring the image
    mirrors the road but leaves the distance to the lead vehicle unchanged, so
    the label is preserved. Crops or scalings would not preserve it -- apparent
    size is precisely the cue being learned -- so they are excluded.
    """

    def __init__(self, cache: Path, tag: str, train: bool):
        self.x = np.load(cache / f"{tag}_images.npy", mmap_mode="r")
        self.y = np.load(cache / f"{tag}_spacing.npy")
        self.train = train

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        img = np.asarray(self.x[i])
        if self.train and np.random.rand() < 0.5:
            img = img[:, ::-1]
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        if self.train:
            t = (t * np.random.uniform(0.8, 1.2)).clamp(0, 1)     # brightness
        return t, torch.tensor(self.y[i], dtype=torch.float32)


def build_model() -> nn.Module:
    import torchvision.models as tvm
    net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net


@torch.no_grad()
def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    P, G = [], []
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    for x, y in loader:
        x = ((x.to(device, non_blocking=True) - mean) / std)
        with torch.autocast("cuda", dtype=torch.float16):
            p = model(x).squeeze(1).float()
        P.append(p.cpu().numpy()); G.append(y.numpy())
    return np.concatenate(P), np.concatenate(G)


def metrics(pred: np.ndarray, gt: np.ndarray, train_median: float) -> Dict:
    err = np.abs(pred - gt)
    base = np.abs(train_median - gt)
    ss_res = float(((pred - gt) ** 2).sum())
    ss_tot = float(((gt - gt.mean()) ** 2).sum())
    return {
        "n": int(len(gt)),
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(((pred - gt) ** 2).mean())),
        "within_5m": float((err <= 5).mean()),
        "within_2m": float((err <= 2).mean()),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
        "baseline_mae": float(base.mean()),
        "baseline_within_5m": float((base <= 5).mean()),
        "pred_mean": float(pred.mean()), "gt_mean": float(gt.mean()),
    }


def bootstrap_delta(pred: np.ndarray, gt: np.ndarray, train_median: float,
                    n_boot: int, seed: int) -> Dict:
    """Paired bootstrap CI on (baseline MAE - model MAE).

    Paired, because both predictors are scored on the same frames; a CI that
    excludes zero is what licenses the claim that one beats the other, rather
    than eyeballing two point estimates.
    """
    rng = np.random.default_rng(seed)
    e_m = np.abs(pred - gt)
    e_b = np.abs(train_median - gt)
    idx = rng.integers(0, len(gt), size=(n_boot, len(gt)))
    d = e_b[idx].mean(axis=1) - e_m[idx].mean(axis=1)
    return {"delta_mae": float(e_b.mean() - e_m.mean()),
            "ci95_low": float(np.percentile(d, 2.5)),
            "ci95_high": float(np.percentile(d, 97.5)),
            "p_model_worse": float((d <= 0).mean()),
            "n_boot": n_boot}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=10000)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cache, out = Path(args.cache), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = SpacingSet(cache, "train", True)
    va = SpacingSet(cache, "val", False)
    te = SpacingSet(cache, "test", False)
    pr = SpacingSet(cache, "probes", False)
    train_median = float(np.median(tr.y))
    logger.info("train %d | val %d | test %d | probes %d | train median gap %.2f m",
                len(tr), len(va), len(te), len(pr), train_median)

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch, shuffle=sh,
                                   num_workers=args.workers, pin_memory=True,
                                   drop_last=sh, persistent_workers=True)
    tl, vl, tel, prl = dl(tr, True), dl(va, False), dl(te, False), dl(pr, False)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(tl), pct_start=0.3)
    scaler = torch.cuda.amp.GradScaler()
    lossf = nn.L1Loss()          # matches the reported metric (MAE, metres)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    best, best_state, history = float("inf"), None, []
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in tl:
            x = ((x.to(device, non_blocking=True) - mean) / std)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(x).squeeze(1), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y)
        vp, vg = predict(model, vl, device)
        vmae = float(np.abs(vp - vg).mean())
        history.append({"epoch": ep, "train_l1": tot / len(tr), "val_mae": vmae})
        logger.info("epoch %2d  train L1 %.3f  val MAE %.3f m  (%.0f s)",
                    ep, tot / len(tr), vmae, time.time() - t0)
        if vmae < best:
            best = vmae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    logger.info("best val MAE %.3f m; evaluating that checkpoint", best)
    model.load_state_dict(best_state)
    torch.save(best_state, out / "spacing_resnet18.pt")

    tp, tg = predict(model, tel, device)
    pp, pg = predict(model, prl, device)
    res = {
        "config": vars(args),
        "train_median_gap_m": train_median,
        "best_val_mae": best,
        "history": history,
        "test_all_frames": metrics(tp, tg, train_median),
        "vlm_probe_frames": metrics(pp, pg, train_median),
        "vlm_probe_bootstrap": bootstrap_delta(pp, pg, train_median,
                                               args.n_boot, args.seed),
        "test_bootstrap": bootstrap_delta(tp, tg, train_median,
                                          args.n_boot, args.seed),
    }
    with open(out / "spacing_cnn_results.json", "w") as f:
        json.dump(res, f, indent=2)
    np.save(out / "probe_predictions.npy", pp)

    for tag in ("test_all_frames", "vlm_probe_frames"):
        m = res[tag]
        print(f"\n{tag}: n={m['n']}")
        print(f"  MAE       {m['mae']:.2f} m   (median baseline {m['baseline_mae']:.2f} m)")
        print(f"  RMSE      {m['rmse']:.2f} m     R2 {m['r2']:.3f}")
        print(f"  within 5m {100*m['within_5m']:.1f}%  (baseline {100*m['baseline_within_5m']:.1f}%)")
        print(f"  within 2m {100*m['within_2m']:.1f}%")
    b = res["vlm_probe_bootstrap"]
    print(f"\nprobe set, paired bootstrap: baseline MAE - model MAE = "
          f"{b['delta_mae']:.2f} m  95% CI [{b['ci95_low']:.2f}, {b['ci95_high']:.2f}]")
    print(f"wrote {out}/spacing_cnn_results.json")


if __name__ == "__main__":
    main()
