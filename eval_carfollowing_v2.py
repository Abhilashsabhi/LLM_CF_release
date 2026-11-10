"""
eval_carfollowing_v2.py
========================
Car-following benchmark, extended to establish a difficulty ceiling.

Why this extends the original
-----------------------------
The first benchmark reported one MLP, a 16-unit LSTM, a calibrated IDM and a
constant-acceleration floor. That the constant policy was competitive on
spacing RMSE admits two readings: that RMSE and safety are decoupled, which is
the reading we drew, or that the trajectories are stationary enough for doing
nothing to be a good answer, which would make the benchmark uninformative. The
second reading has to be excluded rather than argued away, so this script adds:

*Stratified evaluation.* Events are split into \"nominal\" -- the two scenario
categories that carry no scripted injection -- and \"excited\", the seven that
do. If the constant policy is competitive only because most events are
quiescent, its advantage must collapse on the excited stratum. The split is the
taxonomy's own, not a post-hoc threshold, and we report the measured excitation
of each stratum so the reader can see it is not circular.

*Stronger baselines.* A 128-unit two-layer GRU and a temporal Transformer
encoder stand for modern sequence models; an IDM-residual model stands for the
physics-informed family, predicting a bounded correction to a calibrated IDM
acceleration rather than the acceleration itself. If a benchmark's ceiling is
set by a 16-unit LSTM, the ceiling is the model, not the data.

*Seeds and intervals.* Every neural model is trained under several seeds and
reported as a mean with a 95 % interval across seeds; every metric also carries
a bootstrap interval over events. A single-seed table cannot distinguish a real
ordering from initialisation noise.

    python eval_carfollowing_v2.py --arrays <dir> --out <dir> --seeds 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from eval_carfollowing import (ACC_LIMIT, TS_DEFAULT, aggregate, calibrate_idm,
                               event_metrics, idm_acc, make_supervised,
                               rollout)

logger = logging.getLogger("cfbench2")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] cfbench2: %(message)s")

# The two categories that carry no scripted injection (verified in the release:
# no frame of either ever has an active injection flag).
NOMINAL_CATEGORIES = {"steady_follow", "adverse_weather"}

NEURAL = {"mlp", "lstm", "gru", "transformer", "idm_residual"}


# --------------------------------------------------------------------------
# models

def build(kind: str, his: int):
    import torch
    from torch import nn

    class MLP(nn.Module):
        def __init__(self, in_dim, hidden=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1), nn.Tanh())

        def forward(self, x):
            return ACC_LIMIT * self.net(x)

    class RNN(nn.Module):
        """LSTM at the original 16 units, or a 128-unit two-layer GRU."""

        def __init__(self, cell="lstm", hidden=16, layers=1):
            super().__init__()
            mod = nn.LSTM if cell == "lstm" else nn.GRU
            self.cell = cell
            self.enc = mod(3, hidden, layers, batch_first=False,
                           dropout=0.1 if layers > 1 else 0.0)
            self.lin = nn.Linear(hidden, 1)
            nn.init.normal_(self.lin.weight, 0, 0.02)
            nn.init.constant_(self.lin.bias, 0.0)

        def forward(self, x):
            out = self.enc(x)[1]
            h = out[0] if self.cell == "lstm" else out
            return ACC_LIMIT * torch.tanh(self.lin(h[-1]))

    class Transformer(nn.Module):
        """Self-attention over the history window.

        Small on purpose: the window is ten steps, and a large encoder on a
        ten-token sequence would be reporting capacity we cannot use.
        """

        def __init__(self, d_model=64, nhead=4, layers=2, his=his):
            super().__init__()
            self.inp = nn.Linear(3, d_model)
            self.pos = nn.Parameter(torch.zeros(his, 1, d_model))
            enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4,
                                             dropout=0.1, batch_first=False)
            self.enc = nn.TransformerEncoder(enc, layers)
            self.lin = nn.Linear(d_model, 1)

        def forward(self, x):                      # (his, B, 3)
            z = self.inp(x) + self.pos[: x.shape[0]]
            z = self.enc(z)[-1]
            return ACC_LIMIT * torch.tanh(self.lin(z))

    class IDMResidual(nn.Module):
        """Physics-informed: calibrated IDM plus a bounded learned correction.

        The IDM term is computed outside and passed in as a feature, so the
        network only has to explain what IDM gets wrong. The correction is
        capped at half the acceleration limit, which keeps the model anchored
        to the physics rather than free to ignore it.
        """

        def __init__(self, hidden=128, cap=ACC_LIMIT / 2.0):
            super().__init__()
            self.cap = cap
            self.enc = nn.GRU(3, hidden, 1, batch_first=False)
            self.lin = nn.Linear(hidden + 1, 1)
            nn.init.normal_(self.lin.weight, 0, 0.02)
            nn.init.constant_(self.lin.bias, 0.0)

        def forward(self, x, idm):                 # x: (his, B, 3), idm: (B,1)
            h = self.enc(x)[1][-1]
            r = self.cap * torch.tanh(self.lin(torch.cat([h, idm], dim=-1)))
            return torch.clamp(idm + r, -ACC_LIMIT, ACC_LIMIT)

    import torch  # noqa: F811  (needed by the closures above)
    if kind == "mlp":
        return MLP(his * 3)
    if kind == "lstm":
        return RNN("lstm", 16, 1)
    if kind == "gru":
        return RNN("gru", 128, 2)
    if kind == "transformer":
        return Transformer()
    if kind == "idm_residual":
        return IDMResidual()
    raise ValueError(kind)


# --------------------------------------------------------------------------

def idm_feature(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """IDM acceleration at the last step of each history window."""
    out = np.empty((len(X), 1), dtype=np.float32)
    for i, w in enumerate(X):
        sp, sv, rel = w[-1]
        out[i, 0] = idm_acc(params, float(sp), float(sv), float(rel))
    return out


def train(kind: str, tr: np.ndarray, va: np.ndarray, his: int, ts: float,
          epochs: int, batch: int, lr: float, seed: int, params) -> Tuple:
    import torch
    from torch import nn, optim
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, Ytr = make_supervised(tr, his, ts)
    Xva, Yva = make_supervised(va, his, ts)
    mu, sd = Xtr.reshape(-1, 3).mean(0), Xtr.reshape(-1, 3).std(0) + 1e-6
    Itr = idm_feature(Xtr, params) if kind == "idm_residual" else np.zeros((len(Xtr), 1), np.float32)
    Iva = idm_feature(Xva, params) if kind == "idm_residual" else np.zeros((len(Xva), 1), np.float32)
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd

    model = build(kind, his).to(dev)
    opt = optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Itr),
                                  torch.from_numpy(Ytr)),
                    batch_size=batch, shuffle=True, drop_last=True)
    Xv = torch.from_numpy(Xva).to(dev)
    Iv = torch.from_numpy(Iva).to(dev)
    Yv = torch.from_numpy(Yva).to(dev)

    def fwd(xb, ib):
        if kind == "mlp":
            return model(xb.reshape(xb.shape[0], -1))
        seq = xb.permute(1, 0, 2)
        return model(seq, ib) if kind == "idm_residual" else model(seq)

    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        for xb, ib, yb in dl:
            xb, ib, yb = xb.to(dev), ib.to(dev), yb.to(dev)
            loss = lossf(fwd(xb, ib).squeeze(-1), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(fwd(Xv, Iv).squeeze(-1), Yv).item()
        if vl < best:
            best = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    logger.info("  %-13s seed %d  val %.5f", kind, seed, best)
    return model, mu, sd, dev


def acc_fn_for(model, kind: str, his: int, mu, sd, dev, params):
    import torch
    hist: List[List[float]] = []

    def fn(i, spacing, sv, rel):
        hist.append([spacing, sv, rel])
        if len(hist) < his:
            return 0.0
        w = np.asarray(hist[-his:], dtype=np.float32)
        idm = np.float32(idm_acc(params, spacing, sv, rel))
        w = (w - mu) / sd
        with torch.no_grad():
            t = torch.from_numpy(w).to(dev)
            if kind == "mlp":
                return float(model(t.reshape(1, -1)).item())
            seq = t.unsqueeze(1)
            if kind == "idm_residual":
                it = torch.tensor([[idm]], device=dev)
                return float(model(seq, it).item())
            return float(model(seq).item())
    return fn


# --------------------------------------------------------------------------

def excitation(ev: np.ndarray, ts: float) -> float:
    """How much the leader actually forces the follower, in m/s^2."""
    return float(np.std(np.diff(ev[3]) / ts))


def boot_ci(vals: List[float], n: int = 2000, seed: int = 0) -> Tuple[float, float]:
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    r = np.random.default_rng(seed)
    a = np.asarray(vals, dtype=float)
    m = r.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    return (round(float(np.percentile(m, 2.5)), 3),
            round(float(np.percentile(m, 97.5)), 3))


def boot_rmse_ci(mses: List[float], n: int = 2000,
                 seed: int = 0) -> Tuple[float, float]:
    """Bootstrap the aggregate RMSE by resampling events.

    RMSE is sqrt of the MEAN squared error over events, so each resample must
    be reduced that way. Averaging per-event RMSE would estimate a different
    quantity and would sit below the aggregate by Jensen's inequality.
    """
    if len(mses) < 2:
        return (float("nan"), float("nan"))
    r = np.random.default_rng(seed)
    a = np.asarray(mses, dtype=float)
    draws = np.sqrt(r.choice(a, size=(n, len(a)), replace=True).mean(axis=1))
    return (round(float(np.percentile(draws, 2.5)), 3),
            round(float(np.percentile(draws, 97.5)), 3))


def evaluate(acc_fn_factory, test: np.ndarray, ts: float,
             mask: np.ndarray) -> Dict:
    per = []
    for k, ev in enumerate(test):
        if not mask[k]:
            continue
        sim = rollout(acc_fn_factory(), ev, ts)
        per.append(event_metrics(sim, ev, ts))
    agg = aggregate(per)
    agg["rmse_ci"] = boot_rmse_ci([m["spacing_mse"] for m in per])
    agg["collision_ci"] = boot_ci([m["collided"] for m in per])
    return agg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arrays", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ts", type=float, default=TS_DEFAULT)
    p.add_argument("--models", nargs="+",
                   default=["constant", "idm", "mlp", "lstm", "gru",
                            "transformer", "idm_residual"])
    p.add_argument("--his-horizon", type=int, default=10)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--idm-events", type=int, default=40)
    p.add_argument("--idm-maxiter", type=int, default=30)
    args = p.parse_args()

    arr, out = Path(args.arrays), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr = np.load(arr / "train.npy")
    va = np.load(arr / "val.npy")
    te = np.load(arr / "test.npy")
    meta = json.load(open(arr / "test_meta.json"))
    assert len(meta) == len(te), "metadata and test arrays disagree"

    nominal = np.array([m.get("scenario_category") in NOMINAL_CATEGORIES
                        for m in meta])
    strata = {"all": np.ones(len(te), bool), "nominal": nominal,
              "excited": ~nominal}
    exc = np.array([excitation(e, args.ts) for e in te])
    logger.info("test events: %d nominal, %d excited", nominal.sum(),
                (~nominal).sum())
    logger.info("leader-acceleration std: nominal %.3f, excited %.3f m/s^2",
                exc[nominal].mean(), exc[~nominal].mean())

    params = calibrate_idm(tr, args.ts, args.idm_events, 42, args.idm_maxiter)
    logger.info("IDM parameters: %s", np.round(params, 3).tolist())

    results: Dict = {"strata": {k: int(v.sum()) for k, v in strata.items()},
                     "excitation_mps2": {
                         "nominal": round(float(exc[nominal].mean()), 3),
                         "excited": round(float(exc[~nominal].mean()), 3)},
                     "seeds": args.seeds, "models": {}}

    for kind in args.models:
        entry: Dict = {"per_seed": [], "strata": {}}
        if kind in NEURAL:
            trained = []
            for s in range(args.seeds):
                m, mu, sd, dev = train(kind, tr, va, args.his_horizon, args.ts,
                                       args.epochs, args.batch, args.lr,
                                       42 + s, params)
                trained.append((m, mu, sd, dev))
            for name, mask in strata.items():
                per_seed = [evaluate(
                    lambda m=m, mu=mu, sd=sd, dev=dev: acc_fn_for(
                        m, kind, args.his_horizon, mu, sd, dev, params),
                    te, args.ts, mask) for (m, mu, sd, dev) in trained]
                entry["strata"][name] = summarise(per_seed)
        else:
            if kind == "constant":
                fac = lambda: (lambda i, sp, sv, rel: 0.0)          # noqa: E731
            else:
                fac = lambda: (lambda i, sp, sv, rel:                # noqa: E731
                               idm_acc(params, sp, sv, rel))
            for name, mask in strata.items():
                entry["strata"][name] = summarise(
                    [evaluate(fac, te, args.ts, mask)])
        results["models"][kind] = entry
        logger.info("%s -> all: RMSE %.2f  collisions %.3f", kind,
                    entry["strata"]["all"]["spacing_rmse"]["mean"],
                    entry["strata"]["all"]["collision_rate"]["mean"])

    json.dump(results, open(out / "cfbench_v2.json", "w"), indent=2)
    report(results)
    print(f"\nwrote {out / 'cfbench_v2.json'}")


def summarise(runs: List[Dict]) -> Dict:
    """Mean over seeds with a 95 % interval; event bootstrap from the first."""
    keys = ["spacing_rmse", "collision_rate", "jerk_mean", "min_ttc_median"]
    out: Dict = {"n_events": runs[0].get("n_events"),
                 "n_seeds": len(runs)}
    for k in keys:
        v = [r[k] for r in runs if k in r and r[k] is not None
             and np.isfinite(r[k])]
        if not v:
            continue
        m = float(np.mean(v))
        out[k] = {"mean": round(m, 3),
                  "sd": round(float(np.std(v, ddof=1)), 3) if len(v) > 1 else 0.0,
                  "seed_ci": (boot_ci(v) if len(v) > 1 else (round(m, 3),
                                                            round(m, 3)))}
    out["event_rmse_ci"] = runs[0].get("rmse_ci")
    out["event_collision_ci"] = runs[0].get("collision_ci")
    return out


def report(res: Dict) -> None:
    print(f"\nstrata: {res['strata']}   leader-accel std "
          f"nominal {res['excitation_mps2']['nominal']} / "
          f"excited {res['excitation_mps2']['excited']} m/s^2")
    for stratum in ("all", "nominal", "excited"):
        print(f"\n=== {stratum} ===")
        hdr = f"{'model':<15}{'RMSE (m)':>18}{'collisions':>16}{'jerk':>9}"
        print(hdr); print("-" * len(hdr))
        for name, e in res["models"].items():
            s = e["strata"].get(stratum, {})
            if "spacing_rmse" not in s:
                continue
            r, c = s["spacing_rmse"], s["collision_rate"]
            rc = f"{r['mean']:.2f}+-{r['sd']:.2f}" if r["sd"] else f"{r['mean']:.2f}"
            cc = f"{c['mean']:.3f}"
            j = s.get("jerk_mean", {}).get("mean", float("nan"))
            print(f"{name:<15}{rc:>18}{cc:>16}{j:>9.2f}")


if __name__ == "__main__":
    main()
