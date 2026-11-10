"""
audit_qa_leakage.py
====================
Measure how much of a QA release is answerable without looking at the image.

A question that states the quantity it asks for is not a perception task, and
the only way to know whether a redesign fixed that is to try to answer it blind.
This runs a text-only language model on the questions alone -- no image, ever --
and compares its score against the trivial baseline for that question type: the
majority class for categorical answers, the training median for numeric ones.

Reading the result
------------------
Text-only score at or below the trivial baseline means the question carries no
answer, which is what a perception benchmark requires. Text-only score well
above the baseline means the prompt leaks, and the gap measures how much.

The same procedure is applied to the superseded v1 release, so the comparison
is like-for-like rather than asserted.

    python audit_qa_leakage.py --v2 <dir> --v1 <dir> --out <file.json>
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests

logger = logging.getLogger("qaaudit")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] qaaudit: %(message)s")

OLLAMA = "http://localhost:11434/api/generate"
NUMERIC = {"spacing_estimation", "speed_prediction"}


def ask(model: str, prompt: str, num_predict: int = 160,
        timeout: int = 120) -> str:
    r = requests.post(OLLAMA, json={"model": model, "prompt": prompt,
                                    "stream": False,
                                    "options": {"temperature": 0.0,
                                                "num_predict": num_predict}},
                      timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")


def parse_number(t: str) -> Optional[float]:
    m = re.search(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", t, re.I)
    if m:
        return float(m.group(1))
    n = re.findall(r"-?\d+(?:\.\d+)?", t.replace(",", ""))
    return float(n[-1]) if n else None


def parse_choice(t: str, options: List[str]) -> Optional[str]:
    up = t.upper()
    hits = [(up.find(o.upper()), o) for o in options if o.upper() in up]
    return min(hits)[1] if hits else None


def load_v2(d: Path, split: str = "test") -> Dict[str, List[Dict]]:
    by = collections.defaultdict(list)
    with open(d / f"qa_{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            by[r["qa_type"]].append(r)
    return by


def load_v1(vqa_dir: Path, limit_files: int) -> Dict[str, List[Dict]]:
    by = collections.defaultdict(list)
    files = []
    for i, p in enumerate(vqa_dir.rglob("*.json")):
        files.append(p)
        if len(files) >= limit_files:
            break
    for p in files:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        for qa in d.get("qa_pairs", []):
            by[qa["qa_type"]].append(
                {"question": qa["question"], "answer": qa.get("answer", ""),
                 "answer_value": qa.get("answer_value"),
                 "qa_type": qa["qa_type"]})
    return by


def options_for(qa_type: str, samples: List[Dict]) -> List[str]:
    return sorted({s["answer"] for s in samples if isinstance(s["answer"], str)
                   and len(s["answer"]) < 40})


def score(model: str, qa_type: str, samples: List[Dict], n: int,
          rng: random.Random, balanced: bool, num_predict: int = 160) -> Dict:
    if balanced and qa_type not in NUMERIC:
        by = collections.defaultdict(list)
        for s in samples:
            by[s["answer"]].append(s)
        per = max(1, n // max(len(by), 1))
        sel = [x for v in by.values() for x in rng.sample(v, min(per, len(v)))]
        rng.shuffle(sel)
    else:
        sel = rng.sample(samples, min(n, len(samples)))

    if qa_type in NUMERIC:
        gts = [float(s["answer_value"]) for s in sel if s.get("answer_value") is not None]
        base = float(np.median([float(s["answer_value"]) for s in samples
                                if s.get("answer_value") is not None]))
        preds, gt2 = [], []
        for s in sel:
            if s.get("answer_value") is None:
                continue
            v = parse_number(ask(model, s["question"], num_predict))
            if v is None or not (0 <= v <= 300):
                continue
            preds.append(v); gt2.append(float(s["answer_value"]))
        if not preds:
            return {"qa_type": qa_type, "n": 0}
        preds, gt2 = np.array(preds), np.array(gt2)
        return {"qa_type": qa_type, "kind": "numeric", "n": int(len(preds)),
                "text_only_mae": round(float(np.abs(preds - gt2).mean()), 3),
                "baseline_mae": round(float(np.abs(base - gt2).mean()), 3),
                "leaks": bool(np.abs(preds - gt2).mean() <
                              0.8 * np.abs(base - gt2).mean())}

    opts = options_for(qa_type, samples)
    maj = collections.Counter(s["answer"] for s in sel).most_common(1)[0][1] / len(sel)
    ok = tot = 0
    # Track WHAT the blind model said, not just whether it was right. Scoring
    # exactly at chance is only meaningful if we can show the model collapsed
    # to a near-constant reply for want of anything to condition on; without
    # this the reader has to take a suspiciously round number on trust.
    preds: collections.Counter = collections.Counter()
    unparsed = 0
    for s in sel:
        g = parse_choice(ask(model, s["question"], num_predict), opts)
        if g is None:
            unparsed += 1
            continue
        preds[g] += 1
        tot += 1
        ok += int(g == s["answer"])
    if not tot:
        return {"qa_type": qa_type, "n": 0, "n_unparsed": unparsed}
    acc = ok / tot
    top, topn = preds.most_common(1)[0]
    return {"qa_type": qa_type, "kind": "categorical", "n": tot,
            "n_options": len(opts), "text_only_acc": round(acc, 3),
            "majority_baseline": round(maj, 3),
            "n_unparsed": unparsed,
            "pred_distribution": dict(preds.most_common()),
            "modal_pred": top,
            "modal_pred_share": round(topn / tot, 3),
            "leaks": bool(acc > maj + 0.10)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v2", default=None)
    p.add_argument("--v1", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="qwen2.5:14b-instruct")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--v1-files", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-predict", type=int, default=160,
                   help="token budget per reply; v1 free-text answers run to "
                        "the full budget, so lowering it dominates runtime")
    p.add_argument("--merge", default=None,
                   help="existing result JSON whose arms are carried forward "
                        "rather than recomputed")
    args = p.parse_args()

    rng = random.Random(args.seed)
    out = {"model": args.model, "n_per_type": args.n,
           "num_predict": args.num_predict, "v2": [], "v1": []}
    if args.merge and Path(args.merge).exists():
        prev = json.load(open(args.merge))
        for tag in ("v1", "v2"):
            out[tag] = prev.get(tag, [])
        logger.info("carried forward: %s",
                    {t: len(out[t]) for t in ("v1", "v2")})

    if args.v2:
        v2 = load_v2(Path(args.v2))
        for t in sorted(v2):
            r = score(args.model, t, v2[t], args.n, rng, balanced=True,
                      num_predict=args.num_predict)
            out["v2"].append(r); logger.info("v2 %s -> %s", t, r)

    if args.v1:
        v1 = load_v1(Path(args.v1), args.v1_files)
        logger.info("v1 types: %s",
                    {t: len(v) for t, v in sorted(v1.items())})
        for t in sorted(v1):
            r = score(args.model, t, v1[t], args.n, rng, balanced=True,
                      num_predict=args.num_predict)
            out["v1"].append(r); logger.info("v1 %s -> %s", t, r)
            json.dump(out, open(args.out, "w"), indent=2)  # checkpoint

    json.dump(out, open(args.out, "w"), indent=2)
    for tag in ("v1", "v2"):
        if not out[tag]:
            continue
        print(f"\n=== {tag} (text-only, no image) ===")
        print(f"{'qa_type':<24}{'score':>10}{'baseline':>10}{'leaks?':>9}")
        for r in out[tag]:
            if not r.get("n"):
                continue
            sc = r.get("text_only_mae", r.get("text_only_acc"))
            bl = r.get("baseline_mae", r.get("majority_baseline"))
            print(f"{r['qa_type']:<24}{sc:>10.3f}{bl:>10.3f}"
                  f"{('LEAKS' if r['leaks'] else 'ok'):>9}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
