"""
audit_qa_exact.py
==================
Does a question contain its own answer? Decide it exactly, without a model.

The first attempt at this measured leakage by asking a text-only language model
to answer the questions blind. That was the wrong instrument: it is slow, and it
turns a deterministic property of the text into a noisy estimate that depends on
reply parsing, token budgets and sampling. Whether the queried quantity is
printed in the prompt is not a matter of degree -- it either is or it is not.

Two forms of leakage are checked.

*Numeric copy.* The ground-truth answer value appears in the question as a
literal number, within the rounding the question itself uses. A question stating
"travelling at 25.4 m/s" and expecting 25.39 m/s has handed over the answer.

*Phrase copy.* A run of at least MIN_NGRAM consecutive words of the answer
appears verbatim in the question. This catches the case where a question asserts
a fact and then asks the model to restate it ("The ego vehicle is cruising at
roughly constant speed ... Why is the ego vehicle cruising at roughly constant
speed?").

A multiple-choice prompt necessarily lists every option, so finding the correct
label in the prompt is not by itself leakage. A label only counts as leaked when
it appears and at least one alternative label does not -- that is, when the
prompt singles out the answer rather than enumerating the choices.

    python audit_qa_exact.py --v1 <dir> --v2 <dir> --out <file.json>
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

logger = logging.getLogger("qaexact")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] qaexact: %(message)s")

MIN_NGRAM = 5           # consecutive words that must match to count as a copy
REL_TOL = 0.02          # 2 % of the answer magnitude
ABS_TOL = 0.15          # or this absolute margin, whichever is larger

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_WORD = re.compile(r"[a-z0-9]+")


def numbers_in(text: str) -> List[float]:
    return [float(m) for m in _NUM.findall(text.replace(",", ""))]


def numeric_copy(question: str, value: float) -> bool:
    """Is the answer value printed in the question, to its own rounding?"""
    tol = max(abs(value) * REL_TOL, ABS_TOL)
    return any(abs(q - value) <= tol for q in numbers_in(question))


def words(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def longest_shared_run(answer: str, question: str) -> int:
    """Longest run of consecutive answer words appearing verbatim in question."""
    a, q = words(answer), words(question)
    if not a or not q:
        return 0
    qs = " " + " ".join(q) + " "
    best = 0
    for i in range(len(a)):
        # Extend while the run is still present; stop at the first miss.
        for j in range(i + best + 1, len(a) + 1):
            if f" {' '.join(a[i:j])} " in qs:
                best = max(best, j - i)
            else:
                break
    return best


def label_leak(question: str, answer: str, labels: Set[str]) -> bool:
    """
    True when the prompt names the correct label but withholds at least one
    alternative. A prompt that lists every option has not leaked anything.
    """
    if answer not in labels or len(labels) < 2:
        return False
    ql = question.lower()
    if answer.lower() not in ql:
        return False
    return any(other.lower() not in ql for other in labels if other != answer)


def audit(records: Iterable[Dict], label_sets: Dict[str, Set[str]]) -> Dict:
    by: Dict[str, Dict] = collections.defaultdict(
        lambda: {"n": 0, "numeric_copy": 0, "phrase_copy": 0,
                 "label_leak": 0, "runs": []})
    for r in records:
        t = r["qa_type"]
        e = by[t]
        e["n"] += 1
        q = r["question"]
        val = r.get("answer_value")
        ans = r.get("answer") or ""
        if isinstance(val, (int, float)) and numeric_copy(q, float(val)):
            e["numeric_copy"] += 1
        run = longest_shared_run(ans, q)
        e["runs"].append(run)
        if run >= MIN_NGRAM:
            e["phrase_copy"] += 1
        if label_leak(q, ans, label_sets.get(t, set())):
            e["label_leak"] += 1

    out = []
    for t, e in sorted(by.items()):
        n = e["n"]
        out.append({
            "qa_type": t, "n": n,
            "numeric_copy_rate": round(e["numeric_copy"] / n, 3),
            "phrase_copy_rate": round(e["phrase_copy"] / n, 3),
            "label_leak_rate": round(e["label_leak"] / n, 3),
            "median_shared_run_words": int(sorted(e["runs"])[n // 2]),
            "max_shared_run_words": int(max(e["runs"])),
        })
    return {"min_ngram": MIN_NGRAM, "rel_tol": REL_TOL, "abs_tol": ABS_TOL,
            "types": out}


def load_v2(d: Path, split: str = "test", limit: Optional[int] = None) -> List[Dict]:
    recs = []
    with open(d / f"qa_{split}.jsonl") as f:
        for line in f:
            recs.append(json.loads(line))
            if limit and len(recs) >= limit:
                break
    return recs


def load_v1(d: Path, limit_files: int) -> List[Dict]:
    recs, seen = [], 0
    for p in d.rglob("*.json"):
        seen += 1
        if seen > limit_files:
            break
        try:
            doc = json.load(open(p))
        except Exception:
            continue
        for qa in doc.get("qa_pairs", []):
            recs.append({"qa_type": qa["qa_type"], "question": qa["question"],
                         "answer": qa.get("answer", ""),
                         "answer_value": qa.get("answer_value")})
    return recs


def label_sets_of(records: Iterable[Dict]) -> Dict[str, Set[str]]:
    """Short, repeated answers are treated as a closed label set."""
    vals: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in records:
        a = r.get("answer")
        if isinstance(a, str) and len(a) <= 40:
            vals[r["qa_type"]][a] += 1
    return {t: set(c) for t, c in vals.items() if 2 <= len(c) <= 20}


def report(tag: str, res: Dict) -> None:
    print(f"\n=== {tag}: does the question contain its own answer? ===")
    hdr = (f"{'qa_type':<22}{'n':>7}{'numeric':>10}{'phrase':>10}"
           f"{'label':>9}{'med run':>9}")
    print(hdr)
    print("-" * len(hdr))
    for e in res["types"]:
        print(f"{e['qa_type']:<22}{e['n']:>7}{e['numeric_copy_rate']:>10.2f}"
              f"{e['phrase_copy_rate']:>10.2f}{e['label_leak_rate']:>9.2f}"
              f"{e['median_shared_run_words']:>9}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v1", default=None)
    p.add_argument("--v2", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--v1-files", type=int, default=2000)
    p.add_argument("--v2-limit", type=int, default=40000)
    args = p.parse_args()

    out: Dict[str, Dict] = {}
    if args.v1:
        r1 = load_v1(Path(args.v1), args.v1_files)
        logger.info("v1: %d records", len(r1))
        out["v1"] = audit(r1, label_sets_of(r1))
        report("v1 (superseded release)", out["v1"])
    if args.v2:
        r2 = load_v2(Path(args.v2), limit=args.v2_limit)
        logger.info("v2: %d records", len(r2))
        out["v2"] = audit(r2, label_sets_of(r2))
        report("v2 (current release)", out["v2"])

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
