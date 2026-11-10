"""
eval_vlm_zeroshot.py
=====================
Zero-shot vision--language benchmark on the car-following dataset.

Why the released QA pairs are not used verbatim
-----------------------------------------------
The templated questions state the kinematics in the prompt ("travelling at
25.2 m/s ... gap of 24.1 m ... TTC 9999 s"). A text-only model answers the
speed-prediction question by echoing the number it was given, so the released
pairs measure arithmetic, not perception. We therefore build the probes from
the same ground truth but WITHDRAW the quantity being asked for, so the model
must recover it from the image. The original QA pairs remain valid as
supervision; they are simply not a perception benchmark.

Tasks
-----
  spacing    regress bumper-to-bumper gap [m] from the front camera.
             Exactly the quantity car-following needs, with exact GT.
             Metric: MAE, plus fraction within +/-5 m.
             Baseline: predict the training-set median gap.

  safety     binary "is the current following distance safe?".
             Sampled CLASS-BALANCED, because the natural distribution is
             ~94% safe and accuracy on it is uninformative.
             Metric: balanced accuracy. Baseline: 0.5.

  scenario   9-way scenario classification from the image.
             Metric: accuracy. Baseline: 1/9 = 0.111.

  leakage    control. The ORIGINAL speed-prediction question, text only, no
             image. Establishes that the released phrasing is solvable
             without perception and quantifies by how much.

Usage
-----
    python eval_vlm_zeroshot.py --dataset /path/to/cf_dataset \\
        --model qwen2.5vl:7b --n 150
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger("vlmeval")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] vlmeval: %(message)s")

OLLAMA = "http://localhost:11434/api/generate"

CATEGORIES = ["steady_follow", "stop_and_go", "aggressive_leader", "cut_in",
              "cut_out", "speed_limit_change", "grade_change",
              "adverse_weather", "pedestrian_crossing"]

CATEGORY_GLOSS = {
    "steady_follow":       "following a lead vehicle at roughly constant speed",
    "stop_and_go":         "congested stop-and-go, leader repeatedly halting",
    "aggressive_leader":   "the lead vehicle brakes hard and suddenly",
    "cut_in":              "another vehicle cuts into the lane ahead",
    "cut_out":             "the lead vehicle leaves the lane, revealing a slower one",
    "speed_limit_change":  "the lead vehicle slows for a reduced speed limit",
    "grade_change":        "the road runs up or down a noticeable gradient",
    "adverse_weather":     "rain, fog or night reducing visibility",
    "pedestrian_crossing": "a pedestrian crosses the road ahead",
}


# ---------------------------------------------------------------------------

# Generation cap, applied identically to every model.
#
# Must be large enough that a verbose model still reaches its delimited
# answer: at 256 one model's prose was truncated before the ANSWER line on
# ~48% of probes, which scored as model failure when it was a harness
# artefact. It must still be finite, because without any cap another model
# generated to its context limit on the text-only prompt (~2 min/probe).
MAX_TOKENS = 512


# Every raw model reply is archived here. Extraction rules have had to be
# revised three times, and each revision would otherwise have meant hours of
# re-querying to re-score answers the models had already given. With the raw
# text on disk a change to extract_number can be re-scored offline in seconds,
# and a reviewer can audit exactly what each model said.
RAW_SINK = None


def log_raw(task: str, gt, text: str, parsed) -> None:
    if RAW_SINK is None:
        return
    RAW_SINK.write(json.dumps({"task": task, "gt": gt,
                               "parsed": parsed, "text": text}) + "\n")
    RAW_SINK.flush()


def ask(model: str, prompt: str, image: Optional[Path], timeout: int,
        temperature: float = 0.0, max_tokens: int = MAX_TOKENS) -> str:
    """One zero-shot query.

    `num_predict` is capped. Without it a model may generate until its
    context limit: on the verbose text-only control prompt one model took
    ~2 minutes per probe versus ~1 second on the image tasks, which would
    have made the sweep take days. The cap is applied uniformly so the
    comparison across models stays like-for-like, and it is generous enough
    that the brief-reasoning prompts finish well inside it. Answers are
    extracted by pattern, so a truncated tail cannot corrupt a parsed value.
    """
    payload: Dict[str, Any] = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if image is not None:
        payload["images"] = [base64.b64encode(image.read_bytes()).decode()]
    r = requests.post(OLLAMA, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")


# A numeric token is either a RANGE ("10-15", "10 to 15") or a lone number.
# The range alternative must come first, and the lone-number alternative must
# refuse a leading '-' that follows a digit; otherwise "10-15 metres" is read
# as the number -15. See extract_number for why that mattered.
_TOKEN_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)"      # range
    r"|(?<![\d.\-])(-?\d+(?:\.\d+)?)"                          # lone number
)


def _values(text: str) -> List[float]:
    """Numeric tokens in order; a range collapses to its midpoint."""
    out = []
    for m in _TOKEN_RE.finditer(text.replace(",", "")):
        out.append((float(m.group(1)) + float(m.group(2))) / 2.0
                   if m.group(1) is not None else float(m.group(3)))
    return out


def extract_number(text: str) -> Optional[float]:
    """Pull the model's numeric answer out of free-form text.

    Every rule here exists because a model was mis-scored without it, and in
    each case the harness -- not the model -- was at fault:

      1. Prefer the explicit "ANSWER: <n>" tag the prompts request.
      2. Otherwise take the LAST number, not the first. A model reasoning in
         markdown opened with "## Step 1: ...", and scoring the step index as
         its prediction turned a correct 9.0 m/s into 1.0.
      3. Strip structural noise (headings, list markers, "Step 2") first, so
         it cannot be mistaken for a value.
      4. Read "10-15 metres" as the range it is, not as the number -15. One
         model answered almost exclusively in ranges; the negative reading
         failed the caller's plausibility bound and discarded 98% of its
         answers as unparseable, which looked like a model that could not
         follow instructions rather than a broken regex.

    A range collapses to its midpoint: it is the model's stated estimate, and
    picking either endpoint would bias the error in an arbitrary direction.
    """
    m = re.search(r"ANSWER:\s*([^\n]{0,40})", text, re.IGNORECASE)
    if m:
        # First value after the tag: that is the answer. A later one is a
        # unit conversion or an aside ("100 metres, about 328 feet").
        vs = _values(m.group(1))
        if vs:
            return vs[0]
    cleaned = re.sub(r"(?im)^\s*#{1,6}\s*step\s*\d+\s*:?", " ", text)
    cleaned = re.sub(r"(?im)^\s*step\s*\d+\s*[:.]", " ", cleaned)
    cleaned = re.sub(r"(?m)^\s*\d+[.)]\s", " ", cleaned)
    vs = _values(cleaned)
    return vs[-1] if vs else None


# Retained under the old name for callers that genuinely want the first
# number; nothing in this benchmark should use it.
def first_number(text: str) -> Optional[float]:
    m = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m[0]) if m else None


def parse_category(text: str) -> Optional[str]:
    low = text.lower()
    for c in CATEGORIES:                       # exact token first
        if c in low.replace(" ", "_"):
            return c
    hits = [(c, low.count(w)) for c in CATEGORIES
            for w in [c.split("_")[0]] if w in low]
    return max(hits, key=lambda t: t[1])[0] if hits else None


def parse_safe(text: str) -> Optional[int]:
    low = text.strip().lower()
    if re.search(r"\b(unsafe|not safe|dangerous|too close|critical)\b", low):
        return 0
    if re.search(r"\b(safe|adequate|sufficient|comfortable)\b", low):
        return 1
    if low.startswith("yes"):
        return 1
    if low.startswith("no"):
        return 0
    return None


# ---------------------------------------------------------------------------

def load_pool(dataset: Path, split: str) -> List[Dict[str, Any]]:
    """Frames of the given split, with GT and the front-camera path."""
    with open(dataset / "infos" / f"cf_infos_{split}.json") as f:
        scenes = json.load(f)["data"]
    pool = []
    for sc in scenes:
        cat = sc.get("scene_info", {}).get("scenario_category")
        for fr in sc["scene_data"]:
            img = fr["sensor_paths"].get("RGB-CAM_FRONT")
            if not img:
                continue
            k = fr["kinematics"]
            pool.append({
                "scene_id": sc["scene_id"], "frame_idx": fr["frame_idx"],
                "image": dataset / img, "category": cat,
                "spacing": k["spacing_m"], "ego_speed": k["ego_speed_mps"],
                "leader_speed": k["leader_speed_mps"], "ttc": k["ttc_s"],
                "vqa": dataset / fr["vqa_path"] if "vqa_path" in fr else None,
            })
    return pool


def is_safe(rec: Dict[str, Any]) -> int:
    """Dataset's own safety rule: TTC >= 3 s (see stage5 template)."""
    return 1 if rec["ttc"] >= 3.0 else 0


# ---------------------------------------------------------------------------

def run_spacing(model, pool, n, timeout, rng, train_median) -> Dict:
    sample = rng.sample(pool, min(n, len(pool)))
    # A terse "reply with a number only" instruction drives this model to a
    # constant answer regardless of the image, which would understate its
    # ability. Allowing brief reasoning before a delimited answer removes
    # that artefact, so the reported error reflects capability rather than
    # output formatting.
    prompt = (
        "The image is from a forward-facing camera on a car that is following "
        "another vehicle.\n"
        "Look at the vehicle directly ahead in the same lane. Judge how far "
        "away it is, using the lane markings and the vehicle's size for scale.\n"
        "Answer in at most two short sentences, then end your reply with\n"
        "ANSWER: <number> metres")
    preds, gts, fails = [], [], 0
    for i, r in enumerate(sample, 1):
        try:
            txt = ask(model, prompt, r["image"], timeout)
            v = extract_number(txt)
        except Exception as e:
            logger.debug("spacing call failed: %s", e); txt, v = "", None
        log_raw("spacing", r["spacing"], txt, v)
        if v is None or not (0 <= v <= 300):
            fails += 1; continue
        preds.append(v); gts.append(r["spacing"])
        if i % 25 == 0:
            logger.info("  spacing %d/%d", i, len(sample))
    preds, gts = np.array(preds), np.array(gts)
    base = np.full_like(gts, train_median)
    return {
        "n_scored": int(len(preds)), "n_unparsed": fails,
        "mae":            float(np.abs(preds - gts).mean()) if len(preds) else None,
        "rmse":           float(np.sqrt(((preds - gts) ** 2).mean())) if len(preds) else None,
        "within_5m":      float((np.abs(preds - gts) <= 5).mean()) if len(preds) else None,
        "baseline_mae":   float(np.abs(base - gts).mean()) if len(preds) else None,
        "baseline_desc":  f"predict training median gap ({train_median:.1f} m)",
        "pred_mean":      float(preds.mean()) if len(preds) else None,
        "gt_mean":        float(gts.mean()) if len(preds) else None,
    }


def run_safety(model, pool, n, timeout, rng) -> Dict:
    unsafe = [r for r in pool if is_safe(r) == 0]
    safe   = [r for r in pool if is_safe(r) == 1]
    half = min(n // 2, len(unsafe), len(safe))
    sample = rng.sample(unsafe, half) + rng.sample(safe, half)
    rng.shuffle(sample)
    prompt = (
        "You are looking at the forward-facing camera of a car.\n"
        "Considering the distance to the vehicle ahead and how fast it is "
        "being approached, is the current following distance SAFE or UNSAFE?\n"
        "Reply with exactly one word: SAFE or UNSAFE.")
    y, yh, fails = [], [], 0
    for i, r in enumerate(sample, 1):
        try:
            txt = ask(model, prompt, r["image"], timeout)
            p = parse_safe(txt)
        except Exception:
            txt, p = "", None
        log_raw("safety", is_safe(r), txt, p)
        if p is None:
            fails += 1; continue
        y.append(is_safe(r)); yh.append(p)
        if i % 25 == 0:
            logger.info("  safety %d/%d", i, len(sample))
    y, yh = np.array(y), np.array(yh)
    out = {"n_scored": int(len(y)), "n_unparsed": fails,
           "balanced_sample": True, "baseline_balanced_acc": 0.5}
    if len(y):
        rec = {c: float((yh[y == c] == c).mean()) if (y == c).any() else None
               for c in (0, 1)}
        out.update({
            "accuracy":          float((y == yh).mean()),
            "balanced_accuracy": float(np.mean([v for v in rec.values() if v is not None])),
            "recall_unsafe":     rec[0], "recall_safe": rec[1],
            "predicted_safe_rate": float(yh.mean()),
        })
    return out


def run_scenario(model, pool, n, timeout, rng) -> Dict:
    by_cat = defaultdict(list)
    for r in pool:
        if r["category"]:
            by_cat[r["category"]].append(r)
    per = max(n // max(len(by_cat), 1), 1)
    sample = [r for c in by_cat for r in rng.sample(by_cat[c], min(per, len(by_cat[c])))]
    rng.shuffle(sample)
    options = "\n".join(f"  {c}: {CATEGORY_GLOSS[c]}" for c in CATEGORIES)
    prompt = ("You are looking at the forward-facing camera of a car that is "
              "following another vehicle.\nWhich ONE of these driving "
              f"scenarios is occurring?\n{options}\n"
              "Reply with exactly one category name from the list.")
    y, yh, fails = [], [], 0
    for i, r in enumerate(sample, 1):
        try:
            txt = ask(model, prompt, r["image"], timeout)
            p = parse_category(txt)
        except Exception:
            txt, p = "", None
        log_raw("scenario", r["category"], txt, p)
        if p is None:
            fails += 1; continue
        y.append(r["category"]); yh.append(p)
        if i % 25 == 0:
            logger.info("  scenario %d/%d", i, len(sample))
    out = {"n_scored": len(y), "n_unparsed": fails,
           "baseline_acc": round(1 / len(CATEGORIES), 4)}
    if y:
        out["accuracy"] = float(np.mean([a == b for a, b in zip(y, yh)]))
        per_cls = {}
        for c in CATEGORIES:
            idx = [j for j, a in enumerate(y) if a == c]
            if idx:
                per_cls[c] = float(np.mean([yh[j] == c for j in idx]))
        out["per_class_recall"] = per_cls
        out["prediction_histogram"] = dict(Counter(yh))
    return out


def run_leakage(model, pool, n, timeout, rng) -> Dict:
    """Control: the ORIGINAL question, text only, no image."""
    sample = [r for r in rng.sample(pool, min(n * 4, len(pool))) if r["vqa"]][:n]
    preds, gts, fails = [], [], 0
    for i, r in enumerate(sample, 1):
        try:
            q = json.load(open(r["vqa"]))
            item = next(x for x in q["qa_pairs"] if x["qa_type"] == "speed_prediction")
            txt = ask(model, item["question"] +
                      "\nAnswer in at most two short sentences, then end your "
                      "reply with\nANSWER: <number> m/s",
                      None, timeout)
            v = extract_number(txt)
            gt = item["answer_value"]
        except Exception:
            txt, v, gt = "", None, None
        log_raw("leakage", gt, txt, v)
        if v is None or not (0 <= v <= 60):
            fails += 1; continue
        preds.append(v); gts.append(gt)
        if i % 25 == 0:
            logger.info("  leakage %d/%d", i, len(sample))
    preds, gts = np.array(preds), np.array(gts)
    return {"n_scored": int(len(preds)), "n_unparsed": fails,
            "mae_no_image": float(np.abs(preds - gts).mean()) if len(preds) else None,
            "note": ("text-only, no image; the question states the current "
                     "speed, so low error here indicates the released "
                     "phrasing does not require perception")}


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", default="./vlm_eval")
    p.add_argument("--model", default="qwen2.5vl:7b")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=150, help="probes per task")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tasks", nargs="+",
                   default=["spacing", "safety", "scenario", "leakage"])
    args = p.parse_args()

    dataset = Path(args.dataset); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "raw"; raw_dir.mkdir(exist_ok=True)
    global RAW_SINK
    RAW_SINK = open(raw_dir / f"raw_{args.model.replace(':', '_')}.jsonl", "w")
    rng = random.Random(args.seed)

    pool = load_pool(dataset, args.split)
    logger.info("%s split: %d frames with a front camera", args.split, len(pool))
    train_pool = load_pool(dataset, "train")
    train_median = float(np.median([r["spacing"] for r in train_pool]))

    results: Dict[str, Any] = {}
    for task in args.tasks:
        t0 = time.time()
        logger.info("=== %s ===", task)
        fn = {"spacing":  lambda: run_spacing(args.model, pool, args.n, args.timeout, rng, train_median),
              "safety":   lambda: run_safety(args.model, pool, args.n, args.timeout, rng),
              "scenario": lambda: run_scenario(args.model, pool, args.n, args.timeout, rng),
              "leakage":  lambda: run_leakage(args.model, pool, args.n, args.timeout, rng)}.get(task)
        if fn is None:
            logger.warning("unknown task %s", task); continue
        results[task] = fn()
        results[task]["wallclock_s"] = round(time.time() - t0, 1)
        logger.info("%s -> %s", task,
                    {k: v for k, v in results[task].items()
                     if k in ("mae", "baseline_mae", "balanced_accuracy",
                              "accuracy", "baseline_acc", "mae_no_image")})

    payload = {"model": args.model, "split": args.split, "n_per_task": args.n,
               "seed": args.seed, "results": results}
    with open(out / f"vlm_{args.model.replace(':', '_')}.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Zero-shot VLM gap analysis ===")
    print(f"model: {args.model}   split: {args.split}\n")
    if "spacing" in results:
        r = results["spacing"]
        print(f"spacing regression  MAE {r['mae']:.2f} m   "
              f"(median-baseline {r['baseline_mae']:.2f} m)   "
              f"within 5 m: {100*r['within_5m']:.1f}%")
    if "safety" in results and "balanced_accuracy" in results["safety"]:
        r = results["safety"]
        print(f"safety (balanced)   bal.acc {100*r['balanced_accuracy']:.1f}%   "
              f"(chance 50.0%)   recall unsafe {100*(r['recall_unsafe'] or 0):.1f}%")
    if "scenario" in results and "accuracy" in results["scenario"]:
        r = results["scenario"]
        print(f"scenario (9-way)    acc {100*r['accuracy']:.1f}%   "
              f"(chance {100*r['baseline_acc']:.1f}%)")
    if "leakage" in results:
        r = results["leakage"]
        print(f"leakage control     text-only MAE {r['mae_no_image']:.3f} m/s "
              f"(no image seen)")
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
