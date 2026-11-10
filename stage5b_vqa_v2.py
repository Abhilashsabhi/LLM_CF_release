"""
stage5b_vqa_v2.py
==================
Regenerate the question--answer release so that answering requires the imagery.

Why the first release had to be replaced
---------------------------------------
Four of its five question types stated the queried quantity in the prompt
("travelling at 7.5 m/s ... gap of 16.4 m ... TTC 9999 s"), so a text-only model
answered by copying. Our own benchmark therefore had to build separate,
non-leaky probes, which left the released supervision and the released
benchmark measuring different things. This version fixes the supervision rather
than working around it: no question contains the quantity it asks for, and the
benchmark scores exactly these pairs.

Two families, because one is not enough
---------------------------------------
*Frame-level* questions are answerable from a single image: how far ahead the
lead vehicle is, whether one is present, what the visual conditions are.

*Clip-level* questions are answerable only from a short sequence. Time-to-
collision depends on closing speed, and a still image does not uniquely encode
relative longitudinal velocity -- two vehicles at the same gap may be
converging or separating. Safety, gap trend, relative motion and the
temporally-defined scenario categories therefore take a clip, not a frame.

Every answer is derived from the recorded ground truth, so it is exact; the
prompt never contains it.

    python stage5b_vqa_v2.py --dataset <root> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np

logger = logging.getLogger("vqa2")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] vqa2: %(message)s")

CATEGORIES = ["steady_follow", "stop_and_go", "aggressive_leader", "cut_in",
              "cut_out", "speed_limit_change", "grade_change",
              "adverse_weather", "pedestrian_crossing"]

CATEGORY_GLOSS = {
    "steady_follow": "following a lead vehicle at roughly constant speed",
    "stop_and_go": "congested stop-and-go, leader repeatedly halting",
    "aggressive_leader": "the lead vehicle brakes hard and suddenly",
    "cut_in": "another vehicle cuts into the lane ahead",
    "cut_out": "the lead vehicle leaves the lane, revealing a slower one",
    "speed_limit_change": "the lead vehicle slows for a reduced speed limit",
    "grade_change": "the road runs up or down a noticeable gradient",
    "adverse_weather": "rain, fog or night reducing visibility",
    "pedestrian_crossing": "a pedestrian crosses the road ahead",
}

# Which categories are defined by how the scene EVOLVES. A single frame cannot
# express them, so they are only ever asked at clip level.
TEMPORAL_CATEGORIES = {"stop_and_go", "aggressive_leader", "cut_in", "cut_out",
                       "speed_limit_change"}

Q = {
    "spacing_estimation":
        ("The image is from a forward-facing camera on a car that is following "
         "another vehicle.\nLook at the vehicle directly ahead in the same "
         "lane. Judge how far away it is, using the lane markings and the "
         "vehicle's size for scale.\nAnswer in at most two short sentences, "
         "then end your reply with\nANSWER: <number> metres"),
    "lead_presence":
        ("You are looking at the forward-facing camera of a car.\nIs there a "
         "vehicle in your own lane ahead of you, within roughly 60 metres?\n"
         "Reply with exactly one word: YES or NO."),
    "visual_conditions":
        ("You are looking at the forward-facing camera of a car.\nWhat are the "
         "visibility conditions?\nReply with exactly one word: CLEAR, RAIN, "
         "FOG or NIGHT."),
    "gap_trend":
        ("These frames are consecutive views from a forward-facing camera, in "
         "time order.\nOver this sequence, is the gap to the vehicle ahead "
         "closing, opening, or staying about the same?\nReply with exactly one "
         "word: CLOSING, OPENING or STEADY."),
    "relative_motion":
        ("These frames are consecutive views from a forward-facing camera, in "
         "time order.\nIs the vehicle ahead moving faster than you, slower "
         "than you, or at about the same speed?\nReply with exactly one word: "
         "FASTER, SLOWER or SAME."),
    "safety_ttc":
        ("These frames are consecutive views from a forward-facing camera, in "
         "time order.\nConsidering both the distance to the vehicle ahead and "
         "how quickly you are closing on it, is the current following "
         "situation safe?\nReply with exactly one word: SAFE or UNSAFE."),
    "ego_action":
        ("These frames are consecutive views from a forward-facing camera, in "
         "time order.\nWhat should the driver do now?\nReply with exactly one "
         "word: ACCELERATE, HOLD or BRAKE."),
    "scenario_temporal":
        ("These frames are consecutive views from a forward-facing camera of a "
         "car following another vehicle, in time order.\nWhich ONE of these "
         "driving scenarios is occurring?\n"
         + "\n".join(f"  {c}: {CATEGORY_GLOSS[c]}" for c in CATEGORIES)
         + "\nReply with exactly one category name from the list."),
}


def weather_label(w: Dict) -> str:
    if float(w.get("sun_altitude_angle", 90)) < 0:
        return "NIGHT"
    if float(w.get("fog_density", 0)) > 30:
        return "FOG"
    if float(w.get("precipitation", 0)) > 20:
        return "RAIN"
    return "CLEAR"


def frame_level(sc: Dict, stride: int) -> Iterator[Dict]:
    info, data = sc["scene_info"], sc["scene_data"]
    wl = weather_label(info.get("weather", {}))
    for fr in data[::stride]:
        k = fr["kinematics"]
        img = fr["sensor_paths"].get("RGB-CAM_FRONT")
        if not img:
            continue
        base = {"scene_id": sc["scene_id"], "frame_idx": fr["frame_idx"],
                "level": "frame", "images": [img],
                "scenario_category": info.get("scenario_category"),
                "driving_style": info.get("driving_style"),
                "regime": info.get("regime")}
        yield {**base, "qa_type": "spacing_estimation",
               "question": Q["spacing_estimation"],
               "answer": f"ANSWER: {k['spacing_m']:.1f} metres",
               "answer_value": round(float(k["spacing_m"]), 3),
               "answer_units": "m"}
        yield {**base, "qa_type": "lead_presence",
               "question": Q["lead_presence"],
               "answer": "YES" if k["spacing_m"] <= 60.0 else "NO",
               "answer_value": None, "answer_units": None}
        yield {**base, "qa_type": "visual_conditions",
               "question": Q["visual_conditions"], "answer": wl,
               "answer_value": None, "answer_units": None}


def clip_level(sc: Dict, clip_frames: int, stride: int) -> Iterator[Dict]:
    info, data = sc["scene_info"], sc["scene_data"]
    cat = info.get("scenario_category")
    n = len(data)
    for start in range(0, n - clip_frames + 1, stride):
        win = data[start:start + clip_frames]
        imgs = [f["sensor_paths"].get("RGB-CAM_FRONT") for f in win]
        if any(i is None for i in imgs):
            continue
        s = np.array([f["kinematics"]["spacing_m"] for f in win])
        dv = np.array([f["kinematics"]["relative_speed_mps"] for f in win])
        a = np.array([f["kinematics"]["ego_acceleration_mps2"] for f in win])
        ttc = float(win[-1]["kinematics"]["ttc_s"])
        ds = float(s[-1] - s[0])
        mdv = float(dv.mean())
        base = {"scene_id": sc["scene_id"],
                "frame_idx": win[0]["frame_idx"],
                "frame_idx_end": win[-1]["frame_idx"],
                "level": "clip", "images": imgs,
                "scenario_category": cat,
                "driving_style": info.get("driving_style"),
                "regime": info.get("regime")}
        # Deadbands keep the labels decidable: a 0.4 m change over 2 s is not a
        # perceptible trend, and labelling it as one would teach noise.
        #
        # relative_speed_mps = ego_speed - leader_speed (positive → ego closing).
        # The question asks about the LEADER's speed relative to the ego:
        #   mdv < 0  → leader faster (pulling away)  → FASTER
        #   mdv > 0  → leader slower (ego closing)   → SLOWER
        trend = "CLOSING" if ds < -0.5 else "OPENING" if ds > 0.5 else "STEADY"
        rel = "FASTER" if mdv < -0.3 else "SLOWER" if mdv > 0.3 else "SAME"
        act = ("BRAKE" if a[-1] < -0.3 else
               "ACCELERATE" if a[-1] > 0.3 else "HOLD")
        yield {**base, "qa_type": "gap_trend", "question": Q["gap_trend"],
               "answer": trend, "answer_value": round(ds, 3), "answer_units": "m"}
        yield {**base, "qa_type": "relative_motion",
               "question": Q["relative_motion"], "answer": rel,
               "answer_value": round(mdv, 3), "answer_units": "m/s"}
        yield {**base, "qa_type": "safety_ttc", "question": Q["safety_ttc"],
               "answer": "UNSAFE" if ttc < 3.0 else "SAFE",
               "answer_value": round(ttc, 3), "answer_units": "s"}
        yield {**base, "qa_type": "ego_action", "question": Q["ego_action"],
               "answer": act, "answer_value": round(float(a[-1]), 3),
               "answer_units": "m/s^2"}
        # For temporal categories, only assign the scene's category when the
        # defining event is active in this clip.  Pre-event and post-event clips
        # look like steady following, so label them accordingly.
        event_in_clip = any(
            f["kinematics"].get("injection_active", False) for f in win
        )
        scenario_ans = (
            cat if (cat not in TEMPORAL_CATEGORIES or event_in_clip)
            else "steady_follow"
        )
        yield {**base, "qa_type": "scenario_temporal",
               "question": Q["scenario_temporal"], "answer": scenario_ans,
               "answer_value": None, "answer_units": None,
               "event_in_clip": event_in_clip,
               "temporally_defined": cat in TEMPORAL_CATEGORIES}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--clip-frames", type=int, default=20, help="2 s at 10 Hz")
    p.add_argument("--clip-stride", type=int, default=20)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   help="which splits to process (default: all three)")
    args = p.parse_args()

    root, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    totals: Dict[str, int] = {}

    for split in args.splits:
        f = root / "infos" / f"cf_infos_{split}.json"
        if not f.exists():
            continue
        scenes = json.load(open(f))["data"]
        # One JSONL per split rather than one file per frame: the release lives
        # on exFAT, where creating hundreds of thousands of small files is
        # pathologically slow, and a single stream is easier to consume.
        path = out / f"qa_{split}.jsonl"
        n = 0
        with open(path, "w") as fh:
            for sc in scenes:
                for rec in frame_level(sc, args.frame_stride):
                    fh.write(json.dumps(rec) + "\n"); n += 1
                    totals[rec["qa_type"]] = totals.get(rec["qa_type"], 0) + 1
                for rec in clip_level(sc, args.clip_frames, args.clip_stride):
                    fh.write(json.dumps(rec) + "\n"); n += 1
                    totals[rec["qa_type"]] = totals.get(rec["qa_type"], 0) + 1
        logger.info("%s: %d records -> %s", split, n, path.name)

    meta = {
        "version": 2,
        "design": ("No question states the quantity it asks for. Frame-level "
                   "questions are answerable from one image; clip-level "
                   "questions require a sequence because time-to-collision, "
                   "gap trend and relative motion depend on closing speed, "
                   "which a still image does not uniquely encode."),
        "clip_seconds": args.clip_frames / 10.0,
        "frame_stride": args.frame_stride, "clip_stride": args.clip_stride,
        "counts_by_type": totals,
        "temporal_categories": sorted(TEMPORAL_CATEGORIES),
        "answer_provenance": "derived from the recorded kinematic ground truth",
        "supersedes": ("v1, in which speed_prediction, safety_assessment, "
                       "behavior_explanation and lead_behavior stated the "
                       "queried quantities in the prompt"),
    }
    json.dump(meta, open(out / "qa_meta.json", "w"), indent=2)
    print("\nrecords by type:")
    for k, v in sorted(totals.items()):
        print(f"  {k:<22}{v:>9,}")
    print(f"\ntotal {sum(totals.values()):,}  ->  {out}")


if __name__ == "__main__":
    main()
