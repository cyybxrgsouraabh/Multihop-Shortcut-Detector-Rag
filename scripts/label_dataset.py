"""
scripts/label_dataset.py

Reads the labeling dataset (labeling_dataset.jsonl) and assigns a final
ternary routing label to each query using a 3-signal arbitration system.

Decision matrix
───────────────
A pipeline run is "Correct" ONLY IF:
    judge_score == 1
    AND (f1_score >= F1_THRESHOLD  OR  contains_gold == 1)

Labels
───────
  label =  1  →  Shortcut:   single-hop is Correct → route to cheap pipeline
  label =  0  →  Multi-hop:  single-hop is Incorrect, multi-hop is Correct
  label = -1  →  Discard:    both pipelines are Incorrect (ambiguous data)

Output is a flat feature dataset for training/evaluating the shortcut classifier.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DATA_DIR

# ─── Config ──────────────────────────────────────────────────────────────────

INPUT_PATH   = PROCESSED_DATA_DIR / "labeling_dataset.jsonl"
OUTPUT_JSONL = PROCESSED_DATA_DIR / "classifier_dataset.jsonl"
OUTPUT_JSON  = PROCESSED_DATA_DIR / "classifier_dataset.json"

# F1 threshold: one of the two signals required for a "Correct" verdict
F1_THRESHOLD = 0.70

# ─── 3-Signal Arbitration ─────────────────────────────────────────────────────

def is_pipeline_correct(pipeline_result: dict) -> bool:
    """
    Return True (Correct) only if BOTH conditions hold:
      1. LLM judge agrees (judge_score == 1)
      2. Token-level F1 >= F1_THRESHOLD  OR  gold answer contained in prediction
    This eliminates:
      - False positives: judge hallucinating a pass score
      - False negatives: judge overly strict when contains_gold would rescue
    """
    judge_ok       = pipeline_result.get("judge_score", 0) == 1
    f1_ok          = pipeline_result.get("f1_score", 0.0) >= F1_THRESHOLD
    contains_ok    = bool(pipeline_result.get("contains_gold", 0))
    return judge_ok and (f1_ok or contains_ok)


def assign_label(record: dict) -> int:
    """
    Apply the 3-signal decision matrix and return:
      1  → single-hop Correct (shortcut exists)
      0  → single-hop Incorrect, multi-hop Correct (need multi-hop)
     -1  → both Incorrect (discard — ambiguous training signal)
    """
    sh_correct = is_pipeline_correct(record.get("single_hop", {}))
    mh_correct = is_pipeline_correct(record.get("multi_hop", {}))

    if sh_correct:
        return 1
    elif mh_correct:
        return 0
    else:
        return -1


def flatten_record(record: dict, label: int) -> dict:
    """
    Produces a flat feature record for classifier training strictly matching the schema:
    {
      "id": ...,
      "question": ...,
      "rrf_score_top1": ...,
      "rrf_margin": ...,
      "rrf_score_entropy": ...,
      "entity_overlap_ratio": ...,
      "has_all_query_entities": ...,
      "query_token_count": ...,
      "is_comparison": ...,
      "label": ...
    }
    """
    signals = record.get("retrieval_signals", {})
    return {
        "id":                     record.get("example_id", ""),
        "question":               record.get("question", ""),
        "rrf_score_top1":         signals.get("rrf_score_top1", 0.0),
        "rrf_margin":             signals.get("rrf_margin", 0.0),
        "rrf_score_entropy":      signals.get("rrf_score_entropy", 0.0),
        "entity_overlap_ratio":   signals.get("entity_overlap_ratio", 0.0),
        "has_all_query_entities": signals.get("has_all_query_entities", 0),
        "query_token_count":      signals.get("query_token_count", 0),
        "is_comparison":          signals.get("is_comparison", 0),
        "label":                  label,
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH}")
        print("Run scripts/build_labeling_dataset.py first.")
        sys.exit(1)

    records = []
    with open(INPUT_PATH, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {INPUT_PATH}")

    labeled = []
    label_counts = {1: 0, 0: 0, -1: 0}

    for record in records:
        label = assign_label(record)
        flat  = flatten_record(record, label)
        labeled.append(flat)
        label_counts[label] += 1

    # ── Stats ─────────────────────────────────────────────────────────────
    total = len(labeled)
    print(f"\n{'='*54}")
    print(f"3-Signal Labeling Summary  (F1 threshold = {F1_THRESHOLD})")
    print(f"{'='*54}")
    print(f"  Total queries                  : {total}")
    print(f"  Label  1  (shortcut / SH ok)   : {label_counts[ 1]:3d}  ({100*label_counts[ 1]/total:5.1f}%)")
    print(f"  Label  0  (multi-hop required)  : {label_counts[ 0]:3d}  ({100*label_counts[ 0]/total:5.1f}%)")
    print(f"  Label -1  (both wrong / discard): {label_counts[-1]:3d}  ({100*label_counts[-1]/total:5.1f}%)")

    # Per-dataset breakdown
    per_ds = defaultdict(lambda: {1: 0, 0: 0, -1: 0})
    for record, flat in zip(records, labeled):
        per_ds[record.get("dataset", "unknown")][flat["label"]] += 1

    print(f"\n  Per-dataset breakdown (SH=shortcut, MH=multi-hop, DX=discard):")
    print(f"  {'Dataset':25s}  SH   MH   DX")
    print(f"  {'-'*43}")
    for ds, counts in sorted(per_ds.items()):
        print(f"  {ds:25s}  {counts[1]:3d}  {counts[0]:3d}  {counts[-1]:3d}")

    # ── Write outputs ─────────────────────────────────────────────────────
    with open(OUTPUT_JSONL, "w") as f:
        for item in labeled:
            f.write(json.dumps(item) + "\n")
    print(f"\n  JSONL saved : {OUTPUT_JSONL}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(labeled, f, indent=2, ensure_ascii=False)
    print(f"  JSON  saved : {OUTPUT_JSON}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
