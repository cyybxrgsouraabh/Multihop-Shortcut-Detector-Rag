"""
scripts/build_exp_dataset.py

Creates 'exp_dataset' by combining the complete evaluation records from
labeling_dataset.jsonl with the final 'label' column from classifier_dataset.jsonl.

Outputs:
  - data/processed/exp_dataset.jsonl
  - data/processed/exp_dataset.json
"""

import json
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

LABELING_JSONL = PROCESSED_DIR / "labeling_dataset.jsonl"
CLASSIFIER_JSONL = PROCESSED_DIR / "classifier_dataset.jsonl"

OUTPUT_JSONL = PROCESSED_DIR / "exp_dataset.jsonl"
OUTPUT_JSON = PROCESSED_DIR / "exp_dataset.json"


def main():
    print(f"Reading labels from: {CLASSIFIER_JSONL}")
    # Load labels sequentially and map by ID
    id_to_label = {}
    classifier_labels_list = []
    with open(CLASSIFIER_JSONL, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                qid = item.get("id", "")
                lbl = item.get("label")
                id_to_label[qid] = lbl
                classifier_labels_list.append((qid, lbl))

    print(f"Loaded {len(classifier_labels_list)} records from classifier dataset.")

    print(f"\nReading full records from: {LABELING_JSONL}")
    exp_records = []
    label_counts = Counter()
    per_dataset_counts = Counter()

    with open(LABELING_JSONL, "r") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            eid = rec.get("example_id", "")

            # Match label by exact position if IDs match, else by ID lookup
            if idx < len(classifier_labels_list) and classifier_labels_list[idx][0] == eid:
                lbl = classifier_labels_list[idx][1]
            else:
                lbl = id_to_label.get(eid, -1)

            # Only include valid binary labels (0 or 1), discarding ambiguous (-1) rows
            if lbl not in (0, 1):
                continue

            # Insert 'label' as the final field of the record
            rec["label"] = lbl

            exp_records.append(rec)
            label_counts[lbl] += 1
            per_dataset_counts[(rec.get("dataset", "unknown"), lbl)] += 1

    print(f"Total unified records retained (labels 0 and 1 only): {len(exp_records)}")

    # Save to JSONL
    print(f"Writing JSONL to: {OUTPUT_JSONL}")
    with open(OUTPUT_JSONL, "w") as f:
        for item in exp_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Save to JSON
    print(f"Writing JSON to: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(exp_records, f, indent=2, ensure_ascii=False)

    # Print summary statistics
    total = len(exp_records)
    print("\n" + "=" * 55)
    print("EXP_DATASET GENERATION SUMMARY (BINARY LABELS: 0 or 1)")
    print("=" * 55)
    print(f"  Total valid training rows: {total}")
    print(f"  Label  1 (Shortcut):        {label_counts[1]:5d} ({label_counts[1]/total*100:5.1f}%)")
    print(f"  Label  0 (Multi-hop):       {label_counts[0]:5d} ({label_counts[0]/total*100:5.1f}%)")
    print("-" * 55)
    print("  Breakdown by source dataset:")
    datasets = sorted(set(d for d, _ in per_dataset_counts.keys()))
    print(f"  {'Dataset':22s} {'SH (1)':>8s} {'MH (0)':>8s} {'DX (-1)':>8s}")
    for ds in datasets:
        sh = per_dataset_counts.get((ds, 1), 0)
        mh = per_dataset_counts.get((ds, 0), 0)
        dx = per_dataset_counts.get((ds, -1), 0)
        print(f"  {ds:22s} {sh:8d} {mh:8d} {dx:8d}")
    print("=" * 55)
    print("Successfully created 'exp_dataset'!")


if __name__ == "__main__":
    main()
