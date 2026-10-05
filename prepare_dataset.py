import json
import os
from itertools import islice
from datasets import load_dataset

NUM_ROWS_PER_DATASET = 1000
OUTPUT_FILE = "data/raw/combined_dataset.jsonl"

def process_hotpotqa(n=NUM_ROWS_PER_DATASET):
    print(f"Processing HotpotQA (streaming {n} rows)...")
    ds = load_dataset('hotpotqa/hotpot_qa', 'distractor', split='train', streaming=True)
    data = []
    for row in islice(ds, n):
        supp_facts = [
            {"title": t, "sent_id": s}
            for t, s in zip(row['supporting_facts']['title'], row['supporting_facts']['sent_id'])
        ]
        contexts = [
            {"title": t, "sentences": s}
            for t, s in zip(row['context']['title'], row['context']['sentences'])
        ]
        data.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "supporting_facts": supp_facts,
            "context_passages": contexts,
            "source_dataset": "hotpotqa"
        })
    return data

def process_musique(n=NUM_ROWS_PER_DATASET):
    print(f"Processing Musique (streaming {n} rows)...")
    ds = load_dataset('bdsaglam/musique', split='train', streaming=True)
    data = []
    for row in islice(ds, n):
        supp_facts = [
            {"title": p["title"], "idx": p["idx"]}
            for p in row["paragraphs"] if p["is_supporting"]
        ]
        contexts = [
            {"title": p["title"], "paragraph_text": p["paragraph_text"]}
            for p in row["paragraphs"]
        ]
        data.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "supporting_facts": supp_facts,
            "context_passages": contexts,
            "source_dataset": "musique"
        })
    return data

def process_2wikimultihop(n=NUM_ROWS_PER_DATASET):
    print(f"Processing 2WikiMultihopQA (loading {n} rows from data/raw/dev.json)...")
    data = []
    with open("data/raw/dev.json", "r") as f:
        raw_data = json.load(f)
    
    for row in raw_data[:n]:
        supp_facts = [
            {"title": sf[0], "sent_id": sf[1]}
            for sf in row.get("supporting_facts", [])
        ]
        contexts = [
            {"title": ctx[0], "sentences": ctx[1]}
            for ctx in row.get("context", [])
        ]
        data.append({
            "id": row["_id"],
            "question": row["question"],
            "answer": row["answer"],
            "supporting_facts": supp_facts,
            "context_passages": contexts,
            "source_dataset": "2wikimultihopqa"
        })
    return data

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    hotpot_data = process_hotpotqa()
    print(f"  -> got {len(hotpot_data)} rows")

    musique_data = process_musique()
    print(f"  -> got {len(musique_data)} rows")

    wiki_data = process_2wikimultihop()
    print(f"  -> got {len(wiki_data)} rows")

    combined_data = hotpot_data + musique_data + wiki_data

    # Write as JSONL
    with open(OUTPUT_FILE, 'w') as f:
        for item in combined_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved {len(combined_data)} rows to {OUTPUT_FILE} (JSONL format)")
