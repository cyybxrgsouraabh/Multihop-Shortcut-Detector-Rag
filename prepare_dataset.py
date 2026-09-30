import json
from itertools import islice
from datasets import load_dataset

NUM_ROWS_PER_DATASET = 250

def process_hotpotqa(n=NUM_ROWS_PER_DATASET):
    """Load HotpotQA in streaming mode and normalise the first n rows."""
    ds = load_dataset('hotpotqa/hotpot_qa', 'distractor', split='train', streaming=True)
    data = []
    for row in islice(ds, n):
        # Format supporting facts
        supp_facts = [
            {"title": t, "sent_id": s}
            for t, s in zip(row['supporting_facts']['title'],
                            row['supporting_facts']['sent_id'])
        ]
        # Format context passages
        contexts = [
            {"title": t, "sentences": s}
            for t, s in zip(row['context']['title'],
                            row['context']['sentences'])
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
    """Load Musique in streaming mode and normalise the first n rows."""
    ds = load_dataset('bdsaglam/musique', split='train', streaming=True)
    data = []
    for row in islice(ds, n):
        # Format supporting facts
        supp_facts = [
            {"title": p["title"], "idx": p["idx"]}
            for p in row["paragraphs"] if p["is_supporting"]
        ]
        # Format context passages
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

if __name__ == "__main__":
    print("Processing HotpotQA (streaming)...")
    hotpot_data = process_hotpotqa()
    print(f"  → got {len(hotpot_data)} rows")

    print("Processing Musique (streaming)...")
    musique_data = process_musique()
    print(f"  → got {len(musique_data)} rows")

    combined_data = hotpot_data + musique_data

    output_file = "combined_dataset.json"
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2)

    print(f"\nSaved {len(combined_data)} rows to {output_file}")
