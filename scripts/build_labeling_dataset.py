"""
scripts/build_labeling_dataset.py

Extracts 10 rows from each source dataset (hotpotqa, musique, 2wikimultihopqa)
and runs them through single-hop and multi-hop pipelines, computing all signals
required for the labeling schema used to train the shortcut classifier.
"""
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

from src.config import RAW_DATA_PATH, PROCESSED_DATA_DIR
from src.hybrid_retriever import HybridRetriever
from src.pipelines.single_hop import SingleHopRAGPipeline
from src.pipelines.multi_hop import MultiHopRAGPipeline

# ─── Config ──────────────────────────────────────────────────────────────────

ROWS_PER_DATASET = 10
OUTPUT_PATH = PROCESSED_DATA_DIR / "labeling_dataset.jsonl"
JSON_OUTPUT_PATH = PROCESSED_DATA_DIR / "labeling_dataset.json"

INTERROGATIVES = ["what", "which", "who", "when", "where", "how", "why"]

# ─── NLP Helpers ─────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Lowercase, strip articles and punctuation."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())

def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return round(2 * precision * recall / (precision + recall), 4)

def compute_exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def get_interrogative_type(question: str) -> str:
    q_lower = question.lower().strip()
    for word in INTERROGATIVES:
        if q_lower.startswith(word):
            return word
    return "other"

def is_comparison(question: str) -> int:
    """Returns 1 if the question looks like a comparison."""
    comparison_words = ["first", "older", "newer", "earlier", "later",
                        "before", "after", "more", "less", "larger", "smaller",
                        "founded", "started", "born", "died"]
    q_lower = question.lower()
    return int(any(w in q_lower for w in comparison_words))

def extract_entities_simple(question: str) -> List[str]:
    """Extract likely entities: capitalized word sequences."""
    # Rough NP heuristic: groups of words starting with a capital letter
    tokens = question.split()
    entities = []
    current_entity = []
    for token in tokens:
        clean = re.sub(r"[^\w']", "", token)
        if clean and clean[0].isupper():
            current_entity.append(clean)
        else:
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
    if current_entity:
        entities.append(" ".join(current_entity))
    # Filter out common question words
    stopwords = {"The", "A", "An", "Who", "What", "Which", "Where", "When", "How", "Why", "Is", "Are", "Was", "Were"}
    return [e for e in entities if e not in stopwords and len(e) > 1]

def entity_overlap_ratio(query: str, top_doc_text: str) -> float:
    entities = extract_entities_simple(query)
    if not entities:
        return 0.0
    doc_lower = top_doc_text.lower()
    matched = sum(1 for e in entities if e.lower() in doc_lower)
    return round(matched / len(entities), 4)

def has_all_query_entities(query: str, top_doc_text: str) -> int:
    entities = extract_entities_simple(query)
    if not entities:
        return 0
    doc_lower = top_doc_text.lower()
    return int(all(e.lower() in doc_lower for e in entities))

def compute_rrf_entropy(rrf_score_map: Dict[str, float]) -> float:
    """Shannon entropy of the RRF score distribution over retrieved docs."""
    scores = list(rrf_score_map.values())
    total = sum(scores)
    if total == 0:
        return 0.0
    probs = [s / total for s in scores]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    return round(entropy, 4)

def rrf_margin(sorted_rrf_items: List) -> float:
    """Margin between top-1 and top-2 RRF scores."""
    if len(sorted_rrf_items) < 2:
        return 0.0
    return round(sorted_rrf_items[0][1] - sorted_rrf_items[1][1], 6)

# ─── LLM Setup ───────────────────────────────────────────────────────────────

def get_groq_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")
    client = Groq(api_key=api_key)
    model = os.getenv("GROQ_MODEL_NAME", "qwen/qwen3.8-27b")

    def llm_fn(prompt: str, system_prompt: str = "") -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        max_retries = 4
        base_wait = 1
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model, messages=messages, temperature=0.0
                )
                usage = response.usage
                return {
                    "text": response.choices[0].message.content or "",
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                }
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  [LLM Error after {max_retries} attempts]: {e}")
                    return {"text": "API Failed", "prompt_tokens": 0, "completion_tokens": 0}
                sleep_time = base_wait * (2 ** attempt)
                print(f"  [Retry {attempt+1}/{max_retries} in {sleep_time}s]: {e}")
                time.sleep(sleep_time)
    return llm_fn


def llm_judge(llm_fn, question: str, ground_truth: str, prediction: str) -> Dict[str, Any]:
    """Use LLM as judge to score prediction quality."""
    prompt = (
        f"You are an expert evaluator for question answering systems.\n\n"
        f"Question: {question}\n"
        f"Ground Truth Answer: {ground_truth}\n"
        f"Predicted Answer: {prediction}\n\n"
        f"Does the predicted answer correctly answer the question? "
        f"Score it: 1 if correct or mostly correct, 0 if wrong or hallucinated. "
        f"Respond ONLY in this format:\n"
        f"SCORE: <0 or 1>\n"
        f"REASONING: <one sentence explaining your judgment>"
    )
    resp = llm_fn(prompt, system_prompt="You are a strict QA evaluator. Respond only in the requested format.")
    text = resp.get("text", "")
    score_match = re.search(r"SCORE:\s*([01])", text)
    reason_match = re.search(r"REASONING:\s*(.*)", text, re.DOTALL)
    score = int(score_match.group(1)) if score_match else 0
    reasoning = reason_match.group(1).strip() if reason_match else text.strip()
    return {"judge_score": score, "judge_reasoning": reasoning}

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_test_set(dataset_path: Path, n_per_dataset: int = 10) -> List[Dict]:
    counts = {}
    selected = []
    with open(dataset_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            src = item.get("source_dataset", "unknown")
            if counts.get(src, 0) < n_per_dataset:
                selected.append(item)
                counts[src] = counts.get(src, 0) + 1
            if all(v >= n_per_dataset for v in counts.values()) and len(counts) >= 3:
                break
    print(f"Loaded test set: {counts}")
    return selected

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("Loading HybridRetriever (will auto-build BM25 if needed)...")
    retriever = HybridRetriever()

    llm_fn = get_groq_llm()

    def retriever_fn(query: str, top_k: int = 3):
        return retriever.query(query, top_k=top_k)

    single_hop_pipe = SingleHopRAGPipeline(retriever_fn, llm_fn)
    multi_hop_pipe = MultiHopRAGPipeline(retriever_fn, llm_fn)

    test_set = load_test_set(RAW_DATA_PATH, n_per_dataset=ROWS_PER_DATASET)
    print(f"Total queries to evaluate: {len(test_set)}\n")

    results = []

    for i, item in enumerate(test_set):
        qid = item["id"]
        question = item["question"]
        ground_truth = item["answer"]
        source = item["source_dataset"]

        print(f"\n[{i+1}/{len(test_set)}] ({source}) {question[:80]}...")

        # ── Retrieval Signals ──────────────────────────────────────────────
        print("  Computing retrieval signals...")
        top_docs, raw_scores = retriever.query_with_scores(question, top_k=5)

        top1_doc = top_docs[0] if top_docs else {}
        top1_id = top1_doc.get("id", "")
        top1_text = top1_doc.get("text", "")
        top1_title = top1_doc.get("title", "")

        dense_map = raw_scores["dense_score_map"]
        bm25_map = raw_scores["bm25_score_map"]
        rrf_map = raw_scores["rrf_score_map"]

        dense_sim_top1 = round(dense_map.get(top1_id, 0.0), 6)
        bm25_score_top1 = round(bm25_map.get(top1_id, 0.0), 6)
        rrf_score_top1 = round(rrf_map.get(top1_id, 0.0), 6)

        sorted_rrf = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)
        rrf_margin_val = rrf_margin(sorted_rrf)
        rrf_entropy = compute_rrf_entropy(rrf_map)

        entities = extract_entities_simple(question)
        ent_overlap = entity_overlap_ratio(question, f"{top1_title} {top1_text}")
        all_ents_present = has_all_query_entities(question, f"{top1_title} {top1_text}")
        query_token_count = len(question.split())
        is_comp = is_comparison(question)
        interrog_type = get_interrogative_type(question)

        retrieval_signals = {
            "top1_doc_id": top1_id,
            "dense_similarity_top1": dense_sim_top1,
            "bm25_score_top1": bm25_score_top1,
            "rrf_score_top1": rrf_score_top1,
            "rrf_margin": rrf_margin_val,
            "rrf_score_entropy": rrf_entropy,
            "entity_overlap_ratio": ent_overlap,
            "has_all_query_entities": all_ents_present,
            "query_token_count": query_token_count,
            "is_comparison": is_comp,
            "interrogative_type": interrog_type,
            "detected_entities": entities,
        }

        # ── Single-Hop ─────────────────────────────────────────────────────
        print("  Running Single-Hop...")
        sh_result = single_hop_pipe.run(question, top_k=3)
        sh_pred = sh_result.get("prediction", "")
        sh_f1 = compute_f1(sh_pred, ground_truth)
        sh_em = compute_exact_match(sh_pred, ground_truth)
        print(f"    -> Pred: {sh_pred[:60]} | F1={sh_f1} | EM={sh_em}")
        print("  Running LLM judge for single-hop...")
        sh_judge = llm_judge(llm_fn, question, ground_truth, sh_pred)

        single_hop_entry = {
            "prediction": sh_pred,
            "retrieved_doc_ids": sh_result.get("retrieved_ids", []),
            "f1_score": sh_f1,
            "exact_match": sh_em,
            "judge_score": sh_judge["judge_score"],
            "judge_reasoning": sh_judge["judge_reasoning"],
            "total_tokens": sh_result.get("total_tokens", 0),
            "latency_sec": round(sh_result.get("latency_sec", 0.0), 4),
        }

        # ── Multi-Hop ──────────────────────────────────────────────────────
        print("  Running Multi-Hop (IR-CoT)...")
        mh_result = multi_hop_pipe.run(question, top_k=3)
        mh_pred = mh_result.get("prediction", "")
        mh_f1 = compute_f1(mh_pred, ground_truth)
        mh_em = compute_exact_match(mh_pred, ground_truth)
        print(f"    -> Pred: {mh_pred[:60]} | F1={mh_f1} | EM={mh_em}")
        print("  Running LLM judge for multi-hop...")
        mh_judge = llm_judge(llm_fn, question, ground_truth, mh_pred)

        multi_hop_entry = {
            "prediction": mh_pred,
            "hop_queries": mh_result.get("hop_queries", []),
            "retrieved_doc_ids": mh_result.get("retrieved_ids", []),
            "f1_score": mh_f1,
            "exact_match": mh_em,
            "judge_score": mh_judge["judge_score"],
            "judge_reasoning": mh_judge["judge_reasoning"],
            "total_tokens": mh_result.get("total_tokens", 0),
            "latency_sec": round(mh_result.get("latency_sec", 0.0), 4),
        }

        # ── Final Record ───────────────────────────────────────────────────
        record = {
            "example_id": f"{source}_{qid}",
            "dataset": source,
            "question": question,
            "question_type": "comparison" if is_comp else "factual",
            "ground_truth_answer": ground_truth,
            "retrieval_signals": retrieval_signals,
            "single_hop": single_hop_entry,
            "multi_hop": multi_hop_entry,
        }
        results.append(record)

        # Write incrementally so we don't lose data on crash
        with open(OUTPUT_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"  Saved record {i+1}/{len(test_set)}")

    print(f"\n{'='*60}")
    print(f"Done! {len(results)} records saved to {OUTPUT_PATH}")

    # Write pretty-printed JSON for easy preview
    with open(JSON_OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Preview JSON saved to {JSON_OUTPUT_PATH}")


if __name__ == "__main__":
    # Clear output files if they exist from a previous run
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
    if JSON_OUTPUT_PATH.exists():
        JSON_OUTPUT_PATH.unlink()
    main()
