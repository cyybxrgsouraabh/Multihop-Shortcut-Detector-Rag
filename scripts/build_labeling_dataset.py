"""
scripts/build_labeling_dataset.py

Builds the labeling dataset using the DeepSeek-Flash model across both single-hop
and multi-hop (IR-CoT) pipelines with anti-hallucination 3-signal evaluation.

Supports flexible batch processing (e.g. 500, 1500, 1000 rows, or custom sizes)
with balanced distribution across source datasets (hotpotqa, musique, 2wikimultihopqa).

Usage:
    # Run Batch 1: 500 rows (~167 per dataset)
    python scripts/build_labeling_dataset.py --batch_size 500

    # Run Batch 2: 1500 rows (500 per dataset)
    python scripts/build_labeling_dataset.py --batch_size 1500

    # Run Batch 3: 1000 rows (~333 per dataset)
    python scripts/build_labeling_dataset.py --batch_size 1000

    # Or specify per-dataset quota directly:
    python scripts/build_labeling_dataset.py --per_dataset 200
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Set

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    DEEPSEEK_MODEL_NAME,
    DEEPSEEK_BASE_URL,
    DEFAULT_MAX_TOKENS,
    SINGLE_HOP_MAX_TOKENS,
    MULTI_HOP_MAX_TOKENS,
    JUDGE_MAX_TOKENS,
)
from src.hybrid_retriever import HybridRetriever
from src.pipelines.single_hop import SingleHopRAGPipeline
from src.pipelines.multi_hop import MultiHopRAGPipeline

# ─── Config ──────────────────────────────────────────────────────────────────

INTER_QUERY_DELAY = 0.5  # seconds between queries (DeepSeek has high throughput)
OUTPUT_PATH = PROCESSED_DATA_DIR / "labeling_dataset.jsonl"
JSON_OUTPUT_PATH = PROCESSED_DATA_DIR / "labeling_dataset.json"

INTERROGATIVES = ["what", "which", "who", "when", "where", "how", "why"]
TARGET_DATASETS = ["hotpotqa", "musique", "2wikimultihopqa"]

# ─── NLP & Evaluation Helpers ────────────────────────────────────────────────

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
    comparison_words = [
        "first", "older", "newer", "earlier", "later",
        "before", "after", "more", "less", "larger", "smaller",
        "founded", "started", "born", "died", "both"
    ]
    q_lower = question.lower()
    return int(any(w in q_lower for w in comparison_words))


def extract_entities_simple(question: str) -> List[str]:
    """Extract likely entities: capitalized word sequences."""
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


def normalize_text(s: str) -> str:
    """Lowercase, remove punctuation, strip articles, standardise whitespace."""
    import string
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def check_contains_gold(prediction: str, gold_answer: str) -> int:
    """Return 1 if normalized gold answer is a substring of normalized prediction."""
    norm_pred = normalize_text(prediction)
    norm_gold = normalize_text(gold_answer)
    if not norm_gold:
        return 0
    return int(norm_gold in norm_pred)

# ─── LLM Setup (DeepSeek Flash) ───────────────────────────────────────────────

def get_llm(default_max_tokens: int = DEFAULT_MAX_TOKENS):
    """
    Configures and returns the DeepSeek-Flash LLM callable.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in .env")
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    model = os.getenv("DEEPSEEK_MODEL_NAME", DEEPSEEK_MODEL_NAME)
    print(f"Connected to DeepSeek API (Model: {model})...")

    def llm_fn(prompt: str, system_prompt: str = "", max_tokens: int = None) -> dict:
        tokens_limit = max_tokens if max_tokens is not None else default_max_tokens
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_retries = 5
        base_wait = 2.0

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=tokens_limit,
                )
                usage = response.usage
                content = response.choices[0].message.content or ""
                return {
                    "text": content,
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                }
            except Exception as e:
                sleep_time = base_wait * (2 ** attempt)
                if attempt == max_retries - 1:
                    print(f"  [DeepSeek Error after {max_retries} attempts]: {e}")
                    return {"text": "API Failed", "prompt_tokens": 0, "completion_tokens": 0}
                print(f"  [DeepSeek Error: {e}]: retrying in {sleep_time:.1f}s ({attempt+1}/{max_retries})...")
                time.sleep(sleep_time)

    return llm_fn


# Patterns indicating the model could not find, generate, or determine the answer
REFUSAL_PATTERNS = [
    r"\bcannot determine\b",
    r"\bcan not determine\b",
    r"\bcannot be determined\b",
    r"\bcan not be determined\b",
    r"\bcannot answer\b",
    r"\bcannot be answered\b",
    r"\bcan not be answered\b",
    r"\bunable to determine\b",
    r"\bunable to answer\b",
    r"\bnot provided\b",
    r"\bnot mentioned\b",
    r"\bnot stated\b",
    r"\bnot specified\b",
    r"\bnot specify\b",
    r"\bno information\b",
    r"\bdoes not contain\b",
    r"\bdo not contain\b",
    r"\bdoes not provide\b",
    r"\bdo not provide\b",
    r"\bdoes not state\b",
    r"\bdo not state\b",
    r"\bdoes not mention\b",
    r"\bdo not mention\b",
    r"\binsufficient information\b",
    r"\binsufficient context\b",
    r"\bnot enough information\b",
    r"\bnot enough context\b",
    r"\bunknown from the context\b",
    r"\bnot clear from the context\b",
    r"\bfailed to find\b",
    r"\bcould not find\b",
    r"\bcould not be found\b",
    r"\bno mention of\b",
    r"\bno record of\b",
    r"\bno direct information\b",
    r"\bcontext does not\b",
    r"\bprovided text does not\b",
    r"\bprovided context does not\b",
]

def is_refusal_or_unanswered(text: str) -> bool:
    """Detects any sentence implying the answer could not be found or generated."""
    clean = text.strip()
    if not clean:
        return True
    lower = clean.lower()
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


def llm_judge(llm_fn, question: str, ground_truth: str, prediction: str, f1_score: float = 0.0, contains_gold: int = 0) -> Dict[str, Any]:
    """
    Anti-hallucination LLM judge with Chain-of-Thought extraction:
    1. Short-circuits any prediction where the answer was not found/generated (score 0, 0 API calls).
    2. Sends all substantive predictions directly to DeepSeek judge for evaluation.
    """
    pred_clean = prediction.strip()

    # Rule: Empty, unanswered, or explicit refusal -> Automatically score 0 (NO API CALL)
    if not pred_clean:
        return {
            "judge_score": 0,
            "judge_reasoning": "Answer was not generated (prediction is empty).",
            "judge_extracted_prediction": "",
        }

    if is_refusal_or_unanswered(pred_clean) and contains_gold == 0:
        return {
            "judge_score": 0,
            "judge_reasoning": f"Prediction indicates answer could not be found ('{pred_clean[:60]}...').",
            "judge_extracted_prediction": "unanswered",
        }

    # All substantive predictions are judged by DeepSeek
    prompt = (
        f"You are a strict QA evaluation judge. Your task is to assess whether a predicted answer is correct.\n\n"
        f"Question: {question}\n"
        f"Gold Answer: {ground_truth}\n"
        f"Predicted Answer: {prediction}\n\n"
        f"Follow these steps:\n"
        f"1. Extract the core factual entity or value from the Predicted Answer (ignore filler phrases like 'Based on the context...' or 'The answer is...').\n"
        f"2. Compare that extracted entity to the Gold Answer.\n"
        f"3. Decide: is_correct = 1 if they refer to the same fact, 0 otherwise.\n\n"
        f"Respond ONLY with a JSON object on a single line. No markdown, no extra text:\n"
        f'{{"extracted_prediction": "<core entity from prediction>", "reasoning": "<one sentence>", "is_correct": <0 or 1>}}'
    )
    try:
        resp = llm_fn(
            prompt,
            system_prompt="You are a strict QA evaluator. Output ONLY a single JSON object, no markdown.",
            max_tokens=JUDGE_MAX_TOKENS,
        )
    except TypeError:
        resp = llm_fn(
            prompt,
            system_prompt="You are a strict QA evaluator. Output ONLY a single JSON object, no markdown.",
        )

    text = resp.get("text", "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        score = int(bool(parsed.get("is_correct", 0)))
        reasoning = parsed.get("reasoning", "").strip()
        extracted = parsed.get("extracted_prediction", "").strip()
        return {
            "judge_score": score,
            "judge_reasoning": reasoning,
            "judge_extracted_prediction": extracted,
        }
    except (json.JSONDecodeError, ValueError):
        score_match = re.search(r"is_correct[\"\s:]+([01])", text)
        reason_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', text)
        score = int(score_match.group(1)) if score_match else 0
        reasoning = reason_match.group(1).strip() if reason_match else text[:200]
        return {
            "judge_score": score,
            "judge_reasoning": reasoning,
            "judge_extracted_prediction": "",
        }

# ─── Data Loading with Balanced Source Quotas ─────────────────────────────────

def load_balanced_batch(
    dataset_path: Path,
    targets_per_dataset: Dict[str, int],
    existing_ids: Set[str],
) -> List[Dict]:
    """
    Loads unseen rows from dataset_path ensuring each source dataset reaches
    its specific quota targets_per_dataset[src].
    """
    counts = {src: 0 for src in TARGET_DATASETS}
    selected = []

    with open(dataset_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            src = item.get("source_dataset", "unknown")
            qid = item.get("id", "")
            example_id = f"{src}_{qid}"

            # Skip queries already evaluated in any previous run
            if example_id in existing_ids or qid in existing_ids:
                continue

            target_needed = targets_per_dataset.get(src, 0)
            if counts.get(src, 0) < target_needed:
                selected.append(item)
                counts[src] += 1

            # Stop when all dataset quotas are filled
            if all(counts[s] >= targets_per_dataset.get(s, 0) for s in TARGET_DATASETS):
                break

    print(f"Selected batch distribution: {counts} (Total: {len(selected)} new queries)")
    return selected

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_labeling(batch_size: int = None, per_dataset: int = None):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # 1. Calculate quotas per dataset
    if per_dataset is not None:
        targets_per_dataset = {s: per_dataset for s in TARGET_DATASETS}
        total_target = per_dataset * len(TARGET_DATASETS)
    elif batch_size is not None:
        base = batch_size // len(TARGET_DATASETS)
        remainder = batch_size % len(TARGET_DATASETS)
        targets_per_dataset = {s: base for s in TARGET_DATASETS}
        # Distribute remainder across first datasets
        for i in range(remainder):
            targets_per_dataset[TARGET_DATASETS[i]] += 1
        total_target = batch_size
    else:
        # Default: 500 rows (~167 each)
        base = 500 // len(TARGET_DATASETS)
        remainder = 500 % len(TARGET_DATASETS)
        targets_per_dataset = {s: base for s in TARGET_DATASETS}
        for i in range(remainder):
            targets_per_dataset[TARGET_DATASETS[i]] += 1
        total_target = 500

    print("=" * 65)
    print(f"BUILDING LABELING DATASET: TARGET = {total_target} ROWS")
    print(f"Quotas: {targets_per_dataset}")
    print("=" * 65)

    # 2. Check existing records for deduplication
    existing_ids = set()
    existing_records = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing_records.append(rec)
                    if "example_id" in rec:
                        existing_ids.add(rec["example_id"])
                    if "id" in rec:
                        existing_ids.add(rec["id"])
        print(f"Found {len(existing_records)} existing records in {OUTPUT_PATH}. Deduplicating...")

    # 3. Load queries
    batch = load_balanced_batch(RAW_DATA_PATH, targets_per_dataset, existing_ids)
    if not batch:
        print("No new queries to process. All matching queries are already labeled.")
        return

    print(f"Total queries to evaluate in this batch: {len(batch)}\n")

    # 4. Initialize Retriever and Pipelines
    print("Loading HybridRetriever (Dense MiniLM + Sparse BM25)...")
    retriever = HybridRetriever()
    llm_fn = get_llm()

    def retriever_fn(query: str, top_k: int = 3):
        return retriever.query(query, top_k=top_k)

    single_hop_pipe = SingleHopRAGPipeline(retriever_fn, llm_fn, max_tokens=SINGLE_HOP_MAX_TOKENS)
    multi_hop_pipe = MultiHopRAGPipeline(retriever_fn, llm_fn, max_tokens=MULTI_HOP_MAX_TOKENS)

def process_single_query(item: Dict, retriever, single_hop_pipe, multi_hop_pipe, llm_fn) -> Dict:
    """
    Executes single-hop and multi-hop evaluations uniformly on a single query
    with smart short-circuit judge calls.
    """
    qid = item["id"]
    question = item["question"]
    ground_truth = item["answer"]
    source = item["source_dataset"]

    # ── 1. Retrieval Signals ──
    top_docs, raw_scores = retriever.query_with_scores(question, top_k=5)
    top1_id = top_docs[0]["id"] if top_docs else ""
    top1_text = top_docs[0]["text"] if top_docs else ""

    dense_map = raw_scores.get("dense_score_map", {})
    bm25_map = raw_scores.get("bm25_score_map", {})
    rrf_map = raw_scores.get("rrf_score_map", {})

    dense_sim_top1 = round(dense_map.get(top1_id, 0.0), 4)
    bm25_score_top1 = round(bm25_map.get(top1_id, 0.0), 4)
    rrf_score_top1 = round(rrf_map.get(top1_id, 0.0), 6)

    sorted_rrf = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)
    rrf_margin_val = rrf_margin(sorted_rrf)
    rrf_entropy = compute_rrf_entropy(rrf_map)

    ent_overlap = entity_overlap_ratio(question, top1_text)
    all_ents_present = has_all_query_entities(question, top1_text)
    query_token_count = len(question.split())
    is_comp = is_comparison(question)
    interrog_type = get_interrogative_type(question)
    entities = extract_entities_simple(question)

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

    # ── 2. Single-Hop ──
    sh_result = single_hop_pipe.run(question, top_k=3)
    sh_pred = sh_result.get("prediction", "")
    sh_f1 = compute_f1(sh_pred, ground_truth)
    sh_em = compute_exact_match(sh_pred, ground_truth)
    sh_contains_gold = check_contains_gold(sh_pred, ground_truth)

    # Cost-optimized judge: checks refusal and verbatim match before calling LLM
    sh_judge = llm_judge(llm_fn, question, ground_truth, sh_pred, f1_score=sh_f1, contains_gold=sh_contains_gold)

    single_hop_entry = {
        "prediction": sh_pred,
        "retrieved_doc_ids": sh_result.get("retrieved_ids", []),
        "f1_score": sh_f1,
        "exact_match": sh_em,
        "judge_score": sh_judge["judge_score"],
        "judge_reasoning": sh_judge["judge_reasoning"],
        "judge_extracted_prediction": sh_judge.get("judge_extracted_prediction", ""),
        "contains_gold": sh_contains_gold,
        "total_tokens": sh_result.get("total_tokens", 0),
        "latency_sec": round(sh_result.get("latency_sec", 0.0), 4),
    }

    # ── 3. Multi-Hop (IR-CoT) ──
    mh_result = multi_hop_pipe.run(question, top_k=3)
    mh_pred = mh_result.get("prediction", "")
    mh_f1 = compute_f1(mh_pred, ground_truth)
    mh_em = compute_exact_match(mh_pred, ground_truth)
    mh_contains_gold = check_contains_gold(mh_pred, ground_truth)

    # Cost-optimized judge: checks refusal and verbatim match before calling LLM
    mh_judge = llm_judge(llm_fn, question, ground_truth, mh_pred, f1_score=mh_f1, contains_gold=mh_contains_gold)

    multi_hop_entry = {
        "prediction": mh_pred,
        "hop_queries": mh_result.get("hop_queries", []),
        "retrieved_doc_ids": mh_result.get("retrieved_ids", []),
        "f1_score": mh_f1,
        "exact_match": mh_em,
        "judge_score": mh_judge["judge_score"],
        "judge_reasoning": mh_judge["judge_reasoning"],
        "judge_extracted_prediction": mh_judge.get("judge_extracted_prediction", ""),
        "contains_gold": mh_contains_gold,
        "total_tokens": mh_result.get("total_tokens", 0),
        "latency_sec": round(mh_result.get("latency_sec", 0.0), 4),
    }

    # ── 4. Unified Record ──
    return {
        "example_id": f"{source}_{qid}",
        "dataset": source,
        "question": question,
        "question_type": "comparison" if is_comp else "factual",
        "ground_truth_answer": ground_truth,
        "retrieval_signals": retrieval_signals,
        "single_hop": single_hop_entry,
        "multi_hop": multi_hop_entry,
    }


def run_labeling(batch_size: int = None, per_dataset: int = None, workers: int = 6):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # 1. Calculate quotas per dataset
    if per_dataset is not None:
        targets_per_dataset = {s: per_dataset for s in TARGET_DATASETS}
        total_target = per_dataset * len(TARGET_DATASETS)
    elif batch_size is not None:
        base = batch_size // len(TARGET_DATASETS)
        remainder = batch_size % len(TARGET_DATASETS)
        targets_per_dataset = {s: base for s in TARGET_DATASETS}
        for i in range(remainder):
            targets_per_dataset[TARGET_DATASETS[i]] += 1
        total_target = batch_size
    else:
        base = 500 // len(TARGET_DATASETS)
        remainder = 500 % len(TARGET_DATASETS)
        targets_per_dataset = {s: base for s in TARGET_DATASETS}
        for i in range(remainder):
            targets_per_dataset[TARGET_DATASETS[i]] += 1
        total_target = 500

    print("=" * 65)
    print(f"BUILDING LABELING DATASET: TARGET = {total_target} ROWS (Parallel Workers: {workers})")
    print(f"Quotas: {targets_per_dataset}")
    print("=" * 65)

    # 2. Check existing records for deduplication
    existing_ids = set()
    existing_records = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing_records.append(rec)
                    if "example_id" in rec:
                        existing_ids.add(rec["example_id"])
                    if "id" in rec:
                        existing_ids.add(rec["id"])
        print(f"Found {len(existing_records)} existing records in {OUTPUT_PATH}. Skipping those...")

    # 3. Load queries
    batch = load_balanced_batch(RAW_DATA_PATH, targets_per_dataset, existing_ids)
    if not batch:
        print("No new queries to process. All matching queries are already labeled.")
        return

    print(f"Total new queries to evaluate: {len(batch)}\n")

    # 4. Initialize Retriever and Pipelines
    print("Loading HybridRetriever (Dense MiniLM + Sparse BM25)...")
    retriever = HybridRetriever()
    llm_fn = get_llm()

    def retriever_fn(query: str, top_k: int = 3):
        return retriever.query(query, top_k=top_k)

    single_hop_pipe = SingleHopRAGPipeline(retriever_fn, llm_fn, max_tokens=SINGLE_HOP_MAX_TOKENS)
    multi_hop_pipe = MultiHopRAGPipeline(retriever_fn, llm_fn, max_tokens=MULTI_HOP_MAX_TOKENS)

    batch_results = []
    file_lock = Lock()
    start_time = time.time()
    completed_count = 0

    def worker_task(item):
        record = process_single_query(item, retriever, single_hop_pipe, multi_hop_pipe, llm_fn)
        nonlocal completed_count
        with file_lock:
            completed_count += 1
            idx = len(existing_records) + completed_count
            with open(OUTPUT_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
            sh_stat = f"SH Judge={record['single_hop']['judge_score']}"
            mh_stat = f"MH Judge={record['multi_hop']['judge_score']}"
            print(f"[{completed_count}/{len(batch)}] (Overall #{idx}) [{record['dataset']}] {record['question'][:50]}... -> {sh_stat}, {mh_stat}")
        return record

    # 5. Process queries in parallel
    print(f"Starting parallel execution with {workers} worker threads...\n")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_task, item) for item in batch]
        for future in as_completed(futures):
            try:
                res = future.result()
                batch_results.append(res)
            except Exception as e:
                print(f"  [Error evaluating query]: {e}")

    # 6. Save Complete Combined JSON Preview
    all_records = existing_records + batch_results
    with open(JSON_OUTPUT_PATH, "w") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\n{'='*65}")
    print(f"Batch completed in {elapsed/60:.1f} minutes ({elapsed/len(batch_results):.2f}s/query)")
    print(f"  New queries added : {len(batch_results)}")
    print(f"  Total queries now : {len(all_records)}")
    print(f"  JSONL path        : {OUTPUT_PATH}")
    print(f"  JSON preview path : {JSON_OUTPUT_PATH}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build labeling dataset in batches.")
    parser.add_argument("--batch_size", type=int, default=500, help="Total rows to add in this batch (e.g. 500, 1500, 1000)")
    parser.add_argument("--per_dataset", type=int, default=None, help="Exact quota per dataset (e.g. 167 or 500)")
    parser.add_argument("--workers", type=int, default=6, help="Number of concurrent worker threads (default: 6)")
    args = parser.parse_args()

    run_labeling(batch_size=args.batch_size, per_dataset=args.per_dataset, workers=args.workers)
