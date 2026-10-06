import json
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.build_labeling_dataset import (
    get_llm,
    llm_judge,
    HybridRetriever,
    SingleHopRAGPipeline,
    MultiHopRAGPipeline,
    SINGLE_HOP_MAX_TOKENS,
    MULTI_HOP_MAX_TOKENS,
    RAW_DATA_PATH
)

def run_test():
    print("Initializing HybridRetriever...")
    retriever = HybridRetriever()
    def retriever_fn(q, top_k=3):
        return retriever.query(q, top_k=top_k)

    call_logs = []
    base_llm = get_llm()

    def tracked_llm(prompt, system_prompt="", max_tokens=None):
        t0 = time.time()
        res = base_llm(prompt, system_prompt=system_prompt, max_tokens=max_tokens)
        latency = time.time() - t0
        entry = {
            "prompt_tokens": res.get("prompt_tokens", 0),
            "completion_tokens": res.get("completion_tokens", 0),
            "total_tokens": res.get("prompt_tokens", 0) + res.get("completion_tokens", 0),
            "latency_sec": round(latency, 3),
            "text_snippet": res.get("text", "")[:60].replace("\n", " ")
        }
        call_logs.append(entry)
        return res

    sh_pipe = SingleHopRAGPipeline(retriever_fn, tracked_llm, max_tokens=SINGLE_HOP_MAX_TOKENS)
    mh_pipe = MultiHopRAGPipeline(retriever_fn, tracked_llm, max_tokens=MULTI_HOP_MAX_TOKENS)

    # Pick 1 representative multi-hop query
    with open(RAW_DATA_PATH) as f:
        sample_item = json.loads(f.readline())

    question = sample_item["question"]
    gold = sample_item["answer"]
    dataset = sample_item["source_dataset"]

    print("\n=== RUNNING TEST QUERY ===")
    print(f"Dataset : {dataset}")
    print(f"Question: {question}")
    print(f"Gold    : {gold}\n")

    # 1. Single-Hop
    sh_idx_start = len(call_logs)
    sh_res = sh_pipe.run(question, top_k=3)
    sh_judge = llm_judge(tracked_llm, question, gold, sh_res["prediction"])
    sh_calls = call_logs[sh_idx_start:]

    # 2. Multi-Hop
    mh_idx_start = len(call_logs)
    mh_res = mh_pipe.run(question, top_k=3)
    mh_judge = llm_judge(tracked_llm, question, gold, mh_res["prediction"])
    mh_calls = call_logs[mh_idx_start:]

    # Output metrics
    total_prompt = sum(c["prompt_tokens"] for c in call_logs)
    total_comp = sum(c["completion_tokens"] for c in call_logs)
    total_tokens = total_prompt + total_comp

    # DeepSeek-V3 pricing: $0.14 per 1M input tokens, $0.28 per 1M output tokens
    cost_input = (total_prompt / 1_000_000) * 0.14
    cost_output = (total_comp / 1_000_000) * 0.28
    total_cost = cost_input + cost_output

    print("\n" + "="*60)
    print("DETAILED RUN METRICS (DEEPSEEK-V3)")
    print("="*60)
    print(f"Single-Hop Prediction: {sh_res['prediction']}")
    print(f"Single-Hop Judge     : {sh_judge}")
    print(f"Single-Hop Calls     : {len(sh_calls)} calls | Prompt: {sum(c['prompt_tokens'] for c in sh_calls)} | Comp: {sum(c['completion_tokens'] for c in sh_calls)}")
    print("-"*60)
    print(f"Multi-Hop Prediction : {mh_res['prediction']}")
    print(f"Multi-Hop Judge      : {mh_judge}")
    print(f"Multi-Hop Calls      : {len(mh_calls)} calls | Prompt: {sum(c['prompt_tokens'] for c in mh_calls)} | Comp: {sum(c['completion_tokens'] for c in mh_calls)}")
    print("-"*60)
    print(f"Total API Calls      : {len(call_logs)}")
    print(f"Total Prompt Tokens  : {total_prompt:,}")
    print(f"Total Output Tokens  : {total_comp:,}")
    print(f"Total Combined Tokens: {total_tokens:,}")
    print(f"Cost for this 1 query: ${total_cost:.6f} USD")
    print("-"*60)
    print(f"Extrapolated Cost for 1,000 queries: ${total_cost * 1000:.2f} USD")
    print(f"Extrapolated Cost for 2,000 queries: ${total_cost * 2000:.2f} USD")
    print(f"Extrapolated Cost for 3,000 queries: ${total_cost * 3000:.2f} USD")
    max_queries = int(2.00 / total_cost) if total_cost > 0 else 0
    print(f"Queries possible with your $2.00 balance: ~{max_queries:,} queries!")
    print("="*60)

if __name__ == "__main__":
    run_test()
