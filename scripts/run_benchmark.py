import json
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import re
from openai import OpenAI

from src.config import (
    RAW_DATA_PATH,
    DEEPSEEK_MODEL_NAME,
    DEEPSEEK_BASE_URL,
    DEFAULT_MAX_TOKENS,
)
from src.hybrid_retriever import HybridRetriever
from src.pipelines.multi_hop import MultiHopRAGPipeline
from src.pipelines.single_hop import SingleHopRAGPipeline

load_dotenv()


def get_llm(default_max_tokens: int = DEFAULT_MAX_TOKENS):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in .env")

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    model = os.getenv("DEEPSEEK_MODEL_NAME", DEEPSEEK_MODEL_NAME)
    print(f"Benchmark using DeepSeek API (model: {model})...")

    def llm_fn(prompt: str, system_prompt: str = "", max_tokens: int = None) -> dict:
        tokens_limit = max_tokens if max_tokens is not None else default_max_tokens
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_retries = 5
        base_wait_time = 2.0

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
                sleep_time = base_wait_time * (2 ** attempt)
                if attempt == max_retries - 1:
                    print(f"DeepSeek API Error after {max_retries} attempts: {e}")
                    return {"text": "API Failed", "prompt_tokens": 0, "completion_tokens": 0}
                print(f"DeepSeek API Error: {e}. Retrying in {sleep_time:.1f}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(sleep_time)

    return llm_fn


def main():
    indexer = HybridRetriever()

    def retriever_fn(query: str, top_k: int = 3):
        return indexer.query(query, top_k=top_k)

    llm_fn = get_llm()

    single_hop_pipe = SingleHopRAGPipeline(retriever_fn, llm_fn)
    multi_hop_pipe = MultiHopRAGPipeline(retriever_fn, llm_fn)

    dataset = []
    with open(RAW_DATA_PATH, "r") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    # Test on next 10 queries (10 to 20)
    dataset = dataset[10:20]

    from src.config import BENCHMARK_RESULTS_PATH
    results = []
    if BENCHMARK_RESULTS_PATH.exists():
        with open(BENCHMARK_RESULTS_PATH, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []

    for i, item in enumerate(dataset):
        query = item["question"]
        print(f"\n--- Query {i+1}: {query} ---")
        
        print("Running Single-Hop...")
        single_hop_result = single_hop_pipe.run(query)
        print("# Output of single_hop_pipeline(query)")
        print(json.dumps(single_hop_result, indent=4))
        
        print("\nRunning Multi-Hop...")
        multi_hop_result = multi_hop_pipe.run(query)
        print("# Output of multi_hop_pipeline(query)")
        print(json.dumps(multi_hop_result, indent=4))
        
        results.append({
            "id": item.get("id"),
            "question": query,
            "answer": item.get("answer"),
            "single_hop_result": single_hop_result,
            "multi_hop_result": multi_hop_result,
            "source_dataset": item.get("source_dataset"),
        })

    # Save to BENCHMARK_RESULTS_PATH
    from src.config import BENCHMARK_RESULTS_PATH
    import os
    os.makedirs(BENCHMARK_RESULTS_PATH.parent, exist_ok=True)
    with open(BENCHMARK_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {BENCHMARK_RESULTS_PATH}")


if __name__ == "__main__":
    main()
