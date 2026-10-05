import json
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from groq import Groq

from src.config import RAW_DATA_PATH
from src.hybrid_retriever import HybridRetriever
from src.pipelines.multi_hop import MultiHopRAGPipeline
from src.pipelines.single_hop import SingleHopRAGPipeline

load_dotenv()


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
        base_wait_time = 1  # seconds

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                )
                
                # Calculate tokens
                usage = response.usage
                return {
                    "text": response.choices[0].message.content or "",
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                }
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Groq API Error after {max_retries} attempts: {e}")
                    return {"text": "API Failed", "prompt_tokens": 0, "completion_tokens": 0}
                
                sleep_time = base_wait_time * (2 ** attempt)
                print(f"API Error: {e}. Retrying in {sleep_time} seconds (Attempt {attempt+1}/{max_retries})...")
                time.sleep(sleep_time)

    return llm_fn


def main():
    indexer = HybridRetriever()

    def retriever_fn(query: str, top_k: int = 3):
        return indexer.query(query, top_k=top_k)

    llm_fn = get_groq_llm()

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
