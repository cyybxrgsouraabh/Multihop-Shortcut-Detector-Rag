import re
import time
from typing import Any, Callable, Dict, List

from src.pipelines.base import BaseRAGPipeline


class MultiHopRAGPipeline(BaseRAGPipeline):
    """
    Iterative Multi-Hop RAG using IR-CoT (Interleaving Retrieval with Chain-of-Thought):
    1. Retrieve initial passages.
    2. Prompt LLM to reason based on context.
    3. If LLM outputs 'SEARCH: <query>', retrieve more and append to context.
    4. If LLM outputs 'ANSWER: <answer>', stop and return.
    """

    def __init__(
        self,
        retriever_fn: Callable[[str, int], List[Dict[str, Any]]],
        llm_generate_fn: Callable[[str, str], Dict[str, Any]],
    ):
        self.retriever = retriever_fn
        self.llm = llm_generate_fn

    def run(self, query: str, top_k: int = 3, max_steps: int = 3) -> Dict[str, Any]:
        start_time = time.time()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        all_passages = []
        hop_queries = [query]
        existing_ids = set()

        # --- Initial retrieval ---
        hop1_passages = self.retriever(query, top_k)
        for p in hop1_passages:
            if p.get("id") not in existing_ids:
                all_passages.append(p)
                existing_ids.add(p.get("id"))

        system_prompt = (
            "You are a reasoning assistant designed to answer complex multi-hop questions. "
            "You will be given a question and a set of retrieved passages. "
            "Think step-by-step based on the passages. "
            "If the retrieved passages contain enough information to answer the question, output your reasoning followed by the final answer starting with 'ANSWER: <your concise answer>'. "
            "If you need more information to bridge a gap, output your reasoning followed by a specific search query starting with 'SEARCH: <your query>'. "
            "Never output both SEARCH and ANSWER."
        )

        step = 1
        final_prediction = ""
        
        # --- IR-CoT Loop ---
        while step <= max_steps:
            context_str = "\n".join(
                [f"[{i+1}] {p.get('title', '')}: {p.get('text', '')}" for i, p in enumerate(all_passages)]
            )

            prompt = (
                f"Question: {query}\n\n"
                f"Retrieved Context:\n{context_str}\n\n"
                f"Are you ready to answer? Use 'SEARCH: <query>' if missing info, or 'ANSWER: <answer>' if you have enough info."
            )

            resp = self.llm(prompt, system_prompt=system_prompt)
            total_prompt_tokens += resp.get("prompt_tokens", 0)
            total_completion_tokens += resp.get("completion_tokens", 0)

            response_text = resp.get("text", "").strip()

            # Check if LLM decided to answer
            answer_match = re.search(r"ANSWER:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)
            if answer_match:
                final_prediction = answer_match.group(1).strip()
                break

            # Check if LLM decided to search again
            search_match = re.search(r"SEARCH:\s*(.*)", response_text, re.IGNORECASE)
            if search_match and step < max_steps:
                next_query = search_match.group(1).strip()
                hop_queries.append(next_query)
                
                next_passages = self.retriever(next_query, top_k)
                for p in next_passages:
                    if p.get("id") not in existing_ids:
                        all_passages.append(p)
                        existing_ids.add(p.get("id"))
                step += 1
                continue

            # Fallback if model doesn't follow instructions exactly
            final_prediction = response_text
            break

        # --- Final Fallback Synthesis ---
        # If we reached max_steps and haven't successfully pulled an 'ANSWER: ' tag
        if not final_prediction or "SEARCH:" in final_prediction:
            context_str = "\n".join(
                [f"[{i+1}] {p.get('title', '')}: {p.get('text', '')}" for i, p in enumerate(all_passages)]
            )
            final_prompt = (
                f"Question: {query}\n\n"
                f"Context:\n{context_str}\n\n"
                f"Based on the context, provide a concise, direct answer."
            )
            final_resp = self.llm(final_prompt, system_prompt="Answer the question directly and concisely based on context.")
            total_prompt_tokens += final_resp.get("prompt_tokens", 0)
            total_completion_tokens += final_resp.get("completion_tokens", 0)
            final_prediction = final_resp.get("text", "").strip()

        latency = time.time() - start_time
        retrieved_ids = [p.get("id") for p in all_passages]

        return {
            "prediction": final_prediction,
            "hop_queries": hop_queries,
            "retrieved_ids": retrieved_ids,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "latency_sec": latency
        }
