import time
from typing import Any, Callable, Dict, List

from src.pipelines.base import BaseRAGPipeline


class MultiHopRAGPipeline(BaseRAGPipeline):
    """
    Iterative Multi-Hop RAG:
    1. Hop 1: Retrieve top passages.
    2. Step 2: Use LLM to formulate a follow-up sub-query.
    3. Hop 2: Retrieve passages for the sub-query.
    4. Step 4: Final multi-document synthesis.
    """

    def __init__(
        self,
        retriever_fn: Callable[[str, int], List[Dict[str, Any]]],
        llm_generate_fn: Callable[[str, str], Dict[str, Any]],
    ):
        self.retriever = retriever_fn
        self.llm = llm_generate_fn

    def run(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        start_time = time.time()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        all_passages = []
        hop_queries = [query]

        # --- Hop 1 ---
        hop1_passages = self.retriever(query, top_k)
        all_passages.extend(hop1_passages)

        hop1_context = "\n".join(
            [f"- {p.get('title', '')}: {p.get('text', '')}" for p in hop1_passages]
        )

        # --- Step 2: Generate Follow-up Query ---
        sub_query_prompt = (
            f"Original Question: {query}\n\n"
            f"Passages retrieved so far:\n{hop1_context}\n\n"
            f"Based on what is known and what is missing, generate one specific follow-up search query "
            f"to find the remaining missing information. Output only the search query."
        )

        sub_resp = self.llm(
            sub_query_prompt,
            system_prompt="You are an expert search planner that breaks down complex questions into targeted search steps.",
        )
        total_prompt_tokens += sub_resp.get("prompt_tokens", 0)
        total_completion_tokens += sub_resp.get("completion_tokens", 0)

        follow_up_query = sub_resp.get("text", "").strip()
        if not follow_up_query:
            follow_up_query = query
        hop_queries.append(follow_up_query)

        # --- Hop 2 ---
        hop2_passages = self.retriever(follow_up_query, top_k)
        # Deduplicate
        existing_ids = {p.get("id") for p in all_passages}
        for p in hop2_passages:
            if p.get("id") not in existing_ids:
                all_passages.append(p)
                existing_ids.add(p.get("id"))

        # --- Final Synthesis ---
        all_context_str = "\n\n".join(
            [f"[{i+1}] {p.get('title', '')}: {p.get('text', '')}" for i, p in enumerate(all_passages)]
        )

        final_prompt = (
            f"Context Information from multi-step search:\n{all_context_str}\n\n"
            f"Question: {query}\n\n"
            f"Synthesize the evidence from the context and provide a concise, direct answer."
        )

        final_resp = self.llm(
            final_prompt,
            system_prompt="You are a reasoning assistant that synthesizes multiple sources to answer complex questions.",
        )
        total_prompt_tokens += final_resp.get("prompt_tokens", 0)
        total_completion_tokens += final_resp.get("completion_tokens", 0)

        latency = time.time() - start_time
        retrieved_ids = [p.get("id") for p in all_passages]

        return {
            "prediction": final_resp.get("text", "").strip(),
            "hop_queries": hop_queries,
            "retrieved_ids": retrieved_ids,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "latency_sec": latency
        }
