import time
from typing import Any, Callable, Dict, List

from src.config import SINGLE_HOP_MAX_TOKENS
from src.pipelines.base import BaseRAGPipeline


class SingleHopRAGPipeline(BaseRAGPipeline):
    """
    Standard Single-Hop RAG:
    1. Retrieve top-k passages once.
    2. Pass query + retrieved context directly to the LLM.
    """

    def __init__(
        self,
        retriever_fn: Callable[[str, int], List[Dict[str, Any]]],
        llm_generate_fn: Callable[..., Dict[str, Any]],
        max_tokens: int = SINGLE_HOP_MAX_TOKENS,
    ):
        self.retriever = retriever_fn
        self.llm = llm_generate_fn
        self.max_tokens = max_tokens

    def run(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        start_time = time.time()

        # Step 1: Single retrieval
        passages = self.retriever(query, top_k)
        context_str = "\n\n".join(
            [f"[{i+1}] {p.get('title', '')}: {p.get('text', '')}" for i, p in enumerate(passages)]
        )

        # Step 2: Generation prompt
        prompt = (
            f"Context Information:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            f"Provide a concise, direct answer based strictly on the context."
        )

        try:
            gen_resp = self.llm(
                prompt,
                system_prompt="You are a helpful and concise assistant that answers questions based on retrieved documents.",
                max_tokens=self.max_tokens,
            )
        except TypeError:
            gen_resp = self.llm(
                prompt,
                system_prompt="You are a helpful and concise assistant that answers questions based on retrieved documents.",
            )

        latency = time.time() - start_time
        prompt_tokens = gen_resp.get("prompt_tokens", 0)
        comp_tokens = gen_resp.get("completion_tokens", 0)
        
        retrieved_ids = [p.get("id") for p in passages]
        top_1_text = f"{passages[0].get('title', '')}: {passages[0].get('text', '')}" if passages else ""

        return {
            "prediction": gen_resp.get("text", "").strip(),
            "retrieved_ids": retrieved_ids,
            "top_1_doc_text": top_1_text,
            "total_tokens": prompt_tokens + comp_tokens,
            "latency_sec": latency
        }
