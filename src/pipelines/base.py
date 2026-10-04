from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseRAGPipeline(ABC):
    """Abstract base class for RAG pipelines."""

    @abstractmethod
    def run(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Run the pipeline on a query and return structured dictionary results."""
        pass
