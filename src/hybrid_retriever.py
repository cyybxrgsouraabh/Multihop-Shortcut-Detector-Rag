import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from src.config import CHROMA_DB_PATH, RAW_DATA_PATH
from src.indexer import VectorIndexer, extract_passages_from_item


class HybridRetriever:
    def __init__(self, top_k: int = 60, rrf_k: int = 60):
        self.vector_indexer = VectorIndexer()
        self.top_k = top_k
        self.rrf_k = rrf_k
        
        self.bm25_path = CHROMA_DB_PATH.parent / "bm25_index.pkl"
        self.docs_path = CHROMA_DB_PATH.parent / "bm25_docs.pkl"
        
        self.bm25 = None
        self.bm25_docs = None
        
        if self.bm25_path.exists() and self.docs_path.exists():
            print("Loading BM25 index...")
            with open(self.bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
            with open(self.docs_path, "rb") as f:
                self.bm25_docs = pickle.load(f)
        else:
            print("BM25 index not found. Please build it first by calling build_bm25_index().")

    def build_bm25_index(self, dataset_path: Path = RAW_DATA_PATH):
        print(f"Building BM25 index from {dataset_path}...")
        data = []
        with open(dataset_path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        all_docs = []
        for item in data:
            all_docs.extend(extract_passages_from_item(item))

        seen_ids = set()
        self.bm25_docs = []
        for d in all_docs:
            if d["id"] not in seen_ids:
                seen_ids.add(d["id"])
                self.bm25_docs.append(d)

        # Tokenize using simple whitespace/lower
        print("Tokenizing documents for BM25...")
        tokenized_corpus = [
            (f"{doc['title']} {doc['text']}").lower().split(" ") 
            for doc in self.bm25_docs
        ]
        
        print("Building BM25Okapi...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print("Saving BM25 to disk...")
        with open(self.bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(self.docs_path, "wb") as f:
            pickle.dump(self.bm25_docs, f)
            
        print("BM25 index built successfully!")

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        # 1. Get Dense (Vector) Results
        # Retrieve more than top_k for RRF to work well
        dense_results = self.vector_indexer.query(query_text, top_k=self.top_k)
        
        # 2. Get Sparse (BM25) Results
        tokenized_query = query_text.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Sort and get top-k for BM25
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self.top_k]
        bm25_results = [self.bm25_docs[i] for i in top_bm25_indices]
        
        # 3. Apply Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        
        # Rank dense results
        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"score": 0.0, "doc": doc}
            rrf_scores[doc_id]["score"] += 1.0 / (self.rrf_k + rank + 1)
            
        # Rank sparse results
        for rank, doc in enumerate(bm25_results):
            doc_id = doc["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"score": 0.0, "doc": doc}
            rrf_scores[doc_id]["score"] += 1.0 / (self.rrf_k + rank + 1)
            
        # 4. Sort by RRF score and return top_k
        sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        final_results = [item["doc"] for item in sorted_docs[:top_k]]
        
        return final_results
