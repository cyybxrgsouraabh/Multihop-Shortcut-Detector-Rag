import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
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
            print("BM25 index not found. Building automatically...")
            self.build_bm25_index()

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

    def query_with_scores(self, query_text: str, top_k: int = 3) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Returns (top_k results, raw_scores_dict) where raw_scores_dict contains:
          - dense_scores: {doc_id: cosine_distance from chroma}
          - bm25_scores: {doc_id: bm25_score}
          - rrf_scores: {doc_id: rrf_score}
        """
        # 1. Dense results with distances (thread-safe)
        dense_results_raw = self.vector_indexer.query_raw(
            query_texts=[query_text],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"]
        )

        dense_docs = []
        dense_score_map = {}  # doc_id -> similarity score (1 - distance)
        if dense_results_raw and dense_results_raw["documents"]:
            docs = dense_results_raw["documents"][0]
            metas = dense_results_raw["metadatas"][0]
            ids = dense_results_raw["ids"][0]
            distances = dense_results_raw["distances"][0]

            for doc_id, text, meta, dist in zip(ids, docs, metas, distances):
                similarity = round(1 - dist, 6)
                dense_score_map[doc_id] = similarity
                dense_docs.append({
                    "id": doc_id,
                    "title": meta.get("title", ""),
                    "text": text,
                    "source": meta.get("source", ""),
                })

        # 2. BM25 results
        tokenized_query = query_text.lower().split(" ")
        bm25_scores_array = self.bm25.get_scores(tokenized_query)

        top_bm25_indices = sorted(
            range(len(bm25_scores_array)),
            key=lambda i: bm25_scores_array[i],
            reverse=True
        )[:self.top_k]

        bm25_score_map = {}
        bm25_results = []
        for idx in top_bm25_indices:
            doc = self.bm25_docs[idx]
            bm25_score_map[doc["id"]] = round(float(bm25_scores_array[idx]), 6)
            bm25_results.append(doc)

        # 3. RRF fusion
        rrf_scores = {}

        for rank, doc in enumerate(dense_docs):
            doc_id = doc["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"score": 0.0, "doc": doc}
            rrf_scores[doc_id]["score"] += 1.0 / (self.rrf_k + rank + 1)

        for rank, doc in enumerate(bm25_results):
            doc_id = doc["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"score": 0.0, "doc": doc}
            rrf_scores[doc_id]["score"] += 1.0 / (self.rrf_k + rank + 1)

        rrf_score_map = {doc_id: round(v["score"], 6) for doc_id, v in rrf_scores.items()}
        sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        final_results = [item["doc"] for item in sorted_docs[:top_k]]

        raw_scores = {
            "dense_score_map": dense_score_map,
            "bm25_score_map": bm25_score_map,
            "rrf_score_map": rrf_score_map,
        }
        return final_results, raw_scores

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        results, _ = self.query_with_scores(query_text, top_k=top_k)
        return results
