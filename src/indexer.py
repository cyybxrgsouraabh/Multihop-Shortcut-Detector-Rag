import json
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions

from src.config import CHROMA_DB_PATH, DEFAULT_EMBEDDING_MODEL, RAW_DATA_PATH


def extract_passages_from_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    extracted = []
    item_id = item.get("id", "")
    source = item.get("source_dataset", "")

    for idx, passage in enumerate(item.get("context_passages", [])):
        title = passage.get("title", "")
        
        if "sentences" in passage:
            text = " ".join(passage["sentences"]).strip()
        elif "paragraph_text" in passage:
            text = passage["paragraph_text"].strip()
        else:
            text = ""

        if text:
            doc_id = f"{item_id}_p{idx}"
            extracted.append({
                "id": doc_id,
                "title": title,
                "text": text,
                "source_dataset": source,
                "parent_query_id": item_id,
            })
    return extracted


class VectorIndexer:
    def __init__(
        self,
        persist_directory: Path = CHROMA_DB_PATH,
        collection_name: str = "multihop_passages",
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.persist_directory = str(persist_directory)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.emb_fn,
        )

    def index_dataset(self, dataset_path: Path = RAW_DATA_PATH, batch_size: int = 256):
        with open(dataset_path, "r") as f:
            data = json.load(f)

        all_docs = []
        for item in data:
            all_docs.extend(extract_passages_from_item(item))

        print(f"Total unique passages to index: {len(all_docs)}")

        seen_ids = set()
        unique_docs = []
        for d in all_docs:
            if d["id"] not in seen_ids:
                seen_ids.add(d["id"])
                unique_docs.append(d)

        for i in range(0, len(unique_docs), batch_size):
            batch = unique_docs[i : i + batch_size]
            self.collection.upsert(
                ids=[b["id"] for b in batch],
                documents=[f"{b['title']}: {b['text']}" for b in batch],
                metadatas=[
                    {
                        "title": b["title"],
                        "source": b["source_dataset"],
                        "parent_id": b["parent_query_id"],
                    }
                    for b in batch
                ],
            )
            print(f"Indexed {min(i + batch_size, len(unique_docs))}/{len(unique_docs)} passages...")

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        results = self.collection.query(query_texts=[query_text], n_results=top_k)

        passages = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
            ids = results["ids"][0] if results["ids"] else [""] * len(docs)

            for doc_id, text, meta in zip(ids, docs, metas):
                passages.append({
                    "id": doc_id,
                    "title": meta.get("title", ""),
                    "text": text,
                    "source": meta.get("source", ""),
                })
        return passages
