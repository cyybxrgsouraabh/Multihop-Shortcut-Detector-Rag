import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_PATH
from src.indexer import VectorIndexer


def main():
    print(f"Building vector index from {RAW_DATA_PATH}...")
    indexer = VectorIndexer()
    indexer.index_dataset()
    print("Vector indexing completed successfully!")


if __name__ == "__main__":
    main()
