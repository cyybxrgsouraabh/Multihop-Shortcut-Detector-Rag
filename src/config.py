import os
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "combined_dataset.json"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
BENCHMARK_RESULTS_PATH = PROCESSED_DATA_DIR / "benchmark_dataset.json"
CHROMA_DB_PATH = DATA_DIR / "processed" / "chroma_db"

# Embedding Model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq Model
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "qwen/qwen3.8-27b")
