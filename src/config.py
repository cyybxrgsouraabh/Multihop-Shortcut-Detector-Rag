import os
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "combined_dataset.jsonl"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
BENCHMARK_RESULTS_PATH = PROCESSED_DATA_DIR / "benchmark_dataset.json"
CHROMA_DB_PATH = DATA_DIR / "processed" / "chroma_db"

# Embedding Model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# DeepSeek Model Configuration
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-flash")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Generation Token Budgets (deepseek-flash uses reasoning tokens, so allocate adequate headroom)
DEFAULT_MAX_TOKENS = 512
SINGLE_HOP_MAX_TOKENS = 512
MULTI_HOP_MAX_TOKENS = 768
JUDGE_MAX_TOKENS = 512


