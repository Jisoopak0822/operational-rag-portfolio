"""Configuration for a local-first, portfolio-scale RAG demonstration."""
from dataclasses import dataclass
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / '.env')
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = ROOT / 'data' / 'docs'
    embedding_backend: str = os.getenv('RAG_EMBEDDING_BACKEND', 'tfidf')
    embedding_model: str = os.getenv('RAG_EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    llm_base_url: str = os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1')
    llm_api_key: str = os.getenv('LLM_API_KEY', '')
    llm_model: str = os.getenv('LLM_MODEL', 'gpt-4.1-mini')
    llm_timeout_s: float = float(os.getenv('LLM_TIMEOUT_S', '15'))
    top_k: int = int(os.getenv('TOP_K', '5'))


settings = Settings()
