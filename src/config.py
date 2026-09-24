import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env if present
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ChunkingConfig(BaseModel):
    text_chunk_size: int = 500
    text_chunk_overlap: int = 50
    table_chunk_strategy: str = "single_unit"


class ModelsConfig(BaseModel):
    embedding: str = "BAAI/bge-base-en-v1.5"
    reranker: str = "BAAI/bge-reranker-base"
    llm_provider: str = "anthropic"
    llm_model: str = "claude-3-5-sonnet-20241022"
    vision_model: str = "claude-3-5-sonnet-20241022"


class RouterConfig(BaseModel):
    enabled: bool = True
    table_boost: float = 1.5
    text_boost: float = 1.2
    figure_boost: float = 2.0


class RetrievalConfig(BaseModel):
    top_k_dense: int = 20
    top_k_sparse: int = 20
    rrf_k: int = 60
    default_top_k: int = 5
    router: RouterConfig = Field(default_factory=RouterConfig)


class GenerationConfig(BaseModel):
    temperature: float = 0.0
    strict_grounding: bool = True
    abstention_message: str = (
        "The requested information is not found in the provided document."
    )


class EvalConfig(BaseModel):
    golden_set_path: str = "eval/golden_set.jsonl"
    results_dir: str = "eval/results"
    cache_dir: str = "eval/cache"


class AppConfig(BaseModel):
    pdf_path: str = "data/2022_Q3_AAPL.pdf"
    processed_dir: str = "data/processed"
    chroma_db_dir: str = "data/chroma_db"
    bm25_index_path: str = "data/bm25_index.pkl"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    def resolve_path(self, rel_path: str) -> Path:
        return (PROJECT_ROOT / rel_path).resolve()


def load_config(config_path: str = "config.yaml") -> AppConfig:
    full_path = (PROJECT_ROOT / config_path).resolve()
    if not full_path.exists():
        return AppConfig()

    with open(full_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Override provider / model from environment if specified
    if os.getenv("LLM_PROVIDER"):
        data.setdefault("models", {})["llm_provider"] = os.getenv("LLM_PROVIDER")
    if os.getenv("LLM_MODEL"):
        data.setdefault("models", {})["llm_model"] = os.getenv("LLM_MODEL")

    return AppConfig(**data)
