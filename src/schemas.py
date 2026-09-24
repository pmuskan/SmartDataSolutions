from typing import Any, Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str = Field(..., description="Deterministic unique identifier for the chunk")
    type: Literal["text", "table", "figure"] = Field(..., description="Chunk modality type")
    page: int = Field(..., description="1-indexed page number in PDF")
    section_path: str = Field(..., description="Section hierarchy e.g. Part I > Item 2 > Gross Margin")
    title: str = Field(default="", description="Title of table, figure, or section heading")
    units: str = Field(default="", description="Units e.g. in millions, in thousands, %")
    text: str = Field(..., description="Full text/payload provided to LLM context")
    embed_text: str = Field(..., description="Target text used for generating embeddings")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional parsing/structural metadata")


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    page: int
    section_or_table_title: str
    quote: str | None = None


class ToolCall(BaseModel):
    tool_name: str
    expression: str
    inputs: dict[str, Any]
    result: Any


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    used_chunk_ids: list[str] = Field(default_factory=list)
    abstained: bool = False
    tool_calls: list[ToolCall] = Field(default_factory=list)
    latency_ms: float = 0.0
    tokens_used: int = 0


class EvalItem(BaseModel):
    id: str
    question: str
    type: Literal["text", "table", "cross_modal", "numeric_derived", "figure", "unanswerable"]
    gold_answer: str
    gold_value: float | None = None
    gold_unit: str | None = None
    gold_pages: list[int]
    gold_chunk_hint: str
    difficulty: str = "medium"


class EvalResult(BaseModel):
    item_id: str
    question: str
    type: str
    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False
    recall: float = 0.0
    mrr: float = 0.0
    page_hit: bool = False
    modality_correct: bool = False
    numeric_match: bool | None = None
    text_judge_score: float | None = None
    judge_verdict: str | None = None
    rouge_l_score: float | None = None
    faithfulness_score: float | None = None
    citation_precision: float | None = None
    hallucinated_number_count: int = 0
    abstention_correct: bool | None = None
    latency_ms: float = 0.0
    tokens_used: int = 0
