from typing import Any

from sentence_transformers import CrossEncoder

from src.config import load_config
from src.index import VectorAndSparseIndex
from src.schemas import Chunk, RetrievedChunk


def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    rrf_k: int = 60
) -> dict[str, dict[str, Any]]:
    """Compute Reciprocal Rank Fusion (RRF) scores across dense and sparse runs."""
    fused_scores: dict[str, dict[str, Any]] = {}

    for rank, (cid, score) in enumerate(dense_results):
        if cid not in fused_scores:
            fused_scores[cid] = {"dense_rank": rank + 1, "sparse_rank": None, "rrf_score": 0.0, "dense_score": score, "sparse_score": 0.0}
        else:
            fused_scores[cid]["dense_rank"] = rank + 1
            fused_scores[cid]["dense_score"] = score
        fused_scores[cid]["rrf_score"] += 1.0 / (rrf_k + rank + 1)

    for rank, (cid, score) in enumerate(sparse_results):
        if cid not in fused_scores:
            fused_scores[cid] = {"dense_rank": None, "sparse_rank": rank + 1, "rrf_score": 0.0, "dense_score": 0.0, "sparse_score": score}
        else:
            fused_scores[cid]["sparse_rank"] = rank + 1
            fused_scores[cid]["sparse_score"] = score
        fused_scores[cid]["rrf_score"] += 1.0 / (rrf_k + rank + 1)

    return fused_scores


class QueryRouter:
    def __init__(
        self,
        table_boost: float = 1.5,
        text_boost: float = 1.2,
        figure_boost: float = 2.0
    ):
        self.table_boost = table_boost
        self.text_boost = text_boost
        self.figure_boost = figure_boost

        self.table_keywords = [
            "how much", "revenue", "net sales", "operating income", "gross margin",
            "net income", "eps", "diluted", "share", "shares", "dividend", "cash",
            "debt", "obligation", "tax rate", "expense", "cost", "total", "table",
            "million", "billion", "dollars", "%", "percent"
        ]
        self.text_keywords = [
            "why", "explain", "drove", "factor", "reason", "cause", "describe",
            "policy", "discussion", "impact", "decline", "increase", "fell", "rose"
        ]
        self.figure_keywords = [
            "image", "chart", "figure", "logo", "graph", "picture", "cover page",
            "visual", "diagram"
        ]

    def apply_boost(self, query: str, chunk: Chunk, base_score: float) -> float:
        q_lower = query.lower()
        
        is_figure_query = any(k in q_lower for k in self.figure_keywords)
        is_table_query = any(k in q_lower for k in self.table_keywords)
        is_text_query = any(k in q_lower for k in self.text_keywords)

        boost = 1.0
        if is_figure_query and chunk.type == "figure":
            boost = self.figure_boost
        elif is_table_query and chunk.type == "table":
            boost = self.table_boost
        elif is_text_query and chunk.type == "text":
            boost = self.text_boost

        return base_score * boost


class HybridRetriever:
    def __init__(self, index: VectorAndSparseIndex, config_path: str = "config.yaml"):
        self.index = index
        self.cfg = load_config(config_path)
        self.router = QueryRouter(
            table_boost=self.cfg.retrieval.router.table_boost,
            text_boost=self.cfg.retrieval.router.text_boost,
            figure_boost=self.cfg.retrieval.router.figure_boost
        )
        self.reranker_model_name = self.cfg.models.reranker
        self.reranker = None

    def _init_reranker(self):
        if self.reranker is None:
            print(f"Loading reranker model: {self.reranker_model_name}...")
            try:
                self.reranker = CrossEncoder(self.reranker_model_name)
            except Exception as e:
                print(f"Warning: Failed to load cross-encoder ({e}), falling back to RRF scoring.")
                self.reranker = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid+rerank",
        use_router: bool = True
    ) -> list[RetrievedChunk]:
        chunks_map = self.index.chunks_dict
        if not chunks_map:
            return []

        top_k_dense = self.cfg.retrieval.top_k_dense
        top_k_sparse = self.cfg.retrieval.top_k_sparse

        # Single modes
        if mode == "dense":
            dense_res = self.index.dense_search(query, top_k=top_k_dense)
            retrieved = []
            for rank, (cid, score) in enumerate(dense_res[:top_k]):
                chunk = chunks_map[cid]
                final_score = self.router.apply_boost(query, chunk, score) if use_router else score
                retrieved.append(RetrievedChunk(
                    chunk=chunk,
                    score=final_score,
                    dense_rank=rank + 1
                ))
            return sorted(retrieved, key=lambda x: x.score, reverse=True)[:top_k]

        if mode == "bm25":
            sparse_res = self.index.sparse_search(query, top_k=top_k_sparse)
            retrieved = []
            for rank, (cid, score) in enumerate(sparse_res[:top_k]):
                chunk = chunks_map[cid]
                final_score = self.router.apply_boost(query, chunk, score) if use_router else score
                retrieved.append(RetrievedChunk(
                    chunk=chunk,
                    score=final_score,
                    sparse_rank=rank + 1
                ))
            return sorted(retrieved, key=lambda x: x.score, reverse=True)[:top_k]

        # Candidate retrieval for hybrid modes
        dense_res = self.index.dense_search(query, top_k=top_k_dense)
        sparse_res = self.index.sparse_search(query, top_k=top_k_sparse)

        fused = reciprocal_rank_fusion(dense_res, sparse_res, rrf_k=self.cfg.retrieval.rrf_k)
        
        candidate_list: list[RetrievedChunk] = []
        for cid, meta in fused.items():
            if cid not in chunks_map:
                continue
            chunk = chunks_map[cid]
            base_score = meta["rrf_score"]
            boosted_score = self.router.apply_boost(query, chunk, base_score) if use_router else base_score
            
            candidate_list.append(RetrievedChunk(
                chunk=chunk,
                score=boosted_score,
                dense_rank=meta["dense_rank"],
                sparse_rank=meta["sparse_rank"],
                rrf_score=base_score
            ))

        # Sort candidate list by fused RRF score
        candidate_list.sort(key=lambda x: x.score, reverse=True)

        if mode == "hybrid":
            return candidate_list[:top_k]

        # Rerank mode: cross-encoder over top candidates
        if mode == "hybrid+rerank":
            self._init_reranker()
            top_candidates = candidate_list[:min(15, len(candidate_list))]
            if self.reranker and top_candidates:
                pairs = [(query, rc.chunk.text) for rc in top_candidates]
                scores = self.reranker.predict(pairs)
                for rc, r_score in zip(top_candidates, scores):
                    rc.rerank_score = float(r_score)
                    rc.score = self.router.apply_boost(query, rc.chunk, float(r_score)) if use_router else float(r_score)
                top_candidates.sort(key=lambda x: x.score, reverse=True)
                return top_candidates[:top_k]
            else:
                return candidate_list[:top_k]

        return candidate_list[:top_k]
