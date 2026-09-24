import os
import pickle

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import load_config
from src.schemas import Chunk


class VectorAndSparseIndex:
    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        self.chroma_dir = str(self.cfg.resolve_path(self.cfg.chroma_db_dir))
        self.bm25_path = str(self.cfg.resolve_path(self.cfg.bm25_index_path))
        self.model_name = self.cfg.models.embedding

        self.embedder = None
        self.chroma_client = None
        self.collection = None
        self.bm25 = None
        self.chunks_dict: dict[str, Chunk] = {}
        self.chunk_ids: list[str] = []

    def _init_embedder(self):
        if self.embedder is None:
            self.embedder = SentenceTransformer(self.model_name)

    def _init_chroma(self):
        if self.chroma_client is None:
            os.makedirs(self.chroma_dir, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
            self.collection = self.chroma_client.get_or_create_collection(
                name="aapl_10q_chunks",
                metadata={"hnsw:space": "cosine"}
            )

    def build_index(self, chunks: list[Chunk]):
        """Build both Chroma DB and BM25 index from chunks list."""
        self._init_embedder()
        self._init_chroma()

        self.chunks_dict = {c.id: c for c in chunks}
        self.chunk_ids = [c.id for c in chunks]

        print(f"Embedding {len(chunks)} chunks for Chroma DB...")
        texts_to_embed = [c.embed_text for c in chunks]
        embeddings = self.embedder.encode(texts_to_embed, show_progress_bar=False, normalize_embeddings=True)

        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "type": c.type,
                "page": c.page,
                "section_path": c.section_path,
                "title": c.title,
                "units": c.units
            }
            for c in chunks
        ]

        # Reset collection if exists
        try:
            existing_ids = self.collection.get()["ids"]
            if existing_ids:
                self.collection.delete(ids=existing_ids)
        except Exception:
            pass

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )

        # Build BM25 Index
        tokenized_corpus = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        with open(self.bm25_path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "chunk_ids": self.chunk_ids,
                "chunks_dict": self.chunks_dict
            }, f)

    def load_index(self, chunks: list[Chunk]):
        """Load indices from disk or build if missing or empty."""
        self._init_chroma()
        self.chunks_dict = {c.id: c for c in chunks}
        self.chunk_ids = [c.id for c in chunks]

        # Check if collection is empty
        try:
            count = self.collection.count()
        except Exception:
            count = 0

        if count == 0 or count != len(chunks):
            self.build_index(chunks)
            return

        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
        else:
            tokenized_corpus = [c.text.lower().split() for c in chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)

    def dense_search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        self._init_embedder()
        self._init_chroma()
        try:
            if self.collection.count() == 0:
                return []
        except Exception:
            return []

        query_emb = self.embedder.encode([query], normalize_embeddings=True).tolist()
        try:
            res = self.collection.query(query_embeddings=query_emb, n_results=top_k)
        except Exception as e:
            print(f"Warning: Chroma query failed ({e}), returning empty dense results.")
            return []

        results = []
        if res and res["ids"] and res["ids"][0]:
            ids = res["ids"][0]
            distances = res["distances"][0] if res.get("distances") else [0.0] * len(ids)
            for cid, dist in zip(ids, distances):
                score = max(0.0, 1.0 - dist)
                results.append((cid, float(score)))
        return results

    def sparse_search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        if self.bm25 is None:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:
                cid = self.chunk_ids[idx]
                results.append((cid, score))
        return results
