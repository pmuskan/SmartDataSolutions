import hashlib
from typing import Any

from src.schemas import Chunk


def generate_chunk_id(chunk_type: str, page: int, idx: int, section_path: str, title: str, snippet: str) -> str:
    """Generate deterministic unique chunk ID using sha256 hash."""
    raw = f"{chunk_type}|p{page}_i{idx}|{section_path}|{title}|{snippet[:100]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class ChunkingEngine:
    def __init__(self, target_chunk_size: int = 500, overlap: int = 50):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def chunk_text_pages(self, pages_data: list[dict[str, Any]]) -> list[Chunk]:
        chunks = []

        for page_info in pages_data:
            page_num = page_info["page"]
            section_path = page_info["section_path"]
            text = page_info["text"]

            if not text.strip():
                continue

            # Split text into paragraphs
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [text]

            current_words = []
            chunk_count_on_page = 0
            
            for p in paragraphs:
                words = p.split()
                if len(current_words) + len(words) <= self.target_chunk_size:
                    current_words.extend(words)
                else:
                    if current_words:
                        chunk_count_on_page += 1
                        chunk_str = " ".join(current_words)
                        prefixed_text = f"[{section_path}]\n{chunk_str}"
                        chunk_id = generate_chunk_id("text", page_num, chunk_count_on_page, section_path, "", chunk_str)

                        chunks.append(Chunk(
                            id=chunk_id,
                            type="text",
                            page=page_num,
                            section_path=section_path,
                            title="",
                            units="",
                            text=prefixed_text,
                            embed_text=prefixed_text,
                            metadata={"page": page_num, "word_count": len(current_words)}
                        ))

                    # Start next chunk with overlap
                    overlap_words = current_words[-self.overlap:] if len(current_words) >= self.overlap else current_words
                    current_words = overlap_words + words

            if current_words:
                chunk_count_on_page += 1
                chunk_str = " ".join(current_words)
                prefixed_text = f"[{section_path}]\n{chunk_str}"
                chunk_id = generate_chunk_id("text", page_num, chunk_count_on_page, section_path, "", chunk_str)

                chunks.append(Chunk(
                    id=chunk_id,
                    type="text",
                    page=page_num,
                    section_path=section_path,
                    title="",
                    units="",
                    text=prefixed_text,
                    embed_text=prefixed_text,
                    metadata={"page": page_num, "word_count": len(current_words)}
                ))

        return chunks

    def chunk_tables(self, tables_data: list[dict[str, Any]]) -> list[Chunk]:
        chunks = []
        for idx, t in enumerate(tables_data):
            page_num = t["page"]
            title = t["title"]
            units = t["units"]
            summary = t["summary"]
            markdown = t["markdown"]

            section_path = f"Financial Table > Page {page_num} > {title}"
            
            full_text = f"### [Table] {title} (Page {page_num})\nUnits: {units or 'N/A'}\nSummary: {summary}\n\n{markdown}"
            embed_text = f"Table Title: {title}. Page: {page_num}. Units: {units or 'N/A'}. Summary: {summary}"
            
            chunk_id = generate_chunk_id("table", page_num, idx + 1, section_path, title, summary)

            chunks.append(Chunk(
                id=chunk_id,
                type="table",
                page=page_num,
                section_path=section_path,
                title=title,
                units=units,
                text=full_text,
                embed_text=embed_text,
                metadata={
                    "table_id": t["table_id"],
                    "page": page_num,
                    "num_rows": len(t["rows"]),
                    "headers": t["headers"]
                }
            ))

        return chunks

    def chunk_figures(self, figures_data: list[dict[str, Any]]) -> list[Chunk]:
        chunks = []
        for idx, f in enumerate(figures_data):
            page_num = f["page"]
            title = f["title"]
            caption = f["caption"]
            surrounding = f.get("surrounding_text", "")

            section_path = f"Figure > Page {page_num} > {title}"
            full_text = f"### [Figure] {title} (Page {page_num})\nCaption: {caption}\nContext: {surrounding}"
            embed_text = f"Figure: {title}. Caption: {caption}. Page: {page_num}"

            chunk_id = generate_chunk_id("figure", page_num, idx + 1, section_path, title, caption)

            chunks.append(Chunk(
                id=chunk_id,
                type="figure",
                page=page_num,
                section_path=section_path,
                title=title,
                units="",
                text=full_text,
                embed_text=embed_text,
                metadata={
                    "figure_id": f["figure_id"],
                    "image_path": f["image_path"],
                    "page": page_num
                }
            ))

        return chunks
