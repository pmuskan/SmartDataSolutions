import json

from src.chunking import ChunkingEngine
from src.config import load_config
from src.ingest.figure_parser import FigureParser
from src.ingest.table_parser import TableParser
from src.ingest.text_parser import TextParser
from src.schemas import Chunk


def run_ingestion_pipeline(config_path: str = "config.yaml") -> list[Chunk]:
    cfg = load_config(config_path)
    pdf_path = str(cfg.resolve_path(cfg.pdf_path))
    processed_dir = cfg.resolve_path(cfg.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting ingestion pipeline for: {pdf_path}")

    # 1. Parse text pages
    text_parser = TextParser(pdf_path)
    pages_data = text_parser.parse_pages()
    print(f"Extracted {len(pages_data)} pages of text.")

    # 2. Parse financial tables
    table_parser = TableParser(pdf_path)
    tables_data, qa_report = table_parser.parse_tables()
    print(f"Extracted {len(tables_data)} tables across pages.")

    # 3. Parse figures / images
    figure_parser = FigureParser(pdf_path, output_dir=str(processed_dir / "figures"))
    figures_data, figure_decisions = figure_parser.parse_figures()
    print(f"Extracted {len(figures_data)} figures/logos (decisions logged: {len(figure_decisions)}).")

    # 4. Run Chunking Engine
    chunker = ChunkingEngine(
        target_chunk_size=cfg.chunking.text_chunk_size,
        overlap=cfg.chunking.text_chunk_overlap
    )
    
    text_chunks = chunker.chunk_text_pages(pages_data)
    table_chunks = chunker.chunk_tables(tables_data)
    figure_chunks = chunker.chunk_figures(figures_data)

    all_chunks = text_chunks + table_chunks + figure_chunks
    print(
        f"Generated {len(all_chunks)} chunks "
        f"({len(text_chunks)} text, {len(table_chunks)} table, {len(figure_chunks)} figure)."
    )

    # 5. Write data/processed/chunks.jsonl
    chunks_file = processed_dir / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        f.writelines(chunk.model_dump_json() + "\n" for chunk in all_chunks)
    print(f"Saved chunks to {chunks_file}")

    # 6. Save QA Report & Figure Decisions
    full_qa = {
        "pdf_path": pdf_path,
        "page_count": len(pages_data),
        "total_chunks": len(all_chunks),
        "chunk_counts": {
            "text": len(text_chunks),
            "table": len(table_chunks),
            "figure": len(figure_chunks)
        },
        "table_qa_report": qa_report,
        "figure_decisions": figure_decisions
    }
    qa_file = processed_dir / "parsing_qa_report.json"
    with open(qa_file, "w", encoding="utf-8") as f:
        json.dump(full_qa, f, indent=2)
    print(f"Saved parsing QA report to {qa_file}")

    return all_chunks


if __name__ == "__main__":
    run_ingestion_pipeline()
