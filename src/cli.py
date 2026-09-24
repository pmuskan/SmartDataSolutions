import argparse
import json
import sys

from src.config import load_config
from src.generate import RAGGenerator
from src.index import VectorAndSparseIndex
from src.ingest.pipeline import run_ingestion_pipeline
from src.llm import LLMClient
from src.retrieve import HybridRetriever
from src.schemas import Chunk


def main():
    parser = argparse.ArgumentParser(description="Apple 10-Q Multimodal RAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    ask_parser = subparsers.add_parser("ask", help="Ask a question about the Apple 10-Q filing")
    ask_parser.add_argument("query", type=str, help="Question to ask")
    ask_parser.add_argument("--mode", type=str, default="hybrid+rerank", choices=["dense", "bm25", "hybrid", "hybrid+rerank"], help="Retrieval mode")
    ask_parser.add_argument("--k", type=int, default=5, help="Top k chunks to retrieve")
    ask_parser.add_argument("--json", action="store_true", help="Output raw JSON response")
    ask_parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")

    ingest_parser = subparsers.add_parser("ingest", help="Run ingestion pipeline")
    ingest_parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "ingest":
        run_ingestion_pipeline(args.config)
        sys.exit(0)

    if args.command == "ask":
        cfg = load_config(args.config)
        processed_chunks_file = cfg.resolve_path(cfg.processed_dir) / "chunks.jsonl"
        if not processed_chunks_file.exists():
            print("Processing chunks for index...")
            chunks = run_ingestion_pipeline(args.config)
        else:
            chunks = []
            with open(processed_chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunks.append(Chunk(**json.loads(line)))

        index = VectorAndSparseIndex(args.config)
        index.load_index(chunks)
        retriever = HybridRetriever(index, args.config)
        llm_client = LLMClient(args.config)
        generator = RAGGenerator(llm_client, args.config)

        retrieved = retriever.retrieve(query=args.query, top_k=args.k, mode=args.mode)
        answer = generator.generate_answer(query=args.query, retrieved_chunks=retrieved)

        if args.json:
            out = answer.model_dump()
            print(json.dumps(out, indent=2))
        else:
            print("\n=======================================================")
            print(f"QUESTION: {args.query}")
            print("=======================================================")
            print(f"\nANSWER:\n{answer.text}\n")
            if answer.citations:
                print("CITATIONS:")
                for c in answer.citations:
                    print(f"  - Page {c.page}: {c.section_or_table_title}")
            if answer.tool_calls:
                print("\nCALCULATOR TOOL TRACES:")
                for tc in answer.tool_calls:
                    print(f"  - Formula: {tc.expression} -> Result: {tc.result}")
            print(f"\n[Latency: {answer.latency_ms:.1f} ms | Abstained: {answer.abstained}]")
            print("=======================================================\n")


if __name__ == "__main__":
    main()
