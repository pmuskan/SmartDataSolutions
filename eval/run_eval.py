import json
from datetime import datetime
from typing import Any

import numpy as np

from eval.judge import LLMJudge
from eval.metrics import evaluate_item
from src.config import load_config
from src.generate import RAGGenerator
from src.index import VectorAndSparseIndex
from src.ingest.pipeline import run_ingestion_pipeline
from src.llm import LLMClient
from src.retrieve import HybridRetriever
from src.schemas import Chunk, EvalItem, EvalResult


def load_golden_set(golden_path: str) -> list[EvalItem]:
    items = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                items.append(EvalItem(**data))
    return items


def run_evaluation(
    config_path: str = "config.yaml",
    retrieval_mode: str = "hybrid+rerank",
    use_router: bool = True,
    top_k: int = 5
) -> dict[str, Any]:
    cfg = load_config(config_path)
    golden_path = str(cfg.resolve_path(cfg.eval.golden_set_path))
    results_dir = cfg.resolve_path(cfg.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    processed_chunks_file = cfg.resolve_path(cfg.processed_dir) / "chunks.jsonl"
    if not processed_chunks_file.exists():
        print("Chunks file not found. Running ingestion pipeline...")
        chunks = run_ingestion_pipeline(config_path)
    else:
        chunks = []
        with open(processed_chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(Chunk(**json.loads(line)))

    # 1. Initialize Index and Retriever
    index = VectorAndSparseIndex(config_path)
    index.load_index(chunks)
    retriever = HybridRetriever(index, config_path)
    llm_client = LLMClient(config_path)
    generator = RAGGenerator(llm_client, config_path)
    judge = LLMJudge(llm_client)

    golden_items = load_golden_set(golden_path)
    print(f"Running evaluation on {len(golden_items)} golden set items (Mode: {retrieval_mode}, Router: {use_router}, top_k: {top_k})...")

    eval_results: list[EvalResult] = []
    failures: list[dict[str, Any]] = []

    for idx, item in enumerate(golden_items):
        print(f"[{idx+1}/{len(golden_items)}] Question ({item.id} - {item.type}): {item.question[:60]}...")

        # Retrieval
        retrieved_chunks = retriever.retrieve(
            query=item.question,
            top_k=top_k,
            mode=retrieval_mode,
            use_router=use_router
        )

        # Generation
        answer = generator.generate_answer(query=item.question, retrieved_chunks=retrieved_chunks)

        # Context for Judge
        context_str = "\n".join([rc.chunk.text for rc in retrieved_chunks])
        
        # LLM Judge evaluation
        judge_score, judge_verdict, faithfulness = judge.evaluate_answer(
            question=item.question,
            gold_answer=item.gold_answer,
            assistant_answer=answer.text,
            context=context_str
        )

        result = evaluate_item(
            item=item,
            retrieved_chunks=retrieved_chunks,
            answer=answer,
            judge_score=judge_score,
            judge_verdict=judge_verdict,
            faithfulness_score=faithfulness
        )
        eval_results.append(result)

        # Track failure details
        if judge_score < 1.0 or not result.hit_at_5 or (result.numeric_match is False):
            failures.append({
                "item_id": item.id,
                "question": item.question,
                "type": item.type,
                "gold_answer": item.gold_answer,
                "generated_answer": answer.text,
                "judge_verdict": judge_verdict,
                "hit_at_5": result.hit_at_5,
                "numeric_match": result.numeric_match,
                "retrieved_chunk_ids": [rc.chunk.id for rc in retrieved_chunks],
                "retrieved_pages": [rc.chunk.page for rc in retrieved_chunks]
            })

    # Aggregate Metrics Calculation
    latencies = [r.latency_ms for r in eval_results]
    p50_latency = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    hit_1_rate = np.mean([r.hit_at_1 for r in eval_results])
    hit_3_rate = np.mean([r.hit_at_3 for r in eval_results])
    hit_5_rate = np.mean([r.hit_at_5 for r in eval_results])
    mrr_avg = np.mean([r.mrr for r in eval_results])
    recall_avg = np.mean([r.recall for r in eval_results])
    page_hit_rate = np.mean([r.page_hit for r in eval_results])
    modality_acc = np.mean([r.modality_correct for r in eval_results])

    numeric_results = [r.numeric_match for r in eval_results if r.numeric_match is not None]
    numeric_exact_match = float(np.mean(numeric_results)) if numeric_results else 0.0

    judge_scores = [r.text_judge_score for r in eval_results if r.text_judge_score is not None]
    avg_judge_score = float(np.mean(judge_scores)) if judge_scores else 0.0

    faithfulness_scores = [r.faithfulness_score for r in eval_results if r.faithfulness_score is not None]
    avg_faithfulness = float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0

    citation_precisions = [r.citation_precision for r in eval_results if r.citation_precision is not None]
    avg_citation_precision = float(np.mean(citation_precisions)) if citation_precisions else 0.0

    unanswerable_results = [r.abstention_correct for r in eval_results if r.type == "unanswerable"]
    refusal_rate = float(np.mean(unanswerable_results)) if unanswerable_results else 1.0

    answerable_results = [not r.abstention_correct for r in eval_results if r.type != "unanswerable"]
    false_refusal_rate = float(np.mean(answerable_results)) if answerable_results else 0.0

    summary_metrics = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "mode": retrieval_mode,
            "use_router": use_router,
            "top_k": top_k,
            "llm_provider": cfg.models.llm_provider
        },
        "total_items": len(golden_items),
        "retrieval": {
            "hit_at_1": round(float(hit_1_rate), 4),
            "hit_at_3": round(float(hit_3_rate), 4),
            "hit_at_5": round(float(hit_5_rate), 4),
            "mrr": round(float(mrr_avg), 4),
            "recall": round(float(recall_avg), 4),
            "page_hit_rate": round(float(page_hit_rate), 4),
            "modality_accuracy": round(float(modality_acc), 4)
        },
        "generation": {
            "numeric_exact_match": round(float(numeric_exact_match), 4),
            "text_judge_score": round(float(avg_judge_score), 4),
            "faithfulness_score": round(float(avg_faithfulness), 4),
            "citation_precision": round(float(avg_citation_precision), 4),
            "refusal_rate": round(float(refusal_rate), 4),
            "false_refusal_rate": round(float(false_refusal_rate), 4)
        },
        "operational": {
            "p50_latency_ms": round(p50_latency, 2),
            "p95_latency_ms": round(p95_latency, 2)
        },
        "type_breakdown": {}
    }

    # Breakdown by question type
    types = set([item.type for item in golden_items])
    for t in types:
        t_results = [r for r in eval_results if r.type == t]
        summary_metrics["type_breakdown"][t] = {
            "count": len(t_results),
            "hit_at_5": round(float(np.mean([r.hit_at_5 for r in t_results])), 4),
            "judge_score": round(float(np.mean([r.text_judge_score for r in t_results if r.text_judge_score is not None])), 4) if t_results else 0.0
        }

    # Save timestamped results JSON
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"eval_results_{ts_str}.json"
    latest_json_path = results_dir / "latest_eval_results.json"

    out_payload = {
        "summary": summary_metrics,
        "results": [r.model_dump() for r in eval_results],
        "failures": failures
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    with open(latest_json_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print(f"\nSaved evaluation summary to {json_path}")
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Hit@5: {summary_metrics['retrieval']['hit_at_5']*100:.1f}% | MRR: {summary_metrics['retrieval']['mrr']:.3f} | Modality Acc: {summary_metrics['retrieval']['modality_accuracy']*100:.1f}%")
    print(f"Numeric Exact Match: {summary_metrics['generation']['numeric_exact_match']*100:.1f}% | LLM Judge Score: {summary_metrics['generation']['text_judge_score']*100:.1f}%")
    print(f"Faithfulness: {summary_metrics['generation']['faithfulness_score']*100:.1f}% | Refusal Rate: {summary_metrics['generation']['refusal_rate']*100:.1f}%")
    print(f"Latency p50: {summary_metrics['operational']['p50_latency_ms']} ms | p95: {summary_metrics['operational']['p95_latency_ms']} ms")
    print("====================================================\n")

    return summary_metrics


if __name__ == "__main__":
    run_evaluation()
