import json
from pathlib import Path
from typing import Any

from eval.run_eval import run_evaluation
from src.config import load_config


def run_all_ablations(config_path: str = "config.yaml") -> dict[str, Any]:
    cfg = load_config(config_path)
    results_dir = cfg.resolve_path(cfg.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ablation_configs = [
        {"name": "dense_only", "mode": "dense", "router": True, "k": 5},
        {"name": "bm25_only", "mode": "bm25", "router": True, "k": 5},
        {"name": "hybrid_no_rerank", "mode": "hybrid", "router": True, "k": 5},
        {"name": "hybrid_rerank_no_router", "mode": "hybrid+rerank", "router": False, "k": 5},
        {"name": "full_pipeline_k3", "mode": "hybrid+rerank", "router": True, "k": 3},
        {"name": "full_pipeline_k5", "mode": "hybrid+rerank", "router": True, "k": 5},
        {"name": "full_pipeline_k8", "mode": "hybrid+rerank", "router": True, "k": 8},
    ]

    ablation_results = []
    print(f"Starting ablation study across {len(ablation_configs)} configurations...\n")

    for ac in ablation_configs:
        print(f"=== Running Ablation Config: {ac['name']} ===")
        res = run_evaluation(
            config_path=config_path,
            retrieval_mode=ac["mode"],
            use_router=ac["router"],
            top_k=ac["k"]
        )
        ablation_results.append({
            "name": ac["name"],
            "mode": ac["mode"],
            "router": ac["router"],
            "k": ac["k"],
            "hit_at_5": res["retrieval"]["hit_at_5"],
            "mrr": res["retrieval"]["mrr"],
            "modality_acc": res["retrieval"]["modality_accuracy"],
            "numeric_match": res["generation"]["numeric_exact_match"],
            "text_judge": res["generation"]["text_judge_score"],
            "faithfulness": res["generation"]["faithfulness_score"],
            "p50_latency_ms": res["operational"]["p50_latency_ms"]
        })

    # Generate Markdown summary table
    md_lines = [
        "# Multimodal RAG Ablation Study Results\n",
        "Comparison of retrieval modes, query router, reranker, and top-k context window choices.\n",
        "| Configuration | Mode | Router | Top-K | Hit@5 | MRR | Modality Acc | Numeric Match | LLM Judge | Faithfulness | Latency p50 (ms) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    ]

    for r in ablation_results:
        md_lines.append(
            f"| **{r['name']}** | {r['mode']} | {r['router']} | {r['k']} | "
            f"{r['hit_at_5']*100:.1f}% | {r['mrr']:.3f} | {r['modality_acc']*100:.1f}% | "
            f"{r['numeric_match']*100:.1f}% | {r['text_judge']*100:.1f}% | {r['faithfulness']*100:.1f}% | "
            f"{r['p50_latency_ms']} |"
        )

    summary_md_path = results_dir / "ablation_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    summary_json_path = results_dir / "ablation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    print(f"\nAblation study complete. Results saved to {summary_md_path}")
    
    # Generate failures analysis file
    generate_failure_analysis(results_dir)

    return {"ablations": ablation_results}


def generate_failure_analysis(results_dir: Path):
    latest_json = results_dir / "latest_eval_results.json"
    if not latest_json.exists():
        return

    with open(latest_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    failures = data.get("failures", [])
    md_lines = [
        "# Evaluation Failure Analysis & Error Categorization\n",
        f"Total Failed or Imperfect Items: **{len(failures)}**\n",
    ]

    for f_idx, fail in enumerate(failures):
        md_lines.append(f"### Failure {f_idx+1}: [{fail['item_id']}] ({fail['type']})")
        md_lines.append(f"**Question**: {fail['question']}")
        md_lines.append(f"**Gold Answer**: {fail['gold_answer']}")
        md_lines.append(f"**Generated Answer**: {fail['generated_answer']}")
        md_lines.append(f"**Judge Verdict**: `{fail['judge_verdict']}` | **Hit@5**: `{fail['hit_at_5']}` | **Numeric Match**: `{fail['numeric_match']}`")
        md_lines.append(f"**Retrieved Pages**: {fail['retrieved_pages']}\n")
        md_lines.append("---")

    failures_md_path = results_dir / "failures.md"
    with open(failures_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Failure analysis saved to {failures_md_path}")


if __name__ == "__main__":
    run_all_ablations()
