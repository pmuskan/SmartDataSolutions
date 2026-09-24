import re
from typing import Any

from src.schemas import Answer, EvalItem, EvalResult, RetrievedChunk


def compute_lcs_length(x: list[str], y: list[str]) -> int:
    """Compute length of Longest Common Subsequence between two token lists."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(candidate: str, reference: str) -> float:
    """Calculate ROUGE-L F1 score."""
    cand_tokens = re.findall(r"\w+", candidate.lower())
    ref_tokens = re.findall(r"\w+", reference.lower())

    if not cand_tokens or not ref_tokens:
        return 0.0

    lcs_len = compute_lcs_length(cand_tokens, ref_tokens)
    prec = lcs_len / len(cand_tokens)
    rec = lcs_len / len(ref_tokens)

    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def parse_numeric_values(text: str) -> list[float]:
    """Extract all numeric values from text string."""
    clean_t = text.replace(",", "")
    # Find numbers with optional decimals and percentages
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", clean_t)
    nums = []
    for m in matches:
        try:
            nums.append(float(m))
        except ValueError:
            pass
    return nums


def check_numeric_match(
    assistant_text: str,
    gold_val: float | None,
    gold_unit: str | None = None,
    tolerance: float = 0.05
) -> bool | None:
    """Check if target gold value is present in assistant text within tolerance."""
    if gold_val is None:
        return None

    nums = parse_numeric_values(assistant_text)
    if not nums:
        return False

    for num in nums:
        possible_vals = [num]
        if gold_unit == "billion" or gold_unit == "million":
            possible_vals.extend([num * 1000.0, num / 1000.0])

        for pv in possible_vals:
            if abs(pv - gold_val) <= max(abs(gold_val * tolerance), 0.01):
                return True

    return False


def count_hallucinated_numbers(
    assistant_text: str,
    context_text: str,
    tool_results: list[Any]
) -> int:
    """Count numbers in assistant text that are absent in context and tool results."""
    nums_in_ans = set(parse_numeric_values(assistant_text))
    nums_in_ctx = set(parse_numeric_values(context_text))

    for tr in tool_results:
        if isinstance(tr, (int, float)):
            nums_in_ctx.add(float(tr))

    hallucinated = 0
    for num in nums_in_ans:
        if num in [1.0, 2.0, 3.0, 4.0, 5.0, 2022.0, 2021.0, 2023.0, 25.0, 26.0, 15.0, 10.0, 100.0]:
            continue

        match_found = False
        for ctx_num in nums_in_ctx:
            if abs(num - ctx_num) <= 0.01 or (ctx_num != 0 and abs((num - ctx_num) / ctx_num) <= 0.02):
                match_found = True
                break

        if not match_found:
            hallucinated += 1

    return hallucinated


def evaluate_item(
    item: EvalItem,
    retrieved_chunks: list[RetrievedChunk],
    answer: Answer,
    judge_score: float,
    judge_verdict: str,
    faithfulness_score: float
) -> EvalResult:
    """Compute all evaluation metrics for a single golden set item."""
    retrieved_pages = [rc.chunk.page for rc in retrieved_chunks]
    gold_pages_set = set(item.gold_pages)

    hit_1 = bool(retrieved_pages and retrieved_pages[0] in gold_pages_set) if gold_pages_set else True
    hit_3 = any(p in gold_pages_set for p in retrieved_pages[:3]) if gold_pages_set else True
    hit_5 = any(p in gold_pages_set for p in retrieved_pages[:5]) if gold_pages_set else True

    page_hit = hit_5

    mrr = 0.0
    for rank, p in enumerate(retrieved_pages):
        if p in gold_pages_set:
            mrr = 1.0 / (rank + 1)
            break

    retrieved_matching_pages = len(set(retrieved_pages).intersection(gold_pages_set))
    recall = (retrieved_matching_pages / len(gold_pages_set)) if gold_pages_set else 1.0

    target_modality = item.type
    top_modality = retrieved_chunks[0].chunk.type if retrieved_chunks else "text"
    modality_correct = (
        (target_modality in ["table", "numeric_derived"] and top_modality == "table") or
        (target_modality == "figure" and top_modality == "figure") or
        (target_modality in ["text", "cross_modal", "unanswerable"] and top_modality == "text")
    )

    numeric_match = check_numeric_match(answer.text, item.gold_value, item.gold_unit)
    rouge_l = compute_rouge_l(answer.text, item.gold_answer)

    cited_pages = [c.page for c in answer.citations]
    if cited_pages and gold_pages_set:
        valid_citations = sum(1 for cp in cited_pages if cp in gold_pages_set)
        citation_precision = valid_citations / len(cited_pages)
    else:
        citation_precision = 1.0 if not cited_pages and not gold_pages_set else 0.0

    context_text = " ".join([rc.chunk.text for rc in retrieved_chunks])
    tool_res = [tc.result for tc in answer.tool_calls]
    hallucinated_cnt = count_hallucinated_numbers(answer.text, context_text, tool_res)

    abstention_correct = None
    if item.type == "unanswerable":
        abstention_correct = answer.abstained
    else:
        abstention_correct = not answer.abstained

    return EvalResult(
        item_id=item.id,
        question=item.question,
        type=item.type,
        hit_at_1=hit_1,
        hit_at_3=hit_3,
        hit_at_5=hit_5,
        recall=recall,
        mrr=mrr,
        page_hit=page_hit,
        modality_correct=modality_correct,
        numeric_match=numeric_match,
        text_judge_score=judge_score,
        judge_verdict=judge_verdict,
        rouge_l_score=rouge_l,
        faithfulness_score=faithfulness_score,
        citation_precision=citation_precision,
        hallucinated_number_count=hallucinated_cnt,
        abstention_correct=abstention_correct,
        latency_ms=answer.latency_ms,
        tokens_used=answer.tokens_used
    )
