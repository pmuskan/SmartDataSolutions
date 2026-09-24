import json

from src.llm import LLMClient

JUDGE_PROMPT_TEMPLATE = """You are an impartial financial evaluator judging the correctness and grounding of an AI assistant's answer.

Question: {question}
Gold Reference Answer: {gold_answer}
AI Assistant Answer: {assistant_answer}
Retrieved Context Provided to Assistant: {context}

Rate the AI Assistant Answer based on the following strict rubric:
1. CORRECT (Score: 1.0): The answer is fully accurate, contains all key facts/values from the gold answer, and is supported by context.
2. PARTIALLY_CORRECT (Score: 0.5): The answer gets the main value or fact right, but misses secondary details or context.
3. INCORRECT (Score: 0.0): The answer contains wrong values, incorrect facts, or contradicts the gold reference.
4. HALLUCINATED (Score: 0.0): The answer makes up facts or numbers not found in the context.

Output format (respond with JSON only):
{{
  "verdict": "CORRECT" | "PARTIALLY_CORRECT" | "INCORRECT" | "HALLUCINATED",
  "score": 1.0 | 0.5 | 0.0,
  "faithfulness_score": 1.0 | 0.5 | 0.0,
  "reasoning": "Short explanation"
}}
"""


class LLMJudge:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client or LLMClient()

    def evaluate_answer(
        self,
        question: str,
        gold_answer: str,
        assistant_answer: str,
        context: str
    ) -> tuple[float, str, float]:
        """Judge assistant answer correctness and faithfulness."""
        # Fast path for exact refusals on unanswerable questions
        if "not found in the provided document" in assistant_answer.lower() and "not found" in gold_answer.lower():
            return 1.0, "CORRECT", 1.0

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            gold_answer=gold_answer,
            assistant_answer=assistant_answer,
            context=context[:2000]
        )

        raw_res = self.llm_client.generate(
            prompt=prompt,
            system_prompt="You are a strict financial evaluator. Output only valid JSON."
        )

        try:
            # Extract JSON block
            json_str = raw_res
            if "```json" in raw_res:
                json_str = raw_res.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_res:
                json_str = raw_res.split("```")[1].split("```")[0].strip()

            parsed = json.loads(json_str)
            score = float(parsed.get("score", 0.0))
            verdict = str(parsed.get("verdict", "INCORRECT"))
            faithfulness = float(parsed.get("faithfulness_score", score))
            return score, verdict, faithfulness
        except Exception:
            # Heuristic fallback if judge JSON parsing fails
            if gold_answer.lower() in assistant_answer.lower():
                return 1.0, "CORRECT", 1.0
            return 0.5, "PARTIALLY_CORRECT", 0.5
