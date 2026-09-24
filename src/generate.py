import re
import time

from src.config import load_config
from src.llm import LLMClient
from src.schemas import Answer, Citation, RetrievedChunk, ToolCall
from src.tools import evaluate_calculator_expression

SYSTEM_PROMPT = """You are an expert financial analyst answering questions about Apple Inc.'s Form 10-Q filing for Q3 2022.
Follow these strict guidelines:
1. Answer ONLY using the provided retrieved context. Do NOT use outside knowledge.
2. Every claim or number MUST be backed by a clear citation in the format: [p.X, Section/Table title].
3. ALWAYS state the units (e.g., in millions, in thousands, $) and the time period (e.g., three months ended June 25, 2022) with every single numeric value mentioned.
4. If a calculation (e.g. YoY %, difference, share of total, gross margin change) is needed, specify the formula using CALCULATE: <expression> (e.g. CALCULATE: ((19604 - 17486) / 17486) * 100).
5. IF THE QUERY IS A GREETING, CHITCHAT, OR DOES NOT ASK A SPECIFIC QUESTION ABOUT THE DOCUMENT, OR IF THE CONTEXT DOES NOT CONTAIN THE ANSWER TO THE QUESTION, reply EXACTLY with:
"The requested information is not found in the provided document." and nothing else.
"""

GREETINGS_AND_CHITCHAT = {
    "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
    "good evening", "how are you", "who are you", "what can you do", "test",
    "hi!", "hello!", "hey!", "yo", "sup", "help", "thanks", "thank you"
}


def extract_citations(text: str) -> list[Citation]:
    """Parse inline citations like [p.1, Condensed Consolidated Statements of Operations]."""
    citations = []
    matches = re.findall(r"\[p\.?\s*(\d+)(?:,\s*([^\]]+))?\]", text)
    for m in matches:
        page_num = int(m[0])
        title = m[1].strip() if m[1] else f"Page {page_num}"
        citations.append(Citation(page=page_num, section_or_table_title=title))
    return citations


class RAGGenerator:
    def __init__(self, llm_client: LLMClient | None = None, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        self.llm_client = llm_client or LLMClient(config_path)
        self.abstention_msg = self.cfg.generation.abstention_message

    def generate_answer(self, query: str, retrieved_chunks: list[RetrievedChunk]) -> Answer:
        start_time = time.time()

        clean_q = query.strip().lower().rstrip("!?.").strip()
        if clean_q in GREETINGS_AND_CHITCHAT or len(clean_q) <= 2:
            return Answer(
                text=self.abstention_msg,
                citations=[],
                used_chunk_ids=[],
                abstained=True,
                latency_ms=(time.time() - start_time) * 1000.0
            )

        if not retrieved_chunks:
            return Answer(
                text=self.abstention_msg,
                citations=[],
                used_chunk_ids=[],
                abstained=True,
                latency_ms=(time.time() - start_time) * 1000.0
            )

        # Build context prompt from retrieved chunks
        context_str_list = []
        used_ids = []
        for idx, rc in enumerate(retrieved_chunks):
            c = rc.chunk
            used_ids.append(c.id)
            context_str_list.append(
                f"--- Context [{idx+1}] (Page {c.page} | Modality: {c.type} | Path: {c.section_path}) ---\n{c.text}\n"
            )

        full_context = "\n".join(context_str_list)
        user_prompt = f"Retrieved Context:\n{full_context}\n\nQuestion: {query}\nAnswer:"

        raw_llm_response = self.llm_client.generate(prompt=user_prompt, system_prompt=SYSTEM_PROMPT)

        # Check for exact abstention signal
        if self.abstention_msg.lower() in raw_llm_response.lower() or "not found in the provided document" in raw_llm_response.lower():
            return Answer(
                text=self.abstention_msg,
                citations=[],
                used_chunk_ids=used_ids,
                abstained=True,
                latency_ms=(time.time() - start_time) * 1000.0
            )

        # Check for calculator calls in LLM output
        tool_calls: list[ToolCall] = []
        calc_matches = re.findall(r"CALCULATE:\s*([0-9\.\+\-\*\/\(\)\s\%\$]+)", raw_llm_response)
        processed_text = raw_llm_response

        for expr in calc_matches:
            tool_call = evaluate_calculator_expression(expr)
            tool_calls.append(tool_call)
            # Replace CALCULATE: expr with computed result in answer text
            if not str(tool_call.result).startswith("Error"):
                processed_text = processed_text.replace(
                    f"CALCULATE: {expr}",
                    f"{tool_call.result}% (calculated from {expr})"
                )

        citations = extract_citations(processed_text)
        latency_ms = (time.time() - start_time) * 1000.0

        return Answer(
            text=processed_text,
            citations=citations,
            used_chunk_ids=used_ids,
            abstained=False,
            tool_calls=tool_calls,
            latency_ms=latency_ms
        )
