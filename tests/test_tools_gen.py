import os

from src.generate import RAGGenerator
from src.ingest.figure_parser import FigureParser
from src.llm import LLMClient
from src.schemas import Chunk, RetrievedChunk
from src.tools import (
    calculate_share_percentage,
    calculate_yoy_growth,
    evaluate_calculator_expression,
)
from tests.fixtures.synthetic_figure_pdf import generate_synthetic_figure_pdf


def test_calculator_evaluation():
    # Simple addition and multiplication
    tc1 = evaluate_calculator_expression("19604 + 63355")
    assert tc1.result == 82959

    # YoY calculation
    tc2 = calculate_yoy_growth(current=19604, prior=17486)
    assert abs(tc2.result - 12.1125) < 0.01

    # Share percentage
    tc3 = calculate_share_percentage(part=19604, total=82959)
    assert abs(tc3.result - 23.631) < 0.01

    # Division by zero safety
    tc4 = evaluate_calculator_expression("100 / 0")
    assert "Error" in str(tc4.result)

    # Malicious expression safety (AST should block function calls)
    tc5 = evaluate_calculator_expression("__import__('os').system('dir')")
    assert "Error" in str(tc5.result)


def test_abstention_path():
    generator = RAGGenerator(llm_client=LLMClient())
    # Empty retrieved context should abstain immediately
    ans = generator.generate_answer(query="What was revenue in 2030?", retrieved_chunks=[])
    assert ans.abstained is True
    assert ans.text == "The requested information is not found in the provided document."

    # Greeting should abstain immediately without returning random numbers
    ans_greeting = generator.generate_answer(query="hi", retrieved_chunks=[])
    assert ans_greeting.abstained is True
    assert ans_greeting.text == "The requested information is not found in the provided document."


def test_synthetic_figure_pipeline(tmp_path):
    pdf_path = str(tmp_path / "sample_chart.pdf")
    generate_synthetic_figure_pdf(pdf_path)
    assert os.path.exists(pdf_path)

    fig_parser = FigureParser(pdf_path=pdf_path, output_dir=str(tmp_path / "figures"))
    figures, decisions = fig_parser.parse_figures()

    assert len(figures) >= 1
    assert "Page 1" in figures[0]["title"] or "Figure" in figures[0]["title"] or "Logo" in figures[0]["title"]
    assert decisions[0]["action"] == "extracted"


def test_end_to_end_rag_smoke():
    chunk = Chunk(
        id="c1",
        type="table",
        page=4,
        section_path="Condensed Consolidated Statements of Operations",
        title="Statements of Operations",
        units="in millions",
        text="Total net sales for three months ended June 25, 2022 was $82,959 million [p.4].",
        embed_text="Total net sales 82,959"
    )
    rc = RetrievedChunk(chunk=chunk, score=0.95, dense_rank=1)
    
    generator = RAGGenerator(llm_client=LLMClient())
    ans = generator.generate_answer("What were total net sales for the three months ended June 25, 2022?", [rc])
    
    assert ans.abstained is False
    assert "82,959" in ans.text or "82959" in ans.text
