from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def build_solution_overview_pdf(output_path: str = "report/solution_overview.pdf"):
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=44,
        leftMargin=44,
        topMargin=44,
        bottomMargin=44
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1D1D1F"),
        alignment=0,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#6E6E73"),
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#0071E3"),
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1D1D1F"),
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=10,
        firstLineIndent=-6,
        spaceAfter=2.5
    )

    story = []

    # Title Header
    story.append(Paragraph("System Architecture & Solution Overview", title_style))
    story.append(Paragraph("Multimodal RAG Engine — Apple Inc. Form 10-Q (Q3 2022)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0071E3"), spaceAfter=10))

    # Executive Summary Card
    summary_html = (
        "<b>Selected Core Strategy:</b> Single-unit financial table chunking + Hybrid Retrieval "
        "(Chroma Dense vector search + BM25Okapi sparse keyword search combined via Reciprocal Rank Fusion, RRF k=60) "
        "+ Intent-based Query Routing + Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`) + Safe AST Calculator Tool."
    )
    summary_table = Table([[Paragraph(summary_html, body_style)]], colWidths=[524])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F5F7")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#D2D2D7")),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 8))

    # 1. Approach & Methodology
    story.append(Paragraph("1. Approach & Methodology", h1_style))

    p1_points = [
        "<b>Document Parsing & Normalization:</b> PyMuPDF extracts text blocks and page hierarchy; pdfplumber extracts financial tables. Negative numbers in parentheses (e.g., <i>(1,234)</i>) are converted to standard negatives <i>-1234</i>, and multi-tier headers are flattened into clean column descriptions.",
        "<b>Single-Unit Table Chunking:</b> Text sections are split at 500-token boundaries with 50-token overlap. Financial tables are kept as single atomic markdown units so numbers, rows, and headers are never split across chunks.",
        "<b>Dual Indexing & Hybrid Retrieval:</b> Dense semantic vectors (`BAAI/bge-base-en-v1.5` in Chroma DB) capture intent, while sparse keyword indexes (`rank_bm25`) catch exact numbers and financial terms. Results are merged using Reciprocal Rank Fusion (RRF).",
        "<b>Intent Routing & Reranking:</b> A lightweight router boosts relevant chunk types (table, text, figure) based on query keywords. Top candidates are then reranked with a Cross-Encoder model to ensure high contextual precision.",
        "<b>Grounded Generation & Calculator Tool:</b> Answers must cite explicit pages `[p.X, Title]` and state units (e.g., millions, $). Math expressions (YoY %, margins) are evaluated via Python AST to guarantee exact arithmetic."
    ]
    for pt in p1_points:
        story.append(Paragraph(f"• {pt}", bullet_style))

    story.append(Spacer(1, 6))

    # 2. Design Decisions & Reasoning
    story.append(Paragraph("2. Key Design Decisions & Technical Reasoning", h1_style))

    decisions_data = [
        ["Design Decision", "Reasoning & Business Impact"],
        [
            "Single-Unit Table Chunking",
            "Splitting tables by token count breaks row alignment and column headers. Keeping each table intact as a markdown unit preserved 100% structural context for LLM extraction."
        ],
        [
            "Hybrid RRF (Dense + BM25)",
            "Dense embeddings often fail on exact numeric queries or specific page items. BM25 guarantees keyword matches while Dense handles natural phrasing. RRF merges rankings without needing score scale calibration."
        ],
        [
            "AST Calculator Tool",
            "LLMs frequently miscalculate YoY growth percentages or margins. Parsing expressions like CALCULATE: ((19604-17486)/17486)*100 into a python AST evaluator eliminated math hallucinations."
        ],
        [
            "Strict Abstention & Guardrails",
            "Out-of-scope questions or casual greetings ('hi') previously risked hallucinating financial numbers. Adding greeting filters and context checks enforces clean abstention ('The requested information is not found in the provided document.')."
        ]
    ]

    t_dec = Table(decisions_data, colWidths=[140, 384])
    t_dec.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0071E3")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D2D2D7")),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#FFFFFF")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(t_dec)

    story.append(Spacer(1, 8))

    # 3. Assumptions, Challenges & Improvements
    story.append(Paragraph("3. Assumptions, Challenges & Future Improvements", h1_style))

    p3_points = [
        "<b>Scope Assumption:</b> System is built and benchmarked specifically for Apple Inc.'s 28-page Q3 2022 Form 10-Q filing. The document is heavily text- and table-centric with minimal graphical figures (cover logo).",
        "<b>Extracted Table Challenges:</b> Multi-column financial layouts with indented sub-line items were resolved by concatenating parent section titles into chunk metadata path strings.",
        "<b>Non-Financial Query Handling:</b> Handled non-question inputs ('hi', 'hello', 'who are you') gracefully by returning explicit abstention rather than retrieving arbitrary context.",
        "<b>Future Improvements:</b> Extend dual-index layer to support multi-period 10-K/10-Q filing trends, introduce Table-to-SQL execution for multi-table aggregations, and add vision LLM extraction for chart-heavy filings."
    ]
    for pt in p3_points:
        story.append(Paragraph(f"• {pt}", bullet_style))

    doc.build(story)
    print(f"Solution Overview PDF created successfully at: {output_path}")


if __name__ == "__main__":
    build_solution_overview_pdf()
