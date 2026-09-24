from src.ingest.table_parser import (
    clean_cell,
    flatten_two_tier_headers,
    parse_negative_parentheses,
    rows_to_markdown,
)
from src.ingest.text_parser import clean_text, detect_heading


def test_parse_negative_parentheses():
    assert parse_negative_parentheses("(1,234)") == "-1,234"
    assert parse_negative_parentheses("($5,678)") == "-5,678"
    assert parse_negative_parentheses("(16)%") == "-16%"
    assert parse_negative_parentheses("12,345") == "12,345"
    assert parse_negative_parentheses("") == ""


def test_clean_cell():
    assert clean_cell("  Product Name \n ") == "Product Name"
    assert clean_cell(None) == ""
    assert clean_cell("Three\nMonths\nEnded") == "Three Months Ended"


def test_flatten_two_tier_headers():
    rows = [
        ["", "Three Months Ended", "", "Nine Months Ended", ""],
        ["Category", "June 25, 2022", "June 26, 2021", "June 25, 2022", "June 26, 2021"],
        ["Products", "$63,355", "$63,948", "$245,241", "$232,309"],
    ]
    headers, body = flatten_two_tier_headers(rows)
    assert len(headers) == 5
    assert "Three Months Ended" in headers[1]
    assert "June 25, 2022" in headers[1]
    assert len(body) == 1
    assert body[0][0] == "Products"


def test_rows_to_markdown():
    headers = ["Metric", "Q3 2022", "Q3 2021"]
    rows = [["Revenue", "82959", "81434"], ["Cost", "(47074)", "(46179)"]]
    md = rows_to_markdown(headers, rows)
    assert "| Metric | Q3 2022 | Q3 2021 |" in md
    assert "| Revenue | 82959 | 81434 |" in md
    assert "-47074" in md


def test_clean_text():
    raw = "Apple Inc. is a leading technology com-\npany that manufactures iPhones."
    cleaned = clean_text(raw)
    assert "company" in cleaned
    assert "com-\npany" not in cleaned


def test_detect_heading():
    is_head, _title, level = detect_heading("PART I — FINANCIAL INFORMATION")
    assert is_head is True
    assert level == 1

    is_head, _title, level = detect_heading("Item 2. Management's Discussion and Analysis")
    assert is_head is True
    assert level == 2

    is_head, _title, level = detect_heading("Note 9 – Segment Information")
    assert is_head is True
    assert level == 2

    is_head, _title, level = detect_heading("Gross Margin")
    assert is_head is True
    assert level == 3
