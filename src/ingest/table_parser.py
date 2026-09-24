import re
from typing import Any

import pdfplumber


def parse_negative_parentheses(val: str) -> str:
    """Normalize negative numbers in parentheses e.g. (1,234) -> -1,234."""
    if not val:
        return ""
    val_str = str(val).strip()
    match = re.match(r"^\(([\$\d,]+(?:\.\d+)?)\)%?$", val_str)
    if match:
        num_str = match.group(1).replace("$", "").strip()
        if "%" in val_str:
            return f"-{num_str}%"
        return f"-{num_str}"
    return val_str


def clean_cell(cell: Any | None) -> str:
    if cell is None:
        return ""
    s = str(cell).replace("\n", " ").strip()
    # Replace non-breaking spaces or unprintable chars
    s = re.sub(r"[^\x00-\x7F]+", " ", s).strip()
    return s


def flatten_two_tier_headers(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """
    Detect multi-level headers (e.g. Three Months Ended / Nine Months Ended)
    and flatten them into clean column names.
    """
    if not rows or len(rows) < 2:
        return (rows[0] if rows else [], rows[1:] if len(rows) > 1 else [])

    row0 = [clean_cell(c) for c in rows[0]]
    row1 = [clean_cell(c) for c in rows[1]]

    # Check if top rows look like two-tier header
    combined_str = " ".join(row0 + row1)
    if "Three Months Ended" in combined_str or "Nine Months Ended" in combined_str:
        flattened_headers = []
        last_period = ""
        
        # Build mapping of columns
        max_cols = max(len(row0), len(row1))
        for i in range(max_cols):
            c0 = row0[i] if i < len(row0) else ""
            c1 = row1[i] if i < len(row1) else ""

            if "Three Months Ended" in c0:
                last_period = "Three Months Ended"
            elif "Nine Months Ended" in c0:
                last_period = "Nine Months Ended"

            col_parts = []
            if last_period and c1 and not c1.startswith("Three Months") and not c1.startswith("Nine Months"):
                col_parts.append(last_period)
            if c0 and c0 not in ["Three Months Ended", "Nine Months Ended"]:
                col_parts.append(c0)
            if c1:
                col_parts.append(c1)

            col_name = " ".join(col_parts).strip()
            if not col_name:
                col_name = f"Column_{i+1}"
            flattened_headers.append(col_name)

        # Skip the header rows in body
        body_start_idx = 2
        # If there is a third header row (e.g., June 25, 2022 | June 26, 2021)
        if len(rows) > 2:
            row2 = [clean_cell(c) for c in rows[2]]
            if any(re.search(r"2022|2021", c) for c in row2):
                new_headers = []
                for idx, h in enumerate(flattened_headers):
                    c2 = row2[idx] if idx < len(row2) else ""
                    if c2 and c2 not in h:
                        new_headers.append(f"{h} {c2}".strip())
                    else:
                        new_headers.append(h)
                flattened_headers = new_headers
                body_start_idx = 3

        return flattened_headers, rows[body_start_idx:]

    # Single-tier header default
    headers = [row0[i] if row0[i] else f"Column_{i+1}" for i in range(len(row0))]
    return headers, rows[1:]


def rows_to_markdown(headers: list[str], data_rows: list[list[str]]) -> str:
    """Convert structured header and rows to a clean Markdown table string."""
    if not headers and not data_rows:
        return ""

    # Normalize empty column names
    headers = [h if h else f"Col_{i+1}" for i, h in enumerate(headers)]

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"

    body_lines = []
    for row in data_rows:
        # Pad or truncate row to match headers count
        padded_row = row + [""] * (len(headers) - len(row))
        padded_row = padded_row[:len(headers)]
        cleaned_cells = [parse_negative_parentheses(clean_cell(c)) for c in padded_row]
        body_lines.append("| " + " | ".join(cleaned_cells) + " |")

    return "\n".join([header_line, separator_line] + body_lines)


def generate_table_summary(title: str, page: int, units: str, headers: list[str], data_rows: list[list[str]]) -> str:
    """Generate concise deterministic summary for table chunk embedding."""
    num_rows = len(data_rows)
    non_empty_headers = [h for h in headers if h and not h.startswith("Column_") and not h.startswith("Col_")]
    col_str = ", ".join(non_empty_headers[:5]) if non_empty_headers else "financial metrics"
    unit_str = f" ({units})" if units else ""
    return (
        f"Financial table '{title}' on page {page}{unit_str} with {num_rows} items. "
        f"Columns include: {col_str}."
    )


class TableParser:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def parse_tables(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tables_data = []
        qa_report = {
            "total_tables": 0,
            "tables_per_page": {},
            "flagged_ragged_rows": [],
            "flagged_empty_headers": [],
            "spot_check_sample": []
        }

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                raw_tables = page.extract_tables()
                page_text = page.extract_text() or ""
                
                qa_report["tables_per_page"][page_num] = len(raw_tables)
                qa_report["total_tables"] += len(raw_tables)

                for t_idx, raw_table in enumerate(raw_tables):
                    if not raw_table or len(raw_table) < 2:
                        continue

                    # Filter out empty rows or dollar-only artifact rows
                    cleaned_raw_rows = []
                    for r in raw_table:
                        cells = [clean_cell(c) for c in r]
                        # drop rows that are all empty or just "$" symbols
                        non_empty = [c for c in cells if c and c != "$"]
                        if non_empty:
                            cleaned_raw_rows.append(cells)

                    if not cleaned_raw_rows:
                        continue

                    # Determine title and units from page text preceding table
                    title = f"Table on Page {page_num}"
                    units = ""
                    if "in millions" in page_text.lower():
                        units = "in millions"
                    elif "in thousands" in page_text.lower():
                        units = "in thousands"

                    # Infer title from heading or preceding lines
                    lines_before = [l.strip() for l in page_text.split("\n") if l.strip()]
                    for l in lines_before:
                        if any(k in l for k in ["Statements of", "Balance Sheets", "Note", "Segment", "Net Sales", "Revenue"]):
                            title = l
                            break

                    headers, body_rows = flatten_two_tier_headers(cleaned_raw_rows)

                    # QA Checks
                    col_count = len(headers)
                    has_ragged = any(len(r) != col_count for r in body_rows)
                    has_empty_headers = any(h == "" or h.startswith("Column_") for h in headers)

                    if has_ragged:
                        qa_report["flagged_ragged_rows"].append({"page": page_num, "table_idx": t_idx + 1})
                    if has_empty_headers:
                        qa_report["flagged_empty_headers"].append({"page": page_num, "table_idx": t_idx + 1})

                    markdown_str = rows_to_markdown(headers, body_rows)
                    summary = generate_table_summary(title, page_num, units, headers, body_rows)

                    # Structured row representation
                    structured_rows = []
                    for row in body_rows:
                        if not row:
                            continue
                        row_label = parse_negative_parentheses(clean_cell(row[0]))
                        col_values = {}
                        for c_idx in range(1, len(row)):
                            col_name = headers[c_idx] if c_idx < len(headers) else f"Column_{c_idx+1}"
                            col_values[col_name] = parse_negative_parentheses(clean_cell(row[c_idx]))
                        structured_rows.append({"label": row_label, "values": col_values})

                    table_obj = {
                        "table_id": f"p{page_num}_t{t_idx+1}",
                        "page": page_num,
                        "title": title,
                        "units": units,
                        "headers": headers,
                        "rows": structured_rows,
                        "markdown": markdown_str,
                        "summary": summary
                    }

                    tables_data.append(table_obj)

                    # Spot check recording for 5 key tables
                    if page_num in [4, 6, 8, 10, 16] and len(qa_report["spot_check_sample"]) < 5:
                        qa_report["spot_check_sample"].append({
                            "page": page_num,
                            "table_id": table_obj["table_id"],
                            "title": title,
                            "num_rows": len(body_rows),
                            "sample_header": headers[:3]
                        })

        return tables_data, qa_report
