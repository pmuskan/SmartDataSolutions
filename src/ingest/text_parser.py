import re
from typing import Any

import fitz


def clean_text(text: str) -> str:
    """Rejoin hyphenated/wrapped lines and remove excessive whitespace."""
    # Rejoin hyphenated words across newlines: e.g. "com-\npany" -> "company"
    text = re.sub(r"(\b[a-zA-Z]+)-\n([a-zA-Z]+\b)", r"\1\2", text)
    # Replace lone newlines within paragraphs with spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Normalize multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def is_header_or_footer(line: str, page_num: int, total_pages: int) -> bool:
    """Identify running headers and footers to drop."""
    line_clean = line.strip()
    if not line_clean:
        return True

    # Matching standard Apple 10-Q header/footer patterns
    if re.search(r"Apple Inc\.?\s*\|\s*Q3 2022 Form 10-Q\s*\|\s*\d+", line_clean, re.IGNORECASE):
        return True
    if re.search(r"UNITED STATES\s+SECURITIES AND EXCHANGE COMMISSION", line_clean, re.IGNORECASE):
        if page_num > 1:
            return True
    if line_clean.isdigit() and len(line_clean) <= 3:
        return True
    return False


def detect_heading(line: str) -> tuple[bool, str, int]:
    """Detect heading level and title from text line."""
    line_clean = line.strip()
    
    # Major Part heading
    if re.match(r"^PART\s+[I|V|X]+", line_clean, re.IGNORECASE):
        return True, line_clean, 1

    # Item heading
    if re.match(r"^Item\s+\d+[A-Z]?\.", line_clean, re.IGNORECASE):
        return True, line_clean, 2

    # Note heading
    if re.match(r"^Note\s+\d+\s*[\–\—\-]\s*", line_clean, re.IGNORECASE):
        return True, line_clean, 2

    # Financial Statements heading
    if re.match(r"^CONDENSED CONSOLIDATED STATEMENTS", line_clean, re.IGNORECASE):
        return True, line_clean, 2

    # MD&A Subheadings
    mda_subheadings = [
        "Products and Services Performance",
        "Segment Performance",
        "Gross Margin",
        "Operating Expenses",
        "Capital Resources and Liquidity",
        "Critical Accounting Policies and Estimates",
        "Recent Accounting Pronouncements",
        "Americas", "Europe", "Greater China", "Japan", "Rest of Asia Pacific",
        "Services", "Products"
    ]
    for sub in mda_subheadings:
        if line_clean.lower() == sub.lower():
            return True, line_clean, 3

    return False, "", 0


class TextParser:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def parse_pages(self) -> list[dict[str, Any]]:
        doc = fitz.open(self.pdf_path)
        pages_data = []

        current_part = "Part I"
        current_item = "Item 1. Financial Statements"
        current_sub = ""

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            text = page.get_text("text")
            raw_lines = [l.strip() for l in text.split("\n") if l.strip()]
            
            content_lines = []
            page_headings = []

            for line in raw_lines:
                if is_header_or_footer(line, page_num, len(doc)):
                    continue

                is_head, head_title, level = detect_heading(line)
                if is_head:
                    page_headings.append(head_title)
                    if level == 1:
                        current_part = head_title
                        current_item = ""
                        current_sub = ""
                    elif level == 2:
                        current_item = head_title
                        current_sub = ""
                    elif level == 3:
                        current_sub = head_title

                content_lines.append(line)

            # Reconstruct section path
            path_parts = [p for p in [current_part, current_item, current_sub] if p]
            section_path = " > ".join(path_parts) if path_parts else "Document Root"

            raw_page_text = "\n".join(content_lines)
            cleaned_page_text = clean_text(raw_page_text)

            pages_data.append({
                "page": page_num,
                "section_path": section_path,
                "raw_lines": content_lines,
                "text": cleaned_page_text,
                "headings": page_headings,
            })

        doc.close()
        return pages_data
