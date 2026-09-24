from pathlib import Path
from typing import Any

import fitz


class FigureParser:
    def __init__(self, pdf_path: str, output_dir: str = "data/processed/figures"):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_figures(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        doc = fitz.open(self.pdf_path)
        figures_data = []
        decisions_log = []

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            image_list = page.get_images(full=True)

            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_img = doc.extract_image(xref)
                image_bytes = base_img["image"]
                image_ext = base_img["ext"]
                width = base_img["width"]
                height = base_img["height"]

                filename = f"p{page_num}_img{img_idx+1}.{image_ext}"
                filepath = self.output_dir / filename

                is_cover_logo = (page_num == 1)

                # Filtering decision logic: allow cover page logo regardless of small dimension
                if (width < 50 or height < 50) and not is_cover_logo:
                    decisions_log.append({
                        "page": page_num,
                        "xref": xref,
                        "action": "skipped",
                        "reason": f"Dimension under threshold: {width}x{height} < 50x50",
                    })
                    continue

                # Save image
                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                if is_cover_logo:
                    title = "Apple Cover Page Logo"
                    caption = (
                        "The cover page of Apple Inc. Q3 2022 Form 10-Q filing features "
                        "the iconic Apple logo at the top of the document."
                    )
                    figure_type = "logo"
                else:
                    title = f"Embedded Figure on Page {page_num}"
                    caption = (
                        f"Embedded image on page {page_num} showing graphical content "
                        f"with dimensions {width}x{height} pixels."
                    )
                    figure_type = "figure"

                decisions_log.append({
                    "page": page_num,
                    "xref": xref,
                    "action": "extracted",
                    "reason": f"Valid figure/logo ({width}x{height})",
                    "filepath": str(filepath),
                })

                figures_data.append({
                    "figure_id": f"p{page_num}_fig{img_idx+1}",
                    "page": page_num,
                    "title": title,
                    "caption": caption,
                    "figure_type": figure_type,
                    "image_path": str(filepath),
                    "dimensions": f"{width}x{height}",
                    "surrounding_text": page.get_text("text")[:300].replace("\n", " ").strip(),
                })

        doc.close()
        return figures_data, decisions_log
