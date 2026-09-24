import os
from pathlib import Path

import fitz
import matplotlib.pyplot as plt


def generate_synthetic_figure_pdf(output_path: str = "tests/fixtures/sample_chart.pdf") -> str:
    """Generate a synthetic PDF containing a bar chart image and caption."""
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create a synthetic bar chart image using matplotlib
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ['Q1 2022', 'Q2 2022', 'Q3 2022', 'Q4 2022']
    values = [123.9, 97.3, 83.0, 90.1]
    ax.bar(categories, values, color='#1f77b4')
    ax.set_title('Quarterly Revenue (in Billions)')
    ax.set_ylabel('USD ($B)')
    
    img_buf = os.path.join(out_dir, "temp_chart.png")
    plt.savefig(img_buf, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. Build PDF using PyMuPDF (fitz)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    # Insert heading
    page.insert_text((50, 50), "Synthetic Financial Report - Figure Test", fontsize=16)
    
    # Insert chart image
    rect = fitz.Rect(50, 80, 450, 380)
    page.insert_image(rect, filename=img_buf)

    # Insert caption
    page.insert_text(
        (50, 400),
        "Figure 1: Quarterly Revenue Trend for Fiscal Year 2022 showing Q3 revenue of $83.0 Billion.",
        fontsize=11
    )

    doc.save(output_path)
    doc.close()

    if os.path.exists(img_buf):
        os.remove(img_buf)

    return output_path

if __name__ == "__main__":
    path = generate_synthetic_figure_pdf()
    print(f"Generated synthetic figure PDF at: {path}")
