
from pathlib import Path

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
pdf_path = PROJECT_ROOT / "data" / "company" / "culture_ar.pdf"


# Test 1: pypdf
reader = PdfReader(pdf_path)
pypdf_text = "\n".join(
    page.extract_text() or ""
    for page in reader.pages
)

print("=== PYPDF ===")
print(repr(pypdf_text[:1000]))


# Test 2: pdfplumber
with pdfplumber.open(pdf_path) as pdf:
    pdfplumber_text = "\n".join(
        page.extract_text() or ""
        for page in pdf.pages
    )

print("\n=== PDFPLUMBER ===")
print(repr(pdfplumber_text[:1000]))


import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
# Test 3: OCR on the first page
images = convert_from_path(
    pdf_path,
    first_page=1,
    last_page=1,
    dpi=300,
)

ocr_text = pytesseract.image_to_string(
    images[0],
    lang="ara",
    config="--psm 6",
)

print("\n=== TESSERACT OCR (Arabic) ===")
print(repr(ocr_text[:1000]))
