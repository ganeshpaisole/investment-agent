#!/usr/bin/env python3
"""Simple OCR helper: render PDFs to images and run Tesseract OCR.

Writes outputs to `data/bse_ocr/<pdfname>.txt` and prints excerpts.
Tries `pdf2image`+poppler first, falls back to `pdfplumber` page rendering.
"""
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr_pdfs")

PDF_DIR = Path("data/bse_pdfs")
OUT_DIR = Path("data/bse_ocr")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def try_pdf2image(pdf_path, max_pages=20):
    try:
        from pdf2image import convert_from_path
    except Exception as e:
        logger.debug(f"pdf2image not available: {e}")
        return None

    # try to locate poppler in common Chocolatey locations
    poppler_candidates = [
        r"C:\ProgramData\chocolatey\lib\poppler\tools\poppler-26.02.0\bin",
        r"C:\ProgramData\chocolatey\lib\poppler.portable\tools\poppler\bin",
        r"C:\ProgramData\chocolatey\bin",
    ]
    poppler_path = None
    for p in poppler_candidates:
        if Path(p).exists():
            poppler_path = p
            break

    try:
        if poppler_path:
            imgs = convert_from_path(str(pdf_path), dpi=200, first_page=1, last_page=max_pages, poppler_path=poppler_path)
        else:
            imgs = convert_from_path(str(pdf_path), dpi=200, first_page=1, last_page=max_pages)
        return imgs
    except Exception as e:
        logger.debug(f"pdf2image.convert_from_path failed: {e}")
        return None

def try_pdfplumber_render(pdf_path, max_pages=20):
    try:
        import pdfplumber
    except Exception as e:
        logger.debug(f"pdfplumber not available for rendering: {e}")
        return None

    imgs = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                try:
                    img = page.to_image(resolution=200).original
                    imgs.append(img)
                except Exception as e:
                    logger.debug(f"pdfplumber page render failed on page {i}: {e}")
        return imgs
    except Exception as e:
        logger.debug(f"pdfplumber open failed: {e}")
        return None

def ocr_images(images):
    try:
        import pytesseract
        import shutil
        # If tesseract binary isn't on PATH, try common install location
        if shutil.which('tesseract') is None:
            tpath = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            try:
                import os
                if os.path.exists(tpath):
                    pytesseract.pytesseract.tesseract_cmd = tpath
            except Exception:
                pass
    except Exception as e:
        logger.error(f"pytesseract not installed: {e}")
        return None

    text = []
    for i, img in enumerate(images):
        try:
            page_text = pytesseract.image_to_string(img, lang='eng')
            text.append(page_text)
        except Exception as e:
            logger.debug(f"pytesseract failed on page {i}: {e}")
    return "\n".join(text)

def ocr_pdf(pdf_path: Path):
    logger.info(f"OCRing {pdf_path.name}")

    imgs = try_pdf2image(pdf_path)
    method = "pdf2image"
    if not imgs:
        imgs = try_pdfplumber_render(pdf_path)
        method = "pdfplumber" if imgs else None

    if not imgs:
        logger.error("No renderer available (poppler/pdf2image or pdfplumber). Cannot OCR.")
        return None

    text = ocr_images(imgs)
    if text is None:
        logger.error("OCR failed (pytesseract missing or runtime error)")
        return None

    out_file = OUT_DIR / (pdf_path.stem + ".txt")
    out_file.write_text(text, encoding="utf-8")
    logger.info(f"Wrote OCR text to {out_file}")
    return text

def main():
    pdfs = list(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        logger.error("No PDFs found in data/bse_pdfs/")
        sys.exit(1)

    # Prefer recently downloaded ones by mtime
    pdfs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for pdf in pdfs[:5]:
        text = ocr_pdf(pdf)
        if text:
            excerpt = text[:2000].strip().replace('\n', ' ')
            print('\n' + '='*80)
            print(f"File: {pdf.name} — excerpt (first 2000 chars):\n")
            print(excerpt)
            print('\n' + '='*80 + '\n')

if __name__ == '__main__':
    main()
