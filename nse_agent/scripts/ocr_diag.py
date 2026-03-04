#!/usr/bin/env python3
"""OCR diagnostic: render first page, save image, run Tesseract, report results."""
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr_diag")

PDF_DIR = Path("data/bse_pdfs")
OUT_DIR = Path("data/bse_ocr")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def render_first_page_pdf2image(pdf_path):
    try:
        from pdf2image import convert_from_path
    except Exception as e:
        logger.debug(f"pdf2image not available: {e}")
        return None
    # guess poppler path
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
            imgs = convert_from_path(str(pdf_path), dpi=200, first_page=1, last_page=1, poppler_path=poppler_path)
        else:
            imgs = convert_from_path(str(pdf_path), dpi=200, first_page=1, last_page=1)
        return imgs[0] if imgs else None
    except Exception as e:
        logger.debug(f"pdf2image convert error: {e}")
        return None

def render_first_page_pdfplumber(pdf_path):
    try:
        import pdfplumber
    except Exception as e:
        logger.debug(f"pdfplumber not available: {e}")
        return None
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]
            img = page.to_image(resolution=200).original
            return img
    except Exception as e:
        logger.debug(f"pdfplumber render error: {e}")
        return None

def render_first_page_pdfium(pdf_path):
    try:
        from pypdfium2 import PdfDocument
    except Exception as e:
        logger.debug(f"pypdfium2 not available: {e}")
        return None
    try:
        pdf = PdfDocument(str(pdf_path))
        try:
            page = pdf.get_page(0)
        except Exception:
            pdf.close()
            return None
        try:
            bmp = page.render(scale=200)
            pil = bmp.to_pil()
            try:
                bmp.close()
            except Exception:
                pass
            return pil
        finally:
            try:
                page.close()
            except Exception:
                pass
            try:
                pdf.close()
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"pypdfium2 render error: {e}")
        return None

def run_tesseract_on_image(img):
    try:
        import pytesseract
        import shutil
        # If tesseract binary isn't on PATH for this process, try the common install path
        if shutil.which('tesseract') is None:
            tpath = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            try:
                import os
                if os.path.exists(tpath):
                    pytesseract.pytesseract.tesseract_cmd = tpath
            except Exception:
                pass
    except Exception as e:
        logger.error(f"pytesseract not available: {e}")
        return None
    try:
        text = pytesseract.image_to_string(img, lang='eng')
        return text
    except Exception as e:
        logger.debug(f"pytesseract failed: {e}")
        return None

def save_image(img, path: Path):
    try:
        img.save(str(path))
        return True
    except Exception as e:
        logger.debug(f"image save failed: {e}")
        return False

def diag_pdf(pdf_path: Path):
    logger.info(f"Diagnosing {pdf_path.name}")
    img = render_first_page_pdf2image(pdf_path)
    method = 'pdf2image'
    if not img:
        img = render_first_page_pdfplumber(pdf_path)
        method = 'pdfplumber' if img else None

    if not img:
        logger.error("No renderer available to produce an image of the page.")
        return

    out_img = OUT_DIR / (pdf_path.stem + "_p1.png")
    saved = save_image(img, out_img)
    logger.info(f"Rendered first page using {method}; saved image: {out_img} (saved={saved})")

    text = run_tesseract_on_image(img)
    if text is None:
        logger.error("Tesseract OCR returned None or is not available")
        return

    out_txt = OUT_DIR / (pdf_path.stem + "_ocr_diag.txt")
    out_txt.write_text(text or "", encoding='utf-8')
    logger.info(f"Tesseract produced {len(text or '')} chars; wrote to {out_txt}")
    excerpt = (text or "").strip().replace('\n', ' ')[:1000]
    print('\n' + '-'*72)
    print(f"File: {pdf_path.name} — renderer={method} — OCR chars={len(text or '')}")
    print(f"Excerpt: {excerpt}\n")

def main():
    pdfs = list(PDF_DIR.glob('*.pdf'))
    if not pdfs:
        logger.error('No PDFs found in data/bse_pdfs')
        sys.exit(1)
    for pdf in sorted(pdfs, key=lambda p: p.stat().st_mtime, reverse=True):
        diag_pdf(pdf)

if __name__ == '__main__':
    main()
