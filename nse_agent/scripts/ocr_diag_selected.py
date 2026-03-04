#!/usr/bin/env python3
"""Run OCR diagnostics only on likely investor-presentation PDFs.

Selection heuristics:
 - file size > 150 KB
 - OR filename contains keywords ('presentation','investor','results')
 - OR pdfplumber extracts > 800 chars

Outputs PNG + _ocr_diag.txt files into data/bse_ocr/ for aggregation.
"""
from pathlib import Path
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr_diag_selected")

PDF_DIR = Path("data/bse_pdfs")
OUT_DIR = Path("data/bse_ocr")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEYWORDS = ("presentation", "investor", "investors", "results", "quarter")

def likely_ir_pdf(path: Path) -> bool:
    try:
        if path.stat().st_size > 150 * 1024:
            return True
    except Exception:
        pass
    name = path.name.lower()
    if any(k in name for k in KEYWORDS):
        return True
    # quick pdfplumber sniff
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            txt = ""
            for p in pdf.pages[:2]:
                t = p.extract_text() or ""
                txt += t
        if len(txt) > 800:
            return True
    except Exception:
        pass
    return False


def render_first_page(path: Path):
    # try pdf2image/poppler
    try:
        from pdf2image import convert_from_path
        imgs = convert_from_path(str(path), dpi=200, first_page=1, last_page=1)
        if imgs:
            return imgs[0]
    except Exception:
        pass
    # try pdfplumber rendering
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            page = pdf.pages[0]
            img = page.to_image(resolution=200).original
            return img
    except Exception:
        pass
    # try pypdfium2
    try:
        from pypdfium2 import PdfDocument
        pdf = PdfDocument(str(path))
        page = pdf.get_page(0)
        bmp = page.render(scale=200)
        pil = bmp.to_pil()
        try:
            bmp.close()
        except Exception:
            pass
        try:
            page.close()
        except Exception:
            pass
        try:
            pdf.close()
        except Exception:
            pass
        return pil
    except Exception:
        pass
    return None


def run_tesseract(img):
    try:
        import pytesseract
        import shutil
        if shutil.which('tesseract') is None:
            tpath = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            import os
            if os.path.exists(tpath):
                pytesseract.pytesseract.tesseract_cmd = tpath
    except Exception as e:
        logger.error(f"pytesseract not available: {e}")
        return None
    try:
        return pytesseract.image_to_string(img, lang='eng')
    except Exception as e:
        logger.debug(f"Tesseract failed: {e}")
        return None


def diag_pdf(path: Path):
    logger.info(f"Checking: {path.name}")
    if not likely_ir_pdf(path):
        logger.info("Skipping (not likely IR PDF)")
        return False
    img = render_first_page(path)
    if not img:
        logger.warning("No renderer available")
        return False
    out_img = OUT_DIR / (path.stem + "_p1.png")
    try:
        img.save(str(out_img))
    except Exception as e:
        logger.debug(f"Save image failed: {e}")
    text = run_tesseract(img)
    if text is None:
        logger.warning("Tesseract produced no text")
        return False
    out_txt = OUT_DIR / (path.stem + "_ocr_diag.txt")
    out_txt.write_text(text or "", encoding='utf-8')
    logger.info(f"Wrote OCR for {path.name}: {len(text or '')} chars")
    return True


def main():
    pdfs = sorted(PDF_DIR.glob('*.pdf'), key=lambda p: p.stat().st_mtime, reverse=True)
    count = 0
    for p in pdfs:
        try:
            ok = diag_pdf(p)
            if ok:
                count += 1
        except KeyboardInterrupt:
            logger.warning('Interrupted')
            break
    logger.info(f'Diagnosed {count} PDFs')


if __name__ == '__main__':
    main()
