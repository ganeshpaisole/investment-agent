import requests
import re
from pathlib import Path
from loguru import logger

ATTACH_ID = '77dd1f75-6b85-4aac-8334-7d7c7b7eb410.pdf'
BASES = ['AttachLive', 'AttachHis', 'AttachDiv']
PDF_DIR = Path('data/bse_pdfs')
PDF_DIR.mkdir(parents=True, exist_ok=True)


def download_url(url: str):
    try:
        headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.bseindia.com/'}
        r = requests.get(url, headers=headers, timeout=30, stream=True)
        if r.status_code != 200:
            logger.warning(f'HTTP {r.status_code} for {url}')
            return None
        content = r.content
        if not content[:4] == b'%PDF':
            logger.warning('Content not PDF')
            return None
        fname = (hashlib_name(url) + '.pdf')
        local = PDF_DIR / fname
        with open(local, 'wb') as f:
            f.write(content)
        logger.info(f'Saved {local} ({local.stat().st_size//1024} KB)')
        return local
    except Exception as e:
        logger.warning(f'Download failed: {e}')
        return None


def hashlib_name(s: str) -> str:
    import hashlib
    return hashlib.md5(s.encode()).hexdigest()[:16]


def extract_text(pdf_path: Path) -> str:
    text = ''
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:40]:
                pt = page.extract_text() or ''
                text += pt + '\n'
        if text.strip():
            logger.info(f'pdfplumber extracted {len(text)} chars')
            return text
    except Exception as e:
        logger.debug(f'pdfplumber failed: {e}')

    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(str(pdf_path), maxpages=40) or ''
        if text.strip():
            logger.info(f'pdfminer extracted {len(text)} chars')
            return text
    except Exception:
        pass

    # OCR fallback
    try:
        import pytesseract
    except Exception:
        logger.warning('pytesseract not installed')
        return ''

    images = []
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(str(pdf_path), dpi=200, first_page=1, last_page=5)
    except Exception:
        try:
            from pypdfium2 import PdfDocument
            pdf = PdfDocument(str(pdf_path))
            for i in range(min(5, pdf.page_count)):
                page = pdf.get_page(i)
                bmp = page.render(scale=200)
                pil = bmp.to_pil()
                images.append(pil)
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
        except Exception as e:
            logger.debug(f'pdf rendering failed: {e}')

    ocr_text = ''
    for img in images:
        try:
            import pytesseract
            ocr_text += pytesseract.image_to_string(img, lang='eng') + '\n'
        except Exception as e:
            logger.debug(f'OCR page failed: {e}')

    if ocr_text.strip():
        logger.info(f'OCR extracted {len(ocr_text)} chars')
    return ocr_text


def find_pct(patterns, text):
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            raw = m.group(1).replace(',', '').replace('%', '').strip()
            raw = re.sub(r"^[^0-9\-\.]+|[^0-9\.]+$", "", raw)
            try:
                v = float(raw)
                if 0 <= v < 200:
                    return round(v, 2)
            except:
                pass
    return None


COMPANY_BANK_PATTERNS = {
    'ICICIBANK': {
        # Expanded CASA patterns to match variations seen in investor presentations
        'casa': [
            r"CASA\s+Ratio\s*[:\-]?\s*([\d.]+)\s*%?",
            r"CASA\s*[:\-]?\s*([\d.]+)\s*%?",
            r"Current\s+&\s+Savings\s+[:\-]?\s*([\d.]+)\s*%?",
            r"Current\s+and\s+Savings\s*\(?Deposits\)?\s*\(CASA\)?\s*[:\-]?\s*([\d.]+)\s*%?",
            r"CASA\s+deposits?\s+.*?([\d.]+)\s*%?",
            r"CASA\s*\(as\s*a\s*%\)\s*[:\-]?\s*([\d.]+)\s*%?",
            r"CASA\s+as\s+%\s*[:\-]?\s*([\d.]+)\s*%?",
            r"(?:^|\s)CASA\s+([\d.]+)\s*%?",
        ],
        'car': [r"Capital\s+Adequacy\s+Ratio\s*[:\-]?\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%"],
        'nim': [r"NIM\s*[:\-]?\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%"],
    }
}


# Keywords for nearby-number fallback (used when strict regexes fail)
COMPANY_KEYWORDS = {
    'ICICIBANK': {
        'casa': [r'CASA', r'Current\s+and\s+Savings', r'Current\s*&\s*Savings', r'Current\s+Savings'],
        'car':  [r'Capital\s+Adequacy', r'CRAR', r'Capital\s+Ratio', r'CAR'],
        'nim':  [r'NIM', r'Net\s+Interest\s+Margin'],
    }
}


def _augment_patterns():
    """Programmatically extend COMPANY_BANK_PATTERNS with no-percent
    variants and simple keyword patterns to improve recall for OCR'd text.
    This mutates COMPANY_BANK_PATTERNS in-place.
    """
    for comp, metrics in COMPANY_BANK_PATTERNS.items():
        for metric, pats in list(metrics.items()):
            new = []
            for p in pats:
                # if pattern contains explicit percent sign, add a variant without it
                if '%' in p and p.replace('%', '') not in pats and p.replace('%', '') not in new:
                    new.append(p.replace('%', '').replace('\s*', ' '))
                # add a relaxed number-only capture if not present
                relaxed = re.sub(r"\\\s*%\\\)?", "", p)
                if relaxed not in pats and relaxed not in new:
                    new.append(relaxed)
            # append generated patterns
            for q in new:
                if q not in metrics[metric]:
                    metrics[metric].append(q)


# Run augmentation at import time
try:
    _augment_patterns()
except Exception:
    pass


def simple_extract_metrics(text: str) -> dict:
    res = {}
    comp = COMPANY_BANK_PATTERNS.get('ICICIBANK', {})
    for metric, patterns in comp.items():
        v = find_pct(patterns, text)
        if v is not None:
            res[metric] = v
    return res


def main():
    import hashlib
    for base in BASES:
        url = f'https://www.bseindia.com/xml-data/corpfiling/{base}/{ATTACH_ID}'
        logger.info(f'Trying {url}')
        try:
            headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.bseindia.com/'}
            r = requests.get(url, headers=headers, timeout=15)
        except Exception as e:
            logger.warning(f'HTTP request failed: {e}')
            continue
        is_pdf = (r.status_code == 200 and r.content[:4] == b'%PDF')
        logger.info(f'{base}: status={r.status_code} size={len(r.content)//1024}KB pdf={is_pdf}')
        if not is_pdf:
            continue
        fname = hashlib.md5(url.encode()).hexdigest()[:16] + '.pdf'
        local = PDF_DIR / fname
        with open(local, 'wb') as f:
            f.write(r.content)
        logger.info(f'Saved {local}')
        text = extract_text(local)
        logger.info(f'Extracted {len(text)} chars')
        metrics = simple_extract_metrics(text or '')
        logger.info(f'Metrics: {metrics}')
        break


if __name__ == '__main__':
    main()
