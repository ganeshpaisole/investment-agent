"""Scan all PDFs in data/bse_pdfs for company-specific bank patterns
and write results to data/bse_ocr/icici_pattern_checks.log
"""
from pathlib import Path
import re
from datetime import datetime

root = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(root))

from utils.bse_parser import BSEFilingParser
from download_and_parse_attach import COMPANY_BANK_PATTERNS, COMPANY_KEYWORDS
import re


def find_number_near_keyword(text: str, keywords: list, window: int = 220):
    """Search for keywords in text and extract the nearest numeric token within a window."""
    num_re = re.compile(r"([0-9]{1,3}(?:[,\.][0-9]{1,3})?(?:\.[0-9]+)?)\s*(%|percent)?", re.IGNORECASE)
    for kw in keywords:
        for m in re.finditer(kw, text, re.IGNORECASE):
            start = max(0, m.start() - window)
            end = min(len(text), m.end() + window)
            snippet = text[start:end]
            # prefer percentage-like matches
            pct = re.search(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%", snippet)
            if pct:
                return pct.group(1)
            m2 = num_re.search(snippet)
            if m2:
                return m2.group(1)
    return None


def _parse_number_token(tok: str):
    """Normalize numeric token like '5,000' or '10.3%' to float, or None."""
    if not tok:
        return None
    s = tok.strip()
    s = s.replace('%', '').replace('percent', '')
    s = s.replace(',', '')
    s = re.sub(r"[^0-9\.\-]", "", s)
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _is_sane(metric: str, val_str: str) -> bool:
    """Return True if parsed numeric value lies within sane thresholds for metric."""
    v = _parse_number_token(val_str)
    if v is None:
        return False
    thresholds = {
        'casa': (0.0, 100.0),
        'car':  (0.0, 100.0),
        'nim':  ( -5.0, 20.0),
    }
    lo, hi = thresholds.get(metric, (None, None))
    if lo is None:
        return True
    return lo <= v <= hi


OUT = Path('data/bse_ocr/icici_pattern_checks.log')
OUT.parent.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    with open(OUT, 'a', encoding='utf-8') as f:
        f.write(msg + "\n")

def main():
    log('=== ICICI pattern scan started: ' + datetime.now().isoformat())
    pdf_dir = Path('data/bse_pdfs')
    files = sorted(pdf_dir.glob('*.pdf'))
    if not files:
        log('No PDFs found in data/bse_pdfs')
        return

    for pdf in files:
        log('---')
        log(str(pdf.name))
        parser = BSEFilingParser('ICICIBANK')
        try:
            text = parser._extract_text(pdf) or ''
            log(f'Extracted chars: {len(text)}')
        except Exception as e:
            log(f'Extraction error: {e}')
            continue

        patterns = COMPANY_BANK_PATTERNS.get('ICICIBANK', {})
        if not patterns:
            log('No company patterns for ICICIBANK')
            continue

        for metric, pats in patterns.items():
            matched = False
            for pat in pats:
                try:
                    m = re.search(pat, text, re.IGNORECASE)
                except re.error:
                    m = None
                if m:
                    val = m.group(1).strip()[:120]
                    log(f'MATCH {metric}: {val}')
                    matched = True
                    break

            if not matched:
                # Nearby-number fallback using keywords
                kw_map = COMPANY_KEYWORDS.get('ICICIBANK', {})
                keywords = kw_map.get(metric, [])
                if keywords:
                    v = find_number_near_keyword(text, keywords)
                    if v:
                        # sanity-check fallback
                        if _is_sane(metric, v):
                            log(f'FALLBACK {metric}: {v} (near keywords {keywords[:2]})')
                        else:
                            log(f'DISCARDED FALLBACK {metric}: {v} (out of range)')
                        continue

                log(f'NO MATCH for {metric}')

    log('=== Scan complete: ' + datetime.now().isoformat())


if __name__ == '__main__':
    main()
