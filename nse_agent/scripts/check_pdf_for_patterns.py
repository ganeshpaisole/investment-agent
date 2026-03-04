"""Check a single PDF against company-specific bank patterns.

Usage: python scripts/check_pdf_for_patterns.py data/bse_pdfs/<file>.pdf ICICIBANK
"""
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from utils.bse_parser import BSEFilingParser
from download_and_parse_attach import COMPANY_BANK_PATTERNS

if len(sys.argv) < 3:
    print('Usage: check_pdf_for_patterns.py <pdf> <TICKER>')
    sys.exit(1)

pdf = Path(sys.argv[1])
ticker = sys.argv[2].upper()
if not pdf.exists():
    print('PDF not found:', pdf)
    sys.exit(1)

parser = BSEFilingParser(ticker)
text = parser._extract_text(pdf) or ''
print('Extracted chars:', len(text))
patterns = COMPANY_BANK_PATTERNS.get(ticker, {})
if not patterns:
    print('No company patterns for', ticker)
    sys.exit(0)

import re
for metric, pats in patterns.items():
    for pat in pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            print(f'MATCH {metric}:', m.group(1)[:120])
            break
    else:
        print(f'NO MATCH for {metric}')
