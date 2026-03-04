"""Test company-specific bank patterns against PDFs in data/bse_pdfs.

Runs text extraction on each PDF and reports any pattern matches for
tickers defined in `COMPANY_BANK_PATTERNS`.
"""
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from utils.bse_parser import BSEFilingParser, COMPANY_BANK_PATTERNS

PDF_DIR = Path('data/bse_pdfs')
pdfs = list(PDF_DIR.glob('*.pdf'))
if not pdfs:
    print('No PDFs found in', PDF_DIR)
    sys.exit(1)

for pdf in pdfs:
    print('\n===', pdf.name)
    # Use a generic parser instance (ticker doesn't matter for extraction)
    p = BSEFilingParser('HDFCBANK')
    text = p._extract_text(pdf) or ''
    text_l = text.lower()
    for ticker, patterns in COMPANY_BANK_PATTERNS.items():
        found_any = False
        for metric, pats in patterns.items():
            for pat in pats:
                import re
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    print(f'  {ticker}: matched {metric} ->', m.group(1)[:60])
                    found_any = True
                    break
            # continue to next metric
        if found_any:
            # separate tickers
            print('  ---')
