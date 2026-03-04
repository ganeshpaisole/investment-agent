"""Probe common bank IR pages for PDF links and download them for OCR.

Usage:
  python scripts/check_bank_ir_pages.py

Downloads found PDFs into `data/bse_pdfs/` for later OCR diagnostics.
"""
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from utils.bse_parser import BSEFilingParser

BANK_IRS = {
    'HDFCBANK': 'https://www.hdfcbank.com/investor-relations',
    'ICICIBANK': 'https://www.icicibank.com/about-us/investor-relations',
    'AXISBANK': 'https://www.axisbank.com/investors',
    'KOTAKBANK': 'https://www.kotak.com/en/investors.html',
    'SBIN': 'https://sbi.co.in/web/investor-relations',
}

def main():
    for ticker, ir in BANK_IRS.items():
        print(f"\n=== Checking {ticker} -> {ir}")
        parser = BSEFilingParser(ticker)
        try:
            cands = parser._find_pdf_in_page(ir) or []
        except Exception as e:
            print('  Error finding PDFs:', e)
            continue
        if not cands:
            print('  No PDF candidates found')
            continue
        print(f'  Found {len(cands)} PDF candidates')
        for cand in cands[:8]:
            print('   ', cand)
            try:
                p = parser._download_pdf(cand)
                if p:
                    print('    -> Downloaded:', p.name)
            except Exception as e:
                print('    -> Download failed:', e)

if __name__ == '__main__':
    main()
