"""Batch runner (no OCR): download candidate PDFs for Nifty50 tickers.

Usage:
  python scripts/run_nifty50_batch_no_ocr.py --limit 6

Saves PDFs into `data/bse_pdfs/` for later OCR processing.
"""
import time
import json
import argparse
from pathlib import Path
import logging
import sys

# Ensure `nse_agent` package is on sys.path so `from utils.bse_parser` works
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('run_nifty50_batch_no_ocr')

from utils.bse_parser import BSEFilingParser, BSE_CODE_MAP, PDF_DIR

RESULTS_DIR = Path('data/bse_results')
PDF_DIR = Path('data/bse_pdfs')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)


def process_ticker(ticker: str):
    logger.info(f"=== Processing {ticker} (no OCR) ===")
    parser = BSEFilingParser(ticker)
    data = parser.get_filing_data()
    out_file = RESULTS_DIR / f"{ticker}.json"
    out_file.write_text(json.dumps(data, indent=2), encoding='utf-8')
    logger.info(f"Wrote parsed data to {out_file}")

    # collect candidate PDFs (try categories)
    candidates = []
    for cat in ('Investor Presentation', 'Result'):
        try:
            filings = parser._get_bse_filings(category=cat, max_results=6)
        except Exception:
            filings = []
        for f in filings:
            url = f.get('url')
            if not url:
                continue
            # find candidates from the page
            cands = parser._find_pdf_in_page(url) or []
            if not cands:
                cands = parser._nse_company_fallback_candidates() or []
            if not cands:
                cands = parser._company_ir_candidates() or []
            for c in cands:
                if c not in candidates:
                    candidates.append(c)

    logger.info(f"Found {len(candidates)} unique candidate URLs for {ticker}")
    for cand in candidates:
        try:
            pdf_path = parser._download_pdf(cand)
            if pdf_path:
                logger.info(f"Downloaded {pdf_path.name} for {ticker}")
        except Exception as e:
            logger.debug(f"Error downloading candidate {cand}: {e}")
        time.sleep(0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='Limit to first N tickers (0 = all)')
    args = ap.parse_args()

    tickers = sorted(list(BSE_CODE_MAP.keys()))
    if args.limit and args.limit > 0:
        tickers = tickers[:args.limit]

    logger.info(f"Running no-OCR batch for {len(tickers)} tickers (limit={args.limit})")
    for t in tickers:
        try:
            process_ticker(t)
        except Exception as e:
            logger.warning(f"Failed for {t}: {e}")
        time.sleep(0.5)


if __name__ == '__main__':
    main()
