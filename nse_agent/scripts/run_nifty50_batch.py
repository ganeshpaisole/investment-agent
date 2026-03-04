"""Batch runner: run BSE parser for all tickers in BSE_CODE_MAP (Nifty50 subset).

Usage:
  python scripts/run_nifty50_batch.py [--limit N]

Saves: data/bse_results/<TICKER>.json and OCRs PDFs to data/bse_ocr/<TICKER>_<pdf>.txt
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
logger = logging.getLogger('run_nifty50_batch')

from utils.bse_parser import BSEFilingParser, BSE_CODE_MAP, PDF_DIR

RESULTS_DIR = Path('data/bse_results')
OCR_DIR = Path('data/bse_ocr')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OCR_DIR.mkdir(parents=True, exist_ok=True)

def process_ticker(ticker: str):
    logger.info(f"=== Processing {ticker} ===")
    parser = BSEFilingParser(ticker)
    data = parser.get_filing_data()
    out_file = RESULTS_DIR / f"{ticker}.json"
    out_file.write_text(json.dumps(data, indent=2), encoding='utf-8')
    logger.info(f"Wrote parsed data to {out_file}")

    # If no metrics, attempt to download candidate PDFs and OCR them
    if data.get('source', '').startswith('BSE: no data'):
        logger.info(f"No metrics for {ticker}; attempting to download & OCR candidate PDFs")
        # collect filings from both categories
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
                if not pdf_path:
                    continue
                # try OCR (uses parser._extract_text_ocr)
                ocr_text = parser._extract_text_ocr(pdf_path)
                if ocr_text and ocr_text.strip():
                    out = OCR_DIR / f"{ticker}_{pdf_path.stem}.txt"
                    out.write_text(ocr_text, encoding='utf-8')
                    logger.info(f"Wrote OCR for {ticker} -> {out}")
                else:
                    logger.info(f"OCR returned no text for {pdf_path.name}")
            except Exception as e:
                logger.debug(f"Error processing candidate {cand}: {e}")
            time.sleep(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='Limit to first N tickers (0 = all)')
    args = ap.parse_args()

    tickers = sorted(list(BSE_CODE_MAP.keys()))
    if args.limit and args.limit > 0:
        tickers = tickers[:args.limit]

    logger.info(f"Running batch for {len(tickers)} tickers (limit={args.limit})")
    for t in tickers:
        try:
            process_ticker(t)
        except Exception as e:
            logger.warning(f"Failed for {t}: {e}")
        time.sleep(0.5)

if __name__ == '__main__':
    main()
