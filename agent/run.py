#!/usr/bin/env python3
"""Central runner for nse_agent scripts.

Usage examples:
  python agent\run.py --list
  python agent\run.py batch --limit 10
  python agent\run.py ocr-diag
  python agent\run.py ocr-all
  python agent\run.py check-company
  python agent\run.py search-sitemap
  python agent\run.py parse TICKER

This script runs the existing scripts under `nse_agent/scripts` via the
current Python interpreter so you have a single entry point.
"""
import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / 'nse_agent' / 'scripts'

COMMANDS = {
    'batch': ('run_nifty50_batch.py', 'Run batch for all tickers (use --limit)'),
    'ocr-diag': ('ocr_diag.py', 'Run OCR diagnostic (render first page + OCR)'),
    'ocr-all': ('ocr_pdfs.py', 'OCR all PDFs in data/bse_pdfs/'),
    'check-company': ('check_company_pages.py', 'Probe company IR pages for PDFs'),
    'search-sitemap': ('search_sitemap.py', 'Search company sitemap for PDFs/presentations'),
    'check-tesseract': ('check_tesseract.py', 'Check pytesseract and tesseract binary'),
}

def run_script(script_name, args=None):
    path = SCRIPTS_DIR / script_name
    if not path.exists():
        print(f"Script not found: {path}")
        return 2
    cmd = [sys.executable, str(path)] + (args or [])
    proc = subprocess.run(cmd)
    return proc.returncode

def main():
    ap = argparse.ArgumentParser(prog='agent/run.py')
    ap.add_argument('--list', action='store_true', help='List available commands')

    sub = ap.add_subparsers(dest='cmd')
    sub_batch = sub.add_parser('batch', help='Run batch for Nifty50')
    sub_batch.add_argument('--limit', type=int, default=0, help='Limit to first N tickers')

    sub.add_parser('ocr-diag', help='Run OCR diagnostic')
    sub.add_parser('ocr-all', help='OCR all PDFs')
    sub.add_parser('check-company', help='Check company pages for PDFs')
    sub.add_parser('search-sitemap', help='Search sitemap for PDFs')
    sub.add_parser('check-tesseract', help='Check tesseract/pytesseract')

    sub_parse = sub.add_parser('parse', help='Run parser for a single ticker')
    sub_parse.add_argument('ticker', help='Ticker symbol, e.g. HDFCBANK')

    args, rest = ap.parse_known_args()

    if args.list or not args.cmd:
        print('Available commands:')
        for k, v in COMMANDS.items():
            print(f'  {k:15} - {v[1]}')
        print('  parse           - Run parser for a single ticker')
        print('\nExamples:')
        print('  python agent\\run.py batch --limit 6')
        print('  python agent\\run.py parse HDFCBANK')
        return

    if args.cmd == 'batch':
        cmd = COMMANDS['batch'][0]
        arglist = []
        if args.limit and args.limit > 0:
            arglist = ['--limit', str(args.limit)]
        return run_script(cmd, arglist)

    if args.cmd in COMMANDS:
        return run_script(COMMANDS[args.cmd][0])

    if args.cmd == 'parse':
        # invoke a short runner for single ticker using the parser module
        ticker = args.ticker
        # run via python -c to avoid adding packaging
        code = (
            "from utils.bse_parser import get_bse_data; import json; "
            f"d = get_bse_data('{ticker}'); print(json.dumps(d, indent=2))"
        )
        return subprocess.run([sys.executable, '-c', code]).returncode

if __name__ == '__main__':
    sys.exit(main() or 0)
