"""Check specific company IR pages for PDF links using BSEFilingParser._find_pdf_in_page"""
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from utils.bse_parser import BSEFilingParser

parser = BSEFilingParser('HDFCBANK')

urls = [
    'https://www.hdfcbank.com/',
    'https://www.hdfcbank.com/personal/about-us/investor-relations/events-and-presentations',
    'https://www.hdfcbank.com/personal/about-us/investor-relations/events-and-presentations/analyst-meet',
    'https://www.hdfcbank.com/about-us',
    'https://www.hdfcbank.com/about-us/investor-relations',
    'https://www.hdfcbank.com/investor-relations',
    'https://www.hdfcbank.com/investors',
    'https://www.hdfcbank.com/~/media/feature/our-business/investor-relations',
    'https://www.hdfcbank.com/~/media/feature/our-business/financial-highlights',
]

for u in urls:
    print('\nChecking:', u)
    try:
        cands = parser._find_pdf_in_page(u)
        if cands:
            for c in cands:
                print('  PDF:', c)
        else:
            print('  No PDF candidates found')
    except Exception as e:
        print('  Error:', e)
