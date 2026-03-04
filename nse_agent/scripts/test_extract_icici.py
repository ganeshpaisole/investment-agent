from pathlib import Path
from utils.bse_parser import BSEFilingParser

def main():
    p = BSEFilingParser('ICICIBANK')
    pdf = Path('data/bse_pdfs/4248e51b492eda04.pdf')
    text = p._extract_text(pdf)
    print('Extracted chars:', len(text))
    metrics = p._extract_banking_metrics(text)
    print('Metrics:', metrics)

if __name__ == '__main__':
    main()
