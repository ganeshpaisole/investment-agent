import json
from pathlib import Path
from nse_agent import download_and_parse_attach as dap
from nse_agent.utils.bse_parser import BSEFilingParser

out = Path('data/bse_ocr/icici_check_result.json')
pdf = Path('data/bse_pdfs/4248e51b492eda04.pdf')

result = {'pdf': str(pdf), 'text_chars': 0, 'dap_metrics': None, 'parser_metrics': None}

try:
    if not pdf.exists():
        result['error'] = 'PDF not found'
    else:
        text = dap.extract_text(pdf)
        result['text_chars'] = len(text or '')
        result['dap_metrics'] = dap.simple_extract_metrics(text or '')

        parser = BSEFilingParser('ICICIBANK')
        parser_text = parser._extract_text(pdf)
        result['parser_text_chars'] = len(parser_text or '')
        result['parser_metrics'] = parser._extract_banking_metrics(parser_text or '')
except Exception as e:
    result['exception'] = repr(e)

with open(out, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)

print('Wrote', out)
