import sys
import click
from pathlib import Path

# Import internal modules lazily to keep CLI responsive
def _get_parser(ticker: str):
    from nse_agent.utils.bse_parser import BSEFilingParser
    return BSEFilingParser(ticker)


@click.group()
def cli():
    """nse_agent command-line utilities."""


@cli.command('check-pdf')
@click.argument('pdf_path', type=click.Path(exists=True))
@click.argument('ticker')
def check_pdf(pdf_path, ticker):
    """Check a local PDF for company banking patterns."""
    # Reuse existing script logic
    p = _get_parser(ticker.upper())
    text = p._extract_text(Path(pdf_path)) or ''
    from nse_agent.download_and_parse_attach import COMPANY_BANK_PATTERNS
    import re
    patterns = COMPANY_BANK_PATTERNS.get(ticker.upper(), {})
    if not patterns:
        click.echo(f'No company patterns for {ticker.upper()}')
        sys.exit(0)

    for metric, pats in patterns.items():
        for pat in pats:
            try:
                m = re.search(pat, text, re.IGNORECASE)
            except re.error:
                m = None
            if m:
                click.echo(f'MATCH {metric}: {m.group(1)[:120]}')
                break
        else:
            click.echo(f'NO MATCH for {metric}')


@cli.command('bse-parse')
@click.argument('ticker')
def bse_parse(ticker):
    """Run the BSE filing parser for a ticker and print extracted metrics."""
    p = _get_parser(ticker.upper())
    data = p.get_filing_data(sector_type='BANKING')
    for k, v in data.items():
        click.echo(f'{k}: {v}')


def main():
    cli()


if __name__ == '__main__':
    main()
