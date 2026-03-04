# NSE Agent — CLI Guide

[![CI](https://github.com/your-repo/your-project/actions/workflows/ci.yml/badge.svg)](https://github.com/your-repo/your-project/actions/workflows/ci.yml)

This repository includes a small CLI for working with BSE/NSE filing parsers.

## Install (editable)

Install the package into your Python environment so the console script is available:

```powershell
C:/Path/To/Python/python.exe -m pip install -e .
```

## Run the CLI

You can run the CLI either via the installed console script `bse-agent` or with
the module runner.

```powershell
C:/Path/To/Python/python.exe -m nse_agent.cli --help
# or, if installed and on PATH
bse-agent --help
```

## Add New CLI Commands

The CLI is implemented in `nse_agent/cli.py` using Click. To add a new command:

1. Open `nse_agent/cli.py`.
2. Add a Click command function decorated with `@cli.command()`.

Example:

```python
@cli.command('scan-dir')
@click.argument('pdf_dir', type=click.Path(exists=True))
def scan_dir(pdf_dir):
    """Scan all PDFs in a directory with the existing patterns."""
    from nse_agent.scripts.check_many_pdfs import run_scan
    run_scan(pdf_dir)
```

Test the command locally without reinstalling:

```powershell
C:/Path/To/Python/python.exe -m nse_agent.cli scan-dir data/bse_pdfs
```

## Packaging / Entry Point

The console script entry is defined in `pyproject.toml` under `[project.scripts]`.

## PowerShell Integration (optional)

Add the Python `Scripts` folder to your user PATH or add a helper function in
your PowerShell profile (`$PROFILE`) to wrap the installed exe.

## Tab-completion

If using Click, generate shell completion and append to your PowerShell profile:

```powershell
python -m nse_agent.cli --show-completion powershell | Out-File -FilePath $PROFILE -Append -Encoding utf8
```
