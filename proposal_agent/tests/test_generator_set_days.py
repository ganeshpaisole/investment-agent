import subprocess
import shutil
from pathlib import Path
import csv

HERE = Path(__file__).parents[1]
PARSED = HERE / 'samples' / 'rfps' / 'sap_rfp.parsed.json'
OUTDIR = HERE / 'samples' / 'rfps' / 'output_test_days'


def test_generator_respects_set_days():
    if OUTDIR.exists():
        shutil.rmtree(OUTDIR)
    cmd = [
        'python',
        str(HERE / 'tools' / 'generator.py'),
        str(PARSED),
        '--set-days', 'Developer=42',
        '--outdir', str(OUTDIR)
    ]
    subprocess.run(cmd, check=True)

    pricing_csv = OUTDIR / 'pricing.csv'
    assert pricing_csv.exists(), f"pricing.csv not found at {pricing_csv}"

    found = False
    with pricing_csv.open('r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            if row['role'] == 'Developer':
                found = True
                assert int(row['days']) == 42
    assert found, 'Developer row not found in pricing.csv'
