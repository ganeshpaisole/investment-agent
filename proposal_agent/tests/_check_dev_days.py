import csv
from pathlib import Path
p = Path('proposal_agent/samples/rfps/output_test_days/pricing.csv')
if not p.exists():
    print('MISSING')
    raise SystemExit(2)
with p.open() as f:
    r = csv.DictReader(f)
    for row in r:
        if row['role']=='Developer':
            print('Developer days:', row['days'])
            break
