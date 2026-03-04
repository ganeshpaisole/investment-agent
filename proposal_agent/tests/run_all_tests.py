#!/usr/bin/env python3
"""Run generator for all sample parsed RFPs and verify outputs exist."""
import subprocess
import shutil
from pathlib import Path

HERE = Path(__file__).parents[1]
SAMPLES = [
    HERE / 'samples' / 'rfps' / 'sap_rfp.parsed.json',
    HERE / 'samples' / 'rfps' / 'dynamics_rfp.parsed.json',
    HERE / 'samples' / 'rfps' / 'mes_rfp.parsed.json',
]


def main():
    out_base = HERE / 'samples' / 'rfps' / 'output_all_manual'
    if out_base.exists():
        shutil.rmtree(out_base)
    all_ok = True
    for s in SAMPLES:
        outdir = out_base / s.stem
        cmd = [
            'python',
            str(HERE / 'tools' / 'generator.py'),
            str(s),
            '--outdir', str(outdir)
        ]
        print('Running:', ' '.join(cmd))
        subprocess.run(cmd, check=True)
        missing = []
        for fn in ('proposal.docx','exec_summary.pptx','pricing.csv','gantt.csv'):
            if not (outdir / fn).exists():
                missing.append(fn)
        if missing:
            print(f"FAIL for {s}: missing {missing}")
            all_ok = False
        else:
            print(f"OK: {s} -> {outdir}")
    if not all_ok:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
