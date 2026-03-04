import subprocess
import shutil
from pathlib import Path

HERE = Path(__file__).parents[1]
SAMPLES = [
    HERE / 'samples' / 'rfps' / 'sap_rfp.parsed.json',
    HERE / 'samples' / 'rfps' / 'dynamics_rfp.parsed.json',
    HERE / 'samples' / 'rfps' / 'mes_rfp.parsed.json',
]


def test_generate_all_samples():
    out_base = HERE / 'samples' / 'rfps' / 'output_all'
    if out_base.exists():
        shutil.rmtree(out_base)
    for s in SAMPLES:
        outdir = out_base / s.stem
        cmd = [
            'python',
            str(HERE / 'tools' / 'generator.py'),
            str(s),
            '--outdir', str(outdir)
        ]
        subprocess.run(cmd, check=True)
        # verify outputs
        assert (outdir / 'proposal.docx').exists()
        assert (outdir / 'exec_summary.pptx').exists()
        assert (outdir / 'pricing.csv').exists()
        assert (outdir / 'gantt.csv').exists()
