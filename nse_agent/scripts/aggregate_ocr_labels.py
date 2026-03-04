"""Aggregate OCR diagnostic files and build label-frequency statistics.

Searches `data/bse_ocr/*_ocr_diag.txt` for metric keywords and collects
surrounding context to help tune regexes.

Writes results to `data/bse_ocr/label_stats.json` and prints a short summary.
"""
import re
import json
from pathlib import Path
from collections import Counter, defaultdict

OCR_DIR = Path('data/bse_ocr')
OUT_FILE = OCR_DIR / 'label_stats.json'

KEYWORDS = {
    'casa': ['casa', 'current & savings', 'current and savings', 'current/savings'],
    'car': ['capital adequacy', 'car', 'crar', 'capital ratio'],
    'nim': ['net interest margin', 'nim', 'net interest'],
    'gnpa': ['gross npa', 'gnpa', 'gross non'],
    'nnpa': ['net npa', 'nnpa', 'net non'],
    'credit_cost': ['credit cost', 'credit costs', 'provision coverage'],
    'roa': ['return on assets', 'roa'],
    'roe': ['return on equity', 'roe'],
}

def contexts_for_kw(text, kw, window=60):
    results = []
    for m in re.finditer(re.escape(kw), text, re.IGNORECASE):
        start = max(0, m.start() - window)
        end = min(len(text), m.end() + window)
        results.append(text[start:end].strip())
    return results


def main():
    stats = {k: Counter() for k in KEYWORDS}
    samples = defaultdict(list)

    files = sorted(OCR_DIR.glob('*_ocr_diag.txt'))
    for f in files:
        txt = f.read_text(encoding='utf-8', errors='ignore')
        for metric, kws in KEYWORDS.items():
            for kw in kws:
                ctxs = contexts_for_kw(txt, kw)
                for c in ctxs:
                    # extract short numeric tokens inside context
                    nums = re.findall(r"[\d,]+(?:\.\d+)?%?", c)
                    for n in nums:
                        stats[metric][n] += 1
                    # also keep raw context samples (truncate)
                    if len(samples[metric]) < 30 and c not in samples[metric]:
                        samples[metric].append(c[:260])

    out = {
        'counts': {k: dict(stats[k].most_common(30)) for k in stats},
        'samples': {k: samples[k] for k in samples},
        'files_scanned': [str(p.name) for p in files],
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2), encoding='utf-8')

    print(f"Scanned {len(files)} files; wrote stats to {OUT_FILE}")
    for k in KEYWORDS:
        top = stats[k].most_common(5)
        print(f"{k}: {top}")


if __name__ == '__main__':
    main()
