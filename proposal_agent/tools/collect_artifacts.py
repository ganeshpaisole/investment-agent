#!/usr/bin/env python3
"""Create a ZIP of CI proposal artifacts for local inspection.

Usage:
  python proposal_agent/tools/collect_artifacts.py --src proposal_agent/samples/rfps/ci_output --out proposal-artifacts.zip
"""
import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='proposal_agent/samples/rfps/ci_output')
    p.add_argument('--out', default='proposal-artifacts.zip')
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.exists():
        print('Source folder not found:', src)
        raise SystemExit(2)

    # shutil.make_archive expects a base name without suffix
    base = out.with_suffix('')
    print('Creating archive', out)
    shutil.make_archive(str(base), 'zip', root_dir=str(src))
    print('Done')


if __name__ == '__main__':
    main()
