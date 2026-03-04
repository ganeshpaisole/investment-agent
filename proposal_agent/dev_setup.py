#!/usr/bin/env python3
"""Dev setup: create virtualenv at .venv, install dev requirements, and optionally run tests."""
import subprocess
import sys
from pathlib import Path
import venv


def main(run_tests=False):
    root = Path(__file__).parent
    venv_dir = root / '.venv'
    if not venv_dir.exists():
        print('Creating venv at', venv_dir)
        venv.create(venv_dir, with_pip=True)
    else:
        print('Using existing venv at', venv_dir)

    pip_exe = venv_dir / 'Scripts' / 'pip.exe'
    py_exe = venv_dir / 'Scripts' / 'python.exe'

    if not pip_exe.exists():
        print('pip not found in venv, aborting')
        sys.exit(2)

    req = root / 'requirements-dev.txt'
    if req.exists():
        print('Installing dev requirements...')
        subprocess.check_call([str(pip_exe), 'install', '-r', str(req)])
    else:
        print('No requirements-dev.txt found at', req)

    if run_tests:
        print('Running tests...')
        subprocess.check_call([str(py_exe), '-m', 'pytest', '-q', 'proposal_agent/tests'])

    print('\nDev setup complete. To activate venv (PowerShell):')
    print(f"  {venv_dir / 'Scripts' / 'Activate.ps1'}")


if __name__ == '__main__':
    run_tests = False
    if len(sys.argv) > 1 and sys.argv[1] in ('-t', '--test', 'test'):
        run_tests = True
    main(run_tests)
