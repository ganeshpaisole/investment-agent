#!/usr/bin/env python3
"""Simple CLI to inspect KB clauses and generate files by injecting clauses into templates.

Usage examples:
  python proposal_agent/cli.py list-clauses
  python proposal_agent/cli.py dump-clause iso
  python proposal_agent/cli.py generate --template proposal_agent/kb/templates/sap_s4_template.md --out out.md
"""
import argparse
import re
from pathlib import Path
from proposal_agent.kb.loader import load_clauses


def cmd_list_clauses(args):
    clauses = load_clauses(args.clauses or Path('proposal_agent/kb/clauses'))
    for k in sorted(clauses.keys()):
        print(k)


def cmd_dump_clause(args):
    clauses = load_clauses(args.clauses or Path('proposal_agent/kb/clauses'))
    text = clauses.get(args.name)
    if text is None:
        print('Clause not found:', args.name)
        return 2
    print(text)
    return 0


def cmd_generate(args):
    clauses = load_clauses(args.clauses or Path('proposal_agent/kb/clauses'))
    tpl = Path(args.template).read_text(encoding='utf-8')

    def repl(m):
        key = m.group(1)
        return clauses.get(key, f'<!-- MISSING CLAUSE: {key} -->')

    out = re.sub(r'\{\{CLAUSE:([a-zA-Z0-9_\-]+)\}\}', repl, tpl)
    Path(args.out).write_text(out, encoding='utf-8')
    print('Wrote', args.out)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog='proposal-agent')
    sp = p.add_subparsers(dest='cmd')

    a = sp.add_parser('list-clauses')
    a.add_argument('--clauses', help='Path to clauses dir')
    a.set_defaults(func=cmd_list_clauses)

    a = sp.add_parser('dump-clause')
    a.add_argument('name')
    a.add_argument('--clauses', help='Path to clauses dir')
    a.set_defaults(func=cmd_dump_clause)

    a = sp.add_parser('generate')
    a.add_argument('--template', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--clauses', help='Path to clauses dir')
    a.set_defaults(func=cmd_generate)

    args = p.parse_args(argv)
    if not hasattr(args, 'func'):
        p.print_help()
        return 2
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
