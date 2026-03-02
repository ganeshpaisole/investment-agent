"""Small CLI for interacting with the proposal_agent KB.

Usage:
  python -m proposal_agent.cli list
  python -m proposal_agent.cli get iso
  python -m proposal_agent.cli search security
"""
import argparse
from pathlib import Path
from .kb import loader


def main(argv=None):
    parser = argparse.ArgumentParser(prog='proposal_agent')
    sub = parser.add_subparsers(dest='cmd')

    sub.add_parser('list', help='List available clauses')

    get_p = sub.add_parser('get', help='Print clause by name')
    get_p.add_argument('name')

    search_p = sub.add_parser('search', help='Search clauses by keyword')
    search_p.add_argument('keyword')
    filt_p = sub.add_parser('filter', help='Filter clauses by tag/priority/jurisdiction')
    filt_p.add_argument('--tag', action='append', help='Filter by tag (repeatable)', default=[])
    filt_p.add_argument('--min-priority', type=int, help='Minimum priority (inclusive)', default=0)
    filt_p.add_argument('--jurisdiction', help='Filter by jurisdiction code (e.g. eu, global)', default=None)
    filt_p.add_argument('--risk-level', help='Filter by risk level (e.g. high, critical)', default=None)
    filt_p.add_argument('--owner', help='Filter by owner name', default=None)
    filt_p.add_argument('--json', action='store_true', help='Output results as JSON')
    meta_p = sub.add_parser('metadata', help='Print clause metadata as JSON')
    meta_p.add_argument('name', help='Clause filename/stem to print metadata for')

    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    kb = loader.load_clauses(root / 'kb')

    if args.cmd == 'list':
        for k in sorted(kb.keys()):
            v = kb.get(k)
            title = ''
            if isinstance(v, dict):
                title = v.get('meta', {}).get('title', '')
            print(f"{k}" + (f" - {title}" if title else ''))
    elif args.cmd == 'get':
        name = args.name
        v = kb.get(name)
        if not v:
            print(f'Clause not found: {name}')
            return 2
        text = v['text'] if isinstance(v, dict) else v
        print(text)
    elif args.cmd == 'search':
        matches = loader.find_clauses_by_keyword(kb, args.keyword)
        if not matches:
            print('No matches')
            return 1
        for m in matches:
            print(m)
    elif args.cmd == 'filter':
        tags = [t.lower() for t in (args.tag or [])]
        min_pr = args.min_priority or 0
        jur = args.jurisdiction.lower() if args.jurisdiction else None
        risk_filter = args.risk_level.lower() if args.risk_level else None
        owner_filter = args.owner.lower() if args.owner else None

        results = []
        for name, v in kb.items():
            meta = v.get('meta', {}) if isinstance(v, dict) else {}
            ctags = [t.lower() for t in meta.get('tags', [])] if meta else []
            pr = meta.get('priority') if meta else None
            if pr is None:
                pr = 0
            juris = [j.lower() for j in meta.get('jurisdiction', [])] if meta else []
            rlevel = meta.get('risk_level') if meta else None
            if rlevel:
                rlevel = rlevel.lower()
            owner_meta = meta.get('owner') if meta else None
            if owner_meta:
                owner_meta = owner_meta.lower()

            if tags:
                if not any(t in ctags for t in tags):
                    continue
            if pr < min_pr:
                continue
            if jur and juris and jur not in juris:
                continue
            if risk_filter and (not rlevel or risk_filter != rlevel):
                continue
            if owner_filter and (not owner_meta or owner_filter not in owner_meta):
                continue

            title = meta.get('title') if meta else None
            results.append((name, title or '', pr, juris))

        if not results:
            print('No clauses matched filter')
            return 1
        if args.json:
            import json
            json_out = []
            for name, title, pr, juris in sorted(results, key=lambda x: (-x[2], x[0])):
                meta = kb.get(name, {}).get('meta', {}) if isinstance(kb.get(name), dict) else {}
                json_out.append({
                    'name': name,
                    'title': title,
                    'priority': pr,
                    'jurisdiction': juris,
                    'risk_level': meta.get('risk_level'),
                    'owner': meta.get('owner'),
                })
            print(json.dumps(json_out, ensure_ascii=False))
        else:
            for name, title, pr, juris in sorted(results, key=lambda x: (-x[2], x[0])):
                out = name
                if title:
                    out += ' - ' + title
                out += f' (priority={pr})'
                if juris:
                    out += ' [' + ','.join(juris) + ']'
                print(out)
    elif args.cmd == 'metadata':
        name = args.name
        v = kb.get(name)
        if not v:
            print(f'Clause not found: {name}')
            return 2
        meta = v.get('meta', {}) if isinstance(v, dict) else {}
        import json
        print(json.dumps(meta, indent=2, ensure_ascii=False))
    else:
        parser.print_help()


if __name__ == '__main__':
    raise SystemExit(main())
