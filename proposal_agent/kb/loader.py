from pathlib import Path
from typing import Dict


def _normalize_content(text: str) -> str:
    """Remove surrounding code fences and trim whitespace."""
    text = text.strip()
    if text.startswith('```'):
        parts = text.split('\n')
        # drop the opening fence line
        parts = parts[1:]
        # drop closing fence if present
        if parts and parts[-1].strip().startswith('```'):
            parts = parts[:-1]
        text = '\n'.join(parts)
    return text.strip()


def load_clause(path: Path) -> str:
    """Load a single markdown clause and return its normalized content."""
    text = path.read_text(encoding='utf-8')
    return _normalize_content(text)


def load_clauses(dir_path) -> Dict[str, str]:
    """Load all `.md` clause files from `dir_path`.

    Returns a dict mapping filename stem -> clause content.
    """
    p = Path(dir_path)
    if not p.exists():
        return {}
    clauses = {}
    for f in sorted(p.glob('*.md')):
        try:
            clauses[f.stem] = load_clause(f)
        except Exception:
            # best-effort: skip unreadable files
            continue
    return clauses


def find_clauses_by_keyword(clauses: Dict[str, str], keyword: str) -> list:
    """Return list of clause keys whose filename stem or content matches keyword.

    This is a best-effort, case-insensitive substring match used by the
    generator to select candidate clauses for an RFP keyword.
    """
    if not keyword:
        return []
    k = keyword.lower().strip()
    matches = []
    for name, content in clauses.items():
        if k in name.lower() or k in (content or '').lower():
            matches.append(name)
    return matches


if __name__ == '__main__':
    # quick manual check
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'clauses'
    c = load_clauses(path)
    for k in sorted(c.keys()):
        print(k)
