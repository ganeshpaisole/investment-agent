from pathlib import Path
from typing import Dict, Union, List, Any


def _strip_fenced_markdown(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith('```'):
        lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
    return '\n'.join(lines).strip()


def _parse_front_matter(text: str) -> (dict, str):
    """Parse optional YAML-like front-matter from the top of a markdown file.

    Returns (meta_dict, remaining_text).
    The front-matter is expected between lines with only '---'.
    Supports simple key: value pairs. Values that look like lists [a, b] are parsed.
    """
    lines = text.splitlines()
    if not lines:
        return {}, text
    if lines[0].strip() not in ('---', '+++'):
        return {}, text
    # find closing delimiter
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() in ('---', '+++'):
            end = i
            break
    if end is None:
        return {}, text

    meta_lines = lines[1:end]
    body_lines = lines[end+1:]
    meta: Dict[str, Any] = {}
    for ln in meta_lines:
        if ':' not in ln:
            continue
        k, v = ln.split(':', 1)
        key = k.strip()
        val = v.strip()
        # simple list notation
        if val.startswith('[') and val.endswith(']'):
            inner = val[1:-1].strip()
            items = [it.strip().strip('"').strip("'") for it in inner.split(',') if it.strip()]
            meta[key] = items
        else:
            # unquote if present
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            meta[key] = val

    body = '\n'.join(body_lines)
    body = _strip_fenced_markdown(body)
    # Normalize common fields
    if 'tags' in meta:
        t = meta.get('tags')
        if isinstance(t, str):
            # comma separated
            meta['tags'] = [x.strip().lower() for x in t.split(',') if x.strip()]
        elif isinstance(t, list):
            meta['tags'] = [str(x).strip().lower() for x in t]
        else:
            meta['tags'] = [str(t).strip().lower()]
    if 'priority' in meta:
        try:
            meta['priority'] = int(meta['priority'])
        except Exception:
            try:
                meta['priority'] = int(float(meta['priority']))
            except Exception:
                meta['priority'] = None
    if 'jurisdiction' in meta:
        j = meta.get('jurisdiction')
        if isinstance(j, str):
            meta['jurisdiction'] = [x.strip().lower() for x in j.split(',') if x.strip()]
        elif isinstance(j, list):
            meta['jurisdiction'] = [str(x).strip().lower() for x in j]
        else:
            meta['jurisdiction'] = [str(j).strip().lower()]

    # Normalize optional risk_level and owner
    if 'risk_level' in meta:
        rl = meta.get('risk_level')
        if rl is None:
            meta['risk_level'] = None
        else:
            meta['risk_level'] = str(rl).strip().lower()
    if 'owner' in meta:
        ow = meta.get('owner')
        if ow is None:
            meta['owner'] = None
        else:
            meta['owner'] = str(ow).strip()

    return meta, body


def load_clauses(kb_dir: Union[str, Path]) -> Dict[str, dict]:
    """Load markdown clauses from a `kb` directory and return mapping name -> {meta, text}.

    Backwards-compatible: earlier callers expecting a str can use the 'text' value.
    """
    kb_dir = Path(kb_dir)
    clauses: Dict[str, dict] = {}

    for sub in ('clauses', 'templates'):
        folder = kb_dir / sub
        if not folder.exists():
            continue
        for p in sorted(folder.glob('*.md')):
            try:
                raw = p.read_text(encoding='utf-8')
            except Exception:
                raw = p.read_text(encoding='utf-8', errors='ignore')

            meta, body = _parse_front_matter(raw)
            if not body:
                continue
            clauses[p.stem] = {'meta': meta, 'text': body}

    return clauses


def find_clauses_by_keyword(clauses: Dict[str, Union[str, dict]], keywords: Union[str, List[str]]):
    """Return clause names that match any of the provided keywords (case-insensitive).

    Accepts `clauses` values as either raw text or dicts with a 'text' key.
    """
    if isinstance(keywords, str):
        keywords = [keywords]
    kws = [k.lower() for k in keywords]
    matches = []
    for name, val in clauses.items():
        if isinstance(val, dict):
            texts = [val.get('text',''), ' '.join([str(v) for v in val.get('meta',{}).values()])]
        else:
            texts = [val]
        low = ' '.join(texts).lower()
        if any(k in low for k in kws):
            matches.append(name)
    return matches
