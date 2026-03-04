import runpy
from pathlib import Path


def test_tag_based_selection():
    mod_path = Path(__file__).resolve().parents[2] / 'tools' / 'generator.py'
    globs = runpy.run_path(str(mod_path))
    clauses = globs['load_clauses'](Path(__file__).resolve().parents[2] / 'kb' / 'clauses')
    parsed = {'title': 'Test', 'clause_tags': ['security'], 'scope_items': []}
    selected = globs['select_clauses'](parsed, clauses)
    titles = [t for t, _ in selected]
    assert any(t.lower().startswith('iso') for t in titles), f"Expected ISO clause for 'security' tag, got {titles}"
    # also ensure ISO meta priority present and is int when loaded from kb via generator
    for t, body in selected:
        if t.lower().startswith('iso'):
            # find meta by re-loading from loader to inspect meta
            from proposal_agent.kb import loader as kb_loader
            kbmap = kb_loader.load_clauses(Path(__file__).resolve().parents[2] / 'kb')
            iso_meta = kbmap.get('iso', {}).get('meta', {})
            assert isinstance(iso_meta.get('priority'), int)
