import runpy
from pathlib import Path


def test_selects_gdpr_and_iso():
    mod_path = Path(__file__).resolve().parents[2] / 'tools' / 'generator.py'
    globs = runpy.run_path(str(mod_path))
    clauses = globs['load_clauses'](Path(__file__).resolve().parents[2] / 'kb' / 'clauses')
    parsed = {'title':'SAP S/4 migration','compliance':'ISO, GDPR','scope_items':['Finance migration'],'company':'ExampleCo'}
    selected = globs['select_clauses'](parsed, clauses)
    titles = [t for t, _ in selected]
    assert any('GDPR' == t or 'Gdpr' == t for t in titles)
    assert any(t.lower().startswith('iso') for t in titles)
