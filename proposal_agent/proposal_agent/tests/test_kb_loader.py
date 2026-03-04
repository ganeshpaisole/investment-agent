from pathlib import Path
from proposal_agent.kb import loader


def test_load_clauses_contains_iso():
    root = Path(__file__).resolve().parents[2]
    kb = loader.load_clauses(root / 'kb')
    assert 'iso' in kb, 'iso clause should be present in kb/clauses'
    v = kb['iso']
    assert isinstance(v, dict)
    text = v.get('text', '')
    assert 'ISO/IEC 27001' in text or 'information security' in text.lower()
    # meta fields normalized
    meta = v.get('meta', {})
    assert isinstance(meta.get('tags', []), list)
    assert all(isinstance(t, str) for t in meta.get('tags', []))
    # jurisdiction should be a list and 'global' present for iso
    assert isinstance(meta.get('jurisdiction', []), list)
    assert 'global' in [x.lower() for x in meta.get('jurisdiction', [])]
    # new normalized fields
    assert isinstance(meta.get('risk_level'), str)
    assert isinstance(meta.get('owner'), str)
