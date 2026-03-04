from pathlib import Path
from proposal_agent.kb.loader import load_clauses


def test_load_iso_clause():
    clauses_dir = Path(__file__).parent.parent / 'kb' / 'clauses'
    clauses = load_clauses(clauses_dir)
    assert 'iso' in clauses
    assert 'ISO/IEC 27001' in clauses['iso']
