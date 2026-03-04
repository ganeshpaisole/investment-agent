import json
from proposal_agent import cli
from pathlib import Path


def test_filter_security_returns_iso(capsys):
    cli.main(['filter', '--tag', 'security'])
    out = capsys.readouterr().out
    assert 'iso' in out.lower()


def test_metadata_iso_json(capsys):
    cli.main(['metadata', 'iso'])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert 'title' in data
    assert 'tags' in data


def test_filter_json_output(capsys):
    cli.main(['filter', '--tag', 'security', '--json'])
    out = capsys.readouterr().out
    arr = json.loads(out)
    assert isinstance(arr, list)
    assert any(item.get('name','').lower() == 'iso' for item in arr)
