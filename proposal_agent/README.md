Proposal Agent — README

Quick start (local):

1. Parse a sample RFP:

```bash
python proposal_agent/tools/parser.py proposal_agent/samples/rfps/sap_rfp.txt
```

This creates `sap_rfp.parsed.json` next to the RFP.

2. Generate outputs from the parsed JSON:

```bash
python proposal_agent/tools/generator.py proposal_agent/samples/rfps/sap_rfp.parsed.json
```

Outputs appear under `proposal_agent/samples/rfps/output/` (proposal.docx, pricing.csv, gantt.csv).

Notes
- This is a minimal MVP skeleton. For production, add PDF parsing, Word/PPTX generation (python-docx, python-pptx), richer estimation rules, and testing.

KB (knowledge base) format
- Clauses live under `kb/clauses` as Markdown files. Files may include optional YAML front-matter at the top to provide metadata, for example:

	---
	title: "ISO / Security Clause"
	tags: [security, iso, compliance]
	applicability: [cloud, onprem]
	---

- The agent loader (`proposal_agent.kb.loader.load_clauses`) returns a mapping name -> {meta: {...}, text: "..."}.

- Front-matter fields: common fields include `title` (string), `tags` (list), `priority` (int), `jurisdiction` (list). Additional recommended fields:
	- `risk_level`: a normalized string (e.g. `low`, `medium`, `high`, `critical`)
	- `owner`: string identifying the clause owner or maintainer (e.g., `Security Team`)

- Clause selection: the generator looks for keywords in the parsed RFP `compliance` field and also supports tag-based selection via the parsed fields `clause_tags` or `required_clause_tags` (list or comma-separated string). The CLI also supports listing and searching clauses.

CLI examples
- List clauses: `python -m proposal_agent.cli list`
- Get clause text: `python -m proposal_agent.cli get iso`
- Search by keyword: `python -m proposal_agent.cli search security`
- Filter by tag/priority/jurisdiction:
	- By tag: `python -m proposal_agent.cli filter --tag security`
	- By min priority: `python -m proposal_agent.cli filter --min-priority 80`
	- By jurisdiction: `python -m proposal_agent.cli filter --jurisdiction eu --tag gdpr`
	- Machine-readable JSON: `python -m proposal_agent.cli filter --tag security --json`

CI
- A minimal GitHub Actions workflow is included at `.github/workflows/ci.yml` which installs dev dependencies and runs `pytest` under Python 3.9.
