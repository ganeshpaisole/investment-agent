# Proposal Agent — Spec

Purpose
- Generate fixed-price proposals for SAP S/4, Dynamics 365, and MES (implementation, enhancement, support).

MVP Acceptance Criteria
- Parse a plain-text RFP and extract core sections: scope, timeline, must-have, nice-to-have, evaluation criteria.
- Produce: Word proposal (exec + detailed SOW), PowerPoint executive summary (3-6 slides), Excel/CSV pricing workbook, and a Gantt CSV.
- Estimation: rule-based fixed-price using default role rates and contingency.
- Include compliance clauses: GDPR, ISO, Data Residency, local procurement.

Interfaces
- CLI: `parser.py <rfp.txt>` -> JSON output; `generator.py <parsed.json>` -> outputs in `output/`.

Storage
- Simple filesystem layout under `proposal_agent/` using JSON/CSV and Markdown templates.

Notes
- No external integrations for MVP. Templates are vendor-style, formal tone.
