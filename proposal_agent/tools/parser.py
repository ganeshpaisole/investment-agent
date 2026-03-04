#!/usr/bin/env python3
"""RFP parser: extract sections from plain-text or PDF RFPs.

Features:
- Accepts .txt and .pdf files. If `pdfplumber` is installed, PDFs are extracted.
- Improved heuristics for finding headings and bullet lists.
"""
import sys
import json
import re
from pathlib import Path

try:
    import pdfplumber
except Exception:
    pdfplumber = None

SECTION_TAGS = [
    "Submission deadline",
    "Scope",
    "Timeline",
    "Must-have",
    "Nice-to-have",
    "Evaluation Criteria",
    "Compliance",
]


def extract_text_from_pdf(path: Path) -> str:
    if not pdfplumber:
        raise RuntimeError("pdfplumber not installed; cannot extract PDF text")
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def normalize_headings(text: str) -> str:
    # Normalize common heading markers to a consistent 'HEADING:' form
    # e.g., turn '\nScope\n- item' into '\nScope: ...' to aid regex
    for tag in SECTION_TAGS:
        # replace tag lines that are followed by newline with 'Tag: '
        text = re.sub(rf"(?im)^\s*{re.escape(tag)}\s*\n", f"{tag}:\n", text)
        # ensure 'Tag :' variants normalized
        text = re.sub(rf"(?im)^{re.escape(tag)}\s*:\s*", f"{tag}: ", text)
    return text


def parse_rfp_text(text: str) -> dict:
    # Basic metadata
    lines = [l for l in text.splitlines() if l.strip()]
    title = lines[0].strip() if lines else "RFP"
    data = {"title": title, "raw": text}

    text = normalize_headings(text)

    # Find submission deadline (more flexible)
    m = re.search(r"Submission deadline[:\s]*([0-9A-Za-z,\-/ ]+)", text, re.IGNORECASE)
    if m:
        data["submission_deadline"] = m.group(1).strip()

    # Find company name if present
    m = re.search(r"Company[:\s]*([A-Za-z0-9 &\-\.]+)", text, re.IGNORECASE)
    if m:
        data['company'] = m.group(1).strip()

    # Extract sections by heading keywords: capture until next heading or EOF
    # Build a regex that matches any of the section tags as a group
    tags_pattern = "|".join([re.escape(t) for t in SECTION_TAGS])
    pattern = re.compile(rf"(?ms)^(?:{tags_pattern})[:]?\s*(.*?)^(?=(?:{tags_pattern})[:]?\s*|\Z)", re.IGNORECASE | re.MULTILINE)
    matches = list(pattern.finditer(text))
    for m in matches:
        heading = m.group(0).split(':', 1)[0].strip()
        body = m.group(1).strip()
        key = heading.lower().replace(' ', '_').replace('-', '_')
        data[key] = body

    # Heuristic: split 'Scope' into bullet lines (support '-', '*', numbered lists)
    scope = data.get('scope')
    if scope:
        items = []
        for line in scope.splitlines():
            line = line.strip()
            if not line:
                continue
            # remove leading list markers
            line = re.sub(r"^[\-\*\d\.)\s]+", "", line).strip()
            items.append(line)
        data['scope_items'] = items
        data['modules_count'] = len(items)
    else:
        data['scope_items'] = []
        data['modules_count'] = 0

    return data


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: parser.py <rfp.txt|rfp.pdf>")
        sys.exit(1)
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(1)

    if p.suffix.lower() == '.pdf':
        if not pdfplumber:
            print("pdfplumber not installed. Install with: pip install pdfplumber")
            sys.exit(1)
        text = extract_text_from_pdf(p)
    else:
        text = p.read_text(encoding='utf-8')

    parsed = parse_rfp_text(text)
    out = p.with_suffix('.parsed.json')
    out.write_text(json.dumps(parsed, indent=2), encoding='utf-8')
    print(f"Parsed RFP -> {out}")
