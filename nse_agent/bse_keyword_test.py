"""
BSE PDF Keyword Test
Searches the downloaded HDFCBANK PDF for banking metric keywords
and shows surrounding context so we can write correct regex patterns.
"""
import pdfplumber

pdf_path = "data/bse_pdfs/test_hdfcbank.pdf"

# Extract all text
text = ""
with pdfplumber.open(pdf_path) as p:
    print(f"Total pages: {len(p.pages)}")
    for i, page in enumerate(p.pages):
        t = page.extract_text()
        if t:
            text += t + "\n"

print(f"Total chars extracted: {len(text):,}")
print("=" * 60)

keywords = [
    "CASA",
    "Capital Adequacy",
    "CRAR",
    "NIM",
    "Net Interest Margin",
    "GNPA",
    "NNPA",
    "Gross NPA",
    "Net NPA",
    "ROA",
    "ROE",
    "Credit Cost",
    "Provision Coverage",
    "Cost of Fund",
    "Slippage",
    "PCR",
]

print("KEYWORD SEARCH RESULTS:")
print("=" * 60)
found = []
missing = []

for kw in keywords:
    idx = text.upper().find(kw.upper())
    if idx >= 0:
        # Show 30 chars before and 100 after for context
        snippet = text[max(0, idx-30):idx+100].replace("\n", " | ")
        print(f"✅ {kw}")
        print(f"   ...{snippet}...")
        print()
        found.append(kw)
    else:
        missing.append(kw)

print("=" * 60)
print(f"FOUND:   {found}")
print(f"MISSING: {missing}")
print()

# Also print a sample of raw text around page 3-5 where metrics usually appear
print("=" * 60)
print("RAW TEXT SAMPLE (chars 3000-6000) — where metrics usually appear:")
print("=" * 60)
print(text[3000:6000])
