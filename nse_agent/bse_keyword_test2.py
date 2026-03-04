"""
BSE PDF Keyword Test — Multi Bank
Searches downloaded PDFs for banking metric keywords.
Run AFTER: python utils\bse_parser.py ICICIBANK SBIN AXISBANK --no-cache
"""
import pdfplumber
import pathlib
import glob

KEYWORDS = [
    "CASA", "NIM", "Net Interest Margin",
    "Gross NPA", "GNPA", "Net NPA", "NNPA",
    "NPA Ratio", "% of Gross", "% of Net",
    "ROA", "Return on Asset",
    "Capital Adequacy", "CRAR",
    "Credit Cost", "Provision",
]

def search_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = len(pdf.pages)
            for page in pdf.pages[:30]:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    print(f"  Pages: {pages} | Chars: {len(text):,}")
    print()

    found = []
    for kw in KEYWORDS:
        idx = text.upper().find(kw.upper())
        if idx >= 0:
            snippet = text[max(0, idx-20):idx+120].replace("\n", " | ")
            print(f"  ✅ {kw:25} → ...{snippet[:120]}...")
            found.append(kw)

    missing = [k for k in KEYWORDS if k not in found]
    if missing:
        print(f"\n  ❌ Missing: {missing}")

    # Show raw text from chars 3000-7000 where metrics usually appear
    print()
    print("  --- RAW TEXT (3000-6000 chars) ---")
    print(text[3000:6000])
    print("  ---")


# Find all downloaded PDFs
pdf_dir = pathlib.Path("data/bse_pdfs")
pdfs = sorted(pdf_dir.glob("*.pdf"), key=lambda p: p.stat().st_size, reverse=True)

print(f"Found {len(pdfs)} cached PDFs in {pdf_dir}")
print()

# Show the 3 largest PDFs (most likely to be results/presentations)
# Skip test_hdfcbank.pdf and HDFC IR PDF (already working)
skip_sizes = set()

for pdf_path in pdfs[:10]:
    size_kb = pdf_path.stat().st_size // 1024
    # Skip tiny PDFs (< 100KB — likely governance/ESOP notices)
    if size_kb < 100:
        continue
    # Skip the HDFCBANK ones we already handled
    if "test_hdfcbank" in pdf_path.name:
        continue

    print("=" * 70)
    print(f"FILE: {pdf_path.name} ({size_kb} KB)")
    print("=" * 70)
    search_pdf(pdf_path)
    print()

    # Only show top 4 largest
    skip_sizes.add(size_kb)
    if len(skip_sizes) >= 4:
        break
