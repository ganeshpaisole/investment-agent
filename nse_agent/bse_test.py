"""
BSE Parser Diagnostic — run this first to check connectivity and PDF URLs
Usage: python bse_test.py
"""
import requests
import pathlib

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.bseindia.com/"
}

# ── Step 1: Search for filings ────────────────────────────────
print("=" * 60)
print("Step 1: Searching BSE API for HDFCBANK filings...")
print("=" * 60)

url = (
    "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
    "?strCat=Result&strType=C&strScrip=500180"
    "&strSearch=P&strToDate=&strFromDate=&myClient="
)

try:
    r = requests.get(url, headers=headers, timeout=15)
    print("HTTP Status:", r.status_code)
    data = r.json()
    filings = data.get("Table", [])
    print("Total filings found:", len(filings))
    print()

    for i, f in enumerate(filings[:5]):
        desc = f.get("NEWSSUB", "")
        nsurl = f.get("NSURL", "")
        attach = f.get("ATTACHMENTNAME", "")
        date = f.get("News_submission_dt", "")
        print(f"[{i+1}] {desc[:70]}")
        print(f"     Date: {date}")
        print(f"     NSURL: {nsurl}")
        print(f"     ATTACHMENTNAME: {attach}")
        print()

except Exception as e:
    print("ERROR:", e)
    print()
    print("BSE API unreachable. Check internet connection.")
    exit(1)

# ── Step 2: Try downloading the first PDF ─────────────────────
print("=" * 60)
print("Step 2: Trying to download first PDF...")
print("=" * 60)

if not filings:
    print("No filings found — cannot test download")
    exit(1)

first = filings[0]
attach = first.get("ATTACHMENTNAME", "").strip()

if not attach:
    print("No ATTACHMENTNAME found in first filing")
    exit(1)

# AttachHis is the confirmed working BSE PDF path
pdf_url = "https://www.bseindia.com/xml-data/corpfiling/AttachHis/" + attach
print("ATTACHMENTNAME:", attach)
print("Downloading:", pdf_url)

try:
    r2 = requests.get(pdf_url, headers=headers, timeout=30, stream=True)
    print("HTTP Status:", r2.status_code)
    print("Content-Type:", r2.headers.get("Content-Type", "unknown"))
    print("Content-Length:", r2.headers.get("Content-Length", "unknown"), "bytes")

    # Save to disk
    out = pathlib.Path("data/bse_pdfs/test_hdfcbank.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        for chunk in r2.iter_content(8192):
            f.write(chunk)

    size_kb = out.stat().st_size // 1024
    print("Saved:", out, f"({size_kb} KB)")

except Exception as e:
    print("Download ERROR:", e)
    exit(1)

# ── Step 3: Try extracting text ───────────────────────────────
print()
print("=" * 60)
print("Step 3: Extracting text from PDF...")
print("=" * 60)

pdf_path = pathlib.Path("data/bse_pdfs/test_hdfcbank.pdf")

# Check first bytes — is it actually a PDF?
with open(pdf_path, "rb") as f:
    first_bytes = f.read(10)
print("First bytes:", first_bytes)
print("Is PDF:", first_bytes.startswith(b"%PDF"))

# Try pdfplumber
print()
print("Trying pdfplumber...")
try:
    import pdfplumber
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        print("Total pages:", len(pdf.pages))
        for i, page in enumerate(pdf.pages[:5]):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                print(f"  Page {i+1}: {len(page_text)} chars")
            else:
                print(f"  Page {i+1}: 0 chars (image-based?)")

    print()
    print("Total text extracted:", len(text), "chars")
    if text:
        print()
        print("--- First 500 chars of extracted text ---")
        print(text[:500])
        print("---")
    else:
        print("WARNING: No text extracted — PDF is likely image-based (scanned)")

except Exception as e:
    print("pdfplumber ERROR:", e)

print()
print("=" * 60)
print("Diagnostic complete.")
print("=" * 60)
