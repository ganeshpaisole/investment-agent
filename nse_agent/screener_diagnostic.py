"""
screener_diagnostic.py
======================
Run this ONCE locally to inspect Screener.in's current HTML structure.
It saves the raw HTML and prints what data it can find.

Usage:
    python screener_diagnostic.py

Output:
    - screener_raw_HDFCBANK.html   (full page HTML for manual inspection)
    - screener_raw_TCS.html
    - Prints all found data to terminal
"""

import requests
from bs4 import BeautifulSoup
import json
import time

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

TEST_TICKERS = ["HDFCBANK", "TCS", "BAJFINANCE"]

def diagnose(ticker: str):
    url = f"https://www.screener.in/company/{ticker}/consolidated/"
    print(f"\n{'='*60}")
    print(f"DIAGNOSING: {ticker}")
    print(f"URL: {url}")
    print(f"{'='*60}")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        print(f"Status: {resp.status_code}")

        if resp.status_code != 200:
            print(f"❌ Failed — status {resp.status_code}")
            # Try standalone URL
            url2 = f"https://www.screener.in/company/{ticker}/"
            resp2 = requests.get(url2, headers=HEADERS, timeout=20)
            print(f"Standalone URL status: {resp2.status_code}")
            if resp2.status_code == 200:
                resp = resp2
            else:
                return

        # Save raw HTML for manual inspection
        html_file = f"screener_raw_{ticker}.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(resp.text)
        print(f"✅ Raw HTML saved to {html_file}")
        print(f"   Open in browser to inspect element structure")

        soup = BeautifulSoup(resp.content, "lxml")

        # ── APPROACH 1: Original method (div#top-ratios) ──────────
        print("\n--- Approach 1: div#top-ratios ---")
        section = soup.find("div", {"id": "top-ratios"})
        if section:
            print(f"  ✅ Found #top-ratios")
            items = section.find_all("li")
            print(f"  Found {len(items)} <li> items")
            for item in items[:5]:
                print(f"  Item HTML: {item}")
                name = item.find("span", class_="name")
                val  = item.find("span", class_="value")
                print(f"  → name_tag: {name}, value_tag: {val}")
        else:
            print("  ❌ #top-ratios NOT found")

        # ── APPROACH 2: Any ul with ratios ───────────────────────
        print("\n--- Approach 2: All <ul> tags ---")
        uls = soup.find_all("ul")
        print(f"  Found {len(uls)} <ul> tags")
        for i, ul in enumerate(uls[:10]):
            cls = ul.get("class", [])
            id_ = ul.get("id", "")
            print(f"  ul[{i}] id='{id_}' class='{cls}' — {len(ul.find_all('li'))} li items")

        # ── APPROACH 3: Look for any number-like spans ────────────
        print("\n--- Approach 3: spans with numbers ---")
        # Screener typically has spans with class 'number' or 'value'
        for cls in ["number", "value", "font-size-18", "nowrap"]:
            spans = soup.find_all("span", class_=cls)
            if spans:
                print(f"  span.{cls}: {len(spans)} found — first 3:")
                for s in spans[:3]:
                    print(f"    {s}")

        # ── APPROACH 4: Look for key financial terms ──────────────
        print("\n--- Approach 4: Search for known ratio labels ---")
        page_text = soup.get_text()
        keywords = ["Market Cap", "P/E", "Book Value", "Dividend Yield",
                    "ROCE", "ROE", "Debt / Equity", "Current Ratio",
                    "Promoter", "NPA", "CASA"]
        for kw in keywords:
            found = kw.lower() in page_text.lower()
            print(f"  '{kw}': {'✅ Found' if found else '❌ Not found'}")

        # ── APPROACH 5: Check for JSON data in script tags ────────
        print("\n--- Approach 5: JSON data in <script> tags ---")
        scripts = soup.find_all("script")
        print(f"  Found {len(scripts)} script tags")
        for i, s in enumerate(scripts):
            text = s.get_text()
            if any(kw in text for kw in ["roce", "roe", "market_cap", "book_value"]):
                print(f"  ✅ Script[{i}] contains financial data (first 300 chars):")
                print(f"  {text[:300]}")
                break

        # ── APPROACH 6: Section IDs ───────────────────────────────
        print("\n--- Approach 6: All section/div IDs ---")
        for tag in soup.find_all(["section", "div"], id=True):
            print(f"  <{tag.name} id='{tag['id']}'>")

        # ── APPROACH 7: Tables ────────────────────────────────────
        print("\n--- Approach 7: Tables ---")
        tables = soup.find_all("table")
        print(f"  Found {len(tables)} tables")
        for i, t in enumerate(tables[:5]):
            cls = t.get("class", [])
            print(f"  table[{i}] class='{cls}' — {len(t.find_all('tr'))} rows")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    time.sleep(2)  # polite delay

if __name__ == "__main__":
    print("Screener.in HTML Diagnostic Tool")
    print("="*60)
    print("This will:")
    print("1. Fetch Screener.in pages for 3 test stocks")
    print("2. Save raw HTML files for manual inspection")
    print("3. Try 7 different approaches to find data")
    print("="*60)

    for ticker in TEST_TICKERS:
        diagnose(ticker)
        time.sleep(3)

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("1. Open screener_raw_HDFCBANK.html in browser")
    print("2. Right-click on a ratio (e.g. P/E) → Inspect Element")
    print("3. Note the exact HTML structure")
    print("4. Share the output of this script with Claude")
    print("="*60)
