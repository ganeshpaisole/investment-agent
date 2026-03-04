"""Fetch HDFC Bank sitemap(s) and search for PDF or investor/presentation links."""
import requests
from urllib.parse import urljoin
from pathlib import Path

base='https://www.hdfcbank.com/'
targets=['sitemap.xml','sitemap_index.xml']

found = []
for t in targets:
    url = urljoin(base, t)
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            continue
        text = r.text
        for line in text.splitlines():
            l = line.strip()
            if '.pdf' in l.lower() or 'investor' in l.lower() or 'presentation' in l.lower():
                found.append((t, l))
    except Exception as e:
        print('error fetching', url, e)

if not found:
    print('No interesting entries found in sitemap(s)')
else:
    for t, l in found:
        print(t, l)
