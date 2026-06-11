"""Download Football-Data.co.uk international odds CSVs to data/raw/football_data/.

Usage: python scripts/download_football_data.py
"""

from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
URL_FILE = ROOT / "data" / "raw" / "football_data_urls.txt"
OUT_DIR = ROOT / "data" / "raw" / "football_data"

HEADERS = {"User-Agent": "polymbappe/0.1 (+https://github.com/)"}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    urls = [
        line.strip()
        for line in URL_FILE.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for url in urls:
        name = url.rsplit("/", 1)[-1]
        dest = OUT_DIR / name
        if dest.exists():
            print(f"  skip (exists): {name}")
            continue
        print(f"  downloading: {name} ...", end=" ")
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"OK ({len(resp.content)} bytes)")
    print(f"Done. {len(list(OUT_DIR.glob('*.csv')))} CSVs in {OUT_DIR}")


if __name__ == "__main__":
    main()
