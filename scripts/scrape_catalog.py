from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "shl_assessments_live.json"


def scrape_catalog(max_pages: int = 40) -> list[dict[str, str]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 SHL-assignment-recommender"})
    rows: list[dict[str, str]] = []

    for page in range(max_pages):
        url = f"{BASE_URL}?start={page * 12}&type=1"
        response = session.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        page_rows = parse_rows(soup)
        if not page_rows:
            break
        rows.extend(page_rows)

    return dedupe(rows)


def parse_rows(soup: BeautifulSoup) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for table_row in soup.select("tr"):
        link = table_row.find("a", href=re.compile(r"product-catalog/view"))
        if not link:
            continue
        name = " ".join(link.get_text(" ", strip=True).split())
        url = urljoin(BASE_URL, link["href"])
        text = table_row.get_text(" ", strip=True)
        codes = " ".join(dict.fromkeys(re.findall(r"\b[A-Z]\b", text)))
        rows.append(
            {
                "name": name,
                "url": url,
                "test_type": codes,
                "category": "Individual Test Solution",
            }
        )
    return rows


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = row["url"].rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


if __name__ == "__main__":
    data = scrape_catalog()
    OUTPUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {len(data)} rows to {OUTPUT}")

