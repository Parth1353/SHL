from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import openpyxl
import requests


SOURCE_URL = "https://raw.githubusercontent.com/ankitshah074/shl-assessment-recommender/main/Gen_AI%20Dataset.xlsx"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "public_eval_pairs.json"


def download_public_pairs() -> list[dict[str, str]]:
    response = requests.get(SOURCE_URL, timeout=30)
    response.raise_for_status()
    with NamedTemporaryFile(suffix=".xlsx") as temp_file:
        temp_file.write(response.content)
        temp_file.flush()
        workbook = openpyxl.load_workbook(temp_file.name, data_only=True)
        worksheet = workbook["Train-Set"]
        rows: list[dict[str, str]] = []
        for query, url in worksheet.iter_rows(min_row=2, values_only=True):
            if query and url:
                rows.append({"query": str(query).strip(), "url": str(url).strip()})
        return rows


def main() -> None:
    rows = download_public_pairs()
    OUTPUT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} public eval pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
