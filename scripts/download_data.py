"""Download the Ertarğın et al. (2026) smartphone motor-vibration CSVs from Mendeley Data.

Dataset: https://doi.org/10.17632/rs4vz8n3t5 (CC BY 4.0). Only the .csv copies are
fetched; the .mat files hold the same recordings.

Usage: python scripts/download_data.py [--out data/raw]
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

API = "https://data.mendeley.com/public-api/datasets/rs4vz8n3t5"
# Mendeley rejects urllib's default User-Agent with 403.
HEADERS = {"User-Agent": "Mozilla/5.0 (cse455-motor-pr downloader)"}


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS)) as r:
        return r.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = json.loads(_fetch(API))
    files = [f for f in meta["files"] if f["filename"].endswith(".csv")]
    print(f"dataset version {meta['version']}: {len(files)} CSV files")

    for f in sorted(files, key=lambda f: f["filename"]):
        dest = args.out / f["filename"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            continue
        print("  fetching", f["filename"])
        dest.write_bytes(_fetch(f["content_details"]["download_url"]))
        if dest.stat().st_size != f["size"]:
            raise RuntimeError(f"size mismatch for {f['filename']}")

    (args.out / "SOURCE.json").write_text(json.dumps({
        "doi": meta["doi"]["id"],
        "version": meta["version"],
        "licence": meta["data_licence"]["short_name"],
        "citation": "Ertarğın, M., Orhan, A., Yıldırım, Ö., Chauhan, S., & Vashishtha, G. (2026). "
                    "A smartphone-based vibration dataset for induction motor fault diagnosis under "
                    "different speed and load conditions. Data in Brief, 67, 112916. "
                    "https://doi.org/10.1016/j.dib.2026.112916",
    }, indent=2))
    print("done")


if __name__ == "__main__":
    main()
