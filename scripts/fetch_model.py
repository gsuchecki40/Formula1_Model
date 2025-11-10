#!/usr/bin/env python3
"""Download a model file from a URL into the repo `data/` folder.

Usage:
  python3 scripts/fetch_model.py --url <URL> --out data/streamlit_model.joblib

If no --out is provided the default is data/streamlit_model.joblib
"""
import argparse
import os
import sys
from urllib.request import urlopen, Request


def download(url, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    req = Request(url, headers={"User-Agent": "python-urllib/3"})
    try:
        with urlopen(req) as resp, open(out_path, "wb") as f:
            chunk = resp.read(8192)
            while chunk:
                f.write(chunk)
                chunk = resp.read(8192)
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="Model URL to download")
    p.add_argument("--out", default="data/streamlit_model.joblib", help="Output path")
    args = p.parse_args()

    print(f"Downloading {args.url} -> {args.out}")
    ok = download(args.url, args.out)
    if not ok:
        print("Failed to download model", file=sys.stderr)
        sys.exit(2)
    print("Download complete")


if __name__ == "__main__":
    main()
