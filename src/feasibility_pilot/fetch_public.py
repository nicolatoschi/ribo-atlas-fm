#!/usr/bin/env python3
"""Download public files (RMDB detail pages / RDATs via RMDB's own download links,
RCSB coordinate files) into inputs/downloaded/ and append URL + SHA-256 + bytes to
inputs/download_log.jsonl. Usage: fetch_public.py <subdir> <url> [<url> ...]"""
import datetime as dt, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = ("https://rmdb.stanford.edu/", "https://github.com/DasLab/rmdb.github.io/releases/download/",
           "https://files.rcsb.org/download/")

def main():
    sub = sys.argv[1]
    out = ROOT / "inputs" / "downloaded" / sub
    out.mkdir(parents=True, exist_ok=True)
    log = ROOT / "inputs" / "download_log.jsonl"
    for url in sys.argv[2:]:
        if not url.startswith(ALLOWED):
            raise SystemExit(f"URL not in allowed public sources: {url}")
        name = url.rstrip("/").split("/")[-1]
        if url.startswith("https://rmdb.stanford.edu/detail/"):
            name = name + "_rmdb_detail.html"
        target = out / name
        r = subprocess.run(["curl", "-sSL", "--fail", "--max-time", "120", "-o", str(target),
                            "-w", "%{http_code} %{url_effective}", url], capture_output=True, text=True)
        if r.returncode != 0:
            print("FAILED", url, r.stderr.strip()); continue
        code, effective = r.stdout.split(" ", 1)
        data = target.read_bytes()
        rec = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "url": url, "effective_url": effective,
               "http_status": int(code), "path": str(target.relative_to(ROOT)), "bytes": len(data),
               "sha256": hashlib.sha256(data).hexdigest()}
        with open(log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(rec["path"], rec["bytes"], rec["sha256"][:16])

if __name__ == "__main__":
    main()
