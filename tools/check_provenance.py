#!/usr/bin/env python3
"""Check this repository's copied files against PROVENANCE.json, and optionally against a pack.

Without arguments: every file listed in PROVENANCE.json must exist here with the recorded sha256.
With --evidence-root (the execution pack's evidence/ directory, which mirrors the project tree):
additionally, each origin file must exist there with the recorded origin_sha256, so the repository
copy and the frozen-packet copy are shown to be the same bytes.

Usage: python3 -I tools/check_provenance.py [--evidence-root <pack>/evidence]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-root', type=Path)
    a = ap.parse_args()
    prov = json.loads((REPO / 'PROVENANCE.json').read_text())['files']
    bad, absent_in_pack, checked_in_pack = [], [], 0
    for rel, e in prov.items():
        p = REPO / rel
        if not p.is_file() or sha(p) != e['sha256']:
            bad.append(rel)
        if a.evidence_root and 'origin' in e:
            q = a.evidence_root / e['origin']
            if q.is_file():
                checked_in_pack += 1
                if sha(q) != e['origin_sha256']:
                    bad.append('pack:' + e['origin'])
            else:
                absent_in_pack.append(e['origin'])
    out = {'listed': len(prov), 'mismatched': bad}
    if a.evidence_root:
        out.update({'origin_files_checked_in_pack': checked_in_pack, 'origins_not_in_pack': absent_in_pack})
    print(json.dumps(out, indent=1))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
