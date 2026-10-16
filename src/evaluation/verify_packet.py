#!/usr/bin/env python3
"""Verify a frozen Wave D packet (stdlib only): every file hash, no extra files,
freeze sha recomputed from configuration+files, directory name, optional checkpoint
and sbatch hashes. Run before submission:  python3 -I verify_packet.py --packet DIR
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--sbatch', type=Path, default=None)
    parser.add_argument('--expected-sbatch-sha256', default=None)
    args = parser.parse_args()
    packet = args.packet.resolve()
    manifest = json.loads((packet / 'FROZEN_MANIFEST.json').read_text())
    files = manifest['files']
    freeze = hashlib.sha256(json.dumps({'configuration': manifest['configuration'],
                                        'files': {k: v['sha256'] for k, v in files.items()}},
                                       sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    present = sorted(str(p.relative_to(packet)) for p in packet.rglob('*') if p.is_file())
    extra = [p for p in present if p not in files and p != 'FROZEN_MANIFEST.json']
    missing = [k for k in files if not (packet / k).is_file()]
    mismatched = [k for k in files if k not in missing and digest(packet / k) != files[k]['sha256']]
    report = {'packet': str(packet), 'freeze_sha256_manifest': manifest['freeze_sha256'], 'freeze_sha256_recomputed': freeze,
              'freeze_consistent': freeze == manifest['freeze_sha256'],
              'directory_name_matches': packet.name == 'waveD_' + freeze[:12],
              'files_declared': len(files), 'missing': missing, 'mismatched': mismatched, 'extra_files': extra}
    if args.checkpoint:
        report['checkpoint_sha256_ok'] = digest(args.checkpoint) == manifest['configuration']['checkpoint_sha256']
    if args.sbatch:
        report['sbatch_sha256'] = digest(args.sbatch)
        if args.expected_sbatch_sha256:
            report['sbatch_sha256_ok'] = report['sbatch_sha256'] == args.expected_sbatch_sha256
    ok = (report['freeze_consistent'] and report['directory_name_matches'] and not missing and not mismatched and not extra
          and report.get('checkpoint_sha256_ok', True) and report.get('sbatch_sha256_ok', True))
    report['PASSED'] = ok
    print(json.dumps(report, indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
