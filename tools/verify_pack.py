#!/usr/bin/env python3
"""Verify an unzipped RIBO_ATLAS_FM_Execution_Pack_v2.

1. Every file listed in FILE_SHA256.json exists with the recorded sha256 and size, and no unlisted
   file is present (FILE_SHA256.json and SHA256SUMS.txt themselves excepted).
2. Every frozen packet under evidence/ (a directory named <name>_<12 hex> holding a packet manifest):
   each manifest entry is either present with the recorded sha256, or listed in EXCLUDED_FILES.json
   with the same sha256; the freeze sha256 recomputed from the manifest equals the recorded one and
   its first 12 hex digits equal the directory suffix.
3. Prints a JSON report; exits 1 on any failure.

Usage: python3 -I verify_pack.py --pack <unzipped RIBO_ATLAS_FM_Execution_Pack_v2 directory>
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

MANIFEST_NAMES = ['frozen_manifest.json', 'campaign_manifest.json', 'FROZEN_MANIFEST.json', 'waveE_manifest.json']
PACKET_DIR = re.compile(r'.*_([0-9a-f]{12})$')
EXTERNAL_MANIFESTS = {'work/hpc_pilot/expanded_frozen_af8074e80953': 'work/hpc_pilot/expanded_campaign_manifest.json'}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def freeze_candidates(manifest):
    files = {k: (v if isinstance(v, str) else v['sha256']) for k, v in manifest['files'].items()}
    body = {'configuration': manifest['configuration'], 'files': files}
    yield 'json.dumps(sort_keys=True)', hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    yield 'compact sorted JSON', hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if 'source_data_sha256' in manifest:  # older packets: sha256(source_data_sha256 + sorted configuration JSON)
        yield 'source_data_sha256 + json.dumps(configuration, sort_keys=True)', hashlib.sha256(
            manifest['source_data_sha256'].encode() + json.dumps(manifest['configuration'], sort_keys=True).encode()).hexdigest()


def source_data_sha(d, manifest):
    """sha256 over name+bytes of every manifest file in sorted order; None if any file is absent."""
    h = hashlib.sha256()
    for name in sorted(manifest['files']):
        p = d / name
        if not p.is_file():
            return None
        h.update(name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', type=Path, required=True)
    a = ap.parse_args()
    pack = a.pack.resolve()
    report = {'pack': pack.name, 'ok': True}
    listed = json.loads((pack / 'FILE_SHA256.json').read_text())['files']
    bad, missing = [], []
    for rel, rec in listed.items():
        p = pack / rel
        if not p.is_file():
            missing.append(rel); continue
        if p.stat().st_size != rec['bytes'] or sha_file(p) != rec['sha256']:
            bad.append(rel)
    on_disk = {p.relative_to(pack).as_posix() for p in pack.rglob('*') if p.is_file()}
    extra = sorted(on_disk - set(listed) - {'FILE_SHA256.json', 'SHA256SUMS.txt'})
    report['file_hashes'] = {'listed': len(listed), 'verified': len(listed) - len(bad) - len(missing),
                             'mismatched': bad, 'missing': missing, 'unlisted_extra': extra}
    if bad or missing or extra:
        report['ok'] = False
    excluded = json.loads((pack / 'EXCLUDED_FILES.json').read_text())['files']
    excl = {e['pack_path']: e['sha256'] for e in excluded}
    packets = []
    for mpath in sorted((pack / 'evidence').rglob('*')):
        if mpath.name not in MANIFEST_NAMES or not mpath.is_file():
            continue
        d = mpath.parent
        m = PACKET_DIR.match(d.name)
        if not m:
            continue
        manifest = json.loads(mpath.read_text())
        if 'files' not in manifest or 'freeze_sha256' not in manifest:
            continue
        rec = {'packet': d.relative_to(pack).as_posix(), 'manifest': mpath.name, 'entries': len(manifest['files']),
               'present_ok': 0, 'excluded_ok': 0, 'problems': []}
        for rel, v in manifest['files'].items():
            h = v if isinstance(v, str) else v['sha256']
            p = d / rel
            prel = p.relative_to(pack).as_posix()
            if p.is_file():
                if sha_file(p) == h:
                    rec['present_ok'] += 1
                else:
                    rec['problems'].append('hash mismatch: ' + rel)
            elif excl.get(prel) == h:
                rec['excluded_ok'] += 1
            else:
                rec['problems'].append('absent and not listed as excluded: ' + rel)
        rec['freeze_sha256'] = manifest['freeze_sha256']
        rec['freeze_recomputed'] = None
        for how, value in freeze_candidates(manifest):
            if value == manifest['freeze_sha256']:
                rec['freeze_recomputed'] = how
        if rec['freeze_recomputed'] and rec['freeze_recomputed'].startswith('source_data_sha256'):
            recomputed = source_data_sha(d, manifest)
            rec['source_data_sha256_recomputed'] = ('matches' if recomputed == manifest['source_data_sha256'] else 'MISMATCH') \
                if recomputed else 'not recomputable here (third-party files excluded); recorded value used'
            if recomputed and recomputed != manifest['source_data_sha256']:
                rec['problems'].append('source_data_sha256 mismatch')
        if rec['freeze_recomputed'] is None:
            rec['problems'].append('freeze sha256 not reproduced from manifest')
        if manifest['freeze_sha256'][:12] != m.group(1):
            rec['problems'].append('directory suffix differs from freeze sha256')
        if rec['problems']:
            report['ok'] = False
        packets.append(rec)
    # packets whose SHA-256 manifest lives beside the packet (systems prototype 2297211-16)
    for d_rel, m_rel in EXTERNAL_MANIFESTS.items():
        d, mp = pack / 'evidence' / d_rel, pack / 'evidence' / m_rel
        if not d.is_dir():
            continue
        cfg = json.loads(mp.read_text())['configuration']
        rec = {'packet': d.relative_to(pack).as_posix(), 'manifest': m_rel, 'entries': len(cfg['files']), 'present_ok': 0,
               'excluded_ok': 0, 'problems': [], 'freeze_sha256': cfg['freeze_sha256']}
        for rel, h in cfg['files'].items():
            q = d / rel
            if q.is_file() and sha_file(q) == h:
                rec['present_ok'] += 1
            else:
                rec['problems'].append('missing or mismatched: ' + rel)
        joined = b''.join(n.encode() + (d / n).read_bytes() for n in sorted(cfg['files']) if (d / n).is_file())
        ok = hashlib.sha256(joined).hexdigest() == cfg['source_sha256']
        rec['freeze_recomputed'] = ('source_sha256 recomputed and matches; the freeze hash also covers the campaign CLI arguments, '
                                    'which are recorded in campaign_configuration.json') if ok else None
        if not ok:
            rec['problems'].append('source_sha256 mismatch')
        if cfg['freeze_sha256'][:12] != d.name[-12:]:
            rec['problems'].append('directory suffix differs from freeze sha256')
        if rec['problems']:
            report['ok'] = False
        packets.append(rec)
    report['packets'] = packets
    report['excluded_files'] = {'count': len(excluded), 'bytes': sum(e['bytes'] for e in excluded)}
    print(json.dumps(report, indent=1))
    sys.exit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()
