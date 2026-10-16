#!/usr/bin/env python3
"""Pre-flight check for the Wave E packet (run by the sbatch script on the batch host).

Verifies every file listed in waveE_manifest.json, recomputes the freeze sha256 from
(configuration, files), checks that the dataset hash equals the frozen config, optionally
hashes the RNA-FM weights, and checks the interpreter: fm 0.2.2 importable and the expected
torch version. Exits non-zero on any mismatch; writes a JSON record either way.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--weights', type=Path)
    p.add_argument('--weights-sha256')
    p.add_argument('--expect-torch', default=None, help='required torch version prefix, e.g. 2.10')
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    t0 = time.time()
    record = {'packet': str(args.packet), 'checks': {}, 'errors': []}
    manifest = json.loads((args.packet / 'waveE_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = args.packet / name
        got = sha_file(path) if path.exists() else 'missing'
        if got != expected:
            record['errors'].append(f'{name}: {got} != {expected}')
    present = sorted(str(x.relative_to(args.packet)) for x in args.packet.rglob('*') if x.is_file())
    bytecode = [x for x in present if '__pycache__' in Path(x).parts]
    if bytecode:
        record['checks']['ignored_bytecode_caches'] = bytecode
    extra = sorted(set(present) - set(manifest['files']) - {'waveE_manifest.json'} - set(bytecode))
    if extra:
        record['errors'].append(f'unlisted files in packet: {extra}')
    freeze = hashlib.sha256(json.dumps({'configuration': manifest['configuration'], 'files': manifest['files']}, sort_keys=True).encode()).hexdigest()
    record['checks']['freeze_sha256'] = freeze
    if freeze != manifest['freeze_sha256'] or not args.packet.name.endswith(freeze[:12]):
        record['errors'].append(f'freeze sha mismatch: {freeze} vs {manifest["freeze_sha256"]} / {args.packet.name}')
    cfg = json.loads((args.packet / 'waveE_config.json').read_text())
    if manifest['files'].get('waveE_dataset.tsv') != cfg['dataset_sha256']:
        record['errors'].append('dataset hash in config differs from manifest')
    if args.weights:
        if not args.weights.exists():
            record['errors'].append(f'weights missing: {args.weights}')
        else:
            got = sha_file(args.weights)
            record['checks']['weights_sha256'] = got
            if got != (args.weights_sha256 or cfg['weights_sha256']) or got != cfg['weights_sha256']:
                record['errors'].append(f'weights sha256 {got} != {cfg["weights_sha256"]}')
    try:
        import torch
        import fm
        record['checks']['python'] = sys.version.split()[0]
        record['checks']['torch'] = torch.__version__
        record['checks']['torch_cuda'] = torch.version.cuda
        record['checks']['fm'] = fm.__version__
        record['checks']['fm_file'] = fm.__file__
        if fm.__version__ != cfg['rnafm_package_version']:
            record['errors'].append(f'fm {fm.__version__} != {cfg["rnafm_package_version"]}')
        if args.expect_torch and not torch.__version__.startswith(args.expect_torch):
            record['errors'].append(f'torch {torch.__version__} does not start with {args.expect_torch}')
    except Exception as exc:  # noqa: BLE001
        record['errors'].append(f'import failed: {exc!r}')
    record['ok'] = not record['errors']
    record['seconds'] = round(time.time() - t0, 2)
    text = json.dumps(record, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n')
    print(text)
    sys.exit(0 if record['ok'] else 1)


if __name__ == '__main__':
    main()
