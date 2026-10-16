#!/usr/bin/env python3
"""Wave D preparation step 1: audit every RNAbpFlow training case (read-only).

Streams RNAbpFlow-Train.tar.gz (never extracts it to the project tree), reads the
five metadata CSVs as bytes, applies the upstream PDBNABaseDataset filters
(homomer, num_protein_chains==0, 30<=modeled_na_seq_len<=800), requires both
Feature/train_pkl/<id>/<id>.pkl and pkls/<id>_base/<id>_feat.pkl, and opens the
pickles with a RESTRICTED unpickler (numpy arrays only; torch tensors are either
decoded through torch.load(weights_only=True) or replaced by metadata stubs).
No archive code is imported or executed.

Output: one JSON with per-case checks, hashes, sequence, base-pair statistics
and native-backbone completeness, used by select_and_build_cases.py.
Role of these cases: they are RNAbpFlow TRAINING cases, so any accuracy metric
computed on them is pipeline QA, never held-out accuracy.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np

CSV_SETS = ('RNA3DB_train', 'CASP15_train', 'CASP16_train', 'CASP15_FT', 'CASP16_FT')
ROOT = 'RNAbpFlow_training/'
# Training vocabulary (src/data/vocabulary.py): 21 a, 22 c, 23 g, 24 t, 25 u, 26 x.
TRAIN_AATYPE_TO_BASE = {21: 'A', 22: 'C', 23: 'G', 25: 'U'}
# Inference get_one_hot ordering (src/data/data_transform.py): A0 U1 G2 C3.
INFERENCE_ONEHOT_ORDER = 'AUGC'


class TensorStub:
    def __init__(self, shape, stride):
        self.shape = tuple(shape)
        self.stride = tuple(stride)


def _decode_storage(data):
    import torch
    return torch.load(io.BytesIO(data), weights_only=True, map_location='cpu')


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *args):
    import torch
    if isinstance(storage, TensorStub) or storage is None:
        return TensorStub(size, stride)
    tensor = torch.empty((0,), dtype=storage.dtype)
    tensor.set_(storage, storage_offset, size, stride)
    return tensor


class RestrictedUnpickler(pickle.Unpickler):
    """Only numpy reconstruction, containers and (optionally) torch storages."""

    def __init__(self, handle, decode_torch):
        super().__init__(handle)
        self.decode_torch = decode_torch

    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            import collections
            return collections.OrderedDict
        if (module, name) == ('collections', 'defaultdict'):
            import collections
            return collections.defaultdict
        if (module, name) == ('builtins', 'list'):
            return list
        if (module, name) == ('numpy', 'dtype'):
            return np.dtype
        if (module, name) == ('numpy', 'ndarray'):
            return np.ndarray
        if module in ('numpy.core.numeric', 'numpy._core.numeric') and name == '_frombuffer':
            from numpy._core import numeric
            return numeric._frombuffer
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name in ('_reconstruct', 'scalar'):
            from numpy._core import multiarray
            return getattr(multiarray, name)
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return _rebuild_tensor_v2
        if (module, name) == ('torch.storage', '_load_from_bytes'):
            if self.decode_torch:
                return _decode_storage
            return lambda data: TensorStub((len(data),), (1,))
        raise pickle.UnpicklingError(f'Blocked global {module}.{name}')


def restricted_load(data, decode_torch=False):
    return RestrictedUnpickler(io.BytesIO(data), decode_torch).load()


def literal_assignment(path, name):
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit_train_pkl(data, row, compact_names, atom28):
    L = int(row['modeled_na_seq_len'])
    feats = restricted_load(data)
    out = {'train_pkl_fields': sorted(feats.keys())}
    problems = []
    aatype = np.asarray(feats['aatype'])
    onehot = np.asarray(feats['onehot'], dtype=float)
    ss = np.asarray(feats['ss'])
    positions = np.asarray(feats['atom_positions'], dtype=float)
    mask = np.asarray(feats['atom_mask'], dtype=float)
    modeled = np.asarray(feats['modeled_idx'])
    bb = np.asarray(feats['bb_mask'], dtype=float)
    if aatype.shape != (L,):
        problems.append(f'aatype shape {aatype.shape} != ({L},)')
    unknown = sorted({int(x) for x in aatype.tolist()} - set(TRAIN_AATYPE_TO_BASE))
    if unknown:
        problems.append(f'non-AUGC training aatype codes {unknown}')
        sequence = None
    else:
        sequence = ''.join(TRAIN_AATYPE_TO_BASE[int(x)] for x in aatype)
    if onehot.shape != (L, 4) or not np.all((onehot == 0) | (onehot == 1)) or not np.all(onehot.sum(1) == 1):
        problems.append('onehot is not an exact Lx4 one-hot')
        onehot_order_matches_inference = False
    else:
        onehot_sequence = ''.join(INFERENCE_ONEHOT_ORDER[i] for i in onehot.argmax(1))
        onehot_order_matches_inference = sequence is not None and onehot_sequence == sequence
        if not onehot_order_matches_inference:
            problems.append('training onehot differs from inference get_one_hot(A,U,G,C) of aatype sequence')
    if ss.shape != (L, L, 3):
        problems.append(f'ss shape {ss.shape} != ({L},{L},3)')
        ss_ok = False
    else:
        ss_ok = bool(np.all((ss == 0) | (ss == 1)))
        sym = all(np.array_equal(ss[..., c], ss[..., c].T) for c in range(3))
        diag = all(int(np.trace(ss[..., c])) == 0 for c in range(3))
        if not (ss_ok and sym and diag):
            problems.append('ss not binary/symmetric/zero-diagonal')
            ss_ok = False
    if modeled.shape != (L,) or not np.array_equal(modeled, np.arange(L)):
        problems.append('modeled_idx is not 0..L-1')
    if bb.shape != (L,) or not np.all(bb == 1):
        problems.append('bb_mask not all valid')
    if positions.shape != (L, 23, 3) or mask.shape != (L, 23):
        problems.append('atom_positions/atom_mask shape mismatch')
    native28 = np.zeros((L, 28, 3)); native28_mask = np.zeros((L, 28), dtype=bool)
    if sequence is not None and positions.shape == (L, 23, 3):
        for i, base in enumerate(sequence):
            for j, name in enumerate(compact_names[base]):
                if name and mask[i, j] > 0:
                    native28[i, atom28.index(name)] = positions[i, j]
                    native28_mask[i, atom28.index(name)] = True
        if not np.isfinite(native28[native28_mask]).all():
            problems.append('nonfinite native coordinates')
    c4 = native28_mask[:, atom28.index("C4'")]
    p = native28_mask[:, atom28.index('P')]
    o3 = native28_mask[:, atom28.index("O3'")]
    if not c4.all():
        problems.append(f'native C4prime missing for {int((~c4).sum())} residues')
    channel_pairs = [int(ss[..., c].sum() // 2) for c in range(3)] if ss.ndim == 3 and ss.shape[-1] == 3 else None
    out.update({
        'sequence': sequence,
        'onehot_order_matches_inference_AUGC': bool(onehot_order_matches_inference),
        'ss_valid_binary_symmetric_zero_diagonal': bool(ss_ok),
        'ss_channel_pair_counts': channel_pairs,
        'ss_channels_identical': bool(ss.ndim == 3 and np.array_equal(ss[..., 0], ss[..., 1]) and np.array_equal(ss[..., 0], ss[..., 2])),
        'ss_max_partners_per_residue': int(ss.sum(axis=1).max()) if ss.ndim == 3 else None,
        'custom_ss_mask_sum': int(np.asarray(feats.get('custom_ss_mask', np.zeros(1))).sum()),
        'has_ss_pred': 'ss_pred' in feats,
        'native_C4prime_present': int(c4.sum()), 'native_P_present': int(p.sum()), 'native_O3prime_present': int(o3.sum()),
        'native_heavy_atoms_present': int(native28_mask.sum()),
        'native_heavy_atoms_expected': int(sum(len([n for n in compact_names[b] if n]) for b in sequence)) if sequence else None,
    })
    return out, problems, (native28, native28_mask, ss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--inference-base-constants', type=Path, required=True,
                        help='RNAbpFlow inference src/data/base_constants.py (parsed with ast, not imported)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--native-out', type=Path, default=None,
                        help='optional directory for per-case native/ss arrays (.npz) of usable cases')
    args = parser.parse_args()
    archive_sha = hashlib.sha256()
    with open(args.archive, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            archive_sha.update(block)
    csv_bytes = {}; rows = {}; membership = {}; compact = None
    atom28 = literal_assignment(args.inference_base_constants, 'atom_types')
    train_records = {}; feat_records = {}
    if args.native_out:
        args.native_out.mkdir(parents=True, exist_ok=True)
    # Single streaming pass; member order is metadata, src, pkls/, Feature/ (checked).
    with tarfile.open(args.archive, 'r:gz') as tar:
        for member in tar:
            name = member.name
            if not member.isfile():
                continue
            if name.startswith(ROOT + 'metadata/') and name.endswith('.csv'):
                csv_bytes[Path(name).stem] = tar.extractfile(member).read()
                continue
            if name == ROOT + 'src/data/nucleotide_constants.py':
                scratch_constants = args.out.parent / 'nucleotide_constants_from_archive.py'
                scratch_constants.write_bytes(tar.extractfile(member).read())
                compact = literal_assignment(scratch_constants, 'restype_name_to_compact_atom_names')
                continue
            if name.startswith(ROOT + 'pkls/') and name.endswith('_feat.pkl'):
                data = tar.extractfile(member).read()
                pid = Path(name).parent.name[:-len('_base')]
                record = {'member': name, 'sha256': sha(data), 'bytes': len(data)}
                try:
                    feat = restricted_load(data, decode_torch=True)
                    aatype = np.asarray(feat['aatype']).reshape(-1)
                    # Companion pickles use the inference order A0 G1 C2 U3 (checked on 1e8o_E).
                    record['sequence'] = ''.join('AGCU'[int(x)] if 0 <= int(x) < 4 else '?' for x in aatype)
                except Exception as error:
                    record['error'] = f'{type(error).__name__}: {error}'
                feat_records[pid] = record
                continue
            if name.startswith(ROOT + 'Feature/train_pkl/') and name.endswith('.pkl'):
                if not rows:
                    for set_name in CSV_SETS:
                        for row in csv.DictReader(io.StringIO(csv_bytes[set_name].decode())):
                            rows.setdefault(row['pdb_name'], {})[set_name] = row
                            membership.setdefault(row['pdb_name'], []).append(set_name)
                pid = Path(name).stem
                data = tar.extractfile(member).read()
                record = {'member': name, 'sha256': sha(data), 'bytes': len(data)}
                if pid in rows:
                    row = rows[pid].get('CASP16_train') or next(iter(rows[pid].values()))
                    try:
                        info, issues, arrays = audit_train_pkl(data, row, compact, atom28)
                        record['info'] = info; record['issues'] = issues
                        if args.native_out and not issues:
                            native28, native28_mask, ss = arrays
                            np.savez_compressed(args.native_out / f'{pid}.npz', native_atom28=native28,
                                                native_mask28=native28_mask, ss=ss.astype(np.uint8))
                    except Exception as error:
                        record['issues'] = [f'train pkl restricted read failed: {type(error).__name__}: {error}']
                train_records[pid] = record
    csv_info = {name: {'sha256': sha(b), 'bytes': len(b), 'CRLF_lines': b.count(b'\r\n'), 'LF_lines': b.count(b'\n'),
                       'line_ending': 'CRLF' if b.count(b'\r\n') == b.count(b'\n') else 'LF' if b.count(b'\r\n') == 0 else 'mixed'}
                for name, b in csv_bytes.items()}
    cases = {}
    for pid in sorted(rows):
        sets = rows[pid]
        lengths = {r['modeled_na_seq_len'] for r in sets.values()}
        row = sets.get('CASP16_train') or next(iter(sets.values()))
        L = int(row['modeled_na_seq_len'])
        eligible = (row['quaternary_category'] == 'homomer' and int(row['num_protein_chains']) == 0 and 30 <= L <= 800)
        record = {'pdb_name': pid, 'length': L, 'csv_membership': membership[pid],
                  'length_consistent_across_csvs': len(lengths) == 1, 'upstream_loader_filter_passed': eligible,
                  'train_pkl_present': pid in train_records, 'feat_pkl_present': pid in feat_records}
        problems = []
        if not (eligible and pid in train_records and pid in feat_records):
            problems.append('not eligible under upstream filter + both-pickle rule')
        else:
            t = train_records[pid]; f = feat_records[pid]
            record['train_pkl'] = {k: t[k] for k in ('member', 'sha256', 'bytes')}
            record['feat_pkl'] = {k: f[k] for k in ('member', 'sha256', 'bytes')}
            record.update(t.get('info', {})); problems += t.get('issues', [])
            if 'error' in f:
                problems.append('companion pkl restricted read failed: ' + f['error'])
                record['feat_pkl_sequence_matches'] = None
            else:
                record['feat_pkl_sequence_matches'] = f['sequence'] == record.get('sequence')
                if not record['feat_pkl_sequence_matches']:
                    problems.append('companion _feat.pkl aatype sequence differs')
        record['problems'] = problems
        record['usable_for_waveD'] = not problems
        cases[pid] = record
    result = {'archive': str(args.archive), 'archive_sha256': archive_sha.hexdigest(),
              'csv_files': csv_info, 'train_pkl_count': len(train_records), 'feat_pkl_count': len(feat_records),
              'unique_ids_in_csvs': len(rows), 'usable_count': sum(c['usable_for_waveD'] for c in cases.values()),
              'scope': 'All listed cases are RNAbpFlow training metadata; metrics on them are pipeline QA only, not held-out accuracy.',
              'cases': cases}
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=1))


if __name__ == '__main__':
    main()
