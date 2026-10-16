#!/usr/bin/env python3
"""Wave D preparation step 2: stratified case selection, inputs and static partition.

Deterministic, outcome-blind rules (nothing here looks at model outputs):
  * pool = usable cases from audit_training_cases.py; exact-duplicate sequences
    collapsed to the alphabetically first pdb_name; cases whose three base-pair
    channels are all empty are excluded (degenerate conditioning input);
  * origin strata: PDB chains vs bpRNA_RFAM_* cross-distillation entries (whose
    'native' coordinates are distillation targets, not experimental structures);
  * length bins; bins at/above LONG_BIN_START take every deduplicated case (all
    are PDB chains), shorter bins take a fixed number per origin, evenly spaced
    in (length, pdb_name) order;
  * one calibration anchor, P4P6_ref158 (same input/seeds as jobs 2297413/2298016);
  * refined subset per case: sample IDs 0..R(L)-1, R(L) = nearest power of two to
    REFINE_TARGET_SECONDS / r_est(L), clipped to [R_MIN, 32] (anchor: 32);
  * static LPT partition over 32 nodes x 4 GPUs by estimated per-case GPU-seconds
    (affine in L^2, scaled from the measured 158-nt pilot); per-GPU order is
    descending estimated cost.
Writes case inputs (FASTA + compressed npz), the metadata CSV (byte-identical
CASP16_train rows, original line ending), case_table.json and partition_plan.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import tarfile
from pathlib import Path

import numpy as np

BINS = [30, 50, 75, 100, 130, 160, 200, 250, 300, 400, 500, 600, 801]
# per sampled bin: (PDB chains, bpRNA_RFAM cross-distillation entries)
SAMPLED_BIN_COUNTS = {30: (24, 6), 50: (24, 6), 75: (24, 6), 100: (24, 5), 130: (29, 0), 160: (29, 0)}
LONG_BIN_START = 200
K = 32
TIMESTEPS = 50
NODES = 32
GPUS_PER_NODE = 4
REFINE_TARGET_SECONDS = 900.0
R_MIN = 2
# Planning estimates only (seconds on one GH200 GPU), anchored on the measured
# 158-nt pilot: 56.04 s sampler sum / 32 samples (job 2297413, 1 worker) and
# 969.6 s summed case wall / 32 refined samples (job 2298016).
GEN_A, GEN_B = 0.8, 0.95          # g(L) = GEN_A + GEN_B (L/158)^2 per sample
REF_A, REF_B = 4.0, 26.0          # r(L) = REF_A + REF_B (L/158)^2 per refined sample
CASE_OVERHEAD = 10.0
ANCHOR = 'P4P6_ref158'
CSV_MEMBER = 'RNAbpFlow_training/metadata/CASP16_train.csv'


def g_est(L):
    return GEN_A + GEN_B * (L / 158.0) ** 2


def r_est(L):
    return REF_A + REF_B * (L / 158.0) ** 2


def refine_count(L):
    raw = REFINE_TARGET_SECONDS / r_est(L)
    power = 2 ** int(round(math.log2(raw)))
    return int(min(K, max(R_MIN, power)))


def case_cost(L, R):
    return K * g_est(L) + R * r_est(L) + CASE_OVERHEAD


def sha(data):
    return hashlib.sha256(data).hexdigest()


def origin(pid):
    return 'bpRNA_RFAM_distillation' if pid.startswith('bpRNA_') else 'PDB_chain'


def bin_label(L):
    for lo, hi in zip(BINS, BINS[1:]):
        if lo <= L < hi:
            return f'[{lo},{hi})'
    raise ValueError(L)


def parse_1gid_chain_a(pdb_text, sequence, atom28):
    keys = []; atoms = {}
    for line in pdb_text.splitlines():
        if not line.startswith('ATOM  ') or line[21] != 'A' or line[16].strip() not in ('', 'A'):
            continue
        key = (int(line[22:26]), line[26].strip())
        if key not in atoms:
            keys.append(key); atoms[key] = (line[17:20].strip(), {})
        atoms[key][1][line[12:16].strip().replace('*', "'")] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    keys = keys[:len(sequence)]
    assert ''.join(atoms[k][0] for k in keys) == sequence, 'mapped 1GID chain A differs from P4P6_ref158'
    native = np.zeros((len(sequence), 28, 3)); mask = np.zeros((len(sequence), 28), dtype=bool)
    for i, key in enumerate(keys):
        for name, xyz in atoms[key][1].items():
            if name in atom28:
                native[i, atom28.index(name)] = xyz; mask[i, atom28.index(name)] = True
    return native, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--native-arrays', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--inference-packet', type=Path, required=True)
    parser.add_argument('--native-1gid', type=Path, required=True)
    parser.add_argument('--pilot-1gid-reference', type=Path, required=True)
    parser.add_argument('--packet', type=Path, required=True, help='packet staging directory (created)')
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    import ast
    tree = ast.parse((args.inference_packet / 'RNAbpFlow/src/data/base_constants.py').read_text())
    atom28 = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(getattr(t, 'id', None) == 'atom_types' for t in n.targets))

    usable = {k: v for k, v in audit['cases'].items() if v['usable_for_waveD']}
    by_sequence = {}
    for pid in sorted(usable):
        by_sequence.setdefault(usable[pid]['sequence'], []).append(pid)
    duplicates = {seq: ids for seq, ids in by_sequence.items() if len(ids) > 1}
    pool = []; exclusions = {}
    for seq, ids in by_sequence.items():
        keep = ids[0]
        for other in ids[1:]:
            exclusions[other] = f'exact sequence duplicate of {keep}'
        if max(usable[keep]['ss_channel_pair_counts']) == 0:
            exclusions[keep] = 'no base pairs in any conditioning channel'
            continue
        pool.append(keep)
    selected = []
    for lo, hi in zip(BINS, BINS[1:]):
        members = sorted((usable[p]['length'], p) for p in pool if lo <= usable[p]['length'] < hi)
        if lo >= LONG_BIN_START:
            chosen = [p for _, p in members]
        else:
            chosen = []
            for kind, n in zip(('PDB_chain', 'bpRNA_RFAM_distillation'), SAMPLED_BIN_COUNTS[lo]):
                sub = [m for m in members if origin(m[1]) == kind]
                if n == 0:
                    continue
                idx = sorted(set(int(round(x)) for x in np.linspace(0, len(sub) - 1, n)))
                assert len(idx) == n, (lo, kind)
                chosen += [sub[i][1] for i in idx]
        selected += chosen

    # --- packet inputs --------------------------------------------------
    cases_dir = args.packet / 'cases'
    cases_dir.mkdir(parents=True, exist_ok=False)
    table = []
    for pid in selected:
        info = usable[pid]; L = info['length']
        d = cases_dir / pid; d.mkdir()
        (d / f'{pid}.fasta').write_text(f'>{pid}\n{info["sequence"]}\n')
        arrays = np.load(args.native_arrays / f'{pid}.npz')
        np.savez_compressed(d / 'case_input.npz', ss=arrays['ss'].astype(np.uint8),
                            native_atom28=arrays['native_atom28'], native_mask28=arrays['native_mask28'])
        R = refine_count(L)
        table.append({'case_id': pid, 'length': L, 'length_bin': bin_label(L), 'role': 'training_case_pipeline_QA', 'origin': origin(pid),
                      'csv_membership': info['csv_membership'], 'refined_sample_ids': list(range(R)),
                      'ss_channel_pair_counts': info['ss_channel_pair_counts'], 'ss_channels_identical': info['ss_channels_identical'],
                      'train_pkl': info['train_pkl'], 'feat_pkl': info['feat_pkl'],
                      'estimated_gpu_seconds': case_cost(L, R)})
    # anchor: byte-identical P4P6 inputs, native mapped from 1GID chain A
    anchor_dir = cases_dir / ANCHOR; anchor_dir.mkdir()
    src = args.inference_packet / 'RNAbpFlow/Inputs' / ANCHOR
    shutil.copyfile(src / f'{ANCHOR}.fasta', anchor_dir / f'{ANCHOR}.fasta')
    shutil.copyfile(src / 'annotation_pairs.npy', anchor_dir / 'annotation_pairs.npy')
    sequence = ''.join(x.strip() for x in (src / f'{ANCHOR}.fasta').read_text().splitlines() if not x.startswith('>'))
    pair = np.load(src / 'annotation_pairs.npy')
    assert pair.shape == (158, 158) and set(np.unique(pair)) <= {0.0, 1.0}
    native, mask = parse_1gid_chain_a(args.native_1gid.read_text(), sequence, atom28)
    pilot_ref = np.load(args.pilot_1gid_reference)
    assert np.array_equal(native[:, atom28.index("C4'")], pilot_ref['A_C4prime']), 'C4prime mapping differs from pilot reference'
    np.savez_compressed(anchor_dir / 'case_input.npz', ss=np.stack([pair] * 3, axis=-1).astype(np.uint8),
                        native_atom28=native, native_mask28=mask)
    table.append({'case_id': ANCHOR, 'length': 158, 'length_bin': bin_label(158), 'role': 'calibration_anchor_P4P6', 'origin': 'PDB_chain_1GID_A',
                  'csv_membership': [], 'refined_sample_ids': list(range(K)),
                  'ss_channel_pair_counts': [int(pair.sum() // 2)] * 3, 'ss_channels_identical': True,
                  'anchor_sources': {'fasta_sha256': sha((src / f'{ANCHOR}.fasta').read_bytes()),
                                     'annotation_pairs_sha256': sha((src / 'annotation_pairs.npy').read_bytes()),
                                     'native_1GID_sha256': sha(args.native_1gid.read_bytes())},
                  'anchor_note': ('ID absent from all five RNAbpFlow CSVs, but P4-P6 family structures (e.g. 1l8v_B) are in '
                                  'training; calibration anchor for jobs 2297413/2298016, NOT a held-out case'),
                  'estimated_gpu_seconds': case_cost(158, K)})

    # --- metadata CSV: byte-identical CASP16_train rows -------------------------
    with tarfile.open(args.archive, 'r:gz') as tar:
        for member in tar:
            if member.name == CSV_MEMBER:
                csv_bytes = tar.extractfile(member).read(); break
    lines = csv_bytes.splitlines(keepends=True)
    ending = b'\r\n' if lines[0].endswith(b'\r\n') else b'\n'
    wanted = {r['case_id'] for r in table if r['role'] == 'training_case_pipeline_QA'}
    out_lines = [lines[0]] + [ln for ln in lines[1:] if ln.split(b',', 1)[0].decode() in wanted]
    assert len(out_lines) == len(wanted) + 1
    assert all(ln in lines for ln in out_lines) and all(ln.endswith(ending) for ln in out_lines)
    (args.packet / 'waveD_cases_metadata.csv').write_bytes(b''.join(out_lines))

    # --- static LPT partition ----------------------------------------------------
    slots = [(n, g) for n in range(NODES) for g in range(GPUS_PER_NODE)]
    loads = {s: 0.0 for s in slots}; assign = {s: [] for s in slots}
    for row in sorted(table, key=lambda r: (-r['estimated_gpu_seconds'], r['case_id'])):
        slot = min(slots, key=lambda s: (loads[s], s))
        assign[slot].append(row['case_id']); loads[slot] += row['estimated_gpu_seconds']
        row['node_slot'], row['gpu_slot'] = slot
    assert sorted(c for s in slots for c in assign[s]) == sorted(r['case_id'] for r in table)
    assert all(assign[s] for s in slots), 'every GPU slot must receive work'
    plan = {'plan_version': 'waveD-1', 'nodes': NODES, 'gpus_per_node': GPUS_PER_NODE, 'K': K, 'timesteps': TIMESTEPS,
            'seed_rule': 'seed = 420*(sample_id+1), sample_id 0..K-1 (published RNAbpFlow dataset rule)',
            'refine_rule': {'target_seconds': REFINE_TARGET_SECONDS, 'R_min': R_MIN, 'R_max': K,
                            'definition': 'refine sample IDs 0..R(L)-1; R(L)=nearest power of two to target/r_est(L), clipped; anchor R=32'},
            'estimate_model': {'g_per_sample': f'{GEN_A}+{GEN_B}*(L/158)^2', 'r_per_refined_sample': f'{REF_A}+{REF_B}*(L/158)^2',
                               'case_overhead_seconds': CASE_OVERHEAD,
                               'basis': 'planning-only; scaled from measured 158-nt pilot jobs 2297413/2298016'},
            'case_order_within_gpu': 'descending estimated cost',
            'assignments': {str(n): {str(g): assign[(n, g)] for g in range(GPUS_PER_NODE)} for n in range(NODES)},
            'estimated_gpu_seconds_per_slot': {f'{n}.{g}': round(loads[(n, g)], 1) for n, g in slots},
            'estimated_max_slot_seconds': round(max(loads.values()), 1),
            'estimated_total_gpu_seconds': round(sum(loads.values()), 1)}
    (args.packet / 'partition_plan.json').write_text(json.dumps(plan, indent=1) + '\n')
    selection = {'selection_rules': __doc__, 'bins': BINS, 'sampled_bin_counts': SAMPLED_BIN_COUNTS,
                 'long_bin_start': LONG_BIN_START, 'pool_after_dedupe': len(pool), 'usable': len(usable),
                 'exact_duplicate_sequence_groups': len(duplicates), 'exclusions': exclusions,
                 'selected_training_cases': len(selected), 'anchor': ANCHOR, 'cases': table,
                 'bin_counts_selected': {b: sum(1 for r in table if r['length_bin'] == b) for b in
                                         [f'[{lo},{hi})' for lo, hi in zip(BINS, BINS[1:])]},
                 'bin_counts_pool': {f'[{lo},{hi})': sum(1 for p in pool if lo <= usable[p]['length'] < hi) for lo, hi in zip(BINS, BINS[1:])},
                 'origin_counts_selected': {k: sum(1 for r in table if r['origin'] == k) for k in sorted({r['origin'] for r in table})},
                 'origin_counts_pool': {k: sum(1 for p in pool if origin(p) == k) for k in ('PDB_chain', 'bpRNA_RFAM_distillation')},
                 'origin_note': ('bpRNA_RFAM_* rows exist only in CASP16_train; per the RNAbpFlow README bpRNA-1m(90) was used for '
                                 'cross-distillation, so their reference coordinates are distillation targets, not experimental structures.'),
                 'pool_lengths': sorted(usable[p]['length'] for p in pool),
                 'scope': 'Training cases: lDDT is pipeline QA, not held-out accuracy.'}
    (args.packet / 'case_table.json').write_text(json.dumps(selection, indent=1) + '\n')
    print(json.dumps({k: selection[k] for k in ('pool_after_dedupe', 'selected_training_cases', 'bin_counts_selected', 'origin_counts_selected')}, indent=1))
    print('estimated max slot seconds', plan['estimated_max_slot_seconds'], 'total GPU-s', plan['estimated_total_gpu_seconds'])


if __name__ == '__main__':
    main()
