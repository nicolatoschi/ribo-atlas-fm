#!/usr/bin/env python3
"""Deterministic family/clan-held-out split of the Rfam 15.1 canonical sequence inventory (Wave E).

Inputs come from work/resource_costing/ and are verified against the sha256 values that
rfam15_1_inventory.json recorded on 12 Oct 2026. Clan membership is not in the family
inventory CSV, so it is read from the '#=GF CL' lines of the same hash-verified Rfam.seed.gz.

Grouping. Families are joined into one hold-out group when (a) they belong to the same
Rfam clan or (b) they share an exact sequence (45 sequences carry several memberships).
Whole groups are held out, so no held-out sequence belongs to a training family.

Selection. Groups are visited in the order of sha256(salt|sorted family accessions). A group
is taken unless it would push held-out sequences above MAX_SEQUENCE_FRACTION; selection stops
once held-out families reach TARGET_FAMILY_FRACTION of the families with eligible records.

In-family validation. A small random subset (by hash) of the remaining sequences, kept only
if every family of the sequence still has at least one training sequence. It measures
in-distribution change, as a contrast to the held-out-family change.

Length policy. RNA-FM's learned positions cover 1024 tokens including <cls> and <eos>, so at
most 1022 nucleotides fit. The inventory maximum is 1014 nt, so no sequence is cropped; the
builder asserts this and records it.

Caveat recorded in the summary: RNA-FM was pretrained on RNAcentral, which very likely
contains these Rfam seed sequences. The split controls leakage within Wave E's continued
pretraining only, not with respect to RNA-FM's original corpus.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

SALT = 'RIBO-ATLAS-FM waveE family/clan split v1'
TARGET_FAMILY_FRACTION = 0.05
MAX_SEQUENCE_FRACTION = 0.08
VAL_FRACTION = 0.01
RNAFM_MAX_NUCLEOTIDES = 1022
EXPECTED = {
    'Rfam.seed.gz': '41f014f4ab5628620935f089b1bb77987ebc162fda7c72bff368a32c9cccabac',
    'rfam15_1_canonical_unique.fasta': '3f53f44933a2ed9c79aebbb64f56ac6f79234f65b34854376e3f162c4ec4abbb',
    'rfam15_1_sequence_memberships.jsonl': 'b00482ae24cf1d63a3d4336be4cb8892585de96d799d79852306bfff244ad66d',
    'rfam15_1_family_inventory.csv': '3f424fb5bd93d847b100d595ea36e206bf2c297495eb1fbd92a136abffd77bfa',
}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def unit_hash(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) / float(1 << 64)


def read_clans(path):
    clans, acc, clan = {}, None, None
    with gzip.open(path, 'rt', encoding='latin-1') as lines:
        for line in lines:
            if line.startswith('#=GF AC '):
                acc = line.split()[2]
            elif line.startswith('#=GF CL '):
                clan = line.split()[2]
            elif line.startswith('//'):
                if acc is None:
                    raise ValueError('Stockholm block without accession')
                if clan is not None:
                    clans[acc] = clan
                acc, clan = None, None
    return clans


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def length_stats(values):
    values = sorted(values)
    if not values:
        return {}
    n = len(values)
    return {'n': n, 'min': values[0], 'median': values[n // 2], 'mean': round(sum(values) / n, 3), 'max': values[-1],
            'tokens_incl_cls_eos': sum(values) + 2 * n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True, help='work/resource_costing directory')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    inputs = {name: args.inputs / name for name in EXPECTED}
    observed = {name: sha_file(path) for name, path in inputs.items()}
    for name in EXPECTED:
        assert observed[name] == EXPECTED[name], f'{name}: sha256 {observed[name]} != expected {EXPECTED[name]}'
    inventory = json.loads((args.inputs / 'rfam15_1_inventory.json').read_text())
    assert inventory['source_sha256'] == EXPECTED['Rfam.seed.gz']
    for name, value in inventory['output_sha256'].items():
        assert EXPECTED[name] == value, name

    with open(inputs['rfam15_1_family_inventory.csv'], newline='') as f:
        rows = list(csv.DictReader(f))
    inventory_columns = list(rows[0])
    eligible_families = {r['rfam'] for r in rows if int(r['canonical_30_1024_rows']) > 0}

    sequences, header_families = {}, {}
    with open(inputs['rfam15_1_canonical_unique.fasta']) as f:
        ident = None
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                ident, fam_field = line[1:].split(' ')
                assert fam_field.startswith('families=')
                header_families[ident] = set(fam_field[len('families='):].split(','))
            else:
                assert ident is not None and ident not in sequences
                assert set(line) <= set('ACGU'), ident
                assert sha_bytes(line.encode()) == ident, ident
                sequences[ident] = line
                ident = None
    memberships = {}
    with open(inputs['rfam15_1_sequence_memberships.jsonl']) as f:
        for line in f:
            r = json.loads(line)
            fams = {m['rfam'] for m in r['memberships']}
            assert r['sequence_sha256'] in sequences and r['length'] == len(sequences[r['sequence_sha256']])
            assert fams == header_families[r['sequence_sha256']]
            memberships[r['sequence_sha256']] = fams
    assert set(memberships) == set(sequences)
    families_used = set().union(*memberships.values())
    assert families_used == eligible_families, 'membership families differ from inventory families with eligible rows'
    over = [s for s, seq in sequences.items() if len(seq) > RNAFM_MAX_NUCLEOTIDES]
    assert not over, f'{len(over)} sequences exceed the RNA-FM limit; a crop policy would be needed'

    clans = read_clans(inputs['Rfam.seed.gz'])
    uf = UnionFind(sorted(families_used))
    by_clan = collections.defaultdict(list)
    for fam in sorted(families_used):
        if fam in clans:
            by_clan[clans[fam]].append(fam)
    for members in by_clan.values():
        for fam in members[1:]:
            uf.union(members[0], fam)
    multi = 0
    for fams in memberships.values():
        fams = sorted(fams)
        if len(fams) > 1:
            multi += 1
            for fam in fams[1:]:
                uf.union(fams[0], fam)
    groups = collections.defaultdict(list)
    for fam in sorted(families_used):
        groups[uf.find(fam)].append(fam)
    group_of_family, group_info = {}, {}
    for members in groups.values():
        key = '|'.join(sorted(members))
        gid = 'G' + sha_bytes(key.encode())[:12]
        group_info[gid] = {'families': sorted(members), 'order_key': sha_bytes(f'{SALT}|{key}'.encode()),
                           'clans': sorted({clans[f] for f in members if f in clans})}
        for fam in members:
            group_of_family[fam] = gid
    seq_group = {}
    for s, fams in memberships.items():
        gids = {group_of_family[f] for f in fams}
        assert len(gids) == 1
        seq_group[s] = gids.pop()
    group_sequences = collections.Counter(seq_group.values())

    n_seq, n_fam = len(sequences), len(families_used)
    family_target = math.ceil(TARGET_FAMILY_FRACTION * n_fam)
    seq_cap = math.floor(MAX_SEQUENCE_FRACTION * n_seq)
    held_groups, held_fams, held_seqs, skipped = [], 0, 0, []
    for gid in sorted(group_info, key=lambda g: group_info[g]['order_key']):
        if held_fams >= family_target:
            break
        size = group_sequences[gid]
        if held_seqs + size > seq_cap:
            skipped.append({'group': gid, 'families': len(group_info[gid]['families']), 'sequences': size})
            continue
        held_groups.append(gid)
        held_fams += len(group_info[gid]['families'])
        held_seqs += size
    held_set = set(held_groups)
    held_families = sorted(f for g in held_groups for f in group_info[g]['families'])

    split = {s: ('heldout_family' if seq_group[s] in held_set else 'train') for s in sequences}
    for s in sequences:
        if split[s] == 'heldout_family':
            assert memberships[s] <= set(held_families)
        else:
            assert not (memberships[s] & set(held_families))
    train_count = collections.Counter(f for s in sequences if split[s] == 'train' for f in memberships[s])
    candidates = sorted((unit_hash(f'{SALT}|val|{s}'), s) for s in sequences if split[s] == 'train')
    val_returned = 0
    for u, s in candidates:
        if u >= VAL_FRACTION:
            break
        if all(train_count[f] >= 2 for f in memberships[s]):
            split[s] = 'val_in_family'
            for f in memberships[s]:
                train_count[f] -= 1
        else:
            val_returned += 1

    args.out.mkdir(parents=True, exist_ok=True)
    dataset = args.out / 'waveE_dataset.tsv'
    with open(dataset, 'w', newline='') as f:
        f.write('seq_sha256\tsplit\tlength\tgroup_id\tprimary_family\tfamilies\tclans\tsequence\n')
        for s in sorted(sequences):
            fams = sorted(memberships[s])
            cl = sorted({clans[x] for x in fams if x in clans})
            f.write(f"{s}\t{split[s]}\t{len(sequences[s])}\t{seq_group[s]}\t{fams[0]}\t{','.join(fams)}\t{','.join(cl) or '-'}\t{sequences[s]}\n")

    per_split = {}
    for name in ['train', 'val_in_family', 'heldout_family']:
        ids = sorted(s for s in sequences if split[s] == name)
        fams = set().union(*(memberships[s] for s in ids)) if ids else set()
        per_split[name] = {
            'sequences': len(ids), 'fraction_of_sequences': round(len(ids) / n_seq, 5),
            'families': len(fams), 'groups': len({seq_group[s] for s in ids}),
            'clans': len({clans[f] for f in fams if f in clans}),
            'length': length_stats([len(sequences[s]) for s in ids]),
            'sha256_of_sorted_sequence_ids': sha_bytes('\n'.join(ids).encode()),
        }
    summary = {
        'builder': 'build_waveE_split.py', 'salt': SALT,
        'parameters': {'target_heldout_family_fraction': TARGET_FAMILY_FRACTION, 'max_heldout_sequence_fraction': MAX_SEQUENCE_FRACTION,
                       'val_in_family_fraction': VAL_FRACTION, 'rnafm_max_nucleotides': RNAFM_MAX_NUCLEOTIDES},
        'inputs_sha256': observed,
        'inputs_note': 'verified against rfam15_1_inventory.json (12 Oct 2026); family inventory CSV columns ' + ','.join(inventory_columns) + ' (no clan column, so clans are parsed from #=GF CL lines of Rfam.seed.gz)',
        'sequences': n_seq, 'families_with_eligible_records': n_fam,
        'families_in_a_clan': sum(1 for f in families_used if f in clans), 'clans_present': len(by_clan),
        'sequences_with_multiple_family_memberships': multi,
        'groups_total': len(group_info), 'largest_group_sequences': max(group_sequences.values()),
        'heldout_family_target': family_target, 'heldout_sequence_cap': seq_cap,
        'heldout_groups': [{'group': g, 'families': group_info[g]['families'], 'clans': group_info[g]['clans'], 'sequences': group_sequences[g]} for g in held_groups],
        'groups_skipped_by_sequence_cap': skipped,
        'val_candidates_returned_to_train_to_keep_family_seen': val_returned,
        'heldout_families': held_families,
        'splits': per_split,
        'length_policy': f'no cropping: max length {max(len(x) for x in sequences.values())} nt <= {RNAFM_MAX_NUCLEOTIDES} nt (1024 learned positions incl. <cls> and <eos>); the trainer refuses sequences above the limit',
        'leakage_controls': ['whole clans held out together', 'families sharing an exact sequence held out together',
                             'no held-out sequence belongs to any training family (asserted)'],
        'limitations': ['no sequence-identity (e.g. MMseqs2/cd-hit) filter across unrelated clans/families',
                        'RNA-FM was pretrained on RNAcentral, which very likely contains these Rfam seed sequences: the split controls leakage within Wave E continued pretraining only'],
        'outputs_sha256': {'waveE_dataset.tsv': sha_file(dataset)},
    }
    (args.out / 'waveE_split_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ['sequences', 'families_with_eligible_records', 'groups_total', 'heldout_family_target', 'outputs_sha256']}, indent=2))
    print(json.dumps({k: {x: v[x] for x in ['sequences', 'families', 'groups', 'clans']} for k, v in per_split.items()}, indent=2))


if __name__ == '__main__':
    main()
