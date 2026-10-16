#!/usr/bin/env python3
"""Inventory bounded Rfam15.1 seed data; preserve family membership and hashes.

This is a pretraining-source inventory, not a leakage-cleared or paired
experimental benchmark. Stockholm alignment strings are concatenated across
blocks, gap characters removed, T converted to U, and ambiguous bases excluded.
Exact sequence duplicates retain all original Rfam memberships.
"""
from pathlib import Path
import csv,gzip,hashlib,json,statistics

ROOT=Path(__file__).resolve().parent

def sha(data):return hashlib.sha256(data).hexdigest()

def families(path):
    current={'accession':None,'name':None,'sequences':{}}
    with gzip.open(path,'rt',encoding='latin-1') as lines:
        for line in lines:
            line=line.rstrip('\n')
            if line.startswith('#=GF AC '):current['accession']=line.split()[2]
            elif line.startswith('#=GF ID '):current['name']=line.split()[2]
            elif line=='//':
                if current['accession'] is None:raise ValueError('Missing family accession')
                yield current
                current={'accession':None,'name':None,'sequences':{}}
            elif line and not line.startswith('#'):
                parts=line.split()
                if len(parts)==2:current['sequences'].setdefault(parts[0],[]).append(parts[1])

raw_families=0;rows=0;ambiguous=0;short=0;long=0
unique={};family_stats=[]
for fam in families(ROOT/'Rfam.seed.gz'):
    raw_families+=1;valid=0;nt=0
    for source,strings in fam['sequences'].items():
        rows+=1
        seq=''.join(strings).upper().replace('.','').replace('-','').replace('T','U')
        if set(seq)-set('ACGU'):
            ambiguous+=1;continue
        if len(seq)<30:short+=1;continue
        if len(seq)>1024:long+=1;continue
        valid+=1;nt+=len(seq);ident=sha(seq.encode())
        rec=unique.setdefault(ident,{'sequence':seq,'length':len(seq),'memberships':[]})
        rec['memberships'].append({'rfam':fam['accession'],'family_name':fam['name'],'source_sequence_region':source})
    family_stats.append({'rfam':fam['accession'],'name':fam['name'],'seed_rows':len(fam['sequences']),'canonical_30_1024_rows':valid,'canonical_residue_occurrences':nt})
vals=[r['length'] for r in unique.values()]
counts={}
for r in unique.values():
    for member in r['memberships']:
        counts.setdefault(member['rfam'],set()).add(sha(r['sequence'].encode()))
ordered=sorted(unique.items())
with (ROOT/'rfam15_1_canonical_unique.fasta').open('w') as out:
    for ident,r in ordered:
        fids=','.join(sorted({x['rfam'] for x in r['memberships']}))
        out.write(f'>{ident} families={fids}\n{r["sequence"]}\n')
with (ROOT/'rfam15_1_sequence_memberships.jsonl').open('w') as out:
    for ident,r in ordered:out.write(json.dumps({'sequence_sha256':ident,'length':r['length'],'memberships':r['memberships']})+'\n')
with (ROOT/'rfam15_1_family_inventory.csv').open('w',newline='') as out:
    w=csv.DictWriter(out,fieldnames=list(family_stats[0]));w.writeheader();w.writerows(family_stats)
summary={'source':'Rfam CURRENT release15.1, downloaded12October2026','source_url':'https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz','source_sha256':sha((ROOT/'Rfam.seed.gz').read_bytes()),'source_rights':'CC0, supplied COPYING included',
    'families_in_source':raw_families,'source_sequence_rows':rows,'excluded_ambiguous_rows':ambiguous,'excluded_length_below30':short,'excluded_length_above1024':long,
    'eligible_source_occurrence_rows':rows-ambiguous-short-long,'exact_unique_sequences':len(unique),'unique_nucleotide_tokens':sum(vals),'length_min':min(vals),'length_median':statistics.median(vals),'length_mean':statistics.mean(vals),'length_max':max(vals),
    'families_with_eligible_records':len(counts),'exact_sequences_with_multiple_family_memberships':sum(len({m['rfam'] for m in r['memberships']})>1 for r in unique.values()),
    'length_bins':{f'{lo}-{hi}':{'unique_sequences':sum(lo<=n<=hi for n in vals),'tokens':sum(n for n in vals if lo<=n<=hi)} for lo,hi in [(30,128),(129,256),(257,512),(513,1024)]},
    'not_yet_admitted_to_final_training':'Family/clan/near-identity exclusions and final test overlap controls must be applied; no train/blind split made here','scope':'Sequence-source inventory only; no measured binding/probe labels, independent biological efficacy cohort, or model training',
    'output_sha256':{n:sha((ROOT/n).read_bytes()) for n in ['rfam15_1_canonical_unique.fasta','rfam15_1_sequence_memberships.jsonl','rfam15_1_family_inventory.csv']}}
(ROOT/'rfam15_1_inventory.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
