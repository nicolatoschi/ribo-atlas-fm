"""Stream real RDAT into an auditable, local-only SQLite index; export compact evidence.

Candidate means canonical full construct sequence plus the recorded environmental
condition. It does not mean independent family, quality-admitted or split-admitted.
Chemistry, processing, construct IDs and all annotation tokens remain separate.
No universal connected-component or novelty assignment is made here.
"""
import collections
import csv
import hashlib
import json
import math
import pathlib
import re
import sqlite3

BASE = pathlib.Path(__file__).resolve().parent
SELECTION = json.loads((BASE / 'selected_sources.json').read_text())

def array_stats(tokens):
    values = [float(t) for t in tokens]
    # These assets encode missing observations as NaN. Every finite negative
    # background-subtracted value remains observed; zero is never absent.
    valid = [math.isfinite(v) for v in values]
    return len(values), sum(valid), sum(v < 0 and ok for v, ok in zip(values, valid)), sum(v == 0 and ok for v, ok in zip(values, valid)), sum(v >= 0 and ok for v, ok in zip(values, valid))

def parse_annotation(text):
    fields = text.strip().split('\t')
    seq = next((t.split(':', 1)[1] for t in fields if t.startswith('sequence:')), '').upper().replace('T', 'U')
    read = next((t.split(':', 1)[1] for t in fields if t.startswith('reads:')), None)
    snr = next((t.rsplit(':', 1)[1] for t in fields if t.startswith('signal_to_noise:')), None)
    name = next((t.split(':', 1)[1] for t in fields if t.startswith('name:') or t.startswith('Eterna:design_name:')), '')
    if name.startswith('design_name:'):
        name = name.split(':', 1)[1]
    source_id = next((t.split('Eterna:id:', 1)[1] for t in fields if t.startswith('Eterna:id:')), None)
    bad = int(any(t.startswith('warning:') for t in fields))
    try: reads = int(read)
    except (TypeError, ValueError): reads = None
    try: signal = float(snr)
    except (TypeError, ValueError): signal = None
    return seq, reads, signal, name, source_id, bad, json.dumps(fields, ensure_ascii=False)

def main():
    manifest = json.loads((BASE / 'download_manifest.json').read_text())
    verified = {a['rmdb_id'] for a in manifest['assets'] if a['status'] == 'verified' and a['sha256_verified'] and a['size_verified']}
    dbpath = BASE / 'raw' / 'admission_audit.sqlite'
    if dbpath.exists(): dbpath.unlink()
    con = sqlite3.connect(dbpath)
    con.execute('PRAGMA journal_mode=OFF')
    con.execute('PRAGMA synchronous=OFF')
    con.execute('CREATE TABLE records(source TEXT, group_id TEXT, chemistry TEXT, record_id INTEGER, sequence TEXT, condition_key TEXT, length INTEGER, canonical INTEGER, reads INTEGER, snr REAL, name TEXT, eterna_id TEXT, source_warning INTEGER, annotations TEXT, nvalues INTEGER, finite_values INTEGER, negative_values INTEGER, zero_values INTEGER, nerrors INTEGER, finite_errors INTEGER, nonnegative_errors INTEGER, PRIMARY KEY(source,record_id))')
    sources = []
    for entry in SELECTION['entries']:
        source = entry['rmdb_id']
        if source not in verified:
            print('SKIP unverified', source, flush=True)
            continue
        group, chemistry, _ = source.split('_')
        condition = json.dumps({'chemical_conditions':sorted(entry['chemical_conditions']), 'temperature':sorted(entry['temperature'])}, sort_keys=True, separators=(',', ':'))
        counts = collections.Counter()
        headers = {}
        annotations, values, errors = [], [], []
        seqpos = []
        with (BASE / 'raw' / (source + '.rdat')).open() as stream:
            for line in stream:
                parts = line.rstrip('\n').split(None, 1)
                if not parts: continue
                key = parts[0]
                text = parts[1] if len(parts) > 1 else ''
                kind = key.split(':', 1)[0]
                counts[kind] += 1
                if kind == 'ANNOTATION_DATA':
                    rid = int(key.split(':')[1])
                    seq, reads, snr, name, eid, warning, fields = parse_annotation(text)
                    annotations.append((source, group, chemistry, rid, seq, condition, len(seq), int(bool(seq) and set(seq) <= set('ACGU')), reads, snr, name, eid, warning, fields))
                    if len(annotations) >= 5000:
                        con.executemany('INSERT INTO records(source,group_id,chemistry,record_id,sequence,condition_key,length,canonical,reads,snr,name,eterna_id,source_warning,annotations) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', annotations)
                        annotations.clear()
                elif kind in ('REACTIVITY', 'REACTIVITY_ERROR'):
                    if annotations:
                        con.executemany('INSERT INTO records(source,group_id,chemistry,record_id,sequence,condition_key,length,canonical,reads,snr,name,eterna_id,source_warning,annotations) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', annotations)
                        annotations.clear()
                    rid = int(key.split(':')[1])
                    n, finite, negative, zero, nonneg = array_stats(text.split())
                    if kind == 'REACTIVITY':
                        values.append((n, finite, negative, zero, source, rid))
                        if len(values) >= 5000:
                            con.executemany('UPDATE records SET nvalues=?,finite_values=?,negative_values=?,zero_values=? WHERE source=? AND record_id=?', values)
                            values.clear()
                    else:
                        errors.append((n, finite, nonneg, source, rid))
                        if len(errors) >= 5000:
                            con.executemany('UPDATE records SET nerrors=?,finite_errors=?,nonnegative_errors=? WHERE source=? AND record_id=?', errors)
                            errors.clear()
                elif kind == 'SEQPOS':
                    seqpos = [int(re.search(r'-?\d+$', t).group()) for t in text.split()]
                elif kind not in ('COMMENT',):
                    headers[kind] = text
        if annotations: con.executemany('INSERT INTO records(source,group_id,chemistry,record_id,sequence,condition_key,length,canonical,reads,snr,name,eterna_id,source_warning,annotations) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', annotations)
        if values: con.executemany('UPDATE records SET nvalues=?,finite_values=?,negative_values=?,zero_values=? WHERE source=? AND record_id=?', values)
        if errors: con.executemany('UPDATE records SET nerrors=?,finite_errors=?,nonnegative_errors=? WHERE source=? AND record_id=?', errors)
        con.commit()
        coverage = con.execute('SELECT count(*),sum(canonical),sum(source_warning),sum(nvalues IS NULL),sum(nerrors IS NULL),sum(finite_values),sum(finite_errors),sum(negative_values),sum(zero_values),sum(nonnegative_errors),min(length),max(length),sum(CASE WHEN nvalues != ? OR nerrors != ? THEN 1 ELSE 0 END) FROM records WHERE source=?', (len(seqpos),len(seqpos),source)).fetchone()
        result = dict(zip(('construct_profile_records','canonical_records','source_warning_records','missing_profile_records','missing_error_records','finite_observed_values','finite_reported_errors','negative_observed_values','zero_observed_values','nonnegative_reported_errors','minimum_full_construct_length','maximum_full_construct_length','profile_error_length_mismatch_records'), coverage))
        result.update({'source':source, 'reported_construct_occurrences_from_metadata':entry['reported_construct_occurrences'], 'actual_RDAT_annotation_lines':counts['ANNOTATION_DATA'],'RDAT_counts':dict(counts),'seqpos_1based':seqpos, 'header':headers, 'environment_condition_key':json.loads(condition)})
        sources.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('header','seqpos_1based')}), flush=True)
    con.execute('CREATE INDEX candidate_key ON records(sequence,condition_key)')
    con.execute('CREATE TABLE candidates AS SELECT sequence,condition_key,length,count(*) AS observation_record_count,count(DISTINCT group_id) AS source_group_count,count(DISTINCT chemistry) AS chemistry_count,max(CASE WHEN chemistry="DMS" AND source_warning=0 AND finite_values=nvalues AND finite_errors=nerrors AND nonnegative_errors=nerrors THEN 1 ELSE 0 END) AS has_complete_source_unflagged_DMS,max(CASE WHEN chemistry="2A3" AND source_warning=0 AND finite_values=nvalues AND finite_errors=nerrors AND nonnegative_errors=nerrors THEN 1 ELSE 0 END) AS has_complete_source_unflagged_2A3 FROM records WHERE canonical=1 AND nvalues>0 AND finite_values>0 GROUP BY sequence,condition_key')
    totals = {}
    queries = {
      'actual_construct_chemistry_profile_records':'SELECT count(*) FROM records',
      'exact_unique_sequence_condition_records_before_missing_data_filter':'SELECT count(*) FROM (SELECT sequence,condition_key FROM records WHERE canonical=1 GROUP BY sequence,condition_key)',
      'canonical_construct_chemistry_profile_records':'SELECT count(*) FROM records WHERE canonical=1',
      'unique_sequence_condition_candidates':'SELECT count(*) FROM candidates',
      'unique_full_construct_sequences':'SELECT count(DISTINCT sequence) FROM candidates',
      'both_chemistry_candidates':'SELECT count(*) FROM candidates WHERE chemistry_count=2',
      'both_chemistry_complete_source_unflagged_candidates':'SELECT count(*) FROM candidates WHERE has_complete_source_unflagged_DMS=1 AND has_complete_source_unflagged_2A3=1',
      'at_least_one_chemistry_complete_source_unflagged_candidates':'SELECT count(*) FROM candidates WHERE has_complete_source_unflagged_DMS=1 OR has_complete_source_unflagged_2A3=1',
      'candidate_duplicates_across_source_groups':'SELECT count(*) FROM candidates WHERE source_group_count>1',
      'repeat_observation_records_beyond_one_per_chemistry':'SELECT sum(observation_record_count-chemistry_count) FROM candidates',
    }
    for key, query in queries.items(): totals[key] = con.execute(query).fetchone()[0]
    paired = con.execute('SELECT count(*),sum(a.sequence!=b.sequence),sum(a.condition_key!=b.condition_key) FROM records a JOIN records b ON a.group_id=b.group_id AND a.record_id=b.record_id AND b.chemistry="DMS" WHERE a.chemistry="2A3"').fetchone()
    totals.update(dict(zip(('one_to_one_deposited_construct_chemistry_pairs','paired_record_sequence_mismatches','paired_record_condition_mismatches'),paired)))
    totals['final_split_admitted_candidates'] = None
    totals['independent_RNA_family_count'] = None
    distribution = [{'full_construct_length_nt':r[0], 'unique_sequence_condition_candidates':r[1]} for r in con.execute('SELECT length,count(*) FROM candidates GROUP BY length ORDER BY length')]
    with (BASE / 'actual_length_distribution.csv').open('w') as out:
        writer = csv.DictWriter(out, fieldnames=['full_construct_length_nt','unique_sequence_condition_candidates'])
        writer.writeheader(); writer.writerows(distribution)
    groups = []
    for group in ('OK2TRN','OK1LIB','M2PK90','RFMREP'):
        groups.append({'source_group':group, 'construct_chemistry_profile_records':con.execute('SELECT count(*) FROM records WHERE group_id=?',(group,)).fetchone()[0], 'unique_sequence_condition_candidates_within_group':con.execute('SELECT count(*) FROM (SELECT sequence,condition_key FROM records WHERE group_id=? AND canonical=1 AND nvalues>0 AND finite_values>0 GROUP BY sequence,condition_key)',(group,)).fetchone()[0]})
    sample_queries = con.execute('SELECT min(r.sequence),r.condition_key FROM records r JOIN candidates c ON r.sequence=c.sequence AND r.condition_key=c.condition_key WHERE c.has_complete_source_unflagged_DMS=1 AND c.has_complete_source_unflagged_2A3=1 AND r.group_id="M2PK90" AND r.chemistry="2A3" AND r.reads>=1000 AND r.snr>=1 AND lower(r.name) NOT LIKE "%virus%" AND lower(r.name) NOT LIKE "%sars%" AND lower(r.name) NOT LIKE "%hiv%" GROUP BY substr(r.name,1,instr(r.name,"_")-1),r.condition_key ORDER BY substr(r.name,1,instr(r.name,"_")-1) LIMIT 4').fetchall()
    samples = []
    sample_ids = {}
    for sequence, condition in sample_queries:
        record_data = con.execute('SELECT source,record_id,chemistry,reads,snr,annotations FROM records WHERE sequence=? AND condition_key=? AND source_warning=0 ORDER BY source,record_id', (sequence,condition)).fetchall()
        sample = {'candidate_key_SHA256':hashlib.sha256((sequence+'\n'+condition).encode()).hexdigest(),'sequence':sequence,'environment_condition':json.loads(condition),'status':'true observed probe engineering fixture; not split-admitted, independent biology or an unseen-family test','observations':[]}
        for source,rid,chemistry,reads,snr,annotation in record_data:
            obs = {'source':source,'record_id':rid,'chemistry':chemistry,'reads':reads,'signal_to_noise_reported':snr,'annotation_tokens':json.loads(annotation)}
            sample['observations'].append(obs)
            sample_ids[(source,rid)] = obs
        samples.append(sample)
    for source in {key[0] for key in sample_ids}:
        positions = next(s['seqpos_1based'] for s in sources if s['source']==source)
        with (BASE / 'raw' / (source+'.rdat')).open() as stream:
            for line in stream:
                parts = line.split(None,1)
                if len(parts) != 2: continue
                key, text = parts
                kind = key.split(':',1)[0]
                if kind not in ('REACTIVITY','REACTIVITY_ERROR'): continue
                obs = sample_ids.get((source,int(key.split(':')[1])))
                if obs is None: continue
                vals = [float(t) for t in text.split()]
                obs['seqpos_1based'] = positions
                obs['reactivity' if kind=='REACTIVITY' else 'reported_error'] = vals
                if kind=='REACTIVITY': obs['observed_mask'] = [math.isfinite(v) for v in vals]
    (BASE / 'true_observed_probe_sample.json').write_text(json.dumps({'samples':samples,'source_integrity':'Every source file matched release SHA-256 and byte count','mask_definition':'finite; these assets encode missing values as NaN, while zero and every finite negative normalized value remain observed','lineage':'All annotation tokens retained; samples may be parent/mutant-related'},indent=2,ensure_ascii=False,allow_nan=False))
    evidence = {'scope':'Four selected RNA-only groups; VIRWIN viral-window deposits explicitly excluded', 'definition':'Exact canonical full-construct sequence plus recorded chemical environment and temperature, merging chemistry as separate observations, not averaging profiles or erasing IDs/lineage', 'totals':totals,'groups':groups,'sources':sources,'length_distribution':distribution,'source_sha256_verified':len(sources),'limitations':['Candidate deduplication does not establish independent families or parent/mutant/window lineage separation.','No legacy model/prior sequence exposure or public pseudolabel/template audit has been completed for these candidates.','Final task-specific training/development/blind admission remains prospective.','Unflagged source records are not an assay-independent scientific quality certificate.','Two chemistry views of the same construct are paired observations, not two independent RNAs.']}
    (BASE / 'actual_admission_evidence.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False))
    con.commit(); con.close()
    print('TOTALS',json.dumps(totals),flush=True)

if __name__ == '__main__': main()
