"""Independent compact audit of finite-value masks, uncertainty arrays and lineage."""
import collections
import csv
import hashlib
import json
import math
import pathlib
import re
import sqlite3

BASE = pathlib.Path(__file__).resolve().parent

def main():
    verified = json.loads((BASE / 'download_manifest.json').read_text())['assets']
    audits = []
    histogram_rows = []
    for asset in verified:
        if asset['status'] != 'verified': continue
        source = asset['rmdb_id']
        masks = {}
        histogram = collections.Counter()
        paired_mismatches = missing_error_masks = zero_errors = negative_errors = minus999 = nonfinite_values = nonfinite_errors = 0
        with (BASE / 'raw' / (source + '.rdat')).open() as stream:
            for line in stream:
                if not line.startswith('REACTIVITY'): continue
                key, *tokens = line.split()
                rid = int(key.split(':')[1])
                values = [float(t) for t in tokens]
                finite = [math.isfinite(v) for v in values]
                mask = sum(1 << idx for idx, ok in enumerate(finite) if ok)
                minus999 += sum(v == -999.0 for v in values)
                if key.startswith('REACTIVITY_ERROR:'):
                    nonfinite_errors += len(values)-sum(finite)
                    zero_errors += sum(v == 0 for v in values)
                    negative_errors += sum(v < 0 for v in values if math.isfinite(v))
                    if rid not in masks: missing_error_masks += 1
                    elif masks[rid] != mask: paired_mismatches += 1
                else:
                    masks[rid] = mask
                    nfinite = sum(finite)
                    nonfinite_values += len(values)-nfinite
                    histogram[nfinite] += 1
        audit = {'source':source,'profile_record_count':len(masks),'profile_vs_error_finite_mask_mismatch_records':paired_mismatches,'errors_without_profile_record':missing_error_masks,'nonfinite_observation_positions':nonfinite_values,'nonfinite_error_positions':nonfinite_errors,'finite_negative_error_values':negative_errors,'finite_zero_error_values':zero_errors,'literal_minus999_values_or_errors':minus999,'coverage_histogram_finite_positions':dict(sorted(histogram.items()))}
        audits.append(audit)
        histogram_rows.extend({'source':source,'finite_positions':n,'profile_records':count} for n,count in sorted(histogram.items()))
        print(json.dumps(audit),flush=True)
    con = sqlite3.connect(BASE / 'raw' / 'admission_audit.sqlite')
    expected_records = sum(s['construct_profile_records'] for s in json.loads((BASE / 'actual_admission_evidence.json').read_text())['sources'])
    if con.execute('SELECT count(*) FROM records WHERE nerrors IS NOT NULL').fetchone()[0] != expected_records:
        raise RuntimeError('Primary parser has not committed all expected profile/error records; rerun after primary parsing completes.')
    parents = set()
    rfams = set()
    warnings = collections.Counter()
    for group,name,annotations in con.execute('SELECT group_id,name,annotations FROM records WHERE chemistry="2A3"'):
        if group == 'M2PK90':
            match = re.match(r'(\d+)', name)
            if match: parents.add(match.group(1))
        if group == 'RFMREP': rfams.update(re.findall(r'RF\d{5}',name))
        for token in json.loads(annotations):
            if token.startswith('warning:'): warnings[token]+=1
    group_core = [{'source_group':group,'distinct_probed_100nt_core_windows_before_missing_filter':con.execute('SELECT count(DISTINCT substr(sequence,27,100)) FROM records WHERE group_id=?',(group,)).fetchone()[0],'distinct_probed_100nt_core_windows_with_observed_data':con.execute('SELECT count(DISTINCT substr(sequence,27,100)) FROM records WHERE group_id=? AND finite_values>0',(group,)).fetchone()[0]} for group in ('OK2TRN','OK1LIB','M2PK90','RFMREP')]
    lineage = {'M2PK90_declared_numeric_parent_id_tokens':len(parents),'RFMREP_Rfam_family_id_annotation_tokens':len(rfams),'all_groups_distinct_probed_100nt_core_windows_before_missing_filter':con.execute('SELECT count(DISTINCT substr(sequence,27,100)) FROM records').fetchone()[0],'all_groups_distinct_probed_100nt_core_windows_with_observed_data':con.execute('SELECT count(DISTINCT substr(sequence,27,100)) FROM records WHERE finite_values>0').fetchone()[0],'group_counts':group_core,'warning_tokens_in_2A3_records':dict(warnings),'limitation':'Parent and Rfam IDs are declared source tokens, not independently validated biological families. Adjacent windows, mutants and barcodes remain related and require task-specific grouping.'}
    result = {'masks':audits,'lineage':lineage,'observation_definition':'Every finite value, including zero and negative background-subtracted values, is observed. Nonfinite NaN values are masked.','uncertainty_definition':'Reported RDAT errors remain distinct from calibrated predictive uncertainty; zero reported errors require a train-only calibrated error floor, not infinite confidence.','independent_primary_parser_reference':'https://github.com/ribokit/RDATKit/blob/master/rdat-kit/src/rdat_kit/handler.py','position_mapping':'RDAT SEQPOS is 1-based full-construct indexing; OFFSET is 0. The selected groups probe positions 27–126 of a 177-nt construct.'}
    (BASE / 'mask_and_lineage_audit.json').write_text(json.dumps(result,indent=2))
    with (BASE / 'actual_mask_coverage_distribution.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['source','finite_positions','profile_records']);writer.writeheader();writer.writerows(histogram_rows)
    con.close()

if __name__ == '__main__': main()
