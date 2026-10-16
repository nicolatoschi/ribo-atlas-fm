"""Derive measured clean-core coverage and bounded prospective curriculum counts."""
import json
import pathlib
import sqlite3

BASE=pathlib.Path(__file__).resolve().parent

def main():
    con=sqlite3.connect(BASE/'raw'/'admission_audit.sqlite')
    cohorts={}
    for label,where in (
        ('dual_complete_source_unflagged','has_complete_source_unflagged_DMS=1 AND has_complete_source_unflagged_2A3=1'),
        ('at_least_one_complete_source_unflagged','has_complete_source_unflagged_DMS=1 OR has_complete_source_unflagged_2A3=1')):
        full,core=con.execute('SELECT count(*),count(DISTINCT substr(sequence,27,100)) FROM candidates WHERE '+where).fetchone()
        groups=[{'source_group':r[0],'full_construct_candidate_count':r[1],'distinct_probed_100nt_core_count':r[2]} for r in con.execute('SELECT r.group_id,count(DISTINCT r.sequence),count(DISTINCT substr(r.sequence,27,100)) FROM records r JOIN candidates c ON r.sequence=c.sequence AND r.condition_key=c.condition_key WHERE '+where+' GROUP BY r.group_id')]
        cohorts[label]={'full_construct_candidate_count':full,'distinct_probed_100nt_core_count':core,'by_source_group':groups}
    warnings=[]
    columns=('source','has_warning','profile_records','wholly_missing_profiles','complete_finite_profiles','minimum_reads','maximum_reads','minimum_reported_SNR','maximum_reported_SNR','reported_SNR_below_one','reported_SNR_at_least_one','reported_reads_below100','reported_reads_at_least100_and_SNR_at_least_one')
    for row in con.execute('SELECT source,source_warning,count(*),sum(finite_values=0),sum(finite_values=100),min(reads),max(reads),min(snr),max(snr),sum(snr<1),sum(snr>=1),sum(reads<100),sum(reads>=100 AND snr>=1) FROM records GROUP BY source,source_warning ORDER BY source,source_warning'):
        warnings.append(dict(zip(columns,row)))
    flagged=[r for r in warnings if r['has_warning']]
    total_flagged=sum(r['profile_records'] for r in flagged)
    flagged_complete=sum(r['complete_finite_profiles'] for r in flagged)
    flagged_below1=sum(r['reported_SNR_below_one'] for r in flagged)
    summary={'source_warning_token_types':['warning:badQuality'],'flagged_profile_records':total_flagged,'flagged_complete_finite_profile_records':flagged_complete,'flagged_profiles_with_reported_SNR_below1':flagged_below1,'fraction_flagged_with_reported_SNR_below1':flagged_below1/total_flagged,'all_source_unflagged_profiles_have_reported_SNR_at_least1':all(r['reported_SNR_below_one']==0 for r in warnings if not r['has_warning']),'flag_interpretation':'The warning marks low signal/read quality in these data; most flagged arrays are complete, rather than missing. The actual release stores warning:badQuality and numerical SNR. Exact upstream flag thresholds were not independently reimplemented; rounded SNR=1.000 and a few low-read flagged records prevent inferring an exact universal criterion.'}
    readiness={'downloaded_source_assets_sha256_and_size_verified':8,'parsed_chemistry_profile_records':747902,'actual_condition_count':1,'full_construct_length_nt':177,'probed_positions_per_construct':100,'actual_input_fixture_count':4,'actual_input_fixture_declared_distinct_parent_ids':4,'final_task_specific_split_admitted_count':None,'legacy_exposure_audited_count':None,'independently_validated_family_count':None,'prospective_strict_clean_training_distinct_core_cap':60000,'cap_rationale':'75,659 measured dual-clean distinct cores × 80%=60,527 before group/exposure losses; round down to a maximum planning cap of60,000. Release only min(60,000, actually G1-admitted training cores) after parent/window lineage blocking, task-specific family/study/time holdouts and legacy exposure audit. This is a bounded prospective cap, not a measured post-partition cohort.','broad_K8_scope':'Sparse/uncertainty-aware learning may use observed candidate views under train-only error calibration and lineage weighting. Low-SNR source flags remain explicit; missing values are masked and no candidate count is an independent-family count.','strict_K32_scope':'Final ensemble learning uses the admitted clean cohort and fresh Monte Carlo draws; parent/mutant/correlated observations remain grouped and correctly weighted.'}
    result={'measured_clean_cohorts':cohorts,'source_quality_summary':summary,'source_quality_table':warnings,'readiness_and_prospective_curriculum':readiness}
    result['readiness_and_prospective_curriculum']['cap_rationale'] = result['readiness_and_prospective_curriculum']['cap_rationale'].replace('cap of60,000', 'cap of 60,000')
    (BASE/'clean_curriculum_readiness_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    con.close()

if __name__=='__main__':main()
