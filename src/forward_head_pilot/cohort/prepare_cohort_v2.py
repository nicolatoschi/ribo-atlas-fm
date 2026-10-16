"""Wave F v2: freeze the NON-VIRAL exact-core pilot cohort and lineage folds before fitting.

Derived from v1 prepare_cohort.py (sha256 in ../CHANGES.md). Changes versus v1:
  * Scope restricted to non-viral RNAs (independent benign pilot audit, 2026-10-14):
    Zika xrRNA (ZIKAXR_DMS_0000 / 5TPY) removed; ydaO c-di-AMP riboswitch
    (YDAORS_DMS_0001 profile 1, mutation:WT, 10 uM c-di-AMP) mapped to 4QLM chain A
    residues 39-123 (= RDAT positions 37-121) added, exactly as established by
    ../benign_cohort/ydao_source_mapping_audit.json. The exclusion of Zika is a scope
    decision taken before any model fit (v1 manifest fitted:false); it is not based on
    data quality or on any model outcome.
  * A deposited STRUCTURE whose bracket type is unbalanced is not repaired: every
    position annotated with that bracket type is set to unknown ('x'). This affects
    only the ydaO pseudoknot ('[' x7 vs ']' x6 in the deposited RDAT); the rule was
    fixed from the annotation string alone, without inspecting reactivities or fits.
  * Coordinate chains with unobserved residues may be restricted to an explicit
    contiguous PDB residue range (only 4QLM; range taken from the mapping audit).

Crystal structures are explicit structural proxies for solution measurements,
not claims of identical thermodynamic state or unseen-pretraining benchmarks.
Only deposited wild-type rows are selected. No target-dependent source selection.
"""
from pathlib import Path
import csv, datetime, hashlib, json, re
import numpy as np
BASE=Path(__file__).resolve().parent
SRC=BASE/'sources'
MAPPING_AUDIT=BASE.parent/'benign_cohort'/'ydao_source_mapping_audit.json'
CASES=[
 ('adenine_add_SHAPE_holo','adenine_add','ADDRSW_SHP_0001',1,'SHAPE','1Y26','X','Kladwang2011','SHAPE_2011_HEPES_24C','Ligand-present deposited row selected a priori; adenine5mM in assay and adenine in crystal. Crystallization solvent/temperature differ.'),
 ('P4P6_SHAPE_aqueous','P4P6_groupI','TRP4P6_SHP_0001',1,'SHAPE','1GID','A','Kladwang2011','SHAPE_2011_HEPES_24C','Aqueous no-MPD deposited row selected a priori; crystal includes cobalt hexammine, unlike assay. One native core, not independent crystal-copy states.'),
 ('P4P6_DMS_aqueous','P4P6_groupI','TRP4P6_DMS_0001',1,'DMS','1GID','A','Cordero2012','DMS_2012_HEPES_24C','One average deposited DMS profile; same P4P6 lineage as SHAPE. Crystal includes cobalt hexammine, unlike assay.'),
 ('TPP_DMS_holo_WT','TPP_riboswitch','TPPRSW_DMS_0001',1,'DMS','2GDI','X','Cheng2017','DMS_2017_cacodylate_24C','Only mutation:WT ligand-present row selected; assay10mMTPP and crystalTPP. Crystal X/Y are copies of one core, use only X. Noncanonical GTP/CCC termini outside core are excluded.'),
 ('ydaO_DMS_holo_WT','ydao_cdiamp_riboswitch','YDAORS_DMS_0001',1,'DMS','4QLM','A','Cheng2017','DMS_2017_cacodylate_24C','Only mutation:WT ligand-present row (profile 1) selected a priori; assay 10 uM c-di-AMP and crystal 4QLM contains two c-di-AMP. Exact 85-nt contiguous core = 4QLM chain A residues 39-123 = RDAT positions 37-121 (longest contiguous observed C4prime block, per benign_cohort/ydao_source_mapping_audit.json). Crystal residues 3-12 and 22-34 exist but are outside the contiguous core and residues 1-2, 13-21, 35-38, 124-125 are unobserved, so packing descriptors near that interface are truncated (partial domain). 2.72 A crystal with high B factors (core mean ~90 A^2). Source RDAT has no REACTIVITY_ERROR (unit precision within profile). Deposited pseudoknot bracket type is unbalanced; those 13 positions are set unknown (no repair). RDAT organism label vs PDB organism: see organism_label_check.json.'),
]
# Explicit contiguous residue windows for chains with unobserved residues (from the mapping audit).
RESIDUE_RANGES={('4QLM','A'):(39,123)}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def parse_rdat(path):
 d={'profiles':{},'comments':[]}
 for line in path.read_text().splitlines():
  f=line.split()
  if not f:continue
  k,rest=f[0],f[1:]
  if k in ['SEQUENCE','STRUCTURE','NAME','ANNOTATION']:d[k.lower()]=' '.join(rest)
  elif k=='OFFSET':d['offset']=int(rest[0])
  elif k=='SEQPOS':d['positions']=[int(re.sub(r'^[A-Za-z]+','',x)) for x in rest]
  elif k=='COMMENT':d['comments'].append(' '.join(rest))
  elif k.startswith(('ANNOTATION_DATA:','REACTIVITY:','REACTIVITY_ERROR:')):
   n=int(k.split(':')[1]);p=d['profiles'].setdefault(n,{})
   field='annotation' if k.startswith('ANNOTATION_DATA') else 'stderr' if k.startswith('REACTIVITY_ERROR') else 'reactivity'
   p[field]=' '.join(rest) if field=='annotation' else [float(x) for x in rest]
 return d
def pdb_chain(path,chain,residue_range=None):
 residues={}
 for line in path.read_text().splitlines():
  if not line.startswith('ATOM  ') or line[21]!=chain:continue
  base=line[17:20].strip()
  if base not in ['A','C','G','U']:continue
  key=(int(line[22:26]),line[26].strip())
  if residue_range is not None and not (residue_range[0]<=key[0]<=residue_range[1]):continue
  r=residues.setdefault(key,{'base':base,'atoms':{}})
  assert r['base']==base
  atom=line[12:16].strip().replace('*',"'");alt=line[16].strip();priority=2 if not alt else 1 if alt=='A' else 0;occ=float(line[54:60])
  new={'xyz':[float(line[a:b]) for a,b in [(30,38),(38,46),(46,54)]],'choice':(priority,occ)}
  old=r['atoms'].get(atom)
  if old is None or new['choice']>old['choice']:r['atoms'][atom]=new
 out=list(residues.items())
 if residue_range is not None:
  numbers=[k[0] for k,_ in out]
  assert numbers==list(range(residue_range[0],residue_range[1]+1)),'residue window must be complete and contiguous'
 return out
def restricted_secondary(struct,start,length):
 # Multiple bracket types support deposited pseudoknot annotations; letter-pairs
 # are intentionally not inferred. v2: a bracket type that is unbalanced in the
 # deposited string is NOT repaired; all its positions become unknown ('x').
 openers='([{<';partner=dict(zip(')]}>',openers))
 unbalanced={op for op in openers if struct.count(op)!=struct.count({v:k for k,v in partner.items()}[op])}
 stacks={c:[] for c in openers};pairs=[];unknown=set()
 for i,c in enumerate(struct):
  if c in stacks:
   if c in unbalanced:unknown.add(i);continue
   stacks[c].append(i)
  elif c in partner:
   op=partner[c]
   if op in unbalanced:unknown.add(i);continue
   assert stacks[op],(i,c)
   j=stacks[op].pop()
   if start<=j<start+length and start<=i<start+length:pairs.append((j-start,i-start,op,c))
  else:assert c=='.',c
 assert all(not s for s in stacks.values())
 chars=['.']*length
 for i,j,op,cl in pairs:chars[i]=op;chars[j]=cl
 unknown_core=sorted(i-start for i in unknown if start<=i<start+length)
 for i in unknown_core:chars[i]='x'
 return ''.join(chars),[[i,j] for i,j,_,_ in sorted(pairs)],{'unbalanced_bracket_types':sorted(unbalanced),'positions_set_unknown_core_0based':unknown_core}
examples=[];allrows=[]
for name,lineage,ident,profile,chem,pdb,chain,study,protocol,note in CASES:
 data=parse_rdat(SRC/(ident+'.rdat'));pf=data['profiles'][profile]
 assert profile==1
 if data['annotation'].find('MutateAndMap')>=0:assert pf['annotation']=='mutation:WT'
 window=RESIDUE_RANGES.get((pdb,chain))
 coord=pdb_chain(SRC/(pdb+'.pdb'),chain,window);sequence=''.join(r['base'] for _,r in coord)
 assert len(sequence)>=50
 full=data['sequence'];assert full.upper().count(sequence)==1
 start=full.upper().index(sequence);L=len(sequence)
 # Exact contiguous matching prohibits similarity-based mutation substitution,
 # inference of missing coordinates or splicing around engineered insertions.
 assert full[start:start+L].upper()==sequence
 annotation=data['annotation']+' '+pf.get('annotation','')
 positions={p:i for i,p in enumerate(data['positions'])};ys=[];errors=[];mask=[];mapping=[];coords=[];pcoords=[];fullatoms=[]
 secondary_source=ident
 if 'structure' not in data:
  assert ident=='TRP4P6_DMS_0001'
  shared=parse_rdat(SRC/'TRP4P6_SHP_0001.rdat');assert shared['sequence']==full
  structure=shared['structure'];secondary_source='TRP4P6_SHP_0001'
 else:structure=data['structure']
 secondary,pairs,secondary_handling=restricted_secondary(structure,start,L)
 for i,((resnum,icode),r) in enumerate(coord):
  construct_index=start+i;pos=construct_index+1+data['offset'];j=positions.get(pos)
  y=pf['reactivity'][j] if j is not None else None;e=pf.get('stderr',[None]*len(data['positions']))[j] if j is not None else None
  y=y if y is not None and np.isfinite(y) else None;e=e if e is not None and np.isfinite(e) and e>=0 else None
  observed=y is not None;allowed=observed and (chem!='DMS' or sequence[i] in 'AC')
  ys.append(y);errors.append(e);mask.append(bool(allowed))
  assert "C4'"in r['atoms'];coords.append(r['atoms']["C4'"]['xyz']);pcoords.append(r['atoms'].get('P',{}).get('xyz'))
  fullatoms.append({a:v['xyz'] for a,v in r['atoms'].items()})
  mapping.append({'core_index_1based':i+1,'construct_index_1based':construct_index+1,'deposited_coordinate':pos,'pdb_resnum':resnum,'pdb_icode':icode,'base':sequence[i]})
  allrows.append(dict(example_id=name,lineage_id=lineage,chemistry=chem,core_index_1based=i+1,construct_index_1based=construct_index+1,deposited_coordinate=pos,pdb=pdb,chain=chain,pdb_resnum=resnum,pdb_icode=icode,base=sequence[i],reactivity=y,stderr=e,observed_finite=observed,observed_mask=bool(allowed),secondary_token=secondary[i]))
 # All source/coordinate identities established without inspecting model accuracy.
 provenance={'rmdb_id':ident,'profile_row':profile,'rdat_path':'sources/'+ident+'.rdat','rdat_sha256':sha(SRC/(ident+'.rdat')),'metadata_path':'sources/'+ident+'.md','metadata_sha256':sha(SRC/(ident+'.md')),'secondary_annotation_rmdb_source':secondary_source,'secondary_annotation_rdat_sha256':sha(SRC/(secondary_source+'.rdat')),'pdb_id':pdb,'pdb_chain':chain,'pdb_path':'sources/'+pdb+'.pdb','pdb_sha256':sha(SRC/(pdb+'.pdb')),'PDB_technology':'X-RAY DIFFRACTION','coordinate_atom':"C4'",'coordinate_units':'angstrom','full_assay_sequence':full,'full_assay_length':len(full),'core_construct_1based':[start+1,start+L],'pdb_residue_window':list(window) if window else None,'excluded_5prime_assay_sequence':full[:start],'excluded_3prime_assay_sequence':full[start+L:],'offset':data['offset'],'source_comments':data['comments'],'deposited_row_count_not_independent_RNAs':len(data['profiles']),'secondary_annotation_handling':secondary_handling,'source_name_field':data.get('name')}
 if ident=='YDAORS_DMS_0001':
  provenance['mapping_audit']={'path':'../benign_cohort/ydao_source_mapping_audit.json','sha256':sha(MAPPING_AUDIT),'note':'Mapping established by the earlier exact-core audit; independently re-derived from raw bytes by verify_extraction_from_raw.py.'}
 example={'example_id':name,'lineage_id':lineage,'study_id':study,'chemistry':chem,'protocol_id':protocol,'sequence':sequence,'reactivity':ys,'stderr':errors,'observed_mask':mask,'secondary_dot_bracket':secondary,'secondary_pairs_0based':pairs,'coordinates':coords,'phosphate_coordinates':pcoords,'all_atom_coordinates':fullatoms,'mapping':mapping,
 'admission':{'eligible':True,'geometry_compatibility':'structural_proxy_exact_sequence_core','exact_sequence_core':True,'complete_C4prime':True,'identical_crystal_solution_conditions':False,'interpretation':'Small empirical cross-lineage structure-to-probe validation using experimental crystal geometry as a solution-state proxy; no structure-generation or condition-response efficacy claim.','limitations':note},
 'protocol':{'source_annotation':annotation,'reported_error_available':'stderr'in pf,'reported_error_semantics':'REACTIVITY_ERROR as deposited, in deposited reactivity units; whether it is a standard deviation or a standard error of replicate means is not stated in the local RDAT/metadata files (unknown). Absent field = no reported uncertainty (null).','DMS_primary_mask':'A/C only' if chem=='DMS' else 'All finite measured nucleotides','target_normalization':'Original deposited values unchanged, including finite negatives; any transform/scaling must be fixed from training folds only.','temperature_C':24,'MgCl2_mM':10,'buffer':'50mMNaHEPESpH8' if 'HEPES'in annotation else '300mMNa-cacodylatepH7','native_core_geometry':True,'deposited_secondary_annotation':'Source-provided annotation restricted to this exact core, not independently predicted sequence-only secondary structure. Its original derivation may have used chemical-mapping data from the same group/experiment class (not established either way): treat the secondary arm as an annotation-informed baseline that may carry target-derived information.'},
 'provenance':provenance}
 examples.append(example)
lineages=sorted(set(e['lineage_id']for e in examples));folds=[]
for held in lineages:
 folds.append({'fold_id':'holdout_'+held,'test_lineages':[held],'train_lineages':[x for x in lineages if x!=held],'test_examples':[e['example_id']for e in examples if e['lineage_id']==held],'train_examples':[e['example_id']for e in examples if e['lineage_id']!=held]})
contract={'schema_version':2,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'Frozen before any learned-head fit','scope_restriction':'Non-viral RNAs only (adenine riboswitch, P4-P6 group I domain, TPP riboswitch, ydaO c-di-AMP riboswitch), per independent_benign_pilot_audit.txt. Zika xrRNA excluded by scope (see excluded_primary_candidates.json).','independent_lineage_count':len(lineages),'case_profile_count':len(examples),'outer_folds':folds,'grouping':'Entire RNA lineage, including both P4P6 chemistries, held out together. Shared study is disclosed; no study-disjoint claim.','primary_scoring':'DMS A/C finite observations; SHAPE all finite observations. Missing errors remain null and must not be replaced by invented measurement errors. Equal-lineage macro metrics; no nucleotide-independence p-value.','model_scope':'Small forward measurement-operator pilot. No pretrained structural model used to establish unseen-structure generalization; these old RNAs may be exposed in public pretrained corpora.','test_profile_use':'Held-out targets used solely for final scoring; never input descriptors, target normalization, hyperparameter selection or refinement. Inner CV grouped by training lineage only.','crystal_state_scope':'Every example is a crystal-geometry proxy for a solution measurement, with exact sequence core. Different buffers/packing/ions mean these are not identical solution ensembles.','examples':examples}
dump(BASE/'paired_core_cohort.json',contract)
split={'schema_version':2,'frozen_utc':contract['created_utc'],'frozen_before_model_fitting':True,'cohort_sha256':sha(BASE/'paired_core_cohort.json'),'outer_folds':folds,'source_choice_rule':'Selected wild-type ligand-present rows and exact contiguous canonical coordinate cores by source annotation/sequence integrity, before fitting. Excluded engineered-loop/mutation/partner mismatches from primary. v2: non-viral scope; Zika xrRNA replaced by ydaO (4QLM A 39-123).','primary_case_ids':[e['example_id']for e in examples]}
dump(BASE/'frozen_lineage_splits.json',split)
with (BASE/'paired_core_observations.csv').open('w',newline='')as f:
 w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
counts={e['example_id']:{'lineage_id':e['lineage_id'],'length':len(e['sequence']),'chemistry':e['chemistry'],'scored_observations_before_structure_mask':sum(e['observed_mask']),'scored_observations_after_secondary_unknown_mask':sum(m and s!='x' for m,s in zip(e['observed_mask'],e['secondary_dot_bracket'])),'finite_reactivity':sum(x is not None for x in e['reactivity']),'finite_negative_reactivity':sum(x is not None and x<0 for x in e['reactivity']),'reported_error_count':sum(x is not None for x in e['stderr']),'zero_reported_error_count':sum(x==0 for x in e['stderr'] if x is not None),'secondary_unknown_positions':e['secondary_dot_bracket'].count('x'),'coordinate_source':e['provenance']['pdb_id']+'_'+e['provenance']['pdb_chain']}for e in examples}
dump(BASE/'admission_summary.json',{'cohort_sha256':sha(BASE/'paired_core_cohort.json'),'split_sha256':sha(BASE/'frozen_lineage_splits.json'),'status':'Prepared authenticated finite pilot data; no fit has been run by data-preparation script','scope_restriction':contract['scope_restriction'],'independent_lineages':len(lineages),'examples':counts,'total_scored_observations_after_secondary_unknown_mask':sum(c['scored_observations_after_secondary_unknown_mask'] for c in counts.values()),'chemistry_by_lineage':{l:sorted({e['chemistry'] for e in examples if e['lineage_id']==l}) for l in lineages},'study_by_lineage':{l:sorted({e['study_id'] for e in examples if e['lineage_id']==l}) for l in lineages},'all_C4prime_complete':True,'exact_contiguous_core_mapping':True,'secondary_labels_are_deposited_not_sequence_predicted':True,'condition_exactness_claim':False,'coordinate_augmentation_count':0,'source_copy_replication_count':0})
dump(BASE/'excluded_primary_candidates.json',{'schema_version':2,'note':'v2 supersedes the v1 list: the v1 ydaO entry is replaced (see ydaO entries below) and Zika xrRNA is added (scope).','exclusions':[
 {'source':'ZIKAXR_DMS_0000/5TPY (Zika_xrRNA_DMS_WT)','reason':'SCOPE: the pilot is restricted to non-viral RNAs per independent_benign_pilot_audit.txt (adenine, P4-P6, TPP, ydaO). The source passed the v1 exact-core mapping (71 nt, 36 DMS A/C sites, no REACTIVITY_ERROR). Exclusion decided before any model fit (v1 prepared manifest fitted:false; no v1 results exist); not a data-quality or outcome-based exclusion. Source files retained in sources/.'},
 {'source':'YDAORS_DMS_0000 (ligand-absent WT) with 4QLM','reason':'Apo assay versus c-di-AMP-bound crystal is not a state-compatible primary pair (same rule as TPPRSW_DMS_0000); same-lineage condition, not an independent lineage. Ligand-present YDAORS_DMS_0001 is admitted instead.'},
 {'source':'YDAORS_DMS_0001 profiles 2-192 (mutate-and-map)','reason':'Mutation perturbations of the same parent RNA; only mutation:WT profile 1 admitted.'},
 {'source':'4OQU, 4P8Z as ydaO coordinate candidates','reason':'Different RNAs (SAM-I/IV riboswitch; lariat-capping ribozyme); no whole-SEQRES exact source match (ydao_source_mapping_audit.json).'},
 {'source':'4QLM chain A residues outside 39-123','reason':'Residues 1-2, 13-21, 35-38, 124-125 unobserved; observed residues 3-12 and 22-34 are not contiguous with the 85-nt core and are not used (no splicing). Partial-domain packing caveat recorded.'},
 {'source':'v1 entry "YDAORS_DMS_0000/0001 with4QLM,4OQU"','reason':'SUPERSEDED: v1 judged no exact contiguous core; the later exact-core audit (benign_cohort/ydao_source_mapping_audit.json) established 4QLM A 39-123 = RDAT 37-121 with sequence identity 1.0 and zero indels/substitutions.'},
 {'source':'CIDGMP_SHP_0001/3MXH,3IWN,3MUT,3MUV','reason':'Engineered U1A-binding insertion/mutations means full coordinate chain is not an exact contiguous assayed core; no splicing/substitution admitted.'},
 {'source':'GLYCFN_SHP_0001/3P49','reason':'Engineered insertion in one loop; full chain not exact contiguous assayed core. Shared-core proxy could be separate sensitivity analysis, not primary.'},
 {'source':'TRNAPH_SHP_0001/1EHZ','reason':'Probe transcript versus modified native tRNA coordinate modification state not established; canonicalization insufficient.'},
 {'source':'5SRRNA_SHP_0001','reason':'Isolated probing versus ribosome-bound coordinates requires partner compatibility; excluded primary.'},
 {'source':'GIR1RZ_DMS_0000/4P8Z','reason':'Assay U versus crystal C at corresponding core140 (construct159); one mutation, not exact contiguous core. Excluded primary rather than silently repaired.'},
 {'source':'RNASEP_DMS_0000/1NBS,1U9S','reason':'Wrong structural domain/species; no exact native catalytic-domain coordinate match admitted. Source retained as failed candidate lead.'},
 {'source':'TPPRSW_DMS_0000/2GDI','reason':'Apo assay and holo crystal are not a state-compatible primary pair; ligand-present0001 used instead.'},
 {'source':'ADDRSW_SHP_0001 references1Y27/2G9C/3GO2','reason':'Alternative1Y27/2G9C sequences do not exactly match assayed add core;3GO2 is a protein entry despite deposit source lead. Exact1Y26 chosen.'},
 {'source':'Crystal chains1GID_B/2GDI_Y and mutate-map rows','reason':'Copies/perturbations of the same parent RNA do not create independent native-coordinate cases; retained source files only.'}]})
print(json.dumps({'cohort':str(BASE/'paired_core_cohort.json'),'sha256':sha(BASE/'paired_core_cohort.json'),'lineages':len(lineages),'examples':counts},indent=2))
