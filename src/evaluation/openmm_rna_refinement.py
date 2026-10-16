#!/usr/bin/env python3
"""Bounded OH-fragment OL3/GBn2 engineering minimization, never job submission.

The model boundary is explicit: same 5'-OH/3'-OH RNA fragment for native and
generated inputs. Original files remain untouched. No Mg/solution populations,
novel learned-model efficacy or complete chemical-validity certification follows.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
import traceback

import numpy as np
import openmm as mm
from openmm import app,unit

FF_FILES=('amber14/RNA.OL3.xml','implicit/gbn2.xml')
RESTraint_STAGES=((1000.0,250),(100.0,250)) # kJ/mol/nm^2; bounded per stage
TOLERANCE=10.0 # kJ/mol/nm force RMS criterion requested, no convergence assumption

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def finite(value,label):
    if not np.isfinite(np.asarray(value)).all():raise RuntimeError('Nonfinite '+label)
def summary(a):
    a=np.asarray(a,dtype=float);finite(a,'QC values')
    return {'count':int(a.size),'minimum':float(a.min()),'median':float(np.median(a)),
            'maximum':float(a.max()),'RMS':float(np.sqrt((a*a).mean()))}

def canonical_fragment(text,sequence):
    """Select only canonical chain-A RNA heavy atoms and declare OH termini."""
    original=[];keys=[];bases={};seen=set()
    for line in text.splitlines():
        if not line.startswith('ATOM  ') or line[21]!='A':continue
        base=line[17:20].strip();atom=line[12:16].strip().replace('*',"'")
        if base not in 'AUGC' or len(base)!=1:raise ValueError('Noncanonical RNA residue')
        if line[16].strip() not in ('','A'):continue
        if atom.startswith('H'):raise ValueError('Input must be the frozen heavy-atom structure')
        key=(int(line[22:26]),line[26].strip())
        if key not in bases:keys.append(key);bases[key]=base
        if (key,atom) in seen:raise ValueError('Duplicate admitted input atom')
        seen.add((key,atom));xyz=[float(line[30:38]),float(line[38:46]),float(line[46:54])]
        finite(xyz,'input coordinates');original.append((key,atom,base,xyz))
    if ''.join(bases[k] for k in keys)!=sequence or len(keys)!=158:raise ValueError('Exact 158-nt frozen sequence does not match input')
    numbering={key:i+1 for i,key in enumerate(keys)};lines=[];removed=[]
    for key,atom,base,xyz in original:
        residue=numbering[key]
        if residue==1 and atom in ('P','OP1','OP2','OP3'):
            removed.append({'original_author_residue':key[0],'atom':atom});continue
        element='P' if atom=='P' else atom[0]
        serial=len(lines)+1;name=(' '+atom).ljust(4) if len(atom)<4 else atom
        lines.append(f'ATOM  {serial:5d} {name} {base:>3s} A{residue:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}  ')
    if len(lines)!=3379 or sorted(x['atom'] for x in removed) not in (['OP1','OP2','P'],['OP1','OP2','OP3','P']):raise ValueError('Boundary atom accounting differs')
    result='\n'.join(lines+['TER','END',''])
    return result,{'input_heavy_atom_records':len(original),'admitted_RNA_residues':158,
                   'removed_5prime_phosphate_records':removed,'OH_fragment_heavy_atoms':3379,
                   'boundary':'Explicit engineered 5prime-OH/3prime-OH fragment; first phosphate group removed (native P/OP1/OP2/OP3, generated P/OP1/OP2), all 157 interresidue phosphates retained',
                   'canonical_fragment_SHA256':hashlib.sha256(result.encode()).hexdigest()}

def forcefield_hashes():
    data=Path(app.__file__).resolve().parent/'data'
    return {name:digest(data/name) for name in FF_FILES}

def topology_preflight(text,sequence):
    """Actual topology/template/system parameterization; no energy context.

    Add template hydrogen topology only, without pretending those unplaced
    hydrogens have valid positions. GPU preparation places actual hydrogens.
    """
    canonical,metadata=canonical_fragment(text,sequence)
    pdb=app.PDBFile(io.StringIO(canonical));ff=app.ForceField(*FF_FILES)
    top=app.Topology();chain=top.addChain('A');records=list(pdb.topology.residues());maps=[];chosen=[]
    for i,residue in enumerate(records):
        template_name=residue.name+('5' if i==0 else '3' if i==len(records)-1 else '')
        template=ff._templates[template_name]
        observed={a.name for a in residue.atoms()}
        expected={a.name for a in template.atoms if a.element!=app.element.hydrogen}
        if observed!=expected:raise ValueError('Heavy-atom template mismatch at residue '+str(i+1))
        expected_external={"O3'"} if i==0 else {'P'} if i==len(records)-1 else {'P',"O3'"}
        if {template.atoms[j].name for j in template.externalBonds}!=expected_external:raise ValueError('Unexpected OL3 polymer connectivity')
        newres=top.addResidue(residue.name,chain,id=str(i+1));atoms={a.name:top.addAtom(a.name,a.element,newres) for a in template.atoms}
        for one,two in template.bonds:top.addBond(atoms[template.atoms[one].name],atoms[template.atoms[two].name])
        maps.append(atoms);chosen.append(template_name)
    for i in range(157):top.addBond(maps[i]["O3'"],maps[i+1]['P'])
    matches=ff.getMatchingTemplates(top)
    if [t.name for t in matches]!=chosen:raise ValueError('OL3 matched templates differ from explicit boundary')
    system=ff.createSystem(top,nonbondedMethod=app.NoCutoff,constraints=None,rigidWater=False,
                           soluteDielectric=1.0,solventDielectric=78.3,implicitSolventKappa=0.0/unit.nanometer)
    charge=0.0;forces=[]
    for force in system.getForces():
        forces.append(type(force).__name__)
        if isinstance(force,mm.NonbondedForce):
            for i in range(force.getNumParticles()):
                q,sigma,epsilon=force.getParticleParameters(i)
                values=[q.value_in_unit(unit.elementary_charge),sigma.value_in_unit(unit.nanometer),epsilon.value_in_unit(unit.kilojoule_per_mole)]
                finite(values,'nonbonded parameters');charge+=values[0]
    if abs(charge+157)>1e-5:raise ValueError('OH fragment formal charge differs from 157 retained phosphate groups')
    return {**metadata,'status':'Actual CPU topology + OL3/GBn2 full system parameterization passed; no hydrogen-position, energy-context or minimization execution',
            'system_particles_including_template_hydrogens':system.getNumParticles(),'matched_template_count':len(matches),
            'first_last_templates':[matches[0].name,matches[-1].name],'interresidue_O3prime_P_bonds':157,
            'formal_charge_elementary_charge':charge,'force_classes':forces,
            'forcefield_hashes':forcefield_hashes(),'OpenMM_version':mm.__version__}

def platform_configuration(name,gpu):
    platform=mm.Platform.getPlatformByName(name)
    if name=='CUDA':properties={'DeviceIndex':str(gpu),'Precision':'double','DeterministicForces':'true'}
    elif name=='CPU':properties={'Threads':'2','DeterministicForces':'true'}
    else:raise ValueError('Only explicit CPU or allocated CUDA platforms supported')
    available=set(platform.getPropertyNames())
    if not set(properties)<=available:raise RuntimeError('Required platform properties unavailable')
    # addHydrogens accepts the platform but no properties mapping; defaults here
    # are process-local and do not modify installed files or shared environments.
    for key,value in properties.items():platform.setPropertyDefaultValue(key,value)
    return platform,properties

def geometry_qc(topology,positions,system):
    p=np.asarray(positions,dtype=float);finite(p,'geometry coordinates')
    atoms=list(topology.atoms());heavy=np.asarray([a.element!=app.element.hydrogen for a in atoms],dtype=bool)
    keys={(int(a.residue.id),a.name):a.index for a in atoms}
    bridges=np.asarray([np.linalg.norm(p[keys[(i,"O3'")]]-p[keys[(i+1,'P')]])*10 for i in range(1,158)])
    bond_errors=[];angle_errors=[];nonbonded=None
    for force in system.getForces():
        if isinstance(force,mm.HarmonicBondForce):
            for j in range(force.getNumBonds()):
                one,two,length,k=force.getBondParameters(j)
                if heavy[one] and heavy[two]:bond_errors.append(abs(np.linalg.norm(p[one]-p[two])-length.value_in_unit(unit.nanometer))*10)
        if isinstance(force,mm.HarmonicAngleForce):
            for j in range(force.getNumAngles()):
                one,two,three,target,k=force.getAngleParameters(j)
                if not (heavy[one] and heavy[two] and heavy[three]):continue
                a=p[one]-p[two];b=p[three]-p[two]
                actual=np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))
                angle_errors.append(abs(actual-target.value_in_unit(unit.radian))*180/np.pi)
        if isinstance(force,mm.NonbondedForce):nonbonded=force
    if nonbonded is None:raise RuntimeError('OL3 nonbonded force missing')
    selected=np.flatnonzero(heavy);sigs=[];eps=[];excluded=set()
    for i in range(nonbonded.getNumParticles()):
        q,sigma,epsilon=nonbonded.getParticleParameters(i);sigs.append(sigma.value_in_unit(unit.nanometer));eps.append(epsilon.value_in_unit(unit.kilojoule_per_mole))
    for i in range(nonbonded.getNumExceptions()):
        one,two,q,sigma,epsilon=nonbonded.getExceptionParameters(i);excluded.add((min(one,two),max(one,two)))
    sigs=np.asarray(sigs);eps=np.asarray(eps);clashes=0;eligible=0
    # A descriptive FF-dependent heavy-atom LJ proximity screen excludes all
    # explicit exceptions (1-2/1-3/1-4), without assigning an energy certification.
    for offset in range(0,len(selected),128):
        first=selected[offset:offset+128];d=np.linalg.norm(p[first,None,:]-p[selected][None,:,:],axis=-1)
        threshold=.75*2**(1/6)*(sigs[first,None]+sigs[selected][None,:])/2
        mask=(first[:,None]<selected[None,:])&(eps[first,None]>0)&(eps[selected][None,:]>0)
        for row,one in enumerate(first):
            for col,two in enumerate(selected):
                if (int(one),int(two)) in excluded:mask[row,col]=False
        clashes+=int(((d<threshold)&mask).sum());eligible+=int(mask.sum())
    return {'heavy_atom_count':int(heavy.sum()),'all_atom_count':len(atoms),
            'O3prime_next_P_angstrom':summary(bridges),'O3prime_next_P_above_2p1_angstrom_count':int((bridges>2.1).sum()),
            'O3prime_next_P_below_1p3_angstrom_count':int((bridges<1.3).sum()),
            'heavy_bond_absolute_error_to_OL3_target_angstrom':summary(bond_errors),
            'heavy_bond_error_above_0p15_angstrom_count':int((np.asarray(bond_errors)>.15).sum()),
            'heavy_angle_absolute_error_to_OL3_target_degree':summary(angle_errors),
            'heavy_angle_error_above_15_degree_count':int((np.asarray(angle_errors)>15).sum()),
            'heavy_nonbonded_LJ_proximity_screen_count':clashes,'heavy_nonbonded_LJ_screen_eligible_pairs':eligible,
            'LJ_screen_definition':'r < 0.75*2^(1/6)*Lorentz-combined OL3 sigma; positive-epsilon heavy atoms only; all explicit nonbonded exceptions excluded; descriptive screen'}

class BoundedReporter(mm.MinimizationReporter):
    def __init__(self,start,maximum_seconds):
        super().__init__();self.start=start;self.maximum_seconds=maximum_seconds;self.calls=0;self.maximum_iteration=-1;self.aborted_for_wall=False
    def report(self,iteration,x,grad,args):
        finite(x,'minimizer positions');finite(grad,'minimizer gradient')
        self.calls+=1;self.maximum_iteration=max(self.maximum_iteration,int(iteration))
        if time.perf_counter()-self.start>self.maximum_seconds:self.aborted_for_wall=True;return True
        return False

def energy_state(context):
    state=context.getState(getEnergy=True,getForces=True,getPositions=True)
    energy=float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    positions=np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    forces=np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))
    finite(energy,'potential energy');finite(positions,'positions');finite(forces,'forces')
    return energy,positions,float(np.sqrt(np.mean(np.sum(forces*forces,axis=1))))

def refine_case(label,path,manifest,out,gpu,platform_name='CUDA',maximum_seconds=120):
    start=time.perf_counter();text=Path(path).read_text();original_sha=digest(path)
    expected=manifest['native']['sha256'] if label=='native_control' else manifest['generated'][int(label.split('_')[-1])]['sha256']
    if original_sha!=expected:raise RuntimeError('Original input SHA differs: '+label)
    out.mkdir(parents=True,exist_ok=False)
    canonical,boundary=canonical_fragment(text,manifest['sequence'])
    (out/'OH_fragment_heavy_input.pdb').write_text(canonical)
    topology_start=time.perf_counter();preflight=topology_preflight(text,manifest['sequence'])
    if preflight['forcefield_hashes']!=manifest['forcefield_hashes']:raise RuntimeError('Forcefield source hashes differ')
    topology_seconds=time.perf_counter()-topology_start
    platform,properties=platform_configuration(platform_name,gpu)
    preparation_start=time.perf_counter();pdb=app.PDBFile(io.StringIO(canonical));modeller=app.Modeller(pdb.topology,pdb.positions)
    input_heavy_positions={
        (int(a.residue.id),a.name):np.asarray(pdb.positions[a.index].value_in_unit(unit.nanometer))
        for a in pdb.topology.atoms()}
    # Hydrogens use the declared OL3 topology and explicit allocated platform.
    # Standard canonical base protonation is selected at engineering pH=7.0.
    hydrogen_seed=20262012 if label=='native_control' else 20261012+int(label.split('_')[-1])
    np.random.seed(hydrogen_seed)
    import random
    random.seed(hydrogen_seed)
    ff=app.ForceField(*FF_FILES);modeller.addHydrogens(ff,pH=7.0,platform=platform)
    if len(list(modeller.topology.atoms()))!=preflight['system_particles_including_template_hydrogens']:raise RuntimeError('Placed hydrogen count differs from qualified templates')
    system=ff.createSystem(modeller.topology,nonbondedMethod=app.NoCutoff,constraints=None,rigidWater=False,
                           soluteDielectric=1.0,solventDielectric=78.3,implicitSolventKappa=0.0/unit.nanometer)
    original_positions=np.asarray(modeller.positions.value_in_unit(unit.nanometer))
    finite(original_positions,'hydrogen-prepared coordinates')
    restraint=mm.CustomExternalForce('0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.addGlobalParameter('k',RESTraint_STAGES[0][0])
    for name in ('x0','y0','z0'):restraint.addPerParticleParameter(name)
    atoms=list(modeller.topology.atoms());heavy_indices=[a.index for a in atoms if a.element!=app.element.hydrogen]
    hydrogen_prep_heavy_drift=max(float(np.linalg.norm(original_positions[i]-input_heavy_positions[(int(atoms[i].residue.id),atoms[i].name)])) for i in heavy_indices)
    if hydrogen_prep_heavy_drift>1e-6:raise RuntimeError('Hydrogen preparation unexpectedly moved admitted heavy atoms')
    for i in heavy_indices:restraint.addParticle(i,original_positions[i].tolist())
    system.addForce(restraint)
    integrator=mm.VerletIntegrator(.001*unit.picoseconds)
    context=mm.Context(system,integrator,platform,properties);context.setPositions(modeller.positions)
    preparation_seconds=time.perf_counter()-preparation_start
    before_energy,before_positions,before_force_RMS=energy_state(context)
    qc_start=time.perf_counter();before_qc=geometry_qc(modeller.topology,before_positions,system);before_qc_seconds=time.perf_counter()-qc_start
    stages=[]
    for strength,iterations in RESTraint_STAGES:
        context.setParameter('k',strength)
        energy_before,_,_=energy_state(context);stage_start=time.perf_counter()
        reporter=BoundedReporter(stage_start,maximum_seconds)
        mm.LocalEnergyMinimizer.minimize(context,TOLERANCE,iterations,reporter)
        energy_after,positions,force_RMS=energy_state(context)
        elapsed=time.perf_counter()-stage_start
        if reporter.aborted_for_wall:raise RuntimeError('Declared per-stage minimizer wall limit exceeded')
        if energy_after>energy_before+max(1e-4,abs(energy_before)*1e-8):raise RuntimeError('Minimization energy increased under the same restraint Hamiltonian')
        stages.append({'restraint_k_kJ_per_mol_nm2':strength,'maximum_iterations_requested':iterations,
                       'minimizer_tolerance_requested_kJ_per_mol_nm':TOLERANCE,'reporter_calls':reporter.calls,
                       'maximum_iteration_index_reported':reporter.maximum_iteration,'seconds':elapsed,
                       'potential_energy_before_kJ_per_mol':energy_before,'potential_energy_after_kJ_per_mol':energy_after,
                       'force_RMS_after_kJ_per_mol_nm':force_RMS,'finite_energy_positions_forces':True,
                       'convergence_claimed':False})
    qc_start=time.perf_counter();after_qc=geometry_qc(modeller.topology,positions,system);after_qc_seconds=time.perf_counter()-qc_start
    # Route qualification is numerical and uses the predeclared connection screen.
    # Remaining angle/LJ flags are retained and prohibit a broad validity claim.
    connection_screen_passed=(after_qc['O3prime_next_P_above_2p1_angstrom_count']==0 and after_qc['O3prime_next_P_below_1p3_angstrom_count']==0)
    output_start=time.perf_counter()
    with (out/'refined_with_hydrogens.pdb').open('w') as f:app.PDBFile.writeFile(modeller.topology,positions*unit.nanometer,f,keepIds=True)
    np.savez_compressed(out/'refined_heavy_coordinates.npz',coordinates_angstrom=positions[heavy_indices]*10,
                        residue_index=np.asarray([int(atoms[i].residue.id) for i in heavy_indices]),
                        atom_name=np.asarray([atoms[i].name for i in heavy_indices],dtype='U4'))
    restraint.setGlobalParameterDefaultValue(0,RESTraint_STAGES[-1][0])
    (out/'system_with_positional_restraint.xml').write_text(mm.XmlSerializer.serialize(system))
    output_seconds=time.perf_counter()-output_start
    result={'label':label,'original_input_path':str(path),'original_input_sha256':original_sha,
            'boundary':boundary,'topology_preflight':preflight,'forcefield_hashes':forcefield_hashes(),
            'OpenMM_version':mm.__version__,'platform':platform_name,'platform_properties':properties,
            'CPU_arithmetic':'OpenMM CPU double precision; no CPU energy context in topology preflight',
            'CUDA_arithmetic':'Explicit double precision; DeterministicForces=true; no blanket reproducibility claim',
            'solvent_model':'GBn2 implicit; solute dielectric1, solvent78.3, kappa0; no Mg, water box or condition/population inference',
            'hydrogen_preparation_pH':7.0,'hydrogen_preparation_seed':hydrogen_seed,
            'added_hydrogen_count':len(atoms)-len(heavy_indices),
            'maximum_hydrogen_preparation_heavy_atom_drift_angstrom':hydrogen_prep_heavy_drift*10,
            'chemical_reactions_simulated':0,'covalent_topology_during_minimization':'Fixed OL3 polymer bonds; no bond formation/breaking simulation',
            'restraint_reference':'Each structure own hydrogen-prepared heavy positions; no native-coordinate steering of generated inputs',
            'before_potential_energy_kJ_per_mol':before_energy,'before_force_RMS_kJ_per_mol_nm':before_force_RMS,
            'minimization_stages':stages,'before_geometry_QC':before_qc,'after_geometry_QC':after_qc,
            'predeclared_connection_screen_passed':connection_screen_passed,
            'heavy_atom_displacement_RMS_angstrom':float(np.sqrt(np.mean(np.sum((positions[heavy_indices]-before_positions[heavy_indices])**2,axis=1))))*10,
            'timing_seconds':{'topology_template_parameter_preflight':topology_seconds,'hydrogen_system_context_preparation':preparation_seconds,
                              'minimizer_stage_total':sum(s['seconds'] for s in stages),'before_QC':before_qc_seconds,'after_QC':after_qc_seconds,'output_IO':output_seconds,
                              'complete_case_wall':time.perf_counter()-start},
            'output_hashes':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()},
            'claim_scope':'Bounded numerical local refinement of declared OH fragment; no MD integration, equilibrium populations, learned efficacy, Mg-specific chemistry or complete stereochemistry certification'}
    dump(out/'case_refinement.json',result)
    del context,integrator
    return result

def cohort_worker(gpu,manifest,output,messages,maximum_seconds):
    try:
        reports=[];start=time.perf_counter()
        for sample_id in range(gpu,32,4):
            item=manifest['generated'][sample_id]
            reports.append(refine_case('sample_'+str(sample_id),item['remote_path'],manifest,output/f'sample_{sample_id:02d}',gpu,maximum_seconds=maximum_seconds))
        value={'gpu':gpu,'sample_ids':[r['label'] for r in reports],'complete_worker_wall_seconds':time.perf_counter()-start,
               'complete_cases':len(reports),'all_connections_pass_screen':all(r['predeclared_connection_screen_passed'] for r in reports),
               'sum_case_minimizer_stage_seconds':sum(r['timing_seconds']['minimizer_stage_total'] for r in reports)}
        dump(output/f'worker_{gpu}.json',value);messages.put({'event':'complete',**value})
    except BaseException:
        messages.put({'event':'error','gpu':gpu,'traceback':traceback.format_exc()});raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage-wall-seconds',type=float,default=120.0);args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text());output=args.out.resolve();output.mkdir(parents=True,exist_ok=False)
    if digest(Path(__file__))!=manifest['driver_sha256']:raise RuntimeError('Refinement driver changed')
    if mm.__version__!=manifest['OpenMM_version']:raise RuntimeError('OpenMM version differs from CPU qualification')
    if forcefield_hashes()!=manifest['forcefield_hashes']:raise RuntimeError('Forcefield XML source changed')
    if len(manifest['generated'])!=32:raise RuntimeError('Full fixed K32 cohort not declared')
    campaign_start=time.perf_counter()
    native=refine_case('native_control',Path(manifest['native']['packet_path']),manifest,output/'qualification/native',0,maximum_seconds=args.stage_wall_seconds)
    generated=refine_case('sample_0',Path(manifest['generated'][0]['remote_path']),manifest,output/'qualification/generated_0',0,maximum_seconds=args.stage_wall_seconds)
    if not native['predeclared_connection_screen_passed'] or not generated['predeclared_connection_screen_passed']:
        dump(output/'qualification_gate.json',{'status':'failed connection screen; K32 not started','native':native['predeclared_connection_screen_passed'],'generated_0':generated['predeclared_connection_screen_passed']})
        raise RuntimeError('Native/generated numerical qualification did not pass declared connection screen')
    dump(output/'qualification_gate.json',{'status':'CPU topology + CUDA finite-energy/bounded-minimizer and connection screen passed; K32 gate opened',
                                          'native':True,'generated_0':True,'full_chemical_validity_claimed':False})
    cohort=output/'cohort';cohort.mkdir();context=mp.get_context('spawn');messages=context.Queue()
    cohort_budget=min(1400.0,1500.0-(time.perf_counter()-campaign_start))
    if cohort_budget<=0:raise RuntimeError('Qualification consumed declared complete-campaign wall budget')
    processes=[context.Process(target=cohort_worker,args=(g,manifest,cohort,messages,args.stage_wall_seconds)) for g in range(4)]
    cohort_start=time.perf_counter()
    for p in processes:p.start()
    records=[]
    try:
        while len(records)<4:
            try:r=messages.get(timeout=20)
            except queue.Empty:
                if any(p.exitcode not in (None,0) for p in processes):raise RuntimeError('Cohort worker failed')
                if time.perf_counter()-cohort_start>cohort_budget:raise RuntimeError('Declared complete-campaign/cohort wall budget exceeded')
                continue
            if r['event']=='error':raise RuntimeError(r['traceback'])
            records.append(r)
        for p in processes:p.join(timeout=10)
        if not all(p.exitcode==0 for p in processes):raise RuntimeError('Cohort worker did not exit cleanly')
    finally:
        for p in processes:
            if p.is_alive():p.terminate();p.join(timeout=10)
    reports=[json.loads((cohort/f'sample_{i:02d}/case_refinement.json').read_text()) for i in range(32)]
    result={'status':'Executed bounded K32 OL3/GBn2 OH-fragment engineering refinement',
            'driver_sha256':digest(Path(__file__)),'manifest_sha256':digest(args.manifest),
            'source_inference_job':2297413,'fixed_unique_sample_ids':list(range(32)),
            'forcefield_hashes':forcefield_hashes(),'OpenMM_version':mm.__version__,
            'boundary':native['boundary']['boundary'],'CPU_preflight_scope':'Actual heavy-template + full-system parameterization without energy contexts',
            'CUDA_qualification_native_and_generated_0_passed':True,
            'all_32_case_energies_coordinates_and_forces_finite':True,
            'all_32_connection_screens_pass':all(r['predeclared_connection_screen_passed'] for r in reports),
            'before_O3prime_next_P_above_2p1_count':sum(r['before_geometry_QC']['O3prime_next_P_above_2p1_angstrom_count'] for r in reports),
            'after_O3prime_next_P_above_2p1_count':sum(r['after_geometry_QC']['O3prime_next_P_above_2p1_angstrom_count'] for r in reports),
            'after_O3prime_next_P_below_1p3_count':sum(r['after_geometry_QC']['O3prime_next_P_below_1p3_angstrom_count'] for r in reports),
            'worker_records':records,'cohort_wall_seconds':time.perf_counter()-cohort_start,'complete_campaign_wall_seconds':time.perf_counter()-campaign_start,
            'completed_case_report_hashes':{str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('case_refinement.json'))},
            'completed_worker_report_hashes':{str(p.relative_to(output)):digest(p) for p in sorted(cohort.glob('worker_*.json'))},
            'qualification_gate_SHA256':digest(output/'qualification_gate.json'),
            'energy_scope':'Per-stage before/after comparisons use the same positional-restraint Hamiltonian; energies across different restraint strengths are not presented as comparable',
            'scientific_scope':'Engineering local correction/cost under declared OH fragment and implicit solvent. Remaining angle/LJ flags and lack of minimizer convergence certificate are retained; no equilibrium/conditional efficacy/complete physical-validity claim'}
    dump(output/'refinement_campaign.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
