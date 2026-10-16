#!/usr/bin/env python3
"""Length-generic form of the frozen OL3/GBn2 OH-fragment refinement (Wave D).

The physics, constants and helpers are imported unchanged from the byte-identical
copy of the frozen pilot driver openmm_rna_refinement.py (refinement packet
e665e8ed11ae, sha256 219f0267...): force-field files amber14/RNA.OL3.xml +
implicit/gbn2.xml, NoCutoff, solute/solvent dielectric 1/78.3, kappa 0, pH 7
template hydrogens, own-start heavy-atom positional restraints 1000 then
100 kJ/mol/nm^2 with at most 250 L-BFGS iterations each, tolerance 10
kJ/mol/nm, CUDA double precision with DeterministicForces, same hydrogen seed
rule (20261012 + sample_id), same QC definitions.

Only the 158-nt literals are generalised: residue count, heavy-atom count,
157 inter-residue links, the -157 e charge check and the per-stage wall cap
(120 s at 158 nt in the pilot; here stage_cap_seconds is supplied by the
caller). The O(N^2) Python exclusion loop of the LJ proximity screen is
vectorised; equality with the frozen geometry_qc is checked on the 158-nt
pilot structures (see smoke test). No convergence or chemical-validity claim.
"""
from __future__ import annotations

import io
import hashlib
import random
import time
import traceback
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit

import openmm_rna_refinement as frozen

FF_FILES = frozen.FF_FILES
STAGES = frozen.RESTraint_STAGES
TOLERANCE = frozen.TOLERANCE
finite = frozen.finite
summary = frozen.summary
digest = frozen.digest


class RefinementFailure(RuntimeError):
    def __init__(self, category, message):
        super().__init__(message)
        self.category = category


def expected_heavy_atoms(sequence, forcefield):
    total = 0
    for i, base in enumerate(sequence):
        name = base + ('5' if i == 0 else '3' if i == len(sequence) - 1 else '')
        total += sum(1 for a in forcefield._templates[name].atoms if a.element != app.element.hydrogen)
    return total


def canonical_fragment(text, sequence, forcefield):
    """Generic copy of frozen.canonical_fragment (chain A, canonical RNA, OH termini)."""
    original = []; keys = []; bases = {}; seen = set()
    for line in text.splitlines():
        if not line.startswith('ATOM  ') or line[21] != 'A':
            continue
        base = line[17:20].strip(); atom = line[12:16].strip().replace('*', "'")
        if base not in 'AUGC' or len(base) != 1:
            raise RefinementFailure('input_parse', 'Noncanonical RNA residue')
        if line[16].strip() not in ('', 'A'):
            continue
        if atom.startswith('H'):
            raise RefinementFailure('input_parse', 'Input must be the heavy-atom structure')
        key = (int(line[22:26]), line[26].strip())
        if key not in bases:
            keys.append(key); bases[key] = base
        if (key, atom) in seen:
            raise RefinementFailure('input_parse', 'Duplicate admitted input atom')
        seen.add((key, atom)); xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        finite(xyz, 'input coordinates'); original.append((key, atom, base, xyz))
    if ''.join(bases[k] for k in keys) != sequence or len(keys) != len(sequence):
        raise RefinementFailure('input_parse', 'Sequence does not match input structure')
    numbering = {key: i + 1 for i, key in enumerate(keys)}; lines = []; removed = []
    for key, atom, base, xyz in original:
        residue = numbering[key]
        if residue == 1 and atom in ('P', 'OP1', 'OP2', 'OP3'):
            removed.append({'original_author_residue': key[0], 'atom': atom}); continue
        element = 'P' if atom == 'P' else atom[0]
        serial = len(lines) + 1; name = (' ' + atom).ljust(4) if len(atom) < 4 else atom
        lines.append(f'ATOM  {serial:5d} {name} {base:>3s} A{residue:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}  ')
    heavy = expected_heavy_atoms(sequence, forcefield)
    if len(lines) != heavy or sorted(x['atom'] for x in removed) not in (['OP1', 'OP2', 'P'], ['OP1', 'OP2', 'OP3', 'P']):
        raise RefinementFailure('input_parse', f'Boundary atom accounting differs ({len(lines)} vs {heavy} heavy atoms)')
    result = '\n'.join(lines + ['TER', 'END', ''])
    return result, {'input_heavy_atom_records': len(original), 'admitted_RNA_residues': len(sequence),
                    'removed_5prime_phosphate_records': removed, 'OH_fragment_heavy_atoms': heavy,
                    'boundary': 'Explicit engineered 5prime-OH/3prime-OH fragment; first phosphate group removed, all interresidue phosphates retained',
                    'canonical_fragment_SHA256': hashlib.sha256(result.encode()).hexdigest()}


def topology_preflight(canonical, sequence, forcefield):
    """Generic copy of frozen.topology_preflight (no energy context)."""
    L = len(sequence)
    pdb = app.PDBFile(io.StringIO(canonical))
    top = app.Topology(); chain = top.addChain('A'); records = list(pdb.topology.residues()); maps = []; chosen = []
    for i, residue in enumerate(records):
        template_name = residue.name + ('5' if i == 0 else '3' if i == len(records) - 1 else '')
        template = forcefield._templates[template_name]
        observed = {a.name for a in residue.atoms()}
        expected = {a.name for a in template.atoms if a.element != app.element.hydrogen}
        if observed != expected:
            raise RefinementFailure('template_mismatch', 'Heavy-atom template mismatch at residue ' + str(i + 1))
        expected_external = {"O3'"} if i == 0 else {'P'} if i == len(records) - 1 else {'P', "O3'"}
        if {template.atoms[j].name for j in template.externalBonds} != expected_external:
            raise RefinementFailure('template_mismatch', 'Unexpected OL3 polymer connectivity')
        newres = top.addResidue(residue.name, chain, id=str(i + 1))
        atoms = {a.name: top.addAtom(a.name, a.element, newres) for a in template.atoms}
        for one, two in template.bonds:
            top.addBond(atoms[template.atoms[one].name], atoms[template.atoms[two].name])
        maps.append(atoms); chosen.append(template_name)
    for i in range(L - 1):
        top.addBond(maps[i]["O3'"], maps[i + 1]['P'])
    matches = forcefield.getMatchingTemplates(top)
    if [t.name for t in matches] != chosen:
        raise RefinementFailure('template_mismatch', 'OL3 matched templates differ from explicit boundary')
    system = forcefield.createSystem(top, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False,
                                     soluteDielectric=1.0, solventDielectric=78.3, implicitSolventKappa=0.0 / unit.nanometer)
    charge = 0.0; forces = []
    for force in system.getForces():
        forces.append(type(force).__name__)
        if isinstance(force, mm.NonbondedForce):
            for i in range(force.getNumParticles()):
                q, sigma, epsilon = force.getParticleParameters(i)
                values = [q.value_in_unit(unit.elementary_charge), sigma.value_in_unit(unit.nanometer), epsilon.value_in_unit(unit.kilojoule_per_mole)]
                finite(values, 'nonbonded parameters'); charge += values[0]
    if abs(charge + (L - 1)) > 1e-5:
        raise RefinementFailure('template_mismatch', 'OH fragment formal charge differs from retained phosphate count')
    return {'system_particles_including_template_hydrogens': system.getNumParticles(), 'matched_template_count': len(matches),
            'first_last_templates': [matches[0].name, matches[-1].name], 'interresidue_O3prime_P_bonds': L - 1,
            'formal_charge_elementary_charge': charge, 'force_classes': forces}


def geometry_qc(topology, positions, system):
    """Same outputs as frozen.geometry_qc; LJ exclusion test vectorised, bridges generic."""
    p = np.asarray(positions, dtype=float); finite(p, 'geometry coordinates')
    atoms = list(topology.atoms()); heavy = np.asarray([a.element != app.element.hydrogen for a in atoms], dtype=bool)
    keys = {(int(a.residue.id), a.name): a.index for a in atoms}
    nres = len(list(topology.residues()))
    bridges = np.asarray([np.linalg.norm(p[keys[(i, "O3'")]] - p[keys[(i + 1, 'P')]]) * 10 for i in range(1, nres)])
    bond_errors = []; angle_errors = []; nonbonded = None
    for force in system.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            for j in range(force.getNumBonds()):
                one, two, length, k = force.getBondParameters(j)
                if heavy[one] and heavy[two]:
                    bond_errors.append(abs(np.linalg.norm(p[one] - p[two]) - length.value_in_unit(unit.nanometer)) * 10)
        if isinstance(force, mm.HarmonicAngleForce):
            for j in range(force.getNumAngles()):
                one, two, three, target, k = force.getAngleParameters(j)
                if not (heavy[one] and heavy[two] and heavy[three]):
                    continue
                a = p[one] - p[two]; b = p[three] - p[two]
                actual = np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
                angle_errors.append(abs(actual - target.value_in_unit(unit.radian)) * 180 / np.pi)
        if isinstance(force, mm.NonbondedForce):
            nonbonded = force
    if nonbonded is None:
        raise RuntimeError('OL3 nonbonded force missing')
    selected = np.flatnonzero(heavy); sigs = []; eps = []
    for i in range(nonbonded.getNumParticles()):
        q, sigma, epsilon = nonbonded.getParticleParameters(i)
        sigs.append(sigma.value_in_unit(unit.nanometer)); eps.append(epsilon.value_in_unit(unit.kilojoule_per_mole))
    sigs = np.asarray(sigs); eps = np.asarray(eps); n = len(atoms)
    positive_heavy = heavy & (eps > 0)
    excluded_codes = set()
    for i in range(nonbonded.getNumExceptions()):
        one, two, q, sigma, epsilon = nonbonded.getExceptionParameters(i)
        a, b = min(one, two), max(one, two)
        excluded_codes.add(a * n + b)
    excluded_eligible = sum(1 for c in excluded_codes if positive_heavy[c // n] and positive_heavy[c % n])
    m = int(positive_heavy[selected].sum())
    eligible = m * (m - 1) // 2 - excluded_eligible
    clashes = 0
    for offset in range(0, len(selected), 128):
        first = selected[offset:offset + 128]
        d = np.linalg.norm(p[first, None, :] - p[selected][None, :, :], axis=-1)
        threshold = .75 * 2 ** (1 / 6) * (sigs[first, None] + sigs[selected][None, :]) / 2
        mask = (first[:, None] < selected[None, :]) & (eps[first, None] > 0) & (eps[selected][None, :] > 0)
        rows, cols = np.nonzero((d < threshold) & mask)
        clashes += sum(1 for r, c in zip(rows, cols) if int(first[r]) * n + int(selected[c]) not in excluded_codes)
    return {'heavy_atom_count': int(heavy.sum()), 'all_atom_count': len(atoms),
            'O3prime_next_P_angstrom': summary(bridges), 'O3prime_next_P_above_2p1_angstrom_count': int((bridges > 2.1).sum()),
            'O3prime_next_P_below_1p3_angstrom_count': int((bridges < 1.3).sum()),
            'heavy_bond_absolute_error_to_OL3_target_angstrom': summary(bond_errors),
            'heavy_bond_error_above_0p15_angstrom_count': int((np.asarray(bond_errors) > .15).sum()),
            'heavy_angle_absolute_error_to_OL3_target_degree': summary(angle_errors),
            'heavy_angle_error_above_15_degree_count': int((np.asarray(angle_errors) > 15).sum()),
            'heavy_nonbonded_LJ_proximity_screen_count': clashes, 'heavy_nonbonded_LJ_screen_eligible_pairs': eligible,
            'LJ_screen_definition': 'r < 0.75*2^(1/6)*Lorentz-combined OL3 sigma; positive-epsilon heavy atoms only; all explicit nonbonded exceptions excluded; descriptive screen'}


def refine_sample(pdb_path, sequence, sample_id, out, platform_name='CUDA', device_index=0,
                  stage_cap_seconds=120.0, write_system_xml=False, progress=None):
    """Refine one generated structure. Returns a record; never raises."""
    start = time.perf_counter(); timing = {}; phase = 'start'
    record = {'sample_id': sample_id, 'input_path': str(pdb_path), 'platform': platform_name,
              'stage_wall_cap_seconds': stage_cap_seconds, 'status': 'failed'}
    try:
        phase = 'input'
        text = Path(pdb_path).read_text(); record['input_sha256'] = digest(pdb_path)
        out = Path(out); out.mkdir(parents=True, exist_ok=False)
        ff = app.ForceField(*FF_FILES)
        canonical, boundary = canonical_fragment(text, sequence, ff)
        (out / 'OH_fragment_heavy_input.pdb').write_text(canonical)
        phase = 'topology'; t0 = time.perf_counter()
        preflight = topology_preflight(canonical, sequence, ff)
        timing['topology_template_parameter_preflight'] = time.perf_counter() - t0
        platform, properties = frozen.platform_configuration(platform_name, device_index)
        phase = 'hydrogens_context'; t0 = time.perf_counter()
        pdb = app.PDBFile(io.StringIO(canonical)); modeller = app.Modeller(pdb.topology, pdb.positions)
        input_heavy = {(int(a.residue.id), a.name): np.asarray(pdb.positions[a.index].value_in_unit(unit.nanometer))
                       for a in pdb.topology.atoms()}
        hydrogen_seed = 20261012 + int(sample_id)
        np.random.seed(hydrogen_seed); random.seed(hydrogen_seed)
        modeller.addHydrogens(ff, pH=7.0, platform=platform)
        if len(list(modeller.topology.atoms())) != preflight['system_particles_including_template_hydrogens']:
            raise RefinementFailure('hydrogen_count', 'Placed hydrogen count differs from templates')
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False,
                                 soluteDielectric=1.0, solventDielectric=78.3, implicitSolventKappa=0.0 / unit.nanometer)
        original_positions = np.asarray(modeller.positions.value_in_unit(unit.nanometer))
        finite(original_positions, 'hydrogen-prepared coordinates')
        restraint = mm.CustomExternalForce('0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
        restraint.addGlobalParameter('k', STAGES[0][0])
        for name in ('x0', 'y0', 'z0'):
            restraint.addPerParticleParameter(name)
        atoms = list(modeller.topology.atoms()); heavy_indices = [a.index for a in atoms if a.element != app.element.hydrogen]
        drift = max(float(np.linalg.norm(original_positions[i] - input_heavy[(int(atoms[i].residue.id), atoms[i].name)])) for i in heavy_indices)
        if drift > 1e-6:
            raise RefinementFailure('hydrogen_moved_heavy', 'Hydrogen preparation moved admitted heavy atoms')
        for i in heavy_indices:
            restraint.addParticle(i, original_positions[i].tolist())
        system.addForce(restraint)
        integrator = mm.VerletIntegrator(.001 * unit.picoseconds)
        context = mm.Context(system, integrator, platform, properties); context.setPositions(modeller.positions)
        timing['hydrogen_system_context_preparation'] = time.perf_counter() - t0
        if progress:
            progress('context_created')
        phase = 'before_energy'
        before_energy, before_positions, before_force_RMS = frozen.energy_state(context)
        phase = 'before_QC'; t0 = time.perf_counter()
        before_qc = geometry_qc(modeller.topology, before_positions, system); timing['before_QC'] = time.perf_counter() - t0
        stages = []
        for strength, iterations in STAGES:
            phase = f'minimize_k{int(strength)}'
            context.setParameter('k', strength)
            energy_before, _, _ = frozen.energy_state(context); stage_start = time.perf_counter()
            reporter = frozen.BoundedReporter(stage_start, stage_cap_seconds)
            mm.LocalEnergyMinimizer.minimize(context, TOLERANCE, iterations, reporter)
            energy_after, positions, force_RMS = frozen.energy_state(context)
            elapsed = time.perf_counter() - stage_start
            stages.append({'restraint_k_kJ_per_mol_nm2': strength, 'maximum_iterations_requested': iterations,
                           'minimizer_tolerance_requested_kJ_per_mol_nm': TOLERANCE, 'reporter_calls': reporter.calls,
                           'maximum_iteration_index_reported': reporter.maximum_iteration, 'seconds': elapsed,
                           'potential_energy_before_kJ_per_mol': energy_before, 'potential_energy_after_kJ_per_mol': energy_after,
                           'force_RMS_after_kJ_per_mol_nm': force_RMS, 'aborted_for_wall_cap': reporter.aborted_for_wall,
                           'convergence_claimed': False})
            record['minimization_stages'] = stages
            if reporter.aborted_for_wall:
                raise RefinementFailure('stage_wall_cap', f'stage k={strength} exceeded {stage_cap_seconds:.0f} s')
            if energy_after > energy_before + max(1e-4, abs(energy_before) * 1e-8):
                raise RefinementFailure('energy_increase', 'Energy increased under the same restraint Hamiltonian')
        timing['minimizer_stage_total'] = sum(s['seconds'] for s in stages)
        if progress:
            progress('minimized')
        phase = 'after_QC'; t0 = time.perf_counter()
        after_qc = geometry_qc(modeller.topology, positions, system); timing['after_QC'] = time.perf_counter() - t0
        phase = 'output'; t0 = time.perf_counter()
        with (out / 'refined_with_hydrogens.pdb').open('w') as handle:
            app.PDBFile.writeFile(modeller.topology, positions * unit.nanometer, handle, keepIds=True)
        heavy_xyz = positions[heavy_indices] * 10
        np.savez_compressed(out / 'refined_heavy_coordinates.npz', coordinates_angstrom=heavy_xyz,
                            residue_index=np.asarray([int(atoms[i].residue.id) for i in heavy_indices]),
                            atom_name=np.asarray([atoms[i].name for i in heavy_indices], dtype='U4'))
        if write_system_xml:
            restraint.setGlobalParameterDefaultValue(0, STAGES[-1][0])
            (out / 'system_with_positional_restraint.xml').write_text(mm.XmlSerializer.serialize(system))
        timing['output_IO'] = time.perf_counter() - t0
        timing['complete_case_wall'] = time.perf_counter() - start
        record.update({'status': 'ok', 'boundary': boundary, 'topology_preflight': preflight,
                       'forcefield_hashes': frozen.forcefield_hashes(), 'OpenMM_version': mm.__version__,
                       'platform_properties': properties, 'hydrogen_preparation_seed': hydrogen_seed,
                       'added_hydrogen_count': len(atoms) - len(heavy_indices),
                       'before_potential_energy_kJ_per_mol': before_energy, 'before_force_RMS_kJ_per_mol_nm': before_force_RMS,
                       'before_geometry_QC': before_qc, 'after_geometry_QC': after_qc,
                       'predeclared_connection_screen_passed': (after_qc['O3prime_next_P_above_2p1_angstrom_count'] == 0 and
                                                                after_qc['O3prime_next_P_below_1p3_angstrom_count'] == 0),
                       'heavy_atom_displacement_RMS_angstrom': float(np.sqrt(np.mean(np.sum((positions[heavy_indices] - before_positions[heavy_indices]) ** 2, axis=1)))) * 10,
                       'refined_C4prime_angstrom': heavy_xyz[[k for k, i in enumerate(heavy_indices) if atoms[i].name == "C4'"]].tolist(),
                       'output_hashes': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}})
        del context, integrator
    except BaseException as error:  # recorded per sample; the rank continues
        record['failure_category'] = getattr(error, 'category', 'exception')
        record['failure_phase'] = phase
        record['failure_message'] = f'{type(error).__name__}: {error}'
        record['traceback'] = traceback.format_exc()[-4000:]
        if isinstance(error, KeyboardInterrupt):
            record['failure_category'] = 'interrupted'
    timing.setdefault('complete_case_wall', time.perf_counter() - start)
    record['timing_seconds'] = timing
    return record
