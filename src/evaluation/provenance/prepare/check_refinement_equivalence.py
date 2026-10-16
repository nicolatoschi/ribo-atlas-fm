#!/usr/bin/env python3
"""Local CPU check: generic Wave D refinement == frozen pilot refine_case on a 158-nt input.

Runs the frozen openmm_rna_refinement.refine_case (packet e665e8ed11ae) and
waveD_refine.refine_sample on the same job-2297413 P4-P6 generated PDB with the
OpenMM CPU platform, then compares stage energies, iteration counts, every
before/after QC field (this also checks the vectorised LJ screen) and the final
heavy-atom coordinates. CPU only; no GPU, no job submission.

OpenMM Modeller.addHydrogens is not bitwise reproducible run-to-run (even the frozen code
against itself), so --fixed-hydrogens places hydrogens once and hands the identical
hydrogenated Modeller state to both code paths; --iterations shortens both minimiser stages
identically (code-path identity test; the full protocol is the default).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=Path, required=True, help='directory with openmm_rna_refinement.py and waveD_refine.py')
    parser.add_argument('--pdb', type=Path, required=True)
    parser.add_argument('--refinement-manifest', type=Path, required=True)
    parser.add_argument('--sample-id', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fixed-hydrogens', action='store_true')
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--cpu-threads', type=int, default=None,
                        help='override the frozen CPU Threads=2 property for BOTH paths (1 = bitwise deterministic)')
    args = parser.parse_args()
    sys.path.insert(0, str(args.src.resolve()))
    import openmm_rna_refinement as frozen
    import waveD_refine
    if args.iterations:
        stages = tuple((k, args.iterations) for k, _ in frozen.RESTraint_STAGES)
        frozen.RESTraint_STAGES = stages; waveD_refine.STAGES = stages
    if args.cpu_threads:
        original_platform = frozen.platform_configuration
        def platform_configuration(name, gpu):
            platform, properties = original_platform(name, gpu)
            if name == 'CPU':
                properties = dict(properties, Threads=str(args.cpu_threads))
                platform.setPropertyDefaultValue('Threads', str(args.cpu_threads))
            return platform, properties
        frozen.platform_configuration = platform_configuration
    if args.fixed_hydrogens:
        from openmm import app
        original = app.Modeller.addHydrogens; cache = {}
        def fixed(self, forcefield=None, pH=7.0, variants=None, platform=None, residueTemplates=dict()):
            if 'state' not in cache:
                result = original(self, forcefield, pH=pH, variants=variants, platform=platform)
                cache['state'] = (self.topology, self.positions)
                return result
            self.topology, self.positions = cache['state']
            return None
        app.Modeller.addHydrogens = fixed
    manifest = json.loads(args.refinement_manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.perf_counter()
    frozen_result = frozen.refine_case(f'sample_{args.sample_id}', args.pdb, manifest, args.out / 'frozen', 0,
                                       platform_name='CPU', maximum_seconds=7200)
    frozen_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    generic = waveD_refine.refine_sample(args.pdb, manifest['sequence'], args.sample_id, args.out / 'generic',
                                         platform_name='CPU', device_index=0, stage_cap_seconds=7200)
    generic_seconds = time.perf_counter() - t0
    assert generic['status'] == 'ok', generic.get('traceback')
    a = np.load(args.out / 'frozen/refined_heavy_coordinates.npz'); b = np.load(args.out / 'generic/refined_heavy_coordinates.npz')
    stage_keys = ('potential_energy_before_kJ_per_mol', 'potential_energy_after_kJ_per_mol', 'force_RMS_after_kJ_per_mol_nm',
                  'reporter_calls', 'maximum_iteration_index_reported')
    stages_equal = all(fs[k] == gs[k] for fs, gs in zip(frozen_result['minimization_stages'], generic['minimization_stages']) for k in stage_keys)
    result = {
        'input_pdb': str(args.pdb), 'platform': f'CPU (OpenMM Threads={args.cpu_threads or 2}, DeterministicForces=true)',
        'fixed_hydrogens': args.fixed_hydrogens, 'iterations_per_stage': args.iterations or 250, 'cpu_threads': args.cpu_threads or 2,
        'before_QC_identical': frozen_result['before_geometry_QC'] == generic['before_geometry_QC'],
        'after_QC_identical': frozen_result['after_geometry_QC'] == generic['after_geometry_QC'],
        'stage_energies_iterations_identical': stages_equal,
        'before_energy_identical': frozen_result['before_potential_energy_kJ_per_mol'] == generic['before_potential_energy_kJ_per_mol'],
        'final_heavy_coordinates_identical': bool(np.array_equal(a['coordinates_angstrom'], b['coordinates_angstrom'])),
        'refined_pdb_identical': frozen_result['output_hashes']['refined_with_hydrogens.pdb']['sha256'] == generic['output_hashes']['refined_with_hydrogens.pdb']['sha256'],
        'connection_screen': [frozen_result['predeclared_connection_screen_passed'], generic['predeclared_connection_screen_passed']],
        'frozen_wall_seconds': frozen_seconds, 'generic_wall_seconds': generic_seconds,
        'frozen_timing': frozen_result['timing_seconds'], 'generic_timing': generic['timing_seconds'],
        'frozen_stages': frozen_result['minimization_stages'], 'generic_stages': generic['minimization_stages'],
        'after_QC': generic['after_geometry_QC']}
    result['all_identical'] = all(result[k] for k in ('before_QC_identical', 'after_QC_identical', 'stage_energies_iterations_identical',
                                                      'before_energy_identical', 'final_heavy_coordinates_identical', 'refined_pdb_identical'))
    (args.out / 'equivalence.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('_stages') and k != 'after_QC'}, indent=1))


if __name__ == '__main__':
    main()
