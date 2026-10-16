#!/usr/bin/env python3
"""Re-run build_facts_v2.py unchanged, with its hard-coded paths remapped.

build_facts_v2.py is shipped byte-identical to the copy that produced facts_v2.json. It hard-codes
the build machine's project root (ROOT) and derives ET, AO and SRC from it at import. It reads its
inputs from SRC, globs rank reports under AO (host RSS and gradient-spike facts) and writes to
ROOT/work/rebuild_v2/facts_v2.json.

This wrapper imports the builder as a module and replaces every module-level path under the original
root, then calls its main(). No line of the builder changes.
  - With --evidence-root (an execution pack's evidence/ tree, which mirrors the project layout), every
    path is mapped to the same relative path under that root.
  - Without it, the inputs come from this repository: results/ and resource_plan/, with the globbed
    rank reports under results/rank_reports/.
The output goes to --out. With --compare, every value must equal the shipped facts file. The only
permitted difference is the 'sources' field, whose paths must agree after the roots are stripped.
"""
import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
REPO_INPUTS = {
    'waveB': 'results/waveB_job2339575/waveB_analysis.json',
    'gate': 'results/waveB_job2339575/gate_B0.json',
    'B1': 'results/waveB_job2339575/B1_reliability_qualification.json',
    'waveC': 'results/waveC_job2342412/waveC_analysis.json',
    'waveD': 'results/waveD_job2340563/waveD_findings.json',
    'waveF': 'results/waveF_forward_head/forward_head_results.json',
    'plan': 'resource_plan/resource_plan_v2.json',
    'rel2319595': 'results/job2319595_reliability/reliability_qualification_summary.json',
    'admission': 'results/data_admission/actual_admission_evidence.json',
    'readiness': 'results/data_admission/clean_curriculum_readiness_audit.json',
    'rfam': 'results/data_admission/rfam15_1_inventory.json',
    'expansion': 'results/data_expansion/actual_admission_evidence.json',
    'expansion_breakdown': 'results/data_expansion/expansion_breakdown.json',
}
REPO_GLOBALS = {'AO': 'results/rank_reports', 'ET': 'results/rank_reports'}  # AO is the only one used after import


def load_builder():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('build_facts_v2', HERE / 'build_facts_v2.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def strip(sources, root):
    root = str(root).rstrip('/') + '/'
    return {k: (v[len(root):] if v.startswith(root) else v) for k, v in sources.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--evidence-root', type=Path, help='execution-pack evidence/ directory (mirrors the project tree)')
    p.add_argument('--plan', type=Path, help='use this resource_plan_v2.json instead (e.g. one just rebuilt)')
    p.add_argument('--out', type=Path, required=True, help='where to write the rebuilt facts JSON')
    p.add_argument('--compare', type=Path, help='shipped facts_v2.json to compare against')
    a = p.parse_args()
    mod = load_builder()
    orig = mod.ROOT
    path_globals = {k for k, v in vars(mod).items() if isinstance(v, Path) and (v == orig or orig in v.parents)}
    if path_globals != {'ROOT', 'ET', 'AO'}:
        sys.exit(f'build_facts_v2.py path globals changed ({sorted(path_globals)}); update this wrapper')
    if set(mod.SRC) != set(REPO_INPUTS):
        sys.exit(f'build_facts_v2.py inputs changed ({sorted(set(mod.SRC) ^ set(REPO_INPUTS))}); update this wrapper')
    if a.evidence_root:
        root = a.evidence_root.resolve()
        src = {k: root / Path(v).relative_to(orig) for k, v in mod.SRC.items()}
        new_globals = {k: root / getattr(mod, k).relative_to(orig) for k in ('ET', 'AO')}
    else:
        root = REPO
        src = {k: REPO / rel for k, rel in REPO_INPUTS.items()}
        new_globals = {k: REPO / rel for k, rel in REPO_GLOBALS.items()}
    if a.plan:
        src['plan'] = a.plan.resolve()
    missing = [str(v) for v in list(src.values()) + list(new_globals.values()) if not Path(v).exists()]
    if missing:
        sys.exit('missing inputs: ' + ', '.join(missing))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'work/rebuild_v2').mkdir(parents=True)
        mod.ROOT = tmp                      # output location only
        for k, v in new_globals.items():
            setattr(mod, k, v)
        mod.SRC = src
        mod.main()
        a.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp / 'work/rebuild_v2/facts_v2.json', a.out)
    if a.compare:
        new = json.loads(a.out.read_text()); old = json.loads(a.compare.read_text())
        ns, os_ = new.pop('sources'), old.pop('sources')
        same = new == old
        new_rel, old_rel = strip(ns, root), strip(os_, orig)
        if a.plan:
            new_rel.pop('plan', None); old_rel.pop('plan', None)
        report = {'values_identical_except_sources': same, 'request_NH': new.get('request_NH'), 'request_GPUh': new.get('request_GPUh'),
                  'model_total_NH': new.get('model_total_NH'),
                  'sources_same_relative_paths': (new_rel == old_rel) if a.evidence_root else 'n/a (repository inputs)',
                  'plan_input': str(a.plan) if a.plan else 'as recorded',
                  'differing_top_level_keys': sorted(k for k in set(new) | set(old) if new.get(k) != old.get(k))}
        print(json.dumps(report, indent=1))
        if not same or (a.evidence_root and new_rel != old_rel):
            sys.exit(1)


if __name__ == '__main__':
    main()
