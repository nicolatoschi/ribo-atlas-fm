#!/usr/bin/env python3
"""Reference metric implementation for the mapped public RNA geometry pilot.

C4-prime distance-difference score: reference neighbours within15 A,
four tolerances0.5/1/2/4 A, per-residue then construct mean. This is an
explicit selected-backbone-atom score, not an all-atom lDDT result.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def c4prime_lddt(prediction, reference, cutoff=15.0):
    prediction=np.asarray(prediction, dtype=float)
    reference=np.asarray(reference, dtype=float)
    if prediction.shape!=reference.shape or prediction.ndim!=2 or prediction.shape[1]!=3:
        raise ValueError('Coordinates require matching N-by-3 arrays')
    if not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError('All admitted reference/prediction coordinates must be finite')
    dref=np.linalg.norm(reference[:,None,:]-reference[None,:,:],axis=-1)
    dpred=np.linalg.norm(prediction[:,None,:]-prediction[None,:,:],axis=-1)
    mask=(dref<cutoff)&~np.eye(len(reference),dtype=bool)
    delta=np.abs(dpred-dref)
    score=sum((delta<t).astype(float) for t in (0.5,1.0,2.0,4.0))/4
    neighbours=mask.sum(axis=1)
    included=neighbours>0
    if not included.any():
        raise ValueError('Reference has no eligible local distance pairs')
    per_residue=(score*mask).sum(axis=1)/np.maximum(neighbours,1)
    return dict(mean_score=float(per_residue[included].mean()),
        eligible_residues=int(included.sum()), directed_reference_pairs=int(mask.sum()),
        per_residue_scores=per_residue.tolist(), reference_cutoff_angstrom=cutoff,
        distance_tolerances_angstrom=[0.5,1.0,2.0,4.0], atom_selection='C4-prime',
        aggregation='Equal weight per eligible residue, then construct; family mean is applied across constructs')


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path(__file__).resolve().parent/'data')
    a=p.parse_args(); archive=np.load(a.data/'1GID_AB_coordinates.npz')
    # Inspect explicit archive names; atom order is residue-mapped by prepare_rna_pilot.
    ref=archive['A_C4prime']; target=archive['B_C4prime']
    cross=c4prime_lddt(target,ref)
    identity=c4prime_lddt(ref,ref)['mean_score']
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    rotated=c4prime_lddt(ref@rotation+np.array([17.,-8.,3.]),ref)['mean_score']
    assert identity==1.0 and abs(rotated-1.0)<1e-12
    result=dict(status='Executed on public1GID reference coordinates',
        interpretation='Two crystal copies of one RNA; metric/curation qualification, not learned-model accuracy or multi-state evidence',
        chainB_against_chainA=cross,
        checks=dict(self_score=identity, rigid_transform_score=rotated))
    (a.data/'c4prime_metric_qc.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='chainB_against_chainA'}))
    print('C4-prime chain-copy score',cross['mean_score'])


if __name__=='__main__':main()
