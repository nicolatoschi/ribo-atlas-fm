RIBO-ATLAS-FM Wave D frozen packet: evaluation-pipeline cost and robustness across RNA length

What runs (per GPU, sequentially over a static case list in partition_plan.json):
  1. Published pretrained RNAbpFlow generation (RNA3DB.ckpt, sha256 bb598a73...), K=32 samples,
     T=50 steps, seeds 420*(sample_id+1); original FlowModule.predict_step, Interpolant.sample,
     all-atom mapping and PDB writer from RNAbpFlow/ (byte-identical copy of inference packet
     6cbee47ef9de; upstream commit e8b1c07 plus the documented torch_native_scatter port).
     Anomaly mode is forced off and asserted after model construction.
  2. The project's OL3/GBn2 OH-fragment refinement (frozen openmm_rna_refinement.py from
     refinement packet e665e8ed11ae, imported unchanged; waveD_refine.py only generalises the
     158-nt literals) on the pre-declared sample IDs 0..R(L)-1, OpenMM CUDA double precision
     in a persistent spawned worker process on the same GPU (the pilot also used one OpenMM
     CUDA worker per GPU).
  3. Descriptive validity panel and C4' lDDT (rna_backbone_metrics.py, byte-identical) of every
     generated and refined sample against native coordinates from the training features.
One JSON per case (cases/<id>/case_result.json, atomic write); per-sample failures are
recorded and the rank continues; deadline guards skip (and record) work that would not finish.

Inputs: cases/<id>/<id>.fasta and case_input.npz (ss = the three training base-pair channels,
native atom28 coordinates/mask), derived with a restricted unpickler from
RNAbpFlow-Train.tar.gz Feature/train_pkl/<id>/<id>.pkl (sha256 of both that pickle and the
companion pkls/<id>_base/<id>_feat.pkl in case_table.json). waveD_cases_metadata.csv holds the
byte-identical CASP16_train.csv rows (LF line endings, as in the source) of the 255 training cases.
Calibration anchor P4P6_ref158: identical input and seeds to jobs 2297413/2298016.
Origins: 232 PDB chains and 23 bpRNA_RFAM_* entries. bpRNA_RFAM rows exist only in CASP16_train
and, per the RNAbpFlow README, come from bpRNA-1m(90) cross-distillation, so their reference
coordinates are distillation targets, not experimental structures; they are summarised separately.

SCOPE: every training case is in RNAbpFlow's own training metadata, and the P4-P6 family is in
training too. lDDT and validity numbers are PIPELINE QA, NOT held-out accuracy. No case in this
packet is verifiably outside RNAbpFlow training.

Verify before submission:  python3 -I verify_packet.py --packet <this dir> --checkpoint <RNA3DB.ckpt>
Aggregate after retrieval: python aggregate_waveD.py --results <job dir> --packet <this dir> --out <dir>
