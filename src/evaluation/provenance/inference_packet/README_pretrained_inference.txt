Published pretrained RNAbpFlow inference campaign

Code: inference_benchmark.py. The checkpoint is external to this packet and is never copied into it. Official inference source is pinned to GitHub commit e8b1c077f7951d4d6ebe00c851d22c37777dfc9f, with the previously documented narrow Torch-native grouped-reduction portability import. The frozen training baseline is not changed.

Actual CPU qualification on JUPITER passed:
  Strictly loaded official RNA3DB.ckpt into the published inference FlowModule.
  16,876,313 parameters, all finite.
  Sequence onehot [1,158,4]; pair channels [1,158,158,3]; 28 all-atom slots.
  Actual test JSON is preserved here as qualification/cpu_checkpoint_input_qualification.json. Its recorded driver, all 22 Python source files and two inputs match this packet byte-for-byte; only their directory locations change.
  CPU qualification performs no sampling or GPU computation.

Checkpoint:
  Official https://zenodo.org/records/18305861, verified full archive MD5 f79ce1555126ee77651594b65a95cb4b.
  RNA3DB.ckpt is 203034292 bytes, SHA256 bb598a736193734541f63c59dae75d45db8c04f01a8414a758677367c4b1e213, checked on both local and remote copies.

Engineering input:
  P4P6_ref158 canonical PDB1GID core sequence and48 structural-reference annotation pairs. The single map is repeated to the three published input channels exactly as the authors' one-map interface supports. It is not a predicted secondary structure or a chemical-evidence-inferred map. P4-P6-family examples occur in baseline training data, so this is an in-distribution public-model engineering test.

Generation:
  Original FlowModule.predict_step and Interpolant.sample, trained checkpoint weights,50 sampling steps, public all-atom mapping and original PDB writer. Sample IDs0..31 and seeds420*(sample_id+1) are identical across1/2/4 GPU worker counts. Each worker loads the checkpoint independently, sets eval/no_grad and the published float32 matmul precision high.
  The wrapper normalises seed to its fixed Python integer value instead of passing a CUDA scalar to NumPy. It captures the return of the original sampler for checks and .npy export, without replacing decoder/sampler equations.
  Each worker generates one separate initial sample before the common cohort barrier. These samples have separate paths and negative IDs. The requested cohort consists only of IDs 0..K-1. Process-start timing includes checkpoint load, transfer and this initial sample; total run wall time includes all these phases. The cohort wall clock starts at the common barrier. The maximum worker sum of original sampler call times covers only the requested cohort; it is a derived sum rather than a complete end-to-end wall clock.

Parent reviews/submits one allocated node with4 visible GPUs; no jobs are submitted by this wrapper or the data agent:
  env/bin/python -u inference_benchmark.py \
    --source FROZEN_INFERENCE_SOURCE \
    --checkpoint /e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/pretrained_checkpoint/RNA3DB.ckpt \
    --out /ABSOLUTE/NEW_JOB_NAMESPACE \
    --K 32 --timesteps 50 --GPU-counts 1 2 4 --repetitions 3 --smoke-K 8 \
    --campaign-max-seconds 1500

The initial K8/one-GPU smoke produces real generated structures and finite-coordinate checks. Its observed timing selects up to three balanced complete1/2/4 GPU repetition cycles; a further cycle is skipped when the observed projected duration exceeds the campaign time target. The actual completed repetition count is recorded; it is never inferred from the request. GPU allocation cap/accounting remain the parent's Slurm submission responsibility.

Measurements:
  Fresh process startup until all workers ready; checkpoint load/H2D; warm-up; cohort sampling wall time; maximum worker sum of original published sampling times; original predict_step duration including PDB writes; peak allocated/reserved HBM per worker.
  Generation-only timing covers the original sampler, full50 decoder steps, final all-atom construction and CPU coordinate transfer. Cohort wall time additionally includes PDB generation/writes, descriptive QC and .npy/checksum I/O. Both are reported explicitly.
  Every sample must have finite coordinates with shape[1,158,28,3]. Descriptive geometry checks report adjacent P/C4' distances, O3'(i)-P(i+1) connections and a coarse nonadjacent C4' proximity screen. These are not an all-atom energy test or a learned-efficacy assessment.
  Coordinate arrays and PDB outputs have SHA256 hashes. Driver, input files, public Python source and checkpoint hashes are recorded. Existing example Predictions directories are never written.
  Corresponding sample arrays across GPU counts are compared at absolute tolerance1e-3 angstrom plus relative tolerance1e-4, with maximum/RMS component differences reported. Bitwise determinism is not claimed.

Outputs:
  NEW_JOB_NAMESPACE/smoke_K8_G1/
  NEW_JOB_NAMESPACE/rep_N_G1_K32/, rep_N_G2_K32/, rep_N_G4_K32/
  Each has samples/Sample_ID.pdb and.npy, separate warm-up folders, worker JSON and run_measurements.json.
  campaign_measurements.json records provenance, actual completed runs, timing/QC and coordinate comparisons. Sampling throughput is independent conformer-generation scaling, distinct from synchronous gradient-training scaling.
