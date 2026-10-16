RIBO-specific ensemble-mean regularizer engineering qualification

Scientific mapping
Each ensemble member is an independent corruption of the SAME real native RNA,
followed by one complete published RNAbpFlow denoising prediction. The same
synthetic observed profile/error/mask/chemistry input conditions every member
of a construct. The predicted frames/rotations/torsions produce native-order
C4' coordinates. This is a one-step denoised-ensemble regularizer; it is separate
from the final50-step generated-ensemble objective and does not claim to execute
full generative training. K_train is declared as1,8or32. Mean-profile squared
loss at finiteK contains a MonteCarlo variance/K term; population-mean unbiasedness
is not claimed. No training exposure is selected to consume an allocation quota.

Frozen forward projection
The4,673-parameter random fixed head takes only C4' coordinates, chemistry ID
and native residue validity. Its inputs exclude observed values, standard errors
and the probe observation mask. It has no trainable parameters in this campaign.
Geometric invariants plus chemistry project to a synthetic reactivity profile.
This benchmarks a frozen forward-head interface; a scientifically trained,
training-only cross-fitted head must be established separately and frozen before
joint conditioning. No learned head, experimental cross-fit or efficacy claim.

Inverse path and matched controls
The full16,876,313-parameter published decoder and complete SE3+auxiliary loss
remain unchanged. A330,752-parameter four-feature probe encoder/cross-attention
module has matched inactive/active computational paths; inactive gates its
residual tozero. Total model17,211,738parameters, of which17,207,065inverse
parameters train and4,673forward-head parameters freeze. A head-state SHA is
recorded and checked unchanged. Active adapter gradients must be finite and
nonzero on timed updates; inactive adapter gradients are exactlyzero by design.

Exact streamed objective
L=geometry_weight*mean_K(published geometry loss)
  +bulk_weight*sum(weights*(mean_K(profile)-target)^2)/sum(weights).
1. No-grad first pass obtains the ensemble mean.
2. alpha=2*weights*(mean-target)/sum(weights) is detached.
3. Repeat every corruption/dropout seed and backpropagate
   geometry_weight*geometry_loss/K +bulk_weight*dot(alpha,profile)/K.
This is the exact derivative of the finiteK loss when replayed predictions
match; only one geometric graph is resident. Replay profiles must satisfy the
declared numerical tolerance (atol1e-5,rtol1e-4); CPU toy tests establish algebra,
and GPU results establish actual full-decoder replay behavior. The logged loss
is the true mean-profile MSE, not the replay surrogate's scalar value. DDP.no_sync
is used until the final ensemble member; optimizer updates once per construct.
No head parameter gradient is permitted. A separate bulk-only inverse/adapter
gradient check follows one geometric warmup and does not enter timing averages.

Actual preliminary proof and planned allocation
cpu_gradient_test.json was actually produced on the existing isolated JUPITER
Python environment using only tiny CPU tensors, without a GPU job. It compares
retained-graph and streamed gradients for inactive/active paths atK1/8/32.
The full-native GPU campaign is a new parent-submitted qualification. It uses
archived1g59_B75nt,1l8v_B157nt,2r8s_R159nt and7ptk_B720nt frames/features.
Targets are explicitly synthetic systems profiles, not experimental chemistry
assigned to these structures. Real published target frames/torsions/pairs provide
the geometric supervision. First-pass/model/loss/backward/AdamW/CUDA completion
and all numerical checks are timed. Data are preloaded; loader/shard throughput
and final50-step inference receive separate measurements. Actual walltime
accounting includes import, lookup, warmup, data and report costs.

First qualification:1node/4GPUs,1warmup+2timed updates,1repetition, everycase
and inactive/active K1/8/32path plus bulk-only checks.20min ceiling=4/3GPUh.
Maximum reserved HBM must remain below85%device capacity. This small test
establishes the genuine ensemble objective's cost/memory/gradient path. Further
production profiles use admitted corpus length distributions and data loading;
the parent selects any additional repetitions or scales after inspecting results.
