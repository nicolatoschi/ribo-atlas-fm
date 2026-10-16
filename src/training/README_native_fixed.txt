EXACT NATIVE FULL-GENERATED FIXED-WORK QUALIFICATION
The batched training driver, loss/solver/gradient/replay/finite/HBM guards and
all native RNAbpFlow code are byte-identical to submitted8d2. No optimized
quaternion converter is used. This retains the exact native eigensolver primal
and the qualified analytic isolated-largest-eigenvector derivative.

Actual175nt5zwm_B, fixed global128 stochastic construct instances:
 N8 K32/T50:32ranks, micro1/local4, accumulation4,45minute cap;
 N32K32/T50:128ranks, micro1/local1, accumulation1,25minute cap;
 N8 K8/T50:32ranks, micro1/local4, accumulation4,15minute cap.
K32N8 andN32 cover identical logicalIDs0..127, per-case priors/synthetic
observations and retained teacher frames. K8 is a separate unit-cost job,
not a third point on the K32 strong-scaling curve. One initial update each,
no warm repetitions, no sustained-throughput or convergence claim.
Maximum total=(8*45/60+32*25/60+8*15/60)*4=256/3GPU-hours.
Root reviews/uploads/verifies/submits; preparation is not execution.

New CPU fixture retains128 actual singleton corruptions of this physical RNA;
allFloat inputs and corrupted frames finite, all residue masks valid. Actual
same-source CPUbatch2/K2/T2 generated-gradient equality, replay, nonzero full
model/probe and teacher tests passed before freezing. Original157nt fixture,
old CPU proofs and base metadata feature files remain historical evidence.
The standalone launcher adds only metadata/hash capture, changing no tensor,
RNG, loss or optimization operation. Per-case observation hashes are captured
before model setup/timing. Teacher/hash report decoration occurs after timing.
Reported model-call counters include real forward/backward/recompute/teacher.

Fixed global case instances are not128 independent RNAs or biological samples.
Observations are synthetic and the fixed geometry/chemistry head is random.
No pairing with unrelated177nt measured probing constructs is claimed.
Teacher physical inputs match across layouts; native teacher train-mode dropout
RNG layout differs with nodes/accumulation. There is no bitwise optimizer-
gradient equivalence claim across allocations. FiniteK MSE retains variance/K
bias. Exact profile replay(atol1e-5,rtol1e-4), finite/nonzero gradients, absent
head gradients/source identity and<85%actualHBM guards remain unchanged.

Strong efficiency is(T8/T32)/4 at the sameK32/global128 work. Per-case node
seconds are N*T/128, using the complete all-rank maximum update time, not a
sum of stage maxima. A quote requires every rank's committed metadata plus
parent AND exact application.0 COMPLETED0:0. Partial/failed output is unqualified.
This is feasibility and one-update forecasting; production efficiency and
scientific efficacy have not been established by preparing this packet.
