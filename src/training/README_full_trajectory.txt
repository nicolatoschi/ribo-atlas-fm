Differentiable final-generated ensemble training qualification

Scientific purpose
The bulk-reactivity objective applies the frozen forward model to the final
generated structures, then compares its K-sample mean with the observed profile.
One-step denoised consistency is a useful separate regularizer; it does not
replace this final-generated objective. The proposed production forward model
is fitted on training-only cross-fitting folds and frozen before joint inverse
conditioning. This packet instead uses a RANDOM FIXED geometry/chemistry head
and synthetic observations solely to establish the differentiable execution,
gradient, memory and compute path. It makes no cross-fit or scientific benefit
claim. FiniteK mean-profile MSE has a MonteCarlo variance/K component; K_train
is declared, and unbiased population-mean consistency is not claimed.

Native sampler and differentiable port
The pinned public inference interpolant_inf.py executes every denoiser under
torch.no_grad and returns detached CPU NumPy atoms. That inference function
cannot train the proposed bulk objective unchanged. The public training source
supplies differentiable Torch Euler and SO3 maps but has no full sampler.
full_trajectory_core.py implements the same float32 linspace(.01,1,T), T-1
translation/SO3 Euler intervals and final t=1 denoiser call: T50 means49intervals
plus one final prediction, hence50vector-field evaluations per trajectory.
Initial translation/rotation priors are parameter-independent centered Gaussian
and uniformSO3 draws. All later states remain Torch tensors in the graph,
including native atom23-to-C4' conversion (native ordering index5). There is no
detach or NumPy conversion between parameter-dependent states and the loss.
Self-conditioning is disabled in the pinned configuration, matching this port.
Production sampling eval mode disables dropout. MHA fused no-grad fastpath is
disabled so first-pass and gradient-pass math follow the same Torch route.

Bounded memory and true measured recomputation
Non-reentrant checkpointing wraps each full model/Euler/SO3 interval and final
prediction, preserving RNG state. Step times are bound in the function defaults
to prevent late-binding recomputation errors. No-grad ensemble mean then
common-RNG graph replay obtains the exact finiteK mean-loss gradient with one
checkpointed trajectory resident. Replay profiles must agree numerically at
atol1e-5/rtol1e-4. Checkpoint recomputation may stop after all required tensors
are recovered; its ACTUAL entry counts and actual backward/recompute elapsed
time are recorded. Model evaluation counts alone are not used as a cost factor.
GPU qualification must establish finite full-model gradients and memory.
PyTorch2.10 API: https://docs.pytorch.org/docs/2.10/checkpoint.html

Objective and frozen-head boundary
L=teacher_weight*original published native flow-matching loss
 +constraint_weight*mean_K(native adjacent-C4' distance constraint)
 +bulk_weight*weightedMSE(mean_K(frozenhead(finalgeneratedC4',chemistry)),target).
The adjacent-distance term uses native local geometry and does not align every
generated sample to one global target conformation. The mean-profile term
reaches the final generated distribution through every solver interval.
The same observed profile/error/mask/context is supplied to the inverse probe
encoder for every K member. Forward-head inputs are ONLY finalC4' coordinates,
chemistry ID and native residue validity; measured values/errors/probe masks
are absent. The4,673forward-head parameters freeze and their SHA remains fixed.
Original16,876,313parameter vector field initializes from the verified official
RNA3DB.ckpt and loads strictly before attaching the330,752parameter matched
inactive/active probe module. Total17,211,738parameters;17,207,065trainable.

Distributed derivative
All generated replay/backward passes use DDP.no_sync. The final original
flow-matching teacher backward touches every parameter hook, including the
otherwise-unused generated pair projection, and reduces accumulated gradients
once. Even a zero-weight teacher supplies hooks for a bulk-only qualification.
Generated-only gradient norms are checked before the teacher contribution.
Active probe gradients must be finite/nonzero; inactive ones are exactlyzero.
Combined gradients clip at norm1 before AdamW; preclip norms are retained.

Actual CPU proof and planned full-model qualification
cpu_trajectory_test.json is an ACTUALLY executed CPU K2/T2 test of native public
SO3/Euler maps with a tiny vector field. It compares a retained uncheckpointed
graph with streamed checkpointed gradients, checks rotation/translation/active
probe gradients and unchanged frozen head, and records actual call entries.
This proves implementation algebra/primitives; the full RNAbpFlow GPU graph
requires its own qualification and is not claimed to have run yet.
The new GPU script first performs a full-model K2/T2 BULK-ONLY gradient check.
It then profiles native1l8v_B157nt with K8/K32,T50, inactive/active conditioning,
one initial optimizer update per configuration. No full-T50 warmup is charged
before these initial qualification timings; initialization cost is included and
recorded as such. A30min1node/4GPU ceiling is2GPUh, not consumed time. Further
steady-state profiling uses scientifically admitted length/condition exposure,
data loading and the production model after initial graph qualification. No
updates, parameters or scales are selected to consume the20knodehcall minimum
or to fill the original85,500nodehrequest.

CORRECTIVE NATIVE-PRIMAL QUATERNION DERIVATIVE
The original submitted361fa418 packet/job2298365 failed its full-model T2/K2
bulk-only preflight with nonfinite gradients. Actual75nt CPU anomaly tracing
identified LinalgEighBackward0 in rigid_utils.rot_to_quat. Native SO3 quaternion
K has an isolated largest eigenvalue and three repeated unused lower values.
The correction retains the exact native K/eigh largest-vector forward output
and sign. Its analytic derivative divides only largest-to-lower spectral gaps.
It never perturbs eigenvalues or replaces/clips nonfinite gradients. Strict
isolated-gap and finite-derivative checks remain. Original native source stays
unchanged; the driver installs this one explicit converter derivative at runtime.
Actual CPU tests:1020 rotations each inFP32/FP64, identity/nearidentity/pi,
exact native quaternion and reconstructed-matrix differences0, finite gradients
and phase-invariant finite-difference gradcheck. Actual real1g59_B75nt K2/T2
CPU full-model preflight plus inactive/active optimizer updates pass with anomaly
tracing, positive inverse/active-probe gradients, inactive probe gradient0,
frozen-head gradients absent, and exact profile replay. This evidence covers
CPU shortRNA/T2 only. GPU T2 qualification precedes the target T50 measurement.
Original failed source/job and logs remain preserved under their own identity.
Reference:https://docs.pytorch.org/docs/2.10/generated/torch.linalg.eigh.html
