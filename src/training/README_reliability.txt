FULL GENERATED-ENSEMBLE PRODUCTION RELIABILITY PILOT (PREPARED, NOT EXECUTED)

Why: current native175 results qualify one initial complete optimizer update
at each allocation. They establish full-objective fixed-work scaling but do
not establish sustained updates or checkpoint recovery of this objective.

Planned: one 32-node, 128-rank job, B128, micro1, K32 and T50. A first srun
process performs two genuinely consecutive AdamW updates. Each rank commits
its full model, optimizer moments, all Python/NumPy/Torch CPU/device RNG,
constant learning rate and fixed logical-instance cursor after updates 1/2.
A second fresh srun process loads update1 exactly and recomputes update2.
Fresh-process restoration must match model/optimizer/RNG hashes exactly.
Final model and optimizer tensors must match uninterrupted reference within
prospective atol1e-6/rtol1e-5; RNG/constant-rate/cursor must match exactly;
bulk, adjacency and teacher losses must match atol1e-5/rtol1e-4. Final
bitwise identity is separately reported and is not assumed on CUDA.

The numerical Workload class is AST-identical to the successful frozen
670450638bf1 native driver. The numerical update body is AST-identical
except removal of model-initialization/optimizer-reset statements, allowing
optimizer moments to persist. Source changes outside that body introduce
the stage orchestration, input/state hashes and the site-enforced collective
limit. All source, fixture, official checkpoint and input identities remain
strictly checked. Existing profile replay, finite/nonzero inverse and probe
gradients, absent frozen-head gradients, exact native eigensolver derivative
and reserved-HBM<85% guards remain. Post-update model/optimizer finite checks
are additionally required. No native projection, loss or derivative is
replaced and no original immutable packet is edited.

12-hour site maximum: batch time and collective timeout are43200seconds;
there is no shorter assistant-imposed elapsed cap. Expected3*5362.857s
complete update work is4.47hours (~572.04GPUh), plus setup and checkpoint
I/O. The job reserves at most384node-hours=1536GPU-hours. Per-rank update1
and update2 model/optimizer snapshots need approximately56GB temporarily.
Checkpointing and hashes are outside measured optimizer-update timing and
are reported separately. Do not publish binary checkpoint payloads by
default: hashes, reports and the complete reproducible source are sufficient
for the reviewer archive and avoid a large unnecessary download.

Qualification requires parent and both exact application steps COMPLETED0:0,
all128 continuous plus128 resumed reports (384 accepted update reports),
matching frozen source/input identity and all prospective numerical guards.
Any partial report or failure is unqualified. Preparation and CPU checkpoint
orchestration tests are not GPU execution or a passed full-objective restart.

Scope: fixed128 synthetic stochastic instances of one actual175nt RNA,
5zwm_B; a random frozen geometry/chemistry head; identical fixtures on
consecutive updates. This tests engineering reliability and timing, not
learning efficacy, biological independence or convergence. Recovery uses
fresh processes on the same allocated nodes, not a different-node preemption
or shuffled-data experiment. Native sampler and teacher explicitly reseed
their stochastic work; this is not refreshed-Monte-Carlo continuity.
The original driver has a constant AdamW
learning rate and no learning-rate scheduler; that exact policy is restored.

Before root submits: compare semantic condition against jobs in both
authorized projects; verify existing qualified scaling jobs are not rerun;
hash every uploaded file and the public checkpoint; verify effective site
QoS limit and free storage; retain this immutable packet and job/account
records. No submission occurs in the preparation script.
