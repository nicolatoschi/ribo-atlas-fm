ANOMALY-OFF RE-MEASUREMENT PACKETS (derived from native 670450638bf1)

Diagnosis: RNAbpFlow src/models/flow_module.py line 25 calls
torch.autograd.set_detect_anomaly(True) at import. The batched driver never
disabled it, so every 175-nt timing (2305296/97/98, 2319595) recorded Python
stack traces per autograd node and NaN-checked every backward output. The
157-nt full_trajectory_driver (2298974) disables it after model construction.

Change: one statement after model construction,
  torch.autograd.set_detect_anomaly(False);assert not torch.is_anomaly_enabled()
plus a report field autograd_anomaly_enabled and corrected launcher metadata
strings. Workload and update are AST-identical to 670450638bf1; inputs,
fixture, checkpoint, head, losses, guards and optimizer are unchanged.
Anomaly mode changes speed only: local CPU A/B gave identical gradient norms.

Runs: 0 = 1-node A/B gate (this packet vs untouched 670450638bf1, K8/T50,
4 ranks x 1 case); 1 = 32-node reliability (2 updates + fresh-process
resume); 2 = 8-node K8 and K32 at local 4, two repeats each.
Scope unchanged: one 175-nt RNA, synthetic observations, random frozen head.
Cost and scaling only; no biological or convergence claim.
