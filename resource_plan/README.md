# Resource plan and facts file

| File | Role |
|---|---|
| `design_v2.json` | The PI's design: arms, seeds, exposure (cases and passes per stage), batch, microbatch, ablations, reproduction, data-scaling fractions, inference cohorts, development panel, evaluation packing (80%), fixed fitting and operations blocks |
| `build_resource_plan_v2.py` | Measured unit costs × design → `resource_plan_v2.json`. Unit costs are read from the Wave B, C and D analyses; nothing is back-solved |
| `resource_plan_v2.json` | The ledger. The model total is 83,239.0 node-hours |
| `build_facts_v2.py` | Ledger + analyses + data-admission evidence + rank reports → `facts_v2.json`. Each block is rounded up to the next 100 NH, and the monthly schedule is also computed here. Shipped byte-identical; it hard-codes the build machine's root |
| `facts_v2.json` | The single source of numbers for the proposal documents. Request 83,400 NH = 333,600 GPU-h |
| `rebuild_facts.py` | Runs `build_facts_v2.py` unchanged with remapped paths (inputs, rank-report root, output), and compares the result with a shipped facts file |

Commands are in `docs/REPRODUCE.md`. The numbers that `build_facts_v2.py` types instead of reading are traced in `docs/NUMBER_SOURCES.md`.
