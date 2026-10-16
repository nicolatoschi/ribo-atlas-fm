#!/usr/bin/env bash
set -euo pipefail
GEOMETRY_BASE_PYTHON=${GEOMETRY_BASE_PYTHON:-/e/project1/e-aif-2026fl01-991/aeris_ai_v019_20261003/envs/openfe-1.12.0-cu129/bin/python3.12}
GEOMETRY_REMOTE_ROOT=${GEOMETRY_REMOTE_ROOT:-/e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012}
GEOMETRY_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$GEOMETRY_REMOTE_ROOT"
if [[ ! -f "$GEOMETRY_REMOTE_ROOT/env/pyvenv.cfg" ]]; then
  "$GEOMETRY_BASE_PYTHON" -m venv --system-site-packages "$GEOMETRY_REMOTE_ROOT/env"
fi
"$GEOMETRY_REMOTE_ROOT/env/bin/python" -m pip install --only-binary=dm-tree,gemmi -r "$GEOMETRY_SCRIPT_DIR/requirements-added.in"
"$GEOMETRY_REMOTE_ROOT/env/bin/python" -m pip freeze > "$GEOMETRY_REMOTE_ROOT/environment_freeze.txt"
"$GEOMETRY_REMOTE_ROOT/env/bin/python" - <<'PY'
import importlib.metadata as md,json,sys
required=['torch','numpy','scipy','pandas','pytorch-lightning','biopython','hydra-core','omegaconf','beartype','jaxtyping','dm-tree','gemmi']
print(json.dumps({'interpreter':sys.executable,'packages':{p:md.version(p) for p in required}},indent=2))
assert md.version('torch').startswith('2.10.0'), 'Do not replace the working CUDA Torch'
PY
