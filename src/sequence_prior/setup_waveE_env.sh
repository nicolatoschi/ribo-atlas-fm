#!/usr/bin/env bash
# Wave E one-time setup on a JUPITER login node (internet access needed). Run from inside the
# uploaded packet directory:   bash setup_waveE_env.sh
# Prepared by Claude on 15 Oct 2026 and NOT executed by it. Idempotent: it re-verifies
# an existing venv/weights file and never overwrites a verified weights file.
#
#  1. Creates a separate venv on top of the geometry-pilot env (torch 2.10, Python 3.12, aarch64).
#     A venv made from a venv's python inherits only the *base* interpreter's site-packages,
#     so a .pth file additionally exposes the geometry env's own site-packages (same order
#     as inside that env). torch/numpy/pandas/tqdm come from there; nothing is upgraded.
#  2. python -m pip install --no-deps --require-hashes rna-fm 0.2.2 + ptflops 0.7.5 (wheels).
#  3. Downloads RNA-FM_pretrained.pth (pinned HuggingFace revision named by the RNA-FM README)
#     to waveE_assets/, verifies size + sha256, makes it read-only.
#  4. Builds the model on CPU through the packet's own loader and checks parameter count and
#     the initial-state digest measured on the Mac (catches a wrong file or fm version).
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1   # never leave __pycache__ inside the frozen packet
BASE_ENV=/e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/env
ROOT=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015
ENV_DIR=$ROOT/waveE_env
ASSETS=$ROOT/waveE_assets
WEIGHTS=$ASSETS/RNA-FM_pretrained.pth
WEIGHTS_URL=https://huggingface.co/cuhkaih/rnafm/resolve/91d4a46d28d8054a7b429955e8fc0c253ba0afd6/RNA-FM_pretrained.pth
WEIGHTS_SHA256=5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99
WEIGHTS_BYTES=1194424423
EXPECTED_PARAMS=99521546
EXPECTED_DIGEST=24c733cc0ae85c876ee732e52d274ae0266e7271315a56c27798d462ee29e172
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
die() { echo "SETUP FAILED: $*" >&2; exit 1; }

[ -x "$BASE_ENV/bin/python" ] || die "base env python missing: $BASE_ENV/bin/python"
[ -f "$HERE/requirements-waveE.txt" ] && [ -f "$HERE/waveE_mlm.py" ] || die "run from inside the Wave E packet"
"$BASE_ENV/bin/python" -c 'import torch,sys; assert torch.__version__.startswith("2.10"), torch.__version__; print("base env:", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda)'

# 1. venv ---------------------------------------------------------------------------
if [ ! -f "$ENV_DIR/pyvenv.cfg" ]; then
  "$BASE_ENV/bin/python" -m venv --system-site-packages "$ENV_DIR"
fi
SP_NEW=$("$ENV_DIR/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
SP_BASE=$("$BASE_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf '%s\n' "$SP_BASE" > "$SP_NEW/_ribo_geometry_env.pth"

# 2. packages -------------------------------------------------------------------------
"$ENV_DIR/bin/python" -m pip install --no-deps --require-hashes --only-binary=:all: -r "$HERE/requirements-waveE.txt"
mkdir -p "$ASSETS"
"$ENV_DIR/bin/python" -m pip freeze > "$ASSETS/waveE_env_pip_freeze.txt"
"$ENV_DIR/bin/python" - <<'PY'
import importlib.metadata as md, sys, torch, numpy, fm
assert torch.__version__.startswith('2.10'), f'torch changed: {torch.__version__}'
assert fm.__version__ == '0.2.2', fm.__version__
print('waveE env:', sys.executable, 'torch', torch.__version__, 'numpy', numpy.__version__, 'fm', fm.__version__, fm.__file__,
      'ptflops', md.version('ptflops'))
PY

# 3. weights --------------------------------------------------------------------------
verify_weights() {
  [ "$(stat -c %s "$1")" = "$WEIGHTS_BYTES" ] || return 1
  [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$WEIGHTS_SHA256" ] || return 1
}
if [ -f "$WEIGHTS" ]; then
  verify_weights "$WEIGHTS" || die "existing $WEIGHTS fails size/sha256 (not overwritten; inspect manually)"
  echo "weights present and verified"
else
  curl -fL --retry 3 --retry-delay 10 -o "$WEIGHTS.part" -D "$ASSETS/RNA-FM_pretrained.download_headers.txt" "$WEIGHTS_URL"
  verify_weights "$WEIGHTS.part" || { rm -f "$WEIGHTS.part"; die "downloaded weights fail size/sha256"; }
  mv "$WEIGHTS.part" "$WEIGHTS"
fi
chmod 444 "$WEIGHTS"
printf '%s  %s\n' "$WEIGHTS_SHA256" "RNA-FM_pretrained.pth" > "$ASSETS/RNA-FM_pretrained.pth.sha256"
printf '%s\n' "source $WEIGHTS_URL" "sha256 $WEIGHTS_SHA256" "bytes $WEIGHTS_BYTES" "verified_utc $(date -u +%FT%TZ)" > "$ASSETS/RNA-FM_pretrained.provenance.txt"

# 4. CPU load test through the packet loader -----------------------------------------------
"$ENV_DIR/bin/python" -B - "$HERE/waveE_mlm.py" "$WEIGHTS" "$EXPECTED_PARAMS" "$EXPECTED_DIGEST" <<'PY'
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('waveE_mlm', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
model, info = m.load_rnafm(sys.argv[2], 'official')
print(json.dumps({k: info[k] for k in ['parameters_total', 'parameters_trainable', 'loaded_embed_equals_checkpoint_lm_head', 'layer_norm_class', 'initial_state_digest_sha256', 'load_seconds']}, indent=1))
assert info['parameters_total'] == int(sys.argv[3]), 'parameter count'
assert info['loaded_embed_equals_checkpoint_lm_head'], 'tied embedding check'
assert info['initial_state_digest_sha256'] == sys.argv[4], 'initial-state digest differs from the Mac measurement'
print('LOAD TEST OK')
PY
echo "Wave E setup complete: env $ENV_DIR, weights $WEIGHTS"
