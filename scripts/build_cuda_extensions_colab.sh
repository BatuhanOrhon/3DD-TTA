#!/usr/bin/env bash
set -euo pipefail

TTA_PHASE="${1:-all}"
case "$TTA_PHASE" in
  prepare|emd|chamfer|pointnet2|pvcnn-smoke|all) ;;
  *)
    echo "Unknown build phase: $TTA_PHASE" >&2
    exit 2
    ;;
esac

TTA_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TTA_STATE_DIR="$TTA_REPO_DIR/.colab_setup/native"
mkdir -p "$TTA_STATE_DIR"
TTA_ENV_PREFIX="$(conda run --no-capture-output -n 3dd_tta_env python -c 'import sys; print(sys.prefix)')"
TTA_TORCH_LIB="$(conda run --no-capture-output -n 3dd_tta_env python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
TTA_HOST_CC="$(conda run --no-capture-output -n 3dd_tta_env python -c 'import os; print(os.environ["CC"])')"
TTA_HOST_CXX="$(conda run --no-capture-output -n 3dd_tta_env python -c 'import os; print(os.environ["CXX"])')"
TTA_GPU_ARCH="$(conda run --no-capture-output -n 3dd_tta_env python -c 'import torch; assert torch.version.cuda == "12.1", torch.version.cuda; assert torch.cuda.is_available(), "Select a Colab GPU runtime"; print(".".join(map(str, torch.cuda.get_device_capability())))')"
TTA_CUDA_INCLUDE="$TTA_ENV_PREFIX/targets/x86_64-linux/include"
TTA_CUDA_LIB="$TTA_ENV_PREFIX/targets/x86_64-linux/lib"

run_with_cuda_env() {
  conda run --no-capture-output -n 3dd_tta_env env \
    CC="$TTA_HOST_CC" CXX="$TTA_HOST_CXX" \
    CUDA_HOME="$TTA_ENV_PREFIX" \
    CPATH="$TTA_CUDA_INCLUDE${CPATH:+:$CPATH}" \
    LIBRARY_PATH="$TTA_CUDA_LIB${LIBRARY_PATH:+:$LIBRARY_PATH}" \
    LD_LIBRARY_PATH="$TTA_TORCH_LIB:$TTA_CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="$TTA_GPU_ARCH" MAX_JOBS="${MAX_JOBS:-1}" "$@"
}

heartbeat() {
  printf '%s phase=%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2"
}

run_with_heartbeat() {
  local phase="$1"
  shift
  heartbeat "$phase" "started"
  "$@" &
  local child=$!
  while kill -0 "$child" 2>/dev/null; do
    local tick
    for tick in {1..12}; do
      sleep 10
      kill -0 "$child" 2>/dev/null || break
    done
    kill -0 "$child" 2>/dev/null && heartbeat "$phase" "still running"
  done
  if wait "$child"; then
    heartbeat "$phase" "completed"
  else
    local status=$?
    heartbeat "$phase" "failed status=$status"
    return "$status"
  fi
}

require_phase() {
  test -f "$TTA_STATE_DIR/$1.complete" || {
    echo "Required build phase is incomplete: $1" >&2
    exit 1
  }
}

mark_phase() {
  touch "$TTA_STATE_DIR/$1.complete"
}

prepare_sources() {
  if [[ -f "$TTA_STATE_DIR/prepare.complete" ]]; then
    heartbeat prepare "already complete; skipping"
    return
  fi
  heartbeat prepare "validating CUDA 12.1 toolchain"
  # NVIDIA toolkit releases use either targets/ or the environment root layout.
  if [[ ! -f "$TTA_CUDA_INCLUDE/cuda_runtime_api.h" ]]; then
    TTA_CUDA_INCLUDE="$TTA_ENV_PREFIX/include"
  fi
  if [[ ! -e "$TTA_CUDA_LIB/libcudart.so" ]]; then
    TTA_CUDA_LIB="$TTA_ENV_PREFIX/lib"
  fi
  test -x "$TTA_ENV_PREFIX/bin/nvcc"
  test -f "$TTA_CUDA_INCLUDE/cuda_runtime_api.h"
  test -e "$TTA_CUDA_LIB/libcudart.so"
  conda run --no-capture-output -n 3dd_tta_env python - "$TTA_ENV_PREFIX/bin/nvcc" <<'PY'
import subprocess, sys
version = subprocess.check_output([sys.argv[1], '--version'], text=True)
print(version)
assert 'release 12.1,' in version, 'CUDA compiler must match PyTorch cu121'
PY
  echo "GPU architecture: $TTA_GPU_ARCH"
  # Future conda-run/JIT calls must use the same toolkit as this build.
  conda env config vars set -n 3dd_tta_env \
    CUDA_HOME="$TTA_ENV_PREFIX" \
    CPATH="$TTA_CUDA_INCLUDE${CPATH:+:$CPATH}" \
    LIBRARY_PATH="$TTA_CUDA_LIB${LIBRARY_PATH:+:$LIBRARY_PATH}" \
    LD_LIBRARY_PATH="$TTA_TORCH_LIB:$TTA_CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="$TTA_GPU_ARCH"

  # The notebook checks out dev, whose setup.py overwrites the selected GPU target.
  conda run --no-capture-output -n 3dd_tta_env python - "$TTA_REPO_DIR" <<'PY'
import hashlib, sys
from pathlib import Path
path = Path(sys.argv[1]) / 'Pointnet2_PyTorch/pointnet2_ops_lib/setup.py'
source = path.read_text()
old = 'os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6"'
updated = source.replace(old, 'os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")')
if updated != source:
    print('PointNet++ setup SHA256:', hashlib.sha256(source.encode()).hexdigest(),
          '->', hashlib.sha256(updated.encode()).hexdigest())
    path.write_text(updated)

# The extension copy lacks cost kernels; reuse the project's complete EMD source.
kernel = Path(sys.argv[1]) / 'extensions/emd/cuda/emd_kernel.cu'
complete = Path(sys.argv[1]) / 'third_party/PyTorchEMD/cuda/emd_kernel.cu'
required = ('at::Tensor MatchCostForward(', 'std::vector<at::Tensor> MatchCostBackward(')
source = kernel.read_text()
if not all(symbol in source for symbol in required):
    replacement = complete.read_text()
    assert all(symbol in replacement for symbol in required), 'Complete EMD kernels missing'
    print('EMD kernel SHA256:', hashlib.sha256(source.encode()).hexdigest(),
          '->', hashlib.sha256(replacement.encode()).hexdigest())
    kernel.write_text(replacement)
PY
  mark_phase prepare
  heartbeat prepare "completed"
}

build_extension() {
  local phase="$1"
  local relative_dir="$2"
  require_phase prepare
  if [[ -f "$TTA_STATE_DIR/$phase.complete" ]]; then
    heartbeat "$phase" "already complete; skipping"
    return
  fi
  cd "$TTA_REPO_DIR/$relative_dir"
  # Objects built with the system toolkit must not enter the cu121 rebuild.
  rm -rf -- "$TTA_REPO_DIR/$relative_dir/build"
  run_with_heartbeat "$phase" run_with_cuda_env python setup.py install --force
  mark_phase "$phase"
}

build_pvcnn_and_smoke() {
  require_phase emd
  require_phase chamfer
  require_phase pointnet2
  if [[ -f "$TTA_STATE_DIR/pvcnn-smoke.complete" ]]; then
    heartbeat pvcnn-smoke "already complete; skipping"
    return
  fi
  cd "$TTA_REPO_DIR"
  run_with_heartbeat pvcnn run_with_cuda_env python build_pkg.py
  heartbeat smoke "native GPU checks started"
  run_with_cuda_env python - <<'PY'
import torch, chamfer, emd_cuda, pointnet2_ops._ext
from pointnet2_ops import pointnet2_utils
points = (torch.rand(1, 32, 3, device='cuda') + 1).contiguous()
indices = pointnet2_utils.furthest_point_sample(points, 8)
assert tuple(indices.shape) == (1, 8)
assert bool(((indices >= 0) & (indices < 32)).all())
features = points.transpose(1, 2).contiguous().requires_grad_(True)
sampled = pointnet2_utils.gather_operation(features, indices)
assert tuple(sampled.shape) == (1, 3, 8)
assert torch.allclose(sampled.transpose(1, 2), points[:, indices[0].long(), :])
sampled.sum().backward()
assert bool(torch.isfinite(features.grad).all())
torch.cuda.synchronize()
print('CUDA extension checks passed:', torch.__version__, torch.version.cuda,
      torch.cuda.get_device_name(0), 'FPS + gather forward/backward')
PY
  mark_phase pvcnn-smoke
  heartbeat pvcnn-smoke "completed"
}

case "$TTA_PHASE" in
  prepare) prepare_sources ;;
  emd) build_extension emd extensions/emd ;;
  chamfer) build_extension chamfer extensions/chamfer_dist ;;
  pointnet2) build_extension pointnet2 Pointnet2_PyTorch/pointnet2_ops_lib ;;
  pvcnn-smoke) build_pvcnn_and_smoke ;;
  all)
    prepare_sources
    build_extension emd extensions/emd
    build_extension chamfer extensions/chamfer_dist
    build_extension pointnet2 Pointnet2_PyTorch/pointnet2_ops_lib
    build_pvcnn_and_smoke
    ;;
esac
