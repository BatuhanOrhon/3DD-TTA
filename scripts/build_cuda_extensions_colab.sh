#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="3dd_tta_env"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_PREFIX="$(conda run --no-capture-output -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
TORCH_LIB_DIR="$(conda run --no-capture-output -n "$ENV_NAME" python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
HOST_CC="$(conda run --no-capture-output -n "$ENV_NAME" python -c 'import os; print(os.environ["CC"])')"
HOST_CXX="$(conda run --no-capture-output -n "$ENV_NAME" python -c 'import os; print(os.environ["CXX"])')"
TORCH_CUDA_ARCH_LIST="$(conda run --no-capture-output -n "$ENV_NAME" python -c 'import torch; assert torch.cuda.is_available(), "Select a Colab GPU runtime"; assert torch.version.cuda == "12.1", torch.version.cuda; print(".".join(map(str, torch.cuda.get_device_capability())))')"
CUDA_INCLUDE_DIR="$CONDA_PREFIX/targets/x86_64-linux/include"
CUDA_LIBRARY_DIR="$CONDA_PREFIX/targets/x86_64-linux/lib"

if [[ ! -x "$CONDA_PREFIX/bin/nvcc" ]]; then
  echo "Pinned CUDA 12.8 nvcc is missing from $CONDA_PREFIX/bin" >&2
  exit 1
fi

if [[ ! -f "$CUDA_INCLUDE_DIR/cuda_runtime_api.h" ]]; then
  echo "CUDA runtime headers are missing from $CUDA_INCLUDE_DIR" >&2
  exit 1
fi

echo "CUDA compiler:"
"$CONDA_PREFIX/bin/nvcc" --version
echo "Host C compiler:"
conda run --no-capture-output -n "$ENV_NAME" "$HOST_CC" --version
echo "Host C++ compiler:"
conda run --no-capture-output -n "$ENV_NAME" "$HOST_CXX" --version
echo "GPU compute capability: $TORCH_CUDA_ARCH_LIST"

build_extension() {
  local relative_dir="$1"
  cd "$REPO_DIR/$relative_dir"
  run_with_cuda_env python setup.py install --force
}

run_with_cuda_env() {
  conda run --no-capture-output -n "$ENV_NAME" env \
    CC="$HOST_CC" \
    CXX="$HOST_CXX" \
    CUDA_HOME="$CONDA_PREFIX" \
    CPATH="$CUDA_INCLUDE_DIR${CPATH:+:$CPATH}" \
    LIBRARY_PATH="$CUDA_LIBRARY_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}" \
    LD_LIBRARY_PATH="$TORCH_LIB_DIR:$CUDA_LIBRARY_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    "$@"
}

build_extension extensions/emd
build_extension extensions/chamfer_dist
build_extension Pointnet2_PyTorch/pointnet2_ops_lib

cd "$REPO_DIR"
run_with_cuda_env python build_pkg.py
run_with_cuda_env python -c \
  'import torch; import chamfer, emd_cuda, pointnet2_ops._ext; print("EMD, Chamfer and PointNet++ extensions imported; CUDA", torch.version.cuda)'
