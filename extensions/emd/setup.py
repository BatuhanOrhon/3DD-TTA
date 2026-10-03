"""Setup extension

Notes:
    If extra_compile_args is provided, you need to provide different instances for different extensions.
    Refer to https://github.com/pytorch/pytorch/issues/20169

"""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6"
# The extension-local kernel omits MatchCostForward/Backward definitions.
emd_cuda_dir = (
    Path(__file__).resolve().parents[2] / "third_party" / "PyTorchEMD" / "cuda"
)
setup(
    name='emd_ext',
    ext_modules=[
        CUDAExtension(
            name='emd_cuda',
            sources=[
                str(emd_cuda_dir / 'emd.cpp'),
                str(emd_cuda_dir / 'emd_kernel.cu'),
            ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
