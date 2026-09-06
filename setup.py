import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# The GTX 1050 Ti is Pascal compute capability 6.1. Respect an explicit user
# setting so the project can also be compiled for a different GPU when needed.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.1")


setup(
    name="cuda-attention-optimization",
    version="0.1.0",
    description="Educational scaled dot-product attention optimization project",
    packages=find_packages(),
    python_requires=">=3.9",
    ext_modules=[
        CUDAExtension(
            name="cuda_attention",
            sources=[
                "csrc/attention.cpp",
                "csrc/attention_naive.cu",
                "csrc/attention_optimized.cu",
            ],
            extra_compile_args={
                "cxx": ["/O2", "/std:c++17"],
                # Deliberately omit --use_fast_math to retain conservative FP32
                # numerical behavior for the reference comparison.
                "nvcc": ["-O3", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
