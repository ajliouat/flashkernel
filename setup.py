"""
FlashKernel — setup.py for torch C++ extension build.

Uses torch.utils.cpp_extension to compile CUDA kernels with nvcc
and bind them via pybind11 (bundled with PyTorch).

Usage:
    pip install -e ".[dev]"           # Development install
    python setup.py build_ext --inplace   # Build extension in-place
"""

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.abspath(__file__))
CUDA_DIR = os.path.join(ROOT, "src", "cuda")
BINDINGS_DIR = os.path.join(ROOT, "src", "bindings")

# ─── Source files ────────────────────────────────────────────────────────────

cuda_sources = [
    os.path.join(CUDA_DIR, f)
    for f in os.listdir(CUDA_DIR)
    if f.endswith(".cu")
]

binding_sources = [
    os.path.join(BINDINGS_DIR, f)
    for f in os.listdir(BINDINGS_DIR)
    if f.endswith(".cpp")
]

all_sources = binding_sources + cuda_sources

# ─── Compiler flags ─────────────────────────────────────────────────────────

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-arch=sm_75",                    # T4 (Turing)
    "--expt-relaxed-constexpr",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]

cxx_flags = [
    "-O3",
    "-std=c++17",
]

# ─── Extension ───────────────────────────────────────────────────────────────

ext_modules = [
    CUDAExtension(
        name="flashkernel._flashkernel_C",
        sources=all_sources,
        include_dirs=[CUDA_DIR],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    ),
]

# ─── Setup ───────────────────────────────────────────────────────────────────

setup(
    name="flashkernel",
    version="1.0.10",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
