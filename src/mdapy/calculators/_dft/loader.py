# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The module _dft of the GPU part, the DFT on the device: that of the GPU part
that calculators given device="cuda" use, mdapy_cuda12._dft or
mdapy_rocm6._dft of the packages mdapy-cuda12 and mdapy-rocm6, or mdapy._dft of
an MDAPY built with it."""

import importlib
from typing import Any

from mdapy.calculators import _gpu
from mdapy.errors import InvalidParameter

_NO_GPU_PART = (
    "the DFT runs in the GPU part of MDAPY, which is not installed: on Linux "
    'with an NVIDIA GPU, pip install "mdapy[cuda12]"; for an AMD GPU, build '
    "the package mdapy-rocm6 with ROCm, pip install ./rocm6 in a clone of MDAPY"
)
# why the module did not load, by the API of the GPU part: it names no library
# of the vendor (it loads cuBLAS, cuFFT and cuSOLVER, or hipBLAS, hipFFT and
# hipSOLVER, where they are asked for, vendor_libraries); with HIP it needs the
# HIP runtime of ROCm, as the potentials of the GPU part do
_LIBRARIES = {
    "CUDA": "the module _dft of the GPU part could not be loaded",
    "HIP": (
        "the module _dft of the GPU part could not be loaded: it needs the HIP "
        "runtime of ROCm 6 on the machine, as module load rocm gives it on a cluster"
    ),
}


def module() -> Any:
    """The module _dft of the GPU part, loaded.

    Returns
    -------
    module
        mdapy_cuda12._dft, mdapy_rocm6._dft or mdapy._dft.

    Raises
    ------
    InvalidParameter
        If there is no GPU part, if it was built without the DFT, or if the
        libraries of the DFT are missing.
    """
    part = _gpu.module()
    if part is None:
        raise InvalidParameter(_NO_GPU_PART)
    package, _, name = part.__name__.rpartition(".")
    api = "HIP" if name == "_hip" else "CUDA"
    try:
        return importlib.import_module(f"{package}._dft")
    except ModuleNotFoundError as error:
        if error.name != f"{package}._dft":
            raise
        where = "MDAPY" if package == "mdapy" else package.replace("_", "-")
        raise InvalidParameter(
            f"the GPU part of {where} was built without the DFT: build it again "
            "with -Ccmake.define.MDAPY_DFT=ON"
        ) from None
    except ImportError as error:
        raise InvalidParameter(f"{_LIBRARIES[api]} ({error})") from None


def vendor_libraries(dft: Any) -> list:
    """The libraries of the vendor loaded by the module _dft, for
    ``kernels="vendor"``: NVIDIA's cuBLAS, cuFFT and cuSOLVER with CUDA,
    hipBLAS, hipFFT and hipSOLVER of ROCm with HIP, loaded here if they were
    not.

    Parameters
    ----------
    dft : module
        The module _dft, as module gives it.

    Returns
    -------
    list of str
        The files of the libraries loaded, and of those they need that the
        module loaded the same way.

    Raises
    ------
    InvalidParameter
        If the libraries cannot be loaded, saying what to install.
    """
    try:
        return list(dft.vendor_libraries())
    except RuntimeError as error:
        raise InvalidParameter(str(error)) from None
