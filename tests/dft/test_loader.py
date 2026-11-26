# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Where the DFT finds its module _dft: in the GPU part that calculators on the
GPU use, of the package mdapy-cuda12 or mdapy-rocm6 or built into MDAPY; and
the errors where it cannot, with stand-ins for the packages and for modules
whose libraries are missing, and where the libraries of the vendor cannot be
loaded for kernels="vendor"."""

import sys
import types

import pytest

import mdapy
from mdapy.calculators import DFT
from mdapy.calculators._dft import loader
from mdapy.errors import InvalidParameter

LIBSTDCXX = "version `GLIBCXX_3.4.32' not found (required by _dft.abi3.so)"
HIPBLAS = "libhipblas.so.2: cannot open shared object file: No such file or directory"


class Unloadable:
    """A finder of modules under which one module fails to load, as an
    extension module whose shared libraries are missing does."""

    def __init__(self, name: str, message: str):
        self.name, self.message = name, message

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name:
            raise ImportError(self.message)
        return None


def not_built_in(monkeypatch):
    """An MDAPY built without a GPU part, and neither package installed."""
    for part in ("_cuda", "_hip", "_dft"):
        monkeypatch.setitem(sys.modules, f"mdapy.{part}", None)
        monkeypatch.delattr(mdapy, part, raising=False)
    for name in ("mdapy_cuda12", "mdapy_rocm6"):
        monkeypatch.setitem(sys.modules, name, None)


def install(monkeypatch, name, part, dft):
    """A stand-in for the package name at the version of MDAPY with its module
    part, and its module _dft: dft itself, an ImportError of this message when
    it loads, or (None) no such module, as a build without the DFT."""
    package = types.ModuleType(name)
    package.__version__ = mdapy.__version__
    package.__path__ = []
    gpu = types.ModuleType(f"{name}.{part}")
    gpu.device_count = lambda: 1
    setattr(package, part, gpu)
    monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, f"{name}.{part}", gpu)
    monkeypatch.delitem(sys.modules, f"{name}._dft", raising=False)
    finders = [f for f in sys.meta_path if not isinstance(f, Unloadable)]
    if isinstance(dft, types.ModuleType):
        monkeypatch.setitem(sys.modules, f"{name}._dft", dft)
    elif isinstance(dft, str):
        finders.insert(0, Unloadable(f"{name}._dft", dft))
    monkeypatch.setattr(sys, "meta_path", finders)


def test_without_a_gpu_part(monkeypatch, tmp_path):
    not_built_in(monkeypatch)
    with pytest.raises(InvalidParameter, match=r'pip install "mdapy\[cuda12\]"'):
        loader.module()
    # the calculator says so when it is made, after its parameters
    with pytest.raises(InvalidParameter, match="encut must be positive"):
        DFT(-1.0, pseudopotentials=tmp_path)
    with pytest.raises(InvalidParameter, match=r"GPU part of MDAPY, which is not"):
        DFT(600.0, pseudopotentials=tmp_path)


def test_the_dft_of_mdapy_cuda12(monkeypatch, tmp_path):
    not_built_in(monkeypatch)
    dft = types.ModuleType("mdapy_cuda12._dft")
    install(monkeypatch, "mdapy_cuda12", "_cuda", dft)
    assert loader.module() is dft
    # a module that does not load, which names no library of NVIDIA
    install(monkeypatch, "mdapy_cuda12", "_cuda", LIBSTDCXX)
    with pytest.raises(InvalidParameter) as error:
        DFT(600.0, pseudopotentials=tmp_path)
    message = str(error.value)
    assert "the module _dft of the GPU part could not be loaded" in message
    assert LIBSTDCXX in message
    # built without the DFT
    install(monkeypatch, "mdapy_cuda12", "_cuda", None)
    with pytest.raises(
        InvalidParameter, match="mdapy-cuda12 was built without the DFT"
    ):
        loader.module()


def test_the_dft_of_mdapy_rocm6(monkeypatch):
    not_built_in(monkeypatch)
    dft = types.ModuleType("mdapy_rocm6._dft")
    install(monkeypatch, "mdapy_rocm6", "_hip", dft)
    assert loader.module() is dft
    install(monkeypatch, "mdapy_rocm6", "_hip", HIPBLAS)
    with pytest.raises(InvalidParameter) as error:
        loader.module()
    message = str(error.value)
    assert "hipBLAS, hipFFT and hipSOLVER of ROCm" in message
    assert HIPBLAS in message
    assert "nvidia" not in message.lower()


def test_the_dft_built_with_mdapy(monkeypatch):
    not_built_in(monkeypatch)
    part = types.ModuleType("mdapy._cuda")
    monkeypatch.setitem(sys.modules, "mdapy._cuda", part)
    monkeypatch.setattr(mdapy, "_cuda", part, raising=False)
    # the GPU part built in comes first, as for the calculators on the GPU
    install(monkeypatch, "mdapy_cuda12", "_cuda", types.ModuleType("other"))
    with pytest.raises(InvalidParameter, match="MDAPY was built without the DFT"):
        loader.module()
    dft = types.ModuleType("mdapy._dft")
    monkeypatch.setitem(sys.modules, "mdapy._dft", dft)
    assert loader.module() is dft


def test_another_error_of_import(monkeypatch):
    """A module missing inside _dft is not taken for a build without it."""
    not_built_in(monkeypatch)
    install(monkeypatch, "mdapy_cuda12", "_cuda", None)

    class Inner:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "mdapy_cuda12._dft":
                raise ModuleNotFoundError("No module named 'inner'", name="inner")
            return None

    monkeypatch.setattr(sys, "meta_path", [Inner()] + sys.meta_path)
    with pytest.raises(ModuleNotFoundError, match="inner"):
        loader.module()


def test_the_libraries_of_the_vendor(monkeypatch):
    """vendor_libraries: the files the module loaded, or what to install as
    InvalidParameter where it cannot load them."""
    dft = types.ModuleType("mdapy_cuda12._dft")
    dft.vendor_libraries = lambda: ["/nvidia/cublas/lib/libcublas.so.12"]
    assert loader.vendor_libraries(dft) == ["/nvidia/cublas/lib/libcublas.so.12"]
    missing = 'NVIDIA\'s cuBLAS, cuFFT and cuSOLVER ... pip install "mdapy[cuda12,dft]"'

    def refuse():
        raise RuntimeError(missing)

    dft.vendor_libraries = refuse
    with pytest.raises(InvalidParameter) as error:
        loader.vendor_libraries(dft)
    assert str(error.value) == missing
