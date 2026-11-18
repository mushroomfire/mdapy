"""The device and the precision of calculators, and what a calculator on the
GPU does with the module of the GPU, here a stand-in that computes on the
processor; tests/cuda compares the GPU itself with the processor."""

import sys

import numpy as np
import pytest

from mdapy.calculators import EAM, LennardJones, Morse, _gpu
from mdapy.errors import InvalidParameter

from .eam_files import alloy
from .gpu_stand_in import stand_in


def test_device_and_precision():
    assert (EAM.zhou(["Cu"]).device, EAM.zhou(["Cu"]).precision) == ("cpu", "double")
    assert _gpu.resolve("cuda", None) == ("cuda", "mixed")
    assert _gpu.resolve("cuda", "double") == ("cuda", "double")
    with pytest.raises(InvalidParameter, match="device must be one of"):
        EAM.zhou(["Cu"], device="gpu")
    with pytest.raises(InvalidParameter, match="precision must be one of"):
        LennardJones((0.01, 3.4), 8.0, device="cuda", precision="single")
    with pytest.raises(InvalidParameter, match="double precision only"):
        Morse({"Cu": (0.3, 1.4, 2.5)}, 6.0, precision="mixed")


@pytest.mark.skipif(
    "mdapy._cuda" in sys.modules or _gpu.__spec__ is None,
    reason="the GPU part is there",
)
def test_without_the_gpu_part():
    try:
        import mdapy._cuda  # noqa: F401
    except ImportError:
        with pytest.raises(InvalidParameter, match="built without its GPU part"):
            EAM.zhou(["Cu"], device="cuda")
    else:
        pytest.skip("the GPU part is there")


def test_a_calculator_on_the_gpu(monkeypatch):
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    system = alloy(repeat=4, a=3.6)
    cpu, gpu = EAM.zhou(["Cu", "Ni"]), EAM.zhou(["Cu", "Ni"], device="cuda")
    everything = ("energy", "atomic_energies", "forces", "virial", "atomic_virials")
    a, b = gpu.compute(system, everything), cpu.compute(system, everything)
    for name in everything:
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=1e-12)
    # only what is asked, and the virial computed only then
    forces = gpu.compute(system, ("forces",))
    assert forces.energy is None and forces.atomic_virials is None
    potential = gpu._EAM__gpu.potential
    assert potential.asked[-2:] == [(True, True, True), (False, False, False)]
    gpu.compute(system, ("virial",))
    assert potential.asked[-1] == (False, True, False)
    # other atoms, other arrays
    small = alloy(repeat=3, a=3.6)
    with pytest.raises(InvalidParameter, match="too thin"):
        gpu.compute(small)
    lj = LennardJones({"Cu": (0.4, 2.3), "Ni": (0.5, 2.2)}, 4.0, device="cuda")
    np.testing.assert_allclose(
        lj.compute(system).forces,
        LennardJones({"Cu": (0.4, 2.3), "Ni": (0.5, 2.2)}, 4.0).compute(system).forces,
    )
    assert "device='cuda', precision='mixed'" in repr(lj)
    assert "device='cuda', precision='mixed'" in repr(gpu)
    assert "average atom" in repr(gpu.average({"Cu": 0.5, "Ni": 0.5}))


def test_no_device(monkeypatch):
    fake = stand_in(found=False)
    monkeypatch.setitem(sys.modules, "mdapy._cuda", fake)
    import mdapy

    monkeypatch.setattr(mdapy, "_cuda", fake, raising=False)
    with pytest.raises(InvalidParameter, match="no CUDA device"):
        EAM.zhou(["Cu"], device="cuda")
    found = stand_in()
    monkeypatch.setattr(mdapy, "_cuda", found)
    assert _gpu.cuda() is found
