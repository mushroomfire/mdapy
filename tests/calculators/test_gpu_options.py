"""The device and the precision of calculators, and what a calculator on the
GPU does with the module of the GPU, here a stand-in that computes on the
processor; tests/cuda compares the GPU itself with the processor."""

import sys
import types

import numpy as np
import pytest

from mdapy import _core
from mdapy.calculators import EAM, LennardJones, Morse, _gpu
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList

from .eam_files import alloy


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


class _Atoms:
    def __init__(self, n):
        self.size = n
        self.species = self.positions = self.result = None

    def set_species(self, species):
        self.species = species.copy()

    def set_positions(self, positions):
        self.positions = positions.copy()

    def forces(self):
        return self.result[1]

    def energies(self):
        return self.result[0]

    def virials(self):
        return self.result[2]

    def total_energy(self):
        return float(self.result[0].sum())

    def total_virial(self):
        return self.result[2].sum(axis=0)


class _Neighbors:
    def __init__(self, cutoff, skin):
        self.cutoff, self.skin = cutoff, skin

    def update(self, atoms, h, lo, pbc):
        if min(np.diag(h)) <= 2 * (self.cutoff + self.skin):
            raise ValueError("the box is too thin")
        return True


class _Potential:
    """A potential of the stand-in: the kernels of the processor."""

    def __init__(self, make, compute):
        self.native, self.compute_native = make, compute
        self.asked = []

    def compute(self, atoms, neighbors, h, lo, pbc, energy, virial, atomic):
        self.asked.append((energy, virial, atomic))
        system = System(atoms.positions, Box(h, origin=lo, pbc=pbc))
        nl = NeighborList.build(system, neighbors.cutoff)
        atoms.result = self.compute_native(
            self.native, atoms.species, atoms.positions, h, nl.offsets, nl.j,
            nl.shifts, None, None,
        )  # fmt: skip


def stand_in(found=True):
    native = _core.calculators

    def make_eam(embedding, density, pair, drho, dr, cutoff, rho_max, precision):
        made = native.make_eam(
            embedding, density, pair, drho, dr, cutoff, rho_max=rho_max
        )
        return _Potential(made, native.compute_eam)

    def make_pair(style, species, cutsq, offset, coefficients, precision):
        made = native.make_pair(style, cutsq, offset, coefficients)
        return _Potential(made, native.compute_pair)

    def device_name():
        if not found:
            raise RuntimeError("no CUDA device: none was found")
        return "stand-in"

    return types.SimpleNamespace(
        Atoms=_Atoms, Neighbors=_Neighbors, make_eam=make_eam, make_pair=make_pair,
        device_name=device_name,
    )  # fmt: skip


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
