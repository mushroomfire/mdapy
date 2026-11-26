# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""DFT-D3 on the GPU against the processor: atom by atom, in double precision
to rounding and in mixed to single precision, in cells smaller than its
cutoffs, slabs and molecules, with both dampings and the three-body term,
many triangles in batches; overlaid on EAM; and the same runs of molecular
dynamics. Skipped without the GPU part of MDAPY or a device."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import D3, EAM, Overlay, _gpu
from mdapy.simulation import MolecularDynamics

cuda = _gpu.module()
if cuda is None:
    pytest.skip("no GPU part of MDAPY", allow_module_level=True)
try:
    cuda.device_name()
except RuntimeError:
    pytest.skip("no CUDA device", allow_module_level=True)

ALL = ("energy", "atomic_energies", "forces", "virial", "atomic_virials")
TOLERANCE = {"double": 1e-12, "mixed": 2e-5}


def random_cell(elements, seed, pbc=(True, True, True)):
    rng = np.random.default_rng(seed)
    h = np.diag([7.0, 6.5, 8.0]) + rng.normal(size=(3, 3)) * 0.8
    while True:
        x = rng.random((len(elements), 3)) @ h
        system = mdapy.System(x, mdapy.Box(h, pbc=pbc), elements=list(elements))
        if len(mdapy.NeighborList.build(system, 1.6)) == 0:
            return system


def molecule(seed=1):
    rng = np.random.default_rng(seed)
    elements = rng.choice(["H", "C", "N", "O", "S", "Fe"], 20)
    while True:
        x = rng.normal(size=(len(elements), 3)) * 3.0
        d = np.linalg.norm(x[:, None] - x[None], axis=-1) + 10 * np.eye(len(x))
        if d.min() > 1.4:
            return mdapy.System(x, mdapy.Box(30.0, pbc=False), elements=list(elements))


def crystal(repeat=3, seed=0):
    system = build_crystal("diamond", 5.43, elements="Si", repeat=(repeat,) * 3)
    rng = np.random.default_rng(seed)
    return system.with_columns(
        positions=system["positions"] + rng.normal(0, 0.05, (len(system), 3))
    )


SYSTEMS = {
    "small cell": lambda: random_cell(["C", "H", "O", "Mo", "S", "Ag", "C", "H"], 7),
    "slab": lambda: random_cell(
        ["C", "H", "O", "S", "Fe", "N"], 3, (True, True, False)
    ),
    "molecule": molecule,
    "crystal": crystal,
}


def assert_close(a, b, tolerance):
    assert a.energy == pytest.approx(b.energy, rel=tolerance)
    for name in ALL[1:]:
        scale = max(np.abs(getattr(b, name)).max(), 1e-3)
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=0,
                                   atol=tolerance * scale, err_msg=name)  # fmt: skip


@pytest.mark.parametrize("name", list(SYSTEMS))
@pytest.mark.parametrize("damping", ["bj", "zero"])
@pytest.mark.parametrize("three_body", [False, True])
@pytest.mark.parametrize("precision", ["double", "mixed"])
def test_as_on_the_processor(name, damping, three_body, precision):
    system = SYSTEMS[name]()
    options = dict(three_body=three_body, three_body_cutoff=9.0)
    cpu = D3("pbe", damping, **options).compute(system, ALL)
    gpu = D3("pbe", damping, device="cuda", precision=precision, **options)
    assert_close(gpu.compute(system, ALL), cpu, TOLERANCE[precision])


def test_triangles_in_batches():
    """More corners than the device holds at once, 2^20: in batches."""
    system = crystal(repeat=3)  # 216 atoms, ~1900 corners each within 20 Å
    options = dict(three_body=True, cutoff=20.0, cn_cutoff=12.0, three_body_cutoff=20.0)
    cpu = D3("pbe", **options).compute(system, ALL)
    big = mdapy.transform.replicate(system, (2, 2, 1))  # 1.4e6 corners
    gpu = D3("pbe", device="cuda", precision="mixed", **options).compute(big, ALL)
    assert gpu.energy == pytest.approx(4 * cpu.energy, rel=2e-5)
    np.testing.assert_allclose(gpu.forces[: len(system)], cpu.forces, atol=2e-5)


def test_the_elements_of_an_overlay():
    copper = build_crystal("fcc", 3.615, elements="Cu", repeat=(5, 5, 5))
    rng = np.random.default_rng(1)
    copper = copper.with_columns(
        element=rng.choice(["Cu", "Ni"], len(copper)).tolist(),
        positions=copper["positions"] + rng.normal(0, 0.05, (len(copper), 3)),
    )
    options = dict(cutoff=12.0, cn_cutoff=8.0)
    gpu = Overlay([
        EAM.zhou(["Cu", "Ni"], device="cuda", precision="double"),
        D3("pbe", elements=["Cu", "Ni"], device="cuda", precision="double", **options),
    ])  # fmt: skip
    cpu = Overlay([EAM.zhou(["Cu", "Ni"]), D3("pbe", **options)])
    assert gpu._on_device.cutoff == EAM.zhou(["Cu"]).cutoff
    assert_close(gpu.compute(copper, ALL), cpu.compute(copper, ALL), 1e-12)


def run(system, potential, steps, **options):
    with MolecularDynamics(system, potential, timestep=0.002) as md:
        md.velocities(600, seed=1)
        md.run(steps, thermo=steps // 4, **options)
        return md.system, md.thermo


@pytest.mark.parametrize(
    "options",
    [{"ensemble": "nve"}, {"ensemble": "npt", "temperature": 300, "pressure": 0.0}],
)
def test_the_runs_of_the_processor(options):
    """EAM with D3 added, in double precision: the trajectory of the processor
    to rounding; D3 alone in mixed precision, close to it."""
    copper = build_crystal("fcc", 3.615, elements="Cu", repeat=(5, 5, 5))
    d3 = dict(cutoff=12.0, cn_cutoff=8.0, elements=["Cu"])
    gpu = Overlay([EAM.zhou(["Cu"], device="cuda", precision="double"),
                   D3("pbe", device="cuda", precision="double", **d3)])  # fmt: skip
    cpu = Overlay([EAM.zhou(["Cu"]), D3("pbe", **d3)])
    a, ta = run(copper, gpu, 200, **options)
    b, tb = run(copper, cpu, 200, **options)
    d = a["positions"] - b["positions"]
    d -= np.round(d @ np.linalg.inv(b.box.matrix)) @ b.box.matrix
    assert np.abs(d).max() < 1e-7
    np.testing.assert_allclose(a["velocities"], b["velocities"], rtol=0, atol=1e-6)
    for column in ("temperature", "potential_energy", "pressure"):
        np.testing.assert_allclose(ta[column], tb[column], rtol=1e-8, atol=1e-8)
    alone = random_cell(["C", "H", "O", "S"] * 4, seed=2)
    x, _ = run(alone, D3("pbe", device="cuda"), 100)
    y, _ = run(alone, D3("pbe"), 100)
    d = x["positions"] - y["positions"]
    d -= np.round(d @ np.linalg.inv(y.box.matrix)) @ y.box.matrix
    assert np.abs(d).max() < 1e-5
