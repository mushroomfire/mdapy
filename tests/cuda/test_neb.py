# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The nudged elastic band with a potential on the GPU against the same
potential on the processor. Skipped without the GPU part of MDAPY or a
device."""

import numpy as np
import pytest

from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.simulation import minimize, neb

cuda = pytest.importorskip("mdapy._cuda")
try:
    cuda.device_name()
except RuntimeError:
    pytest.skip("no CUDA device", allow_module_level=True)


def vacancy_hop(repeat=4):
    eam = EAM.zhou(["Cu"])
    a = 3.615
    crystal = build_crystal("fcc", a, elements="Cu", repeat=(repeat,) * 3)
    x = crystal["positions"]
    hop = int(np.argmin(np.linalg.norm(x - [a / 2, a / 2, 0.0], axis=1)))
    keep = np.arange(len(crystal)) > 0
    moved = x.copy()
    moved[hop] = x[0]
    initial = minimize(crystal.filter(keep), eam, max_force=1e-6).system
    final = minimize(
        crystal.with_columns(positions=moved).filter(keep), eam, max_force=1e-6
    )
    return initial, final.system


@pytest.mark.parametrize("precision, energy", [("double", 1e-8), ("mixed", 1e-4)])
def test_the_band_on_the_gpu_is_the_band_on_the_processor(precision, energy):
    initial, final = vacancy_hop()
    cpu = neb(initial, final, EAM.zhou(["Cu"]), images=3, max_force=1e-4)
    gpu = neb(
        initial, final, EAM.zhou(["Cu"], device="cuda", precision=precision),
        images=3, max_force=1e-4,
    )  # fmt: skip
    assert gpu.converged and gpu.highest == cpu.highest
    np.testing.assert_allclose(gpu.energies, cpu.energies, rtol=0, atol=energy)
    assert gpu.barrier == pytest.approx(cpu.barrier, abs=energy)
    for a, b in zip(gpu.images, cpu.images):
        np.testing.assert_allclose(a["positions"], b["positions"], rtol=0, atol=1e-3)
