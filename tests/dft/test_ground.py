# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The ground state of the DFT in a file: on the processor the file and its
checks, from a ground state made up; on a GPU the ground state of real runs,
its sums, and computations started from it."""

import json

import numpy as np
import pytest

import mdapy
from mdapy.calculators._dft import ground
from mdapy.errors import InvalidParameter

from dft import espresso, gpu

CELL = np.diag([4.05, 4.05, 4.05])
POSITIONS = np.array([[0.04, 0, 0], [2.025, 2.025, 0]])


def made_up(becsum=True):
    rng = np.random.default_rng(1)
    state = {
        "parameters": {
            "mdapy": "2.3.0",
            "pseudopotentials": {"Al": {"file": "Al.upf", "sha256": "ab" * 32}},
        },  # fmt: skip
        "elements": ["Al", "Al"],
        "cell": CELL,
        "positions": POSITIONS,
        "energy": -115.25,
        "fermi": 7.5,
        "forces": rng.normal(size=(2, 3)),
        "density": rng.random((1, 6, 6, 6)),
        "kpoints": np.zeros((1, 3)),
        "weights": np.ones(1),
        "eigenvalues": rng.normal(size=(1, 1, 8)),
        "occupations": rng.random((1, 1, 8)),
    }
    if becsum:
        state["becsum"] = rng.random((1, 10))
    return state


def test_a_file_reads_back(tmp_path):
    path = tmp_path / "al.npz"
    state = made_up()
    ground.write(path, state)
    back = ground.read(path)
    assert set(back) == set(state)
    assert back["parameters"] == state["parameters"] and back["elements"] == [
        "Al",
        "Al",
    ]
    assert back["energy"] == -115.25 and back["fermi"] == 7.5
    for name in ("cell", "positions", "forces", "density", "kpoints", "weights",
                 "eigenvalues", "occupations", "becsum"):  # fmt: skip
        np.testing.assert_array_equal(back[name], state[name])
    # NumPy reads it alone, without pickles
    with np.load(path, allow_pickle=False) as data:
        assert (
            str(data["format"]) == "MDAPY DFT ground state"
            and int(data["version"]) == 1
        )
        assert json.loads(str(data["parameters"])) == state["parameters"]
    # written whole: no file left beside it
    assert [p.name for p in tmp_path.iterdir()] == ["al.npz"]
    ground.write(path, made_up(becsum=False))
    assert "becsum" not in ground.read(path)


def test_files_that_are_not_ground_states(tmp_path):
    with pytest.raises(InvalidParameter, match="not a ground state"):
        ground.read(tmp_path / "none.npz")
    (tmp_path / "text.npz").write_text("density\n")
    with pytest.raises(InvalidParameter, match="not a ground state"):
        ground.read(tmp_path / "text.npz")
    np.save(tmp_path / "one.npy", np.zeros(3))
    with pytest.raises(InvalidParameter, match="one array"):
        ground.read(tmp_path / "one.npy")
    np.savez(tmp_path / "other.npz", density=np.zeros(3))
    with pytest.raises(InvalidParameter, match="not a ground state of MDAPY's DFT$"):
        ground.read(tmp_path / "other.npz")
    ground.write(tmp_path / "al.npz", made_up())
    with np.load(tmp_path / "al.npz") as data:
        contents = dict(data)
    np.savez(tmp_path / "later.npz", **{**contents, "version": np.array(2)})
    with pytest.raises(InvalidParameter, match="version 2 .* reads version 1"):
        ground.read(tmp_path / "later.npz")
    del contents["density"]
    np.savez(tmp_path / "part.npz", **contents)
    with pytest.raises(InvalidParameter, match="lacks density"):
        ground.read(tmp_path / "part.npz")
    with pytest.raises(InvalidParameter, match="cannot write"):
        ground.write(tmp_path / "no" / "al.npz", made_up())
    # a failure once the file is written leaves nothing beside the target
    (tmp_path / "folder").mkdir()
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(InvalidParameter, match="cannot write"):
        ground.write(tmp_path / "folder", made_up())
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_a_computation_must_be_of_the_ground_state():
    state = made_up()
    digests = {"Al": "ab" * 32}
    # the same atoms, one moved by a vector of the cell
    ground.check(state, "al.npz", ["Al", "Al"], CELL, POSITIONS + [[4.05, 0, 0], [0, 0, 0]],
                 digests)  # fmt: skip
    cases = [
        ((["Al", "Ni"], CELL, POSITIONS, digests), "other atoms"),
        ((["Al"], CELL, POSITIONS[:1], digests), "other atoms"),
        ((["Al", "Al"], CELL * 1.001, POSITIONS, digests), "another cell"),
        ((["Al", "Al"], CELL, POSITIONS + [0, 0, 1e-4], digests), "atoms elsewhere"),
        ((["Al", "Al"], CELL, POSITIONS, {"Al": "cd" * 32}), "another pseudopotential of Al"),
    ]  # fmt: skip
    for arguments, message in cases:
        with pytest.raises(InvalidParameter, match=message):
            ground.check(state, "al.npz", *arguments)
    ground.check_grid(state, "al.npz", 1, (6, 6, 6))
    with pytest.raises(
        InvalidParameter, match="1 spin channels on the grid 6 6 6, this computation 2"
    ):
        ground.check_grid(state, "al.npz", 2, (6, 6, 6))
    with pytest.raises(InvalidParameter, match="this computation 1 on 8 8 8"):
        ground.check_grid(state, "al.npz", 1, (8, 8, 8))


def test_the_calculator_reads_it_before_the_gpu(tmp_path):
    from mdapy.calculators import DFT

    with pytest.raises(InvalidParameter, match="not a ground state"):
        DFT(300.0, pseudopotentials={}, start=tmp_path / "none.npz")


# -- on a GPU -----------------------------------------------------------------


def aluminium(shift=0.0):
    a = 4.05
    cell = np.diag([a, a, a])
    fractions = np.array([[0.01, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    return mdapy.System(fractions @ cell + shift, mdapy.Box(cell), elements=["Al"] * 4)


def calculator(**options):
    from mdapy.calculators import DFT

    settings = dict(pseudopotentials={"Al": espresso.pseudo("Al_std.upf")}, kpoints=(2, 2, 2),
                    sigma=0.2, tolerance=1e-9)  # fmt: skip
    settings.update(options)
    return DFT(300.0, **settings)


def test_the_ground_state_of_a_run(tmp_path):
    gpu.device()
    dft = calculator()
    with pytest.raises(InvalidParameter, match="nothing to save"):
        dft.save(tmp_path / "none.npz")
    system = aluminium()
    result = dft.compute(system)
    path = tmp_path / "al.npz"
    dft.save(path)
    state = ground.read(path)
    assert state["parameters"] == dft.last["parameters"]
    assert state["energy"] == result.energy and state["fermi"] == dft.last["fermi"]
    np.testing.assert_array_equal(state["forces"], result.forces)
    np.testing.assert_array_equal(state["positions"], system["positions"])
    grid = dft.last["grid"]
    assert state["density"].shape == (1, *grid)
    # 12 electrons in the cell, the density in electrons per Å^3
    volume = system.box.volume
    assert state["density"].mean() * volume == pytest.approx(12.0, rel=1e-8)
    kpoints, bands = len(state["weights"]), dft.last["parameters"]["arguments"]["bands"]
    assert state["kpoints"].shape == (kpoints, 3) and state[
        "weights"
    ].sum() == pytest.approx(1.0)
    assert (
        state["eigenvalues"].shape == state["occupations"].shape == (1, kpoints, bands)
    )
    # two electrons a band without spin
    filled = 2 * (state["occupations"][0] * state["weights"][:, None]).sum()
    assert filled == pytest.approx(12.0, abs=1e-8)
    assert state["eigenvalues"].min() < state["fermi"] < state["eigenvalues"].max()
    assert "becsum" not in state and "magnetic_moments" not in state


def test_a_computation_started_from_it(tmp_path):
    gpu.device()
    path = tmp_path / "al.npz"
    cold = calculator()
    result = cold.compute(aluminium())
    cold.save(path)
    stream_log = tmp_path / "dft.log"
    warm = calculator(start=path, log=stream_log)
    again = warm.compute(aluminium())
    assert again.energy == pytest.approx(result.energy, abs=1e-8)
    np.testing.assert_allclose(again.forces, result.forces, atol=1e-6)
    assert warm.last["iterations"] < cold.last["iterations"]
    assert f"from the ground state of {path}" in stream_log.read_text()
    assert "start" not in warm.last["parameters"]["arguments"]
    # the next computation starts cold, as without it
    warm.compute(aluminium())
    assert abs(warm.last["iterations"] - cold.last["iterations"]) <= 1
    # the atoms moved by a vector of the cell are the same structure
    calculator(start=path).compute(aluminium(shift=np.array([4.05, 0, 0])))
    with pytest.raises(InvalidParameter, match="atoms elsewhere"):
        calculator(start=path).compute(aluminium(shift=0.01))
    with pytest.raises(InvalidParameter, match="spin channels"):
        calculator(start=path, magmoms={"Al": 0.5}).compute(aluminium())
    with pytest.raises(InvalidParameter, match="on the grid"):
        calculator(start=path, ecutrho=1600.0).compute(aluminium())


def test_spin_and_the_occupations_of_paw(tmp_path):
    from mdapy.calculators import DFT

    gpu.device()
    dft = calculator(magmoms={"Al": 0.5}, max_iterations=60)
    dft.compute(aluminium())
    dft.save(tmp_path / "spin.npz")
    state = ground.read(tmp_path / "spin.npz")
    assert state["density"].shape[0] == 2 and state["eigenvalues"].shape[0] == 2
    np.testing.assert_array_equal(
        state["magnetic_moments"], dft.last["magnetic_moments"]
    )
    # one electron a band with spin
    filled = (state["occupations"] * state["weights"][None, :, None]).sum()
    assert filled == pytest.approx(12.0, abs=1e-8)
    path = espresso.pseudo("pslibrary/Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    h = 5.431 / 2
    cell = np.array([[0, h, h], [h, 0, h], [h, h, 0]])
    silicon = mdapy.System(np.array([[0.01, -0.02, 0.0], [0.27, 0.25, 0.23]]) @ cell,
                           mdapy.Box(cell), elements=["Si"] * 2)  # fmt: skip
    settings = dict(pseudopotentials={"Si": path}, kpoints=(2, 2, 2), smearing="fixed",
                    tolerance=1e-9)  # fmt: skip
    paw = DFT(300.0, **settings)
    first = paw.compute(silicon)
    paw.save(tmp_path / "si.npz")
    assert ground.read(tmp_path / "si.npz")["becsum"].shape[0] == 1
    warm = DFT(300.0, start=tmp_path / "si.npz", **settings)
    assert warm.compute(silicon).energy == pytest.approx(first.energy, abs=1e-7)
    assert warm.last["iterations"] < paw.last["iterations"]
