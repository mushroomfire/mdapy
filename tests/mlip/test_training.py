# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM, NEP
from mdapy.calculators._base import Calculator, CalculatorResult
from mdapy.errors import InvalidParameter
from mdapy.mlip import check_frames, committee, novelty, predict, read_gpumd

from calculators.nep_files import write_nep


class Scaled(Calculator):
    """Another model: the energies, forces and virials of one scaled."""

    def __init__(self, model, factor, virial=True):
        self.model, self.factor = model, factor
        self.cutoff, self.elements = model.cutoff, model.elements
        self.implemented = frozenset(
            ["energy", "forces", "virial"] if virial else ["energy", "forces"]
        )

    def compute(self, system, properties=("energy", "forces"), *, neighbors=None):
        r = self.model.compute(system, ("energy", "forces", "virial"))
        return CalculatorResult(
            energy=self.factor * r.energy,
            forces=self.factor * r.forces,
            virial=self.factor * r.virial if "virial" in properties else None,
        )


@pytest.fixture(scope="module")
def frames():
    """Six frames of shaken and strained copper of 32 to 108 atoms, labeled
    by the EAM of Zhou."""
    rng = np.random.default_rng(0)
    model = EAM.zhou(["Cu"])
    found = []
    for k, repeat in enumerate([(2, 2, 2), (3, 3, 3), (2, 2, 3)] * 2):
        cell = build_crystal("fcc", 3.615, elements="Cu", repeat=repeat)
        box = mdapy.Box(cell.box.matrix @ (np.eye(3) + rng.normal(0, 0.01, (3, 3))))
        shaken = cell.with_box(box, scale_positions=True)
        shaken = shaken.with_columns(
            positions=shaken["positions"]
            + rng.normal(0, 0.05 * (k + 1), (len(cell), 3))
        )
        r = model.compute(shaken, ("energy", "forces", "virial"))
        found.append(
            shaken.with_columns(forces=r.forces).with_info(
                energy=r.energy, virial=r.virial
            )
        )
    return found


def test_predict_against_the_labels(frames, tmp_path):
    """The model of the labels predicts them; another one, its energies
    and forces scaled by 1.1, errs by a tenth of them; through extended
    XYZ the same."""
    model = EAM.zhou(["Cu"])
    same = predict(frames, model)
    np.testing.assert_allclose(same.energy[:, 1], same.energy[:, 0], rtol=1e-12)
    assert max(same.rmse().values()) < 1e-10
    other = predict(frames, Scaled(model, 1.1))
    energies = np.array([f.info["energy"] / len(f) for f in frames])
    np.testing.assert_allclose(
        other.energy[:, 1] - other.energy[:, 0], 0.1 * energies, rtol=1e-12
    )
    forces = np.concatenate([f["forces"] for f in frames])
    assert other.rmse()["forces"] == pytest.approx(
        0.1 * np.sqrt((forces**2).mean()), rel=1e-12
    )
    table = other.table()
    assert table.columns == [
        "frame",
        "atoms",
        "energy",
        "energy_error",
        "force_rmse",
        "force_max_error",
        "virial_rmse",
    ]
    for k, frame in enumerate(frames):
        f = frame["forces"]
        assert table["force_rmse"][k] == pytest.approx(0.1 * np.sqrt((f**2).mean()))
        assert table["force_max_error"][k] == pytest.approx(0.1 * np.abs(f).max())
    worst = other.worst(2)
    assert worst.tolist() == np.argsort(-table["force_rmse"].to_numpy())[:2].tolist()
    assert other.worst(1, by="energy_error")[0] == np.argmax(np.abs(energies))
    with pytest.raises(InvalidParameter, match="by must be"):
        other.worst(1, by="energy")
    np.testing.assert_array_equal(other.frame_of_atom[: len(frames[0])], 0)
    mdapy.write(tmp_path / "train.xyz", frames)
    read = predict(
        list(mdapy.read_trajectory(tmp_path / "train.xyz")), Scaled(model, 1.1)
    )
    np.testing.assert_allclose(read.forces, other.forces, rtol=1e-6, atol=1e-8)
    no_virial = predict(frames, Scaled(model, 1.1, virial=False))
    assert np.isnan(no_virial.virial[:, 1]).all() and np.isnan(
        no_virial.rmse()["virial"]
    )


def test_frames_without_labels(frames):
    bare = [
        f.with_table(f.table.drop("forces")).with_info(energy=None, virial=None)
        for f in frames[:2]
    ]
    result = predict(bare, EAM.zhou(["Cu"]))
    assert np.isnan(result.energy[:, 0]).all() and np.isnan(result.forces[:, 0]).all()
    assert np.isnan(result.rmse()["energy"])


def test_gpumd_outputs(frames, tmp_path):
    """The files NEP of GPUMD writes: per-atom energies and virials, the
    prediction first, a missing reference virial marked."""
    model = EAM.zhou(["Cu"])
    expected = predict(frames, Scaled(model, 1.1))
    mdapy.write(tmp_path / "train.xyz", frames)
    with open(tmp_path / "train.xyz", "a") as file:  # blank lines at the end
        file.write("\n\n")
    energy = np.column_stack([expected.energy[:, 1], expected.energy[:, 0]])
    np.savetxt(tmp_path / "energy_train.out", energy)
    forces = np.column_stack([expected.forces[:, 1], expected.forces[:, 0]])
    np.savetxt(tmp_path / "force_train.out", forces)
    order = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]
    virial = np.column_stack(
        [expected.virial[:, 1, i, j] for i, j in order]
        + [expected.virial[:, 0, i, j] for i, j in order]
    )
    virial[0, 6:] = -1e6
    np.savetxt(tmp_path / "virial_train.out", virial)
    read = read_gpumd(tmp_path)
    np.testing.assert_array_equal(read.atoms, expected.atoms)
    np.testing.assert_allclose(read.energy, expected.energy, rtol=1e-12)
    np.testing.assert_allclose(read.forces, expected.forces, rtol=1e-12)
    np.testing.assert_allclose(read.virial[1:], expected.virial[1:], rtol=1e-12)
    assert np.isnan(read.virial[0, 0]).all()
    np.savetxt(tmp_path / "energy_test.out", energy[:2])
    with pytest.raises(InvalidParameter, match="cannot read"):
        read_gpumd(tmp_path, "test")
    np.savetxt(tmp_path / "force_test.out", forces[:5])
    np.savetxt(tmp_path / "virial_test.out", virial[:2])
    mdapy.write(tmp_path / "test.xyz", frames[:2])
    with pytest.raises(InvalidParameter, match="do not hold the frames"):
        read_gpumd(tmp_path, "test")
    np.savetxt(tmp_path / "virial_test.out", virial[:2, :6])
    with pytest.raises(InvalidParameter, match="columns"):
        read_gpumd(tmp_path, "test")
    with pytest.raises(InvalidParameter, match="train or test"):
        read_gpumd(tmp_path, "valid")


def test_torchnep_outputs(frames, tmp_path):
    """The predictions torchnep writes for frames elsewhere, one without an
    energy and one without a virial: those of the same NEP model in mdapy,
    to the ten digits of the files."""
    pytest.importorskip("torchnep")
    from torchnep.predict import predict_dataset

    labeled = list(frames)
    labeled[2] = labeled[2].with_info(virial=None)
    labeled[4] = labeled[4].with_info(energy=None)
    xyz = tmp_path / "data" / "all.xyz"
    xyz.parent.mkdir()
    mdapy.write(xyz, labeled)
    nep = write_nep(tmp_path / "nep.txt", ("Cu",))
    out = tmp_path / "predicted"
    predict_dataset(
        str(nep), str(xyz), str(out), dtype="float64", device="cpu", verbose=False
    )
    read = read_gpumd(out, xyz=xyz)
    expected = predict(labeled, NEP(nep))
    np.testing.assert_array_equal(read.atoms, expected.atoms)
    for name in ("energy", "forces", "virial"):
        np.testing.assert_allclose(
            getattr(read, name), getattr(expected, name), rtol=1e-8, atol=1e-8
        )
    assert np.isnan(read.energy[4, 0]) and np.isnan(read.virial[2, 0]).all()


def test_committee(frames):
    """Two models of energies and forces 1 and 1.2 times those of one: half
    their difference."""
    model = EAM.zhou(["Cu"])
    spread = committee(frames, [model, Scaled(model, 1.2)])
    energies = np.array([f.info["energy"] / len(f) for f in frames])
    np.testing.assert_allclose(spread.energy, 0.1 * np.abs(energies), rtol=1e-12)
    forces = np.concatenate([f["forces"] for f in frames])
    np.testing.assert_allclose(
        spread.forces, 0.1 * np.linalg.norm(forces, axis=1), rtol=1e-10
    )
    start = 0
    for k, n in enumerate(spread.atoms):
        assert spread.max_force[k] == pytest.approx(
            spread.forces[start : start + n].max()
        )
        start += n
    with pytest.raises(InvalidParameter, match="two or more"):
        committee(frames, [model])
    with pytest.raises(InvalidParameter, match="no frames"):
        committee([], [model, model])
    with pytest.raises(InvalidParameter, match="no frames"):
        predict([], model)


def test_novelty():
    """Distances to the nearest reference point, or to the nearest other
    point, by brute force; a repeated point has none."""
    rng = np.random.default_rng(1)
    points = rng.normal(size=(50, 7))
    reference = rng.normal(size=(30, 7))
    expected = np.linalg.norm(points[:, None] - reference[None], axis=2).min(axis=1)
    np.testing.assert_allclose(novelty(points, reference), expected, rtol=1e-10)
    repeated = np.vstack([points, points[3:4]])
    alone = novelty(repeated)
    distances = np.linalg.norm(repeated[:, None] - repeated[None], axis=2)
    np.fill_diagonal(distances, np.inf)
    np.testing.assert_allclose(alone, distances.min(axis=1), atol=1e-7)
    assert alone[3] == alone[-1] == pytest.approx(0, abs=1e-7)
    for bad in [
        (points[0], None),
        (points, reference[:, :3]),
        (points[:1], None),
        (points, np.zeros((0, 7))),
    ]:
        with pytest.raises(InvalidParameter):
            novelty(*bad)


def test_check_frames(frames):
    """Atoms too close, a net force and missing labels, frame by frame."""
    x = frames[0]["positions"].copy()
    x[0] = x[1] + [0.8, 0, 0]  # 0.3 of the two covalent radii of copper
    close = frames[0].with_columns(positions=x)
    pushed = frames[1].with_columns(forces=frames[1]["forces"] + [0.5, 0, 0])
    bare = frames[2].with_table(frames[2].table.drop("forces")).with_info(energy=None)
    one = mdapy.System([[0.0, 0, 0]], mdapy.Box(10.0), elements=["Cu"])
    table = check_frames([close, pushed, bare, frames[3], one])
    assert table["too_close"].to_list() == [True, False, False, False, False]
    assert table["closest"][0] == pytest.approx(0.8 / (2 * 1.32), rel=0.01)
    assert table["closest"][3] == np.inf
    assert table["net_force"][1] == pytest.approx(0.5 * len(pushed), rel=1e-6)
    assert table["net_force"][3] < 1e-8 and np.isnan(table["net_force"][2])
    assert table["has_energy"].to_list() == [True, True, False, True, False]
    assert table["has_forces"].to_list() == [True, True, False, True, False]
    assert table["has_virial"].to_list()[:4] == [True, True, True, True]
    loose = check_frames([frames[3]], close=1.2)
    assert loose["closest"][0] < 1.2 and loose["too_close"][0]
    with pytest.raises(InvalidParameter, match="close"):
        check_frames(frames, close=0)
