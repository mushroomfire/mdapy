# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Many frames at a time, a frame on a thread: the same as frame by frame."""

import numpy as np
import pytest
from mdapy._core import neighbor as _neighbors

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import NEP
from mdapy.core._frames import Frames, batches
from mdapy.elements import COVALENT_RADII
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.mlip import SOAP, check_frames, committee, predict, structure_descriptors
from mdapy.neighbor import NeighborList

from calculators.nep_files import write_nep


class Alone:
    """A model computed frame by frame: one without the paths of many frames."""

    def __init__(self, model):
        self.model = model
        self.implemented = getattr(model, "implemented", frozenset())

    def compute(self, system, properties=("energy", "forces")):
        return self.model.compute(system, properties)

    def descriptors(self, system):
        return self.model.descriptors(system)


@pytest.fixture(scope="module")
def frames():
    """Frames of all kinds: a box smaller than the cutoff, so that atoms see
    their images; shaken crystals; a triclinic box; a cluster without
    periodic directions and a slab with one; a single atom; no atom."""
    rng = np.random.default_rng(4)
    found = []
    for repeat, a in [((2, 2, 2), 3.6), ((3, 3, 3), 3.55), ((1, 1, 2), 3.6)]:
        cell = build_crystal("fcc", a, elements="Cu", repeat=repeat)
        elements = rng.choice(["Cu", "Ni"], len(cell))
        x = cell["positions"] + rng.normal(0, 0.08, (len(cell), 3))
        found.append(cell.with_columns(positions=x, element=elements))
    cell = found[1]
    tilted = mdapy.Box(
        cell.box.matrix @ (np.eye(3) + [[0, 0.1, 0], [0, 0, 0.05], [0, 0, 0]])
    )
    found.append(cell.with_box(tilted, scale_positions=True))
    cluster = found[1].with_box(mdapy.Box(np.eye(3) * 30.0, pbc=(False, False, False)))
    found.append(cluster)
    found.append(
        found[1].with_box(mdapy.Box(found[1].box.matrix, pbc=(True, True, False)))
    )
    found.append(mdapy.System([[0.0, 0, 0]], mdapy.Box(8.0), elements=["Ni"]))
    found.append(mdapy.System(np.zeros((0, 3)), mdapy.Box(8.0), elements=[]))
    return found


@pytest.fixture(scope="module")
def nep(tmp_path_factory):
    path = write_nep(
        tmp_path_factory.mktemp("nep") / "nep.txt", ("Cu", "Ni"), zbl="typewise"
    )
    return NEP(path)


def test_nep_of_many_frames(frames, nep):
    """The energies and forces of the atoms bit for bit; the virials and the
    mean descriptors of the frames to the rounding of their sums; whatever
    the threads."""
    packed = Frames(frames)
    energies, forces, virials = nep._frame_results(packed)
    means = nep._frame_descriptors(packed)
    for k, frame in enumerate(frames):
        lo, hi = packed.starts[k], packed.starts[k + 1]
        if len(frame) == 0:
            assert np.isnan(means[k]).all() and (virials[k] == 0).all()
            continue
        alone = nep.compute(frame, ("energy", "forces", "virial"))
        np.testing.assert_array_equal(energies[lo:hi], alone.atomic_energies)
        np.testing.assert_array_equal(forces[lo:hi], alone.forces)
        np.testing.assert_allclose(virials[k], alone.virial, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            means[k], nep.descriptors(frame).mean(axis=0), rtol=1e-12
        )
    with mdapy.num_threads(1):
        one = nep._frame_results(packed)
        np.testing.assert_array_equal(nep._frame_descriptors(packed), means)
    for got, want in zip(one, (energies, forces, virials)):
        np.testing.assert_array_equal(got, want)


def test_predict_committee_descriptors(frames, nep, tmp_path):
    """predict, committee and structure_descriptors of models of mdapy,
    many frames at a time, against the same models frame by frame; of a
    trajectory read as needed, in batches."""
    labeled = []
    for frame in frames[:-1]:
        r = nep.compute(frame, ("energy", "forces", "virial"))
        labeled.append(
            frame.with_columns(forces=r.forces * 1.1).with_info(
                energy=r.energy, virial=r.virial
            )
        )
    fast, slow = predict(labeled, nep), predict(labeled, Alone(nep))
    for name in ("energy", "forces", "virial"):
        np.testing.assert_allclose(
            getattr(fast, name), getattr(slow, name), rtol=1e-12, atol=1e-12
        )
    other = NEP(write_nep(tmp_path / "other.txt", ("Cu", "Ni"), seed=1))
    fast = committee(labeled, [nep, other])
    slow = committee(labeled, [Alone(nep), Alone(other)])
    np.testing.assert_allclose(fast.forces, slow.forces, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fast.energy, slow.energy, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(fast.max_force, slow.max_force)
    soap = SOAP(["Cu", "Ni"], r_cut=4.5, n_max=4, l_max=3)
    for model in (nep, soap):
        np.testing.assert_allclose(
            structure_descriptors(labeled, model),
            structure_descriptors(labeled, Alone(model)),
            rtol=1e-12,
        )
    mdapy.write(tmp_path / "frames.xyz", labeled)
    read = mdapy.read_trajectory(tmp_path / "frames.xyz")
    np.testing.assert_allclose(
        structure_descriptors(read, nep),
        structure_descriptors(labeled, nep),
        rtol=1e-10,
    )
    assert structure_descriptors([], nep).size == 0


def test_closest_pairs(frames):
    """check_frames against the pairs of a neighbor list of each frame."""
    close = 0.95
    table = check_frames(frames, close=close)
    for k, frame in enumerate(frames):
        expected = np.inf
        if len(frame) > 1:
            radii = COVALENT_RADII[frame.table["element"].to_physical().to_numpy()]
            pairs = NeighborList.build(frame, close * 2 * radii.max())
            i = np.repeat(np.arange(len(frame)), np.diff(pairs.offsets))
            ratio = pairs.distances / (radii[i] + radii[pairs.j])
            if len(ratio) and ratio.min() < close:
                expected = ratio.min()
        assert table["closest"][k] == expected
    assert (
        np.isfinite(table["closest"].to_numpy()).sum() >= 2
    )  # shaken frames come that close
    assert table["frame"].to_list() == list(range(len(frames)))


def test_batches_and_errors(frames, nep):
    assert [len(b) for b in batches(range(25), 10)] == [10, 10, 5]
    assert list(batches([], 10)) == []
    copper = Frames([build_crystal("fcc", 3.6, elements="Cu", repeat=(2, 2, 2))])
    with pytest.raises(InvalidParameter, match=r"no parameters for \['Al'\]"):
        nep._frame_results(
            Frames([build_crystal("fcc", 4.05, elements="Al", repeat=(2, 2, 2))])
        )
    with pytest.raises(InvalidParameter, match="Al is not among the species"):
        SOAP(["Cu"])._frame_descriptors(
            Frames([build_crystal("fcc", 4.05, elements="Al", repeat=(2, 2, 2))])
        )
    bare = mdapy.System([[0.0, 0, 0], [1, 1, 1]], mdapy.Box(5.0))
    for call in (
        lambda: nep._frame_descriptors(Frames([bare])),
        lambda: SOAP(["Cu"])._frame_descriptors(Frames([bare])),
        lambda: check_frames([bare]),
    ):
        with pytest.raises(MissingProperty, match="element column"):
            call()
    starts, positions, matrices, origins, pbc = copper.arrays()
    with pytest.raises(ValueError, match="do not match"):
        _neighbors.closest_pairs(
            starts[:1], positions, matrices, origins, pbc, np.ones(4), 0.5
        )
    with pytest.raises(ValueError, match="one value per atom"):
        _neighbors.closest_pairs(
            starts, positions, matrices, origins, pbc, np.ones(3), 0.5
        )
    with pytest.raises(ValueError, match="decrease"):
        _neighbors.closest_pairs(
            np.array([0, 3, 2, 4]),
            positions[:4],
            np.repeat(matrices, 3, 0),
            np.repeat(origins, 3, 0),
            np.repeat(pbc, 3, 0),
            np.ones(4),
            0.5,
        )
