# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Restart files: a run stopped in its middle and finished from the file
gives the run that did not stop, in every ensemble and with every control,
to the rounding of the forces, whose neighbor lists are built again and sum
in another order; between runs too; and files of others or of nothing are
refused."""

import io
import os
import pickle

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM, Springs
from mdapy.errors import InvalidParameter
from mdapy.simulation import MolecularDynamics, _restart
from mdapy.simulation.controls import (
    AdaptiveTimestep,
    AddForce,
    Deform,
    ElectronicStopping,
    Hold,
    Indenter,
    Langevin,
    Move,
    Switch,
    Wall,
)


def copper(repeat=4, tilt=0.0):
    crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(repeat,) * 3)
    if tilt:
        matrix = crystal.box.matrix.copy()
        matrix[1, 0] += tilt * matrix[0, 0]
        crystal = crystal.with_box(mdapy.Box(matrix), scale_positions=True)
    return crystal


def z_of(system):
    return system["positions"][:, 2]


def runs():
    """The settings of a run and the system it starts from."""
    crystal = copper()
    z = z_of(crystal)
    bottom, top = z < z.min() + 2.0, z > z.max() - 2.0
    middle = ~(bottom | top)
    springs = Springs(crystal["positions"], np.full(len(crystal), 2.0))
    return {
        "nve": (crystal, {}),
        "nvt": (crystal, {"ensemble": "nvt", "temperature": (300, 500)}),
        "npt iso": (
            crystal,
            {"ensemble": "npt", "temperature": 300, "pressure": (0.0, 2.0)},
        ),
        "npt tri": (
            copper(tilt=0.1),
            {"ensemble": "npt", "temperature": 300, "pressure": 1.0, "barostat": "tri"},
        ),
        "nph aniso": (
            crystal,
            {"ensemble": "nph", "pressure": 1.0, "barostat": "aniso"},
        ),
        "langevin": (
            crystal,
            {
                "ensemble": "nvt",
                "temperature": 400,
                "thermostat": "langevin",
                "seed": 5,
            },
        ),
        "langevin momentum": (
            crystal,
            {
                "ensemble": "nvt",
                "temperature": 400,
                "thermostat": "langevin",
                "seed": 5,
                "keep_momentum": True,
            },
        ),  # fmt: skip
        "nphug": (crystal, {"ensemble": "nphug", "pressure": 5.0, "barostat": "z"}),
        "msst": (
            crystal,
            {"ensemble": "msst", "barostat": "z", "shock_velocity": 60.0},
        ),
        "groups": (
            crystal,
            {
                "controls": [
                    Hold(bottom),
                    Move(top, (0.0, 0.0, 0.5)),
                    AddForce(middle, (0.01, 0.0, 0.0)),
                    Langevin(middle, 300.0, damping=0.1, seed=9),
                ]
            },
        ),  # fmt: skip
        "deform": (
            crystal,
            {"ensemble": "nvt", "temperature": 300, "controls": [Deform(xx=1e-3)]},
        ),  # fmt: skip
        "adaptive": (crystal, {"controls": [AdaptiveTimestep(0.02)]}),
        "stopping": (
            crystal,
            {
                "controls": [
                    ElectronicStopping(
                        {
                            "Cu": (
                                np.geomspace(0.01, 10.0, 20),
                                np.geomspace(0.1, 3.0, 20),
                            )
                        },
                        min_energy=0.02,
                        mask=middle,
                    )
                ]
            },
        ),  # fmt: skip
        "wall": (
            crystal.with_box(mdapy.Box(crystal.box.matrix, pbc=(True, True, False))),
            {"controls": [Wall("z", side="low", velocity=0.5)]},
        ),
        "indenter": (
            crystal,
            {
                "controls": [
                    Indenter(
                        "sphere",
                        center=(7.2, 7.2, 16.0),
                        radius=4.0,
                        velocity=(0.0, 0.0, -0.5),
                    )
                ]
            },
        ),  # fmt: skip
        "switch": (
            crystal,
            {
                "ensemble": "nvt",
                "temperature": 300,
                "thermostat": "langevin",
                "seed": 2,
                "controls": [Switch(springs)],
            },
        ),  # fmt: skip
    }


CASES = runs()


def continued(tmp_path, crystal, options, steps=60, stop=30):
    """The run that did not stop, and the same stopped after stop steps and
    finished from its restart file."""
    potential = EAM.zhou(["Cu"])
    options = {"thermo": 10, **options}
    with MolecularDynamics(crystal, potential, timestep=0.002) as md:
        md.velocities(300, seed=1)
        md.run(steps, **options)
        whole = (md.system, md.thermo)
    path = tmp_path / "md.restart"
    with MolecularDynamics(crystal, potential, timestep=0.002) as md:
        md.velocities(300, seed=1)
        frames = md.iterate(
            steps, save_every=stop, restart=path, restart_every=stop, **options
        )
        for frame in frames:
            if frame.info["timestep"] >= 2 * stop:
                break
        frames.close()
    others = [c.target for c in options.get("controls", ()) if isinstance(c, Switch)]
    again = MolecularDynamics.load(path, potential, others=others)
    assert again.step == stop
    again.resume()
    return whole, (again.system, again.thermo)


@pytest.mark.parametrize("case", list(CASES))
def test_a_run_finished_from_its_restart_is_the_run(tmp_path, case):
    crystal, options = CASES[case]
    (a, ta), (b, tb) = continued(tmp_path, crystal, options)
    np.testing.assert_allclose(b["positions"], a["positions"], rtol=0, atol=1e-11)
    np.testing.assert_array_equal(b["image"], a["image"])  # the boxes crossed
    np.testing.assert_allclose(b["velocities"], a["velocities"], rtol=0, atol=1e-11)
    np.testing.assert_allclose(b.box.matrix, a.box.matrix, rtol=0, atol=1e-11)
    assert tb.columns == ta.columns and tb.height == ta.height
    for name in ta.columns:
        x, y = np.stack(ta[name].to_numpy()), np.stack(tb[name].to_numpy())
        np.testing.assert_allclose(y, x, rtol=1e-10, atol=1e-10, err_msg=name)


def test_between_runs(tmp_path):
    crystal, potential = copper(), EAM.zhou(["Cu"])
    with MolecularDynamics(crystal, potential, timestep=0.002) as md:
        md.velocities(300, seed=1)
        md.run(40, ensemble="nvt", temperature=300, thermostat="langevin", seed=4)
        md.run(40, ensemble="npt", temperature=300, pressure=0.5)
        whole = md.system
    path = tmp_path / "between.restart"
    with MolecularDynamics(crystal, potential, timestep=0.002) as md:
        md.velocities(300, seed=1)
        md.run(40, ensemble="nvt", temperature=300, thermostat="langevin", seed=4)
        md.save(path)
    again = MolecularDynamics.load(path, potential)
    with pytest.raises(InvalidParameter, match="no run to resume"):
        again.resume()
    again.run(40, ensemble="npt", temperature=300, pressure=0.5)
    np.testing.assert_allclose(
        again.system["positions"], whole["positions"], atol=1e-11
    )
    # a run's own restart file at its end is written between runs
    with MolecularDynamics(crystal, potential, timestep=0.002) as md:
        md.velocities(300, seed=1)
        md.run(40, ensemble="nvt", temperature=300, thermostat="langevin", seed=4,
               restart=path)  # fmt: skip
    again = MolecularDynamics.load(path, potential)
    assert again.step == 40
    again.run(40, ensemble="npt", temperature=300, pressure=0.5)
    np.testing.assert_allclose(
        again.system["positions"], whole["positions"], atol=1e-11
    )


def test_the_log_goes_on(tmp_path):
    crystal, potential = copper(3), EAM.zhou(["Cu"])
    log, path = tmp_path / "md.log", tmp_path / "md.restart"
    with MolecularDynamics(crystal, potential, timestep=0.002, log=log) as md:
        md.velocities(300, seed=1)
        frames = md.iterate(
            40, save_every=20, thermo=10, restart=path, restart_every=20
        )
        for frame in frames:
            if frame.info["timestep"] == 40:
                break
        frames.close()
    again = MolecularDynamics.load(path, potential, log=log)
    again.resume()
    again.close()
    text = log.read_text()
    assert "continued from" in text and "run 1 resumed after 20 of 40 steps" in text
    assert text.index("molecular dynamics") < text.index("continued from")
    rows = np.loadtxt(log)
    assert rows[-1, 0] == 40 and set(rows[:, 0]) == {0, 10, 20, 30, 40}


def test_the_file_is_whole_or_the_last(tmp_path):
    crystal, potential = copper(3), EAM.zhou(["Cu"])
    path = tmp_path / "md.restart"
    with MolecularDynamics(crystal, potential, timestep=0.002) as md:
        md.velocities(300, seed=1)
        md.run(20, restart=path, restart_every=10)
        md.save(path)
    assert sorted(os.listdir(tmp_path)) == ["md.restart"]  # no half-written file left
    assert path.read_bytes().startswith(b"MDAPY restart\n")


def test_refused_files(tmp_path):
    crystal, potential = copper(3), EAM.zhou(["Cu"])
    path = tmp_path / "md.restart"
    (tmp_path / "other").write_bytes(b"not a restart")
    with pytest.raises(InvalidParameter, match="not a restart file of MDAPY"):
        MolecularDynamics.load(tmp_path / "other", potential)
    with pytest.raises(InvalidParameter, match="cannot read the restart file"):
        MolecularDynamics.load(tmp_path / "missing", potential)
    with MolecularDynamics(crystal, potential, timestep=0.002) as md:
        md.save(path)
    # a file of another version
    data = path.read_bytes()
    at = len(_restart._MAGIC)
    size = int.from_bytes(data[at : at + 8], "little")
    head = pickle.loads(data[at + 8 : at + 8 + size])
    head["version"] = "1.0.0"
    other = pickle.dumps(head, 5)
    old = tmp_path / "old.restart"
    old.write_bytes(
        _restart._MAGIC
        + len(other).to_bytes(8, "little")
        + other
        + data[at + 8 + size :]
    )
    with pytest.raises(InvalidParameter, match="written by MDAPY 1.0.0"):
        MolecularDynamics.load(old, potential)
    # a potential of another kind
    from mdapy.calculators import LennardJones

    with pytest.raises(InvalidParameter, match="ran with EAM"):
        MolecularDynamics.load(path, LennardJones({"Cu": (0.4, 2.3)}, 5.0))
    # a pickle that would call something else
    evil = io.BytesIO()
    pickle.dump(os.system, evil)
    with pytest.raises(InvalidParameter, match="holds no"):
        _restart._Reader(io.BytesIO(evil.getvalue()), []).load()


def test_a_switch_needs_its_potential(tmp_path):
    crystal, potential = copper(3), EAM.zhou(["Cu"])
    springs = Springs(crystal["positions"], np.full(len(crystal), 2.0))
    path = tmp_path / "md.restart"
    with MolecularDynamics(crystal, potential, timestep=0.002) as md:
        md.velocities(300, seed=1)
        frames = md.iterate(40, save_every=20, controls=[Switch(springs)], restart=path,
                            restart_every=20)  # fmt: skip
        for frame in frames:
            if frame.info["timestep"] == 40:
                break
        frames.close()
    with pytest.raises(InvalidParameter, match="needs 2 calculators"):
        MolecularDynamics.load(path, potential)


def test_restart_options():
    crystal, potential = copper(3), EAM.zhou(["Cu"])
    with MolecularDynamics(crystal, potential, timestep=0.002) as md:
        with pytest.raises(InvalidParameter, match="restart_every needs restart"):
            md.run(10, restart_every=5)
        for bad in (0, 2.5, True):
            with pytest.raises(InvalidParameter, match="restart_every must be"):
                md.run(10, restart="x", restart_every=bad)
        with pytest.raises(InvalidParameter, match="restart must be a path"):
            md.run(10, restart=3)
        with pytest.raises(InvalidParameter, match="cannot write the restart file"):
            md.save("/nonexistent-folder/md.restart")
