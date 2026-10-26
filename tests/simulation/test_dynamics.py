# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Molecular dynamics against the same runs written as LAMMPS input.
Skipped where LAMMPS is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy import units
from mdapy.build import build_crystal
from mdapy.calculators import EAM, LAMMPS
from mdapy.errors import Cancelled, InvalidParameter, MissingProperty
from mdapy.simulation import MolecularDynamics

from calculators.eam_files import write_setfl

lammps = pytest.importorskip("lammps")


@pytest.fixture(scope="module")
def setfl(tmp_path_factory):
    return write_setfl(tmp_path_factory.mktemp("eam") / "CuNi.eam.alloy")


@pytest.fixture(scope="module")
def potential(setfl):
    return LAMMPS(f"pair_style eam/alloy\npair_coeff * * {setfl} Cu Ni", ["Cu", "Ni"])


def crystal(seed=0):
    system = build_crystal("fcc", 3.4, elements="Cu", repeat=(3, 3, 3))
    rng = np.random.default_rng(seed)
    kinds = rng.choice(["Cu", "Ni"], len(system))
    return system.with_columns(
        element=kinds,
        positions=system["positions"] + rng.normal(0, 0.05, (len(system), 3)),
    )


def plain_lammps(tmp_path, system, setfl, commands):
    """Final positions and velocities (by id) of a run written as LAMMPS input."""
    data = tmp_path / "start.data"
    mdapy.write(data, system, atom_style="atomic", elements=["Cu", "Ni"])
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_string(
            f"""
            units metal
            atom_style atomic
            atom_modify map array
            read_data {data}
            pair_style eam/alloy
            pair_coeff * * {setfl} Cu Ni
            {commands}
            """
        )
        n = len(system)
        x = np.ctypeslib.as_array(lmp.gather_atoms("x", 1, 3)).reshape(n, 3).copy()
        v = np.ctypeslib.as_array(lmp.gather_atoms("v", 1, 3)).reshape(n, 3).copy()
        return x, v
    finally:
        lmp.close()


def test_the_same_run_as_lammps(tmp_path, setfl, potential):
    system = crystal()
    commands = """
        timestep 0.002
        velocity all create 600.0 7 dist gaussian mom yes rot no loop all
        fix 1 all nvt temp 300.0 500.0 0.2
        run 300
        unfix 1
        fix 1 all nve
        run 100
    """
    x, v = plain_lammps(tmp_path, system, setfl, commands)
    with MolecularDynamics(system, potential, timestep=0.002) as md:
        md.velocities(600, seed=7)
        md.run(300, ensemble="nvt", temperature=(300, 500), thermo=70)
        md.run(100, thermo=30)
        final = md.system
        assert (
            md.step == 400
            and repr(md) == f"MolecularDynamics({len(system)} atoms, step 400)"
        )
    # both wrapped into the box, which may differ in the last digit
    lengths = np.diag(final.box.matrix)
    apart = final["positions"] - x
    np.testing.assert_allclose(
        apart - np.round(apart / lengths) * lengths, 0, atol=1e-9
    )
    np.testing.assert_allclose(final["velocities"], v, atol=1e-9)
    assert final.info == {"timestep": 400, "time": pytest.approx(0.8)}
    assert repr(md) == f"MolecularDynamics({len(system)} atoms, closed)"


def test_thermo_and_frames(potential):
    system = crystal()
    calls = []
    with MolecularDynamics(system, potential) as md:
        md.velocities(300, seed=1)
        md.run(
            250,
            ensemble="nvt",
            temperature=300,
            thermo=100,
            frames=125,
            progress=lambda done, total: calls.append((done, total)),
        )
        md.run(100, ensemble="nve", thermo=100)
        thermo, frames = md.thermo, md.frames
    assert thermo["step"].to_list() == [0, 100, 200, 250, 350]
    assert thermo.columns == [
        "step", "time", "temperature", "potential_energy", "kinetic_energy",
        "total_energy", "pressure", "volume", "stress",
    ]  # fmt: skip
    assert thermo["time"].to_list() == pytest.approx([0, 0.1, 0.2, 0.25, 0.35])
    assert thermo["temperature"][0] == pytest.approx(300, rel=1e-9)
    np.testing.assert_allclose(
        thermo["total_energy"], thermo["potential_energy"] + thermo["kinetic_energy"]
    )
    # the pressure is minus a third of the trace of the stress
    stress = np.stack(thermo["stress"].to_numpy())
    np.testing.assert_allclose(
        thermo["pressure"], -stress[:, :3].mean(axis=1), rtol=1e-9
    )
    assert [f.info["timestep"] for f in frames] == [0, 125, 250]
    assert calls[-1] == (250, 250) and [c[0] for c in calls] == sorted(
        c[0] for c in calls
    )


def test_the_pressure_of_the_potential(potential):
    # at rest, the stress of LAMMPS is the virial of mdapy's EAM
    system = crystal()
    eam = EAM(potential.pair.split()[-3])
    expected = eam.compute(system, ("virial",)).stress(system)
    with MolecularDynamics(system, potential) as md:
        md.run(0)
        stress = np.array(md.thermo["stress"][0])
    voigt = [
        expected[a, b] for a, b in ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    ]
    np.testing.assert_allclose(stress, voigt, atol=1e-6)


@pytest.mark.parametrize(
    "matrix",
    [
        [[3.1, 4.2, 0], [-4.0, 3.0, 0.5], [0.3, -0.2, 7.0]],  # rotated
        [[10.2, 0, 0], [0, 0, 10.2], [0, 10.2, 0]],  # left-handed
    ],
)
def test_any_orientation_of_the_box(potential, matrix):
    # the same atoms in a box rotated, or with two vectors swapped, run alike
    system = crystal()
    rotation = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))[0]
    rotation *= np.sign(np.linalg.det(rotation))
    matrix = np.asarray(matrix, dtype=float)
    if np.linalg.det(matrix) < 0:  # the same box, its vectors in another order
        turned = system.with_box(mdapy.Box(system.box.matrix[[0, 2, 1]]))
        rotation = np.eye(3)
    else:
        turned = system.with_box(
            mdapy.Box(
                system.box.matrix @ rotation.T, origin=system.box.origin @ rotation.T
            )
        ).with_columns(positions=system["positions"] @ rotation.T)
    results = []
    for start in (system, turned):
        with MolecularDynamics(start, potential, timestep=0.002) as md:
            md.velocities(400, seed=5)
            md.run(100, ensemble="npt", temperature=400, pressure=1.0, barostat="tri")
            results.append(md.system)
    ours, theirs = results
    np.testing.assert_allclose(
        theirs["positions"] + theirs["image"] @ theirs.box.matrix,
        (ours["positions"] + ours["image"] @ ours.box.matrix) @ rotation.T,
        atol=1e-6,
    )
    assert theirs.box.volume == pytest.approx(ours.box.volume, rel=1e-9)


def test_velocities_of_the_system(potential):
    # atoms of a perfect crystal at rest stay put; given velocities move them
    system = build_crystal("fcc", 3.3, elements="Cu", repeat=(3, 3, 3))
    with MolecularDynamics(system, potential) as md:
        md.run(20)
        np.testing.assert_allclose(
            md.system["positions"], system["positions"], atol=1e-12
        )
    moving = system.with_columns(velocities=np.tile([1.0, 0.0, 0.0], (len(system), 1)))
    with MolecularDynamics(moving, potential) as md:
        md.run(20)
        shift = md.system["positions"] + md.system["image"] @ system.box.matrix
        np.testing.assert_allclose(
            shift - system["positions"],
            np.broadcast_to([0.02, 0, 0], (len(system), 3)),
            atol=1e-12,
        )


def test_a_cluster(potential):
    system = build_crystal("fcc", 3.4, elements="Cu", repeat=(2, 2, 2))
    cluster = system.with_box(mdapy.Box(20.0, pbc=False))
    with MolecularDynamics(cluster, potential) as md:
        md.velocities(100, seed=3)
        md.run(50, ensemble="nvt", temperature=100)
        assert not md.system.box.pbc.any()
        assert md.thermo["total_energy"].len() == 2


def test_cancel(potential):
    token = mdapy.CancelToken()
    with MolecularDynamics(crystal(), potential) as md:
        md.velocities(300, seed=1)
        with pytest.raises(Cancelled):
            md.run(
                1000,
                ensemble="nve",
                thermo=10,
                progress=lambda d, t: d >= 50 and token.cancel(),
                cancel=token,
            )
        assert 50 <= md.step < 1000
        md.run(10)  # the run's fix is gone: another run works


def test_commands_and_the_instance(potential):
    with MolecularDynamics(crystal(), potential) as md:
        md.command("variable n equal count(all)")
        assert md.lammps.extract_variable("n") == 108
        with pytest.raises(InvalidParameter, match="LAMMPS"):
            md.command("no_such_command")
    with pytest.raises(InvalidParameter, match="closed"):
        md.lammps


@pytest.mark.parametrize(
    "options, message",
    [
        ({"steps": -1}, "steps must not be negative"),
        ({"ensemble": "nph"}, "ensemble must be one of"),
        ({"ensemble": "nvt"}, "nvt needs a temperature"),
        ({"ensemble": "npt", "temperature": 300}, "npt needs a pressure"),
        ({"barostat": "z"}, "barostat must be one of"),
        ({"thermo": 0}, "thermo and frames must be positive"),
        ({"frames": 0}, "thermo and frames must be positive"),
        ({"ensemble": "nvt", "temperature": (1, 2, 3)}, "temperature must be"),
    ],
)
def test_invalid_runs(potential, options, message):
    with MolecularDynamics(crystal(), potential) as md:
        with pytest.raises(InvalidParameter, match=message):
            md.run(**({"steps": 10} | options))


def test_invalid_setups(potential):
    system = crystal()
    with pytest.raises(
        InvalidParameter, match="LAMMPS calculator|mdapy.calculators.LAMMPS"
    ):
        MolecularDynamics(system, object())
    with pytest.raises(InvalidParameter, match="timestep"):
        MolecularDynamics(system, potential, timestep=0)
    with pytest.raises(MissingProperty, match="element"):
        MolecularDynamics(system.drop("element"), potential)
    with pytest.raises(InvalidParameter, match="no atom type has the elements"):
        MolecularDynamics(system.with_columns(element="Fe"), potential)
    with MolecularDynamics(system, potential) as md:
        with pytest.raises(InvalidParameter, match="temperature must not be negative"):
            md.velocities(-1, seed=1)
        with pytest.raises(InvalidParameter, match="seed must be positive"):
            md.velocities(300, seed=0)
    cluster = system.with_box(mdapy.Box(30.0, pbc=False))
    with MolecularDynamics(cluster, potential) as md:
        with pytest.raises(InvalidParameter, match="periodic in every direction"):
            md.run(10, ensemble="npt", temperature=300, pressure=0, barostat="aniso")
    broken = LAMMPS("pair_style zero 3.0\npair_coeff * *", ["Cu"])
    object.__setattr__(broken, "pair", "pair_style nonsense")
    with pytest.raises(InvalidParameter, match="LAMMPS"):
        MolecularDynamics(system.with_columns(element="Cu"), broken)


def test_pressure_units(potential):
    # npt at 2 GPa compresses the crystal: its pressure averages near 2 GPa
    with MolecularDynamics(crystal(), potential, timestep=0.002) as md:
        md.velocities(200, seed=2)
        md.run(3000, ensemble="npt", temperature=100, pressure=2.0, thermo=50)
        late = md.thermo.filter(md.thermo["step"] > 1500)
    assert late["pressure"].mean() == pytest.approx(2.0, abs=0.3)
    assert units.BAR_TO_GPA == 1e-4
