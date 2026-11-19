"""Molecular dynamics on the GPU against MDAPY's own dynamics on the
processor, the same equations: in double precision the same trajectories to
rounding, in every ensemble; in mixed precision the energy kept as well.
Skipped without the GPU part of MDAPY or a device."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM, LennardJones
from mdapy.errors import CalculationFailed, InvalidParameter
from mdapy.simulation import MolecularDynamics

cuda = pytest.importorskip("mdapy._cuda")
try:
    cuda.device_name()
except RuntimeError:
    pytest.skip("no CUDA device", allow_module_level=True)


def crystal(repeat=6, tilt=0.0, seed=0):
    system = build_crystal("fcc", 3.6, elements="Cu", repeat=(repeat,) * 3)
    matrix = system.box.matrix.copy()
    matrix[1, 0] += tilt * matrix[0, 0]
    system = system.with_box(mdapy.Box(matrix, pbc=True), scale_positions=True)
    rng = np.random.default_rng(seed)
    kinds = rng.choice(["Cu", "Ni"], len(system))
    return system.with_columns(element=list(kinds))


def run(system, potential, steps, **options):
    with MolecularDynamics(system, potential, timestep=0.002) as md:
        md.velocities(600, seed=1)
        md.run(steps, thermo=steps // 4, **options)
        return md.system, md.thermo


RUNS = [
    {"ensemble": "nve"},
    {"ensemble": "nvt", "temperature": (300, 500)},
    {"ensemble": "npt", "temperature": 300, "pressure": 1.0},
    {
        "ensemble": "npt",
        "temperature": 300,
        "pressure": (0.0, 2.0),
        "barostat": "aniso",
    },
    {"ensemble": "nph", "pressure": 0.5, "barostat": "tri"},
]


@pytest.mark.parametrize("options", RUNS)
def test_the_runs_of_the_processor_in_double_precision(options):
    system = crystal(tilt=0.2 if options.get("barostat") == "tri" else 0.0)
    cpu = EAM.zhou(["Cu", "Ni"])
    gpu = EAM.zhou(["Cu", "Ni"], device="cuda", precision="double")
    a, ta = run(system, gpu, 200, **options)
    b, tb = run(system, cpu, 200, **options)
    np.testing.assert_allclose(a.box.matrix, b.box.matrix, rtol=0, atol=1e-8)
    d = a["positions"] - b["positions"]
    d -= np.round(d @ np.linalg.inv(b.box.matrix)) @ b.box.matrix
    assert np.abs(d).max() < 1e-7
    np.testing.assert_allclose(a["velocities"], b["velocities"], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(a["image"], b["image"])
    for column in ("temperature", "potential_energy", "pressure", "volume"):
        np.testing.assert_allclose(ta[column], tb[column], rtol=1e-8, atol=1e-8)


def test_stillinger_weber_runs_as_on_the_processor(tmp_path):
    """The terms of each atom kept with it, which leaves the totals."""
    from mdapy.calculators import StillingerWeber

    from calculators.potential_files import write_sw

    path = write_sw(tmp_path / "SiGe.sw", ("Si", "Ge"))
    system = build_crystal("diamond", 5.43, elements="Si", repeat=(4, 4, 4))
    rng = np.random.default_rng(0)
    system = system.with_columns(element=list(rng.choice(["Si", "Ge"], len(system))))
    options = {"ensemble": "npt", "temperature": 600, "pressure": 0.0}
    a, ta = run(
        system, StillingerWeber(path, device="cuda", precision="double"), 200, **options
    )
    b, tb = run(system, StillingerWeber(path), 200, **options)
    np.testing.assert_allclose(a.box.matrix, b.box.matrix, atol=1e-8)
    d = a["positions"] - b["positions"]
    d -= np.round(d @ np.linalg.inv(b.box.matrix)) @ b.box.matrix
    assert np.abs(d).max() < 1e-7
    for column in ("temperature", "potential_energy", "pressure"):
        np.testing.assert_allclose(ta[column], tb[column], rtol=1e-8, atol=1e-8)


def test_tersoff_runs_as_on_the_processor(tmp_path):
    """The terms of each atom kept with it, which leaves the totals."""
    from mdapy.calculators import Tersoff

    from calculators.potential_files import write_tersoff

    path = write_tersoff(tmp_path / "SiC.tersoff", ("Si", "C"))
    system = build_crystal("diamond", 5.43, elements="Si", repeat=(4, 4, 4))
    rng = np.random.default_rng(0)
    system = system.with_columns(
        element=list(rng.choice(["Si", "C"], len(system), p=[0.8, 0.2]))
    )
    options = {"ensemble": "npt", "temperature": 600, "pressure": 0.0}
    a, ta = run(
        system, Tersoff(path, device="cuda", precision="double"), 100, **options
    )
    b, tb = run(system, Tersoff(path), 100, **options)
    np.testing.assert_allclose(a.box.matrix, b.box.matrix, atol=1e-8)
    d = a["positions"] - b["positions"]
    d -= np.round(d @ np.linalg.inv(b.box.matrix)) @ b.box.matrix
    assert np.abs(d).max() < 1e-7
    for column in ("temperature", "potential_energy", "pressure"):
        np.testing.assert_allclose(ta[column], tb[column], rtol=1e-8, atol=1e-8)


def test_mixed_precision_keeps_the_energy():
    system = crystal(repeat=8)
    gpu = EAM.zhou(["Cu", "Ni"], device="cuda")
    with MolecularDynamics(system, gpu, timestep=0.002) as md:
        md.velocities(600, seed=1)
        md.run(5000, thermo=500)
        total = md.thermo["total_energy"].to_numpy()
    # eV per atom over 10 ps, after the first record: from the perfect
    # crystal, where no force acts, the error of the time step moves the
    # energy by 5e-5 in both precisions
    assert np.ptp(total[1:]) / len(system) < 1e-5
    assert abs(total[0] - total[1:].mean()) / len(system) < 1e-4
    cpu = EAM.zhou(["Cu", "Ni"])
    a, _ = run(system, gpu, 40)
    b, _ = run(system, cpu, 40)
    d = a["positions"] - b["positions"]
    d -= np.round(d @ np.linalg.inv(b.box.matrix)) @ b.box.matrix
    assert np.abs(d).max() < 1e-4


def test_langevin_dynamics_holds_the_temperature_and_repeats():
    system = crystal()
    gpu = EAM.zhou(["Cu", "Ni"], device="cuda")
    options = {
        "ensemble": "nvt",
        "temperature": 400,
        "thermostat": "langevin",
        "seed": 7,
    }
    with MolecularDynamics(system, gpu, timestep=0.002) as md:
        md.velocities(400, seed=1)
        md.run(2000, thermo=10, **options)
        temperature = md.thermo["temperature"].to_numpy()[50:]
        first = md.system
    assert temperature.mean() == pytest.approx(400, rel=0.03)
    second, _ = run(system, gpu, 2000, **options)
    with MolecularDynamics(system, gpu, timestep=0.002) as md:
        md.velocities(400, seed=1)
        md.run(2000, thermo=10, **options)
        np.testing.assert_array_equal(md.system["positions"], first["positions"])


def test_frames_of_iterate_and_a_cluster():
    """iterate on the GPU, and atoms in a box periodic in no direction."""
    system = crystal()
    gpu = EAM.zhou(["Cu", "Ni"], device="cuda", precision="double")
    with MolecularDynamics(system, gpu, timestep=0.002) as md:
        md.velocities(300, seed=1)
        frames = list(md.iterate(100, save_every=50, ensemble="nvt", temperature=300))
    assert [f.info["timestep"] for f in frames] == [0, 50, 100]
    cluster = crystal(repeat=4).with_box(
        mdapy.Box(40.0, origin=(-12.0, -12.0, -12.0), pbc=False)
    )
    a, _ = run(cluster, gpu, 100, ensemble="nvt", temperature=300)
    b, _ = run(cluster, EAM.zhou(["Cu", "Ni"]), 100, ensemble="nvt", temperature=300)
    assert np.abs(a["positions"] - b["positions"]).max() < 1e-7


def test_errors():
    gpu = EAM.zhou(["Cu", "Ni"], device="cuda")
    with pytest.raises(InvalidParameter, match="thicker than twice the cutoff"):
        run(crystal(repeat=3), gpu, 10)
    held = crystal().with_columns(movable=np.zeros((864, 3), dtype=bool))
    with pytest.raises(InvalidParameter, match="movable column"):
        MolecularDynamics(held, gpu)
    # two atoms on one site
    lj = LennardJones({"Cu": (0.4, 2.3), "Ni": (0.5, 2.2)}, 5.0, device="cuda")
    system = crystal()
    positions = system["positions"].copy()
    positions[1] = positions[0]
    with pytest.raises(CalculationFailed, match="not finite"):
        run(system.with_columns(positions=positions), lj, 20)
