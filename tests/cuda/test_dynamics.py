"""Molecular dynamics on the GPU against MDAPY's own dynamics on the
processor, the same equations: in double precision the same trajectories to
rounding, in every ensemble; in mixed precision the energy kept as well.
Skipped without the GPU part of MDAPY or a device."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM, LennardJones, _gpu
from mdapy.errors import CalculationFailed, InvalidParameter
from mdapy.simulation import MolecularDynamics

cuda = _gpu.module()
if cuda is None:
    pytest.skip("no GPU part of MDAPY", allow_module_level=True)
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


def test_meam_runs_as_on_the_processor(tmp_path):
    """Screening beyond rc, the shares of each centre kept with it."""
    from mdapy.calculators import MEAM

    from calculators.meam_files import CASES, write_library, write_parameters
    from calculators.test_meam import crystal as meam_crystal

    names, lines, structure, a = CASES["silicon and carbon, dia, screening beyond rc"]
    library = write_library(tmp_path / "library.meam")
    parameters = write_parameters(tmp_path / "x.meam", lines)
    cpu = MEAM(library, names, parameters)
    gpu = MEAM(library, names, parameters, device="cuda", precision="double")
    repeat = int(np.ceil((2 * (cpu.cutoff + 1.0) + 1.0) / a)) + 1
    system = meam_crystal(
        structure, a, cpu.elements, repeat=repeat, jitter=0.05, seed=3
    )
    options = {"ensemble": "npt", "temperature": 300, "pressure": 0.0}
    x, tx = run(system, gpu, 100, **options)
    y, ty = run(system, cpu, 100, **options)
    np.testing.assert_allclose(x.box.matrix, y.box.matrix, atol=1e-8)
    d = x["positions"] - y["positions"]
    d -= np.round(d @ np.linalg.inv(y.box.matrix)) @ y.box.matrix
    assert np.abs(d).max() < 1e-7
    for column in ("temperature", "potential_energy", "pressure"):
        np.testing.assert_allclose(tx[column], ty[column], rtol=1e-8, atol=1e-8)


def test_nep_runs_as_on_the_processor(tmp_path):
    """The derivatives of each atom's energy, its virial from its neighbors'."""
    from mdapy.calculators import NEP

    from calculators.nep_files import write_nep

    path = write_nep(tmp_path / "nep.txt", zbl="universal")
    system = crystal(repeat=4)
    options = {"ensemble": "nvt", "temperature": 600}
    x, tx = run(system, NEP(path, device="cuda", precision="double"), 100, **options)
    y, ty = run(system, NEP(path), 100, **options)
    d = x["positions"] - y["positions"]
    d -= np.round(d @ np.linalg.inv(y.box.matrix)) @ y.box.matrix
    assert np.abs(d).max() < 1e-7
    for column in ("temperature", "potential_energy", "pressure"):
        np.testing.assert_allclose(tx[column], ty[column], rtol=1e-8, atol=1e-8)


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


def held(system):
    """The crystal turned, so that the frame of the dynamics is not the
    system's, with atoms held along all, one or two of its axes."""
    c, s = np.cos(0.3), np.sin(0.3)
    q = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.array(
        [[1, 0, 0], [0, c, -s], [0, s, c]]
    )
    turned = system.with_box(
        mdapy.Box(system.box.matrix @ q.T, pbc=True), scale_positions=True
    )
    movable = np.ones((len(system), 3), dtype=bool)
    movable[:100] = False
    movable[100:300, 0] = False
    movable[300:400, 1:] = False
    return turned.with_columns(movable=movable)


@pytest.mark.parametrize(
    "options",
    [
        {"ensemble": "nvt", "temperature": 300},
        {"ensemble": "npt", "temperature": 300, "pressure": 1.0, "barostat": "tri"},
    ],
)
def test_held_atoms_as_on_the_processor(options):
    system = held(crystal(tilt=0.2))
    gpu = EAM.zhou(["Cu", "Ni"], device="cuda", precision="double")
    a, ta = run(system, gpu, 200, **options)
    b, tb = run(system, EAM.zhou(["Cu", "Ni"]), 200, **options)
    np.testing.assert_allclose(a.box.matrix, b.box.matrix, rtol=0, atol=1e-8)
    # held atoms on a face of the turned box may wrap to either face
    d = a["positions"] - b["positions"]
    d -= np.round(d @ np.linalg.inv(a.box.matrix)) @ a.box.matrix
    assert np.abs(d).max() < 1e-7
    np.testing.assert_allclose(a["velocities"], b["velocities"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(ta["temperature"], tb["temperature"], rtol=1e-8)
    assert np.abs(a["velocities"][~system["movable"]]).max() < 1e-12


@pytest.mark.parametrize("precision", ["mixed", "double"])
def test_held_atoms_stay(precision):
    """Held atoms keep still along their held directions, also under a
    Langevin thermostat, whose noise the GPU draws."""
    system = held(crystal())
    gpu = EAM.zhou(["Cu", "Ni"], device="cuda", precision=precision)
    options = {"ensemble": "nvt", "temperature": 300, "thermostat": "langevin"}
    with MolecularDynamics(system, gpu, timestep=0.002) as md:
        md.velocities(600, seed=1)
        md.run(500, seed=2, **options)
        a = md.system
    still = ~system["movable"]
    assert np.abs(a["velocities"][still]).max() < 1e-12
    unwrapped = a["positions"] + a["image"] @ a.box.matrix
    np.testing.assert_allclose(unwrapped[still], system["positions"][still], atol=1e-9)
    assert np.abs(a["velocities"][~still]).max() > 1.0


def test_errors():
    gpu = EAM.zhou(["Cu", "Ni"], device="cuda")
    with pytest.raises(InvalidParameter, match="thicker than twice the cutoff"):
        run(crystal(repeat=3), gpu, 10)
    # two atoms on one site
    lj = LennardJones({"Cu": (0.4, 2.3), "Ni": (0.5, 2.2)}, 5.0, device="cuda")
    system = crystal()
    positions = system["positions"].copy()
    positions[1] = positions[0]
    with pytest.raises(CalculationFailed, match="not finite"):
        run(system.with_columns(positions=positions), lj, 20)


def walls(seed=0):
    """A shaken crystal with its bottom, top and middle layers."""
    system = crystal(seed=seed)
    rng = np.random.default_rng(seed)
    system = system.with_columns(
        positions=system["positions"] + rng.normal(0, 0.03, (len(system), 3))
    )
    z = system.box.to_fractional(system["positions"])[:, 2]
    return system, z < 0.2, z > 0.8, (z > 0.45) & (z < 0.55)


def test_controls_as_on_the_processor():
    from mdapy.simulation.controls import AddForce, Hold, Move

    system, bottom, top, middle = walls()
    controls = [
        Hold(bottom, directions="xz"),
        Move(top, (0.4, -0.2, 0.1), name="top"),
        AddForce(middle, (0.0, 0.05, 0.02)),
    ]
    out = []
    for potential in (
        EAM.zhou(["Cu", "Ni"], device="cuda", precision="double"),
        EAM.zhou(["Cu", "Ni"]),
    ):
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.velocities(300, seed=1)
            md.run(200, ensemble="nvt", temperature=300, thermo=50, controls=controls)
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    d = a["positions"] - b["positions"]
    d -= np.round(d @ np.linalg.inv(a.box.matrix)) @ a.box.matrix
    assert np.abs(d).max() < 1e-7
    np.testing.assert_allclose(a["velocities"], b["velocities"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(ta["temperature"], tb["temperature"], rtol=1e-8)
    np.testing.assert_allclose(
        np.stack(ta["top_force"]), np.stack(tb["top_force"]), rtol=0, atol=1e-6
    )


@pytest.mark.parametrize("precision", ["mixed", "double"])
def test_heat_baths(precision):
    """A hot and a cold bath: the energy changes by the heat they gave, and a
    seed gives the same run."""
    from mdapy.simulation.controls import Hold, Langevin

    system, bottom, top, middle = walls()
    potential = EAM.zhou(["Cu", "Ni"], device="cuda", precision=precision)
    runs = []
    for _ in range(2):
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.velocities(300, seed=1)
            md.run(
                2000,
                thermo=100,
                controls=[
                    Langevin(bottom, 600.0, damping=0.1, seed=3, name="hot"),
                    Langevin(top, 100.0, damping=0.1, seed=4, name="cold"),
                    Hold(middle, directions="z"),
                ],
            )
            runs.append((md.thermo, md.system))
    thermo, end = runs[0]
    energy = thermo["total_energy"].to_numpy()
    heat = thermo["hot_energy"].to_numpy() + thermo["cold_energy"].to_numpy()
    assert thermo["hot_energy"][-1] > 5.0 and thermo["cold_energy"][-1] < -5.0
    assert np.abs(energy - energy[0] - heat).max() < 0.05
    np.testing.assert_array_equal(runs[1][1]["velocities"], end["velocities"])
    assert np.abs(end["velocities"][middle, 2]).max() < 1e-12


def test_deformation_as_on_the_processor():
    """A box stretched, compressed and sheared on the GPU, then with a
    barostat on the lengths left: the runs of the processor."""
    from mdapy.simulation.controls import Deform

    system, *_ = walls()
    out = []
    for potential in (
        EAM.zhou(["Cu", "Ni"], device="cuda", precision="double"),
        EAM.zhou(["Cu", "Ni"]),
    ):
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.velocities(300, seed=1)
            md.run(
                200,
                ensemble="nvt",
                temperature=300,
                thermo=50,
                controls=Deform(xx=0.2, zz=-0.1, xy=0.3),
            )
            md.run(
                100,
                ensemble="npt",
                temperature=300,
                pressure=0.0,
                barostat="aniso",
                controls=Deform(yy=0.2, kind="true"),
            )
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    np.testing.assert_allclose(a.box.matrix, b.box.matrix, rtol=0, atol=1e-9)
    d = a["positions"] - b["positions"]
    d -= np.round(d @ np.linalg.inv(a.box.matrix)) @ a.box.matrix
    assert np.abs(d).max() < 1e-7
    np.testing.assert_allclose(a["velocities"], b["velocities"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(ta["temperature"], tb["temperature"], rtol=1e-8)
    np.testing.assert_allclose(
        np.stack(ta["stress"]), np.stack(tb["stress"]), rtol=0, atol=1e-6
    )


def test_adaptive_timestep_as_on_the_processor():
    """The time step of a fast atom on the GPU: the steps and the runs of
    the processor."""
    from mdapy.simulation.controls import AdaptiveTimestep

    system = crystal()
    velocities = np.zeros((len(system), 3))
    velocities[7] = (300.0, -150.0, 80.0)
    system = system.with_columns(velocities=velocities)
    out = []
    for potential in (
        EAM.zhou(["Cu", "Ni"], device="cuda", precision="double"),
        EAM.zhou(["Cu", "Ni"]),
    ):
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.run(200, thermo=20, controls=AdaptiveTimestep(0.04, every=2))
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    np.testing.assert_allclose(ta["time"], tb["time"], rtol=1e-9)
    assert ta["time"][-1] < 0.3  # shorter steps than 0.002 ps
    d = a["positions"] - b["positions"]
    d -= np.round(d @ np.linalg.inv(a.box.matrix)) @ a.box.matrix
    assert np.abs(d).max() < 1e-7
    np.testing.assert_allclose(a["velocities"], b["velocities"], rtol=0, atol=1e-6)


@pytest.mark.parametrize("target", ["ufm", "springs", "gas"])
def test_a_switch_as_on_the_processor(target):
    """A Switch on the GPU to the Uhlenbeck-Ford model (one kind of atom, its
    cutoff the longer), to springs and to nothing: the runs and the
    differences of the energies of the processor."""
    from mdapy.calculators import Springs, UhlenbeckFord
    from mdapy.simulation.controls import Switch

    system = crystal()
    out = []
    for device in ("cuda", "cpu"):
        if target == "ufm":
            other = UhlenbeckFord(
                (0.5, 1.5),
                7.0,
                device=device,
                **({"precision": "double"} if device == "cuda" else {}),
            )
        elif target == "springs":
            other = Springs(system["positions"], 1.5)
        else:
            other = None
        options = {"precision": "double"} if device == "cuda" else {}
        potential = EAM.zhou(["Cu", "Ni"], device=device, **options)
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.velocities(300.0, seed=1)
            md.run(200, thermo=50, controls=Switch(other, name="s"))
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    d = a["positions"] - b["positions"]
    d -= np.round(d @ np.linalg.inv(a.box.matrix)) @ a.box.matrix
    assert np.abs(d).max() < 1e-7
    np.testing.assert_allclose(a["velocities"], b["velocities"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(ta["s_difference"], tb["s_difference"], rtol=1e-8)
    np.testing.assert_allclose(
        ta["potential_energy"], tb["potential_energy"], rtol=1e-9
    )


@pytest.mark.parametrize("precision", ["mixed", "double"])
def test_the_kept_momentum(precision):
    """The noise keeps the momentum: to the rounding in double precision; in
    mixed precision the forces of single precision move it a little, far less
    than the noise does without it."""
    system = crystal()
    potential = EAM.zhou(["Cu", "Ni"], device="cuda", precision=precision)
    momenta = []
    for keep in (True, False):
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.velocities(300.0, seed=1)
            md.run(
                500,
                ensemble="nvt",
                temperature=300.0,
                thermostat="langevin",
                keep_momentum=keep,
                seed=3,
            )
            end = md.system
        masses = mdapy.elements.atomic_masses(list(end["element"]))
        momenta.append(np.abs(masses @ end["velocities"]).max())
    kept, free = momenta
    assert kept < (1e-8 if precision == "double" else 1e-3 * free)


def test_the_free_energy_of_a_solid():
    """Frenkel-Ladd on the GPU: the solid of the processor, within the noise
    of short switchings."""
    from mdapy.properties.thermodynamics import free_energy

    system = build_crystal("fcc", 3.6, elements="Cu", repeat=(6, 6, 6))
    options = dict(
        pressure=None, timestep=0.002, equilibration=2.0, switching=10.0, seed=2
    )
    gpu = free_energy(
        system, EAM.zhou(["Cu"], device="cuda", precision="double"), 300.0, **options
    )
    cpu = free_energy(system, EAM.zhou(["Cu"]), 300.0, **options)
    assert gpu.free_energy[0] == pytest.approx(cpu.free_energy[0], abs=1e-3)
    assert abs(gpu.dissipation[0]) < 2e-3


def test_hnemd_as_on_the_processor():
    """The field of HNEMD on the GPU: the driven run and the heat flux of the
    processor in double precision; in mixed precision the work of the field
    is still the change of the energy."""
    from mdapy.simulation._controls import _HeatField

    system = crystal()
    field = np.array([0.01, -0.004, 0.002])
    out = []
    for potential in (
        EAM.zhou(["Cu", "Ni"], device="cuda", precision="double"),
        EAM.zhou(["Cu", "Ni"]),
    ):
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.velocities(300.0, seed=1)
            md.run(200, thermo=20, controls=_HeatField(field))
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    np.testing.assert_allclose(a["velocities"], b["velocities"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        np.stack(ta["heat_flux"]), np.stack(tb["heat_flux"]), rtol=1e-7, atol=1e-9
    )
    mixed = EAM.zhou(["Cu", "Ni"], device="cuda", precision="mixed")
    with MolecularDynamics(system, mixed, timestep=0.001) as md:
        md.velocities(300.0, seed=1)
        md.run(400, thermo=1, controls=_HeatField(field * 3))
        thermo = md.thermo
    energy = thermo["total_energy"].to_numpy()
    power = np.stack(thermo["heat_flux"].to_numpy()) @ (field * 3)
    work = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * 0.001)])
    assert abs((energy - energy[0] - work)[-1]) < 0.05 * abs(energy[-1] - energy[0])


def test_hnemd_of_stillinger_weber_as_on_the_processor(tmp_path):
    """The field on the GPU with Stillinger-Weber, whose energies and virials
    of the heat flux are those of compute_heat: the driven run and the heat
    flux of the processor in double precision; in mixed precision the work of
    the field is still the change of the energy."""
    from mdapy.calculators import StillingerWeber
    from mdapy.simulation._controls import _HeatField

    from calculators.potential_files import write_sw

    path = write_sw(tmp_path / "Si.sw", ("Si",))
    system = build_crystal("diamond", 5.43, elements="Si", repeat=5)
    field = np.array([0.05, -0.025, 0.0125])
    out = []
    for potential in (
        StillingerWeber(path, device="cuda", precision="double"),
        StillingerWeber(path),
    ):
        with MolecularDynamics(system, potential, timestep=0.0005) as md:
            md.velocities(600.0, seed=1)
            md.run(400, thermo=20, controls=_HeatField(field))
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    np.testing.assert_allclose(a["velocities"], b["velocities"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        np.stack(ta["heat_flux"]), np.stack(tb["heat_flux"]), rtol=1e-7, atol=1e-8
    )
    mixed = StillingerWeber(path, device="cuda", precision="mixed")
    with MolecularDynamics(system, mixed, timestep=0.0005) as md:
        md.velocities(600.0, seed=1)
        md.run(600, thermo=1, controls=_HeatField(field))
        thermo = md.thermo
    energy = thermo["total_energy"].to_numpy()
    power = np.stack(thermo["heat_flux"].to_numpy()) @ field
    work = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * 0.0005)])
    assert abs((energy - energy[0] - work)[-1]) < 0.05 * abs(energy[-1] - energy[0])


def test_hnemd_with_the_energies_carried_as_on_the_processor():
    """The field with the energies each atom carries, in a hot argon fluid:
    on the GPU in double precision the run and the heat flux of the processor;
    in mixed precision the work of the field is still the change of the
    energy."""
    from mdapy.simulation._controls import _HeatField

    crystal = build_crystal("fcc", 5.6, elements="Ar", repeat=6)
    with MolecularDynamics(
        crystal, LennardJones({"Ar": (0.0104, 3.405)}, 8.5, shift=True), timestep=0.002
    ) as md:
        md.velocities(300.0, seed=1)
        md.run(1000, ensemble="nvt", temperature=150.0)
        system = md.system
    field = np.array([0.02, -0.01, 0.005])
    out = []
    for options in ({"device": "cuda", "precision": "double"}, {}):
        potential = LennardJones({"Ar": (0.0104, 3.405)}, 8.5, shift=True, **options)
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.run(500, thermo=10, controls=_HeatField(field, convective=True))
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    np.testing.assert_allclose(a["velocities"], b["velocities"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        np.stack(ta["heat_flux"]), np.stack(tb["heat_flux"]), rtol=1e-7, atol=1e-7
    )
    mixed = LennardJones(
        {"Ar": (0.0104, 3.405)}, 8.5, shift=True, device="cuda", precision="mixed"
    )
    with MolecularDynamics(system, mixed, timestep=0.002) as md:
        md.run(1000, thermo=1, controls=_HeatField(field, convective=True))
        thermo = md.thermo
    energy = thermo["total_energy"].to_numpy()
    power = np.stack(thermo["heat_flux"].to_numpy()) @ field
    work = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * 0.002)])
    assert abs((energy - energy[0] - work)[-1]) < 0.02 * abs(energy[-1] - energy[0])


@pytest.mark.parametrize("precision", ["mixed", "double"])
def test_nemd(precision):
    """Two baths on the GPU drive heat through solid argon: its conductivity
    of the order of that of argon, the gradient set by the baths."""
    from mdapy.calculators import LennardJones
    from mdapy.properties.transport import nemd

    argon = LennardJones(
        {"Ar": (0.0104, 3.405)}, 8.5, shift=True, device="cuda", precision=precision
    )
    result = nemd(
        build_crystal("fcc", 5.28, elements="Ar", repeat=(4, 4, 16)),
        argon,
        40.0,
        difference=10.0,
        timestep=0.004,
        equilibration=2000,
        relaxation=10000,
        production=40000,
        every=50,
    )
    assert 0.2 < result.kappa < 2.0
    assert result.temperatures[10] > result.temperatures[0] + 5.0


@pytest.mark.parametrize("precision", ["mixed", "double"])
def test_walls(precision):
    """A crystal open along z thrown against a still wall, then pushed by a
    piston, on the GPU: the runs of the processor in double precision, the
    atoms kept on their side of the walls in both."""
    from mdapy.simulation.controls import Wall

    system = crystal()
    system = system.with_box(mdapy.Box(system.box.matrix, pbc=(True, True, False)))
    rng = np.random.default_rng(1)
    system = system.with_columns(
        velocities=rng.normal(0, 2.0, (len(system), 3)) + np.array([0.0, 0.0, -5.0])
    )
    top = float(system["positions"][:, 2].max())
    mirrors = [Wall(position=-0.5), Wall(side="high", position=top + 0.5)]
    out = []
    for potential in (
        EAM.zhou(["Cu", "Ni"], device="cuda", precision=precision),
        EAM.zhou(["Cu", "Ni"]),
    ):
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.run(200, thermo=50, controls=mirrors)
            z = md.system["positions"][:, 2]
            assert z.min() > -0.5 and z.max() < top + 0.5
            md.run(200, thermo=50, controls=Wall(position=-1.0, velocity=10.0))
            assert md.system["positions"][:, 2].min() > 3.0
            out.append((md.system, md.thermo))
    if precision == "mixed":
        return
    (a, ta), (b, tb) = out
    d = a["positions"] - b["positions"]
    d[:, :2] -= (
        np.round(d[:, :2] / np.diag(a.box.matrix)[:2]) * np.diag(a.box.matrix)[:2]
    )
    assert np.abs(d).max() < 1e-7
    np.testing.assert_allclose(a["velocities"], b["velocities"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(ta["temperature"], tb["temperature"], rtol=1e-8)


@pytest.mark.parametrize("precision", ["mixed", "double"])
def test_nphug(precision):
    """The Hugoniostat along z on the GPU: the run of the processor in double
    precision; in mixed precision the box compressed to the pressure."""
    system = crystal()
    out = []
    for potential in (
        EAM.zhou(["Cu", "Ni"], device="cuda", precision=precision),
        EAM.zhou(["Cu", "Ni"]),
    ):
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.velocities(300, seed=1)
            md.run(
                1000,
                ensemble="nphug",
                pressure=20.0,
                barostat="z",
                thermo=50,
                thermostat_damping=0.1,
                barostat_damping=0.5,
            )
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    late = -np.stack(ta["stress"].to_numpy())[10:, 2]
    assert late.mean() == pytest.approx(20.0, abs=1.5)
    assert a.box.matrix[2, 2] < 0.97 * system.box.matrix[2, 2]
    if precision == "mixed":
        return
    np.testing.assert_allclose(a.box.matrix, b.box.matrix, rtol=0, atol=1e-9)
    d = a["positions"] - b["positions"]
    d -= np.round(d @ np.linalg.inv(a.box.matrix)) @ a.box.matrix
    assert np.abs(d).max() < 1e-7
    np.testing.assert_allclose(a["velocities"], b["velocities"], rtol=0, atol=1e-6)
    for column in ("hugoniot", "shock_velocity", "particle_velocity"):
        np.testing.assert_allclose(ta[column], tb[column], rtol=1e-7, atol=1e-6)


@pytest.mark.parametrize("precision", ["mixed", "double"])
def test_msst(precision):
    """The multiscale shock technique on the GPU: the run of the processor
    in double precision; in mixed precision the length of the box along the
    shock that of the processor's run, its chaos aside."""
    system = crystal()
    out = []
    for potential in (
        EAM.zhou(["Cu", "Ni"], device="cuda", precision=precision),
        EAM.zhou(["Cu", "Ni"]),
    ):
        with MolecularDynamics(system, potential, timestep=0.001) as md:
            md.velocities(600, seed=1)
            md.run(
                1000,
                ensemble="msst",
                barostat="z",
                shock_velocity=50.0,
                msst={"mu": 1.0},
                thermo=50,
            )
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    assert abs(b.box.matrix[2, 2] - system.box.matrix[2, 2]) > 0.05  # it moves
    assert a.box.matrix[2, 2] == pytest.approx(b.box.matrix[2, 2], rel=0.02)
    if precision == "mixed":
        return
    np.testing.assert_allclose(a.box.matrix, b.box.matrix, rtol=0, atol=1e-9)
    d = a["positions"] - b["positions"]
    d -= np.round(d @ np.linalg.inv(a.box.matrix)) @ a.box.matrix
    assert np.abs(d).max() < 1e-7
    np.testing.assert_allclose(a["velocities"], b["velocities"], rtol=0, atol=1e-6)
    for column in ("hugoniot", "rayleigh", "particle_velocity", "msst_energy"):
        np.testing.assert_allclose(ta[column], tb[column], rtol=1e-7, atol=1e-6)


@pytest.mark.parametrize("precision", ["mixed", "double"])
def test_indenters(precision):
    """A sphere pressed into a crystal open along z, and a plane holding up
    its bottom, on the GPU: the run of the processor in double precision; in
    mixed precision the force on the sphere that of the processor's run, its
    chaos aside."""
    from mdapy.simulation.controls import Indenter

    system = crystal()
    system = system.with_box(mdapy.Box(system.box.matrix, pbc=(True, True, False)))
    top = float(system["positions"][:, 2].max())
    middle = system.box.matrix[0, 0] / 2
    controls = [
        Indenter(center=(middle, middle, top + 9.0), radius=10.0,
                 velocity=(0.0, 0.0, -2.0), name="tip"),
        Indenter("plane", axis="z", position=0.5, side="low", name="floor"),
    ]  # fmt: skip
    out = []
    for potential in (
        EAM.zhou(["Cu", "Ni"], device="cuda", precision=precision),
        EAM.zhou(["Cu", "Ni"]),
    ):
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.velocities(300, seed=1)
            md.run(500, thermo=50, controls=controls)
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    force = np.stack(tb["tip_force"].to_numpy())
    assert force[:, 2].max() > 1.0  # it presses
    np.testing.assert_allclose(
        np.stack(ta["tip_force"].to_numpy())[:, 2].max(), force[:, 2].max(), rtol=0.05
    )
    if precision == "mixed":
        return
    d = a["positions"] - b["positions"]
    d -= np.round(d @ np.linalg.inv(a.box.matrix)) * np.array([1, 1, 0]) @ a.box.matrix
    assert np.abs(d).max() < 1e-7
    np.testing.assert_allclose(a["velocities"], b["velocities"], rtol=0, atol=1e-6)
    for name in ("tip", "floor"):
        np.testing.assert_allclose(
            np.stack(ta[f"{name}_force"].to_numpy()),
            np.stack(tb[f"{name}_force"].to_numpy()),
            rtol=0,
            atol=1e-6,
        )


def test_the_noise_of_successive_steps_is_independent():
    """The random kicks of the Langevin thermostat, of its form that keeps
    the momentum and of a bath, at two successive steps: unit Gaussians, no
    component of one correlated with one of the other, as the canonical
    ensemble needs. A stream that moves on by fewer numbers than a step draws
    repeats a component of the step before in the next."""
    n = 50000
    atoms = cuda.Atoms(n)
    atoms.set_species(np.zeros(n, dtype=np.int32))
    atoms.set_positions(np.zeros((n, 3)))
    atoms.set_masses(np.ones(n))
    atoms.set_groups(np.ones(n, dtype=np.int32))

    def kicks(draw):
        out = []
        for counter in (0, 1):
            atoms.set_velocities(np.zeros((n, 3)))
            draw(counter)
            out.append(atoms.velocities())
        return out

    for draw in (
        lambda counter: atoms.langevin(0.0, 1.0, 1.0, 7, counter),
        lambda counter: atoms.langevin(0.0, 1.0, 1.0, 7, counter, True),
        lambda counter: atoms.bath(0, 0.0, 1.0, 1.0, 7, counter),
    ):
        first, second = kicks(draw)
        assert first.std() == pytest.approx(1.0, abs=0.01)
        correlations = np.corrcoef(first.T, second.T)[:3, 3:]
        assert np.abs(correlations).max() < 0.02


@pytest.mark.parametrize("keep_momentum", [False, True])
def test_the_noise_of_the_processor_and_the_gpu_is_the_same(keep_momentum):
    """Both draw the noise of each atom from its index and the step with the
    same counter-based generator (core/philox.h), whatever order the GPU keeps
    its atoms in: Langevin dynamics and Langevin baths run alike, in double
    precision to the rounding of their functions."""
    from mdapy.simulation.controls import Langevin

    system = crystal()
    cpu = EAM.zhou(["Cu", "Ni"])
    gpu = EAM.zhou(["Cu", "Ni"], device="cuda", precision="double")
    z = system["positions"][:, 2]
    bottom, top = z < z.min() + 4.0, z > z.max() - 4.0
    runs = []
    for potential in (gpu, cpu):
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.velocities(400, seed=1)
            md.run(
                200, ensemble="nvt", temperature=400, thermostat="langevin", seed=7,
                keep_momentum=keep_momentum,
            )  # fmt: skip
            md.run(
                200, ensemble="nve",
                controls=[Langevin(bottom, 600.0, damping=0.1, seed=3),
                          Langevin(top, 100.0, damping=0.1, seed=4)],
            )  # fmt: skip
            runs.append(md.system)
    a, b = runs
    np.testing.assert_allclose(a["positions"], b["positions"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(a["velocities"], b["velocities"], rtol=0, atol=1e-8)
