"""The logs of molecular dynamics and of minimizations: their headings, the
settings of each run, rows that read back as the thermodynamic state, the
speed or where a run stopped, files and streams, and the errors."""

import io
import os

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM, LennardJones, _gpu
from mdapy.core.threads import CancelToken
from mdapy.errors import Cancelled, InvalidParameter
from mdapy.simulation import MolecularDynamics, minimize
from mdapy.simulation._log import _Log

from calculators.gpu_stand_in import stand_in

# the columns of the rows of MD and half the last decimal written
COLUMNS = {
    "step": 0.0,
    "time": 5e-5,
    "temperature": 5e-4,
    "potential_energy": 5e-7,
    "kinetic_energy": 5e-7,
    "total_energy": 5e-7,
    "pressure": 5e-7,
    "volume": 5e-5,
}


def copper(repeat=3):
    return build_crystal("fcc", 3.615, elements="Cu", repeat=(repeat,) * 3)


def comments(text):
    return [line for line in text.splitlines() if line.startswith("#")]


def test_rows_read_back_as_the_thermodynamic_state(tmp_path):
    path = tmp_path / "md.log"
    with MolecularDynamics(copper(), EAM.zhou(["Cu"]), timestep=0.002, log=path) as md:
        md.velocities(600, seed=1)
        md.run(60, thermo=20)
        md.run(
            40,
            ensemble="npt",
            temperature=(300, 400),
            pressure=0.0,
            thermostat="langevin",
            seed=3,
            thermo=20,
            save_every=20,
        )
        thermo = md.thermo
    text = path.read_text(encoding="utf-8")
    lines = comments(text)
    assert lines[0].startswith("# MDAPY ") and "molecular dynamics" in lines[0]
    assert any("108 atoms" in line and "periodic along a b c" in line for line in lines)
    assert any(line.startswith("# engine: MDAPY on the processor") for line in lines)
    assert any(line.startswith("# potential: EAM(") for line in lines)
    assert "# run 1: 60 steps, nve; thermo every 20" in lines
    settings = next(line for line in lines if line.startswith("# run 2: 40 steps"))
    assert "300.0 to 400.0 K, langevin" in settings and "seed 3" in settings
    assert "barostat iso" in settings and "atoms every 20" in settings
    assert sum("steps/s" in line and "ns/day" in line for line in lines) == 2
    rows = np.loadtxt(path)
    assert rows.shape == (len(thermo), len(COLUMNS) + 1)
    for k, (name, half) in enumerate(COLUMNS.items()):
        np.testing.assert_allclose(
            rows[:, k], thermo[name].to_numpy(), rtol=0, atol=half * 1.0001
        )
    assert (rows[:, -1] >= 0).all()  # the wall time


def test_a_stream_stays_open_and_a_stopped_run_says_so():
    stream = io.StringIO()
    lj = LennardJones((0.4, 2.3), 5.0)
    with MolecularDynamics(copper(4), lj, timestep=0.002, log=stream) as md:
        md.velocities(300, seed=1)
        frames = md.iterate(100, save_every=10, thermo=10)
        next(frames)
        next(frames)
        frames.close()  # stops the run at step 10
        token = CancelToken()
        token.cancel()
        with pytest.raises(Cancelled):
            md.run(50, cancel=token)
        md.run(0)
    assert not stream.closed
    lines = comments(stream.getvalue())
    assert "# run 1 stopped after 10 of 100 steps" in "\n".join(lines)
    assert any(line.startswith("# run 2 stopped after 0 of 50 steps") for line in lines)
    assert "# run 3: 0 steps" in lines


def test_the_engine_on_the_gpu(monkeypatch):
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    stream = io.StringIO()
    potential = EAM.zhou(["Cu"], device="cuda")
    with MolecularDynamics(copper(4), potential, timestep=0.002, log=stream) as md:
        md.run(2, thermo=1)
    assert (
        "# engine: MDAPY on the GPU, stand-in, 12.0 GB, compute capability 8.6, "
        "26 multiprocessors; mixed precision"
    ) in stream.getvalue()
    # a minimization with a potential there names the GPU
    stream = io.StringIO()
    minimize(copper(4), potential, max_steps=1, log=stream)
    assert "# GPU: stand-in, 12.0 GB, compute capability 8.6" in stream.getvalue()


def test_the_machine_is_named():
    """The heading of each log names the processor, its logical CPUs and the
    memory of the machine, as show_versions does."""
    rows = {}
    report = io.StringIO()
    mdapy.show_versions(file=report)
    for line in report.getvalue().splitlines():
        rows[line[:21].strip()] = line[21:]
    stream = io.StringIO()
    with MolecularDynamics(copper(3), EAM.zhou(["Cu"]), log=stream) as md:
        md.run(1)
    minimize(copper(3), EAM.zhou(["Cu"]), max_steps=1, log=stream)
    (machine, other) = [
        line
        for line in stream.getvalue().splitlines()
        if line.startswith("# machine: ")
    ]
    assert machine == other
    assert machine.startswith(
        f"# machine: {rows['CPU']}, {os.cpu_count()} logical CPUs, "
    )
    assert rows["Memory"] in machine or rows["Memory"] == "unknown"
    assert "GPU:" not in stream.getvalue()


def test_errors(tmp_path):
    with pytest.raises(InvalidParameter, match="a path or a stream"):
        MolecularDynamics(copper(), EAM.zhou(["Cu"]), log=3)
    with pytest.raises(InvalidParameter, match="cannot write the log"):
        MolecularDynamics(copper(), EAM.zhou(["Cu"]), log=tmp_path / "no" / "md.log")
    with pytest.raises(InvalidParameter, match="log_every"):
        minimize(copper(), EAM.zhou(["Cu"]), log_every=0)


def test_a_log_that_does_not_write():
    stream = io.StringIO()
    log = _Log(stream, active=False)
    assert not log.on
    log.comment("nothing")
    log.columns((("step", "{:d}", 10),))
    log.row(1)
    log.close()
    assert stream.getvalue() == ""


def test_minimization(tmp_path):
    rng = np.random.default_rng(0)
    shaken = copper().with_columns(
        positions=copper()["positions"] + rng.normal(0, 0.05, (108, 3))
    )
    path = tmp_path / "min.log"
    result = minimize(shaken, EAM.zhou(["Cu"]), cell="full", log=path, log_every=7)
    lines = comments(path.read_text(encoding="utf-8"))
    assert "energy minimization (FIRE 2.0)" in lines[0]
    assert any("the box relaxes (full) at 0.0 GPa" in line for line in lines)
    assert lines[-1].startswith(f"# converged after {result.steps} steps")
    rows = np.loadtxt(path)
    expected = sorted({*range(0, result.steps + 1, 7), result.steps})
    np.testing.assert_array_equal(rows[:, 0], expected)
    assert rows[-1, 1] == pytest.approx(result.energy, abs=1e-8)
    assert rows[-1, 2] == pytest.approx(result.max_force, rel=1e-6)
    # the box kept: no stress column; the ends of a minimization
    stream = io.StringIO()
    stopped = minimize(shaken, EAM.zhou(["Cu"]), max_steps=5, log=stream)
    text = stream.getvalue()
    assert "the box is kept" in text and comments(text)[-1].startswith(
        "# reached max_steps after 5 steps"
    )
    assert np.isnan(np.loadtxt(io.StringIO(text))[:, 3]).all()
    assert not stopped.converged
    stream = io.StringIO()
    minimize(shaken, EAM.zhou(["Cu"]), callback=lambda *_: True, log=stream)
    assert comments(stream.getvalue())[-1].startswith(
        "# stopped by the callback after 1"
    )
