# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The log of the DFT and the record of the parameters of each computation:
the form of the log on the processor, from steps made up; on a GPU the log of
real runs, the record in the labels, and a calculator made again from it."""

import io
import json
from collections.abc import Mapping

import numpy as np
import pytest

import mdapy
from mdapy.calculators._base import CalculatorResult
from mdapy.calculators._dft import log as logs
from mdapy.errors import InvalidParameter
from mdapy.io import read, write

from dft import espresso, gpu

HEADING = ["engine: MDAPY on a GPU", "machine: a machine"]


def plain(value):
    """A value of a system's info, which keeps mappings and lists read-only,
    as JSON gives it."""
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def comments(text):
    return [line for line in text.splitlines() if line.startswith("#")]


def steps(writer, energies, single=(), rejected=()):
    for i, e in enumerate(energies, 1):
        writer.step(
            i, e, 10.0**-i, 0.5 * i, 7.25, 3 * i, i in single, i not in rejected
        )


def test_rows_read_back_as_one_table(tmp_path):
    path = tmp_path / "dft.log"
    writer = logs.Writer(path, HEADING)
    assert writer.on
    parameters = {"arguments": {"encut": 300.0, "kpoints": [2, 2, 2]}}
    writer.start(1, "4 atoms, Al 4", parameters)
    steps(writer, [-230.5, -230.75, -230.8125], single=(1, 2))
    writer.end("converged in 3 steps, 1.00 s")
    # the next computation adds to the file; its parameters, the same, are
    # not written again
    writer.start(2, "4 atoms, Al 4", parameters)
    steps(writer, [-230.0, -230.25], rejected=(2,))
    writer.end("not converged in 2 steps, 0.50 s")
    text = path.read_text(encoding="utf-8")
    lines = comments(text)
    assert lines[0].startswith("# MDAPY ") and ", DFT, " in lines[0]
    assert lines[1:3] == ["# " + line for line in HEADING]
    assert lines.count("# calculation 1: 4 atoms, Al 4") == 1
    assert lines.count("# calculation 2: 4 atoms, Al 4") == 1
    written = [line for line in lines if line.startswith("# parameters: ")]
    assert len(written) == 1
    assert json.loads(written[0].removeprefix("# parameters: ")) == parameters
    assert "# calculation 1, step 1: single precision from here" in lines
    assert "# calculation 1, step 3: double precision from here" in lines
    assert "# calculation 2, step 2: tried back" in lines
    assert lines[-1] == "# calculation 2: not converged in 2 steps, 0.50 s"
    heading = [line for line in lines if "free_energy" in line]
    names = [name for name, _, _ in logs.COLUMNS] + ["wall"]
    assert len(heading) == 2 and heading[0].split()[1:] == names
    rows = np.loadtxt(path)
    assert rows.shape == (5, len(names))
    np.testing.assert_array_equal(rows[:, 0], [1, 1, 1, 2, 2])
    np.testing.assert_array_equal(rows[:, 1], [1, 2, 3, 1, 2])
    np.testing.assert_allclose(
        rows[:, 2], [-230.5, -230.75, -230.8125, -230.0, -230.25]
    )
    # the change from the step before in the same computation, none for the first
    assert np.isnan(rows[0, 3]) and np.isnan(rows[3, 3])
    np.testing.assert_allclose(rows[[1, 2, 4], 3], [-0.25, -0.0625, -0.25])
    np.testing.assert_allclose(rows[:, 4], [0.1, 0.01, 0.001, 0.1, 0.01])
    np.testing.assert_allclose(rows[:, 5:8], [[0.5, 7.25, 3], [1.0, 7.25, 6], [1.5, 7.25, 9],
                                              [0.5, 7.25, 3], [1.0, 7.25, 6]])  # fmt: skip
    assert (rows[:, 8] >= 0).all()


def test_new_parameters_are_written_again(tmp_path):
    # a new calculator writes the file anew
    path = tmp_path / "dft.log"
    path.write_text("old\n")
    writer = logs.Writer(path, HEADING)
    for number, mesh in enumerate(([2, 2, 2], [2, 2, 2], [3, 3, 3]), 1):
        writer.start(number, "1 atoms, Si 1", {"kpoints": mesh})
        steps(writer, [-100.0])
        writer.end("converged in 1 steps, 0.10 s")
    text = path.read_text()
    assert "old" not in text and text.count("# MDAPY ") == 1
    assert [line for line in comments(text) if "parameters" in line] == [
        '# parameters: {"kpoints": [2, 2, 2]}',
        '# parameters: {"kpoints": [3, 3, 3]}',
    ]


def test_a_stream_and_a_log_that_does_not_write():
    stream = io.StringIO()
    writer = logs.Writer(stream, HEADING)
    writer.start(1, "1 atoms, Si 1", {})
    steps(writer, [-100.0])
    writer.end("converged in 1 steps, 0.10 s")
    assert stream.getvalue().startswith("# MDAPY ") and not stream.closed
    for quiet in (
        logs.Writer(None, []),
        logs.Writer(io.StringIO(), HEADING, active=False),
    ):
        assert not quiet.on
        quiet.start(1, "1 atoms, Si 1", {})
        steps(quiet, [-100.0])
        quiet.end("converged")
    # the end of a computation that never started writes nothing
    logs.Writer(stream, HEADING).end("stopped")


def test_targets():
    for target in (None, "dft.log", io.StringIO()):
        logs.check_target(target)
    with pytest.raises(InvalidParameter, match="a path or a stream"):
        logs.check_target(3)
    with pytest.raises(InvalidParameter, match="a path or a stream"):
        logs.Writer(3, HEADING)


def test_the_info_of_a_result_goes_with_the_labels(tmp_path):
    record = {"mdapy": "2.3.0", "arguments": {"encut": 300.0, "kpoints": [2, 2, 2],
              "magmoms": None, "hubbard": {}}}  # fmt: skip
    system = mdapy.System([[0.0, 0, 0]], mdapy.Box(3.0), elements="Al")
    result = CalculatorResult(
        energy=-1.5, forces=np.zeros((1, 3)), info={"dft": record}
    )
    labelled = result.attach(system)
    assert labelled.info["energy"] == -1.5 and plain(labelled.info["dft"]) == record
    write(tmp_path / "labels.xyz", labelled)
    assert plain(read(tmp_path / "labels.xyz").info["dft"]) == record
    # without info, as other calculators give it
    bare = CalculatorResult(energy=-1.5).attach(system)
    assert set(bare.info) == {"energy"}


def test_every_setting_of_the_field_has_its_units():
    # the record gives the settings of scf.run in eV and Å: each one with a
    # default is either in hartree or bohr, with its factor, or without units
    import inspect

    from mdapy.calculators._dft import calculator, scf

    settings = {
        name
        for name, p in inspect.signature(scf.run).parameters.items()
        if isinstance(p.default, (bool, int, float, str, tuple))
    }
    assert settings == set(calculator.SCF_UNITS) | calculator.SCF_WITHOUT_UNITS
    assert not set(calculator.SCF_UNITS) & calculator.SCF_WITHOUT_UNITS


# -- on a GPU -----------------------------------------------------------------


def aluminium():
    a = 4.05
    cell = np.diag([a, a, a])
    fractions = np.array([[0.01, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    return mdapy.System(fractions @ cell, mdapy.Box(cell), elements=["Al"] * 4)


def calculator(**options):
    from mdapy.calculators import DFT

    settings = dict(pseudopotentials={"Al": espresso.pseudo("Al_std.upf")}, kpoints=(2, 2, 2),
                    sigma=0.2, tolerance=1e-8)  # fmt: skip
    settings.update(options)
    return DFT(300.0, **settings)


def test_the_log_of_a_run(tmp_path):
    gpu.device()
    path = tmp_path / "dft.log"
    dft = calculator(log=path)
    first = dft.compute(aluminium(), ("energy", "forces", "virial"))
    record = dft.last["parameters"]
    iterations = dft.last["iterations"]
    moved = aluminium().with_columns(positions=aluminium()["positions"] + 0.01)
    dft.compute(moved)
    text = path.read_text(encoding="utf-8")
    lines = comments(text)
    assert lines[1].startswith("# engine: MDAPY on ") and lines[2].startswith(
        "# machine: "
    )
    line = [
        x for x in lines if x.startswith("# calculation 1: 4 atoms, Al 4; mesh 2 2 2, ")
    ]
    assert len(line) == 1 and "k-points" in line[0] and "bands, grid" in line[0]
    # the parameters once, those of the record
    written = [line for line in lines if line.startswith("# parameters: ")]
    assert len(written) == 1
    assert json.loads(written[0].removeprefix("# parameters: ")) == record
    rows = np.loadtxt(path)
    first_rows = rows[rows[:, 0] == 1]
    assert len(first_rows) == iterations and set(rows[:, 0]) == {1, 2}
    assert dft.last["calculation"] == 2
    # the last step is the energy of the label, to the decimals written
    assert first_rows[-1, 2] == pytest.approx(first.energy, abs=1e-6)
    assert first_rows[-1, 4] < 1e-8
    summary = [
        line for line in lines if line.startswith("# calculation 1: converged in")
    ]
    assert (
        len(summary) == 1 and "pressure" in summary[0] and "largest force" in summary[0]
    )
    assert f"converged in {iterations} steps" in summary[0]


def test_the_record_gives_the_calculator_again(tmp_path):
    from mdapy.calculators import DFT

    gpu.device()
    dft = calculator()
    result = dft.compute(aluminium())
    record = dft.last["parameters"]
    assert result.info == {"dft": record}
    arguments = record["arguments"]
    # defaults and "auto" resolved
    assert arguments["ecutrho"] == 1200.0 and arguments["mixing"] == 0.7
    assert arguments["kernels"] in ("mdapy", "vendor")
    assert arguments["precision"] == "double"
    assert arguments["functional"] == "pbe_pw" and isinstance(arguments["bands"], int)
    assert record["pseudopotentials"]["Al"]["file"] == "Al_std.upf"
    assert record["internal"]["history"] == 8
    assert record["internal"]["first_tolerance"] == 0.0
    assert record["internal"]["metric_weight"] == 20.0
    assert "on_step" not in record["internal"] and "kgrid" not in record["internal"]
    assert record["gpu"] and record["processes"] == 1
    json.dumps(record)  # it is JSON
    again = DFT(pseudopotentials={"Al": espresso.pseudo("Al_std.upf")}, **arguments)
    assert again.compute(aluminium()).energy == pytest.approx(result.energy, abs=1e-8)
    assert again.last["parameters"]["arguments"] == arguments
    # the labels keep it
    write(tmp_path / "labels.xyz", result.attach(aluminium()))
    assert plain(read(tmp_path / "labels.xyz").info["dft"]) == record


def test_the_log_of_a_run_that_stops(tmp_path, monkeypatch):
    from mdapy.errors import CalculationFailed

    gpu.device()
    path = tmp_path / "dft.log"
    with pytest.raises(CalculationFailed):
        calculator(log=path, max_iterations=2).compute(aluminium())
    assert "calculation 1: not converged in 2 steps" in path.read_text()
    assert len(np.loadtxt(path)) == 2
    # an error inside the run ends its log too
    dft = calculator(log=path)

    def fails(*args):
        raise RuntimeError("no more")

    monkeypatch.setattr(logs.Writer, "step", fails)
    with pytest.raises(RuntimeError, match="no more"):
        dft.compute(aluminium())
    assert comments(path.read_text())[-1].startswith(
        "# calculation 1: stopped: RuntimeError"
    )


def test_magnetic_runs_and_streams():
    gpu.device()
    stream = io.StringIO()
    dft = calculator(log=stream, magmoms={"Al": 0.5}, max_iterations=60)
    dft.compute(aluminium())
    text = stream.getvalue()
    assert ", collinear spins" in text
    record = dft.last["parameters"]["arguments"]
    assert record["magmoms"] == {"Al": 0.5} and record["mixing"] == 0.4
    assert record["moment_radius"] == pytest.approx(dft.last["moment_radius"])
    per_atom = calculator(magmoms=[0.5, 0.5, 0.5, 0.5], max_iterations=60)
    per_atom.compute(aluminium())
    assert per_atom.last["parameters"]["arguments"]["magmoms"] == [0.5] * 4
