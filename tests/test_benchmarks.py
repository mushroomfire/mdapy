# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""The comparison with OVITO and freud in benchmarks/ runs, and every program
agrees with mdapy on every case of it. Skipped where OVITO or freud is not
installed."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("ovito")
pytest.importorskip("freud")

PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "comparison.py"
spec = importlib.util.spec_from_file_location("comparison", PATH)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)

SIDES = [(name, p) for name, case in comparison._cases().items() for p in case.others]


@pytest.mark.parametrize("name, program", SIDES, ids=[f"{n}-{p}" for n, p in SIDES])
def test_every_program_agrees(name, program):
    if program == "ovito":
        result = comparison.measure(name, 300, "check-ovito", 1)
    else:
        # freud's Voronoi tessellation crashes in a process that has loaded
        # OVITO, as this one has
        result = comparison.spawn(name, 300, f"check-{program}", 1, 2, 2**33)
    assert result["agree"], result


def test_differences_are_reported():
    result = comparison.measure("csp", 300, "check-ovito", 1)
    assert result["agree"] and result["peak"] > 0
    case = comparison._cases()["csp"]
    inputs = case.inputs(300)
    other = case.others["ovito"]
    with pytest.raises(AssertionError):
        other.agree(
            case.mdapy(inputs)(), other.prepare(comparison.copper(500))(), inputs
        )
    message = comparison.measure("clusters", 300, "check-ovito", 1)
    assert message["agree"]


def test_run_table_and_plot(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    results = tmp_path / "results.json"
    options = ["--sizes", "300", "2000", "--repeat", "2", "--threads", "2"]
    comparison.main(["run", "rdf", "csp", *options, "--output", str(results)])
    entries = comparison.load(results)
    assert [(e["case"], e["atoms"]) for e in entries] == [
        ("rdf", 256),
        ("rdf", 2048),
        ("csp", 256),
        ("csp", 2048),
    ], entries
    assert all(len(e["mdapy"]["times"]) == 2 for e in entries)
    assert all(len(e["freud"]["times"]) == 2 for e in entries if e["case"] == "rdf")
    assert entries[0]["threads"] == 2
    assert set(entries[0]["versions"]) == {"mdapy", "ovito", "freud"}
    comparison.main(["table", str(results)])
    out = capsys.readouterr().out
    assert "Centro-symmetry parameter" in out and "| freud" in out
    # a table per program, a row per size
    assert out.count("| Radial distribution function, 6 Å, 200 bins |") == 4
    rows = comparison.table(entries + [dict(entries[-1], ovito={"stopped": "x"})])
    assert "stopped: x" in rows
    comparison.main(["plot", str(results), str(tmp_path / "ratio.png")])
    assert (tmp_path / "ratio.png").stat().st_size > 1000


def test_memory_limit_and_failures(tmp_path):
    stopped = comparison.spawn("csp", 2000, "mdapy", 1, 1, limit=1)
    assert "stopped" in stopped
    failed = comparison.spawn("unknown", 300, "mdapy", 1, 1, limit=2**33)
    assert "failed" in failed
    # a size predicted to need more than the limit is skipped, and without
    # an agreeing program nothing is timed
    results = tmp_path / "results.json"
    comparison.run(["csp"], [300, 10**9], 1, 1, 2**31, results)
    last = json.loads(results.read_text().splitlines()[-1])
    assert last["check"] == {"ovito": {"skipped": "would need too much memory"}}
    assert "mdapy" not in last
    with pytest.raises(SystemExit):
        comparison.main(["run", "nothing"])
