# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""The comparison with OVITO in benchmarks/ runs, and both programs agree on
every case of it. Skipped where OVITO is not installed."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("ovito")

PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "ovito_comparison.py"
spec = importlib.util.spec_from_file_location("ovito_comparison", PATH)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


@pytest.mark.parametrize("name", list(comparison._cases()))
def test_both_programs_agree(name):
    assert comparison.measure(name, 300, "check", 1)["agree"]


def test_differences_are_reported():
    result = comparison.measure("csp", 300, "check", 1)
    assert result["agree"] and result["peak"] > 0
    case = comparison._cases()["csp"]
    inputs = case.inputs(300)
    with pytest.raises(AssertionError):
        case.agree(case.mdapy(inputs)(), case.ovito(comparison.copper(500))())


def test_run_table_and_plot(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    results = tmp_path / "results.json"
    options = ["--sizes", "300", "2000", "--repeat", "2", "--threads", "2"]
    comparison.main(["run", "csp", *options, "--output", str(results)])
    entries = comparison.load(results)
    assert [e["atoms"] for e in entries] == [256, 2048], entries
    assert all(len(e["mdapy"]["times"]) == 2 for e in entries)
    assert entries[0]["threads"] == 2
    comparison.main(["table", str(results)])
    assert "Centro-symmetry parameter" in capsys.readouterr().out
    comparison.main(["plot", str(results), str(tmp_path / "ratio.png")])
    assert (tmp_path / "ratio.png").stat().st_size > 1000


def test_memory_limit_and_failures(tmp_path):
    stopped = comparison.spawn("csp", 2000, "mdapy", 1, 1, limit=1)
    assert "stopped" in stopped
    failed = comparison.spawn("unknown", 300, "mdapy", 1, 1, limit=2**33)
    assert "failed" in failed
    # a size predicted to need more than the limit is skipped, and without
    # the check nothing is timed
    results = tmp_path / "results.json"
    comparison.run(["csp"], [300, 10**9], 1, 1, 2**31, results)
    last = json.loads(results.read_text().splitlines()[-1])
    assert last["check"] == {"skipped": "would need too much memory"}
    assert "mdapy" not in last
    with pytest.raises(SystemExit):
        comparison.main(["run", "nothing"])
