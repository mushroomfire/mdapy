# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""mdapy against OVITO: each case computes the same quantity with both
programs on the same system, and is timed only where both give the same
result.

Every measurement runs in a fresh process, so that the time and the memory
of one program do not include the other, and a process that grows beyond
the memory limit is stopped. Sizes grow eightfold; a size is skipped where
the previous one predicts more memory than the limit.

    python benchmarks/ovito_comparison.py run                  # every case
    python benchmarks/ovito_comparison.py run cna csp --sizes 4000 32000 --threads 1
    python benchmarks/ovito_comparison.py table results.json   # Markdown
    python benchmarks/ovito_comparison.py plot results.json timings.png
"""

import argparse
import gc
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

SIZES = (4_000, 32_000, 256_000, 2_048_000)
WARM_UP = 500  # atoms of the system each process computes once before timing


@dataclass(frozen=True)
class Case:
    """One quantity computed by both programs.

    ``inputs(atoms)`` makes the systems, the same in every process.
    ``mdapy(inputs)`` and ``ovito(inputs)`` prepare everything that is not
    timed and return the timed call. ``agree(ours, theirs)`` raises
    AssertionError where the results differ.
    """

    title: str
    inputs: Callable[[int], dict]
    mdapy: Callable[[dict], Callable[[], Any]]
    ovito: Callable[[dict], Callable[[], Any]]
    agree: Callable[[Any, Any], None]


# ------------------------------------------------------------ systems


def crystal(structure: str, a: float, atoms: int, sigma: float, seed: int = 1):
    """A cubic periodic crystal of about that many atoms, each displaced at
    random by sigma Å along each axis."""
    from mdapy.build import build_crystal

    per_cell = len(build_crystal(structure, a))
    n = max(1, round((atoms / per_cell) ** (1 / 3)))
    perfect = build_crystal(structure, a, repeat=(n, n, n))
    rng = np.random.default_rng(seed)
    return perfect.with_positions(
        perfect.positions + rng.normal(0.0, sigma, (len(perfect), 3))
    )


def copper(atoms: int) -> dict:
    """fcc copper at about room temperature."""
    return {"system": crystal("fcc", 3.615, atoms, 0.1)}


def silicon(atoms: int) -> dict:
    return {"system": crystal("diamond", 5.431, atoms, 0.05)}


def gas(atoms: int) -> dict:
    """Random points at the density of copper; with a cutoff of 2 Å, close
    to percolation, so clusters of every size."""
    import mdapy

    side = (atoms / 0.0847) ** (1 / 3)
    rng = np.random.default_rng(1)
    return {"system": mdapy.System(rng.random((atoms, 3)) * side, mdapy.Box(side))}


def deformed(atoms: int) -> dict:
    """A perfect fcc reference and the same crystal strained, its atoms
    displaced."""
    import mdapy

    reference = crystal("fcc", 3.615, atoms, 0.0)
    strain = np.eye(3) + [[0.02, 0.03, 0.0], [0.0, -0.01, 0.01], [0.0, 0.0, 0.01]]
    rng = np.random.default_rng(2)
    moved = reference.positions + rng.normal(0.0, 0.05, (len(reference), 3))
    current = mdapy.System(moved @ strain.T, mdapy.Box(reference.box.matrix @ strain.T))
    return {"system": current, "reference": reference}


def damaged(atoms: int) -> dict:
    """A perfect fcc reference and the same crystal with its atoms displaced
    and 0.5 % of them gone."""
    reference = crystal("fcc", 3.615, atoms, 0.0)
    current = crystal("fcc", 3.615, atoms, 0.2)
    rng = np.random.default_rng(3)
    kept = rng.random(len(current)) > 0.005
    return {"system": current.select(kept), "reference": reference}


# ------------------------------------------------------------ OVITO


def ovito_data(system):
    """The system as an OVITO data collection."""
    from ovito.data import DataCollection

    data = DataCollection()
    cell = np.zeros((3, 4))
    cell[:, :3] = system.box.matrix.T  # OVITO keeps the cell vectors as columns
    cell[:, 3] = system.box.origin
    data.create_cell(cell, pbc=tuple(bool(p) for p in system.box.pbc))
    particles = data.create_particles(count=len(system))
    particles.create_property("Position", data=system.positions)
    return data


def pipeline(modifiers: Callable[[dict], list]) -> Callable[[dict], Callable]:
    """The OVITO side of a case: a pipeline of the modifiers, new for each
    call so that nothing is cached."""

    def prepare(inputs: dict) -> Callable:
        from ovito.pipeline import Pipeline, StaticSource

        data = ovito_data(inputs["system"])
        references = {"reference": None}
        if "reference" in inputs:
            references["reference"] = StaticSource(data=ovito_data(inputs["reference"]))

        def run():
            p = Pipeline(source=StaticSource(data=data))
            for modifier in modifiers(references):
                p.modifiers.append(modifier)
            return p.compute()

        return run

    return prepare


class Member(str):
    """A member of an enumeration of the modifier, such as "Mode.FixedCutoff",
    looked up only in the process that runs OVITO."""


def modifier(name: str, **options) -> Callable[[dict], list]:
    def make(references: dict) -> list:
        import ovito.modifiers

        kind = getattr(ovito.modifiers, name)
        for key, value in options.items():
            if isinstance(value, Member):
                enumeration, member = value.split(".")
                options[key] = getattr(getattr(kind, enumeration), member)
        made = kind(**options)
        if references["reference"] is not None:
            made.reference = references["reference"]
        return [made]

    return make


def particles(name: str):
    return lambda data: np.asarray(data.particles[name])


# ------------------------------------------------------------ agreement


def equal(ours, theirs) -> None:
    np.testing.assert_array_equal(ours, theirs)


def close(ours, theirs, tolerance: float = 1e-10) -> None:
    np.testing.assert_allclose(ours, theirs, rtol=tolerance, atol=tolerance)


def structures(result, data, mapping=None) -> None:
    theirs = np.asarray(data.particles["Structure Type"])
    equal(result.structure, theirs if mapping is None else np.asarray(mapping)[theirs])


def analysis(make: Callable[[dict], Any]) -> Callable[[dict], Callable]:
    """The mdapy side of a case: make(inputs) is the analysis, computed for
    the system."""

    def prepare(inputs: dict) -> Callable:
        return lambda: make(inputs).compute(inputs["system"])

    return prepare


# ------------------------------------------------------------ cases


def _neighbors_ovito(inputs: dict) -> Callable:
    from ovito.data import CutoffNeighborFinder

    data = ovito_data(inputs["system"])
    return lambda: CutoffNeighborFinder(5.0, data).find_all()


def _neighbors_mdapy(inputs: dict) -> Callable:
    from mdapy.neighbor import NeighborList

    return lambda: NeighborList.build(inputs["system"], 5.0)


def _neighbors_agree(ours, theirs) -> None:
    pairs, vectors = theirs
    n = len(ours.counts)
    equal(ours.counts, np.bincount(pairs[:, 0], minlength=n))
    summed = np.bincount(pairs[:, 0], np.linalg.norm(vectors, axis=1), minlength=n)
    close(np.add.reduceat(ours.distances, ours.offsets[:-1]), summed)


def _nearest_ovito(inputs: dict) -> Callable:
    from ovito.data import NearestNeighborFinder

    data = ovito_data(inputs["system"])
    return lambda: NearestNeighborFinder(12, data).find_all()


def _nearest_mdapy(inputs: dict) -> Callable:
    from mdapy.neighbor import NeighborList

    return lambda: NeighborList.build(inputs["system"], num_neighbors=12)


def _nearest_agree(ours, theirs) -> None:
    _, vectors = theirs
    distances = np.sort(np.linalg.norm(vectors, axis=2), axis=1)
    close(ours.distances.reshape(-1, 12), distances, 1e-12)


def _cases() -> dict[str, Case]:
    from mdapy.analysis import (
        AtomicStrain,
        BondAngleDistribution,
        ClusterAnalysis,
        RadialDistributionFunction,
        SteinhardtParameters,
        VoronoiAnalysis,
        WignerSeitz,
    )
    from mdapy.identify import (
        AcklandJonesAnalysis,
        CentroSymmetryParameter,
        CommonNeighborAnalysis,
        DiamondStructure,
    )

    degrees = tuple(range(2, 7))

    def steinhardt(ours, data):
        theirs = np.asarray(data.particles["Steinhardt"])
        for degree in degrees:
            close(ours[degree], theirs[:, degree - 2])

    def rdf(ours, data):
        close(ours.g, data.tables["coordination-rdf"].xy()[:, 1])

    def bond_angles(ours, data):
        equal(ours.counts, data.tables["bond-angle-distr"].xy()[:, 1])

    def voronoi(ours, data):
        close(ours.volume, particles("Atomic Volume")(data))
        equal(ours.coordination, particles("Coordination")(data))

    def clusters(ours, data):
        theirs = np.bincount(particles("Cluster")(data))
        equal(ours.sizes, np.sort(theirs)[::-1])

    def wigner_seitz(ours, data):
        equal(ours.occupancy, particles("Occupancy")(data))

    def strain(ours, data):
        close(ours.shear, particles("Shear Strain")(data))
        close(ours.volumetric, particles("Volumetric Strain")(data))

    return {
        "neighbors": Case(
            "Neighbor list, 5 Å",
            copper,
            _neighbors_mdapy,
            _neighbors_ovito,
            _neighbors_agree,
        ),
        "nearest": Case(
            "12 nearest neighbors",
            copper,
            _nearest_mdapy,
            _nearest_ovito,
            _nearest_agree,
        ),
        "cna": Case(
            "Adaptive common neighbor analysis",
            copper,
            analysis(lambda inputs: CommonNeighborAnalysis()),
            pipeline(
                modifier(
                    "CommonNeighborAnalysisModifier", mode=Member("Mode.AdaptiveCutoff")
                )
            ),
            structures,
        ),
        "cna-fixed": Case(
            "Common neighbor analysis, cutoff 3.09 Å",
            copper,
            analysis(lambda inputs: CommonNeighborAnalysis(3.086)),
            pipeline(
                modifier(
                    "CommonNeighborAnalysisModifier",
                    mode=Member("Mode.FixedCutoff"),
                    cutoff=3.086,
                )
            ),
            structures,
        ),
        "ackland-jones": Case(
            "Ackland–Jones analysis",
            copper,
            analysis(lambda inputs: AcklandJonesAnalysis()),
            pipeline(modifier("AcklandJonesModifier")),
            structures,
        ),
        "csp": Case(
            "Centro-symmetry parameter, 12 neighbors",
            copper,
            analysis(lambda inputs: CentroSymmetryParameter(12)),
            pipeline(modifier("CentroSymmetryModifier", num_neighbors=12)),
            lambda ours, data: close(ours.csp, particles("Centrosymmetry")(data)),
        ),
        "diamond": Case(
            "Diamond structure",
            silicon,
            analysis(lambda inputs: DiamondStructure()),
            pipeline(modifier("IdentifyDiamondModifier")),
            # OVITO also marks the neighbors of diamond atoms
            lambda ours, data: structures(ours, data, [0, 1, 0, 0, 2, 0, 0]),
        ),
        "steinhardt": Case(
            "Steinhardt q2–q6, 12 neighbors",
            copper,
            analysis(lambda inputs: SteinhardtParameters(degrees, num_neighbors=12)),
            pipeline(
                modifier(
                    "SteinhardtParametersModifier",
                    neighbor_mode=Member("NeighborMode.NearestNeighbors"),
                    num_neighbors=12,
                    max_l=6,
                )
            ),
            steinhardt,
        ),
        "steinhardt-voronoi": Case(
            "Steinhardt q2–q6, Voronoi faces weighted by area",
            copper,
            analysis(
                lambda inputs: SteinhardtParameters(
                    degrees, voronoi=True, weighted=True
                )
            ),
            pipeline(
                modifier(
                    "SteinhardtParametersModifier",
                    neighbor_mode=Member("NeighborMode.Voronoi"),
                    max_l=6,
                )
            ),
            steinhardt,
        ),
        "rdf": Case(
            "Radial distribution function, 6 Å, 200 bins",
            copper,
            analysis(lambda inputs: RadialDistributionFunction(6.0, bins=200)),
            pipeline(
                modifier("CoordinationAnalysisModifier", cutoff=6.0, number_of_bins=200)
            ),
            rdf,
        ),
        "bond-angles": Case(
            "Bond angle distribution, 3 Å, 180 bins",
            copper,
            analysis(lambda inputs: BondAngleDistribution(cutoff=3.0, bins=180)),
            pipeline(
                lambda references: (
                    modifier("CreateBondsModifier", cutoff=3.0)(references)
                    + modifier("BondAngleDistributionModifier", bins=180)(references)
                )
            ),
            bond_angles,
        ),
        "voronoi": Case(
            "Voronoi volumes and coordination",
            copper,
            analysis(lambda inputs: VoronoiAnalysis()),
            pipeline(modifier("VoronoiAnalysisModifier")),
            voronoi,
        ),
        "clusters": Case(
            "Cluster analysis, 2 Å",
            gas,
            analysis(lambda inputs: ClusterAnalysis(cutoff=2.0)),
            pipeline(
                modifier("ClusterAnalysisModifier", cutoff=2.0, sort_by_size=True)
            ),
            clusters,
        ),
        "wigner-seitz": Case(
            "Wigner–Seitz occupancies",
            damaged,
            analysis(lambda inputs: WignerSeitz(inputs["reference"])),
            pipeline(modifier("WignerSeitzAnalysisModifier")),
            wigner_seitz,
        ),
        "atomic-strain": Case(
            "Atomic strain, 3 Å",
            deformed,
            analysis(lambda inputs: AtomicStrain(inputs["reference"], 3.0)),
            pipeline(modifier("AtomicStrainModifier", cutoff=3.0)),
            strain,
        ),
    }


# ------------------------------------------------------------ one process


def peak_memory() -> int:
    """The largest resident memory of this process so far, in bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def measure(name: str, atoms: int, side: str, repeat: int) -> dict:
    """In a process of its own: whether both programs agree (side "check"),
    or the times of one of them."""
    case = _cases()[name]
    if side != "mdapy":
        import ovito.data  # noqa: F401
        import ovito.modifiers  # noqa: F401
        import ovito.pipeline  # noqa: F401
    base = peak_memory()  # the libraries, before any system
    if side == "check":
        inputs = case.inputs(atoms)
        try:
            case.agree(case.mdapy(inputs)(), case.ovito(inputs)())
        except AssertionError as error:
            return {"agree": False, "message": str(error)[:2000]}
        size = len(inputs["system"])
        return {"agree": True, "atoms": size, "base": base, "peak": peak_memory()}
    prepare = case.mdapy if side == "mdapy" else case.ovito
    prepare(case.inputs(WARM_UP))()
    inputs = case.inputs(atoms)
    run = prepare(inputs)
    gc.collect()
    before = peak_memory()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = run()
        times.append(time.perf_counter() - start)
        del result
    return {"times": times, "base": base, "before": before, "peak": peak_memory()}


# ------------------------------------------------------------ the runs


def resident(pid: int) -> int:
    """The resident memory of a process now, in bytes."""
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()
    return int(out) * 1024 if out else 0


def spawn(name, atoms, side, repeat, threads, limit, timeout=None) -> dict:
    """measure() in a new process, stopped above limit bytes or after
    timeout seconds."""
    env = dict(
        os.environ, MDAPY_NUM_THREADS=str(threads), OVITO_THREAD_COUNT=str(threads)
    )
    command = [sys.executable, __file__, "_measure", name, str(atoms), side]
    command.append(str(repeat))
    process = subprocess.Popen(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    largest, start = 0, time.perf_counter()
    while process.poll() is None:
        largest = max(largest, resident(process.pid))
        stopped = None
        if largest > limit:
            stopped = f"more than {limit / 2**30:.1f} GB"
        elif timeout is not None and time.perf_counter() - start > timeout:
            stopped = f"more than {timeout:g} s"
        if stopped is not None:
            process.kill()
            process.communicate()  # waits, and closes the pipes
            return {"stopped": stopped}
        time.sleep(0.05)
    out, err = process.communicate()
    if process.returncode != 0:
        return {"failed": err.strip().splitlines()[-1] if err.strip() else "?"}
    return json.loads(out.strip().splitlines()[-1])


def run(names, sizes, threads, repeat, limit, output, timeout=None) -> list[dict]:
    """Every case at every size: check, then time each program; results
    appended to output as they come."""
    import mdapy

    results = []
    machine = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.platform(),
        "cpus": os.cpu_count(),
        "threads": threads,
        "mdapy_version": mdapy.__version__,
        "ovito_version": subprocess.run(
            [sys.executable, "-c", "import ovito; print(ovito.version_string)"],
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    for name in names:
        # side -> atoms, peak memory and memory of the libraries alone
        peaks: dict[str, tuple[int, int, int]] = {}
        stopped: set[str] = set()  # sides stopped at a smaller size
        for atoms in sizes:
            entry = {"case": name, "size": atoms, "atoms": atoms, **machine}
            for side in ("check", "mdapy", "ovito"):
                last = peaks.get(side)
                if side in stopped:
                    entry[side] = {"skipped": "stopped at a smaller size"}
                elif last is not None and predicted(*last, atoms) > limit:
                    entry[side] = {"skipped": "would need too much memory"}
                else:
                    entry[side] = spawn(
                        name, atoms, side, repeat, threads, limit, timeout
                    )
                    if "peak" in entry[side]:
                        peaks[side] = (atoms, entry[side]["peak"], entry[side]["base"])
                    if "stopped" in entry[side]:
                        stopped.add(side)
                if side == "check":
                    # timed only where both programs agree
                    if not entry["check"].get("agree"):
                        break
                    entry["atoms"] = entry["check"]["atoms"]
            results.append(entry)
            print(line(entry), flush=True)
            with open(output, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            if not entry["check"].get("agree"):
                break
    return results


def predicted(atoms: int, peak: int, base: int, more: int) -> float:
    """The peak memory for more atoms: that of the libraries, and the rest
    in proportion to the atoms."""
    return base + (peak - base) * more / atoms


def best(side: dict) -> float | None:
    return min(side["times"]) if "times" in side else None


def line(entry: dict) -> str:
    ours, theirs = best(entry.get("mdapy", {})), best(entry.get("ovito", {}))
    text = f"{entry['case']:20s} {entry['atoms']:>10,d}"
    if not entry["check"].get("agree"):
        return text + "  " + json.dumps(entry["check"])[:200]
    for side, value in (("mdapy", ours), ("ovito", theirs)):
        text += f"  {side} " + (
            f"{value:9.4f} s" if value is not None else json.dumps(entry[side])
        )
    if ours and theirs:
        text += f"  ×{theirs / ours:.2f}"
    return text


# ------------------------------------------------------------ reports


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(row) for row in f if row.strip()]


def extra(side: dict) -> float:
    """The memory a computation took beyond the systems, in MB."""
    return (side["peak"] - side["before"]) / 2**20


def table(entries: list[dict]) -> str:
    """The results as a Markdown table: the time of each program, how many
    times faster mdapy is, and the memory each computation took."""
    titles = {name: case.title for name, case in _cases().items()}
    rows = [
        "| Case | Atoms | mdapy (s) | OVITO (s) | OVITO / mdapy "
        "| mdapy (MB) | OVITO (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in entries:
        if not entry["check"].get("agree"):
            continue
        ours, theirs = best(entry["mdapy"]), best(entry["ovito"])
        if ours is None or theirs is None:
            continue
        rows.append(
            f"| {titles.get(entry['case'], entry['case'])} | {entry['atoms']:,d} "
            f"| {ours:.4f} | {theirs:.4f} | {theirs / ours:.2f} "
            f"| {extra(entry['mdapy']):.0f} | {extra(entry['ovito']):.0f} |"
        )
    return "\n".join(rows)


def plot(entries: list[dict], path: str) -> None:
    """The ratio of the times of OVITO and mdapy for each case and size."""
    from mdapy.visualization import plot as plots

    titles = {name: case.title for name, case in _cases().items()}
    done = [e for e in entries if e["check"].get("agree") and best(e.get("mdapy", {}))]
    done = [e for e in done if best(e.get("ovito", {}))]
    names = list(dict.fromkeys(e["case"] for e in done))
    sizes = sorted({e["size"] for e in done})
    ratios = {(e["case"], e["size"]): best(e["ovito"]) / best(e["mdapy"]) for e in done}
    with plots.style():
        fig, ax = plots.figure("double", 25 + 7 * len(names))
        height = 0.8 / len(sizes)
        for k, size in enumerate(sizes):
            ax.barh(
                np.arange(len(names)) + (k - (len(sizes) - 1) / 2) * height,
                [ratios.get((name, size), np.nan) for name in names],
                height,
                color=plots.COLORS[k % len(plots.COLORS)],
                label=f"{size:,d} atoms",
            )
        ax.axvline(1.0, color="k", lw=0.5)
        ax.set_xscale("log")
        ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)]
        low, high = ax.get_xlim()
        ticks = [t for t in ticks if low <= t <= high]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_yticks(np.arange(len(names)), [titles.get(n, n) for n in names])
        ax.invert_yaxis()
        ax.set_xlabel("time of OVITO / time of mdapy (above 1: mdapy is faster)")
        ax.legend(frameon=False, loc="lower right")
        plots.save(fig, path.rsplit(".", 1)[0], formats=(path.rsplit(".", 1)[1],))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    runs = commands.add_parser("run", help="check and time the cases")
    runs.add_argument("cases", nargs="*", help="default: every case")
    runs.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    runs.add_argument("--threads", type=int, default=os.cpu_count())
    runs.add_argument("--repeat", type=int, default=3)
    runs.add_argument("--memory", type=float, default=8.0, help="limit in GB")
    runs.add_argument(
        "--timeout", type=float, default=600.0, help="limit of a process in s"
    )
    runs.add_argument("--output", default="results.json")
    for name in ("table", "plot"):
        report = commands.add_parser(name)
        report.add_argument("results")
        if name == "plot":
            report.add_argument("figure")
    child = commands.add_parser("_measure")
    child.add_argument("case")
    child.add_argument("atoms", type=int)
    child.add_argument("side", choices=("check", "mdapy", "ovito"))
    child.add_argument("repeat", type=int)
    options = parser.parse_args(argv)
    if options.command == "_measure":
        result = measure(options.case, options.atoms, options.side, options.repeat)
        print(json.dumps(result))
    elif options.command == "run":
        names = options.cases or list(_cases())
        unknown = set(names) - set(_cases())
        if unknown:
            parser.error(f"unknown cases {sorted(unknown)}; known {list(_cases())}")
        limit = int(options.memory * 2**30)
        run(
            names,
            options.sizes,
            options.threads,
            options.repeat,
            limit,
            options.output,
            options.timeout,
        )
    elif options.command == "table":
        print(table(load(options.results)))
    else:
        plot(load(options.results), options.figure)


if __name__ == "__main__":
    main()
