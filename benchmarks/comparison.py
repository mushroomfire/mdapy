# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""mdapy against OVITO, freud, ASE and Atomsk: each case computes the same
quantity, reads or writes the same file or builds the same structure with
mdapy and with the programs that have it, and a program is timed only where
it gives the result of mdapy. The cases come in three groups: analyses,
files and builders.

Every measurement runs in a fresh process, so that the time and the memory
of one program do not include another, and a process that grows beyond the
memory limit is stopped. Sizes grow eightfold; a size is skipped where the
previous one predicts more memory than the limit.

    python benchmarks/comparison.py run                  # every case
    python benchmarks/comparison.py run cna rdf --sizes 4000 32000 --threads 1
    python benchmarks/comparison.py run --group files
    python benchmarks/comparison.py table results.json   # Markdown
    python benchmarks/comparison.py plot results.json timings.png
"""

import argparse
import atexit
import gc
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

SIZES = (4_000, 32_000, 256_000, 2_048_000)
WARM_UP = 500  # atoms of the system each process computes once before timing
PROGRAMS = {"ovito": "OVITO", "freud": "freud", "ase": "ASE", "atomsk": "Atomsk"}
GROUPS = ("analyses", "files", "builders")


@dataclass(frozen=True)
class Other:
    """The side of another program: ``prepare(inputs)`` does everything that
    is not timed and returns the timed call; ``agree(ours, theirs, inputs)``
    raises AssertionError where its result differs from that of mdapy."""

    prepare: Callable[[dict], Callable[[], Any]]
    agree: Callable[[Any, Any, dict], None]


@dataclass(frozen=True)
class Case:
    """One quantity computed by mdapy and other programs.

    ``inputs(atoms)`` makes the systems, the same in every process.
    ``mdapy(inputs)`` prepares everything that is not timed and returns the
    timed call; ``others`` holds the programs that compute the same, by name.
    """

    title: str
    inputs: Callable[[int], dict]
    mdapy: Callable[[dict], Callable[[], Any]]
    others: dict[str, Other]
    group: str = "analyses"


# ------------------------------------------------------------ systems


def crystal(structure: str, a: float, atoms: int, sigma: float, seed: int = 1):
    """A cubic periodic crystal of about that many atoms, each displaced at
    random by sigma Å along each axis."""
    from mdapy.build import build_crystal

    per_cell = len(build_crystal(structure, a))
    n = max(1, round((atoms / per_cell) ** (1 / 3)))
    perfect = build_crystal(structure, a, repeat=(n, n, n))
    rng = np.random.default_rng(seed)
    return perfect.with_columns(
        positions=perfect["positions"] + rng.normal(0.0, sigma, (len(perfect), 3))
    )


def copper(atoms: int) -> dict:
    """fcc copper at about room temperature."""
    return {"system": crystal("fcc", 3.615, atoms, 0.1)}


def silicon(atoms: int) -> dict:
    return {"system": crystal("diamond", 5.431, atoms, 0.05)}


def gas(atoms: int) -> dict:
    """Random points at the density of copper; with a cutoff of 2 Å, close
    to percolation, so clusters of every size. Points with a neighbor within
    1e-4 Å of 2 Å are left out, so that the bonds are the same in single
    precision."""
    import mdapy
    from mdapy.neighbor import NeighborList

    side = (atoms / 0.0847) ** (1 / 3)
    rng = np.random.default_rng(1)
    system = mdapy.System(rng.random((atoms, 3)) * side, mdapy.Box(side))
    pairs = NeighborList.build(system, 2.0 + 1e-4)
    close = np.abs(pairs.distances - 2.0) < 1e-4
    kept = np.ones(len(system), dtype=bool)
    kept[pairs.i[close]] = False
    return {"system": system.filter(kept)}


def deformed(atoms: int) -> dict:
    """A perfect fcc reference and the same crystal strained, its atoms
    displaced."""
    import mdapy

    reference = crystal("fcc", 3.615, atoms, 0.0)
    strain = np.eye(3) + [[0.02, 0.03, 0.0], [0.0, -0.01, 0.01], [0.0, 0.0, 0.01]]
    rng = np.random.default_rng(2)
    moved = reference["positions"] + rng.normal(0.0, 0.05, (len(reference), 3))
    current = mdapy.System(moved @ strain.T, mdapy.Box(reference.box.matrix @ strain.T))
    return {"system": current, "reference": reference}


def damaged(atoms: int) -> dict:
    """A perfect fcc reference and the same crystal with its atoms displaced
    and 0.5 % of them gone."""
    reference = crystal("fcc", 3.615, atoms, 0.0)
    current = crystal("fcc", 3.615, atoms, 0.2)
    rng = np.random.default_rng(3)
    kept = rng.random(len(current)) > 0.005
    return {"system": current.filter(kept), "reference": reference}


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
    particles.create_property("Position", data=system["positions"])
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


def ovito(prepare: Callable[[dict], Callable], agree: Callable[[Any, Any], None]):
    """The OVITO side of a case; OVITO computes in double precision, as
    mdapy, so its results are compared without the inputs."""
    return Other(prepare, lambda ours, theirs, inputs: agree(ours, theirs))


# ------------------------------------------------------------ freud


def freud_side(compute: Callable[..., Any]) -> Callable[[dict], Callable]:
    """The freud side of a case: compute(box, points) with the system in
    freud's box, converted before the timing."""

    def prepare(inputs: dict) -> Callable:
        from mdapy.interop import to_freud

        box, points = to_freud(inputs["system"])
        return lambda: compute(box, points)

    return prepare


def _freud_neighbors(query: dict) -> Callable[[dict], Callable]:
    def compute(box, points):
        import freud

        found = freud.locality.AABBQuery(box, points).query(points, query)
        return found.toNeighborList()

    return freud_side(compute)


def _freud_voronoi_steinhardt(box, points):
    """Steinhardt q2 to q6 over the Voronoi faces, weighted by their areas."""
    import freud

    voronoi = freud.locality.Voronoi().compute((box, points))
    steinhardt = freud.order.Steinhardt(list(range(2, 7)), weighted=True)
    return steinhardt.compute((box, points), neighbors=voronoi.nlist)


def _freud(name: str, *args, **options) -> Callable[[dict], Callable]:
    """freud's compute of the class at that dotted name, made with these
    arguments, for the box and points of the system."""
    compute_options = options.pop("compute", {})

    def compute(box, points):
        import freud

        module, cls = name.split(".")
        made = getattr(getattr(freud, module), cls)(*args, **options)
        return made.compute((box, points), **compute_options)

    return freud_side(compute)


# ------------------------------------------------------------ agreement


def equal(ours, theirs) -> None:
    np.testing.assert_array_equal(ours, theirs)


def close(ours, theirs, tolerance: float = 1e-10) -> None:
    np.testing.assert_allclose(ours, theirs, rtol=tolerance, atol=tolerance)


# freud computes in single precision: positions in a box of 300 Å are
# rounded by some 1e-5 Å, so a pair this close to a cutoff may fall on either
# side of it, and values agree to this much
SINGLE = 1e-4


def structures(result, data, mapping=None) -> None:
    theirs = np.asarray(data.particles["Structure Type"])
    equal(result.structure, theirs if mapping is None else np.asarray(mapping)[theirs])


def analysis(make: Callable[[dict], Any]) -> Callable[[dict], Callable]:
    """The mdapy side of a case: make(inputs) is the analysis, computed for
    the system."""

    def prepare(inputs: dict) -> Callable:
        return lambda: make(inputs).compute(inputs["system"])

    return prepare


def together(make: Callable[[dict], list], cutoff: float) -> Callable[[dict], Callable]:
    """The mdapy side of analyses together: one list within cutoff, built
    inside the timing, given to each of make(inputs)."""

    def prepare(inputs: dict) -> Callable:
        from mdapy.neighbor import NeighborList

        analyses, system = make(inputs), inputs["system"]

        def run():
            nl = NeighborList.build(system, cutoff)
            return [a.compute(system, neighbors=nl) for a in analyses]

        return run

    return prepare


def _freud_steinhardt_rdf(box, points):
    """Steinhardt q2 to q6 of the 12 nearest and the radial distribution
    function within 5 Å, with one tree of freud for both queries; given a
    list of the pairs instead, freud's radial distribution function takes
    fifty times as long."""
    import freud

    query = freud.locality.AABBQuery(box, points)
    rdf = freud.density.RDF(bins=200, r_max=5.0).compute(query)
    nearest = {"num_neighbors": 12, "exclude_ii": True}
    steinhardt = freud.order.Steinhardt(list(range(2, 7))).compute(
        query, neighbors=nearest
    )
    return steinhardt, rdf


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


# ------------------------------------------------------------ files

_FOLDERS: list[str] = []


def _folder() -> str:
    """A folder of this process for the files of the cases, removed when it
    ends."""
    if not _FOLDERS:
        folder = tempfile.mkdtemp(prefix="mdapy-comparison-")
        atexit.register(shutil.rmtree, folder, True)
        _FOLDERS.append(folder)
    return _FOLDERS[0]


# each format: its suffix, the columns written, and the options of OVITO
# and ASE to read and to write it (None where a program cannot)
FILES = {
    "lammps-dump": {
        "title": "LAMMPS dump",
        "suffix": ".dump",
        "columns": ("id", "type", "positions"),
        "ovito read": {},
        "ovito write": {
            "format": "lammps/dump",
            "columns": [
                "Particle Identifier",
                "Particle Type",
                "Position.X",
                "Position.Y",
                "Position.Z",
            ],
        },
        "ase read": {"format": "lammps-dump-text"},
        "ase write": None,
    },
    "lammps-data": {
        "title": "LAMMPS data file",
        "suffix": ".data",
        "columns": ("id", "type", "element", "positions"),
        "ovito read": {"atom_style": "atomic"},
        "ovito write": {"format": "lammps/data", "atom_style": "atomic"},
        "ase read": {"format": "lammps-data", "atom_style": "atomic", "units": "metal"},
        "ase write": {"format": "lammps-data", "atom_style": "atomic", "masses": True},
    },
    "extxyz": {
        "title": "extended XYZ",
        "suffix": ".xyz",
        "columns": ("element", "positions"),
        "ovito read": {},
        "ovito write": {
            "format": "xyz",
            "columns": ["Particle Type", "Position.X", "Position.Y", "Position.Z"],
        },
        "ase read": {"format": "extxyz"},
        "ase write": {"format": "extxyz"},
    },
}


def copper_file(fmt: str, write: bool) -> Callable[[int], dict]:
    """Inputs of a file case: copper with the columns of the format, and the
    file mdapy writes of it to read, or the name of the file to write."""

    def inputs(atoms: int) -> dict:
        import mdapy

        spec = FILES[fmt]
        system = copper(atoms)["system"]
        n = len(system)
        system = system.with_columns(id=np.arange(1, n + 1), type=1, element="Cu")
        system = system.select(*spec["columns"])
        name = f"{'written' if write else 'copper'}-{n}{spec['suffix']}"
        path = os.path.join(_folder(), name)
        if not write:
            mdapy.write(path, system)
        return {"system": system, "path": path}

    return inputs


def _ovito_collection(system):
    """The system as an OVITO data collection with identifiers and types."""
    from ovito.data import ParticleType

    data = ovito_data(system)
    particles = data.particles_
    columns = system.table.columns
    if "id" in columns:
        particles.create_property("Particle Identifier", data=system["id"])
    types = system["type"] if "type" in columns else np.ones(len(system), int)
    kinds = particles.create_property("Particle Type", data=types)
    kinds.types.append(ParticleType(id=1, name="Cu", mass=63.546))
    return data


def _read(fmt: str) -> tuple[Callable, dict[str, Other]]:
    """The mdapy side and the other programs of reading a format."""
    spec = FILES[fmt]

    def mdapy_side(inputs: dict) -> Callable:
        import mdapy

        return lambda: mdapy.read(inputs["path"])

    def ovito_side(inputs: dict) -> Callable:
        from ovito.io import import_file

        return lambda: import_file(inputs["path"], **spec["ovito read"]).compute()

    def ovito_agree(ours, data, inputs) -> None:
        theirs = np.asarray(data.particles.positions)
        np.testing.assert_allclose(theirs, ours["positions"], rtol=0, atol=1e-12)
        if "type" in ours.table.columns:
            equal(np.asarray(data.particles.particle_types), ours["type"])

    def ase_side(inputs: dict) -> Callable:
        import ase.io

        return lambda: ase.io.read(inputs["path"], **spec["ase read"])

    def ase_agree(ours, atoms, inputs) -> None:
        np.testing.assert_allclose(
            atoms.positions, ours["positions"], rtol=0, atol=1e-12
        )
        if "element" in ours.table.columns:
            equal(atoms.get_chemical_symbols(), ours["element"].tolist())

    return mdapy_side, {
        "ovito": Other(ovito_side, ovito_agree),
        "ase": Other(ase_side, ase_agree),
    }


def _write(fmt: str) -> tuple[Callable, dict[str, Other]]:
    """The mdapy side and the other programs of writing a format."""
    spec = FILES[fmt]

    def mdapy_side(inputs: dict) -> Callable:
        import mdapy

        def run():
            mdapy.write(inputs["path"], inputs["system"])
            return inputs["path"]

        return run

    def written(tolerance: float) -> Callable:
        """The file a program wrote holds the system, read by mdapy."""

        def agree(ours, theirs, inputs) -> None:
            import mdapy

            system, back = inputs["system"], mdapy.read(theirs)
            equal(len(back), len(system))
            np.testing.assert_allclose(
                back["positions"], system["positions"], rtol=0, atol=tolerance
            )
            for column in ("type", "element"):
                if column in system.table.columns:
                    equal(back[column], system[column])

        return agree

    def ovito_side(inputs: dict) -> Callable:
        from ovito.io import export_file

        data = _ovito_collection(inputs["system"])
        path = inputs["path"].replace("written-", "ovito-")

        def run():
            export_file(data, path, precision=17, **spec["ovito write"])
            return path

        return run

    def ase_side(inputs: dict) -> Callable:
        import ase.io

        from mdapy.interop import to_ase

        atoms = to_ase(inputs["system"])
        path = inputs["path"].replace("written-", "ase-")

        def run():
            ase.io.write(path, atoms, **spec["ase write"])
            return path

        return run

    others = {"ovito": Other(ovito_side, written(1e-12))}
    if spec["ase write"] is not None:
        # ASE writes extended XYZ with 8 decimals
        others["ase"] = Other(ase_side, written(1e-8 if fmt == "extxyz" else 1e-12))
    return mdapy_side, others


def _file_cases() -> dict[str, Case]:
    cases = {}
    for fmt, spec in FILES.items():
        for write, (mdapy_side, others) in ((False, _read(fmt)), (True, _write(fmt))):
            verb = "Write" if write else "Read"
            name = f"{verb.lower()}-{fmt.removeprefix('lammps-')}"
            cases[name] = Case(
                f"{verb} a {spec['title']}"
                if fmt != "extxyz"
                else f"{verb} {spec['title']}",
                copper_file(fmt, write),
                mdapy_side,
                others,
                group="files",
            )
    return cases


# ------------------------------------------------------------ builders


def atomsk() -> str | None:
    """The Atomsk program: ATOMSK, or atomsk on the path."""
    return shutil.which(os.environ.get("ATOMSK", "atomsk"))


def _atomsk(*arguments: str) -> None:
    program = atomsk()
    if program is None:
        raise RuntimeError("Atomsk not found: set ATOMSK or put atomsk on the path")
    subprocess.run(
        [program, *arguments, "-ow", "-v", "0"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _repeat(atoms: int) -> int:
    return max(1, round((atoms / 4) ** (1 / 3)))


def single_crystal(atoms: int) -> dict:
    """fcc copper of n x n x n cubic cells, and where to write it."""
    from mdapy.build import build_crystal

    n = _repeat(atoms)
    system = build_crystal("fcc", 3.615, elements="Cu", repeat=(n, n, n))
    return {
        "system": system,
        "repeat": n,
        "path": os.path.join(_folder(), f"crystal-{n}"),
    }


# grains of the polycrystal, at random places and orientations
GRAINS = 20


def _rotation(angles) -> np.ndarray:
    """Atomsk's rotation of angles about x, then y, then z, in degrees."""
    a, b, c = np.radians(angles)
    x = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    y = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    z = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return z @ y @ x


def polycrystal(atoms: int) -> dict:
    """A copper polycrystal of GRAINS grains at fixed places and
    orientations, the parameter file of Atomsk for it, and where to write
    it."""
    from mdapy.build import build_crystal

    side = (atoms * 3.615**3 / 4) ** (1 / 3)
    rng = np.random.default_rng(1)
    seeds = rng.random((GRAINS, 3)) * side
    angles = np.round(rng.uniform(-180.0, 180.0, (GRAINS, 3)), 2)
    folder = _folder()
    parameters = os.path.join(folder, f"polycrystal-{atoms}.txt")
    with open(parameters, "w", encoding="utf-8") as f:
        f.write(f"box {side:.6f} {side:.6f} {side:.6f}\n")
        for (x, y, z), (a, b, c) in zip(seeds, angles):
            f.write(f"node {x:.6f} {y:.6f} {z:.6f} {a:.2f} {b:.2f} {c:.2f}\n")
    unit = build_crystal("fcc", 3.615, elements="Cu")
    made = {
        "unit": unit,
        "box": float(f"{side:.6f}"),
        "seeds": np.round(seeds, 6),
        "orientations": np.array([_rotation(g) for g in angles]),
        "parameters": parameters,
        "path": os.path.join(folder, f"polycrystal-{atoms}"),
    }
    made["system"] = _polycrystal(made)
    return made


def _polycrystal(made: dict):
    """mdapy's polycrystal of the inputs; atoms are removed only where two
    are at the same place, as Atomsk removes none."""
    import mdapy
    from mdapy.build import build_polycrystal

    return build_polycrystal(
        made["unit"],
        mdapy.Box(made["box"]),
        made["seeds"],
        orientations=made["orientations"],
        min_distance=1e-2,
    )


def _build_crystal_mdapy(inputs: dict) -> Callable:
    import mdapy
    from mdapy.build import build_crystal

    n, path = inputs["repeat"], inputs["path"] + "-mdapy.data"

    def run():
        mdapy.write(path, build_crystal("fcc", 3.615, elements="Cu", repeat=(n, n, n)))
        return path

    return run


def _build_crystal_atomsk(inputs: dict) -> Callable:
    n, path = str(inputs["repeat"]), inputs["path"] + "-atomsk.lmp"

    def run():
        _atomsk("--create", "fcc", "3.615", "Cu", "-duplicate", n, n, n, path)
        return path

    return run


def _same_sites(ours, theirs, inputs) -> None:
    """Both files hold the same atoms, in any order, within the 12 decimals
    of Atomsk."""
    import mdapy

    mine, other = mdapy.read(ours), mdapy.read(theirs)
    np.testing.assert_allclose(other.box.matrix, mine.box.matrix, atol=1e-9)
    equal(len(other), len(mine))

    def ordered(system):
        r = np.round(system["positions"], 6)
        return system["positions"][np.lexsort(r.T)]

    np.testing.assert_allclose(ordered(other), ordered(mine), rtol=0, atol=1e-9)


def _build_polycrystal_mdapy(inputs: dict) -> Callable:
    import mdapy

    path = inputs["path"] + "-mdapy.data"

    def run():
        mdapy.write(path, _polycrystal(inputs))
        return path

    return run


def _build_polycrystal_atomsk(inputs: dict) -> Callable:
    unit = os.path.join(_folder(), "unit.xsf")
    _atomsk("--create", "fcc", "3.615", "Cu", unit)
    path = inputs["path"] + "-atomsk.lmp"

    def run():
        _atomsk("--polycrystal", unit, inputs["parameters"], path)
        return path

    return run


def _same_polycrystal(ours, theirs, inputs) -> None:
    """The same box, as many atoms within 0.5 %, and in each grain the same
    lattice: away from the faces of its cell, the atoms of Atomsk are those
    of mdapy moved by one vector, since each program places the lattice of
    a grain in its own way."""
    from mdapy._core import neighbor as search

    import mdapy

    mine, other = mdapy.read(ours), mdapy.read(theirs)
    np.testing.assert_allclose(other.box.matrix, mine.box.matrix, atol=1e-6)
    assert abs(len(other) - len(mine)) <= 0.005 * len(mine), (len(other), len(mine))
    box = mine.box
    frame = tuple(np.ascontiguousarray(a) for a in (box.matrix, box.origin, box.pbc))
    theirs_at = np.ascontiguousarray(other["positions"])
    seeds = search.KdTree(np.ascontiguousarray(inputs["seeds"]), *frame)
    grain, _, d = seeds.nearest(theirs_at, 2)
    inside = d[:, 1] - d[:, 0] > 4.0
    atoms = search.KdTree(np.ascontiguousarray(mine["positions"]), *frame)
    j, shift, _ = atoms.nearest(theirs_at, 1)
    # from each atom of Atomsk to the nearest atom of mdapy
    moved = mine["positions"][j[:, 0]] + shift[:, 0] @ box.matrix - theirs_at
    checked = 0
    for g in range(len(inputs["seeds"])):
        here = moved[(grain[:, 0] == g) & inside]
        if len(here):  # small grains may have no atom away from their faces
            spread = np.ptp(here, axis=0).max()
            assert spread < 1e-6, (
                f"grain {g}: the atoms move by vectors {spread} Å apart"
            )
            checked += 1
    assert checked > 0, "no grain has atoms away from its faces"


def _builder_cases() -> dict[str, Case]:
    return {
        "single-crystal": Case(
            "fcc single crystal, written as a LAMMPS data file",
            single_crystal,
            _build_crystal_mdapy,
            {"atomsk": Other(_build_crystal_atomsk, _same_sites)},
            group="builders",
        ),
        "polycrystal": Case(
            f"Polycrystal of {GRAINS} grains, written as a LAMMPS data file",
            polycrystal,
            _build_polycrystal_mdapy,
            {"atomsk": Other(_build_polycrystal_atomsk, _same_polycrystal)},
            group="builders",
        ),
    }


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
        PolyhedralTemplateMatching,
    )

    degrees = tuple(range(2, 7))

    def steinhardt(ours, data, tolerance=1e-10):
        theirs = np.asarray(data.particles["Steinhardt"])
        for degree in degrees:
            close(ours[degree], theirs[:, degree - 2], tolerance)

    def steinhardt_voronoi(ours, data):
        # the areas of faces come from two tessellations, with different
        # rounding, and weight the sums
        steinhardt(ours, data, 1e-8)

    def steinhardt_freud(ours, theirs, inputs):
        from mdapy.interop import from_freud, to_freud
        from mdapy.neighbor import NeighborList

        # freud rounds the positions to single precision, by some 3e-5 Å in a
        # box of 600 Å: mdapy computes the parameters again from the
        # positions freud has. An atom whose 12th and 13th neighbors are this
        # close may have other 12 nearest. freud's harmonics in single
        # precision then differ by 2e-5 for all but a few atoms in a million,
        # by up to 2e-4 for those.
        rounded = from_freud(*to_freud(inputs["system"]))
        mine = SteinhardtParameters(degrees, num_neighbors=12).compute(rounded)
        d = NeighborList.build(rounded, num_neighbors=13).distances.reshape(-1, 13)
        clear = d[:, 12] - d[:, 11] > SINGLE
        assert clear.mean() > 0.9999
        close(mine.q[clear], theirs.particle_order[clear], 3 * SINGLE)

    def steinhardt_voronoi_freud(ours, theirs, inputs):
        # freud leaves out some faces of distant neighbors below some 2e-3 Å²,
        # which have that weight in the sums
        close(ours.q, theirs.particle_order, 1e-3)

    def neighbors_freud(ours, theirs, inputs):
        from mdapy.neighbor import NeighborList

        n = len(ours.counts)
        counts = np.bincount(theirs.query_point_indices, minlength=n)
        wider = NeighborList.build(inputs["system"], 5.0 + SINGLE)
        edge = np.bincount(ours.i[ours.distances > 5.0 - SINGLE], minlength=n)
        assert np.all(counts <= wider.counts) and np.all(counts >= ours.counts - edge)
        same = counts == ours.counts
        summed = np.bincount(theirs.query_point_indices, theirs.distances, minlength=n)
        ours_summed = np.add.reduceat(ours.distances, ours.offsets[:-1])
        close(ours_summed[same], summed[same], SINGLE)

    def nearest_freud(ours, theirs, inputs):
        order = np.lexsort((theirs.distances, theirs.query_point_indices))
        close(
            ours.distances.reshape(-1, 12),
            theirs.distances[order].reshape(-1, 12),
            SINGLE,
        )

    def rdf_freud(ours, theirs, inputs):
        from mdapy.neighbor import NeighborList

        # a pair this close to the edge of a bin may fall into the next one in
        # single precision: each bin may differ by the pairs at its edges
        reach = ours.edges[-1] + SINGLE
        distances = NeighborList.build(inputs["system"], reach).distances
        width = ours.edges[1] - ours.edges[0]
        edge = np.rint(distances / width).astype(np.int64)
        near = np.abs(distances - edge * width) < SINGLE
        slack = np.bincount(edge[near], minlength=len(ours.edges))
        counts = np.histogram(distances, ours.edges)[0]
        per_pair = np.divide(
            ours.g, counts, out=np.zeros_like(ours.g), where=counts > 0
        )
        allowed = (slack[:-1] + slack[1:]) * per_pair.max()
        # and freud normalizes in single precision
        assert np.all(np.abs(ours.g - theirs.rdf) <= allowed + SINGLE * ours.g + 1e-6)

    def voronoi_volumes_freud(ours, theirs, inputs):
        close(ours.volume, theirs.volumes, SINGLE)

    def voronoi_freud(ours, theirs, inputs):
        close(ours.volume, theirs.volumes, SINGLE)
        # freud leaves out some faces of distant neighbors, all below 2e-2 Å²
        # (0.1 % of them for 16 million atoms); every face of freud above
        # 1e-2 Å² is one of mdapy
        n = len(ours.volume)
        mine = np.repeat(np.arange(n), np.diff(ours.offsets)) * n + ours.j
        nlist = theirs.nlist
        found = nlist.query_point_indices.astype(np.int64) * n + nlist.point_indices
        assert np.isin(found[nlist.weights > 1e-2], mine).all()
        assert ours.areas[~np.isin(mine, found)].max(initial=0.0) < 2e-2

    def metal(ours, data):
        # Ackland-Jones analysis runs last of the two and leaves its types
        structures(ours[1], data)
        close(ours[2].csp, particles("Centrosymmetry")(data))
        steinhardt(ours[3], data)
        rdf(ours[4], data)
        bond_angles(ours[5], data)

    def steinhardt_rdf(ours, data):
        steinhardt(ours[0], data)
        rdf(ours[1], data)

    def steinhardt_rdf_freud(ours, theirs, inputs):
        steinhardt_freud(ours[0], theirs[0], inputs)
        rdf_freud(ours[1], theirs[1], inputs)

    def clusters_freud(ours, theirs, inputs):
        equal(ours.sizes, np.sort(np.bincount(theirs.cluster_idx))[::-1])

    def rdf(ours, data):
        close(ours.g, data.tables["coordination-rdf"].xy()[:, 1])

    def bond_angles(ours, data):
        equal(ours.counts, data.tables["bond-angle-distr"].xy()[:, 1])

    def voronoi(ours, data):
        close(ours.volume, particles("Atomic Volume")(data))
        # faces of no area come and go with the rounding of two
        # tessellations of millions of atoms; the volumes agree
        theirs = particles("Coordination")(data)
        differ = ours.coordination != theirs
        assert differ.sum() <= max(1, 1e-5 * len(theirs))
        assert np.abs(ours.coordination - theirs).max(initial=0) <= 1

    def voronoi_faces(ours, data):
        # a bond for each two faces between two atoms, one of each
        voronoi(ours, data)
        bonds = np.asarray(data.particles.bonds.topology)
        theirs = np.bincount(bonds.ravel(), minlength=len(ours.volume))
        mine = np.bincount(ours.j[ours.j >= 0], minlength=len(ours.volume))
        differ = (mine != theirs) | (ours.coordination != theirs)
        assert differ.sum() <= max(2, 2e-5 * len(theirs))

    def clusters(ours, data):
        theirs = np.bincount(particles("Cluster")(data))
        equal(ours.sizes, np.sort(theirs)[::-1])

    def wigner_seitz(ours, data):
        equal(ours.occupancy, particles("Occupancy")(data))

    def strain(ours, data):
        close(ours.shear, particles("Shear Strain")(data))
        close(ours.volumetric, particles("Volumetric Strain")(data))

    cases = {
        "neighbors": Case(
            "Neighbor list, 5 Å",
            copper,
            _neighbors_mdapy,
            {
                "ovito": ovito(_neighbors_ovito, _neighbors_agree),
                "freud": Other(
                    _freud_neighbors({"r_max": 5.0, "exclude_ii": True}),
                    neighbors_freud,
                ),
            },
        ),
        "nearest": Case(
            "12 nearest neighbors",
            copper,
            _nearest_mdapy,
            {
                "ovito": ovito(_nearest_ovito, _nearest_agree),
                "freud": Other(
                    _freud_neighbors({"num_neighbors": 12, "exclude_ii": True}),
                    nearest_freud,
                ),
            },
        ),
        "cna": Case(
            "Adaptive common neighbor analysis",
            copper,
            analysis(lambda inputs: CommonNeighborAnalysis()),
            {
                "ovito": ovito(
                    pipeline(
                        modifier(
                            "CommonNeighborAnalysisModifier",
                            mode=Member("Mode.AdaptiveCutoff"),
                        )
                    ),
                    structures,
                )
            },
        ),
        "cna-fixed": Case(
            "Common neighbor analysis, cutoff 3.09 Å",
            copper,
            analysis(lambda inputs: CommonNeighborAnalysis(3.086)),
            {
                "ovito": ovito(
                    pipeline(
                        modifier(
                            "CommonNeighborAnalysisModifier",
                            mode=Member("Mode.FixedCutoff"),
                            cutoff=3.086,
                        )
                    ),
                    structures,
                )
            },
        ),
        "ackland-jones": Case(
            "Ackland–Jones analysis",
            copper,
            analysis(lambda inputs: AcklandJonesAnalysis()),
            {"ovito": ovito(pipeline(modifier("AcklandJonesModifier")), structures)},
        ),
        "ptm": Case(
            "Polyhedral template matching",
            copper,
            analysis(lambda inputs: PolyhedralTemplateMatching()),
            {
                "ovito": ovito(
                    pipeline(modifier("PolyhedralTemplateMatchingModifier")), structures
                )
            },
        ),
        "csp": Case(
            "Centro-symmetry parameter, 12 neighbors",
            copper,
            analysis(lambda inputs: CentroSymmetryParameter(12)),
            {
                "ovito": ovito(
                    pipeline(modifier("CentroSymmetryModifier", num_neighbors=12)),
                    lambda ours, data: close(
                        ours.csp, particles("Centrosymmetry")(data)
                    ),
                )
            },
        ),
        "diamond": Case(
            "Diamond structure",
            silicon,
            analysis(lambda inputs: DiamondStructure()),
            {
                "ovito": ovito(
                    pipeline(modifier("IdentifyDiamondModifier")),
                    # OVITO also marks the neighbors of diamond atoms
                    lambda ours, data: structures(ours, data, [0, 1, 0, 0, 2, 0, 0]),
                )
            },
        ),
        "steinhardt": Case(
            "Steinhardt q2–q6, 12 neighbors",
            copper,
            analysis(lambda inputs: SteinhardtParameters(degrees, num_neighbors=12)),
            {
                "ovito": ovito(
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
                "freud": Other(
                    _freud(
                        "order.Steinhardt",
                        list(degrees),
                        compute={
                            "neighbors": {"num_neighbors": 12, "exclude_ii": True}
                        },
                    ),
                    steinhardt_freud,
                ),
            },
        ),
        "steinhardt-voronoi": Case(
            "Steinhardt q2–q6, Voronoi faces weighted by area",
            copper,
            analysis(
                lambda inputs: SteinhardtParameters(
                    degrees, voronoi=True, weighted=True
                )
            ),
            {
                "ovito": ovito(
                    pipeline(
                        modifier(
                            "SteinhardtParametersModifier",
                            neighbor_mode=Member("NeighborMode.Voronoi"),
                            max_l=6,
                        )
                    ),
                    steinhardt_voronoi,
                ),
                "freud": Other(
                    freud_side(_freud_voronoi_steinhardt), steinhardt_voronoi_freud
                ),
            },
        ),
        "rdf": Case(
            "Radial distribution function, 6 Å, 200 bins",
            copper,
            analysis(lambda inputs: RadialDistributionFunction(6.0, bins=200)),
            {
                "ovito": ovito(
                    pipeline(
                        modifier(
                            "CoordinationAnalysisModifier",
                            cutoff=6.0,
                            number_of_bins=200,
                        )
                    ),
                    rdf,
                ),
                "freud": Other(_freud("density.RDF", bins=200, r_max=6.0), rdf_freud),
            },
        ),
        "bond-angles": Case(
            "Bond angle distribution, 3 Å, 180 bins",
            copper,
            analysis(lambda inputs: BondAngleDistribution(cutoff=3.0, bins=180)),
            {
                "ovito": ovito(
                    pipeline(
                        lambda references: (
                            modifier("CreateBondsModifier", cutoff=3.0)(references)
                            + modifier("BondAngleDistributionModifier", bins=180)(
                                references
                            )
                        )
                    ),
                    bond_angles,
                )
            },
        ),
        "voronoi": Case(
            "Voronoi volumes and coordination",
            copper,
            analysis(lambda inputs: VoronoiAnalysis()),
            {
                "ovito": ovito(pipeline(modifier("VoronoiAnalysisModifier")), voronoi),
                "freud": Other(_freud("locality.Voronoi"), voronoi_volumes_freud),
            },
        ),
        "voronoi-faces": Case(
            "Voronoi cells with their faces",
            copper,
            analysis(lambda inputs: VoronoiAnalysis(faces=True)),
            {
                "ovito": ovito(
                    pipeline(modifier("VoronoiAnalysisModifier", generate_bonds=True)),
                    voronoi_faces,
                ),
                "freud": Other(_freud("locality.Voronoi"), voronoi_freud),
            },
        ),
        "clusters": Case(
            "Cluster analysis, 2 Å",
            gas,
            analysis(lambda inputs: ClusterAnalysis(cutoff=2.0)),
            {
                "ovito": ovito(
                    pipeline(
                        modifier(
                            "ClusterAnalysisModifier", cutoff=2.0, sort_by_size=True
                        )
                    ),
                    clusters,
                ),
                "freud": Other(
                    _freud("cluster.Cluster", compute={"neighbors": {"r_max": 2.0}}),
                    clusters_freud,
                ),
            },
        ),
        "wigner-seitz": Case(
            "Wigner–Seitz occupancies",
            damaged,
            analysis(lambda inputs: WignerSeitz(inputs["reference"])),
            {
                "ovito": ovito(
                    pipeline(modifier("WignerSeitzAnalysisModifier")), wigner_seitz
                )
            },
        ),
        "metal": Case(
            "Six analyses of copper, one neighbor search",
            copper,
            together(
                lambda inputs: [
                    CommonNeighborAnalysis(),
                    AcklandJonesAnalysis(),
                    CentroSymmetryParameter(12),
                    SteinhardtParameters(degrees, num_neighbors=12),
                    RadialDistributionFunction(5.0, bins=200),
                    BondAngleDistribution(cutoff=3.0, bins=180),
                ],
                5.0,
            ),
            {
                "ovito": ovito(
                    pipeline(
                        lambda references: [
                            m
                            for make in (
                                modifier(
                                    "CommonNeighborAnalysisModifier",
                                    mode=Member("Mode.AdaptiveCutoff"),
                                ),
                                modifier("AcklandJonesModifier"),
                                modifier("CentroSymmetryModifier", num_neighbors=12),
                                modifier(
                                    "SteinhardtParametersModifier",
                                    neighbor_mode=Member(
                                        "NeighborMode.NearestNeighbors"
                                    ),
                                    num_neighbors=12,
                                    max_l=6,
                                ),
                                modifier(
                                    "CoordinationAnalysisModifier",
                                    cutoff=5.0,
                                    number_of_bins=200,
                                ),
                                modifier("CreateBondsModifier", cutoff=3.0),
                                modifier("BondAngleDistributionModifier", bins=180),
                            )
                            for m in make(references)
                        ]
                    ),
                    metal,
                )
            },
        ),
        "steinhardt-rdf": Case(
            "Steinhardt q2–q6 and the RDF, one neighbor search",
            copper,
            together(
                lambda inputs: [
                    SteinhardtParameters(degrees, num_neighbors=12),
                    RadialDistributionFunction(5.0, bins=200),
                ],
                5.0,
            ),
            {
                "ovito": ovito(
                    pipeline(
                        lambda references: (
                            modifier(
                                "SteinhardtParametersModifier",
                                neighbor_mode=Member("NeighborMode.NearestNeighbors"),
                                num_neighbors=12,
                                max_l=6,
                            )(references)
                            + modifier(
                                "CoordinationAnalysisModifier",
                                cutoff=5.0,
                                number_of_bins=200,
                            )(references)
                        )
                    ),
                    steinhardt_rdf,
                ),
                "freud": Other(freud_side(_freud_steinhardt_rdf), steinhardt_rdf_freud),
            },
        ),
        "atomic-strain": Case(
            "Atomic strain, 3 Å",
            deformed,
            analysis(lambda inputs: AtomicStrain(inputs["reference"], 3.0)),
            {
                "ovito": ovito(
                    pipeline(modifier("AtomicStrainModifier", cutoff=3.0)), strain
                )
            },
        ),
    }
    return {**cases, **_file_cases(), **_builder_cases()}


# ------------------------------------------------------------ one process


def peak_memory() -> int:
    """The largest resident memory of this process so far, in bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _load(program: str) -> None:
    """Imports a program before any system, so that its memory counts as
    that of the libraries, and gives it the threads of the run."""
    if program == "ovito":
        import ovito.data  # noqa: F401
        import ovito.modifiers  # noqa: F401
        import ovito.pipeline  # noqa: F401
    elif program == "freud":
        import freud

        freud.parallel.set_num_threads(int(os.environ.get("MDAPY_NUM_THREADS", "0")))
    elif program == "ase":
        import ase.io  # noqa: F401


def measure(name: str, atoms: int, side: str, repeat: int) -> dict:
    """In a process of its own: whether a program agrees with mdapy (side
    "check-<program>"), or the times of one program (side "mdapy" or the
    program)."""
    case = _cases()[name]
    program = side.removeprefix("check-")
    if program != "mdapy":
        _load(program)
    base = peak_memory()  # the libraries, before any system
    if side.startswith("check-"):
        inputs = case.inputs(atoms)
        other = case.others[program]
        try:
            other.agree(case.mdapy(inputs)(), other.prepare(inputs)(), inputs)
        except AssertionError as error:
            return {"agree": False, "message": str(error)[:2000]}
        size = len(inputs["system"])
        return {"agree": True, "atoms": size, "base": base, "peak": peak_memory()}
    prepare = case.mdapy if side == "mdapy" else case.others[side].prepare
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
        os.environ,
        MDAPY_NUM_THREADS=str(threads),
        OVITO_THREAD_COUNT=str(threads),
        OMP_NUM_THREADS=str(threads),  # Atomsk
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


def version(program: str) -> str:
    """The version of a program, asked in a process of its own."""
    if program == "atomsk":
        found = atomsk()
        if found is None:
            return "not installed"
        out = subprocess.run([found, "--version"], capture_output=True, text=True)
        return out.stdout.strip().rsplit("Version", 1)[-1].strip() or "?"
    code = {
        "mdapy": "import mdapy; print(mdapy.__version__)",
        "ovito": "import ovito; print(ovito.version_string)",
        "freud": "import freud; print(freud.__version__)",
        "ase": "import ase; print(ase.__version__)",
    }[program]
    found = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return found.stdout.strip() or "not installed"


def run(names, sizes, threads, repeat, limit, output, timeout=None) -> list[dict]:
    """Every case at every size: check each program against mdapy, then
    time mdapy and every program that agrees; results appended to output as
    they come."""
    cases = _cases()
    results = []
    machine = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.platform(),
        "cpus": os.cpu_count(),
        "threads": threads,
        "versions": {p: version(p) for p in ("mdapy", *PROGRAMS)},
    }
    for name in names:
        # side -> atoms, peak memory and memory of the libraries alone
        peaks: dict[str, tuple[int, int, int]] = {}
        stopped: set[str] = set()  # sides stopped at a smaller size
        programs = list(cases[name].others)
        for atoms in sizes:
            entry = {
                "case": name,
                "size": atoms,
                "atoms": atoms,
                **machine,
                "check": {},
            }

            def measured(side: str) -> dict:
                last = peaks.get(side)
                if side in stopped:
                    return {"skipped": "stopped at a smaller size"}
                if last is not None and predicted(*last, atoms) > limit:
                    return {"skipped": "would need too much memory"}
                found = spawn(name, atoms, side, repeat, threads, limit, timeout)
                if "peak" in found:
                    peaks[side] = (atoms, found["peak"], found["base"])
                if "stopped" in found:
                    stopped.add(side)
                return found

            agreed = []
            for program in programs:
                entry["check"][program] = measured(f"check-{program}")
                if entry["check"][program].get("agree"):
                    agreed.append(program)
                    entry["atoms"] = entry["check"][program]["atoms"]
            # timed only where a program agrees with mdapy
            for side in ["mdapy", *agreed] if agreed else []:
                entry[side] = measured(side)
            results.append(entry)
            print(line(entry), flush=True)
            with open(output, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            if not agreed:
                break
    return results


def predicted(atoms: int, peak: int, base: int, more: int) -> float:
    """The peak memory for more atoms: that of the libraries, and the rest
    in proportion to the atoms."""
    return base + (peak - base) * more / atoms


def best(side: dict) -> float | None:
    return min(side["times"]) if "times" in side else None


def line(entry: dict) -> str:
    text = f"{entry['case']:20s} {entry['atoms']:>10,d}"
    ours = best(entry.get("mdapy", {}))
    text += "  mdapy " + (f"{ours:9.4f} s" if ours is not None else "-")
    for program, check in entry["check"].items():
        if not check.get("agree"):
            text += f"  {program} " + json.dumps(check)[:200]
            continue
        theirs = best(entry.get(program, {}))
        if theirs is None:
            text += f"  {program} " + json.dumps(entry.get(program))[:200]
        else:
            text += f"  {program} {theirs:9.4f} s"
            if ours:
                text += f" ×{theirs / ours:.2f}"
    return text


# ------------------------------------------------------------ reports


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(row) for row in f if row.strip()]


def extra(side: dict) -> float:
    """The memory a computation took beyond the systems, in MB."""
    return (side["peak"] - side["before"]) / 2**20


def timed(entries: list[dict], program: str) -> list[dict]:
    """The entries where mdapy and the program were both timed."""
    return [
        e
        for e in entries
        if e["check"].get(program, {}).get("agree")
        and best(e.get("mdapy", {})) is not None
        and best(e.get(program, {})) is not None
    ]


def label(entries: list[dict], program: str) -> str:
    """The program and its version, as the runs recorded it."""
    found = {
        e["versions"][program] for e in entries if program in e.get("versions", {})
    }
    return " ".join([PROGRAMS.get(program, program), *sorted(found)])


def outcome(side: dict | None) -> str:
    """Why a program has no time at a size."""
    if not side:
        return "not run"
    for key in ("stopped", "skipped", "failed"):
        if key in side:
            return f"{key}: {side[key]}"
    return "differs" if side.get("agree") is False else "?"


def table(entries: list[dict]) -> str:
    """The results as Markdown tables, one per program, every case at every
    size: the time of each program, how many times faster mdapy is, and the
    memory each computation took; why a program has no time where it has
    none."""
    titles = {name: case.title for name, case in _cases().items()}
    tables = []
    for program in PROGRAMS:
        mine = [e for e in entries if program in e["check"]]
        if not timed(entries, program):
            continue
        name, other = label(entries, program), PROGRAMS[program]
        rows = [
            f"| Case | Atoms | {label(entries, 'mdapy')} (s) | {name} (s) "
            f"| {other} / mdapy | mdapy (MB) | {other} (MB) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for case in titles:
            for entry in (e for e in mine if e["case"] == case):
                ours = best(entry.get("mdapy", {}))
                theirs = best(entry.get(program, {}))
                cells = [titles[case], f"{entry['atoms']:,d}"]
                cells.append(f"{ours:.3f}" if ours is not None else "")
                if theirs is None:
                    check = entry["check"][program]
                    why = outcome(
                        entry.get(program) or (None if check.get("agree") else check)
                    )
                    cells += [
                        why,
                        "",
                        f"{extra(entry['mdapy']):.0f}" if ours else "",
                        "",
                    ]
                else:
                    cells += [
                        f"{theirs:.3f}",
                        f"{theirs / ours:.2f}",
                        f"{extra(entry['mdapy']):.0f}",
                        f"{extra(entry[program]):.0f}",
                    ]
                rows.append("| " + " | ".join(cells) + " |")
        tables.append("\n".join(rows))
    return "\n\n".join(tables)


def plot(entries: list[dict], path: str) -> None:
    """The ratio of the times of each program and mdapy for each case and
    size, one panel per program."""
    from mdapy.visualization import plot as plots

    titles = {name: case.title for name, case in _cases().items()}
    panels = [(p, timed(entries, p)) for p in PROGRAMS if timed(entries, p)]
    counts = [len({e["case"] for e in done}) for _, done in panels]
    sizes = sorted({e["size"] for _, done in panels for e in done})
    with plots.style():
        fig, axes = plots.figure(
            "double",
            30 * len(panels) + 6.5 * sum(counts),
            nrows=len(panels),
            gridspec_kw={"height_ratios": [c + 4 for c in counts]},
        )
        axes = np.atleast_1d(axes)
        for ax, (program, done) in zip(axes, panels):
            names = [n for n in titles if any(e["case"] == n for e in done)]
            ratios = {
                (e["case"], e["size"]): best(e[program]) / best(e["mdapy"])
                for e in done
            }
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
            ticks = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200)
            low, high = ax.get_xlim()
            ticks = [t for t in ticks if low <= t <= high]
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
            ax.minorticks_off()
            ax.set_yticks(np.arange(len(names)), [titles.get(n, n) for n in names])
            ax.invert_yaxis()
            name = PROGRAMS[program]
            ax.set_title(f"{label(entries, 'mdapy')} against {label(entries, program)}")
            ax.set_xlabel(f"time of {name} / time of mdapy")
            # where mdapy is better, and worse, under the axis
            ax.annotate(
                "mdapy better →",
                (1.0, 0.0),
                xycoords="axes fraction",
                xytext=(0, -26),
                textcoords="offset points",
                ha="right",
                va="top",
            )
            ax.annotate(
                "← mdapy worse",
                (0.0, 0.0),
                xycoords="axes fraction",
                xytext=(0, -26),
                textcoords="offset points",
                ha="left",
                va="top",
            )
        axes[0].legend(frameon=False, loc="best")  # the same sizes below
        plots.save(fig, path.rsplit(".", 1)[0], formats=(path.rsplit(".", 1)[1],))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    runs = commands.add_parser("run", help="check and time the cases")
    runs.add_argument("cases", nargs="*", help="default: every case of the groups")
    runs.add_argument("--group", choices=GROUPS, action="append", help="default: all")
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
    child.add_argument("side")
    child.add_argument("repeat", type=int)
    options = parser.parse_args(argv)
    if options.command == "_measure":
        result = measure(options.case, options.atoms, options.side, options.repeat)
        print(json.dumps(result))
    elif options.command == "run":
        groups = options.group or GROUPS
        names = options.cases or [n for n, c in _cases().items() if c.group in groups]
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
