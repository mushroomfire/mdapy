# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The verification of MDAPY's plane-wave DFT against the all-electron
equations of state of Bosoni et al., Nat. Rev. Phys. 6, 45 (2024), of the
unaries (FCC, BCC, SC, Diamond) and of the oxides (X2O, XO, X2O3, XO2, X2O5,
XO3), with their protocol (verification-PBE-v1): PBE without spin, Fermi-Dirac smearing
of 0.0045 Ry, the free energy at 7 volumes from 94 % to 106 % of the central
one, a Gamma-centered mesh of at most 0.06 1/Å (with 2 pi) between k-points
on the smallest volume, the same mesh for all; the Birch-Murnaghan fit of the
Delta project (a cubic in V^-2/3); their metrics nu and epsilon, against the
average of WIEN2k and FLEUR and against Quantum ESPRESSO with the same
pseudopotentials (PseudoDojo 0.4, PBE, scalar relativistic, standard), at the
high cutoff of their hints.

    python benchmarks/dft_verification.py prepare FOLDER
    python benchmarks/dft_verification.py run FOLDER Fe BCC [Fe XO ...] [--cutoff HA] [--library sssp] [--hint normal]
    python benchmarks/dft_verification.py report FOLDER [OUTPUT.md] [--library sssp] [--hint normal]
    python benchmarks/dft_verification.py figure FOLDER IMAGE [unaries|oxides] [--library sssp] [--hint normal]

prepare downloads into FOLDER the data and scripts of the paper from the
Materials Cloud Archive (doi:10.24435/materialscloud:s4-3h, CC BY 4.0) and
the PseudoDojo table with its hints; run writes FOLDER/results/El-CONF.json,
the free energies of the volumes and what they were computed with (with
--cutoff, at that cutoff in hartree into FOLDER/results-HA, for the tests of
convergence); report fits them and prints the table of the comparisons
(markdown); figure draws nu of each structure (of one family, or all)
against QE and against the all-electron codes beside QE's from
FOLDER/summary.json. The cutoff of an oxide is the higher of the hints of
its two elements. With --library sssp, the pseudopotentials of SSSP 1.3 PBE
precision (norm-conserving, ultrasoft and PAW of several libraries) at the
cutoffs it recommends, the highest of the elements of the structure, against
Quantum ESPRESSO with the same, into FOLDER/results-sssp and
FOLDER/summary-sssp.json. With --hint normal, PseudoDojo's normal hints, as
the published Quantum ESPRESSO took them (the default stringency of the
families of aiida-pseudo), into FOLDER/results-normal and
FOLDER/summary-normal.json.
"""

import hashlib
import json
import math
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ARCHIVE = "https://archive.materialscloud.org/api/records/yf0rj-w3r97/files/ACWF-verification-data-and-scripts.zip/content"
DOJO = "https://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard_upf.tgz"
HINTS = "https://raw.githubusercontent.com/PseudoDojo/ONCVPSP-PBE-PDv0.4/master/standard.djson"
SSSP = "https://archive.materialscloud.org/api/records/rcyfm-68h65/files/SSSP_1.3.0_PBE_precision"
CODES = {
    "dojo": "quantum_espresso-PseudoDojo-0.4-PBE-SR-standard-upf",
    "sssp": "quantum_espresso-SSSP-1.3-PBE-precision",
}
BOHR = 0.529177210903  # Å
HARTREE = 27.211386245988  # eV
RY = 0.5 * HARTREE
SCALES = (0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06)
SPACING = 0.06  # 1/Å, with 2 pi
SMEARING = 0.0045 / 2  # hartree, 0.0045 Ry
CONFIGURATIONS = ("FCC", "BCC", "SC", "Diamond")
OXIDES = ("X2O", "XO", "X2O3", "XO2", "X2O5", "XO3")
FAMILIES = {"unaries": CONFIGURATIONS, "oxides": OXIDES}


def _outputs(library: str, hint: str) -> tuple[str, str]:
    """The folder of the results and the name of the summary."""
    tag = library if library != "dojo" else (None if hint == "high" else hint)
    return (
        ("results", "summary.json")
        if tag is None
        else (f"results-{tag}", f"summary-{tag}.json")
    )


def download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=600) as response:
            path.write_bytes(response.read())
    return path


def prepare(folder: Path) -> None:
    import tarfile

    archive = download(ARCHIVE, folder / "downloads" / "acwf.zip")
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            keep = (
                "reference-dataset/" in name
                or "code-data/results-unaries" in name
                or "code-data/results-oxides" in name
                or "xsfs-unaries-verification-PBE-v1/" in name
                or "xsfs-oxides-verification-PBE-v1/" in name
            ) and not name.startswith("__MACOSX")
            if keep and not name.endswith("/"):
                target = folder / "acwf" / Path(name).name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))
    tgz = download(DOJO, folder / "downloads" / "dojo-upf.tgz")
    with tarfile.open(tgz) as t:
        for member in t.getmembers():
            if member.isfile() and member.name.endswith(".upf"):
                target = folder / "pseudo-dojo" / Path(member.name).name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(t.extractfile(member).read())
    download(HINTS, folder / "pseudo-dojo" / "standard.djson")
    sssp = folder / "sssp"
    download(SSSP + ".json/content", sssp / "SSSP_1.3.0_PBE_precision.json")
    with tarfile.open(
        download(SSSP + ".tar.gz/content", folder / "downloads" / "sssp.tar.gz")
    ) as t:
        for member in t.getmembers():
            if member.isfile() and member.name.lower().endswith(".upf"):
                (sssp / Path(member.name).name).write_bytes(
                    t.extractfile(member).read()
                )
    print(
        f"prepared {folder}: {len(list((folder / 'pseudo-dojo').glob('*.upf')))} pseudopotentials"
    )


def read_xsf(path: Path):
    """The cell (rows, Å), the species of the atoms as written (atomic numbers
    in the files of the unaries, symbols in those of the oxides) and the
    fractional positions."""
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    tags = [row[0] for row in lines]
    i = tags.index("PRIMVEC")
    cell = np.array([[float(x) for x in lines[i + k]] for k in (1, 2, 3)])
    j = tags.index("PRIMCOORD")
    n = int(lines[j + 1][0])
    rows = lines[j + 2 : j + 2 + n]
    species = [r[0] for r in rows]
    cart = np.array([[float(x) for x in r[1:4]] for r in rows])
    return cell, species, cart @ np.linalg.inv(cell)


def mesh_of(cell: np.ndarray) -> tuple[int, int, int]:
    """The Gamma-centered mesh of at most SPACING between k-points."""
    b = 2 * math.pi * np.linalg.inv(cell).T
    return tuple(
        max(1, math.ceil(round(np.linalg.norm(row) / SPACING, 5))) for row in b
    )


def run(
    folder: Path,
    element: str,
    configuration: str,
    cutoff: float | None = None,
    library: str = "dojo",
    hint: str = "high",
) -> None:
    from mdapy.calculators._dft import scf
    from mdapy.calculators._dft.upf import read_upf

    cell, species, fractions = read_xsf(
        folder / "acwf" / f"{element}-{configuration}.xsf"
    )
    # one pseudopotential per species, in the order they first appear
    names = [element if s.isdigit() else s for s in dict.fromkeys(species)]
    kinds = [names.index(element if s.isdigit() else s) for s in species]
    density = None
    if library == "sssp":
        sssp = json.loads(
            (folder / "sssp" / "SSSP_1.3.0_PBE_precision.json").read_text()
        )
        paths = [folder / "sssp" / sssp[name]["filename"] for name in names]
        results = folder / "results-sssp"
        # the recommended cutoffs in Ry, the highest of the elements, in hartree
        cutoff = max(sssp[name]["cutoff_wfc"] for name in names) / 2
        density = max(sssp[name]["cutoff_rho"] for name in names) / 2
    else:
        paths = [folder / "pseudo-dojo" / f"{name}.upf" for name in names]
        hints = json.loads((folder / "pseudo-dojo" / "standard.djson").read_text())[
            "pseudos_metadata"
        ]
        results = folder / (
            _outputs(library, hint)[0] if cutoff is None else f"results-{cutoff:g}"
        )
        if cutoff is None:  # hartree
            cutoff = max(float(hints[name]["hints"][hint]["ecut"]) for name in names)
    pseudos = [read_upf(path) for path in paths]
    volume = abs(np.linalg.det(cell))
    mesh = mesh_of(cell * SCALES[0] ** (1 / 3))
    out = {
        "element": element, "configuration": configuration, "atoms": len(species),
        "cutoff_hartree": cutoff, "density_cutoff_hartree": density, "mesh": mesh, "library": library, "smearing": "fermi-dirac", "sigma_hartree": SMEARING,
        "functional": "pbe_pw", "pseudopotentials": [path.name for path in paths],
        "sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
        "central_volume": volume, "volumes": [], "free_energies_ev": [], "steps": [], "seconds": [],
        "operations": [], "kpoints": [],
    }  # fmt: skip
    import mdapy

    out["mdapy"] = mdapy.__version__
    for scale in SCALES:
        start = time.time()
        r = scf.run(
            cell * scale ** (1 / 3) / BOHR, fractions, kinds, pseudos, cutoff, kgrid=mesh,
            smearing="fermi-dirac", sigma=SMEARING, functional="pbe_pw", density_cutoff=density,
            energy_tolerance=1e-9 * len(species), density_tolerance=1e-11, max_iterations=200, stress=False,
        )  # fmt: skip
        if not r["converged"]:
            raise RuntimeError(f"{element} {configuration} at {scale}: no convergence")
        out["volumes"].append(volume * scale)
        out["free_energies_ev"].append(r["energy"] * HARTREE)
        out["steps"].append(int(r["iterations"]))
        out["seconds"].append(time.time() - start)
        out["operations"].append(int(r["operations"]))
        out["kpoints"].append(len(r["kpoints"]))
        print(f"{element} {configuration} {scale:.2f}: {r['energy'] * HARTREE:.8f} eV, {r['iterations']} steps, "
              f"{len(r['kpoints'])} k, {time.time() - start:.1f} s", flush=True)  # fmt: skip
    target = results / f"{element}-{configuration}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1))


def fit(volumes, energies):
    """V0, E0, B0 (eV/Å^3) and B1 of the Birch-Murnaghan equation of state, as
    the Delta project fits it: E a cubic in x = V^-2/3."""
    v, e = np.asarray(volumes), np.asarray(energies)
    p = np.poly1d(np.polyfit(v ** (-2 / 3), e, 3))
    d1, d2, d3 = p.deriv(1), p.deriv(2), p.deriv(3)
    roots = [
        x.real for x in d1.roots if abs(x.imag) < 1e-8 and x.real > 0 and d2(x.real) > 0
    ]
    if not roots:
        raise ValueError("no minimum of the fit")
    x = roots[0]
    v0 = x ** (-3 / 2)
    dv2 = 4 / 9 * x**5 * d2(x)
    dv3 = -20 / 9 * x ** (13 / 2) * d2(x) - 8 / 27 * x ** (15 / 2) * d3(x)
    return v0, p(x), dv2 / x ** (3 / 2), -1 - x ** (-3 / 2) * dv3 / dv2


def birch_murnaghan(v, v0, b0, b1):
    eta = (v0 / v) ** (2 / 3)
    return 9 * v0 * b0 / 16 * ((eta - 1) ** 3 * b1 + (eta - 1) ** 2 * (6 - 4 * eta))


def nu(a, b) -> float:
    """100 times the length of the relative differences of V0, B0 / 20 and
    B1 / 400."""
    rel = [2 * (x - y) / (x + y) for x, y in zip(a, b)]
    return 100 * math.sqrt(rel[0] ** 2 + (rel[1] / 20) ** 2 + (rel[2] / 400) ** 2)


def epsilon(a, b) -> float:
    """The root of <(E_a - E_b)^2> / sqrt(var E_a var E_b) over 94 % to 106 %
    of the mean V0, both curves from their minimum."""
    low, high = 0.94 * (a[0] + b[0]) / 2, 1.06 * (a[0] + b[0]) / 2
    t, w = np.polynomial.legendre.leggauss(400)
    v = low + (high - low) * (t + 1) / 2
    w = w / 2
    ea, eb = birch_murnaghan(v, *a), birch_murnaghan(v, *b)

    def mean(f):
        return float((w * f).sum())

    var = (mean(ea**2) - mean(ea) ** 2) * (mean(eb**2) - mean(eb) ** 2)
    return math.sqrt(mean((ea - eb) ** 2) / math.sqrt(var))


def report(
    folder: Path, output: Path | None = None, library: str = "dojo", hint: str = "high"
) -> None:
    acwf = folder / "acwf"

    def data(family: str, code: str) -> dict:
        path = acwf / f"results-{family}-verification-PBE-v1-{code}.json"
        return json.loads(path.read_text())["BM_fit_data"] if path.exists() else {}

    references = {f: data(f, "AE-average") for f in FAMILIES}
    espressos = {f: data(f, CODES[library]) for f in FAMILIES}
    rows, summary = [], []
    results, name = _outputs(library, hint)
    for path in sorted((folder / results).glob("*.json")):
        r = json.loads(path.read_text())
        oxide = r["configuration"] in OXIDES
        family = "oxides" if oxide else "unaries"
        key = (
            f"{r['element']}-{r['configuration']}"
            if oxide
            else f"{r['element']}-X/{r['configuration']}"
        )
        reference, espresso = references[family], espressos[family]
        v0, _, b0, b1 = fit(r["volumes"], r["free_energies_ev"])
        ours = (v0, b0, b1)
        ae = reference[key]
        ae = (ae["min_volume"], ae["bulk_modulus_ev_ang3"], ae["bulk_deriv"])
        qe = espresso.get(key)
        qe = (
            None
            if qe is None
            else (qe["min_volume"], qe["bulk_modulus_ev_ang3"], qe["bulk_deriv"])
        )
        entry = {
            "key": key, "V0": v0, "B0_GPa": b0 * 160.21766208, "B1": b1,
            "nu_ae": nu(ours, ae), "epsilon_ae": epsilon(ours, ae),
            "nu_qe": None if qe is None else nu(ours, qe), "epsilon_qe": None if qe is None else epsilon(ours, qe),
            "qe_nu_ae": None if qe is None else nu(qe, ae), "cutoff": r["cutoff_hartree"], "mesh": r["mesh"],
            "seconds": sum(r["seconds"]),
        }  # fmt: skip
        summary.append(entry)
        rows.append(
            f"| {r['element']} | {r['configuration']} | {v0:.3f} | {ae[0]:.3f} | {entry['B0_GPa']:.1f} | "
            f"{ae[1] * 160.21766208:.1f} | {b1:.2f} | {ae[2]:.2f} | {entry['nu_ae']:.3f} | {entry['epsilon_ae']:.3f} | "
            + (
                "— | — | — |"
                if qe is None
                else f"{entry['qe_nu_ae']:.3f} | {entry['nu_qe']:.3f} | {entry['epsilon_qe']:.3f} |"
            )
        )
    head = (
        "| Element | Structure | V0 (Å³) | AE V0 | B0 (GPa) | AE B0 | B1 | AE B1 | ν vs AE | ε vs AE "
        "| QE ν vs AE | ν vs QE | ε vs QE |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    text = head + "\n" + "\n".join(rows) + "\n"
    print(text)
    (folder / name).write_text(json.dumps(summary, indent=1))
    if output is not None:
        output.write_text(text)


def figure(
    folder: Path,
    image: Path,
    family: str | None = None,
    library: str = "dojo",
    hint: str = "high",
) -> None:
    from mdapy.visualization import plot

    name = _outputs(library, hint)[1]
    summary = json.loads((folder / name).read_text())
    shapes = "os^Dvp"
    markers = {
        c: shapes[i % len(shapes)]
        for configurations in FAMILIES.values()
        for i, c in enumerate(configurations)
    }
    wanted = FAMILIES[family] if family else CONFIGURATIONS + OXIDES
    with plot.style():
        fig, (left, right) = plot.figure("double", 70, nrows=1, ncols=2)
        for conf in wanted:
            rows = [
                x for x in summary if x["key"].split("/")[-1].split("-")[-1] == conf
            ]
            if not rows:
                continue
            left.plot([x["epsilon_qe"] for x in rows], [x["nu_qe"] for x in rows], markers[conf],
                      ms=3, mfc="none", label=conf)  # fmt: skip
            right.plot([x["qe_nu_ae"] for x in rows], [x["nu_ae"] for x in rows], markers[conf],
                       ms=3, mfc="none", label=conf)  # fmt: skip
        left.axhline(0.1, ls="--", c="0.5", lw=0.8)
        left.axvline(0.06, ls="--", c="0.5", lw=0.8)
        left.set(
            xscale="log",
            yscale="log",
            xlabel=r"$\varepsilon$ against QE",
            ylabel=r"$\nu$ against QE",
        )
        line = np.array([1e-3, 5.0])
        right.plot(line, line, c="0.5", lw=0.8)
        right.set(xscale="log", yscale="log", xlabel=r"$\nu$ of QE against all-electron",
                  ylabel=r"$\nu$ of MDAPY against all-electron")  # fmt: skip
        left.legend()
        for ax, label in zip((left, right), "ab"):
            plot.panel_label(ax, label)
        plot.save(fig, str(image.with_suffix("")), formats=("png",))


def _option(args: list, name: str) -> tuple[list, str | None]:
    """args without --name VALUE, and VALUE (None without it)."""
    if name not in args:
        return args, None
    at = args.index(name)
    return args[:at] + args[at + 2 :], args[at + 1]


if __name__ == "__main__":
    command, folder = sys.argv[1], Path(sys.argv[2])
    rest, library = _option(sys.argv[3:], "--library")
    library = library or "dojo"
    rest, hint = _option(rest, "--hint")
    hint = hint or "high"
    if command == "prepare":
        prepare(folder)
    elif command == "run":
        pairs, cutoff = _option(rest, "--cutoff")
        for element, configuration in zip(pairs[::2], pairs[1::2]):
            run(
                folder,
                element,
                configuration,
                None if cutoff is None else float(cutoff),
                library,
                hint,
            )
    elif command == "report":
        report(folder, Path(rest[0]) if rest else None, library, hint)
    elif command == "figure":
        figure(folder, Path(rest[0]), rest[1] if len(rest) > 1 else None, library, hint)
    else:
        raise SystemExit(__doc__)
