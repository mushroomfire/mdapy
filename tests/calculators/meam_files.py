# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""MEAM library and parameter files for the tests, of published parameters
(Baskes, Phys. Rev. B 46, 2727 (1992), and later) and options chosen to take
every branch."""

# name: lattice z atomic-number mass / alpha b0 b1 b2 b3 alat esub asub /
# t0 t1 t2 t3 rozero ibar
LIBRARY = {
    "Cu": ("fcc", 12, 29, 63.54, 5.106, 3.634, 2.20, 6.0, 2.20, 3.62, 3.54, 1.07, 1, 3.14, 2.49, 2.95, 1.0, 3),
    "Ni": ("fcc", 12, 28, 58.6934, 4.99, 2.45, 2.20, 6.0, 2.20, 3.52, 4.45, 0.94, 1, 3.57, 1.60, 3.70, 1.0, 3),
    "Al": ("fcc", 12, 13, 26.98, 4.64, 2.04, 3.00, 6.0, 1.50, 4.05, 3.353, 1.07, 1, 3.05, 0.51, 7.75, 1.0, 3),
    "Fe": ("bcc", 8, 26, 55.847, 5.65, 3.50, 2.00, 1.0, 1.00, 2.86, 4.29, 0.89, 1, 3.94, 4.12, -1.50, 1.0, 3),
    "Mo": ("bcc", 8, 42, 95.94, 5.85, 4.48, 1.00, 1.0, 1.00, 3.15, 6.81, 0.99, 1, 2.30, 1.90, -1.60, 1.0, 4),
    "Co": ("hcp", 12, 27, 58.933, 5.12, 3.50, 0.00, 0.0, 4.00, 2.50, 4.41, 0.90, 1, 3.00, 5.00, -1.00, 1.0, 3),
    "Si": ("dia", 4, 14, 28.086, 4.87, 4.80, 4.80, 4.8, 4.80, 5.431, 4.63, 1.00, 1, 3.30, 5.105, -0.80, 1.0, 1),
    "C": ("dia", 4, 6, 12.0111, 4.38, 4.10, 4.20, 5.0, 3.00, 3.567, 7.37, 1.18, 1, 5.00, 9.34, -1.00, 2.25, 1),
    "Bi": ("sc", 6, 83, 208.98, 5.31, 4.25, 6.90, 1.0, 0.10, 3.24, 2.18, 0.75, 1, 1.00, -0.29, 0.50, 1.0, 0),
    "H": ("dim", 1, 1, 1.0079, 2.96, 2.96, 3.00, 3.0, 3.00, 0.74, 2.235, 2.50, 1, 0.20, -0.10, 0.00, 1.0, -5),
    # an entry of no element's name, of copper
    "CuX": ("fcc", 12, 29, 63.54, 5.106, 3.634, 2.20, 6.0, 2.20, 3.62, 3.54, 1.07, 1, 3.14, 2.49, 2.95, 1.0, 0),
}


def write_library(path, names=None):
    names = list(LIBRARY) if names is None else names
    lines = ["# elt lat z ielement atwt", "# alpha b0 b1 b2 b3 alat esub asub", "# t0 t1 t2 t3 rozero ibar"]
    for name in names:
        v = LIBRARY[name]
        lines.append(f"'{name}' '{v[0]}' {v[1]} {v[2]} {v[3]}")
        lines.append(" ".join(str(x) for x in v[4:12]))
        lines.append(" ".join(str(x) for x in v[12:]))
    path.write_text("\n".join(lines) + "\n")
    return path


def write_parameters(path, lines):
    path.write_text("# MEAM parameters for the tests\n" + "\n".join(lines) + "\n")
    return path


# alloys and options, each a set of entries and the lines of its parameter
# file: every option of LAMMPS that MDAPY takes
CASES = {
    "copper of the library alone": (["Cu"], None, "fcc", 3.62),
    "copper with second neighbours": (
        ["Cu"],
        ["rc = 4.8", "nn2(1,1) = 1", "Cmin(1,1,1) = 0.8", "erose_form = 2", "zbl(1,1) = 0"],
        "fcc",
        3.62,
    ),
    "copper and nickel, l12": (
        ["Cu", "Ni"],
        [
            "lattce(1,2) = 'l12'", "Ec(1,2) = 4.2", "re(1,2) = 2.52", "alpha(1,2) = 5.0",
            "Cmin(1,1,2) = 1.5", "Cmax(1,2,1) = 2.6", "rc = 4.5", "delr = 0.2",
        ],
        "fcc",
        3.57,
    ),  # fmt: skip
    "nickel and aluminium, l12 of second neighbours, ialloy 1": (
        ["Ni", "Al"],
        [
            "lattce(1,2) = 'l12'", "Ec(1,2) = 4.65", "delta(1,2) = 0.1", "alpha(1,2) = 4.8",
            "nn2(1,1) = 1", "nn2(1,2) = 1", "nn2(2,2) = 1", "Cmin(1,1,1) = 0.81",
            "Cmin(2,2,2) = 0.49", "Cmin(1,1,2) = 0.64", "Cmin(1,2,1) = 0.9",
            "ialloy = 1", "rc = 4.8", "attrac(1,2) = 0.05", "repuls(1,2) = 0.05",
        ],
        "fcc",
        3.6,
    ),  # fmt: skip
    "iron and copper, b2 of second neighbours, ialloy 2, erose_form 1": (
        ["Fe", "Cu"],
        [
            "lattce(1,2) = 'b2'", "Ec(1,2) = 3.8", "re(1,2) = 2.55", "nn2(1,1) = 1",
            "nn2(1,2) = 1", "nn2(2,2) = 1", "Cmin(1,1,1) = 0.36", "Cmin(1,2,1) = 0.5",
            "ialloy = 2", "erose_form = 1", "attrac(1,1) = 0.05", "repuls(1,1) = 0.1",
            "rc = 5.0", "augt1 = 0",
        ],
        "bcc",
        2.9,
    ),  # fmt: skip
    "iron and molybdenum, b1, mixture_ref_t, emb_lin_neg, bkgd_dyn": (
        ["Fe", "Mo"],
        [
            "lattce(1,2) = 'b1'", "Ec(1,2) = 5.0", "re(1,2) = 2.6", "mixture_ref_t = 1",
            "emb_lin_neg = 1", "bkgd_dyn = 1", "rc = 4.5",
        ],
        "bcc",
        3.0,
    ),  # fmt: skip
    "silicon and carbon, dia, screening beyond rc": (
        ["Si", "C"],
        [
            "lattce(1,2) = 'dia'", "Ec(1,2) = 6.4325", "alpha(1,2) = 4.37", "re(1,2) = 1.8878",
            "rho0(2) = 2.25", "rc = 4.0", "Cmax(1,2,1) = 4.0", "Cmax(1,2,2) = 4.0",
            "Cmax(2,2,1) = 4.0", "Cmax(1,1,2) = 4.0", "Cmin(1,2,1) = 1.0",
        ],
        "zincblende",
        4.36,
    ),  # fmt: skip
    "cobalt, hcp": (["Co"], ["rc = 4.2"], "hcp", 2.5),
    "bismuth, sc, ibar 0 with gsmooth_factor": (
        ["Bi"],
        ["rc = 5.0", "gsmooth_factor = 10", "erose_form = 2"],
        "sc",
        3.24,
    ),
    "copper of ibar 0": (["CuX"], ["rc = 4.4"], "fcc", 3.62),
    "hydrogen, dim, ibar -5": (["H"], ["rc = 3.0"], "fcc", 2.4),
}
