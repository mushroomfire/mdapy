# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""OUTCAR files written in the layout of VASP 5 for the tests, from known
frames."""

import numpy as np


def frames(count=3, n=4, seed=0):
    """Known frames: lattice, positions, forces, energy, virial."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        lattice = np.diag([5.0, 5.2, 5.4]) + rng.normal(scale=0.2, size=(3, 3))
        positions = rng.random((n, 3)) @ lattice
        forces = rng.normal(scale=0.3, size=(n, 3))
        energy = -3.5 * n + rng.normal(scale=0.1)
        v = rng.normal(scale=0.5, size=(3, 3))
        out.append((lattice, positions, forces, energy, 0.5 * (v + v.T)))
    return out


def outcar_text(
    steps, symbols=("Pd", "Ag"), counts=(0, 4), converged=None, stress=True
):
    """The text of an OUTCAR of ionic steps; converged marks each electronic
    loop that reached EDIFF (all by default)."""
    converged = [True] * len(steps) if converged is None else converged
    n = sum(counts)
    lines = [
        " vasp.5.4.4.18Apr17-6-g9f103f2a35 (build Jun 17 2019) complex",
        " ",
        " executed on             LinuxIFC date 2026.10.22  21:00:00",
        "",
        " INCAR:",
    ]
    lines += [f" POTCAR:    PAW_PBE {s} 04Jan2005" for s in symbols]
    lines += [f" POTCAR:    PAW_PBE {s} 04Jan2005" for s in symbols]
    lines += [
        "",
        f"   number of dos      NEDOS =    301   number of ions     NIONS = {n:7d}",
        "   ions per type =  " + "".join(f"{c:4d}" for c in counts),
        f"   ISIF   =      {3 if stress else 0}    stress and relaxation",
        "",
    ]
    first = steps[0][0]
    lines += _lattice(first)
    for (lattice, positions, forces, energy, virial), ok in zip(steps, converged):
        lines += [
            "",
            "----------------------------------------- Iteration    1(   1)  ---------------------------------------",
            "",
        ]
        if ok:
            lines += [
                "------------------------ aborting loop because EDIFF is reached ----------------------------------------",
                "",
            ]
        if stress:
            xx, yy, zz = virial[0, 0], virial[1, 1], virial[2, 2]
            xy, yz, zx = virial[0, 1], virial[1, 2], virial[2, 0]
            volume = abs(np.linalg.det(lattice))
            kb = -np.array([xx, yy, zz, xy, yz, zx]) / volume * 1602.1766208 * -1
            lines += [
                "  FORCE on cell =-STRESS in cart. coord.  units (eV):",
                "  Direction    XX          YY          ZZ          XY          YZ          ZX",
                "  --------------------------------------------------------------------------------------",
                "  -------------------------------------------------------------------------------------",
                "  Total  " + "".join(f"{x:12.5f}" for x in (xx, yy, zz, xy, yz, zx)),
                "  in kB  " + "".join(f"{x:12.5f}" for x in kb),
                "  external pressure =        1.75 kB  Pullay stress =        0.00 kB",
                "",
            ]
        lines += [" VOLUME and BASIS-vectors are now :", " -----------------------------------------------------------------------------",
                  "  energy-cutoff  :      400.00", f"  volume of cell : {abs(np.linalg.det(lattice)):11.2f}"]  # fmt: skip
        lines += _lattice(lattice)[2:]
        lines += [
            "",
            " POSITION                                       TOTAL-FORCE (eV/Angst)",
            " -----------------------------------------------------------------------------------",
        ]
        lines += [
            "   "
            + "".join(f"{x:13.5f}" for x in p)
            + "   "
            + "".join(f"{f:14.6f}" for f in force)
            for p, force in zip(positions, forces)
        ]
        lines += [
            " -----------------------------------------------------------------------------------",
            "    total drift:                                0.000000     -0.000000      0.000000",
            "",
            "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
            "  ---------------------------------------------------",
            f"  free  energy   TOTEN  = {energy:18.8f} eV",
            "",
            f"  energy  without entropy= {energy - 0.007:18.8f}  energy(sigma->0) = {energy - 0.003:18.8f}",
            "",
        ]
    lines += ["", " General timing and accounting informations for this job:", ""]
    return "\n".join(lines) + "\n"


def _lattice(lattice):
    inverse = np.linalg.inv(lattice).T
    return [
        "  energy-cutoff  :      400.00",
        f"  volume of cell : {abs(np.linalg.det(lattice)):11.2f}",
        "      direct lattice vectors                 reciprocal lattice vectors",
    ] + ["   " + "".join(f"{x:13.9f}" for x in row) + "   " + "".join(f"{x:13.9f}" for x in inv) for row, inv in zip(lattice, inverse)] + [""]  # fmt: skip
