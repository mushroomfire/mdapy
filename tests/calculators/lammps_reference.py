# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""What LAMMPS computes for a system with a pair style, atom by atom, to
compare the native calculators with: the energy, the energy of each atom
(compute pe/atom), the forces, the virial (compute pressure) and the virial
of each atom (compute stress/atom), in eV and eV/Å."""

from pathlib import Path

import numpy as np

import mdapy

# LAMMPS writes the virial as pressure (bar) and per-atom stress (bar Å^3),
# converted with its own constant for metal units, nktv2p = 1.6021765e6 bar
# per eV/Å^3 (the exact value is 1.602176634e6).
BAR_A3_TO_EV = 1 / 1.6021765e6


def has_style(style: str) -> bool:
    """Whether the installed LAMMPS has a pair style."""
    import lammps

    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        return lmp.has_style("pair", style)
    finally:
        lmp.close()


def _by_id(lmp, values):
    n = lmp.extract_global("nlocal")
    ids = np.array(lmp.numpy.extract_atom("id")[:n], copy=True)
    return np.array(values[:n], copy=True)[np.argsort(ids)]


def reference(
    system: mdapy.System, elements: list[str], pair: str, folder: Path
) -> dict[str, np.ndarray | float]:
    """What LAMMPS computes for the system with the commands ``pair``
    (pair_style, pair_coeff and any pair_modify), its atom types the
    elements in that order."""
    import lammps

    data = Path(folder) / "reference.data"
    if not all(system.box.pbc):
        # LAMMPS reads only atoms inside the box: along a free direction, a
        # box around them all, which changes nothing else
        matrix = system.box.matrix.copy()
        origin = system.box.origin.copy()
        positions = system["positions"]
        for k in np.flatnonzero(~np.asarray(system.box.pbc)):
            if np.count_nonzero(matrix[k]) != 1 or np.count_nonzero(matrix[:, k]) != 1:
                raise ValueError(
                    "a free direction must be along an axis of an orthogonal box"
                )
            low, high = positions[:, k].min() - 5.0, positions[:, k].max() + 5.0
            origin[k], matrix[k, k] = low, high - low
        system = system.with_box(mdapy.Box(matrix, origin=origin, pbc=system.box.pbc))
    mdapy.io.write(data, system, elements=elements)
    boundary = " ".join("p" if p else "s" for p in system.box.pbc)
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_string(
            f"""
units metal
atom_style atomic
atom_modify map array
boundary {boundary}
read_data {data}
{pair}
compute pe all pe/atom
compute w all stress/atom NULL virial
compute p all pressure NULL virial
thermo_style custom step pe c_p[1]
run 0
"""
        )
        energy = lmp.get_thermo("pe")
        atomic = _by_id(lmp, lmp.numpy.extract_compute("pe", 1, 1))
        forces = _by_id(lmp, lmp.numpy.extract_atom("f"))
        stress = _by_id(lmp, lmp.numpy.extract_compute("w", 1, 2))
        # the global vector of the pressure compute: xx yy zz xy xz yz
        pressure = np.array(lmp.numpy.extract_compute("p", 0, 1), copy=True)
        volume = lmp.get_thermo("vol")
    finally:
        lmp.close()
    xx, yy, zz, xy, xz, yz = pressure
    virial = (
        np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) * volume * BAR_A3_TO_EV
    )
    s = stress  # xx yy zz xy xz yz, the negative of the virial of each atom
    atomic_virials = (
        -np.stack([s[:, [0, 3, 4]], s[:, [3, 1, 5]], s[:, [4, 5, 2]]], axis=1)
        * BAR_A3_TO_EV
    )
    return {
        "energy": energy,
        "atomic_energies": atomic,
        "forces": forces,
        "virial": virial,
        "atomic_virials": atomic_virials,
    }


def assert_matches(result, expected, *, energy=1e-12, forces=1e-10, virials=1e-9):
    """The result of a native calculator against LAMMPS: relative tolerances
    of the energies, forces and virials, absolute ones of the same size
    times the scale of each."""
    scale = max(1.0, float(np.abs(expected["atomic_energies"]).max()))
    np.testing.assert_allclose(result.energy, expected["energy"], rtol=energy)
    np.testing.assert_allclose(
        result.atomic_energies,
        expected["atomic_energies"],
        rtol=energy,
        atol=energy * scale,
    )
    scale = max(1.0, float(np.abs(expected["forces"]).max()))
    np.testing.assert_allclose(
        result.forces, expected["forces"], rtol=forces, atol=forces * scale
    )
    scale = max(1.0, float(np.abs(expected["atomic_virials"]).max()))
    np.testing.assert_allclose(
        result.virial,
        expected["virial"],
        rtol=virials,
        atol=virials * scale * len(result.forces),
    )
    # the six components LAMMPS gives, r_x f_y and so on: the virial of an
    # atom from three-body terms need not be symmetric
    upper = (np.array([0, 1, 2, 0, 0, 1]), np.array([0, 1, 2, 1, 2, 2]))
    np.testing.assert_allclose(
        result.atomic_virials[:, upper[0], upper[1]],
        expected["atomic_virials"][:, upper[0], upper[1]],
        rtol=virials,
        atol=virials * scale,
    )
