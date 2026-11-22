# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The self-consistent field of a cell on the GPU, in hartree atomic units: the
setup on the host (the grid, the k-points, V_loc, the core charge and the first
density from the pseudopotentials, the Ewald energy), then one call of
mdapy_dft_cuda12 for the whole loop."""

import math

import numpy as np

from . import radial
from .ewald import ewald

_FUNCTIONALS = {"PBE": "pbe", "SLA PW PBX PBC": "pbe", "PW": "lda_pw", "LDA": "lda_pw"}


def functional_of(pseudos) -> str:
    """The functional of the pseudopotentials, which must agree."""
    names = {pp.functional.strip().upper() for pp in pseudos}
    if len(names) != 1:
        raise ValueError(
            f"the pseudopotentials were made with different functionals: {sorted(names)}"
        )
    name = names.pop()
    if name not in _FUNCTIONALS:
        raise ValueError(f"no functional {name!r} here; PBE or PW LDA")
    return _FUNCTIONALS[name]


def on_grid(components: np.ndarray, miller: np.ndarray, grid) -> np.ndarray:
    """The real function on the grid of Fourier components on a sphere."""
    box = np.zeros(tuple(grid), dtype=complex)
    m = np.mod(miller, grid)
    box[m[:, 0], m[:, 1], m[:, 2]] = components
    return np.fft.ifftn(box).real * box.size


def form_factors(cell, pseudos, miller, slopes=False):
    """The form factors of each species on the sphere of the density, (species,
    n), per unit volume of the cell: of V_loc, of rho_core (None without core
    charges) and of the free atom, interpolated from their tables; with slopes
    also the derivatives by |G| of the first two at a fixed volume, for the
    stress."""
    volume = abs(np.linalg.det(cell))
    q = np.linalg.norm(miller @ (2 * math.pi * np.linalg.inv(cell).T), axis=1)

    def each(transform, core=False):
        return np.array([
            np.zeros(len(q)) if core and pp.core is None else transform(pp)
            for pp in pseudos
        ])  # fmt: skip

    has_core = any(pp.core is not None for pp in pseudos)
    local = each(lambda pp: radial.local_potential(pp, q, volume))
    atom = each(lambda pp: radial.radial_density(pp.atom, pp, q, volume, False))
    core = None
    if has_core:
        core = each(
            lambda pp: radial.radial_density(pp.core, pp, q, volume, True), True
        )
    if not slopes:
        return local, core, atom
    local_slope = each(lambda pp: radial.local_potential_slope(pp, q, volume))
    core_slope = None
    if has_core:
        core_slope = each(
            lambda pp: radial.radial_density_slope(pp.core, pp, q, volume, True), True
        )
    return local, core, atom, local_slope, core_slope


def radial_parts(cell, positions, kinds, pseudos, miller, magmoms=None):
    """V_loc(G), rho_core(G) (None without core charges), rho_atom(G) and the
    magnetization of the atoms (None without magmoms) on the sphere of the
    density: the form factors of each species times the structure factor of
    its atoms; the magnetization is the density of each atom times its moment
    over its valence."""
    local_f, core_f, atom_f = form_factors(cell, pseudos, miller)
    phases = np.exp(-2j * math.pi * (miller @ np.asarray(positions, dtype=float).T))
    kinds = np.asarray(kinds)
    local = (phases * local_f[kinds].T).sum(axis=1)
    atom = (phases * atom_f[kinds].T).sum(axis=1)
    core = None if core_f is None else (phases * core_f[kinds].T).sum(axis=1)
    moment = None
    if magmoms is not None:
        valence = np.array([pseudos[k].valence for k in kinds])
        moment = (
            phases * (atom_f[kinds] * (np.asarray(magmoms) / valence)[:, None]).T
        ).sum(axis=1)
    return local, core, atom, moment


def run(
    cell,
    positions,
    kinds,
    pseudos,
    cutoff: float,
    kgrid=(1, 1, 1),
    offset=(0.0, 0.0, 0.0),
    smearing: str = "fixed",
    sigma: float = 0.0,
    bands: int | None = None,
    density_cutoff: float | None = None,
    grid=None,
    beta: float = 0.4,
    kerker: float = 0.0,
    minimum: float = 0.1,
    history: int = 8,
    metric_weight: float = 20.0,
    energy_tolerance: float = 1e-8,
    density_tolerance: float = 1e-10,
    max_iterations: int = 100,
    davidson_iterations: int = 100,
    davidson_basis: int = 4,
    seed: int = 0,
    functional: str | None = None,
    magmoms=None,
    magnetic_beta: float = 0.4,
    stress: bool = True,
    basis_cell=None,
    correct_forces: bool = True,
) -> dict:
    """The ground state of a cell of rows cell (bohr), atoms at fractional
    positions of species kinds (indices of pseudos), plane waves to cutoff
    (hartree) and the density to density_cutoff (4 cutoff by default) on grid
    (the FFT grid of the density cutoff by default). The forces are in
    hartree / bohr, with correct_forces QE's correction for the change of the
    input density with the atoms, their mean taken out and kept as net_force; with stress,
    strain_derivative is dE/d epsilon (3, 3) of a homogeneous strain at fixed
    Miller indices, hartree; basis_cell chooses the plane waves (and the
    sphere of the density), the cell by default, for differences of the energy
    at a fixed basis. With magmoms, the first
    moments of the atoms in Bohr magnetons, the spins are collinear; the
    eigenvalues and occupations are then (2, K, bands) and the density (2, n1,
    n2, n3), up and down."""
    from mdapy_dft_cuda12 import _dft

    cell = np.ascontiguousarray(cell, dtype=float)
    positions = np.ascontiguousarray(positions, dtype=float)
    kinds = np.ascontiguousarray(kinds, dtype=np.int32)
    density_cutoff = 4.0 * cutoff if density_cutoff is None else density_cutoff
    grid = tuple(_dft.fft_grid(cell, density_cutoff)) if grid is None else tuple(grid)
    volume = abs(np.linalg.det(cell))
    electrons = float(sum(pseudos[s].valence for s in kinds))
    if bands is None:
        occupied = math.ceil(electrons / 2)
        if smearing == "fixed":
            bands = occupied + max(4, math.ceil(0.1 * occupied))
        else:
            bands = max(math.ceil(1.2 * electrons / 2), occupied + 4)
    spins = 1
    if magmoms is not None:
        magmoms = np.asarray(magmoms, dtype=float)
        valence = np.array([pseudos[s].valence for s in kinds])
        if magmoms.shape != (len(kinds),) or (np.abs(magmoms) > valence).any():
            raise ValueError("a moment for every atom, at most its valence")
        spins = 2
    # the G of the sphere of the density in the order of the GPU's
    chooser = (
        cell if basis_cell is None else np.ascontiguousarray(basis_cell, dtype=float)
    )
    miller, _, _, _ = _dft.sphere(chooser, [0.0, 0.0, 0.0], density_cutoff, grid)
    local_s = core_s = None
    if stress:
        local_f, core_f, atom_f, local_s, core_s = form_factors(
            cell, pseudos, miller, True
        )
    else:
        local_f, core_f, atom_f = form_factors(cell, pseudos, miller)
    _, _, atom_g, moment_g = radial_parts(
        cell, positions, kinds, pseudos, miller, magmoms
    )
    density = on_grid(atom_g, miller, grid)
    scale = electrons / (density.mean() * volume)
    density *= scale
    if spins == 2:
        moment = on_grid(moment_g, miller, grid) * scale
        density = np.stack([0.5 * (density + moment), 0.5 * (density - moment)])
    qmax = math.sqrt(2 * cutoff) + 1.0
    tables = [radial.projector_table(pp, qmax, volume) for pp in pseudos]
    angular = [[p.angular for p in pp.projectors] for pp in pseudos]
    dij = [np.ascontiguousarray(pp.dij, dtype=float) for pp in pseudos]
    charges = np.array([pseudos[s].valence for s in kinds])
    e_ewald, f_ewald, s_ewald = ewald(cell, positions @ cell, charges, True, True)
    kpoints, weights = _dft.monkhorst_pack(list(kgrid), list(offset), True)
    result = _dft.scf(
        cell, list(grid), cutoff, density_cutoff, tables, angular, dij, radial.DQ,
        positions, kinds, kpoints, weights, local_f, core_f, local_s, core_s,
        atom_f if correct_forces else None,
        np.ascontiguousarray(density),
        e_ewald, spins, bands, electrons, functional or functional_of(pseudos),
        smearing, sigma, beta, kerker, minimum, magnetic_beta, history, metric_weight,
        energy_tolerance, density_tolerance, max_iterations, davidson_iterations,
        davidson_basis, seed,
        None if basis_cell is None else np.ascontiguousarray(basis_cell, dtype=float),
    )  # fmt: skip
    if spins == 1:
        for key in ("eigenvalues", "occupations", "density"):
            result[key] = result[key][0]
    # the forces with those of the ions, less their mean: the net force of a
    # periodic cell is zero but for the grid
    if stress:
        result["strain_derivative"] = result["strain_derivative"] + s_ewald
    forces = result["forces"] + f_ewald
    result["net_force"] = forces.mean(axis=0)
    result["forces"] = forces - result["net_force"]
    result.update(
        grid=grid, kpoints=kpoints, weights=weights, electrons=electrons, bands=bands,
        spins=spins,
    )  # fmt: skip
    return result
