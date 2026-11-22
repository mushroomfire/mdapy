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


def radial_parts(cell, positions, kinds, pseudos, miller, magmoms=None):
    """V_loc(G), rho_core(G) (None without core charges), rho_atom(G) and the
    magnetization of the atoms (None without magmoms) on the sphere of the
    density: the radial transforms of each species, once per shell of |G|,
    times the structure factor of its atoms; the magnetization is the density
    of each atom times its moment over its valence."""
    volume = abs(np.linalg.det(cell))
    b = 2 * math.pi * np.linalg.inv(cell).T
    q = np.linalg.norm(miller @ b, axis=1)
    shells, where = np.unique(np.round(q, 10), return_inverse=True)
    local = np.zeros(len(q), dtype=complex)
    atom = np.zeros(len(q), dtype=complex)
    core = (
        np.zeros(len(q), dtype=complex)
        if any(pp.core is not None for pp in pseudos)
        else None
    )
    moment = None if magmoms is None else np.zeros(len(q), dtype=complex)
    for s, pp in enumerate(pseudos):
        mine = np.asarray(kinds) == s
        if not mine.any():
            continue
        phases = np.exp(-2j * math.pi * (miller @ positions[mine].T))
        factor = phases.sum(axis=1)
        local += factor * radial.local_potential(pp, shells, volume)[where]
        free = radial.radial_density(pp.atom, pp, shells, volume, False)[where]
        atom += factor * free
        if moment is not None:
            moment += (
                phases @ (np.asarray(magmoms, dtype=float)[mine] / pp.valence) * free
            )
        if pp.core is not None:
            core += (
                factor * radial.radial_density(pp.core, pp, shells, volume, True)[where]
            )
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
) -> dict:
    """The ground state of a cell of rows cell (bohr), atoms at fractional
    positions of species kinds (indices of pseudos), plane waves to cutoff
    (hartree) and the density to density_cutoff (4 cutoff by default) on grid
    (the FFT grid of the density cutoff by default). With magmoms, the first
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
    miller, _, _, _ = _dft.sphere(cell, [0.0, 0.0, 0.0], density_cutoff, grid)
    local_g, core_g, atom_g, moment_g = radial_parts(
        cell, positions, kinds, pseudos, miller, magmoms
    )
    local = on_grid(local_g, miller, grid)
    core = None if core_g is None else on_grid(core_g, miller, grid)
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
    e_ewald = ewald(cell, positions @ cell, charges)
    kpoints, weights = _dft.monkhorst_pack(list(kgrid), list(offset), True)
    result = _dft.scf(
        cell, list(grid), cutoff, density_cutoff, tables, angular, dij, radial.DQ,
        positions, kinds, kpoints, weights, local, core, np.ascontiguousarray(density),
        e_ewald, spins, bands, electrons, functional or functional_of(pseudos),
        smearing, sigma, beta, kerker, minimum, magnetic_beta, history, metric_weight,
        energy_tolerance, density_tolerance, max_iterations, davidson_iterations,
        davidson_basis, seed,
    )  # fmt: skip
    if spins == 1:
        for key in ("eigenvalues", "occupations", "density"):
            result[key] = result[key][0]
    result.update(
        grid=grid, kpoints=kpoints, weights=weights, electrons=electrons, bands=bands,
        spins=spins,
    )  # fmt: skip
    return result
