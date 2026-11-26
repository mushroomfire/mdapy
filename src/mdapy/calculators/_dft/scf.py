# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The self-consistent field of a cell on the GPU, in hartree atomic units: the
setup on the host (the grid, the k-points, V_loc, the core charge and the first
density from the pseudopotentials, the Ewald energy), then one call of
the module _dft of the GPU part for the whole loop."""

import math

import numpy as np

from . import augmentation, loader, paw, radial
from . import symmetry as symmetries

_FUNCTIONALS = {
    "PBE": "pbe_pw",
    "SLA PW PBX PBC": "pbe_pw",
    "SLA PW PBE PBE": "pbe_pw",
    "PBESOL": "pbesol_pw",
    "SLA PW PSX PSC": "pbesol_pw",
    "PW": "lda_pw",
    "LDA": "lda_pw",
}


def functional_of(pseudos) -> str:
    """The functional of the pseudopotentials, which must agree; the names of
    their headers taken without the spaces between their parts."""
    found = set()
    for pp in pseudos:
        name = " ".join(pp.functional.upper().split())
        if name not in _FUNCTIONALS:
            raise ValueError(f"no functional {name!r} here; PBE, PBEsol or PW LDA")
        found.add(_FUNCTIONALS[name])
    if len(found) != 1:
        raise ValueError(
            f"the pseudopotentials were made with different functionals: {sorted(found)}"
        )
    return found.pop()


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


def first_density(atom_f, positions, kinds, pseudos, miller, magmoms=None):
    """rho_atom(G) and the magnetization of the atoms (None without magmoms) on
    the sphere of the density, from the form factors of the free atoms (species,
    n): each times the structure factor of its atoms on the GPU; the
    magnetization is the density of each atom times its moment over its
    valence."""
    _dft = loader.module()
    tau = np.ascontiguousarray(positions, dtype=float)
    kinds = np.ascontiguousarray(kinds, dtype=np.int32)
    m = np.ascontiguousarray(miller, dtype=np.int32)
    factors = np.ascontiguousarray(atom_f, dtype=float)
    atom = _dft.structure_sum(m, tau, kinds, np.ones(len(kinds)), factors)
    moment = None
    if magmoms is not None:
        valence = np.array([pseudos[k].valence for k in kinds])
        weights = np.ascontiguousarray(np.asarray(magmoms, dtype=float) / valence)
        moment = _dft.structure_sum(m, tau, kinds, weights, factors)
    return atom, moment


def radial_parts(cell, positions, kinds, pseudos, miller, magmoms=None):
    """V_loc(G), rho_core(G) (None without core charges), rho_atom(G) and the
    magnetization of the atoms (None without magmoms) on the sphere of the
    density: the form factors of each species times the structure factor of
    its atoms; the magnetization is the density of each atom times its moment
    over its valence. In NumPy, the reference of first_density."""
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


def _hubbard_blocks(kinds, angular, hubbard_of):
    """Of each atom where its packed pairs start, its width, and for a
    Hubbard atom where its orbital starts among its functions and its l."""
    rows, start = [], 0
    for s in kinds:
        width = sum(2 * ell + 1 for ell in angular[s])
        function = hubbard_of[s][0]
        first = (
            sum(2 * ell + 1 for ell in angular[s][:function]) if function >= 0 else -1
        )
        rows.append(
            (start, width, first, angular[s][function] if function >= 0 else -1)
        )
        start += width * (width + 1) // 2
    return rows, start


def _pair(width, a, b):
    """The index of the pair a <= b among the packed pairs of a block."""
    a, b = min(a, b), max(a, b)
    return a * width - a * (a - 1) // 2 + b - a


def _hubbard_start(kinds, angular, hubbard_of, electrons, spins, magmoms):
    """The first occupations (spins, pairs) as Quantum ESPRESSO starts them,
    of the electrons of each Hubbard orbital: with a moment the orbitals of
    the majority spin filled first, one electron each, the rest spread over
    the minority; without, spread over both."""
    rows, pairs = _hubbard_blocks(kinds, angular, hubbard_of)
    becsum = np.zeros((spins, pairs))
    for i, (s, (start, width, first, ell)) in enumerate(zip(kinds, rows)):
        if first < 0:
            continue
        total, d = electrons[s], 2 * ell + 1
        moment = 0.0 if magmoms is None else float(magmoms[i])
        for m in range(first, first + d):
            k = start + _pair(width, m, m)
            if spins == 1:
                becsum[0, k] = total / d
            elif moment == 0.0:
                becsum[:, k] = total / 2 / d
            else:
                major, minor = (0, 1) if moment > 0 else (1, 0)
                becsum[major, k] = min(1.0, total / d)
                becsum[minor, k] = max(0.0, (total - d) / d)
    return becsum


def _hubbard_occupations(becsum, kinds, angular, hubbard_of, spins):
    """n (spins, 2l + 1, 2l + 1) of each Hubbard atom, a spin channel each."""
    rows, _ = _hubbard_blocks(kinds, angular, hubbard_of)
    out = []
    for start, width, first, ell in rows:
        if first < 0:
            continue
        d = 2 * ell + 1
        n = np.zeros((spins, d, d))
        for x in range(d):
            for y in range(d):
                k = start + _pair(width, first + x, first + y)
                n[:, x, y] = becsum[:, k] * (1.0 if x == y else 0.5)
        out.append(n / (2.0 if spins == 1 else 1.0))
    return out


# compute capabilities whose double precision runs at half the rate of single
# (P100, V100, A100, H100 and GH200, B200); the others, of cards for games and
# workstations, at 1/32 or 1/64
_FULL_DOUBLE = {(6, 0), (7, 0), (8, 0), (9, 0), (10, 0)}


def full_double() -> bool:
    """Whether the double precision of the current GPU runs at half the rate
    of its single: the NVIDIA GPUs of _FULL_DOUBLE, and the AMD GPUs HIP runs
    on (the Instinct series), whatever capability HIP reports for them."""
    _dft = loader.module()
    if _dft.api == "HIP":
        return True
    return tuple(_dft.device_capability()) in _FULL_DOUBLE


def mixed_precision(precision: str) -> bool:
    """Whether a run of this precision ("double", "mixed" or "auto") takes
    the single-precision steps on the current GPU: "auto" on GPUs whose double
    precision is a small part of their single."""
    if precision != "auto":
        return precision == "mixed"
    return not full_double()


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
    beta: float = 0.7,
    kerker: float = 0.8,
    minimum: float = 0.0,
    history: int = 8,
    metric_weight: float = 20.0,
    energy_tolerance: float = 1e-8,
    density_tolerance: float = 1e-10,
    max_iterations: int = 100,
    davidson_iterations: int = 100,
    davidson_basis: int = 2,
    seed: int = 0,
    functional: str | None = None,
    magmoms=None,
    magnetic_beta: float = 1.6,
    stress: bool = True,
    basis_cell=None,
    correct_forces: bool = True,
    state=None,
    magnetic_kerker: float = 0.0,
    smooth_grid=None,
    onecenter=None,
    atomic_start: bool = True,
    symmetry: bool | None = None,
    gamma: bool | None = None,
    comm=None,
    hubbard=None,
    kpoint_threads: int = 0,
    start_density=None,
    start_becsum=None,
    first_tolerance: float = 0.0,
    conditioning: float = 1e6,
    output_energies: bool = False,
    adaptive_damping: bool = False,
    first_held: int = 0,
    preconditioner: str = "kerker",
    kpoints=None,
    weights=None,
    precision: str = "double",
    kernels: str = "mdapy",
    on_start=None,
    on_step=None,
    hold: float = 1e-2,
    hold_steps: int = 12,
) -> dict:
    """The ground state of a cell of rows cell (bohr), atoms at fractional
    positions of species kinds (indices of pseudos), plane waves to cutoff
    (hartree) and the density to density_cutoff (4 cutoff by default, 8 with
    ultrasoft pseudopotentials) on grid (the FFT grid of the density cutoff
    by default); the bands on smooth_grid, that of 4 cutoff when the density
    takes more, else grid. The forces are in
    hartree / bohr, with correct_forces QE's correction for the change of the
    input density with the atoms, their mean taken out and kept as net_force; with stress,
    strain_derivative is dE/d epsilon (3, 3) of a homogeneous strain at fixed
    Miller indices, hartree; basis_cell chooses the plane waves (and the
    sphere of the density), the cell by default, for differences of the energy
    at a fixed basis; state, a State of the module _dft, starts the run
    from the last one kept in it when they fit and keeps this one (its
    steps, 3 instead of 1, extrapolates the density to second order from the
    last three runs). With
    magmoms, the first
    moments of the atoms in Bohr magnetons, the spins are collinear; the
    eigenvalues and occupations are then (2, K, bands) and the density (2, n1,
    n2, n3), up and down. With PAW datasets, onecenter(data, becsum) gives E^1
    and D (spins, pairs) of an atom of paw.onecenter data from its
    occupations; the result then holds onecenter (E^1) and becsum, the
    occupations of all atoms (spins, pairs). With atomic_start the first
    bands are those of the pseudo-orbitals of the free atoms in the first
    potential (random ones fill up to the bands), else random. With symmetry
    (by default when spglib is installed, never with basis_cell) the
    operations of the crystal reduce the k-points and make the density, the
    occupations of the projectors, the forces and the stress symmetric; the
    result holds their number as operations. With gamma (by default when the
    only k-point is Gamma) the bands are real, on half the plane waves; the
    result says which as gamma. With comm, an mpi4py communicator of several
    processes (each on its own GPU, chosen before), the pairs of spin and
    k-point are shared among them as pools, every process taking part with
    the same arguments; each gets the whole result. With hubbard, a mapping of
    species (indices of pseudos, norm-conserving) to the label of one of
    their pseudo-orbitals and U in hartree, DFT+U of Dudarev on those
    orbitals as Quantum ESPRESSO's atomic projectors, started from its
    occupations; the result then holds hubbard_energy and
    hubbard_occupations, n of each Hubbard atom (spins, 2l + 1, 2l + 1) in
    the order of the atoms. kpoint_threads host threads solve the k-points
    side by side on the GPU, each with its own Hamiltonian: 0 for as many as
    small cells of many k-points use, 1 for one. start_density, the
    density on grid (as the result's, (2, n1, n2, n3) with spin), replaces the
    superposition of the free atoms as the first input density; with
    first_tolerance (hartree) the bands of the first step are solved to that
    residual norm, as a non-self-consistent pass at its potential needs, and
    start_becsum, the occupations of the projectors (spins, pairs) as the
    result's becsum, replaces the first ones of PAW or DFT+U; with
    first_held only the first first_held bands of each k-point, those
    that may hold electrons; the others to the loose tolerance of empty bands.
    conditioning bounds the condition number of the residual differences
    that Pulay's mixing combines: directions of smaller singular values are
    dropped. With output_energies the result holds output_energies, the
    Kohn-Sham free energy of the output density and occupations of each step
    (variational, where energies are the Harris-Foulkes functional of the
    inputs; NaN with DFT+U). With adaptive_damping each step is a trial,
    taken back and tried closer when neither its output energy nor its
    residual is below those of the step it came from (Herbst and Levitt,
    J. Comput. Phys. 459, 111127 (2022)), and the shares of the mixing
    adapt: the result holds output_energies, accepted (1 for the steps
    taken) and dampings (the factor on each step of the mixing); not with
    DFT+U. preconditioner, of the residual of the total density: "kerker"
    (with kerker) or "ldos", from the local density of states at the Fermi
    level of each step (Herbst and Levitt, J. Phys.: Condens. Matter 33,
    085503 (2021)), for cells of metal and vacuum or insulator. kpoints (K, 3),
    fractions of the reciprocal vectors, replace the mesh of kgrid and offset
    and are taken as they are, none reduced or added (the symmetry, when on,
    still acts on the density), with weights (K,) that add up to one, equal
    ones by default: with max_iterations 1, start_density and
    first_tolerance (and start_becsum with PAW), the bands at those k-points
    in the potential of that density. The result holds unconverged, the bands that Davidson left
    above their tolerance at each step, summed over the k-points. precision
    "mixed": the first steps in single precision (complex bands, and the
    real ones of gamma), then double precision to the end; the result holds single (1 for
    the steps in single precision); "auto": mixed on GPUs of little double
    precision (mixed_precision). kernels "mdapy": the kernels of MDAPY where
    it has its own (the FFTs, the dense linear algebra, the eigensolver and
    the Cholesky factorization), "vendor": the GPU libraries of the vendor
    throughout, loaded here (InvalidParameter if they cannot be); the result
    holds it as kernels. on_start(setup), before the loop, gets a
    dict of what the run takes: grid, smooth_grid, density_cutoff (hartree),
    bands, kpoints (their number), operations, gamma, spins, electrons,
    kernels and mixed (the precision); on_step(step, energy, accuracy,
    magnetization, absolute_magnetization, fermi, davidson, single, accepted,
    held), after each step, its number from 1 and its values in hartree, as
    the result's histories (a log of the run). With spin the result holds
    magnetizations and absolute_magnetizations, the integrals of m and of |m|
    of the output density of each step over the cell, in Bohr magnetons.
    With moments, from the free atoms, the magnetization of the input density
    (and of the occupations of PAW and DFT+U) stays that of the first moments
    while the total density settles: until the Hartree energy of the residual
    of the total density is below hold (hartree per atom), for at most
    hold_steps steps (0 for none); the bands of the first steps do not carry
    the moments they are given, and mixing their magnetization loses the
    start. The history of the mixing starts anew when the moments are let
    go. Not from start_density or a state, nor with adaptive_damping or
    moments that are all zero. The result holds held (1 for the steps whose
    magnetization was not taken)."""
    if preconditioner not in ("kerker", "ldos"):
        raise ValueError("preconditioner must be 'kerker' or 'ldos'")
    if precision not in ("double", "mixed", "auto"):
        raise ValueError("precision must be 'double', 'mixed' or 'auto'")
    if kernels not in ("mdapy", "vendor"):
        raise ValueError("kernels must be 'mdapy' or 'vendor'")
    _dft = loader.module()
    if kernels == "vendor":
        loader.vendor_libraries(_dft)
    cell = np.ascontiguousarray(cell, dtype=float)
    positions = np.ascontiguousarray(positions, dtype=float)
    kinds = np.ascontiguousarray(kinds, dtype=np.int32)
    if density_cutoff is None:
        # ultrasoft augmentation charges need more of the density
        density_cutoff = (8.0 if any(pp.ultrasoft for pp in pseudos) else 4.0) * cutoff
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
    ops = None
    wanted = symmetries.available() if symmetry is None else bool(symmetry)
    if wanted and basis_cell is None:
        ops = symmetries.operations(cell, positions, kinds, magmoms)
        if len(ops) == 1:
            ops = None
    # the grids fit the fractional translations of the operations, as QE's
    factors = (1, 1, 1) if ops is None else symmetries.fft_factors(ops)
    grid = (
        tuple(_dft.fft_grid(cell, density_cutoff, factors))
        if grid is None
        else tuple(grid)
    )
    if smooth_grid is None:
        smooth_grid = (
            grid
            if density_cutoff <= 4.0 * cutoff
            else tuple(_dft.fft_grid(cell, 4.0 * cutoff, factors))
        )
    smooth_grid = tuple(smooth_grid)
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
    atom_g, moment_g = first_density(atom_f, positions, kinds, pseudos, miller, magmoms)
    density = on_grid(atom_g, miller, grid)
    scale = electrons / (density.mean() * volume)
    density *= scale
    if spins == 2:
        moment = on_grid(moment_g, miller, grid) * scale
        density = np.stack([0.5 * (density + moment), 0.5 * (density - moment)])
    if start_density is not None:
        density = np.asarray(start_density, dtype=float)
    qmax = math.sqrt(2 * cutoff) + 1.0
    tables = [radial.projector_table(pp, qmax, volume) for pp in pseudos]
    # the pseudo-orbitals of the free atoms, for the first bands
    orbitals = [
        (
            np.ascontiguousarray(radial.orbital_table(pp, qmax, volume)),
            [o.angular for o in pp.orbitals],
        )
        if pp.orbitals
        else (np.zeros((0, tables[i].shape[1])), [])
        for i, pp in enumerate(pseudos)
    ]
    angular = [[p.angular for p in pp.projectors] for pp in pseudos]
    dij = [np.ascontiguousarray(pp.dij, dtype=float) for pp in pseudos]
    hubbard_of = hubbard_electrons = None
    if hubbard:
        # the pseudo-orbital of each Hubbard species as one more projector
        # function of D = 0, whose strengths the field fills with v = U (1/2 - n)
        hubbard_of = [(-1, 0.0)] * len(pseudos)
        hubbard_electrons = [0.0] * len(pseudos)
        for s, (label, u) in hubbard.items():
            pp = pseudos[s]
            if pp.ultrasoft:
                raise ValueError(
                    "DFT+U takes norm-conserving pseudopotentials only, for now"
                )
            found = [
                i
                for i, o in enumerate(pp.orbitals)
                if o.label.lower() == str(label).lower()
            ]
            if len(found) != 1:
                labels = [o.label for o in pp.orbitals]
                raise ValueError(f"no single pseudo-orbital {label} among {labels}")
            table = radial.orbital_table(pp, qmax, volume)[found[0]]
            tables[s] = np.ascontiguousarray(np.vstack([tables[s], table]))
            angular[s] = angular[s] + [pp.orbitals[found[0]].angular]
            grown = np.zeros((len(angular[s]), len(angular[s])))
            grown[:-1, :-1] = dij[s]
            dij[s] = grown
            hubbard_of[s] = (len(angular[s]) - 1, float(u))
            hubbard_electrons[s] = float(pp.orbitals[found[0]].occupation)
    charges = np.array([pseudos[s].valence for s in kinds])
    e_ewald, f_ewald, s_ewald = _dft.ewald(
        cell,
        np.ascontiguousarray(positions @ cell),
        np.ascontiguousarray(charges, dtype=float),
    )
    if kpoints is not None:
        kpoints = np.ascontiguousarray(kpoints, dtype=float)
        if kpoints.ndim != 2 or kpoints.shape[1] != 3 or len(kpoints) == 0:
            raise ValueError("kpoints are (K, 3)")
        if weights is None:
            weights = np.full(len(kpoints), 1.0 / len(kpoints))
        weights = np.ascontiguousarray(weights, dtype=float)
        if weights.shape != (len(kpoints),) or abs(weights.sum() - 1.0) > 1e-9:
            raise ValueError("a weight for every k-point, adding up to one")
    elif weights is not None:
        raise ValueError("weights come with kpoints")
    elif ops is None:
        kpoints, weights = _dft.monkhorst_pack(list(kgrid), list(offset), True)
    else:
        full, _ = _dft.monkhorst_pack(list(kgrid), list(offset), False)
        kpoints, weights = symmetries.irreducible(full, ops.rotations)
        kpoints, weights = np.ascontiguousarray(kpoints), np.ascontiguousarray(weights)
    at_gamma = len(kpoints) == 1 and not np.any(kpoints[0])
    if gamma is None:
        gamma = at_gamma
    elif gamma and not at_gamma:
        raise ValueError("real bands take the one k-point Gamma")
    ultrasoft = any(pp.ultrasoft for pp in pseudos)
    augmented = None
    if ultrasoft:
        augmented = [
            augmentation.terms(pp, math.sqrt(2 * density_cutoff) + 1, volume)
            if pp.ultrasoft
            else None
            for pp in pseudos
        ]
    becsum = callback = data = None
    paw_atoms = [i for i, s in enumerate(kinds) if pseudos[s].paw is not None]
    if paw_atoms:
        name = functional or functional_of(pseudos)
        becsum, callback, data = _paw(
            pseudos, kinds, paw_atoms, spins, magmoms, name, onecenter
        )
    if hubbard_of is not None:
        if paw_atoms:
            raise ValueError(
                "DFT+U takes norm-conserving pseudopotentials only, for now"
            )
        becsum = _hubbard_start(
            kinds, angular, hubbard_of, hubbard_electrons, spins, magmoms
        )
    if start_becsum is not None:
        start_becsum = np.asarray(start_becsum, dtype=float)
        if becsum is None or start_becsum.shape != becsum.shape:
            raise ValueError(
                "start_becsum takes the occupations (spins, pairs) of PAW or DFT+U"
            )
        becsum = start_becsum
    mixed = mixed_precision(precision)
    if hold < 0 or hold_steps < 0:
        raise ValueError("hold and hold_steps must be 0 or more")
    holding = (
        spins == 2
        and start_density is None
        and not adaptive_damping
        and bool(np.any(magmoms != 0.0))
    )
    if on_start is not None:
        on_start({
            "grid": grid, "smooth_grid": smooth_grid, "density_cutoff": density_cutoff,
            "bands": bands, "kpoints": len(kpoints),
            "operations": 1 if ops is None else len(ops), "gamma": bool(gamma),
            "spins": spins, "electrons": electrons, "kernels": kernels, "mixed": mixed,
        })  # fmt: skip
    if density.shape != ((spins,) if spins == 2 else ()) + tuple(grid):
        raise ValueError("start_density on the grid of the run, with each spin")
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
        state, magnetic_kerker, augmented,
        None if smooth_grid == grid else list(smooth_grid), callback, becsum,
        data if callback is None else None, orbitals if atomic_start else None,
        None if ops is None else symmetries.payload(ops), bool(gamma), _pools(comm),
        hubbard_of, int(kpoint_threads), float(first_tolerance), float(conditioning),
        bool(output_energies), bool(adaptive_damping), int(first_held),
        preconditioner == "ldos", mixed, kernels, on_step,
        float(hold), int(hold_steps) if holding else 0,
    )  # fmt: skip
    if hubbard_of is not None:
        result["hubbard_occupations"] = _hubbard_occupations(
            result["becsum"], kinds, angular, hubbard_of, spins
        )
    else:  # the occupations of PAW come with it
        result.pop("hubbard_energy", None)
    if spins == 1:
        for key in ("eigenvalues", "occupations", "density"):
            result[key] = result[key][0]
    # the forces with those of the ions, less their mean: the net force of a
    # periodic cell is zero but for the grid
    if stress:
        result["strain_derivative"] = result["strain_derivative"] + s_ewald
    forces = result["forces"] + f_ewald
    if ops is not None:
        forces = symmetries.forces(ops, forces)
        if stress:
            result["strain_derivative"] = symmetries.tensor(
                ops, result["strain_derivative"]
            )
    result["kernels"] = kernels
    result["net_force"] = forces.mean(axis=0)
    result["forces"] = forces - result["net_force"]
    result.update(
        smooth_grid=smooth_grid,
        grid=grid, kpoints=kpoints, weights=weights, electrons=electrons, bands=bands,
        spins=spins, operations=1 if ops is None else len(ops),
    )  # fmt: skip
    return result


def _pools(comm):
    """(rank, ranks, allreduce) of an mpi4py communicator of several
    processes, allreduce returning the sum of an array over them; None for
    one."""
    if comm is None or comm.Get_size() == 1:
        return None
    from mpi4py import MPI

    def allreduce(x):
        out = np.empty_like(x)
        comm.Allreduce(np.ascontiguousarray(x), out, op=MPI.SUM)
        return out

    return comm.Get_rank(), comm.Get_size(), allreduce


def _paw(pseudos, kinds, paw_atoms, spins, magmoms, functional, onecenter):
    """The first occupations (spins, pairs) of all atoms, those of PAW atoms
    of their free atom; the callable of the one-center terms of all of them
    from onecenter of one, None without; and the one-center data of each
    species, None for those without PAW."""
    widths = [sum(2 * p.angular + 1 for p in pseudos[s].projectors) for s in kinds]
    offsets = np.cumsum([0] + [w * (w + 1) // 2 for w in widths])
    gga = functional.startswith("pbe")
    data = [paw.onecenter(pp, gga) if pp.paw is not None else None for pp in pseudos]
    becsum = np.zeros((spins, offsets[-1]))
    for i in paw_atoms:
        moment = 0.0 if magmoms is None else float(magmoms[i])
        becsum[:, offsets[i] : offsets[i + 1]] = paw.initial_becsum(
            pseudos[kinds[i]], spins, moment
        )
    if onecenter is None:
        return becsum, None, data

    def callback(b):
        energy, d = 0.0, np.zeros_like(b)
        for i in paw_atoms:
            part = slice(offsets[i], offsets[i + 1])
            e, d[:, part] = onecenter(
                data[int(kinds[i])], np.ascontiguousarray(b[:, part])
            )
            energy += e
        return energy, d

    return becsum, callback, data
