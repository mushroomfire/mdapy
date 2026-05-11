# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

"""
Quasi-harmonic-approximation (QHA) temperature-dependent elastic constants
==========================================================================

Energy-strain QHA workflow:

1. Detect crystal class via spglib. Cubic and hexagonal are supported.
   For chemically-disordered cells (SQS HEAs) pass
   ``ignore_elements_for_symmetry=True`` so the spglib check ignores species
   and recognises the parent FCC / BCC / HCP lattice.
2. Build a (V, mode, eps) grid of strained unit cells. The strain modes are
   chosen by crystal class (cubic: 3 modes -> C11, C12, C44; hexagonal:
   5 modes -> C11, C12, C13, C33, C44).
3. For each cell, build force constants:
     - ``force_constants_method='finite-displacement'`` (default): phonopy
       generates per-(atom, axis) displacement cells; one force per cell.
       Use this for the MD path (NEP / EAM) and small DFT systems.
     - ``force_constants_method='dfpt'``: DFT-only. Each cell is one VASP
       DFPT (IBRION=8) SCF that returns the full force-constant matrix.
4. MD path runs ``self.calc`` to compute static energies + supercell forces
   in place. DFT path: ``export_inputs(...)`` writes POSCARs the user runs
   externally, then ``import_results(...)`` reads vasprun.xml back.
5. ``compute()`` does:
     - phonopy thermal_properties on every cell -> F_phonon(V, mode, eps, T)
     - F_total = E_static + F_phonon
     - Bulk EOS (Birch-Murnaghan 3rd order over the V scan at eps=0) ->
       V(T), B_T(T)
     - alpha(T) = (1/V) dV/dT (centred difference)
     - At each T: interpolate F_total over V to V(T), parabolic-fit over
       eps to get per-mode curvatures kappa_k, invert to C_ij(T) via the
       per-class formulas
     - Adiabatic conversion via thermal-stress matrix
     - Voigt-Reuss-Hill polycrystalline averages on the full 6x6
     - Born stability per crystal class

Output: a polars DataFrame with one row per T, columns
``T, V_eq, alpha, B_T, <independent C_ij>_iso, <independent C_ij>_adi,
K_VRH, G_VRH, E_VRH, nu_VRH, stable``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np
import polars as pl

from mdapy.system import System
from mdapy.calculator import CalculatorMP

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.qha.eos import get_eos, fit_to_eos
except ImportError:
    raise ImportError(
        "phonopy is required: https://phonopy.github.io/phonopy/install.html"
    )


# ============================================================
# Unit constants
# ============================================================

EV_PER_KJMOL = 1.0 / 96.48533212331  # ~0.010364, eV per (kJ/mol)
EV_PER_A3_TO_GPA = 160.21766208  # 1 eV/A^3 = 160.218 GPa


# ============================================================
# Per-crystal-class strain modes + curvature -> Cij formulas
# ============================================================
#
# Each mode is a Voigt 6-vector pattern multiplied by amplitude eps to form
# the actual strain. Curvature kappa_k = (a_k from polyfit) * 160.218 / V0
# expresses a particular linear combination of the independent C_ij; we
# invert that map to recover the C_ij. Patterns + formulas come from
# elastemp (Karthik et al., Comp. Mater. Sci. 2023).

# Cubic: 3 independent C_ij (C11, C12, C44), 3 strain modes.
#   mode 0 [e,-e,0,0,0,0]  -> kappa_0 = C11 - C12          (volume-preserving,
#                                                           pure tetragonal shear)
#   mode 1 [e,e,e,0,0,0]   -> kappa_1 = (3/2)(C11 + 2 C12) (also bulk EOS)
#   mode 2 [0,0,0,e,e,e]   -> kappa_2 = (3/2) C44
#
# Mode 0 is intentionally volume-preserving so the curvature does not mix
# bulk and shear: extracting C11-C12 directly avoids the subtraction
# (C11+C12) vs (C11+2*C12)*2/3 that amplifies fit noise (and gives strange
# C12 trends in low-symmetry / magnetic DFT runs).
CUBIC_STRAIN_MODES: Tuple[np.ndarray, ...] = (
    np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
)

# Hexagonal: 5 independent C_ij (C11, C12, C13, C33, C44; C66=(C11-C12)/2),
# 5 strain modes.
#   mode 0 [e,e,0,0,0,0]   -> kappa_0 = C11 + C12
#   mode 1 [0,0,0,0,0,e]   -> kappa_1 = (1/2) C66 = (C11-C12)/4
#   mode 2 [0,0,e,0,0,0]   -> kappa_2 = (1/2) C33
#   mode 3 [0,0,0,e,e,0]   -> kappa_3 = C44
#   mode 4 [e,e,e,0,0,0]   -> kappa_4 = C11 + C12 + 2 C13 + (1/2) C33
HEXAGONAL_STRAIN_MODES: Tuple[np.ndarray, ...] = (
    np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
    np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
)


def _cubic_curvatures_to_cij(kappa: Sequence[float]) -> Tuple[float, float, float]:
    """Cubic inversion: returns (C11, C12, C44) in GPa given (k0, k1, k2).

    With CUBIC_STRAIN_MODES:
        k0 = C11 - C12           (mode [e,-e,0,0,0,0])
        k1 = (3/2)(C11 + 2 C12)  (mode [e,e,e,0,0,0])
        k2 = (3/2) C44           (mode [0,0,0,e,e,e])
    """
    k0, k1, k2 = kappa
    c11_plus_2c12 = (2.0 / 3.0) * k1
    c11_minus_c12 = k0
    c11 = (c11_plus_2c12 + 2.0 * c11_minus_c12) / 3.0
    c12 = (c11_plus_2c12 - c11_minus_c12) / 3.0
    c44 = (2.0 / 3.0) * k2
    return c11, c12, c44


def _hexagonal_curvatures_to_cij(
    kappa: Sequence[float],
) -> Tuple[float, float, float, float, float]:
    """Hex inversion: returns (C11, C12, C13, C33, C44) in GPa given 5 kappas."""
    k0, k1, k2, k3, k4 = kappa
    # k0 = C11 + C12
    # k1 = (C11 - C12) / 4
    # k2 = C33 / 2
    # k3 = C44
    # k4 = C11 + C12 + 2 C13 + C33/2
    c11_minus_c12 = 4.0 * k1
    c11_plus_c12 = k0
    c11 = 0.5 * (c11_plus_c12 + c11_minus_c12)
    c12 = 0.5 * (c11_plus_c12 - c11_minus_c12)
    c33 = 2.0 * k2
    c44 = k3
    c13 = 0.5 * (k4 - c11_plus_c12 - 0.5 * c33)
    return c11, c12, c13, c33, c44


def _strain_modes_for(crystal_class: str) -> Tuple[np.ndarray, ...]:
    if crystal_class == "cubic":
        return CUBIC_STRAIN_MODES
    if crystal_class == "hexagonal":
        return HEXAGONAL_STRAIN_MODES
    raise ValueError(f"unsupported crystal_class: {crystal_class!r}")


def _build_cij_matrix(crystal_class: str, kappa: Sequence[float]) -> np.ndarray:
    """Construct the full symmetric 6x6 stiffness matrix in GPa from per-mode kappas."""
    C = np.zeros((6, 6))
    if crystal_class == "cubic":
        c11, c12, c44 = _cubic_curvatures_to_cij(kappa)
        for i in range(3):
            C[i, i] = c11
        for i, j in ((0, 1), (0, 2), (1, 2)):
            C[i, j] = C[j, i] = c12
        for i in range(3, 6):
            C[i, i] = c44
    elif crystal_class == "hexagonal":
        c11, c12, c13, c33, c44 = _hexagonal_curvatures_to_cij(kappa)
        C[0, 0] = C[1, 1] = c11
        C[0, 1] = C[1, 0] = c12
        C[0, 2] = C[2, 0] = c13
        C[1, 2] = C[2, 1] = c13
        C[2, 2] = c33
        C[3, 3] = C[4, 4] = c44
        C[5, 5] = 0.5 * (c11 - c12)  # hex constraint
    else:
        raise ValueError(f"unsupported crystal_class: {crystal_class!r}")
    return C


def _born_stable(crystal_class: str, C: np.ndarray) -> bool:
    """Born mechanical stability of a 6x6 Cij matrix."""
    if crystal_class == "cubic":
        c11, c12, c44 = C[0, 0], C[0, 1], C[3, 3]
        return (c11 > abs(c12)) and (c44 > 0.0) and (c11 + 2.0 * c12 > 0.0)
    if crystal_class == "hexagonal":
        c11, c12, c13, c33, c44 = C[0, 0], C[0, 1], C[0, 2], C[2, 2], C[3, 3]
        c66 = 0.5 * (c11 - c12)
        cond1 = c11 > abs(c12)
        cond2 = 2.0 * c13**2 < c33 * (c11 + c12)
        cond3 = c44 > 0.0
        cond4 = c66 > 0.0
        return bool(cond1 and cond2 and cond3 and cond4)
    raise ValueError(f"unsupported crystal_class: {crystal_class!r}")


# ============================================================
# Geometry helpers
# ============================================================


def _voigt_to_strain_tensor(v: np.ndarray) -> np.ndarray:
    """Voigt 6-vector (e1,e2,e3,e4,e5,e6) -> 3x3 symmetric strain tensor.

    Engineering shear convention: e4 = 2 e_yz, e5 = 2 e_xz, e6 = 2 e_xy.
    """
    return np.array(
        [
            [v[0], v[5] / 2.0, v[4] / 2.0],
            [v[5] / 2.0, v[1], v[3] / 2.0],
            [v[4] / 2.0, v[3] / 2.0, v[2]],
        ],
        dtype=float,
    )


def _strain_to_deformation(strain: np.ndarray) -> np.ndarray:
    """Symmetric 3x3 strain -> upper-triangular deformation gradient F via
    Cholesky of (I + 2 E)."""
    M = 2.0 * strain + np.eye(3)
    return np.linalg.cholesky(M).T


def _apply_isotropic_volume(cell: np.ndarray, vstrain: float) -> np.ndarray:
    """Scale all lattice vectors so cell volume changes by factor (1 + vstrain)."""
    return cell * (1.0 + vstrain) ** (1.0 / 3.0)


def _apply_deformation_to_cell(cell: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Apply deformation gradient F to a row-vector cell: new = old @ F.T."""
    return cell @ F.T


# ============================================================
# Phonopy <-> mdapy.System bridges
# ============================================================


def _system_to_phonopy(sys: System) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=sys.data["element"].to_numpy(),
        cell=np.asarray(sys.box.box, dtype=float),
        positions=sys.get_positions().to_numpy(),
    )


def _phonopy_to_system(atoms: PhonopyAtoms) -> System:
    data = pl.DataFrame(
        {
            "element": atoms.symbols,
            "x": atoms.positions[:, 0],
            "y": atoms.positions[:, 1],
            "z": atoms.positions[:, 2],
        },
        schema={
            "element": pl.Utf8,
            "x": pl.Float64,
            "y": pl.Float64,
            "z": pl.Float64,
        },
    )
    return System(data=data, box=np.asarray(atoms.cell, dtype=float))


# ============================================================
# Symmetry detection (cubic only for v1)
# ============================================================


def _detect_crystal_class(
    unitcell: System,
    symprec: float,
    ignore_elements: bool = False,
) -> Tuple[str, int]:
    """Detect crystal class via spglib. Returns (crystal_class, spacegroup_number).

    crystal_class is ``"cubic"`` (space groups 195-230) or ``"hexagonal"``
    (168-194). Other classes raise ValueError.

    Parameters
    ----------
    ignore_elements : bool
        If True, all atoms are treated as the same species for the spglib
        symmetry analysis (atomic_number=1 everywhere). Use this for
        chemically-disordered cells (SQS HEAs etc.) where the parent lattice
        is cubic / hexagonal but species disorder breaks spglib's strict
        detection. The crystal class controls the strain-mode set; phonopy
        and the force-constant calculation still see the real species.
    """
    try:
        import spglib
    except ImportError:
        raise ImportError("spglib is required (typically bundled with phonopy)")

    cell = np.asarray(unitcell.box.box, dtype=float)
    pos_cart = unitcell.get_positions().to_numpy() - unitcell.box.origin
    frac = pos_cart @ np.linalg.inv(cell)

    n = len(unitcell.data)
    if ignore_elements:
        numbers = np.ones(n, dtype=int)
        # Disordered cells typically have ~0.1 A species-driven atomic
        # displacements from ideal lattice sites; bump symprec accordingly.
        # This only relaxes the spglib *parent-lattice* check; phonopy's
        # own symmetry handling for displacements uses self.symprec unchanged.
        sym_for_detect = max(symprec, 0.5)
    else:
        elements = unitcell.data["element"].to_numpy()
        unique = list(dict.fromkeys(elements))
        species_map = {e: i + 1 for i, e in enumerate(unique)}
        numbers = np.array([species_map[e] for e in elements])
        sym_for_detect = symprec

    spg = spglib.get_spacegroup((cell, frac, numbers), symprec=sym_for_detect)
    if spg is None:
        raise ValueError("spglib failed to determine space group of unitcell")
    sg_num = int(spg.split("(")[-1].rstrip(")").strip())

    if 195 <= sg_num <= 230:
        return "cubic", sg_num
    if 168 <= sg_num <= 194:
        return "hexagonal", sg_num
    raise ValueError(
        f"Unsupported crystal class (space group {spg}). QHAElastic supports "
        "cubic (195-230) and hexagonal (168-194). Use "
        "ignore_elements_for_symmetry=True if a chemically disordered cell "
        "has the parent lattice but spglib sees lower symmetry."
    )


# ============================================================
# QHAElastic main class
# ============================================================


class QHAElastic:
    """
    Compute temperature-dependent elastic constants under the
    quasi-harmonic approximation (full QHA, energy-strain method).

    Parameters
    ----------
    unitcell : System
        Reference unit cell (cubic or hexagonal). Must contain element
        information. Relax beforehand if you want V(T=0) to coincide
        with the input.
    calc : CalculatorMP, optional
        If provided, MD path: ``run()`` will compute every static energy and
        supercell force using this calculator. If None, use DFT path:
        ``export_inputs(...)`` then ``import_results(...)``.
    t_min, t_max, t_step : float
        Temperature grid (Kelvin), inclusive of t_min, exclusive of t_max+1.
    volume_strains : sequence of float
        Isotropic volume strains used for the bulk EOS (V(T) determination).
    strain_values : sequence of float
        Strain magnitudes applied within each elastic mode. Must include 0.0;
        the eps=0 column doubles as the V_i base for the bulk EOS.
    supercell : sequence of 3 int, optional
        Supercell repeats. Auto-chosen to ~15 A per axis when None.
    displacement : float
        Finite-displacement amplitude for phonopy (Angstrom).
    mesh : tuple of 3 int
        q-point mesh for thermal_properties.
    symprec : float
        Symmetry tolerance.
    quiet : bool
        Suppress progress prints.
    ignore_elements_for_symmetry : bool
        If True, the spglib symmetry check that selects the strain-mode set
        treats all atoms as a single species (atomic_number=1 everywhere).
        This recognises an SQS HEA's *parent lattice* (e.g. FCC for CrCoNi)
        as cubic / hexagonal, even though strict species-aware detection
        returns P1. Phonopy still sees the real species — full P1
        displacements are generated, no fake-element averaging.
    force_constants_method : {"finite-displacement", "dfpt"}
        How phonopy fits force constants.
        ``"finite-displacement"`` (default): phonopy generates per-(atom, axis)
        displacement cells; one force evaluation per displacement. Use for
        the MD path (NEP / EAM / etc.) and for DFT on small systems.
        ``"dfpt"``: DFT-only. Each (V, mode, eps) cell is one VASP DFPT SCF
        (``IBRION=8``) that returns the full force-constant matrix in one
        shot. Drops cost from ``3N x N_cells`` SCFs to ``N_cells`` SCFs.
    """

    def __init__(
        self,
        unitcell: System,
        calc: Optional[CalculatorMP] = None,
        t_min: float = 0.0,
        t_max: float = 1000.0,
        t_step: float = 100.0,
        volume_strains: Sequence[float] = (-0.06, -0.03, 0.0, 0.03, 0.06),
        strain_values: Sequence[float] = (-0.02, -0.01, 0.0, 0.01, 0.02),
        supercell: Optional[Sequence[int]] = None,
        displacement: float = 0.01,
        mesh: Tuple[int, int, int] = (10, 10, 10),
        symprec: float = 1e-3,
        quiet: bool = False,
        ignore_elements_for_symmetry: bool = False,
        force_constants_method: str = "finite-displacement",
    ) -> None:
        assert "element" in unitcell.data.columns, (
            "unitcell must contain 'element' column. "
            "Use system.set_element(...) before passing to QHAElastic."
        )

        self.unitcell = unitcell
        self.calc = calc
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.t_step = float(t_step)
        self.volume_strains = np.asarray(volume_strains, dtype=float)
        self.strain_values = np.asarray(strain_values, dtype=float)
        self.displacement = float(displacement)
        self.mesh = tuple(int(x) for x in mesh)
        self.symprec = float(symprec)
        self.quiet = bool(quiet)
        if force_constants_method not in ("finite-displacement", "dfpt"):
            raise ValueError(
                "force_constants_method must be 'finite-displacement' or 'dfpt'"
            )
        self.force_constants_method = force_constants_method

        # Strain values must include 0 — it's the parabola anchor / V_i base.
        if not np.any(np.isclose(self.strain_values, 0.0, atol=1e-12)):
            raise ValueError(
                "strain_values must include 0.0 (used as parabola anchor and V_i base)."
            )
        if len(self.volume_strains) < 5:
            raise ValueError(
                "volume_strains must contain at least 5 points (Birch-Murnaghan "
                "3rd order needs >4 data points)."
            )
        if len(self.strain_values) < 3:
            raise ValueError("strain_values must contain at least 3 points.")

        # Detect crystal class to choose strain modes + curvature->Cij formulas.
        # ``ignore_elements_for_symmetry`` lets a chemically-disordered cell
        # (SQS HEA) be classified by its *parent lattice* — e.g., a CrCoNi SQS
        # is recognised as cubic FCC even though spglib's strict (atomic
        # number aware) check returns P1.
        self.crystal_class, self.spacegroup = _detect_crystal_class(
            unitcell, self.symprec, ignore_elements=ignore_elements_for_symmetry
        )
        self.ignore_elements_for_symmetry = bool(ignore_elements_for_symmetry)

        # Default supercell ~15 Å per axis (mirrors mdapy.Phonon).
        if supercell is None:
            lengths = unitcell.box.get_thickness()
            supercell = np.ceil(15.0 / np.asarray(lengths)).astype(int)
        self.supercell = tuple(int(x) for x in supercell)

        # Reference geometry.
        self._ref_cell = np.asarray(unitcell.box.box, dtype=float).copy()
        self._ref_pos = unitcell.get_positions().to_numpy() - unitcell.box.origin
        self._ref_frac = self._ref_pos @ np.linalg.inv(self._ref_cell)
        self._elements = unitcell.data["element"]
        self._n_unitcell = len(self._elements)
        self._ref_volume = float(abs(np.linalg.det(self._ref_cell)))

        # Build the unique cell set (deduplicating eps=0 across modes) and the
        # full (V, mode, eps) grid that maps each grid point to a unique cell.
        self._build_grid()

        # Per-temperature output (filled by compute()).
        self.results_df: Optional[pl.DataFrame] = None
        self.cij_iso: Optional[np.ndarray] = None  # shape (N_T, 6, 6) full Voigt
        self.cij_adi: Optional[np.ndarray] = None
        self.eos_params: Optional[List[np.ndarray]] = None  # BM3 fit per T

    # ----------------------------------------------------------
    # Grid construction
    # ----------------------------------------------------------

    def _cell_key(self, i_V: int, i_mode: int, i_eps: int) -> Tuple:
        """Hash that collapses eps=0 to a single key per V_i (mode-independent)."""
        eps = self.strain_values[i_eps]
        if abs(eps) < 1e-12:
            return ("base", i_V)
        return ("strain", i_V, i_mode, i_eps)

    def _supercell_with_real_species(self, atoms: "PhonopyAtoms") -> System:
        """Convert a phonopy supercell to an mdapy.System with REAL species
        labels. Phonopy tiles the unitcell when supercell_matrix > I; we
        replicate the real species pattern in the same tiling order
        (phonopy's tile order is: copy 0 atoms first, then copy 1, ...)."""
        n_atoms = len(atoms.symbols)
        n_unit = self._n_unitcell
        if n_atoms % n_unit != 0:
            raise RuntimeError(
                f"supercell atoms ({n_atoms}) not divisible by unitcell atoms ({n_unit})"
            )
        n_tiles = n_atoms // n_unit
        # Phonopy with supercell_matrix=N*I lays out atoms as: for each
        # tile (cell vector translation), atoms 0..n_unit-1 copy. So the
        # species pattern repeats every n_unit atoms.
        real_symbols = list(self._elements.to_numpy()) * n_tiles

        data = pl.DataFrame(
            {
                "element": real_symbols,
                "x": atoms.positions[:, 0],
                "y": atoms.positions[:, 1],
                "z": atoms.positions[:, 2],
            },
            schema={
                "element": pl.Utf8,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
            },
        )
        return System(data=data, box=np.asarray(atoms.cell, dtype=float))

    def _build_grid(self) -> None:
        self.unique_cells: List[dict] = []  # one entry per unique unitcell
        cell_key_to_idx: dict = {}
        self.grid: List[dict] = []  # one entry per (V, mode, eps) grid point
        self.strain_modes = _strain_modes_for(self.crystal_class)

        for i_V, vstr in enumerate(self.volume_strains):
            for i_mode, pattern in enumerate(self.strain_modes):
                for i_eps, eps in enumerate(self.strain_values):
                    key = self._cell_key(i_V, i_mode, i_eps)
                    if key not in cell_key_to_idx:
                        # Build this unique cell.
                        v_cell = _apply_isotropic_volume(self._ref_cell, vstr)
                        if abs(eps) < 1e-12:
                            new_cell = v_cell
                        else:
                            voigt = pattern * eps
                            E = _voigt_to_strain_tensor(voigt)
                            F = _strain_to_deformation(E)
                            new_cell = _apply_deformation_to_cell(v_cell, F)
                        new_pos = self._ref_frac @ new_cell

                        sys = System(box=new_cell, pos=new_pos)
                        sys.set_element(self._elements)

                        # phonopy uses the *real* species — no fake-element
                        # tricks. For an SQS HEA this means full P1 displacements
                        # (= 3N or 6N), which is fine for fast NEP/EAM. For DFT
                        # use force_constants_method='dfpt' (1 SCF / cell).
                        phon = Phonopy(
                            unitcell=_system_to_phonopy(sys),
                            supercell_matrix=np.diag(self.supercell),
                            primitive_matrix="auto",
                            symprec=self.symprec,
                        )
                        if self.force_constants_method == "dfpt":
                            # DFPT (VASP IBRION=8) returns the full force-
                            # constant matrix from a single SCF — no
                            # displacement supercells needed.
                            sc_systems = []
                        else:  # "finite-displacement"
                            phon.generate_displacements(distance=self.displacement)
                            sc_systems = [
                                self._supercell_with_real_species(s)
                                for s in phon.supercells_with_displacements
                            ]
                        unique_idx = len(self.unique_cells)
                        self.unique_cells.append(
                            {
                                "key": key,
                                "i_V": i_V,
                                "V_strain": float(vstr),
                                "V_base": float(self._ref_volume * (1.0 + vstr)),
                                "unitcell": sys,
                                "phonon": phon,
                                "supercells": sc_systems,
                                "n_disp": len(sc_systems),
                                "volume": float(abs(np.linalg.det(new_cell))),
                                "E_static": None,  # eV per unit cell
                                "forces": None,  # list of (N, 3) ndarray
                                "F_phonon_T": None,  # ndarray (N_T,) eV per unit cell
                                "Cv_T": None,  # ndarray (N_T,) eV/K per unit cell
                                "T": None,  # ndarray (N_T,)
                            }
                        )
                        cell_key_to_idx[key] = unique_idx

                    self.grid.append(
                        {
                            "i_V": i_V,
                            "i_mode": i_mode,
                            "i_eps": i_eps,
                            "V_strain": float(vstr),
                            "eps": float(self.strain_values[i_eps]),
                            "unique_idx": cell_key_to_idx[key],
                        }
                    )

        if not self.quiet:
            n_disp_total = sum(uc["n_disp"] for uc in self.unique_cells)
            print(
                f"[QHAElastic] grid built: {len(self.unique_cells)} unique unitcells, "
                f"{len(self.grid)} (V, mode, eps) points, "
                f"{n_disp_total} total supercell forces; "
                f"crystal_class={self.crystal_class}, "
                f"force_constants={self.force_constants_method}, "
                f"supercell={self.supercell}, "
                f"V0={self._ref_volume:.3f} A^3"
            )

    # ----------------------------------------------------------
    # MD path
    # ----------------------------------------------------------

    def run(self) -> None:
        """Compute every E_static and supercell force using self.calc."""
        if self.calc is None:
            raise ValueError(
                "calc=None; for DFT path use export_inputs() / import_results()."
            )
        if self.force_constants_method == "dfpt":
            raise ValueError(
                "force_constants_method='dfpt' is DFT-only — use export_inputs() / "
                "import_results() with VASP IBRION=8 outputs. The MD path uses "
                "finite displacements."
            )
        for k, uc in enumerate(self.unique_cells):
            unit = uc["unitcell"]
            unit.calc = self.calc
            uc["E_static"] = float(unit.get_energy())

            forces = []
            for sc in uc["supercells"]:
                sc.calc = self.calc
                f = sc.get_force()
                f -= np.mean(f, axis=0)
                forces.append(np.asarray(f, dtype=float))
            uc["forces"] = forces
            uc["phonon"].forces = np.array(forces)
            uc["phonon"].produce_force_constants(show_drift=False)

            if not self.quiet:
                print(
                    f"[QHAElastic] run {k + 1}/{len(self.unique_cells)}: "
                    f"V_strain={uc['V_strain']:+.3f}, "
                    f"E_static={uc['E_static']:.4f} eV, "
                    f"n_disp={uc['n_disp']}"
                )

    # ----------------------------------------------------------
    # DFT path
    # ----------------------------------------------------------

    def _unique_path(self, root: Path, uc: dict) -> Path:
        key = uc["key"]
        if key[0] == "base":
            return root / f"V-{key[1]}" / "base"
        return root / f"V-{key[1]}" / f"mode-{key[2]}" / f"strain-{key[3]}"

    def export_inputs(self, root: Union[str, os.PathLike]) -> None:
        """Write a VASP POSCAR tree plus a manifest.json under ``root``.

        For ``force_constants_method='finite-displacement'``::

            root/
              manifest.json
              V-{i}/base/                          # mode-independent (eps=0)
                static/POSCAR                       (-> static SCF, get E_static)
                disp-001/POSCAR                     (-> static SCF, get forces)
                ...
              V-{i}/mode-{k}/strain-{j}/           # eps != 0
                static/POSCAR
                disp-001/POSCAR
                ...

        For ``force_constants_method='dfpt'``::

            root/
              manifest.json
              V-{i}/base/POSCAR                    (-> 1 DFPT SCF, get E + force constants)
              V-{i}/mode-{k}/strain-{j}/POSCAR
              ...

        With DFPT each (V, mode, eps) is a single VASP run (the user
        supplies INCAR / KPOINTS / POTCAR — typical settings: IBRION=8,
        NSW=1, EDIFF=1e-8). vasprun.xml contains ``e_fr_energy`` and the
        ``<varray name="hessian">`` block — import_results parses both.
        """
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        manifest = {
            "version": 2,
            "crystal_class": self.crystal_class,
            "strain_modes": [m.tolist() for m in self.strain_modes],
            "spacegroup": self.spacegroup if self.spacegroup is not None else "skipped",
            "supercell": list(self.supercell),
            "displacement": self.displacement,
            "mesh": list(self.mesh),
            "symprec": self.symprec,
            "volume_strains": [float(x) for x in self.volume_strains],
            "strain_values": [float(x) for x in self.strain_values],
            "t_min": self.t_min,
            "t_max": self.t_max,
            "t_step": self.t_step,
            "n_atoms_unitcell": self._n_unitcell,
            "ref_volume_A3": self._ref_volume,
            "unique_cells": [],
            "grid": [],
        }

        for uc in self.unique_cells:
            sub = self._unique_path(root, uc)
            sub.mkdir(parents=True, exist_ok=True)
            if self.force_constants_method == "dfpt":
                # One POSCAR per cell; user runs DFPT (IBRION=8) here, drops
                # vasprun.xml back into the same directory.
                uc["unitcell"].write_poscar(str(sub / "POSCAR"))
            else:
                # finite-displacement: separate static + per-disp subdirs
                (sub / "static").mkdir(parents=True, exist_ok=True)
                uc["unitcell"].write_poscar(str(sub / "static" / "POSCAR"))
                for d, sc_sys in enumerate(uc["supercells"], start=1):
                    d_dir = sub / f"disp-{d:03d}"
                    d_dir.mkdir(parents=True, exist_ok=True)
                    sc_sys.write_poscar(str(d_dir / "POSCAR"))

            manifest["unique_cells"].append(
                {
                    "key": list(uc["key"]),
                    "i_V": uc["i_V"],
                    "V_strain": uc["V_strain"],
                    "n_disp": uc["n_disp"],
                    "volume_A3": uc["volume"],
                    "path": str(sub.relative_to(root)),
                }
            )

        for g in self.grid:
            manifest["grid"].append(
                {
                    "i_V": g["i_V"],
                    "i_mode": g["i_mode"],
                    "i_eps": g["i_eps"],
                    "V_strain": g["V_strain"],
                    "eps": g["eps"],
                    "unique_idx": g["unique_idx"],
                }
            )

        with open(root / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        if not self.quiet:
            if self.force_constants_method == "dfpt":
                print(
                    f"[QHAElastic] exported {len(self.unique_cells)} POSCAR files "
                    f"to {root}/. Run VASP DFPT (IBRION=8) on each, drop "
                    f"vasprun.xml back into the same directory, then call "
                    f"qha.import_results({root!r})."
                )
            else:
                print(
                    f"[QHAElastic] exported {len(self.unique_cells)} unique unitcells "
                    f"to {root}/. Run VASP static SCF in each */static and each "
                    f"*/disp-NNN subdirectory."
                )

    def import_results(self, root: Union[str, os.PathLike]) -> None:
        """Read OSZICAR (energies) and vasprun.xml (forces) back from the
        directory tree created by export_inputs()."""
        root = Path(root)
        with open(root / "manifest.json") as f:
            manifest = json.load(f)

        if len(manifest["unique_cells"]) != len(self.unique_cells):
            raise ValueError(
                "manifest.json grid does not match this QHAElastic instance "
                "(different volume_strains / strain_values?). "
                "Re-create the QHAElastic with the same parameters."
            )

        # Strict check: strain-mode patterns must match. This catches the case
        # where existing DFT data was generated with an older version of the
        # CUBIC_STRAIN_MODES (e.g. mode 0 = [e,e,0] from version 1) and the
        # current code expects a different pattern (mode 0 = [e,-e,0]). Silent
        # mismatch would give garbage C_ij.
        manifest_modes = manifest.get("strain_modes")
        if manifest_modes is not None:
            mm = np.asarray(manifest_modes, dtype=float)
            sm = np.stack([np.asarray(m, dtype=float) for m in self.strain_modes])
            if mm.shape != sm.shape or not np.allclose(mm, sm):
                raise ValueError(
                    "manifest.json strain_modes do not match the current "
                    "CUBIC_STRAIN_MODES / HEXAGONAL_STRAIN_MODES. "
                    "This usually means your DFT data was generated with an "
                    "older version of mdapy. Re-run the relevant strain cells "
                    "with the new modes (run qha.export_inputs() again).\n"
                    f"  manifest: {mm.tolist()}\n"
                    f"  current:  {sm.tolist()}"
                )
        elif manifest.get("version", 1) < 2:
            import warnings

            warnings.warn(
                "manifest.json is version 1 (no strain_modes field). The "
                "cubic mode 0 changed from [e,e,0] to [e,-e,0] in version 2; "
                "if your DFT data predates that change, C12 / C11 will be "
                "incorrect. Re-export with the current code if unsure.",
                stacklevel=2,
            )

        for entry, uc in zip(manifest["unique_cells"], self.unique_cells):
            sub = root / entry["path"]
            if self.force_constants_method == "dfpt":
                # One vasprun.xml per cell; both energy and force_constants
                # come from the single DFPT SCF.
                vrx = sub / "vasprun.xml"
                if not vrx.exists():
                    raise FileNotFoundError(
                        f"DFPT mode expects {vrx}; did you run VASP IBRION=8 here?"
                    )
                uc["E_static"] = _vasprun_last_energy(vrx)
                uc["forces"] = []  # not used in DFPT
                fc = _read_dfpt_force_constants(vrx)
                uc["phonon"].force_constants = fc
            else:
                uc["E_static"] = float(_read_vasp_energy(sub / "static"))
                forces = []
                for d in range(1, entry["n_disp"] + 1):
                    f = _read_vasp_forces(sub / f"disp-{d:03d}")
                    f -= np.mean(f, axis=0)
                    forces.append(f)
                uc["forces"] = forces
                uc["phonon"].forces = np.array(forces)
                uc["phonon"].produce_force_constants(show_drift=False)

        if not self.quiet:
            print(f"[QHAElastic] imported {len(self.unique_cells)} VASP results.")

    # ----------------------------------------------------------
    # Post-processing
    # ----------------------------------------------------------

    def _phonon_thermal(self) -> np.ndarray:
        """Run thermal_properties on every unique cell; cache F_phonon and Cv
        on each entry. Returns the temperature grid."""
        T = None
        for uc in self.unique_cells:
            phon = uc["phonon"]
            phon.run_mesh(self.mesh)
            phon.run_thermal_properties(
                t_min=self.t_min, t_max=self.t_max, t_step=self.t_step
            )
            td = phon.get_thermal_properties_dict()
            if T is None:
                T = np.asarray(td["temperatures"], dtype=float)

            # phonopy returns kJ/mol per primitive cell.
            n_prim = len(phon.primitive)
            scale_eV_per_unit = EV_PER_KJMOL * (self._n_unitcell / n_prim)
            uc["F_phonon_T"] = (
                np.asarray(td["free_energy"], dtype=float) * scale_eV_per_unit
            )
            # heat_capacity in J/K/mol per primitive -> eV/K per unit cell
            uc["Cv_T"] = (
                np.asarray(td["heat_capacity"], dtype=float)
                * (EV_PER_KJMOL / 1000.0)  # J/(K*mol) -> kJ/(K*mol)/1000 -> eV/K
                * (self._n_unitcell / n_prim)
            )
            uc["T"] = T
        return T

    def _eos_fit_at_T(self, T_idx: int) -> Tuple[float, float, np.ndarray]:
        """Fit Birch-Murnaghan F(V) at temperature index T_idx.

        Uses both the base cells (eps=0) and the isotropic-mode strain cells
        ([e,e,e,0,0,0] applied to each V_base). The latter cells are exact
        isotropic deformations of the base lattice; their actual volume is
        ``V_base * det(F) = V_base * (1+2*eps)**1.5`` (the Cholesky-of-I+2E
        convention used here). Including them gives 3x the V-range and a
        much more stable BM3 fit at high T — without these the 4-parameter
        BM3 on 5 base cells frequently produces unphysical Bp<0 once V_eq
        drifts near the edge of the V_base window.

        Returns (V_eq, B_T_GPa, fit_params).
        """
        iso = self._isotropic_mode_index()
        volumes: list[float] = []
        F_total: list[float] = []
        for uc in self.unique_cells:
            key = uc["key"]
            if key[0] == "base":
                V_actual = uc["V_base"]
            elif iso is not None and key[0] == "strain" and key[2] == iso:
                eps_val = float(self.strain_values[key[3]])
                V_actual = uc["V_base"] * (1.0 + 2.0 * eps_val) ** 1.5
            else:
                continue
            volumes.append(V_actual)
            F_total.append(uc["E_static"] + uc["F_phonon_T"][T_idx])
        volumes = np.asarray(volumes, dtype=float)
        F_total = np.asarray(F_total, dtype=float)
        order = np.argsort(volumes)
        volumes = volumes[order]
        F_total = F_total[order]

        eos = get_eos("birch_murnaghan")
        params = fit_to_eos(volumes, F_total, eos)
        # params = [F_0, B_0 (eV/A^3), B'_0, V_0 (A^3)]
        F_0, B_0, _Bp, V_0 = params
        return (
            float(V_0),
            float(B_0 * EV_PER_A3_TO_GPA),
            np.asarray(params, dtype=float),
        )

    def _isotropic_mode_index(self) -> Optional[int]:
        """Index of the pure isotropic strain mode [1,1,1,0,0,0] in
        ``self.strain_modes``, or None if absent. For cubic this is mode 1;
        for hexagonal it is mode 4 (the [e,e,e,...] entry)."""
        target = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        for i, pat in enumerate(self.strain_modes):
            if np.allclose(pat, target):
                return i
        return None

    def _interp_F_at_V(
        self, V_target: float, i_mode: int, i_eps: int, T_idx: int
    ) -> float:
        """Interpolate F_total over outer V_base scaling at fixed (mode, eps, T).

        We parameterize F by V_base (= V_ref * (1 + v_strain)), the "outer"
        volume scaling. The mode strain on top deforms the cell further but
        is held fixed at eps_j — so V_base is the right interpolation axis.
        """
        V_list = []
        F_list = []
        for g in self.grid:
            if g["i_mode"] != i_mode or g["i_eps"] != i_eps:
                continue
            uc = self.unique_cells[g["unique_idx"]]
            V_list.append(uc["V_base"])
            F_list.append(uc["E_static"] + uc["F_phonon_T"][T_idx])
        V_arr = np.asarray(V_list, dtype=float)
        F_arr = np.asarray(F_list, dtype=float)
        order = np.argsort(V_arr)
        V_arr = V_arr[order]
        F_arr = F_arr[order]
        if len(V_arr) >= 4:
            from scipy.interpolate import CubicSpline

            cs = CubicSpline(V_arr, F_arr, extrapolate=True)
            return float(cs(V_target))
        return float(np.interp(V_target, V_arr, F_arr))

    def _curvatures_at_T(self, T_idx: int, V_eq: float) -> Tuple[float, ...]:
        """Parabolic fit of F vs eps at fixed V = V_eq for each strain mode.
        Returns the per-mode kappa values in GPa (length = number of modes for
        the detected crystal class)."""
        kappas = []
        for i_mode in range(len(self.strain_modes)):
            xs, ys = [], []
            for i_eps, eps in enumerate(self.strain_values):
                F = self._interp_F_at_V(V_eq, i_mode, i_eps, T_idx)
                xs.append(eps)
                ys.append(F)
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            p = np.polyfit(xs, ys, 2)
            a = p[0]
            kappa = a * EV_PER_A3_TO_GPA / V_eq  # GPa
            kappas.append(kappa)
        return tuple(kappas)

    def _isothermal_to_adiabatic(
        self,
        C_iso: np.ndarray,
        T: float,
        V_eq: float,
        alpha_V: float,
        Cv_per_unit: float,
    ) -> np.ndarray:
        """Convert a 6x6 isothermal Cij to adiabatic via thermal stress.

            C^S_ij = C^T_ij + T * V * lambda_i * lambda_j / Cv

        For cubic with isotropic volume thermal expansion alpha_V:
            lambda_n = alpha_V * B_T  for n=1,2,3 (normal directions)
            lambda_n = 0              for n=4,5,6 (shears)

        For hexagonal: assume isotropic alpha (alpha_a = alpha_c = alpha_V/3).
        Same formula applies because the thermal strain pattern is (1,1,1,0,0,0)/3
        in Voigt; lambda = -C : alpha_strain reduces to:
            lambda_1 = lambda_2 = -alpha_V/3 * (C11 + C12 + C13)
            lambda_3 =           -alpha_V/3 * (2 C13 + C33)
            lambda_4..6 = 0
        Real hex with anisotropic alpha (alpha_a != alpha_c) needs more
        information than QHA at iso-volume gives; we use the isotropic
        approximation here.
        """
        if T <= 0.0 or Cv_per_unit <= 0.0 or alpha_V == 0.0:
            return C_iso.copy()

        # Thermal-strain Voigt vector for isotropic expansion:
        # alpha_eps = (1, 1, 1, 0, 0, 0) * alpha_V / 3
        alpha_eps = np.array(
            [alpha_V / 3.0, alpha_V / 3.0, alpha_V / 3.0, 0.0, 0.0, 0.0]
        )
        # lambda_i = -C_iso[i, :] @ alpha_eps   (thermal-stress derivative)
        lam = -C_iso @ alpha_eps  # GPa/K
        # T V lam_i lam_j / Cv has units K^2 (GPa/K)^2 A^3 / (eV/K)
        # = GPa^2 A^3 / eV; eV/A^3 = 160.218 GPa, so dividing by 160.218 -> GPa.
        prefactor = (T * V_eq / Cv_per_unit) / EV_PER_A3_TO_GPA
        delta = prefactor * np.outer(lam, lam)
        return C_iso + delta

    def compute(self) -> pl.DataFrame:
        """Execute all post-processing. Returns the results DataFrame."""
        # Sanity check: every unique cell must have E_static and forces filled.
        for k, uc in enumerate(self.unique_cells):
            if uc["E_static"] is None or uc["forces"] is None:
                raise RuntimeError(
                    f"unique_cell #{k} has no results — call run() (MD path) "
                    "or import_results() (DFT path) before compute()."
                )

        T = self._phonon_thermal()
        n_T = len(T)

        # Output containers — full 6x6 Cij matrix per T (iso + adi).
        V_eq_arr = np.zeros(n_T)
        BT_arr = np.zeros(n_T)
        cij_iso = np.zeros((n_T, 6, 6))
        cij_adi = np.zeros((n_T, 6, 6))
        K_arr = np.zeros(n_T)
        G_arr = np.zeros(n_T)
        E_arr = np.zeros(n_T)
        nu_arr = np.zeros(n_T)
        stable_arr = np.zeros(n_T, dtype=bool)
        eos_params = []

        # Equilibrium volume at T=0 for alpha computation.
        V_eq_prev = None

        V_base_min = self._ref_volume * (1.0 + float(self.volume_strains.min()))
        V_base_max = self._ref_volume * (1.0 + float(self.volume_strains.max()))
        out_of_range_T: List[float] = []

        # Pass 1: equilibrium volume and B_T at each T (also flag extrapolation).

        for i_T in range(n_T):
            V_eq, B_T, params = self._eos_fit_at_T(i_T)
            V_eq_arr[i_T] = V_eq
            BT_arr[i_T] = B_T
            eos_params.append(params)
            if not (V_base_min <= V_eq <= V_base_max):
                out_of_range_T.append(float(T[i_T]))

        # Pass 2: centered-diff alpha (used both for DataFrame output and the
        # adiabatic-isothermal conversion).

        alpha_arr = np.zeros(n_T)
        for i in range(n_T):
            if n_T == 1:
                break
            if i == 0:
                dT = T[1] - T[0]
                alpha_arr[i] = (
                    (V_eq_arr[1] - V_eq_arr[0]) / dT / V_eq_arr[0] if dT > 0 else 0.0
                )
            elif i == n_T - 1:
                dT = T[i] - T[i - 1]
                alpha_arr[i] = (
                    (V_eq_arr[i] - V_eq_arr[i - 1]) / dT / V_eq_arr[i]
                    if dT > 0
                    else 0.0
                )
            else:
                dT = T[i + 1] - T[i - 1]
                alpha_arr[i] = (
                    (V_eq_arr[i + 1] - V_eq_arr[i - 1]) / dT / V_eq_arr[i]
                    if dT > 0
                    else 0.0
                )

        # Pass 3: per-T 6x6 Cij (isothermal + adiabatic).
        for i_T in range(n_T):
            V_eq = V_eq_arr[i_T]
            kappas = self._curvatures_at_T(i_T, V_eq)
            C_iso = _build_cij_matrix(self.crystal_class, kappas)
            cij_iso[i_T] = C_iso

            Cv_per_unit = self._interp_Cv_at_V(V_eq, i_T)
            C_adi = self._isothermal_to_adiabatic(
                C_iso, T[i_T], V_eq, alpha_arr[i_T], Cv_per_unit
            )
            cij_adi[i_T] = C_adi

            # VRH from full 6x6 (works for any crystal class).
            from mdapy.elastic import ElasticTensor

            v = ElasticTensor(C_adi).vrh()
            K_arr[i_T] = v["K_H"]
            G_arr[i_T] = v["G_H"]
            E_arr[i_T] = v["E"]
            nu_arr[i_T] = v["nu"]
            stable_arr[i_T] = _born_stable(self.crystal_class, C_iso)

        if out_of_range_T:
            import warnings

            warnings.warn(
                f"V_eq at T={out_of_range_T} K is outside the V_base sampling "
                f"window [{V_base_min:.2f}, {V_base_max:.2f}] A^3 — "
                f"C_ij at these temperatures is extrapolated and may be unreliable. "
                "Widen volume_strains to cover the thermal expansion range.",
                stacklevel=2,
            )

        self.cij_iso = cij_iso  # (n_T, 6, 6)
        self.cij_adi = cij_adi
        self.eos_params = eos_params

        # Per-class independent components for the headline DataFrame.
        cols: dict = {
            "T": T,
            "V_eq": V_eq_arr,
            "alpha": alpha_arr,
            "B_T": BT_arr,
        }
        if self.crystal_class == "cubic":
            ind = [(0, 0, "C11"), (0, 1, "C12"), (3, 3, "C44")]
        elif self.crystal_class == "hexagonal":
            ind = [
                (0, 0, "C11"),
                (0, 1, "C12"),
                (0, 2, "C13"),
                (2, 2, "C33"),
                (3, 3, "C44"),
            ]
        else:
            ind = [(i, j, f"C{i+1}{j+1}") for i in range(6) for j in range(i, 6)]
        for i, j, name in ind:
            cols[f"{name}_iso"] = cij_iso[:, i, j]
            cols[f"{name}_adi"] = cij_adi[:, i, j]
        cols.update(
            {
                "K_VRH": K_arr,
                "G_VRH": G_arr,
                "E_VRH": E_arr,
                "nu_VRH": nu_arr,
                "stable": stable_arr,
            }
        )
        self.results_df = pl.DataFrame(cols)
        return self.results_df

    def _interp_Cv_at_V(self, V_target: float, T_idx: int) -> float:
        """Interpolate Cv (eV/K per unit cell) over V_i base cells -> V_target."""
        V_list, Cv_list = [], []
        for uc in self.unique_cells:
            if uc["key"][0] != "base":
                continue
            V_list.append(uc["V_base"])
            Cv_list.append(uc["Cv_T"][T_idx])
        V_arr = np.asarray(V_list, dtype=float)
        Cv_arr = np.asarray(Cv_list, dtype=float)
        order = np.argsort(V_arr)
        return float(np.interp(V_target, V_arr[order], Cv_arr[order]))

    def get_results(self) -> pl.DataFrame:
        if self.results_df is None:
            raise RuntimeError("call compute() first.")
        return self.results_df

    # ----------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------

    def plot(
        self,
        adiabatic: bool = True,
        fig: Optional["Figure"] = None,
        axes=None,
    ):
        """2x2 summary figure: (a) F(V) at every T with the V_eq(T) trajectory,
        (b) V_eq vs T, (c) alpha_V vs T, (d) independent C_ij vs T.

        If ``adiabatic`` is True, both isothermal and adiabatic C_ij are drawn
        (solid = adiabatic, dashed = isothermal); otherwise only isothermal.
        """
        if self.results_df is None:
            raise RuntimeError("call compute() first.")
        if fig is None and axes is None:
            from mdapy.plotset import set_figure

            fig, axes = set_figure(figsize=(17.0, 13.0), nrow=2, ncol=2)
        ax_F = axes[0][0]
        ax_V = axes[0][1]
        ax_a = axes[1][0]
        ax_C = axes[1][1]

        df = self.results_df
        T = df["T"].to_numpy()
        n_atoms = self._n_unitcell

        # ---------- Panel (a): F(V) curves at each T ----------
        base_V, base_F = [], []
        for uc in self.unique_cells:
            if uc["key"][0] != "base":
                continue
            base_V.append(uc["V_base"])
            base_F.append(np.asarray(uc["E_static"] + uc["F_phonon_T"]))
        base_V = np.asarray(base_V, dtype=float)
        order = np.argsort(base_V)
        base_V = base_V[order]
        F_pts = np.asarray(base_F)[order]  # (n_V, n_T)

        from phonopy.qha.eos import get_eos

        eos = get_eos("birch_murnaghan")
        V_smooth = np.linspace(base_V.min(), base_V.max(), 200)

        for i_T in range(len(T)):
            F_smooth = eos(V_smooth, self.eos_params[i_T]) / n_atoms
            ax_F.plot(V_smooth, F_smooth, "-", color="tab:red", lw=1.0)
            ax_F.plot(base_V, F_pts[:, i_T] / n_atoms, "o", color="tab:red", ms=3)

        V_eq = df["V_eq"].to_numpy()
        F_min = np.array(
            [
                eos(np.array([V_eq[i]]), self.eos_params[i])[0] / n_atoms
                for i in range(len(T))
            ]
        )
        ax_F.plot(V_eq, F_min, "k--", lw=1.5)
        ax_F.set_xlabel(r"Volume (Å$^3$)")
        ax_F.set_ylabel("Helmholtz Energy (eV/atom)")

        # ---------- Panel (b): V_eq(T) ----------
        ax_V.plot(T, V_eq, "-o", color="tab:red", lw=1.5)
        ax_V.set_xlabel("Temperature (K)")
        ax_V.set_ylabel(r"Volume (Å$^3$)")
        ax_V.set_xlim(T.min(), T.max())

        # ---------- Panel (c): alpha_V(T) ----------
        ax_a.plot(T, df["alpha"].to_numpy() * 1e6, "-o", color="tab:red", lw=1.5)
        ax_a.set_xlabel("Temperature (K)")
        ax_a.set_ylabel(r"$\alpha_V \times 10^6$ (1/K)")
        ax_a.set_xlim(T.min(), T.max())

        # ---------- Panel (d): C_ij(T) ----------
        if self.crystal_class == "cubic":
            names = [("C11", r"C_{11}"), ("C12", r"C_{12}"), ("C44", r"C_{44}")]
        elif self.crystal_class == "hexagonal":
            names = [
                ("C11", r"C_{11}"),
                ("C12", r"C_{12}"),
                ("C13", r"C_{13}"),
                ("C33", r"C_{33}"),
                ("C44", r"C_{44}"),
            ]
        else:
            names = []

        if adiabatic:
            for col, lab in names:
                (line,) = ax_C.plot(
                    T,
                    df[f"{col}_adi"].to_numpy(),
                    "--",
                    label=rf"${lab}^{{S}}$",
                )
                ax_C.plot(
                    T,
                    df[f"{col}_iso"].to_numpy(),
                    "-o",
                    color=line.get_color(),
                    label=rf"${lab}^{{T}}$",
                )
        else:
            for col, lab in names:
                ax_C.plot(T, df[f"{col}_iso"].to_numpy(), "-o", label=rf"${lab}$")
        ax_C.set_xlabel("Temperature (K)")
        ax_C.set_ylabel("Elastic constant (GPa)")
        ax_C.set_xlim(T.min(), T.max())
        ax_C.legend(ncol=2 if adiabatic else 1, fontsize="small")
        return fig, axes


# ============================================================
# Minimal VASP output readers (no pymatgen / ASE dependency)
# ============================================================


def _read_vasp_energy(static_dir: Path) -> float:
    """Read total energy (eV) from OSZICAR (last F= line) or vasprun.xml."""
    osz = static_dir / "OSZICAR"
    if osz.exists():
        last_F = None
        with open(osz) as f:
            for line in f:
                if " F= " in line:
                    last_F = line
        if last_F is None:
            raise ValueError(f"OSZICAR in {static_dir} contains no 'F=' line")
        # Format: "  N F= -.123E+02 E0= -.123E+02 d E =-.123E-04"
        toks = last_F.split()
        for i, t in enumerate(toks):
            if t == "F=":
                return float(toks[i + 1])
        raise ValueError(f"could not parse F= from {osz}")
    vrx = static_dir / "vasprun.xml"
    if vrx.exists():
        return _vasprun_last_energy(vrx)
    raise FileNotFoundError(f"no OSZICAR or vasprun.xml found in {static_dir}")


def _read_vasp_forces(disp_dir: Path) -> np.ndarray:
    """Read forces (N, 3) from vasprun.xml in disp_dir."""
    vrx = disp_dir / "vasprun.xml"
    if not vrx.exists():
        raise FileNotFoundError(f"no vasprun.xml in {disp_dir}")
    return _vasprun_last_forces(vrx)


def _vasprun_last_energy(path: Path) -> float:
    """Parse final 'e_fr_energy' (free energy F) from a vasprun.xml file."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()
    last_energy = None
    for calc in root.iterfind(".//calculation"):
        for energy_block in calc.iterfind("energy"):
            for i in energy_block.iterfind("i"):
                if i.attrib.get("name") == "e_fr_energy":
                    last_energy = float(i.text.strip())
    if last_energy is None:
        raise ValueError(f"could not find e_fr_energy in {path}")
    return last_energy


def _vasprun_last_forces(path: Path) -> np.ndarray:
    """Parse the last forces varray from vasprun.xml."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()
    last_forces: Optional[np.ndarray] = None
    for calc in root.iterfind(".//calculation"):
        for varray in calc.iterfind("varray"):
            if varray.attrib.get("name") == "forces":
                rows = []
                for v in varray.iterfind("v"):
                    rows.append([float(x) for x in v.text.split()])
                last_forces = np.asarray(rows, dtype=float)
    if last_forces is None:
        raise ValueError(f"could not find forces varray in {path}")
    return last_forces


def _read_dfpt_force_constants(vasprun: Path) -> np.ndarray:
    """Parse VASP DFPT (IBRION=8) force constants from vasprun.xml.

    Uses phonopy's parser, which reads the ``<varray name="hessian">`` block
    and converts to phonopy's expected shape ``(N, N, 3, 3)`` where N is the
    number of atoms in the cell. Returns force constants in eV/A^2.
    """
    from phonopy.interface.vasp import parse_force_constants

    fc = parse_force_constants(str(vasprun))
    # phonopy's parse_force_constants returns either an ndarray or
    # (ndarray, elements) depending on version; normalize.
    if isinstance(fc, tuple):
        fc = fc[0]
    return np.asarray(fc, dtype=float)


if __name__ == "__main__":
    # from mdapy import NEP, FIRE, SQS, build_hea

    # hea = build_hea(
    #     ("Cr", "Co", "Ni"),
    #     (1 / 3, 1 / 3, 1 / 3),
    #     "fcc",
    #     3.53,
    #     nx=2,
    #     ny=2,
    #     nz=2,
    #     random_seed=2,
    # )
    # sqs = SQS(
    #     hea,
    #     cutoffs={2: 4.0, 3: 3.0},
    #     max_steps=20000,
    #     n_replicas=4,
    #     seed=1,
    #     T=2.0,
    # ).compute()
    # sqs.is_sqs()
    # hea = sqs.system
    # hea.calc = NEP("/Users/herrwu/mypkg/nep_gen400000.txt")
    # fy = FIRE(hea, optimize_cell=True, hydrostatic_strain=True)
    # fy.run(fmax=1e-4, steps=1000, show_process=False)

    # qha = QHAElastic(
    #     hea,
    #     calc=hea.calc,
    #     t_min=0,
    #     t_max=1000,
    #     t_step=100,
    #     volume_strains=(-0.06, -0.03, 0.0, 0.03, 0.06),
    #     strain_values=(-0.02, -0.01, 0.0, 0.01, 0.02),
    #     supercell=(1, 1, 1),
    #     mesh=(10, 10, 10),
    #     ignore_elements_for_symmetry=True,
    #     force_constants_method="dfpt",
    # )
    # qha.export_inputs("qha_dft")
    # qha.run()
    # df = qha.compute()
    # print(df)
    # import matplotlib.pyplot as plt

    # qha.plot()
    # plt.show()
    from mdapy import System, build_crystal, FIRE, NEP
    import matplotlib.pyplot as plt

    # nep = NEP("tests/input_files/UNEP-v1.txt")
    # unit = build_crystal("Ni", "fcc", 3.52)
    # unit.calc = nep
    # fy = FIRE(unit, optimize_cell=True, hydrostatic_strain=True)
    # assert fy.run(fmax=1e-4, steps=1000, show_process=False)
    unit = System("compare_qha_md/Ni.POSCAR")
    qha_dft = QHAElastic(
        unit,
        calc=unit.calc,
        t_min=0,
        t_max=1000,
        t_step=100,
        volume_strains=(-0.06, -0.03, 0.0, 0.03, 0.06),
        strain_values=(-0.02, -0.01, 0.0, 0.01, 0.02),
        supercell=(2, 2, 2),
        mesh=(10, 10, 10),
    )
    qha_dft.export_inputs("compare_qha_md/dft_out_new")
    # qha_dft.import_results("compare_qha_md/dft_out")
    # df = qha_dft.compute()
    # qha_dft.plot()
    # plt.show()
