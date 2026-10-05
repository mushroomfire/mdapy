# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

import warnings
import numpy as np
from typing import List, Optional, Sequence, Tuple, Union
from mdapy.minimizer import FIRE
from mdapy import System
from mdapy.calculator import CalculatorMP

# 1 eV/Å^3 in GPa
EV_A3_TO_GPA = 160.2176621
# 1 GPa·Å in N/m: converts the vacuum-diluted 3D stress of a 2D cell to a
# 2D stress (force per unit length) when multiplied by the cell height.
GPA_A_TO_N_M = 0.1

# Default strain magnitudes. Large shear strains (pymatgen's 6%) are fine for
# bulk crystals but bias C66 of soft, strongly nonlinear 2D sheets (graphene
# C66 is ~2% too low), so 2D uses 1%.
DEFAULT_NORM_STRAINS = (-0.01, -0.005, 0.005, 0.01)
DEFAULT_SHEAR_STRAINS_3D = (-0.06, -0.03, 0.03, 0.06)
DEFAULT_SHEAR_STRAINS_2D = (-0.01, -0.005, 0.005, 0.01)

# In-plane Voigt indices of a 2D material: xx, yy, xy
_VOIGT_2D = [0, 1, 5]

# ============================================================
# Low-level helpers
# ============================================================


def _strain_from_index_amount(idx: Tuple[int, int], amount: float) -> np.ndarray:
    """
    Build a symmetric 3×3 strain tensor with `amount` at position idx
    (and its transpose), all other entries zero.

    Matches pymatgen Strain.from_index_amount for 2-tuple idx.
    """
    e = np.zeros((3, 3))
    e[idx[0], idx[1]] = amount
    e[idx[1], idx[0]] = amount  # symmetrise
    return e


def _strain_to_deformation(strain: np.ndarray) -> np.ndarray:
    """
    Convert a symmetric strain tensor to an upper-triangular deformation
    matrix F such that  E = ½(Fᵀ F − I) = strain.

    Matches pymatgen convert_strain_to_deformation(shape='upper'):
        Fᵀ F = 2·strain + I   →  F = cholesky(2·strain + I)
    """
    M = 2.0 * strain + np.eye(3)
    return np.linalg.cholesky(M).T  # numpy returns lower triangular, .T gives upper


def strain_from_deformation(deformation: np.ndarray) -> np.ndarray:
    """
    Green-Lagrange strain from a deformation gradient F.

        E = ½(Fᵀ F − I)

    Matches pymatgen Strain.from_deformation.

    Parameters
    ----------
    deformation : (3,3) deformation matrix F

    Returns
    -------
    strain : (3,3) symmetric strain tensor in Voigt-ready form
    """
    F = np.asarray(deformation, dtype=float)
    return 0.5 * (F.T @ F - np.eye(3))


def strain_to_voigt(strain: np.ndarray) -> np.ndarray:
    """
    3×3 symmetric strain tensor → 6-vector Voigt notation.
    Convention (matches pymatgen):  [e11, e22, e33, 2e23, 2e13, 2e12]
    """
    e = strain
    return np.array(
        [
            e[0, 0],
            e[1, 1],
            e[2, 2],
            2.0 * e[1, 2],
            2.0 * e[0, 2],
            2.0 * e[0, 1],
        ]
    )


def stress_to_voigt(stress: np.ndarray) -> np.ndarray:
    """
    3×3 stress tensor → 6-vector Voigt notation.
    Convention: [s11, s22, s33, s23, s13, s12]
    """
    s = stress
    return np.array(
        [
            s[0, 0],
            s[1, 1],
            s[2, 2],
            s[1, 2],
            s[0, 2],
            s[0, 1],
        ]
    )


def apply_deformation_to_cell(cell: np.ndarray, deformation: np.ndarray) -> np.ndarray:
    """
    Apply deformation matrix F to a cell (row-vector convention).

    pymatgen applies F to lattice row vectors as:  new_lattice = old_lattice @ F.T
    (because lattice rows are basis vectors, F acts on column vectors)

    Parameters
    ----------
    cell        : (3,3) original cell, row vectors [a; b; c]
    deformation : (3,3) deformation gradient F

    Returns
    -------
    new_cell : (3,3) deformed cell
    """
    return cell @ deformation.T


def apply_deformation_to_positions(
    positions: np.ndarray, old_cell: np.ndarray, new_cell: np.ndarray
) -> np.ndarray:
    """
    Map Cartesian positions into the deformed cell via fractional coordinates.
    Atoms stay at the same fractional coordinates — only the box changes.
    """
    frac = positions @ np.linalg.inv(old_cell)
    return frac @ new_cell


def layer_height(cell: np.ndarray) -> float:
    """
    Height of the cell perpendicular to the layer, h = V / |a x b|.

    The stress of a vacuum-padded 2D cell is diluted by h, so the 2D stress
    (N/m) is ``stress_3d (GPa) * h (Å) * 0.1``.
    """
    cell = np.asarray(cell, dtype=float)
    return abs(np.linalg.det(cell)) / np.linalg.norm(np.cross(cell[0], cell[1]))


def _check_2d_system(system: System, tol: float = 1e-6) -> float:
    """
    A 2D material must lie in the xy plane: the first two cell vectors span
    the layer (zero z-component) and the vacuum is along the third.

    Returns the vacuum thickness along the third cell vector in Å.
    """
    cell = system.box.box
    scale = np.linalg.norm(cell[:2], axis=1).max()
    if np.any(np.abs(cell[:2, 2]) > tol * scale) or cell[2, 2] <= 0:
        raise ValueError(
            "For dim=2 the layer must lie in the xy plane: the first two cell "
            "vectors need zero z-components and the vacuum must be along the "
            "third cell vector (positive z)."
        )
    # Largest empty slab along each cell vector, from the periodic gaps
    # between sorted fractional coordinates.
    pos = system.get_positions().to_numpy() - system.box.origin
    frac = np.sort((pos @ np.linalg.inv(cell)) % 1.0, axis=0)
    gaps = np.diff(np.vstack([frac, frac[:1] + 1.0]), axis=0).max(axis=0)
    gaps *= system.box.get_thickness()
    if gaps[2] < gaps[:2].max():
        raise ValueError(
            f"For dim=2 the vacuum must be along the third cell vector, but the "
            f"largest empty gaps along a, b, c are {np.round(gaps, 2).tolist()} Å. "
            "Rotate the structure so the layer lies in the xy plane."
        )
    return gaps[2]


# ============================================================
# 1. DeformedStructureSet
# ============================================================


class DeformedStructureSet:
    """
    Generate a set of deformed cells for elastic constant fitting.

    Deformation strategy (identical to pymatgen):

      - Normal modes (0,0) (1,1) (2,2) : each strain in `norm_strains`

      - Shear  modes (0,1) (0,2) (1,2) : each strain in `shear_strains`

      Total configurations = 3xlen(norm_strains) + 3xlen(shear_strains)

    For ``dim=2`` only the in-plane modes (0,0) (1,1) and (0,1) are generated,
    giving 2xlen(norm_strains) + len(shear_strains) configurations. The layer
    must lie in the xy plane with vacuum along the third cell vector.

    Parameters
    ----------
    system          : optimized structure
    norm_strains    : normal strain magnitudes (default same as pymatgen)
    shear_strains   : shear  strain magnitudes, defaults is
                      (-0.06, -0.03, 0.03, 0.06) for 3D (same as pymatgen) and
                      (-0.01, -0.005, 0.005, 0.01) for 2D
    dim             : 3 for bulk crystals, 2 for 2D materials

    Attributes
    ----------
    deformations    : list of (3,3) deformation matrices F
    deformed_systems  : list of deformed systems
    """

    def __init__(
        self,
        system: System,
        norm_strains: Sequence[float] = DEFAULT_NORM_STRAINS,
        shear_strains: Optional[Sequence[float]] = None,
        dim: int = 3,
    ):
        assert "element" in system.data.columns, (
            "system must contain element information."
        )
        assert dim in (2, 3), "dim must be 2 or 3."
        self.dim = dim
        self.element = system.data["element"]
        self.cell = system.box.box.copy()
        self.positions = system.get_positions().to_numpy() - system.box.origin
        if dim == 2:
            _check_2d_system(system)
            norm_modes = [(0, 0), (1, 1)]
            shear_modes = [(0, 1)]
        else:
            norm_modes = [(0, 0), (1, 1), (2, 2)]
            shear_modes = [(0, 1), (0, 2), (1, 2)]
        if shear_strains is None:
            shear_strains = (
                DEFAULT_SHEAR_STRAINS_2D if dim == 2 else DEFAULT_SHEAR_STRAINS_3D
            )

        self.deformations: List[np.ndarray] = []
        self.deformed_systems: List[System] = []

        # Normal modes: (0,0), (1,1), (2,2)
        for ind in norm_modes:
            for amount in norm_strains:
                strain = _strain_from_index_amount(ind, amount)
                defo = _strain_to_deformation(strain)
                self._add(defo)

        # Shear modes: (0,1), (0,2), (1,2)
        for ind in shear_modes:
            for amount in shear_strains:
                strain = _strain_from_index_amount(ind, amount)
                defo = _strain_to_deformation(strain)
                self._add(defo)

    def _add(self, defo: np.ndarray):
        new_cell = apply_deformation_to_cell(self.cell, defo)
        new_pos = apply_deformation_to_positions(self.positions, self.cell, new_cell)
        self.deformations.append(defo)
        dfm_system = System(box=new_cell, pos=new_pos)
        dfm_system.set_element(self.element)
        self.deformed_systems.append(dfm_system)

    def __len__(self):
        return len(self.deformations)

    def __iter__(self):
        """Iterate over (deformation, deformed_systems) tuples."""
        return zip(self.deformations, self.deformed_systems)


# ============================================================
# 2. ElasticTensor
# ============================================================


def _fit_independent_strains(
    strains: List[np.ndarray],
    stresses: List[np.ndarray],
    eq_stress: Optional[np.ndarray],
    modes: Sequence[int],
    tol: float,
) -> np.ndarray:
    """
    Fit C_ji = d(stress_j)/d(strain_i) for every independent strain mode i
    in `modes` (Voigt indices 0..5). Columns of modes not listed stay zero.
    """
    # Convert to Voigt arrays
    vstrains = np.array([strain_to_voigt(s) for s in strains])  # (M,6)
    vstresses = np.array([stress_to_voigt(s) for s in stresses])  # (M,6)

    # Equilibrium (zero-strain) stress
    if eq_stress is not None:
        veq_stress = stress_to_voigt(np.asarray(eq_stress, dtype=float))
    else:
        # estimate from nearest-to-zero strains (fallback)
        norms = np.linalg.norm(vstrains, axis=1)
        veq_stress = vstresses[np.argmin(norms)]

    # Build strain-state groups — same logic as pymatgen get_strain_state_dict
    # A "strain state" is identified by which Voigt indices are non-zero
    # (0..5 → e11,e22,e33,2e23,2e13,2e12)
    C = np.zeros((6, 6))

    for ii in modes:
        # Select rows where ONLY component ii is non-zero
        active = np.abs(vstrains[:, ii]) > tol
        other = np.array(
            [
                np.all(np.abs(vstrains[k, [j for j in range(6) if j != ii]]) <= tol)
                for k in range(len(vstrains))
            ]
        )
        mask = active & other

        if not np.any(mask):
            raise ValueError(
                f"No strains found for independent mode {ii}. "
                f"Make sure all Voigt modes {list(modes)} are covered in your "
                f"DeformedStructureSet."
            )

        # Add equilibrium point (zero strain)
        mode_strains = np.vstack([vstrains[mask], np.zeros(6)])  # (K+1, 6)
        mode_stresses = np.vstack([vstresses[mask], veq_stress])  # (K+1, 6)

        # Sort by the active strain component
        order = np.argsort(mode_strains[:, ii])
        mode_strains = mode_strains[order]
        mode_stresses = mode_stresses[order]

        # Fit C_ij = d(stress_j) / d(strain_i)  for each j
        x = mode_strains[:, ii]
        for jj in range(6):
            y = mode_stresses[:, jj]
            C[jj, ii] = np.polyfit(x, y, 1)[0]  # slope only

    return C


class ElasticTensor:
    """
    6x6 elastic stiffness tensor in Voigt notation.

    Build via the class method:
        et = ElasticTensor.from_independent_strains(strain_list, stress_list, eq_stress)

    Attributes
    ----------
    voigt : (6,6) Cij matrix
    """

    def __init__(self, voigt: np.ndarray):
        self.voigt = np.asarray(voigt, dtype=float)

    # ----------------------------------------------------------
    @classmethod
    def from_independent_strains(
        cls,
        strains: List[np.ndarray],
        stresses: List[np.ndarray],
        eq_stress: Optional[np.ndarray] = None,
        tol: float = 1e-10,
    ) -> "ElasticTensor":
        """
        Least-squares fit of elastic constants from independent strain–stress pairs.

        Exact port of pymatgen ElasticTensor.from_independent_strains.

        Algorithm:

        1. Convert all strains/stresses to Voigt 6-vectors.
        2. Group by "strain state" — which Voigt component is active.
        3. For each strain state i (0–5) and each response component j (0–5):
               C_ij = slope of  stress_j  vs  strain_i  (linear polyfit, degree 1)
        4. Include the zero-strain equilibrium point in every group's fit.

        Parameters
        ----------
        strains   : list of (3,3) strain tensors (output of strain_from_deformation)
        stresses  : list of (3,3) stress tensors in GPa
        eq_stress : (3,3) equilibrium stress tensor (at zero strain)
        tol       : zero-out entries smaller than this in the final tensor

        Returns
        -------
        ElasticTensor with .voigt attribute = (6,6) Cij in same units as stresses
        """
        C = _fit_independent_strains(strains, stresses, eq_stress, range(6), tol)

        # Zero out near-zero entries (matches pymatgen .zeroed())
        C[np.abs(C) < tol] = 0.0

        return cls(C)

    # ----------------------------------------------------------
    def print(self):
        print(f"Elastic tensor (GPa):")
        for i, row in enumerate(self.voigt):
            vals = "  ".join(f"{v:8.2f}" for v in row)
            print(f"{vals}")

    def vrh(self):
        """
        Voigt-Reuss-Hill polycrystalline averages.

        Converts the single-crystal elastic tensor into isotropic polycrystalline
        properties by averaging over all grain orientations.

        Three averaging schemes:

          - Voigt : assumes uniform strain across grains → upper bound
          - Reuss : assumes uniform stress across grains → lower bound
          - Hill  : arithmetic mean of Voigt and Reuss → best estimate

        The Voigt-Reuss gap reflects single-crystal anisotropy: a larger gap
        means stronger directional dependence (e.g. HCP Mg has a wider gap
        than FCC Cu).

        Returns
        -------
        dict with keys (all in GPa except nu which is dimensionless):
          K_V, K_R, K_H : bulk modulus
          G_V, G_R, G_H : shear modulus
          E              : Young's modulus, Hill only
          nu             : Poisson's ratio, Hill only
        """
        C = self.voigt
        K_V = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
        G_V = (
            C[0, 0]
            + C[1, 1]
            + C[2, 2]
            - C[0, 1]
            - C[0, 2]
            - C[1, 2]
            + 3 * (C[3, 3] + C[4, 4] + C[5, 5])
        ) / 15.0
        S = np.linalg.inv(C)
        K_R = 1.0 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
        G_R = 15.0 / (
            4 * (S[0, 0] + S[1, 1] + S[2, 2])
            - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
            + 3 * (S[3, 3] + S[4, 4] + S[5, 5])
        )
        K_H = (K_V + K_R) / 2.0
        G_H = (G_V + G_R) / 2.0
        E = 9 * K_H * G_H / (3 * K_H + G_H)
        nu = (3 * K_H - 2 * G_H) / (2 * (3 * K_H + G_H))
        return dict(K_V=K_V, G_V=G_V, K_R=K_R, G_R=G_R, K_H=K_H, G_H=G_H, E=E, nu=nu)

    def print_vrh(self):
        """Print Voigt-Reuss-Hill polycrystalline mechanical properties."""
        r = self.vrh()
        print("Voigt-Reuss-Hill averages:")
        print(
            f"  {'Property':<34s} {'Voigt':>8s}  {'Reuss':>8s}  {'Hill':>8s}  {'Unit'}"
        )
        print(f"  {'-' * 66}")
        print(
            f"  {'Bulk modulus K':<34s} {r['K_V']:>8.2f}  {r['K_R']:>8.2f}  {r['K_H']:>8.2f}  GPa"
        )
        print(
            f"  {'Shear modulus G':<34s} {r['G_V']:>8.2f}  {r['G_R']:>8.2f}  {r['G_H']:>8.2f}  GPa"
        )
        y = "Young's modulus E"
        print(f"  {y:<34s} {'':>8s}  {'':>8s}  {r['E']:>8.2f}  GPa")
        p = "Poisson's ratio nu"
        print(f"  {p:<34s} {'':>8s}  {'':>8s}  {r['nu']:>8.4f}  -")
        print("  Voigt = upper bound  |  Reuss = lower bound  |  Hill = best estimate")
        aniso = abs(r["K_V"] - r["K_R"]) + abs(r["G_V"] - r["G_R"])
        print(f"  Voigt-Reuss gap (K+G): {aniso:.2f} GPa  (larger = more anisotropic)")


class ElasticTensor2D:
    """
    3x3 in-plane elastic stiffness tensor of a 2D material in Voigt notation
    [xx, yy, xy], in N/m (force per unit length).

    The 2D values do not depend on the vacuum thickness. To compare with a
    bulk value in GPa, divide by an assumed layer thickness, see
    :meth:`to_gpa`.

    Build via the class method:
        et = ElasticTensor2D.from_independent_strains(
            strain_list, stress_list, eq_stress, height
        )

    Attributes
    ----------
    voigt : (3,3) Cij matrix, ordering [C11 C12 C16; C12 C22 C26; C16 C26 C66]
    """

    def __init__(self, voigt: np.ndarray):
        self.voigt = np.asarray(voigt, dtype=float)

    @classmethod
    def from_independent_strains(
        cls,
        strains: List[np.ndarray],
        stresses: List[np.ndarray],
        eq_stress: Optional[np.ndarray] = None,
        height: Optional[float] = None,
        tol: float = 1e-10,
    ) -> "ElasticTensor2D":
        """
        Least-squares fit of the in-plane elastic constants of a 2D material.

        Same algorithm as :meth:`ElasticTensor.from_independent_strains`,
        restricted to the in-plane strain modes xx, yy and xy.

        Parameters
        ----------
        strains   : list of (3,3) in-plane strain tensors
        stresses  : list of (3,3) stress tensors in GPa, computed in the
                    vacuum-padded cell (as returned by MD codes or DFT)
        eq_stress : (3,3) equilibrium stress tensor (at zero strain) in GPa
        height    : cell height perpendicular to the layer in Å, i.e.
                    V / |a x b| (see :func:`layer_height`). It converts the
                    vacuum-diluted stress to a 2D stress in N/m.
        tol       : zero-out entries smaller than this in the final tensor

        Returns
        -------
        ElasticTensor2D with .voigt attribute = (3,3) Cij in N/m
        """
        assert height is not None and height > 0, (
            "height (cell height perpendicular to the layer, Å) is required."
        )
        C = _fit_independent_strains(strains, stresses, eq_stress, _VOIGT_2D, tol)
        C = C[np.ix_(_VOIGT_2D, _VOIGT_2D)] * height * GPA_A_TO_N_M
        C[np.abs(C) < tol] = 0.0
        return cls(C)

    def print(self):
        print("Elastic tensor (N/m):")
        for row in self.voigt:
            vals = "  ".join(f"{v:8.2f}" for v in row)
            print(f"{vals}")

    def to_gpa(self, thickness: float) -> np.ndarray:
        """
        Convert to a 3D-equivalent stiffness in GPa by assuming an effective
        layer thickness in Å (e.g. 3.35 Å, the interlayer spacing of graphite,
        for graphene).
        """
        return self.voigt / (thickness * GPA_A_TO_N_M)

    def is_stable(self) -> bool:
        """
        Born mechanical stability: the stiffness matrix must be positive
        definite (C11 > 0, C11*C22 > C12^2, C66 > 0 for orthotropic sheets).
        """
        return bool(np.all(np.linalg.eigvalsh(self.voigt) > 0))

    def engineering_constants(self) -> dict:
        """
        Moduli along the Cartesian axes, from the compliance S = C^-1.

        Returns
        -------
        dict with keys (N/m except the dimensionless Poisson ratios):
          E_x, E_y : 2D Young's modulus along x and y (1/S11, 1/S22)
          nu_xy    : Poisson's ratio, contraction along y under load along x
          nu_yx    : Poisson's ratio, contraction along x under load along y
          G_xy     : in-plane shear modulus (1/S66)
        """
        S = np.linalg.inv(self.voigt)
        return dict(
            E_x=1.0 / S[0, 0],
            E_y=1.0 / S[1, 1],
            nu_xy=-S[0, 1] / S[0, 0],
            nu_yx=-S[0, 1] / S[1, 1],
            G_xy=1.0 / S[2, 2],
        )

    def vrh(self):
        """
        Voigt-Reuss-Hill in-plane averages over all in-plane orientations,
        the 2D counterpart of :meth:`ElasticTensor.vrh`.

        Returns
        -------
        dict with keys (all in N/m except nu which is dimensionless):
          K_V, K_R, K_H : layer (2D bulk / area) modulus
          G_V, G_R, G_H : in-plane shear modulus
          E              : 2D Young's modulus, Hill only
          nu             : Poisson's ratio, Hill only
        """
        C = self.voigt
        S = np.linalg.inv(C)
        K_V = (C[0, 0] + C[1, 1] + 2 * C[0, 1]) / 4.0
        G_V = (C[0, 0] + C[1, 1] - 2 * C[0, 1] + 4 * C[2, 2]) / 8.0
        K_R = 1.0 / (S[0, 0] + S[1, 1] + 2 * S[0, 1])
        G_R = 2.0 / (S[0, 0] + S[1, 1] - 2 * S[0, 1] + S[2, 2])
        K_H = (K_V + K_R) / 2.0
        G_H = (G_V + G_R) / 2.0
        E = 4 * K_H * G_H / (K_H + G_H)
        nu = (K_H - G_H) / (K_H + G_H)
        return dict(K_V=K_V, G_V=G_V, K_R=K_R, G_R=G_R, K_H=K_H, G_H=G_H, E=E, nu=nu)

    def print_vrh(self):
        """Print in-plane mechanical properties of the 2D material."""
        r = self.vrh()
        print("Voigt-Reuss-Hill in-plane averages:")
        print(
            f"  {'Property':<34s} {'Voigt':>8s}  {'Reuss':>8s}  {'Hill':>8s}  {'Unit'}"
        )
        print(f"  {'-' * 66}")
        print(
            f"  {'Layer modulus K':<34s} {r['K_V']:>8.2f}  {r['K_R']:>8.2f}  {r['K_H']:>8.2f}  N/m"
        )
        print(
            f"  {'Shear modulus G':<34s} {r['G_V']:>8.2f}  {r['G_R']:>8.2f}  {r['G_H']:>8.2f}  N/m"
        )
        y = "Young's modulus E"
        print(f"  {y:<34s} {'':>8s}  {'':>8s}  {r['E']:>8.2f}  N/m")
        p = "Poisson's ratio nu"
        print(f"  {p:<34s} {'':>8s}  {'':>8s}  {r['nu']:>8.4f}  -")
        e = self.engineering_constants()
        print("Along the Cartesian axes:")
        print(
            f"  E_x = {e['E_x']:.2f} N/m,  E_y = {e['E_y']:.2f} N/m,  "
            f"G_xy = {e['G_xy']:.2f} N/m"
        )
        print(f"  nu_xy = {e['nu_xy']:.4f},  nu_yx = {e['nu_yx']:.4f}")
        print(f"  Mechanically stable (Born): {self.is_stable()}")


def _get_stress(system: System) -> np.ndarray:
    XX, YY, ZZ, YZ, ZX, XY = system.get_stress()
    stress = np.array([[XX, XY, ZX], [XY, YY, YZ], [ZX, YZ, ZZ]], float) * EV_A3_TO_GPA
    return stress


def get_elastic_constant(
    system: System,
    calc: CalculatorMP,
    norm_strains: Sequence[float] = DEFAULT_NORM_STRAINS,
    shear_strains: Optional[Sequence[float]] = None,
    fmax: float = 1e-4,
    dim: int = 3,
) -> Union[ElasticTensor, ElasticTensor2D]:
    """
    Workflow to compute elastic constants from a system.

    For a 2D material (``dim=2``) the layer must lie in the xy plane with
    vacuum along the third cell vector, and the vacuum should be thicker than
    the potential cutoff. Only the in-plane cell (xx, yy, xy) is relaxed, only
    in-plane strains are applied, and the stress is rescaled by the cell
    height so the result is in N/m and independent of the vacuum thickness.

    Parameters
    ----------
    system          : atomic structure
    calc            : calculator for computing energy, force and stress
    norm_strains    : normal strain magnitudes, defaults is (-0.01, -0.005, 0.005, 0.01)
    shear_strains   : shear  strain magnitudes, defaults is (-0.06, -0.03, 0.03, 0.06)
                      for 3D and (-0.01, -0.005, 0.005, 0.01) for 2D
    fmax            : converge limit for minimization, defaults is 1e-4
    dim             : 3 for bulk crystals, 2 for 2D materials, defaults is 3

    Returns
    -------
    ElasticTensor with .voigt attribute = (6,6) Cij in GPa when dim=3, or
    ElasticTensor2D with .voigt attribute = (3,3) Cij in N/m when dim=2
    """
    assert "element" in system.data.columns, "system must contain element information."
    assert dim in (2, 3), "dim must be 2 or 3."
    if dim == 2:
        vacuum = _check_2d_system(system)
        rc = getattr(calc, "rc", None)
        if rc is not None and vacuum < rc:
            warnings.warn(
                f"Vacuum thickness {vacuum:.2f} Å is smaller than the potential "
                f"cutoff {rc:.2f} Å, so the layer interacts with its periodic "
                "images along z. Increase the cell length along z."
            )
        # Relax only the in-plane cell; the vacuum direction is kept fixed.
        mask = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    else:
        mask = None
    system.calc = calc
    fy = FIRE(system, optimize_cell=True, mask=mask)
    assert fy.run(fmax=fmax, steps=10000, show_process=False), (
        "Fail to cell minimization."
    )
    equi_stress = _get_stress(system)

    dfm_ss = DeformedStructureSet(
        system,
        norm_strains=norm_strains,
        shear_strains=shear_strains,
        dim=dim,
    )
    strain_list, stress_list = [], []
    for defo, dfm_system in dfm_ss:
        dfm_system.calc = calc
        fy = FIRE(dfm_system)
        assert fy.run(fmax=fmax, steps=10000, show_process=False), (
            "Fail to energy minimization."
        )
        stress_list.append(_get_stress(dfm_system))
        strain_list.append(strain_from_deformation(defo))
    if dim == 2:
        return ElasticTensor2D.from_independent_strains(
            strain_list,
            stress_list,
            eq_stress=equi_stress,
            height=layer_height(system.box.box),
        )
    et = ElasticTensor.from_independent_strains(
        strain_list, stress_list, eq_stress=equi_stress
    )
    return et


if __name__ == "__main__":
    from mdapy import NEP, build_crystal

    nep = NEP("tests/input_files/UNEP-v1.txt")
    system = build_crystal("Al", "fcc", 4.05)

    et = get_elastic_constant(system, nep)
    et.print()
    et.print_vrh()
