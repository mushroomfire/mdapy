# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
from typing import List, Optional, Sequence, Tuple
from mdapy.minimizer import FIRE
from mdapy import System
from mdapy.calculator import CalculatorMP

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

    Parameters
    ----------
    system          : optimized structure
    norm_strains    : normal strain magnitudes (default same as pymatgen)
    shear_strains   : shear  strain magnitudes (default same as pymatgen)

    Attributes
    ----------
    deformations    : list of (3,3) deformation matrices F
    deformed_systems  : list of deformed systems
    """

    def __init__(
        self,
        system: System,
        norm_strains: Sequence[float] = (-0.01, -0.005, 0.005, 0.01),
        shear_strains: Sequence[float] = (-0.06, -0.03, 0.03, 0.06),
    ):
        assert "element" in system.data.columns, (
            "system must contain element information."
        )
        self.element = system.data["element"]
        self.cell = system.box.box.copy()
        self.positions = system.get_positions().to_numpy() - system.box.origin

        self.deformations: List[np.ndarray] = []
        self.deformed_systems: List[System] = []

        # Normal modes: (0,0), (1,1), (2,2)
        for ind in [(0, 0), (1, 1), (2, 2)]:
            for amount in norm_strains:
                strain = _strain_from_index_amount(ind, amount)
                defo = _strain_to_deformation(strain)
                self._add(defo)

        # Shear modes: (0,1), (0,2), (1,2)
        for ind in [(0, 1), (0, 2), (1, 2)]:
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
        independent_indices = list(range(6))  # 0..5 → e11,e22,e33,2e23,2e13,2e12

        # For each of the 6 independent modes, collect matching strains/stresses
        C = np.zeros((6, 6))

        for ii in independent_indices:
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
                    f"Make sure all 6 Voigt modes are covered in your DeformedStructureSet."
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


def _get_stress(system: System) -> np.ndarray:
    XX, YY, ZZ, YZ, ZX, XY = system.get_stress()
    stress = np.array([[XX, XY, ZX], [XY, YY, YZ], [ZX, YZ, ZZ]], float) * 160.2176621
    return stress


def get_elastic_constant(
    system: System,
    calc: CalculatorMP,
    norm_strains: Sequence[float] = (-0.01, -0.005, 0.005, 0.01),
    shear_strains: Sequence[float] = (-0.06, -0.03, 0.03, 0.06),
    fmax: float = 1e-4,
) -> ElasticTensor:
    """
    Workflow to compute elastic constants from a system.

    Parameters
    ----------
    system          : atomic structure
    calc            : calculator for computing energy, force and stress
    norm_strains    : normal strain magnitudes, defaults is (-0.01, -0.005, 0.005, 0.01)
    shear_strains   : shear  strain magnitudes, defaults is (-0.06, -0.03, 0.03, 0.06)
    fmax            : converge limit for minimization, defaults is 1e-4

    Returns
    -------
    ElasticTensor with .voigt attribute = (6,6) Cij in same units as stresses
    """
    assert "element" in system.data.columns, "system must contain element information."
    system.calc = calc
    fy = FIRE(system, optimize_cell=True)
    assert fy.run(fmax=fmax, steps=10000, show_process=False), (
        "Fail to cell minimization."
    )
    equi_stress = _get_stress(system)

    dfm_ss = DeformedStructureSet(
        system,
        norm_strains=norm_strains,
        shear_strains=shear_strains,
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
