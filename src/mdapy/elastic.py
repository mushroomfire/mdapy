# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from mdapy.minimizer import FIRE
from mdapy.system import System
import numpy as np
from typing import List


class ElasticConstant:
    """
    A class to compute the elastic constants of a crystal structure
    using finite strain and stress analysis based on molecular dynamics
    or energy minimization simulations.

    The workflow involves applying small deformations to the simulation box,
    relaxing the atomic structure, and evaluating the resulting stress
    to fit the elastic stiffness tensor components.

    Parameters
    ----------
    system : System
        The initial system to be used for deformation and stress calculation.

    Attributes
    ----------
    system : System
        The reference atomic system.
    Cij : np.ndarray or None
        The calculated elastic constants in Voigt notation.
    """

    def __init__(self, system: System):
        self.system = system
        self.Cij = None

    # ------------------------------------------------------------
    # Static methods
    # ------------------------------------------------------------
    @staticmethod
    def triclinic_deform(u: np.ndarray) -> np.ndarray:
        """
        Construct the strain-stress coupling matrix for a triclinic system.

        Parameters
        ----------
        u : np.ndarray
            Strain vector in the order of [uxx, uyy, uzz, uyz, uxz, uxy].

        Returns
        -------
        np.ndarray
            A (6, 18) array representing the linear relationship
            between strain and stress components for fitting.
        """
        uxx, uyy, uzz, uyz, uxz, uxy = u
        return np.array(
            [
                [uxx, 0, 0, uyy, uzz, 0, 0, 0, 0, uxy, 0, 0, 0, 0, uyz, uxz, 0, 0],
                [0, uyy, 0, uxx, 0, uzz, 0, 0, 0, 0, uxy, 0, 0, 0, 0, 0, uxz, 0],
                [0, 0, uzz, 0, uxx, uyy, 0, 0, 0, 0, 0, uxy, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 2 * uyz, 0, 0, 0, 0, 0, uxy, 0, uxx, 0, 0, uxz],
                [0, 0, 0, 0, 0, 0, 0, 2 * uxz, 0, 0, 0, 0, 0, uxy, 0, uxx, uyy, uyz],
                [0, 0, 0, 0, 0, 0, 0, 0, 2 * uxy, uxx, uyy, uzz, uyz, uxz, 0, 0, 0, 0],
            ],
            dtype=float,
        )

    @staticmethod
    def get_pressure(stress: np.ndarray) -> float:
        """
        Compute the hydrostatic pressure from the stress tensor.

        Parameters
        ----------
        stress : array-like
            Stress components in ASE format: [sxx, syy, szz, syz, sxz, sxy].

        Returns
        -------
        float
            The hydrostatic pressure (negative of mean normal stress).
        """
        s = np.array(stress)
        return -np.mean(s[:3])

    @staticmethod
    def relax_struc(
        system: System,
        change_cell: bool = False,
        show_process: bool = False,
        fmax: float = 1e-4,
        steps: int = 10000,
    ):
        """
        Relax the atomic structure using the FIRE minimization algorithm.

        Parameters
        ----------
        system : System
            The system to relax.
        change_cell : bool, optional
            Whether to optimize the cell shape and volume (default: False).
        show_process : bool, optional
            Whether to print relaxation progress (default: False).
        fmax : float, optional
            Force convergence criterion in eV/Å (default: 1e-4).
        steps : int, optional
            Maximum number of FIRE steps (default: 10000).
        """
        fy = FIRE(system, optimize_cell=change_cell)
        fy.run(steps, fmax, show_process=show_process)
        system.calc.results = {}

    # ------------------------------------------------------------
    # Deformation and strain
    # ------------------------------------------------------------
    def get_cart_deformed_cell(
        self, base_cryst: System, axis: int = 0, size: float = 1
    ) -> System:
        """
        Generate a deformed structure by applying a small strain to the box.

        Parameters
        ----------
        base_cryst : System
            The reference (undeformed) system.
        axis : int, optional
            Index of deformation mode:
            0–2 for normal strains, 3–5 for shear strains (default: 0).
        size : float, optional
            Magnitude of deformation in percent (default: 1).

        Returns
        -------
        System
            A new system object with deformed box and updated atomic coordinates.
        """
        uc = base_cryst.box.box.copy()
        s = size / 100.0
        L = np.diag(np.ones(3))
        if axis < 3:
            L[axis, axis] += s
        else:
            if axis == 3:
                L[1, 2] += s
            elif axis == 4:
                L[0, 2] += s
            else:
                L[0, 1] += s
        uc = np.dot(uc, L)

        new_pos = base_cryst.data.select("x", "y", "z").to_numpy() @ np.linalg.solve(
            base_cryst.box.box, uc
        )
        data = base_cryst.data.with_columns(
            x=new_pos[:, 0], y=new_pos[:, 1], z=new_pos[:, 2]
        )
        system = System(data=data, box=uc)
        system.calc = self.system.calc
        system.calc.results = {}
        return system

    def generate_deformations(self, n: int = 5, d: float = 2) -> List[System]:
        """
        Generate a series of deformed systems for elastic constant fitting.

        Parameters
        ----------
        n : int, optional
            Number of deformation steps for each mode (default: 5).
        d : float, optional
            Maximum deformation amplitude in percent (default: 2).

        Returns
        -------
        list of System
            List of deformed system objects.
        """
        self.relax_struc(self.system, change_cell=True)
        systems = []
        for a in range(6):
            if a < 3:
                for dx in np.linspace(-d, d, n):
                    systems.append(
                        self.get_cart_deformed_cell(self.system, axis=a, size=dx)
                    )
            else:
                for dx in np.linspace(d / 10.0, d, n):
                    systems.append(
                        self.get_cart_deformed_cell(self.system, axis=a, size=dx)
                    )
        return systems

    @staticmethod
    def get_strain(cryst: System, refcell: System) -> np.ndarray:
        """
        Compute the engineering strain tensor between deformed and reference cells.

        Parameters
        ----------
        cryst : System
            The deformed system.
        refcell : System
            The undeformed (reference) system.

        Returns
        -------
        np.ndarray
            Strain vector [uxx, uyy, uzz, uyz, uxz, uxy].
        """
        c1 = cryst.box.box
        c0 = refcell.box.box
        du = c1 - c0
        m_inv = np.linalg.inv(c0)
        u_mat = np.dot(m_inv, du)
        u_sym = 0.5 * (u_mat + u_mat.T)
        return np.array(
            [
                u_sym[0, 0],
                u_sym[1, 1],
                u_sym[2, 2],
                u_sym[2, 1],
                u_sym[2, 0],
                u_sym[1, 0],
            ],
            dtype=float,
        )

    # ------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------
    def compute(self, n: int = 5, d: float = 2):
        """
        Compute the elastic constants by fitting stress–strain data.

        Parameters
        ----------
        n : int, optional
            Number of deformation steps per mode (default: 5).
        d : float, optional
            Maximum deformation magnitude in percent (default: 2).

        Notes
        -----
        The method performs the following steps:

        1. Generate deformed configurations based on a fully relaxed structure.
        2. Relax each configuration to minimize energy.
        3. Compute the resulting stress tensors.
        4. Fit the linear stress–strain relation to extract Cij.
        """
        systems = self.generate_deformations(n, d)
        p = self.get_pressure(self.system.get_stress())

        ul, sl = [], []
        for g in systems:
            self.relax_struc(g)
            u_vec = self.get_strain(g, self.system)
            s_vec = np.array(g.get_stress(), dtype=float) - np.array(
                [p, p, p, 0.0, 0.0, 0.0], dtype=float
            )
            ul.append(u_vec)
            sl.append(s_vec)

        eqm = np.array([self.triclinic_deform(u) for u in ul])  # list of (6,18)
        eqm = eqm.reshape((-1, 18))
        slm = np.array(sl).ravel()
        Bij = np.linalg.lstsq(eqm, slm, rcond=None)
        correction = np.array(
            [-p, -p, -p, p, p, p, -p, -p, -p, p, p, p, p, p, p, p, p, p], dtype=float
        )
        Cij = Bij[0] - correction
        self.Cij = Cij

    def print_Cij(self, scale: float = 160.2176621):
        """
        Print the calculated elastic constants in GPa.

        Parameters
        ----------
        scale : float, optional
            Unit conversion factor (default: 160.2176621, eV/Å³ → GPa).

        Notes
        -----
        The output follows Voigt notation with components:

        C11, C22, C33, C12, C13, C23, C44, C55, C66,
        C16, C26, C36, C46, C56, C14, C15, C25, C45.
        """
        if self.Cij is None:
            print("Cij has not been computed. Please call compute() first.")
            return
        labels = (
            "C11",
            "C22",
            "C33",
            "C12",
            "C13",
            "C23",
            "C44",
            "C55",
            "C66",
            "C16",
            "C26",
            "C36",
            "C46",
            "C56",
            "C14",
            "C15",
            "C25",
            "C45",
        )
        print("Elastic constants (GPa):")
        for name, val in zip(labels, self.Cij * scale):
            print(f"{name:6s} = {val:10.3f}")


if __name__ == "__main__":
    pass
