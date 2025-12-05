# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

"""
Embedded Atom Method (EAM) Potential Calculator
================================================

This module implements the Embedded Atom Method (EAM) for static simulations.
The EAM potential is widely used for metallic systems and consists of two main components:

1. **Embedding energy**: :math:`F_i(\\rho_i)` - energy to embed atom i into the electron density :math:`\\rho_i`
2. **Pair interaction**: :math:`\\phi_{ij}(r_{ij})` - pair potential between atoms i and j

The total energy is given by:

.. math::

    E_{tot} = \\sum_i F_i(\\rho_i) + \\frac{1}{2} \\sum_{i,j \\neq i} \\phi_{ij}(r_{ij})

where the electron density at atom i is:

.. math::

    \\rho_i = \\sum_{j \\neq i} \\rho_j(r_{ij})

The forces on atom i are computed as:

.. math::

    \\mathbf{F}_i = -\\sum_{j \\neq i} \\left[ \\left(\\frac{dF_i}{d\\rho_i} + \\frac{dF_j}{d\\rho_j}\\right) \\frac{d\\rho_j}{dr_{ij}} + \\frac{d\\phi_{ij}}{dr_{ij}} \\right] \\frac{\\mathbf{r}_{ij}}{r_{ij}}

The stress tensor is computed from the virial theorem:

.. math::

    \\sigma_{\\alpha\\beta} = -\\frac{1}{V} \\sum_i \\sum_{j \\neq i} r_{ij,\\alpha} F_{ij,\\beta}

References
----------
.. [1] Daw, M. S., & Baskes, M. I. (1984). Embedded-atom method: Derivation and
       application to impurities, surfaces, and other defects in metals.
       Physical Review B, 29(12), 6443.

.. [2] Foiles, S. M., Baskes, M. I., & Daw, M. S. (1986). Embedded-atom-method
       functions for the fcc metals Cu, Ag, Au, Ni, Pd, Pt, and their alloys.
       Physical Review B, 33(12), 7983.

"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from mdapy.neighbor import Neighbor
import polars as pl
from mdapy.box import Box
from mdapy.calculator import CalculatorMP
from mdapy import _eam
import mdapy.tool_function as tool
from mdapy.data import atomic_masses, atomic_numbers
from typing import TYPE_CHECKING, Optional, Tuple, List, Dict
import numpy.typing as npt
import datetime

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


class EAM(CalculatorMP):
    """
    Embedded Atom Method (EAM) potential calculator.

    This class implements the EAM potential for metallic systems, supporting both
    single-element and multi-element alloy systems. It reads EAM potential files
    in the LAMMPS alloy format and provides methods to calculate energies, forces,
    stresses, and virials.

    Parameters
    ----------
    filename : str
        Path to the EAM potential file in LAMMPS alloy format.

    Attributes
    ----------
    filename : str
        Path to the EAM potential file.
    header : List[str]
        Header lines from the potential file (first 3 lines).
    Nelements : int
        Number of elements in the potential.
    elements_list : List[str]
        List of element symbols supported by this potential.
    nrho : int
        Number of points in the electron density grid.
    drho : float
        Spacing of the electron density grid.
    nr : int
        Number of points in the distance grid.
    dr : float
        Spacing of the distance grid.
    rc : float
        Cutoff radius for the potential (in Angstroms).
    F_rho : NDArray[np.float64]
        Embedding function F(ρ) for each element. Shape: (Nelements, nrho).
    rho_r : NDArray[np.float64]
        Electron density function ρ(r) for each element. Shape: (Nelements, nr).
    phi_r : NDArray[np.float64]
        Pair potential φ(r) between element pairs. Shape: (Nelements, Nelements, nr).
    r : NDArray[np.float64]
        Distance grid points. Shape: (nr,).
    rho : NDArray[np.float64]
        Electron density grid points. Shape: (nrho,).
    results : Dict[str, NDArray]
        Dictionary storing calculation results (energies, forces, stress, virials).

    """

    def __init__(self, filename: str, mass_list: Optional[List[float]] = None) -> None:
        self.filename: str = filename
        self.header: List[str] = []
        self.Nelements: int = 0
        self.elements_list: List[str] = []
        self.nrho: int = 0
        self.drho: float = 0.0
        self.nr: int = 0
        self.dr: float = 0.0
        self.rc: float = 0.0
        self.F_rho: NDArray[np.float64] = np.array([])
        self.rho_r: NDArray[np.float64] = np.array([])
        self.phi_r: NDArray[np.float64] = np.array([])
        self._rphi_r: NDArray[np.float64] = np.array([])
        self.r: NDArray[np.float64] = np.array([])
        self.rho: NDArray[np.float64] = np.array([])
        self.results: Dict[str, NDArray] = {}
        self._read_eam_alloy()
        if mass_list is None:
            mass_list = []
            for i in range(self.Nelements):
                species = self.elements_list[i]
                if species in atomic_numbers.keys():
                    aindex = atomic_numbers[species]
                    amass = atomic_masses[aindex]
                else:
                    amass = 0.0
                mass_list.append(amass)
        else:
            assert len(mass_list) == len(self.elements_list)
        self.mass_list = mass_list
        self._eam = _eam.EAM(
            self.rc, self.F_rho, self.rho_r, self.phi_r, self.r, self.rho
        )

    def _read_eam_alloy(self) -> None:
        """
        Read EAM potential file in LAMMPS alloy format.

        This method parses the EAM potential file and extracts:

        - Header information
        - Element symbols and count
        - Grid parameters (nrho, drho, nr, dr, rc)
        - Embedding functions F(ρ)
        - Electron density functions ρ(r)
        - Pair potentials φ(r)

        Raises
        ------
        FileNotFoundError
            If the potential file cannot be found.
        ValueError
            If the file format is invalid or data is inconsistent.

        Notes
        -----
        The pair potential is stored as r*φ(r) in the file and converted to φ(r)
        internally. Special handling is applied at r=0 to avoid division by zero.
        """
        with open(self.filename) as f:
            lines = f.readlines()

        self.header = lines[:3]

        line4 = lines[3].split()
        self.Nelements = int(line4[0])
        self.elements_list = line4[1 : 1 + self.Nelements]

        # nrho, drho, nr, dr, rc
        line5 = lines[4].split()
        self.nrho = int(line5[0])
        self.drho = float(line5[1])
        self.nr = int(line5[2])
        self.dr = float(line5[3])
        self.rc = float(line5[4])

        # Initialize storage arrays
        self.F_rho = np.zeros((self.Nelements, self.nrho))
        self.rho_r = np.zeros((self.Nelements, self.nr))
        self.phi_r = np.zeros((self.Nelements, self.Nelements, self.nr))
        self.r = np.arange(0, self.nr) * self.dr
        self.rho = np.arange(0, self.nrho) * self.drho

        # Process interleaved data and parameter lines
        line_idx = 5
        for element in range(self.Nelements):
            # skip this line (element information)
            line_idx += 1

            # Read embedding function and electron density data for this element
            # Total needed: nrho + nr values
            needed = self.nrho + self.nr
            element_data = []
            while len(element_data) < needed and line_idx < len(lines):
                line = lines[line_idx]
                line = line.split("#")[0]  # Remove comments
                values = line.split()
                # Collect all numerical values from this line
                for v in values:
                    try:
                        element_data.append(float(v))
                    except ValueError:
                        # Stop when encountering non-numeric string
                        break
                line_idx += 1
                if len(element_data) >= needed:
                    break

            # Store data for this element
            self.F_rho[element] = np.array(element_data[: self.nrho], dtype=float)
            self.rho_r[element] = np.array(
                element_data[self.nrho : self.nrho + self.nr], dtype=float
            )

        # Read phi_r (pair potentials stored as r*phi)
        rphi_data = []
        while line_idx < len(lines):
            line = lines[line_idx]
            line = line.split("#")[0]
            values = line.split()

            for v in values:
                try:
                    rphi_data.append(float(v))
                except ValueError:
                    pass
            line_idx += 1

        data_idx = 0

        self._rphi_r = np.zeros_like(self.phi_r)
        for element_i in range(self.Nelements):
            for element_j in range(element_i + 1):
                self._rphi_r[element_i, element_j] = np.array(
                    rphi_data[data_idx : data_idx + self.nr], dtype=float
                )
                data_idx += self.nr
                if element_i != element_j:
                    self._rphi_r[element_j, element_i] = self._rphi_r[
                        element_i, element_j
                    ]

        # Convert r*phi to phi (avoiding division by zero at r=0)
        self.phi_r[:, :, 1:] = self._rphi_r[:, :, 1:] / self.r[1:]
        self.phi_r[:, :, 0] = self.phi_r[:, :, 1]

    def write_eam_alloy(self, output_name: Optional[str] = None):
        """Write to an eam.alloy file.

        Args:
            output_name (Optional[str], optional): output filename, it will automatically generated if not given.
        """
        if output_name is None:
            output_name = ""
            for i in self.elements_list:
                output_name += i
            output_name += ".new.eam.alloy"

        with open(output_name, "w") as op:
            header = self.header.copy()
            h0 = f" eam/alloy {self.Nelements}"
            for i in self.elements_list:
                h0 += f" {i}"
            h0 += "\n"
            header[0] = h0
            header[1] = " Generated by mdapy!\n"
            header[2] = "\n"
            op.write("".join(header))

            op.write(f"    {self.Nelements} ")
            for i in self.elements_list:
                op.write(f"{i} ")
            op.write("\n")

            op.write(f" {self.nrho} {self.drho} {self.nr} {self.dr} {self.rc}\n")

            num = 1
            colnum = 5

            for i in range(self.Nelements):
                op.write(f" 0 {self.mass_list[i]} 0 dummy\n ")
                num = 1
                for j in range(self.nrho):
                    op.write(f"{self.F_rho[i, j]:.16E} ")
                    if num >= colnum:
                        op.write("\n ")
                        num = 0
                    num += 1

                if num > 1:
                    op.write("\n ")
                    num = 1

                for j in range(self.nr):
                    op.write(f"{self.rho_r[i, j]:.16E} ")
                    if num >= colnum:
                        op.write("\n ")
                        num = 0
                    num += 1

                if num > 1:
                    op.write("\n ")

            num = 1
            for i1 in range(self.Nelements):
                for i2 in range(i1 + 1):
                    for j in range(self.nr):
                        op.write(f"{self._rphi_r[i1, i2, j]:.16E} ")
                        if num >= colnum:
                            op.write("\n ")
                            num = 0
                        num += 1

            if num > 1:
                op.write("\n")

    def get_energies(self, data: pl.DataFrame, box: Box) -> NDArray[np.float64]:
        """
        Calculate per-atom potential energies.

        Parameters
        ----------
        data : pl.DataFrame
            Atomic configuration containing columns: 'x', 'y', 'z', 'element'.
        box : Box
            Simulation box object containing boundary conditions and dimensions.

        Returns
        -------
        NDArray[np.float64]
            Per-atom potential energies in eV. Shape: (N,) where N is the number of atoms.

        """
        if "energies" not in self.results.keys():
            self._calculate(data, box)
        return self.results["energies"]

    def get_energy(self, data: pl.DataFrame, box: Box) -> float:
        """
        Calculate total system potential energy.

        Parameters
        ----------
        data : pl.DataFrame
            Atomic configuration containing columns: 'x', 'y', 'z', 'element'.
        box : Box
            Simulation box object containing boundary conditions and dimensions.

        Returns
        -------
        float
            Total potential energy in eV.
        """
        return self.get_energies(data, box).sum()

    def get_forces(self, data: pl.DataFrame, box: Box) -> NDArray[np.float64]:
        """
        Calculate atomic forces.

        Parameters
        ----------
        data : pl.DataFrame
            Atomic configuration containing columns: 'x', 'y', 'z', 'element'.
        box : Box
            Simulation box object containing boundary conditions and dimensions.

        Returns
        -------
        NDArray[np.float64]
            Atomic forces in eV/Å. Shape: (N, 3) where N is the number of atoms.
            Columns represent [Fx, Fy, Fz].

        """
        if "forces" not in self.results.keys():
            self._calculate(data, box)
        return self.results["forces"]

    def get_stress(self, data: pl.DataFrame, box: Box) -> NDArray[np.float64]:
        """
        Calculate stress tensor in Voigt notation.

        Parameters
        ----------
        data : pl.DataFrame
            Atomic configuration containing columns: 'x', 'y', 'z', 'element'.
        box : Box
            Simulation box object containing boundary conditions and dimensions.

        Returns
        -------
        NDArray[np.float64]
            Stress tensor in Voigt notation [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy].
            Units: eV/Å³ or GPa depending on unit conversion.
            Shape: (6,).

        """
        if "stress" not in self.results.keys():
            self._calculate(data, box)
        return self.results["stress"]

    def get_virials(self, data: pl.DataFrame, box: Box) -> NDArray[np.float64]:
        """
        Calculate per-atom virial tensors.

        Parameters
        ----------
        data : pl.DataFrame
            Atomic configuration containing columns: 'x', 'y', 'z', 'element'.
        box : Box
            Simulation box object containing boundary conditions and dimensions.

        Returns
        -------
        NDArray[np.float64]
            Per-atom virial tensors. Shape: (N, 9) where N is the number of atoms.
            Components: [W_xx, W_xy, W_xz, W_yx, W_yy, W_yz, W_zx, W_zy, W_zz].
            Units: eV.
        """
        if "virials" not in self.results.keys():
            self._calculate(data, box)
        return self.results["virials"]

    def _calculate(self, data: pl.DataFrame, box: Box) -> None:
        """
        Main calculation method for energies, forces, stress, and virials.

        This method performs the complete EAM calculation:

        1. Validates input data
        2. Constructs neighbor lists
        3. Computes electron densities at each atom
        4. Evaluates embedding energies
        5. Calculates pair interactions
        6. Computes forces and virials
        7. Derives stress tensor

        Parameters
        ----------
        data : pl.DataFrame
            Atomic configuration with required columns: 'x', 'y', 'z', 'element'.
        box : Box
            Simulation box with boundary conditions.

        Raises
        ------
        AssertionError
            If required columns are missing from data.
        AssertionError
            If element types in data are not supported by the potential.

        Notes
        -----
        Results are stored in `self.results` dictionary with keys:

        - 'energies': Per-atom potential energies
        - 'forces': Atomic forces
        - 'stress': Global stress tensor (Voigt notation)
        - 'virials': Per-atom virial tensors

        """
        for i in ["x", "y", "z", "element"]:
            assert i in data.columns, f"Required column '{i}' not found in data."
        for i in data["element"].unique():
            assert i in self.elements_list, (
                f"{i} not in current EAM potential supported elements: {self.elements_list}."
            )
        old_N = data.shape[0]
        ele2type = {j: i for i, j in enumerate(self.elements_list)}
        data = data.with_columns(
            pl.col("element")
            .replace_strict(ele2type, return_dtype=pl.Int32)
            .rechunk()
            .alias("type")
        )

        repeat = box.check_small_box(self.rc)
        if sum(repeat) != 3:
            data, box = tool.replicate(data, box, *repeat)

        type_list = data["type"].to_numpy(allow_copy=False)
        neigh = Neighbor(self.rc, box, data, max_neigh=200)
        neigh.compute()

        N = data.shape[0]  # Number of atoms
        # Allocate arrays for outputs
        potential = np.zeros(N, float)  # Per-atom energies
        force = np.zeros((N, 3), float)  # Per-atom forces [fx, fy, fz]
        virial = np.zeros((N, 9), float)  # Per-atom virials (9 components)

        self._eam.calculate(
            data["x"].to_numpy(allow_copy=False),
            data["y"].to_numpy(allow_copy=False),
            data["z"].to_numpy(allow_copy=False),
            type_list,
            box.box,
            box.origin,
            box.boundary,
            neigh.verlet_list,
            neigh.distance_list,
            neigh.neighbor_number,
            force,
            virial,
            potential,
        )

        # Store results
        self.results["energies"] = potential[:old_N]
        self.results["forces"] = force[:old_N]
        self.results["virials"] = virial[:old_N]

        # Calculate stress tensor from virials
        v = virial.sum(axis=0)  # Sum virials over all atoms
        # Reshape to 3×3 matrix: v_xx, v_xy, v_xz, v_yx, v_yy, v_yz, v_zx, v_zy, v_zz
        v = v.reshape(3, 3)
        # Stress = -(virial + virial^T) / (2 * volume)
        stress = (-0.5 * (v + v.T) / box.volume).ravel()
        # Convert to Voigt notation: [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]
        stress = stress[[0, 4, 8, 5, 2, 1]]
        self.results["stress"] = stress

    def plot(
        self,
        fig: Optional[Figure] = None,
        ax: Optional[List[Axes]] = None,
    ) -> Tuple[Figure, List[Axes]]:
        """
        Plot EAM potential functions: embedding function, electron density, and pair potential.

        Creates three subplots showing:

        1. Embedding function F(ρ) vs electron density ρ
        2. Electron density function ρ(r) vs distance r
        3. Pair potential φ(r) vs distance r

        Parameters
        ----------
        fig : Figure, optional
            Matplotlib figure object. If None, creates a new figure.
        ax : List[Axes], optional
            List of three matplotlib axes objects. If None, creates new axes.

        Returns
        -------
        fig : Figure
            Matplotlib figure object containing the plots.
        ax : List[Axes]
            List of three matplotlib axes objects [ax_F, ax_rho, ax_phi].

        """
        from mdapy.plotset import set_figure

        if fig is None and ax is None:
            fig, ax = set_figure(ncol=3, figsize=(18, 6))
        else:
            assert len(ax) == 3, "ax must be a list of 3 axes objects"

        # Plot data for each element
        for i, element in enumerate(self.elements_list):
            ax[0].plot(self.rho, self.F_rho[i], label=element)
            ax[1].plot(self.r, self.rho_r[i], label=element)
            ax[2].plot(self.r, self.phi_r[i, i], label=f"{element}-{element}")

        # Set labels
        ax[0].set_xlabel("Electron Density ρ")
        ax[0].set_ylabel("Embedding Function F(ρ) (eV)")

        ax[1].set_xlabel("Distance r (Å)")
        ax[1].set_ylabel("Electron Density ρ(r)")

        ax[2].set_xlabel("Distance r (Å)")
        ax[2].set_ylabel("Pair Potential φ(r) (eV)")

        ax[0].set_xlim(0, self.rho[-1])
        ax[1].set_xlim(0, self.rc)
        ax[2].set_xlim(0, self.rc)

        for i in range(3):
            ax[i].legend()

        return fig, ax


class EAMAverage(EAM):
    """
    Generate average EAM (A-atom) potential for alloy investigation.

    This class creates an average EAM potential, which is useful in alloy investigation.
    The A-atom potential has a similar formula to the original EAM potential:

    .. math:: E_{i}=\\sum_{i} F^{A}\\left(\\bar{\\rho}_{i}\\right)+\\frac{1}{2} \\sum_{i, j \\neq i} \\phi_{ij}^{A A},

    .. math:: F^{A}\\left(\\bar{\\rho}_{i}\\right)=\\sum_{\\alpha} c_{\\alpha} F^{\\alpha}\\left(\\bar{\\rho}_{i}\\right),

    .. math:: \\phi_{ij}^{A A}=\\sum_{\\alpha, \\beta } c_{\\alpha} c_{\\beta} \\phi_{ij}^{\\alpha \\beta},

    .. math:: \\quad \\bar{\\rho}_{i}=\\sum_{j \\neq i} \\sum_{\\alpha} c_{\\alpha} \\rho_{ij}^{\\alpha},

    where :math:`A` denotes an average-atom.

    .. note:: If you use this module in publication, you should also cite the original paper.
      `Average-atom interatomic potential for random alloys <https://doi.org/10.1103/PhysRevB.93.104201>`_

    Parameters
    ----------
    filename : str
        Filename of eam.alloy file.
    concentration : List[float]
        Atomic ratio list, such as [0.5, 0.5]. The summation should equal 1.
    output_name : str, optional
        Filename of generated average EAM potential. If None, auto-generates name.

    Examples
    --------
    >>> potential = EAMAverage("CuNiAl.eam.alloy", [0.25, 0.25, 0.5])

    """

    def __init__(
        self,
        filename: str,
        concentration: List[float],
        output_name: Optional[str] = None,
    ) -> None:
        super().__init__(filename)
        self.concentration = concentration

        assert len(self.concentration) == self.Nelements, (
            f"Number of concentration list should be equal to {self.Nelements}."
        )
        assert np.isclose(np.sum(concentration), 1.0), (
            "Concentration summation should be equal to 1."
        )

        # Perform averaging
        self._average()

        # Determine output filename
        if output_name is None:
            self.output_name = ""
            for i in self.elements_list[:-1]:  # Exclude 'A' element
                self.output_name += i
            self.output_name += ".average.eam.alloy"
        else:
            self.output_name = output_name

        # Write the averaged potential
        self.write_eam_alloy(self.output_name)

    def _average(self) -> None:
        """
        Average the EAM potential arrays according to concentration.

        This method:
        1. Adds an 'A' element to represent the average atom
        2. Averages embedding and electron density functions
        3. Averages pair potentials with proper weighting
        """
        # Calculate average mass for 'A' element
        avg_mass = np.average(self.mass_list, weights=self.concentration)

        # Update element count and list
        self.Nelements += 1
        self.elements_list.append("A")
        self.mass_list.append(avg_mass)

        # Average embedded_data and elec_density_data
        new_F_rho = np.r_[
            self.F_rho,
            np.zeros((1, self.F_rho.shape[1]), dtype=self.F_rho.dtype),
        ]
        new_rho_r = np.r_[
            self.rho_r,
            np.zeros((1, self.rho_r.shape[1]), dtype=self.rho_r.dtype),
        ]
        new_F_rho[-1, :] = np.average(self.F_rho, axis=0, weights=self.concentration)
        new_rho_r[-1, :] = np.average(self.rho_r, axis=0, weights=self.concentration)

        # Average rphi_data
        new_rphi_r = np.concatenate(
            (
                self._rphi_r,
                np.zeros(
                    (self._rphi_r.shape[0], 1, self._rphi_r.shape[2]),
                    dtype=self._rphi_r.dtype,
                ),
            ),
            axis=1,
        )
        new_rphi_r = np.concatenate(
            (
                new_rphi_r,
                np.zeros(
                    (1, new_rphi_r.shape[1], new_rphi_r.shape[2]),
                    dtype=new_rphi_r.dtype,
                ),
            ),
            axis=0,
        )

        # Average for A-X interactions
        new_rphi_r[-1, :-1, :] = np.average(
            self._rphi_r, axis=0, weights=self.concentration
        )
        new_rphi_r[:-1, -1, :] = new_rphi_r[-1, :-1, :]

        # Average for A-A interaction
        column = new_rphi_r[:-1, -1, :]
        new_rphi_r[-1, -1, :] = np.average(column, axis=0, weights=self.concentration)

        # Convert rphi to phi
        new_phi_r = np.zeros_like(new_rphi_r)
        new_phi_r[:, :, 1:] = new_rphi_r[:, :, 1:] / self.r[1:]
        new_phi_r[:, :, 0] = new_phi_r[:, :, 1]

        # Update attributes
        self.F_rho = new_F_rho
        self.rho_r = new_rho_r
        self._rphi_r = new_rphi_r
        self.phi_r = new_phi_r


class EAMGenerator:
    """
    Generator for EAM (Embedded Atom Method) potential files.

    This class creates EAM potential files in the eam.alloy format for single
    or multi-element systems. The implementation is based on the Zhou-Johnson-Wadley
    parameterization.

    Parameters
    ----------
    elements_list : List[str]
        List of element symbols (e.g., ['Cu', 'Ni', 'Al']).
    output_filename : str, optional
        Name of the output file. If None, generates name from elements.
    nr : int, optional
        Number of points in distance grid (default: 2000).
    nrho : int, optional
        Number of points in electron density grid (default: 2000).
    rst : float, optional
        Minimum distance for pair potential computation (default: 0.5 Å).

    Raises
    ------
    ValueError
        If any element is not in the supported elements list.

    References
    ----------
    Zhou, X. W., Johnson, R. A., & Wadley, H. N. G. (2004).
    Misfit-energy-increasing dislocations in vapor-deposited CoFe/NiFe multilayers.
    Physical Review B, 69(14), 144113.
    https://doi.org/10.1103/PhysRevB.69.144113

    Examples
    --------
    >>> gen = EAMGenerator(["Cu", "Ni"])
    >>> # This automatically generates a 'CuNi.eam.alloy' file

    """

    # Class constants
    SUPPORTED_ELEMENTS: Tuple[str, ...] = (
        "Cu",
        "Ag",
        "Au",
        "Ni",
        "Pd",
        "Pt",
        "Al",
        "Pb",
        "Fe",
        "Mo",
        "Ta",
        "W",
        "Mg",
        "Co",
        "Ti",
        "Zr",
    )

    DEFAULT_NR: int = 2000
    DEFAULT_NRHO: int = 2000
    DEFAULT_RST: float = 0.5

    # Raw parameter data (Zhou et al. parameters)
    # 27 parameters per element in this order:
    # re, fe, rhoe, rhos, alpha, beta, A, B, kappa, lambda,
    # Fm0, Fm1, Fm2, Fm3, Fm4, Fi0, Fi1, Fi2, Fi3, Fe,
    # atomic_number, atomic_mass, eta, beta1, lambda1, rhol, rhoh
    _RAW_PARAMETERS: Tuple[str, ...] = (
        "Cu",
        "2.556162",
        "1.554485",
        "21.175871",
        "21.175395",
        "8.127620",
        "4.334731",
        "0.396620",
        "0.548085",
        "0.308782",
        "0.756515",
        "-2.170269",
        "-0.263788",
        "1.088878",
        "-0.817603",
        "-2.19",
        "0.00",
        "0.561830",
        "-2.100595",
        "0.310490",
        "-2.186568",
        "29",
        "63.546",
        "-2.100595",
        "4.334731",
        "0.756515",
        "0.85",
        "1.15",
        "Ag",
        "2.891814",
        "1.106232",
        "14.604100",
        "14.604144",
        "9.132010",
        "4.870405",
        "0.277758",
        "0.419611",
        "0.339710",
        "0.750758",
        "-1.729364",
        "-0.255882",
        "0.912050",
        "-0.561432",
        "-1.75",
        "0.00",
        "0.744561",
        "-1.150650",
        "0.783924",
        "-1.748423",
        "47",
        "107.8682",
        "-1.150650",
        "4.870405",
        "0.750758",
        "0.85",
        "1.15",
        "Au",
        "2.885034",
        "1.529021",
        "19.991632",
        "19.991509",
        "9.516052",
        "5.075228",
        "0.229762",
        "0.356666",
        "0.356570",
        "0.748798",
        "-2.937772",
        "-0.500288",
        "1.601954",
        "-0.835530",
        "-2.98",
        "0.00",
        "1.706587",
        "-1.134778",
        "1.021095",
        "-2.978815",
        "79",
        "196.96654",
        "-1.134778",
        "5.075228",
        "0.748798",
        "0.85",
        "1.15",
        "Ni",
        "2.488746",
        "2.007018",
        "27.562015",
        "27.562031",
        "8.383453",
        "4.471175",
        "0.429046",
        "0.633531",
        "0.443599",
        "0.820658",
        "-2.693513",
        "-0.076445",
        "0.241442",
        "-2.375626",
        "-2.70",
        "0.00",
        "0.265390",
        "-0.152856",
        "0.445470",
        "-2.7",
        "28",
        "58.6934",
        "-0.152856",
        "4.471175",
        "0.820658",
        "0.85",
        "1.15",
        "Pd",
        "2.750897",
        "1.595417",
        "21.335246",
        "21.940073",
        "8.697397",
        "4.638612",
        "0.406763",
        "0.598880",
        "0.397263",
        "0.754799",
        "-2.321006",
        "-0.473983",
        "1.615343",
        "-0.231681",
        "-2.36",
        "0.00",
        "1.481742",
        "-1.675615",
        "1.130000",
        "-2.352753",
        "46",
        "106.42",
        "-1.675615",
        "4.638612",
        "0.754799",
        "0.85",
        "1.15",
        "Pt",
        "2.771916",
        "2.336509",
        "33.367564",
        "35.205357",
        "7.105782",
        "3.789750",
        "0.556398",
        "0.696037",
        "0.385255",
        "0.770510",
        "-1.455568",
        "-2.149952",
        "0.528491",
        "1.222875",
        "-4.17",
        "0.00",
        "3.010561",
        "-2.420128",
        "1.450000",
        "-4.145597",
        "78",
        "195.08",
        "-2.420128",
        "3.789750",
        "0.770510",
        "0.25",
        "1.15",
        "Al",
        "2.863924",
        "1.403115",
        "20.418205",
        "23.195740",
        "6.613165",
        "3.527021",
        "0.314873",
        "0.365551",
        "0.379846",
        "0.759692",
        "-2.807602",
        "-0.301435",
        "1.258562",
        "-1.247604",
        "-2.83",
        "0.00",
        "0.622245",
        "-2.488244",
        "0.785902",
        "-2.824528",
        "13",
        "26.981539",
        "-2.488244",
        "3.527021",
        "0.759692",
        "0.85",
        "1.15",
        "Pb",
        "3.499723",
        "0.647872",
        "8.450154",
        "8.450063",
        "9.121799",
        "5.212457",
        "0.161219",
        "0.236884",
        "0.250805",
        "0.764955",
        "-1.422370",
        "-0.210107",
        "0.682886",
        "-0.529378",
        "-1.44",
        "0.00",
        "0.702726",
        "-0.538766",
        "0.935380",
        "-1.439436",
        "82",
        "207.2",
        "-0.538766",
        "5.212457",
        "0.764955",
        "0.85",
        "1.15",
        "Fe",
        "2.481987",
        "1.885957",
        "20.041463",
        "20.041463",
        "9.818270",
        "5.236411",
        "0.392811",
        "0.646243",
        "0.170306",
        "0.340613",
        "-2.534992",
        "-0.059605",
        "0.193065",
        "-2.282322",
        "-2.54",
        "0.00",
        "0.200269",
        "-0.148770",
        "0.391750",
        "-2.539945",
        "26",
        "55.847",
        "-0.148770",
        "5.236411",
        "0.340613",
        "0.85",
        "1.15",
        "Mo",
        "2.728100",
        "2.723710",
        "29.354065",
        "29.354065",
        "8.393531",
        "4.476550",
        "0.708787",
        "1.120373",
        "0.137640",
        "0.275280",
        "-3.692913",
        "-0.178812",
        "0.380450",
        "-3.133650",
        "-3.71",
        "0.00",
        "0.875874",
        "0.776222",
        "0.790879",
        "-3.712093",
        "42",
        "95.94",
        "0.776222",
        "4.476550",
        "0.275280",
        "0.85",
        "1.15",
        "Ta",
        "2.860082",
        "3.086341",
        "33.787168",
        "33.787168",
        "8.489528",
        "4.527748",
        "0.611679",
        "1.032101",
        "0.176977",
        "0.353954",
        "-5.103845",
        "-0.405524",
        "1.112997",
        "-3.585325",
        "-5.14",
        "0.00",
        "1.640098",
        "0.221375",
        "0.848843",
        "-5.141526",
        "73",
        "180.9479",
        "0.221375",
        "4.527748",
        "0.353954",
        "0.85",
        "1.15",
        "W",
        "2.740840",
        "3.487340",
        "37.234847",
        "37.234847",
        "8.900114",
        "4.746728",
        "0.882435",
        "1.394592",
        "0.139209",
        "0.278417",
        "-4.946281",
        "-0.148818",
        "0.365057",
        "-4.432406",
        "-4.96",
        "0.00",
        "0.661935",
        "0.348147",
        "0.582714",
        "-4.961306",
        "74",
        "183.84",
        "0.348147",
        "4.746728",
        "0.278417",
        "0.85",
        "1.15",
        "Mg",
        "3.196291",
        "0.544323",
        "7.132600",
        "7.132600",
        "10.228708",
        "5.455311",
        "0.137518",
        "0.225930",
        "0.5",
        "1.0",
        "-0.896473",
        "-0.044291",
        "0.162232",
        "-0.689950",
        "-0.90",
        "0.00",
        "0.122838",
        "-0.226010",
        "0.431425",
        "-0.899702",
        "12",
        "24.305",
        "-0.226010",
        "5.455311",
        "1.0",
        "0.85",
        "1.15",
        "Co",
        "2.505979",
        "1.975299",
        "27.206789",
        "27.206789",
        "8.679625",
        "4.629134",
        "0.421378",
        "0.640107",
        "0.5",
        "1.0",
        "-2.541799",
        "-0.219415",
        "0.733381",
        "-1.589003",
        "-2.56",
        "0.00",
        "0.705845",
        "-0.687140",
        "0.694608",
        "-2.559307",
        "27",
        "58.9332",
        "-0.687140",
        "4.629134",
        "1.0",
        "0.85",
        "1.15",
        "Ti",
        "2.933872",
        "1.863200",
        "25.565138",
        "25.565138",
        "8.775431",
        "4.680230",
        "0.373601",
        "0.570968",
        "0.5",
        "1.0",
        "-3.203773",
        "-0.198262",
        "0.683779",
        "-2.321732",
        "-3.22",
        "0.00",
        "0.608587",
        "-0.750710",
        "0.558572",
        "-3.219176",
        "22",
        "47.88",
        "-0.750710",
        "4.680230",
        "1.0",
        "0.85",
        "1.15",
        "Zr",
        "3.199978",
        "2.230909",
        "30.879991",
        "30.879991",
        "8.559190",
        "4.564902",
        "0.424667",
        "0.640054",
        "0.5",
        "1.0",
        "-4.485793",
        "-0.293129",
        "0.990148",
        "-3.202516",
        "-4.51",
        "0.00",
        "0.928602",
        "-0.981870",
        "0.597133",
        "-4.509025",
        "40",
        "91.224",
        "-0.981870",
        "4.564902",
        "1.0",
        "0.85",
        "1.15",
    )

    def __init__(
        self,
        elements_list: List[str],
        output_filename: Optional[str] = None,
        nr: int = DEFAULT_NR,
        nrho: int = DEFAULT_NRHO,
        rst: float = DEFAULT_RST,
    ) -> None:
        # Validate input elements
        for element in elements_list:
            if element not in self.SUPPORTED_ELEMENTS:
                raise ValueError(
                    f"Element '{element}' is not supported. "
                    f"Supported elements: {', '.join(self.SUPPORTED_ELEMENTS)}"
                )

        self.elements_list = elements_list
        self.n_elements = len(elements_list)
        self.nr = nr
        self.nrho = nrho
        self.rst = rst

        # Output filename
        if output_filename is None:
            self.output_filename = self._generate_filename()
        else:
            self.output_filename = output_filename

        # Initialize parameter storage
        self._get_eam_parameters()
        # Write to file
        self._write_eam_file()

    def _diedai(self):
        for i1 in range(self.ntypes):
            for i2 in range(i1 + 1):
                if i1 == i2:
                    for i in range(self.nr):
                        r = i * self.dr
                        if r < self.rst:
                            r = self.rst
                        fvalue = self._prof(i1, r)
                        if self.fmax < fvalue:
                            self.fmax = fvalue
                        self.rho[i, i1] = fvalue
                        psi = self._pair(i1, i2, r)
                        self.rphi[i, i1, i2] = r * psi
                else:
                    for i in range(self.nr):
                        r = i * self.dr
                        if r < self.rst:
                            r = self.rst
                        psi = self._pair(i1, i2, r)
                        self.rphi[i, i1, i2] = r * psi
                        self.rphi[i, i2, i1] = self.rphi[i, i1, i2]
        rhom = self.fmax
        if rhom < 2.0 * self.rhoemax:
            rhom = 2.0 * self.rhoemax
        if rhom < 100.0:
            rhom = 100.0
        self.drho = rhom / (self.nrho - 1.0)
        for it in range(self.ntypes):
            for i in range(self.nrho):
                rhoF = i * self.drho
                self.embedding_function[i, it] = self._embed(it, rhoF)

    def _prof(self, it, r):
        f = self.fe[it] * np.exp(-self.beta1[it] * (r / self.re[it] - 1.0))
        f = f / (1.0 + (r / self.re[it] - self.ramda1[it]) ** 20)
        return f

    def _pair(self, it1, it2, r):
        if it1 == it2:
            psi1 = self.A[it1] * np.exp(-self.alpha[it1] * (r / self.re[it1] - 1.0))
            psi1 = psi1 / (1.0 + (r / self.re[it1] - self.cai[it1]) ** 20)
            psi2 = self.B[it1] * np.exp(-self.beta[it1] * (r / self.re[it1] - 1.0))
            psi2 = psi2 / (1.0 + (r / self.re[it1] - self.ramda[it1]) ** 20)
            psi = psi1 - psi2
        else:
            psiab, fab = [], []
            for it in [it1, it2]:
                psi1 = self.A[it] * np.exp(-self.alpha[it] * (r / self.re[it] - 1.0))
                psi1 = psi1 / (1.0 + (r / self.re[it] - self.cai[it]) ** 20)
                psi2 = self.B[it] * np.exp(-self.beta[it] * (r / self.re[it] - 1.0))
                psi2 = psi2 / (1.0 + (r / self.re[it] - self.ramda[it]) ** 20)
                psiab.append(psi1 - psi2)
                fab.append(self._prof(it, r))
            psi = 0.5 * (fab[1] / fab[0] * psiab[0] + fab[0] / fab[1] * psiab[1])
        return psi

    def _embed(self, it, rho):
        if rho < self.rhoe[it]:
            Fm33 = self.Fm3[it]
        else:
            Fm33 = self.Fm4[it]
        if rho < self.rhoin[it]:
            emb = (
                self.Fi0[it]
                + self.Fi1[it] * (rho / self.rhoin[it] - 1.0)
                + self.Fi2[it] * (rho / self.rhoin[it] - 1.0) ** 2
                + self.Fi3[it] * (rho / self.rhoin[it] - 1.0) ** 3
            )
        elif rho < self.rhoout[it]:
            emb = (
                self.Fm0[it]
                + self.Fm1[it] * (rho / self.rhoe[it] - 1.0)
                + self.Fm2[it] * (rho / self.rhoe[it] - 1.0) ** 2
                + Fm33 * (rho / self.rhoe[it] - 1.0) ** 3
            )
        else:
            emb = (
                self.Fn[it]
                * (1.0 - self.fnn[it] * np.log(rho / self.rhos[it]))
                * (rho / self.rhos[it]) ** self.fnn[it]
            )
        return emb

    def _get_eam_parameters(self):
        name, data = [], []
        a = 0
        for _ in range(16):
            name.append(self._RAW_PARAMETERS[a].strip())
            data.append(self._RAW_PARAMETERS[a + 1 : a + 28])
            a += 28

        self.data = pl.from_numpy(np.array(data, dtype=np.float64).T, schema=name)

        (
            self.re,
            self.fe,
            self.rhoe,
            self.rhos,
            self.alpha,
            self.beta,
            self.A,
            self.B,
            self.cai,
            self.ramda,
            self.Fi0,
            self.Fi1,
            self.Fi2,
            self.Fi3,
            self.Fm0,
            self.Fm1,
            self.Fm2,
            self.Fm3,
            self.fnn,
            self.Fn,
            self.atomic_number,
            self.atomic_mass,
            self.Fm4,
            self.beta1,
            self.ramda1,
            self.rhol,
            self.rhoh,
        ) = self.data.select(self.elements_list).to_numpy()
        self.atomic_number = np.array(self.atomic_number, np.int32)
        self.lattice_constant = np.sqrt(2.0) * self.re
        self.rhoin = self.rhol * self.rhoe
        self.rhoout = self.rhoh * self.rhoe

        self.alatmax = self.lattice_constant.max()
        self.rhoemax = self.rhoe.max()
        self.ntypes = len(self.elements_list)
        self.rc = np.sqrt(10.0) / 2.0 * self.alatmax
        self.dr = self.rc / (self.nr - 1.0)
        self.fmax = -1.0
        self.rho = np.zeros((self.nrho, len(self.elements_list)))
        self.rphi = np.zeros(
            (self.nr, len(self.elements_list), len(self.elements_list))
        )
        self.embedding_function = np.zeros((self.nr, len(self.elements_list)))
        self._diedai()

    def _generate_filename(self) -> str:
        """Generate output filename from element list."""
        element_string = "".join(self.elements_list)
        return f"{element_string}.eam.alloy"

    def _write_eam_file(self) -> None:
        """Write the EAM potential to file in eam.alloy format."""
        crystal_structure = "fcc"

        with open(self.output_filename, "w") as file:
            # Header section
            self._write_header(file)

            # Grid parameters
            file.write(
                f" {self.nrho} {self.drho:.16E} "
                f"{self.nr} {self.dr:.16E} {self.rc:.16E}\n"
            )

            # Element-specific data
            for i in range(self.n_elements):
                self._write_element_data(file, i, crystal_structure)

            # Pair potentials
            self._write_pair_potentials(file)

    def _write_header(self, file) -> None:
        """Write file header with metadata."""
        # First line with element count and symbols
        file.write(f" eam/alloy {self.n_elements}")
        for element in self.elements_list:
            file.write(f" {element}")
        file.write("\n")

        # Citation and generation info
        file.write(
            f" Python version generated by EAMGenerator! Based on the Fortran version by Zhou et al. Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        file.write(
            " CITATION: X. W. Zhou, R. A. Johnson, H. N. G. Wadley, "
            "Phys. Rev. B, 69, 144113 (2004)\n"
        )

        # Element list (repeated)
        file.write(f"    {self.n_elements} ")
        for element in self.elements_list:
            file.write(f"{element} ")
        file.write("\n")

    def _write_element_data(
        self, file, element_idx: int, crystal_structure: str
    ) -> None:
        """Write embedding function and electron density for one element."""
        file.write(
            f" {self.atomic_number[element_idx]} "
            f"{self.atomic_mass[element_idx]:.10f} "
            f"{self.lattice_constant[element_idx]:.6f} "
            f"{crystal_structure}\n"
        )

        self._write_array_data(file, self.embedding_function[:, element_idx])
        self._write_array_data(file, self.rho[:, element_idx])

    def _write_pair_potentials(self, file) -> None:
        """Write pair potential data for all element pairs."""
        for i in range(self.n_elements):
            for j in range(i + 1):
                self._write_array_data(file, self.rphi[:, i, j])

    def _write_array_data(
        self, file, data: npt.NDArray[np.float64], values_per_line: int = 5
    ) -> None:
        """Write array data to file with specified formatting."""
        for idx, value in enumerate(data):
            if idx % values_per_line == 0:
                if idx > 0:
                    file.write("\n")
                file.write(" ")
            file.write(f"{value:.16E} ")

        # Ensure we end on a new line
        file.write("\n")


if __name__ == "__main__":
    # import os

    EAMGenerator(
        ["Co", "Ni", "Fe", "Al", "Cu"], "tests/input_files/CoNiFeAlCu.eam.alloy"
    )
    # EAMAverage("CuNiAl.eam.alloy", [0.25, 0.25, 0.5])

    # eam = EAM("CuNiAl.average.eam.alloy")
    # # eam = EAM(r'tests/input_files/ZrCu.lammps.eam.alloy')
    # eam.plot()
    # import matplotlib.pyplot as plt

    # plt.show()
    # os.system("rm CuNiAl*.eam.alloy")
