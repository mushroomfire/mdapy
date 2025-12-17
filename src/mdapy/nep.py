# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from mdapy import _nepcal, _qnepcal
from mdapy.box import Box
from typing import Tuple
import numpy as np
import polars as pl
import os
from mdapy.calculator import CalculatorMP


class NEP(CalculatorMP):
    """
    NEP calculator for mdapy static calculation framework.

    This class provides an interface to compute atomic properties using a
    pre-trained NEP (Neuroevolution Potential) machine learning model. It
    can calculate energies, forces, stresses, virials, descriptors, and
    latent space representations.

    Parameters
    ----------
    filename : str
        Path to the NEP model file (typically 'nep.txt' or similar)

    Attributes
    ----------
    calc : nepcal.NEPCalculator
        Underlying C++ NEP calculator object (NEP_CPU: https://github.com/brucefan1983/NEP_CPU)
    results : dict
        Dictionary storing computed results with keys:

        - 'energies': per-atom potential energies
        - 'forces': per-atom forces (N×3 array)
        - 'virials': per-atom virials (N×9 array)
        - 'stress': system stress tensor (6-component Voigt notation)

    Raises
    ------
    FileNotFoundError
        If the specified model file does not exist

    Examples
    --------
    >>> from mdapy import build_crystal
    >>> from mdapy.nep import NEP
    >>> # Load NEP model
    >>> calc = NEP("nep.txt")
    >>> system = build_crystal("Cu", "fcc", 3.615)
    >>> system.calc = calc
    >>> # Calculate properties
    >>> energy = system.get_energies()  # per-atom potential energy
    >>> forces = system.get_forces()  # per-atom force
    >>> virial = system.get_virials()  # per-atom virials
    >>> stress = system.get_stress()  # system stress
    >>> total_energy = system.get_energy()  # system potential energy
    >>> descriptor = calc.get_descriptor(system.data, system.box)  # descriptor
    >>> laten_space = calc.get_latentspace(system.data, system.box)  # latentspace
    """

    def __init__(self, filename: str):
        """
        Initialize NEP calculator with a trained model.

        Parameters
        ----------
        filename : str
            Path to the NEP model file

        Raises
        ------
        FileNotFoundError
            If the model file does not exist
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} does not exist.")

        self._is_qnep = False
        with open(filename) as op:
            if "charge" in op.readline():
                self._is_qnep = True

        if self._is_qnep:
            self.calc = _qnepcal.qNEPCalculator(filename)
        else:
            self.calc = _nepcal.NEPCalculator(filename)
        self.rc = max(self.calc.info["radial_cutoff"], self.calc.info["angular_cutoff"])
        # Initialize results dictionary
        self.results = {}

    def setAtoms(
        self, data: pl.DataFrame, box: Box
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare atomic configuration data for NEP calculator.

        This method converts the mdapy data format to the format required
        by the underlying NEP calculator, including type mapping and
        coordinate extraction.

        Parameters
        ----------
        data : pl.DataFrame
            DataFrame containing atomic information with columns:

            - 'x', 'y', 'z': atomic coordinates
            - 'element': chemical symbols (e.g., 'Cu', 'Au')
        box : Box
            Simulation box object

        Returns
        -------
        type_list : np.ndarray
            Integer array mapping each atom to its type index
        x : np.ndarray
            X-coordinates of all atoms
        y : np.ndarray
            Y-coordinates of all atoms
        z : np.ndarray
            Z-coordinates of all atoms
        box_array : np.ndarray
            Box matrix (3×3 array)

        Raises
        ------
        AssertionError
            If required columns are missing or if elements are not in the model

        Notes
        -----
        The NEP model must have been trained with all elements present in the data.
        """
        # Validate that required columns exist
        for i in ["x", "y", "z", "element"]:
            assert i in data.columns, f"data must contain {i} property."

        # Check that all elements in data are supported by the NEP model
        for i in data["element"].unique():
            assert i in self.calc.info["element_list"], (
                f"NEP model did not include {i}."
            )

        # Create mapping from element symbols to type indices
        element2type = {j: i for i, j in enumerate(self.calc.info["element_list"])}

        # Convert element symbols to integer type indices
        type_list = data.with_columns(
            pl.col("element").replace_strict(element2type).rechunk().alias("type")
        )["type"].to_numpy(allow_copy=False)

        # Extract coordinates as separate arrays (required by NEP calculator)
        new_box = box.box.copy()
        for i, j in enumerate(box.boundary):
            if j == 0:
                new_box[i, i] += 3 * self.rc

        return (
            type_list,
            data["x"].to_numpy(allow_copy=False),
            data["y"].to_numpy(allow_copy=False),
            data["z"].to_numpy(allow_copy=False),
            new_box,  # 3×3 box matrix
        )

    def calculate(self, data: pl.DataFrame, box: Box):
        """
        Calculate energies, forces, virials, and stress for the system.

        This method performs the main NEP calculation and stores results
        in the `self.results` dictionary. It's automatically called by
        other getter methods if results are not already cached.

        Parameters
        ----------
        data : pl.DataFrame
            DataFrame containing atomic positions and elements
        box : Box
            Simulation box object

        Notes
        -----
        Results are stored in `self.results` with keys:

        - 'energies': per-atom potential energies (N,)
        - 'forces': per-atom forces (N, 3)
        - 'virials': per-atom virials (N, 9)
        - 'stress': system stress tensor (6,) in Voigt notation
        - 'charge' : per-atom charge (N,), only available for qNEP
        - 'bec' : per-atom bec (N, 9), only available for qNEP

        The stress tensor is computed from virials as:
        σ = -(W + W^T) / (2V) where W is the total virial and V is volume.
        """
        N = data.shape[0]  # Number of atoms

        # Allocate arrays for outputs
        potential = np.zeros(N, float)  # Per-atom energies
        force = np.zeros((N, 3), float)  # Per-atom forces [fx, fy, fz]
        virial = np.zeros((N, 9), float)  # Per-atom virials (9 components)
        if self._is_qnep:
            charge = np.zeros(N, float)  # Per-atom charges
            bec = np.zeros((N, 9), float)  # Per-atom bec (9 components)

        # Call the C++NEP calculator
        if self._is_qnep:
            self.calc.calculate(
                *self.setAtoms(data, box), potential, force, virial, charge, bec
            )
        else:
            self.calc.calculate(*self.setAtoms(data, box), potential, force, virial)

        # Store results
        self.results["energies"] = potential
        self.results["forces"] = force
        self.results["virials"] = virial
        if self._is_qnep:
            self.results["charges"] = charge
            self.results["bec"] = bec

        # Calculate stress tensor from virials
        v = virial.sum(axis=0)  # Sum virials over all atoms
        # Reshape to 3×3 matrix: v_xx, v_xy, v_xz, v_yx, v_yy, v_yz, v_zx, v_zy, v_zz
        v = v.reshape(3, 3)
        # Stress = -(virial + virial^T) / (2 * volume)
        stress = (-0.5 * (v + v.T) / box.volume).ravel()
        # Convert to Voigt notation: [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]
        stress = stress[[0, 4, 8, 5, 2, 1]]
        self.results["stress"] = stress

    def get_energies(self, data: pl.DataFrame, box: Box) -> np.ndarray:
        """
        Get per-atom potential energies.

        Parameters
        ----------
        data : pl.DataFrame
            Atomic configuration data
        box : Box
            Simulation box

        Returns
        -------
        np.ndarray
            Array of shape (N,) containing energy for each atom
        """
        if "energies" not in self.results.keys():
            self.calculate(data, box)
        return self.results["energies"]

    def get_energy(self, data: pl.DataFrame, box: Box) -> float:
        """
        Get total potential energy of the system.

        Parameters
        ----------
        data : pl.DataFrame
            Atomic configuration data
        box : Box
            Simulation box

        Returns
        -------
        float
            Total potential energy (sum of per-atom energies)
        """
        return self.get_energies(data, box).sum()

    def get_virials(self, data: pl.DataFrame, box: Box) -> np.ndarray:
        """
        Get per-atom virial tensors.

        Parameters
        ----------
        data : pl.DataFrame
            Atomic configuration data
        box : Box
            Simulation box

        Returns
        -------
        np.ndarray
            Array of shape (N, 9) with virial components for each atom
            Ordered as: [v_xx, v_xy, v_xz, v_yx, v_yy, v_yz, v_zx, v_zy, v_zz]
        """
        if "virials" not in self.results.keys():
            self.calculate(data, box)
        return self.results["virials"]

    def get_forces(self, data: pl.DataFrame, box: Box) -> np.ndarray:
        """
        Get forces acting on each atom.

        Parameters
        ----------
        data : pl.DataFrame
            Atomic configuration data
        box : Box
            Simulation box

        Returns
        -------
        np.ndarray
            Array of shape (N, 3) containing force components [fx, fy, fz]
        """
        if "forces" not in self.results.keys():
            self.calculate(data, box)
        return self.results["forces"]

    def get_stress(self, data: pl.DataFrame, box: Box) -> np.ndarray:
        """
        Get stress tensor of the system.

        Parameters
        ----------
        data : pl.DataFrame
            Atomic configuration data
        box : Box
            Simulation box

        Returns
        -------
        np.ndarray
            Stress tensor in Voigt notation: [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]
        """
        if "stress" not in self.results.keys():
            self.calculate(data, box)
        return self.results["stress"]

    def get_charges(self, data: pl.DataFrame, box: Box) -> np.ndarray:
        """
        Get per-atom charges for qNEP.

        Parameters
        ----------
        data : pl.DataFrame
            Atomic configuration data
        box : Box
            Simulation box

        Returns
        -------
        np.ndarray
            Array of shape (N,) containing charge for each atom
        """
        if self._is_qnep:
            if "charges" not in self.results.keys():
                self.calculate(data, box)
        else:
            raise ValueError("Charges is only available for qNEP.")
        return self.results["charges"]

    def get_bec(self, data: pl.DataFrame, box: Box) -> np.ndarray:
        """
        Get per-atom bec for qNEP.

        Parameters
        ----------
        data : pl.DataFrame
            Atomic configuration data
        box : Box
            Simulation box

        Returns
        -------
        np.ndarray
            Array of shape (N, 9) with bec components for each atom
            Ordered as: [bec_xx, bec_xy, bec_xz, bec_yx, bec_yy, bec_yz, bec_zx, bec_zy, bec_zz]
        """
        if self._is_qnep:
            if "bec" not in self.results.keys():
                self.calculate(data, box)
        else:
            raise ValueError("bec is only available for qNEP.")
        return self.results["bec"]

    def get_descriptor(self, data: pl.DataFrame, box: Box) -> np.ndarray:
        """
        Get atomic descriptors from the NEP model.

        Descriptors are the learned feature representations in the hidden
        layers of the neural network, before the final energy prediction.

        Parameters
        ----------
        data : pl.DataFrame
            Atomic configuration data
        box : Box
            Simulation box

        Returns
        -------
        np.ndarray
            Array of shape (N, num_ndim) containing descriptor vectors
            for each atom, where num_ndim is the descriptor dimension

        Notes
        -----
        Descriptors can be useful for analyzing structural similarities.
        """
        N = data.shape[0]
        descriptor = np.zeros((N, self.calc.info["num_ndim"]), float)
        self.calc.get_descriptors(*self.setAtoms(data, box), descriptor)
        return descriptor

    def get_latentspace(self, data: pl.DataFrame, box: Box) -> np.ndarray:
        """
        Get latent space representations from the NEP model.

        The latent space is the representation in an intermediate layer
        of the neural network, capturing compressed structural information.

        Parameters
        ----------
        data : pl.DataFrame
            Atomic configuration data
        box : Box
            Simulation box

        Returns
        -------
        np.ndarray
            Array of shape (N, num_nlatent) containing latent vectors
            for each atom, where num_nlatent is the latent space dimension

        """
        if self._is_qnep:
            raise ValueError("qNEP dose not support get_latentspace now.")
        N = data.shape[0]
        latentspace = np.zeros((N, self.calc.info["num_nlatent"]), float)
        self.calc.get_latentspace(*self.setAtoms(data, box), latentspace)
        return latentspace


if __name__ == "__main__":
    pass
