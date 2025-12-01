# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

"""
Abstract Base Class for Static Calculators
=======================================================

This module defines the abstract base class for implementing static
calculators in the mdapy project. All calculator implementations should inherit
from this class and implement the required abstract methods.

Classes
-------
CalculatorMP : Abstract base class for MD calculators
"""

from abc import ABC, abstractmethod
from mdapy.box import Box
import polars as pl
import numpy as np


class CalculatorMP(ABC):
    """
    Abstract base class for static property calculators.

    This class defines the interface that all calculators must implement.
    Calculators are responsible for computing physical properties such as
    energies, forces, stresses, and virials for a given atomic configuration.

    Attributes
    ----------
    results : dict
        Dictionary storing computed results. Keys typically include:

        - 'energies': per-atom potential energies
        - 'forces': per-atom forces
        - 'stress': system stress tensor
        - 'virials': per-atom virial tensors
        - 'energy': system potential energy

    Notes
    -----
    All subclasses must implement the five abstract methods defined here.
    The `data` parameter should contain atomic positions and elements, while
    the `box` parameter describes the simulation cell geometry.

    Examples
    --------
    >>> class MyCalculator(CalculatorMP):
    ...     def get_energies(self, data, box):
    ...         # Implementation here
    ...         pass
    ...
    ...     # ... implement other methods
    """

    results = {}  # Class-level dictionary to store calculation results

    @abstractmethod
    def get_energies(self, data: pl.DataFrame, box: Box) -> np.ndarray:
        """
        Calculate per-atom potential energies.

        Parameters
        ----------
        data : pl.DataFrame
            Polars DataFrame containing atomic information. Must include:

            - 'x', 'y', 'z': atomic coordinates (float)
            - 'element': atomic species information
        box : Box
            Simulation box object containing cell dimensions and boundary conditions

        Returns
        -------
        np.ndarray
            1D array of shape (N,) containing potential energy for each atom,
            where N is the number of atoms

        """
        pass

    @abstractmethod
    def get_energy(self, data: pl.DataFrame, box: Box) -> float:
        """
        Calculate total potential energy of the system.

        Parameters
        ----------
        data : pl.DataFrame
            Polars DataFrame containing atomic information. Must include:

            - 'x', 'y', 'z': atomic coordinates (float)
            - 'element': atomic species information
        box : Box
            Simulation box object containing cell dimensions and boundary conditions

        Returns
        -------
        float
            Total potential energy of the system (sum of all per-atom energies)

        Notes
        -----
        This is typically implemented as `return self.get_energies(data, box).sum()`
        """
        pass

    @abstractmethod
    def get_forces(self, data: pl.DataFrame, box: Box) -> np.ndarray:
        """
        Calculate forces acting on each atom.

        Parameters
        ----------
        data : pl.DataFrame
            Polars DataFrame containing atomic information. Must include:

            - 'x', 'y', 'z': atomic coordinates (float)
            - 'element': atomic species information
        box : Box
            Simulation box object containing cell dimensions and boundary conditions

        Returns
        -------
        np.ndarray
            2D array of shape (N, 3) containing force components [fx, fy, fz]
            for each atom, where N is the number of atoms

        """
        pass

    @abstractmethod
    def get_stress(self, data: pl.DataFrame, box: Box) -> np.ndarray:
        """
        Calculate stress tensor of the system.

        Parameters
        ----------
        data : pl.DataFrame
            Polars DataFrame containing atomic information. Must include:

            - 'x', 'y', 'z': atomic coordinates (float)
            - 'element': atomic species information
        box : Box
            Simulation box object containing cell dimensions and boundary conditions

        Returns
        -------
        np.ndarray
            1D array of shape (6,) containing stress tensor components in Voigt notation:
            [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]

        Notes
        -----
        The stress tensor is symmetric, so only 6 independent components are returned.
        Stress is typically computed from virial: σ = -(virial + virial^T) / (2 * volume)
        """
        pass

    @abstractmethod
    def get_virials(self, data: pl.DataFrame, box: Box) -> np.ndarray:
        """
        Calculate per-atom virial tensors.

        Parameters
        ----------
        data : pl.DataFrame
            Polars DataFrame containing atomic information. Must include:

            - 'x', 'y', 'z': atomic coordinates (float)
            - 'element': atomic species information
        box : Box
            Simulation box object containing cell dimensions and boundary conditions

        Returns
        -------
        np.ndarray
            2D array of shape (N, 9) containing virial tensor components for each atom.
            Components ordered as: [v_xx, v_xy, v_xz, v_yx, v_yy, v_yz, v_zx, v_zy, v_zz]
            where N is the number of atoms

        """
        pass
