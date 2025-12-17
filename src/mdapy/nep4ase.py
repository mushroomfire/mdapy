# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

try:
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
except ImportError:
    raise ImportError("One can install ase by pip install ase.")
from mdapy import _nepcal, _qnepcal
from typing import Optional, List, Tuple
import numpy as np
import os


class NEP4ASE(Calculator):
    """
    NEP calculator compatible with ASE (Atomic Simulation Environment).

    This class wraps the NEP calculator to work seamlessly with ASE's
    calculator interface, allowing NEP models to be used in ASE workflows
    for geometry optimization, molecular dynamics, and other simulations.

    Parameters
    ----------
    model_filename : str
        Path to the NEP model file
    atoms : Atoms, optional
        ASE Atoms object to attach the calculator to

    Attributes
    ----------
    implemented_properties : list
        Properties that this calculator can compute:
        ['energy', 'energies', 'forces', 'stress', 'virials']
    calc : nepcal.NEPCalculator
        Underlying NEP calculator object
    results : dict
        Dictionary storing calculation results

    Examples
    --------
    >>> from ase import Atoms
    >>> from ase.optimize import BFGS
    >>> from mdapy.nep import NEP4ASE
    >>>
    >>> # Create atoms object
    >>> atoms = Atoms("Cu2", positions=[[0, 0, 0], [1.5, 0, 0]])
    >>> atoms.set_cell([10, 10, 10])
    >>> atoms.set_pbc(True)
    >>>
    >>> # Attach NEP calculator
    >>> calc = NEP4ASE("nep.txt")
    >>> atoms.calc = calc
    >>>
    >>> # Run geometry optimization
    >>> opt = BFGS(atoms)
    >>> opt.run(fmax=0.01)
    >>>
    >>> # Get energy and forces
    >>> energy = atoms.get_potential_energy()
    >>> forces = atoms.get_forces()
    """

    # Define which properties this calculator can compute
    implemented_properties = ["energy", "energies", "forces", "stress", "virials"]

    def __init__(
        self,
        model_filename: str,
        atoms: Optional[Atoms] = None,
    ):
        """
        Initialize NEP calculator for ASE.

        Parameters
        ----------
        model_filename : str
            Path to the NEP model file
        atoms : Atoms, optional
            ASE Atoms object to attach to this calculator

        Raises
        ------
        FileNotFoundError
            If the model file does not exist
        """
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"{model_filename} does not exist.")

        # Load NEP model
        self._is_qnep = False
        with open(model_filename) as op:
            if "charge" in op.readline():
                self._is_qnep = True

        if self._is_qnep:
            self.calc = _qnepcal.qNEPCalculator(model_filename)
        else:
            self.calc = _nepcal.NEPCalculator(model_filename)
        self.rc = max(self.calc.info["radial_cutoff"], self.calc.info["angular_cutoff"])
        self.results = {}

        # Initialize ASE Calculator base class
        Calculator.__init__(self, atoms=atoms)

    def set_nep(
        self, atoms: Atoms
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare ASE Atoms object for NEP calculation.

        Converts ASE Atoms format to the format required by NEP calculator.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object

        Returns
        -------
        type_list : np.ndarray
            Integer type indices for each atom
        x : np.ndarray
            X-coordinates
        y : np.ndarray
            Y-coordinates
        z : np.ndarray
            Z-coordinates
        box : np.ndarray
            Cell matrix (3×3)

        Raises
        ------
        AssertionError
            If atoms contain elements not in the NEP model
        """
        # Get chemical symbols for all atoms
        symbols = atoms.get_chemical_symbols()

        # Validate that all elements are supported by the model
        for i in np.unique(symbols):
            assert i in self.calc.info["element_list"], (
                f"NEP model did not include {i}."
            )

        # Map symbols to type indices
        type_list = np.array(
            [self.calc.info["element_list"].index(i) for i in symbols], np.int32
        )

        # Get positions and cell
        pos = np.array(atoms.get_positions())
        new_box = np.array(atoms.get_cell())

        for i, j in enumerate(atoms.get_pbc()):
            if j == 0:
                new_box[i, i] += 3 * self.rc

        # Return data in NEP calculator format
        return type_list, pos[:, 0], pos[:, 1], pos[:, 2], new_box

    def calculate(
        self,
        atoms: Atoms = None,
        properties: List[str] = None,
        system_changes: List[str] = all_changes,
    ):
        """
        Perform calculation for requested properties.

        This method is called by ASE when properties are requested.
        It calculates the specified properties and stores them in
        self.results.

        Parameters
        ----------
        atoms : Atoms, optional
            ASE Atoms object (uses self.atoms if None)
        properties : list of str, optional
            List of properties to calculate (default: all implemented)
        system_changes : list of str, optional
            List of changes since last calculation (for caching)

        Notes
        -----
        Special properties 'descriptor', 'latentspace', 'charges' and 'bec' are not in
        the standard ASE interface but are available through this calculator.
        """
        # Set default properties if not specified
        if properties is None:
            properties = self.implemented_properties

        # Call parent class calculate (handles caching and validation)
        Calculator.calculate(self, atoms, properties, system_changes)

        N = len(atoms)  # Number of atoms

        # Handle special properties: descriptor and latent space
        if "descriptor" in properties:
            descriptor = np.zeros((N, self.calc.info["num_ndim"]), float)
            self.calc.get_descriptors(*self.set_nep(atoms), descriptor)
            self.results["descriptor"] = descriptor
        elif "latentspace" in properties:
            if self._is_qnep:
                raise ValueError("qNEP dose not support get_latentspace now.")
            latentspace = np.zeros((N, self.calc.info["num_nlatent"]), float)
            self.calc.get_latentspace(*self.set_nep(atoms), latentspace)
            self.results["latentspace"] = latentspace
        else:
            # Standard calculation: energy, forces, stress, virials
            potential = np.zeros(N, float)
            force = np.zeros((N, 3), float)
            virial = np.zeros((N, 9), float)
            if self._is_qnep:
                charge = np.zeros(N, float)  # Per-atom charges
                bec = np.zeros((N, 9), float)  # Per-atom bec (9 components)

            # Perform NEP calculation
            if self._is_qnep:
                self.calc.calculate(
                    *self.set_nep(atoms), potential, force, virial, charge, bec
                )
            else:
                self.calc.calculate(*self.set_nep(atoms), potential, force, virial)

            # Store results in ASE format
            self.results["energy"] = potential.sum()  # Total energy
            self.results["energies"] = potential  # Per-atom energies
            self.results["forces"] = force  # Forces
            self.results["virials"] = virial  # Per-atom virials
            if self._is_qnep:
                self.results["charges"] = charge
                self.results["bec"] = bec

            # Calculate stress tensor from virials
            v = virial.sum(axis=0).reshape(3, 3)
            stress = (-0.5 * (v + v.T) / atoms.get_volume()).ravel()
            # Voigt notation: [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]
            stress = stress[[0, 4, 8, 5, 2, 1]]
            self.results["stress"] = stress

    def get_descriptor(
        self,
        atoms: Atoms = None,
        system_changes: List[str] = all_changes,
    ) -> np.ndarray:
        """
        Get atomic descriptors (not part of standard ASE interface).

        Parameters
        ----------
        atoms : Atoms, optional
            ASE Atoms object
        system_changes : list of str, optional
            System changes since last calculation

        Returns
        -------
        np.ndarray
            Descriptor array of shape (N, num_ndim)
        """
        self.calculate(atoms, ["descriptor"], system_changes)
        return self.results["descriptor"]

    def get_charges(
        self,
        atoms: Atoms = None,
        system_changes: List[str] = all_changes,
    ) -> np.ndarray:
        """
        Get atomic charges for qNEP model (not part of standard ASE interface).

        Parameters
        ----------
        atoms : Atoms, optional
            ASE Atoms object
        system_changes : list of str, optional
            System changes since last calculation

        Returns
        -------
        np.ndarray
            Charge array of shape (N,)
        """
        if self._is_qnep:
            if "charges" not in self.results.keys():
                self.calculate(atoms, ["charges"], system_changes)
        else:
            raise ValueError("Charges is only available for qNEP.")
        return self.results["charges"]

    def get_bec(
        self,
        atoms: Atoms = None,
        system_changes: List[str] = all_changes,
    ) -> np.ndarray:
        """
        Get atomic bec for qNEP model (not part of standard ASE interface).

        Parameters
        ----------
        atoms : Atoms, optional
            ASE Atoms object
        system_changes : list of str, optional
            System changes since last calculation

        Returns
        -------
        np.ndarray
            Bec array of shape (N, 9)
        """
        if self._is_qnep:
            if "bec" not in self.results.keys():
                self.calculate(atoms, ["bec"], system_changes)
        else:
            raise ValueError("Bec is only available for qNEP.")
        return self.results["bec"]

    def get_latentspace(
        self,
        atoms: Atoms = None,
        system_changes: List[str] = all_changes,
    ) -> np.ndarray:
        """
        Get latent space representations (not part of standard ASE interface).

        Parameters
        ----------
        atoms : Atoms, optional
            ASE Atoms object
        system_changes : list of str, optional
            System changes since last calculation

        Returns
        -------
        np.ndarray
            Latent space array of shape (N, num_nlatent)
        """
        if self._is_qnep:
            raise ValueError("qNEP dose not support get_latentspace now.")

        self.calculate(atoms, ["latentspace"], system_changes)
        return self.results["latentspace"]

    def get_virials(
        self,
        atoms: Atoms = None,
        system_changes: List[str] = all_changes,
    ) -> np.ndarray:
        """
        Get per-atom virials (not part of standard ASE interface).

        Parameters
        ----------
        atoms : Atoms, optional
            ASE Atoms object
        system_changes : list of str, optional
            System changes since last calculation

        Returns
        -------
        np.ndarray
            Virial array of shape (N, 9)
        """
        self.calculate(atoms, ["virials"], system_changes)
        return self.results["virials"]
