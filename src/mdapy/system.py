# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

"""
Core System Class for MDAPY
===========================

This module provides the central System class that serves as the main interface
for atomic simulation data and analysis in mdapy. The System class integrates
all analysis modules and provides a unified API for structural analysis,
property calculations, and data manipulation.

Classes
-------
System : Core class representing an atomic system
"""

from mdapy.box import Box
from mdapy.load_save import BuildSystem, SaveSystem
from mdapy.knn import NearestNeighbor
from mdapy.neighbor import Neighbor
from mdapy.voronoi import Voronoi
from mdapy.structure_entropy import StructureEntropy
from mdapy.cluster_analysis import ClusterAnalysis
from mdapy.calculator import CalculatorMP
from mdapy.identify_diamond_structure import IdentifyDiamondStructure
from mdapy.polyhedral_template_matching import PolyhedralTemplateMatching
from mdapy.common_neighbor_analysis import CommonNeighborAnalysis
from mdapy.steinhardt_bond_orientation import SteinhardtBondOrientation
from mdapy.radial_distribution_function import RadialDistributionFunction
from mdapy.structure_factor import StructureFactor
from mdapy.centro_symmetry_parameter import CentroSymmetryParameter
from mdapy.warren_cowley_parameter import WarrenCowleyParameter
from mdapy.identify_fcc_planar_faults import IdentifyFccPlanarFaults
from mdapy.ackland_jones_analysis import AcklandJonesAnalysis
from mdapy.common_neighbor_parameter import CommonNeighborParameter
from mdapy.atomic_temperature import AtomicTemperature
from mdapy.bond_analysis import BondAnalysis
from mdapy.angular_distribution_function import AngularDistributionFunction
import mdapy.tool_function as tool
from typing import Optional, Dict, TYPE_CHECKING, Union, Iterable, Any, List, Tuple
import numpy as np
import polars as pl


if TYPE_CHECKING:
    from ase import Atoms
    from ovito.data import DataCollection


class System:
    """
    Core class representing an atomic system with integrated analysis capabilities.

    The System class is the central data structure in mdapy, providing a unified
    interface for loading, manipulating, and analyzing atomic configurations. It
    integrates various structural analysis methods, property calculators, and
    visualization tools.

    Parameters
    ----------
    filename : str, optional
        Path to an atomic configuration file. Supports various formats including
        POSCAR, LAMMPS data/dump, XYZ, mp etc. Format is auto-detected or can
        be specified via the `format` parameter.
    data : pl.DataFrame, optional
        Polars DataFrame containing atomic information. Must include columns
        'x', 'y', 'z' for positions. Other common columns include 'type',
        'element', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz', etc.
    pos : np.ndarray, optional
        Array of shape (N, 3) containing atomic positions. Used in conjunction
        with `box` parameter.
    box : int, float, Iterable[float], np.ndarray, or Box, optional
        Simulation box specification. Required when using `data` or `pos` parameters.
        See :class:`~mdapy.box.Box` for supported formats.
    ase_atom : ase.Atoms, optional
        ASE Atoms object to initialize the system from.
    ovito_atom : ovito.data.DataCollection, optional
        OVITO DataCollection object to initialize the system from.
    format : str, optional
        File format specification when loading from `filename`.
        If None, format is auto-detected.
    global_info : dict, optional
        Dictionary containing global system properties such as total energy,
        virial, stress tensor, timestep, etc. Defaults to empty dict.

    Attributes
    ----------
    data : pl.DataFrame
        Polars DataFrame containing per-atom properties. Always includes
        'x', 'y', 'z' columns for positions.
    box : Box
        Simulation box object containing cell vectors and boundary conditions.
    global_info : dict
        Dictionary of global system properties.
    N : int
        Number of atoms in the system.
    calc : CalculatorMP or None
        Attached calculator for energy/force/virial computations.
    verlet_list : np.ndarray, optional
        Neighbor list array when neighbors are built.
    distance_list : np.ndarray, optional
        Distance list array when neighbors are built.
    neighbor_number : np.ndarray, optional
        Number of neighbors for each atom when neighbors are built.
    rc : float, optional
        Cutoff radius used for neighbor list construction.

    Notes
    -----
    The System class uses lazy evaluation for many analyses - neighbor lists
    and other expensive computations are only performed when needed. Results
    are cached and reused when possible.

    At initialization, exactly one of the following must be provided:
    - `filename`
    - `data` and `box`
    - `pos` and `box`
    - `ase_atom`
    - `ovito_atom`

    Examples
    --------
    Creating a system from different sources:

    .. code-block:: python

        from mdapy.system import System
        import numpy as np

        # Load from file (auto-detect format)
        system = System("config.dump")

        # Load from file with explicit format
        system = System("POSCAR", format="poscar")

        # Create from numpy array
        pos = np.random.rand(100, 3) * 10
        system = System(pos=pos, box=10)

        # Create from polars DataFrame
        import polars as pl

        data = pl.DataFrame(
            {
                "x": np.random.rand(100),
                "y": np.random.rand(100),
                "z": np.random.rand(100),
                "type": np.ones(100, dtype=int),
            }
        )
        system = System(data=data, box=[10, 10, 10])

    Performing structural analysis:

    .. code-block:: python

        # Identify crystal structures
        system.cal_polyhedral_template_matching()

        # Calculate radial distribution function
        rdf = system.cal_radial_distribution_function(rc=10.0, nbin=200)

        # Perform cluster analysis
        system.cal_cluster_analysis(rc=3.0)

    Accessing and modifying data:

    .. code-block:: python

        # Get positions
        pos = system.data.select("x", "y", "z")

        # Add a new column
        new_data = system.data.with_columns(energy=np.random.rand(system.N))
        system.update_data(new_data)

        # Save to file
        system.write("output.dump")

    See Also
    --------
    mdapy.box.Box : Simulation box representation
    mdapy.calculator.CalculatorMP : Abstract calculator interface
    mdapy.load_save.BuildSystem : System construction utilities
    mdapy.load_save.SaveSystem : System export utilities
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        data: Optional[pl.DataFrame] = None,
        pos: Optional[np.ndarray] = None,
        box: Union[int, float, Iterable[float], np.ndarray, Box] = None,
        ase_atom: Optional["Atoms"] = None,
        ovito_atom: Optional["DataCollection"] = None,
        format: Optional[str] = None,
        global_info: Optional[Dict[str, Any]] = {},
    ):
        """Initialize a System object from various data sources."""
        self.__global_info = {}
        if isinstance(filename, str):
            self.__data, self.box, self.__global_info = BuildSystem.from_file(
                filename, format
            )
        elif data is not None and box is not None:
            self.__data, self.box = BuildSystem.from_data(data, box)
        elif pos is not None and box is not None:
            self.__data, self.box = BuildSystem.from_array(pos, box)
        elif ase_atom is not None:
            self.__data, self.box = BuildSystem.from_ase(ase_atom)
        elif ovito_atom is not None:
            self.__data, self.box, self.__global_info = BuildSystem.from_ovito(
                ovito_atom
            )
        else:
            raise RuntimeError(
                "One must at least provide filename or [data, box] or [pos, box] or ase_atom or ovito_atom."
            )
        if not len(self.__global_info):
            self.__global_info = global_info
        self.calc: Optional[CalculatorMP] = None

    @property
    def global_info(self) -> Dict[str, Any]:
        """
        Obtain global system information.

        Returns
        -------
        dict
            Dictionary containing global properties such as total energy,
            virial tensor, stress tensor, timestep, temperature, etc.
            Keys and values depend on the data source and attached calculator.
        """
        return self.__global_info

    @property
    def data(self) -> pl.DataFrame:
        """
        Access the atomic data DataFrame.

        Returns
        -------
        pl.DataFrame
            Polars DataFrame containing per-atom properties. Always includes
            columns 'x', 'y', 'z' for Cartesian coordinates. May include
            additional columns such as 'type', 'element', 'vx', 'vy', 'vz'
            (velocities), 'fx', 'fy', 'fz' (forces), and various computed
            properties from analysis methods.

        Examples
        --------
        >>> system = System("config.dump")
        >>> print(system.data.columns)
        ['x', 'y', 'z', 'type', 'vx', 'vy', 'vz']
        """
        return self.__data

    @property
    def N(self) -> int:
        """
        Get the number of atoms in the system.

        Returns
        -------
        int
            Total number of atoms/particles in the system.

        Examples
        --------
        >>> system = System(pos=np.random.rand(100, 3), box=10)
        >>> print(system.N)
        100
        """
        return self.__data.shape[0]

    def __repr__(self) -> str:
        """
        Return string representation of the System.

        Returns
        -------
        str
            Multi-line string containing atom number, box information,
            global properties, and a summary of the data DataFrame.
        """
        if not self.__global_info:
            return f"Atom Number: {self.N}\n{self.box}\nParticle Information:\n{self.__data}"
        else:
            info = f"Atom Number: {self.N}\n{self.box}\n"
            for key in self.__global_info.keys():
                info += f"{key}: {self.__global_info[key]}\n"
            info += f"Particle Information:\n{self.__data}"
            return info

    def set_element(self, element: Union[str, List[str], Tuple[str], np.ndarray]):
        """
        Set element names for atoms in the system.

        Parameters
        ----------
        element : str, list of str, tuple of str, or np.ndarray
            Element specification:

            * **str**: Assigns the same element to all atoms
            * **list/tuple/array**: Must have length equal to N, specifying
              element for each atom individually

        Examples
        --------
        Assign same element to all atoms:

        >>> system.set_element("Cu")

        Assign different elements:

        >>> elements = ["Cu"] * 50 + ["Al"] * 50
        >>> system.set_element(elements)

        Notes
        -----
        This method updates the 'element' column in the data DataFrame.
        If an 'element' column exists, it is replaced.
        """
        if isinstance(element, str):
            df = self.__data.with_columns(pl.lit(element).alias("element"))
        else:
            assert len(element) == self.data.shape[0], (
                "Length of element should be equal to the atom number."
            )
            if "element" in self.data.columns:
                df = self.__data.select(pl.all().exclude("element")).with_columns(
                    pl.lit(np.array(element)).alias("element")
                )
            else:
                df = self.__data.with_columns(
                    pl.lit(np.array(element)).alias("element")
                )
        self.update_data(df, True)

    def set_type_by_element(
        self, element_list: Union[List[str], Tuple[str], np.ndarray]
    ):
        """
        Set atom types based on element ordering.

        Parameters
        ----------
        element_list : list of str, tuple of str, or np.ndarray
            Ordered list of unique elements. Atoms are assigned type numbers
            based on the index of their element in this list (starting from 1).

        Raises
        ------
        AssertionError
            If data does not contain 'element' column or if any element in
            data is not present in the provided element list.

        Examples
        --------
        >>> system.set_element(["Cu", "Cu", "Al", "Al"])
        >>> system.set_type_by_element(["Cu", "Al"])
        >>> print(system.data.select("element", "type"))
        ┌─────────┬──────┐
        │ element │ type │
        ├─────────┼──────┤
        │ Cu      │ 1    │
        │ Cu      │ 1    │
        │ Al      │ 2    │
        │ Al      │ 2    │
        └─────────┴──────┘

        Notes
        -----
        This method creates or updates the 'type' column in the data DataFrame.
        Type numbers start from 1 and correspond to the order in the element list.
        """
        assert "element" in self.data.columns, "Data must contain element column."
        for i in self.data["element"].unique():
            assert i in element_list, f"Element should have {i}."
        ele2type = {j: i for i, j in enumerate(element_list, start=1)}
        self.update_data(
            self.data.with_columns(
                pl.col("element")
                .replace_strict(ele2type, return_dtype=pl.Int32)
                .alias("type")
            ),
            True,
        )

    def get_positions(self, reduced: bool = False) -> pl.DataFrame:
        """
        Extract atomic positions from the system.

        Parameters
        ----------
        reduced : bool, optional
            If True, return fractional (reduced) coordinates in the box basis.
            If False (default), return Cartesian coordinates.

        Returns
        -------
        pl.DataFrame
            DataFrame containing position columns:

            * **reduced=False**: Columns 'x', 'y', 'z' (Cartesian coordinates)
            * **reduced=True**: Columns 'r_x', 'r_y', 'r_z' (fractional coordinates)

        Examples
        --------
        >>> system = System("POSCAR")
        >>> cart_pos = system.get_positions(reduced=False)
        >>> frac_pos = system.get_positions(reduced=True)

        Notes
        -----
        Fractional coordinates are computed using the inverse box matrix:
        reduced = positions @ box.inverse_box
        """
        if reduced:
            inv = self.box.inverse_box
            # pos @ inv
            return self.data.select(
                r_x=pl.col("x") * inv[0, 0]
                + pl.col("y") * inv[1, 0]
                + pl.col("z") * inv[2, 0],
                r_y=pl.col("x") * inv[0, 1]
                + pl.col("y") * inv[1, 1]
                + pl.col("z") * inv[2, 1],
                r_z=pl.col("x") * inv[0, 2]
                + pl.col("y") * inv[1, 2]
                + pl.col("z") * inv[2, 2],
            )
        else:
            return self.data.select("x", "y", "z")

    def get_velocities(self) -> pl.DataFrame:
        """
        Extract atomic velocities from the system.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns 'vx', 'vy', 'vz' containing velocity
            components for each atom.

        Raises
        ------
        AssertionError
            If the data does not contain velocity columns ('vx', 'vy', 'vz').

        Examples
        --------
        >>> system = System("relax.dump")  # File with velocities
        >>> velocities = system.get_velocities()
        """
        for i in ["vx", "vy", "vz"]:
            assert i in self.data.columns
        return self.data.select("vx", "vy", "vz")

    def set_pka(
        self,
        energy: float,
        direction: np.ndarray,
        index: Optional[int] = None,
        element: Optional[str] = None,
        factor: float = 1.0,
    ):
        """
        Set primary knock-on atom (PKA) for radiation damage simulation.

        This method assigns initial kinetic energy and direction to a PKA atom,
        commonly used to initialize cascade simulations in radiation damage studies.
        We assume that the velocity in the units of A/fs, you need to use `factor` to convert your velocity to this units.

        Parameters
        ----------
        energy : float
            Kinetic energy to assign to the PKA (in eV).
        direction : np.ndarray
            1D array of length 3 specifying the velocity direction vector.
            Will be normalized automatically.
        index : int, optional
            Index of the atom to set as PKA. If None, selects by nearest center atom.
        element : str, optional
            Element name to select PKA. If provided and index is None,
            selects by nearest center atom with same element.
        factor : float
            Convert your velocity to the units of A/fs.
            Defaults to 1.0.


        Notes
        -----
        This method updates the velocity columns ('vx', 'vy', 'vz') in the
        data DataFrame.

        Examples
        --------
        Set PKA by atom index:

        >>> system.set_pka(energy=1000.0, direction=np.array([1, 3, 5]), index=50)

        Set PKA by element:

        >>> system.set_pka(energy=2000.0, direction=np.array([1, 3, 5]), element="Fe")
        """
        self.update_data(
            tool._set_pka(
                self.data.with_columns(
                    pl.col("vx") * factor, pl.col("vy") * factor, pl.col("vz") * factor
                ),
                self.box,
                energy,
                direction,
                index,
                element,
            ).with_columns(
                pl.col("vx") / factor, pl.col("vy") / factor, pl.col("vz") / factor
            )
        )

    def get_energy(self) -> float:
        """
        Calculate total potential energy of the system.

        Returns
        -------
        float
            Total potential energy in the units eV.

        Raises
        ------
        AssertionError
            If no calculator is attached to the system.

        Examples
        --------
        >>> from mdapy.eam import EAM
        >>> system = System("config.dump")
        >>> system.calc = EAM("potential.eam")
        >>> total_energy = system.get_energy()
        """
        assert isinstance(self.calc, CalculatorMP), "Must assign a calc first."
        return self.calc.get_energy(self.data, self.box)

    def get_energies(self) -> np.ndarray:
        """
        Calculate per-atom potential energies in the units eV.

        Returns
        -------
        np.ndarray
            1D array of shape (N,) containing potential energy for each atom.

        Raises
        ------
        AssertionError
            If no calculator is attached to the system.
        """
        assert isinstance(self.calc, CalculatorMP), "Must assign a calc first."
        return self.calc.get_energies(self.data, self.box)

    def get_force(self) -> np.ndarray:
        """
        Calculate forces acting on each atom in the units eV/A.

        Returns
        -------
        np.ndarray
            2D array of shape (N, 3) containing force components [fx, fy, fz]
            for each atom.

        Raises
        ------
        AssertionError
            If no calculator is attached to the system.

        Examples
        --------
        >>> from mdapy.eam import EAM
        >>> system = System("config.dump")
        >>> system.calc = EAM("potential.eam")
        >>> forces = system.get_force()
        >>> print(forces.shape)
        (1000, 3)
        """
        assert isinstance(self.calc, CalculatorMP), "Must assign a calc first."
        return self.calc.get_forces(self.data, self.box)

    def get_stress(self) -> np.ndarray:
        """
        Calculate stress tensor of the system in the units eV/A^3.

        Returns
        -------
        np.ndarray
            1D array of shape (6,) containing stress tensor components in
            Voigt notation: [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy].

        Raises
        ------
        AssertionError
            If no calculator is attached to the system.

        Examples
        --------
        >>> from mdapy.eam import EAM
        >>> system = System("config.dump")
        >>> system.calc = EAM("potential.eam")
        >>> stress = system.get_stress()
        >>> print(stress)  # [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]
        """
        assert isinstance(self.calc, CalculatorMP), "Must assign a calc first."
        return self.calc.get_stress(self.data, self.box)

    def get_virials(self) -> np.ndarray:
        """
        Calculate per-atom virial tensors in the units eV.

        Returns
        -------
        np.ndarray
            2D array of shape (N, 9) containing virial tensor components for
            each atom, ordered as: [v_xx, v_xy, v_xz, v_yx, v_yy, v_yz, v_zx, v_zy, v_zz].

        Raises
        ------
        AssertionError
            If no calculator is attached to the system.

        Examples
        --------
        >>> from mdapy.eam import EAM
        >>> system = System("config.dump")
        >>> system.calc = EAM("potential.eam")
        >>> virials = system.get_virials()
        >>> print(virials.shape)
        (1000, 9)
        """
        assert isinstance(self.calc, CalculatorMP), "Must assign a calc first."
        return self.calc.get_virials(self.data, self.box)

    def update_data(
        self,
        data: pl.DataFrame,
        reset_calcolator: bool = False,
        reset_neighbor: bool = False,
    ) -> None:
        """
        Update the atomic data DataFrame.

        Parameters
        ----------
        data : pl.DataFrame
            New DataFrame containing updated atomic information.
            Should maintain the same structure (columns) as the original data.
        reset_calcolator : bool, optional
            If True, clears cached results from the attached calculator.
            Set to True when atomic positions or types/element change. Default is False.
        reset_neighbor : bool, optional
            If True, clear all neighbor information.
            Set to True when atomic positions change. Default is False.

        Examples
        --------
        >>> # Add a new property column
        >>> new_data = system.data.with_columns(
        ...     temperature=np.random.rand(system.N) * 300
        ... )
        >>> system.update_data(new_data)

        >>> # Update positions (reset calculator)
        >>> new_data = system.data.with_columns(x=system.data["x"] + 0.1)
        >>> system.update_data(new_data, reset_calcolator=True)

        Notes
        -----
        When `reset_calcolator=True`, any cached energy, force, or stress
        calculations are invalidated and will be recomputed on next access.
        """
        self.__data = data
        if reset_calcolator and isinstance(self.calc, CalculatorMP):
            self.calc.results = {}
        if reset_neighbor:
            if hasattr(self, "verlet_list"):
                del self.verlet_list, self.neighbor_number, self.distance_list
            if hasattr(self, "rc"):
                del self.rc
            if hasattr(self, "voro_verlet_list"):
                del (
                    self.voro_verlet_list,
                    self.voro_distance_list,
                    self.voro_face_area,
                    self.voro_neighbor_number,
                )
            if hasattr(self, "_enlarge_box"):
                del self._enlarge_box, self._enlarge_data

    def update_box(
        self,
        box: Union[int, float, Iterable[float], np.ndarray, Box],
        scale_pos: bool = False,
    ) -> None:
        """
        Update the simulation box.

        Parameters
        ----------
        box : int, float, Iterable[float], np.ndarray, or Box
            New box specification. See :class:`~mdapy.box.Box` for accepted formats.
        scale_pos : bool, optional
            If True, scales atomic positions affinely with the box change.
            Useful for applying strain or changing lattice constants while
            maintaining fractional coordinates. Default is False.

        Raises
        ------
        AssertionError
            If `scale_pos=True` but not all boundaries are periodic.

        Examples
        --------
        Change box without scaling positions:

        >>> system.update_box([20, 20, 20])

        Apply uniform strain by scaling box and positions:

        >>> system.update_box([11, 11, 11], scale_pos=True)

        Notes
        -----
        When `scale_pos=True`, the transformation is:
        new_pos = old_pos @ (old_box^-1 @ new_box)

        This preserves fractional coordinates in the new box.
        """
        new_box = Box(box)
        if scale_pos:
            assert sum(new_box.boundary) == 3, (
                "only support all periodic boundary condition."
            )
            map_matrix = np.linalg.solve(self.box.box, new_box.box)
            self.__data = self.__data.with_columns(
                pl.col("x") - self.box.origin[0],
                pl.col("y") - self.box.origin[1],
                pl.col("z") - self.box.origin[2],
            ).with_columns(
                x=pl.col("x") * map_matrix[0, 0]
                + pl.col("y") * map_matrix[1, 0]
                + pl.col("z") * map_matrix[2, 0]
                + new_box.origin[0],
                y=pl.col("x") * map_matrix[0, 1]
                + pl.col("y") * map_matrix[1, 1]
                + pl.col("z") * map_matrix[2, 1]
                + new_box.origin[1],
                z=pl.col("x") * map_matrix[0, 2]
                + pl.col("y") * map_matrix[1, 2]
                + pl.col("z") * map_matrix[2, 2]
                + new_box.origin[2],
            )

        self.box = new_box

        if isinstance(self.calc, CalculatorMP):
            self.calc.results = {}
        if hasattr(self, "verlet_list"):
            del self.verlet_list, self.neighbor_number, self.distance_list
        if hasattr(self, "rc"):
            del self.rc
        if hasattr(self, "voro_verlet_list"):
            del (
                self.voro_verlet_list,
                self.voro_distance_list,
                self.voro_face_area,
                self.voro_neighbor_number,
            )
        if hasattr(self, "_enlarge_box"):
            del self._enlarge_box, self._enlarge_data

    def wrap_pos(self) -> None:
        """Wrap positions into box for PBC boundary."""
        self.update_data(tool.wrap_pos(self.data, self.box), True, True)

    def replicate(self, nx: int, ny: int, nz: int) -> None:
        """
        Replicate the system by creating a supercell.

        Parameters
        ----------
        nx : int
            Number of replications along x-direction.
        ny : int
            Number of replications along y-direction.
        nz : int
            Number of replications along z-direction.

        Examples
        --------
        Create a 2x2x2 supercell:

        >>> system = System("unit_cell.poscar")
        >>> system.replicate(2, 2, 2)
        >>> print(system.N)  # 8 times the original number

        Notes
        -----
        This method modifies the system in place. The box is scaled by
        [nx, ny, nz] and atoms are replicated accordingly.
        """
        assert nx > 0
        assert ny > 0
        assert nz > 0
        data, box = tool.replicate(self.__data, self.box, int(nx), int(ny), int(nz))
        self.update_data(data, True, True)
        self.update_box(box)

    def to_ovito(self) -> "DataCollection":
        """
        Convert system to OVITO DataCollection object.

        Returns
        -------
        ovito.data.DataCollection
            OVITO data structure containing the atomic configuration.

        Raises
        ------
        ImportError
            If OVITO is not installed.

        Examples
        --------
        >>> data = system.to_ovito()
        >>> # Use OVITO's analysis or visualization tools
        >>> from ovito.modifiers import CommonNeighborAnalysisModifier
        >>> data.apply(CommonNeighborAnalysisModifier())
        """
        return SaveSystem.to_ovito(self.__data, self.box, self.__global_info)

    def to_ase(self) -> "Atoms":
        """
        Convert system to ASE Atoms object.

        Returns
        -------
        ase.Atoms
            ASE Atoms object containing the atomic configuration.

        Raises
        ------
        ImportError
            If ASE is not installed.
        AssertionError
            If data does not contain an 'element' column.

        Examples
        --------
        >>> atoms = system.to_ase()
        >>> # Use ASE's functionality
        >>> from ase.io import write
        >>> write("output.xyz", atoms)
        """
        return SaveSystem.to_ase(self.__data, self.box)

    def write_mp(self, output_name: str) -> None:
        """
        Save system to mp format file.

        Parameters
        ----------
        nx : int
            Number of replications along x-direction.
        ny : int
            Number of replications along y-direction.
        nz : int
            Number of replications along z-direction.

        Raises
        ------
        ImportError
            If ASE is not installed.
        AssertionError
            If data does not contain an 'element' column.

        Examples
        --------
        >>> atoms = system.to_ase()
        >>> # Use ASE's functionality
        >>> from ase.io import write
        >>> write("output.xyz", atoms)
        """
        SaveSystem.write_mp(output_name, self.data, self.box, self.global_info)

    def write_xyz(self, output_name: str, classical: bool = False, compress=False):
        SaveSystem.write_xyz(
            output_name, self.box, self.data, classical, compress, **self.global_info
        )

    def write_poscar(
        self,
        output_name: str,
        reduced_pos: bool = False,
        selective_dynamics: bool = False,
        compress=False,
    ):
        SaveSystem.write_poscar(
            output_name,
            self.box,
            self.data,
            reduced_pos=reduced_pos,
            selective_dynamics=selective_dynamics,
            compress=compress,
        )

    def write_data(
        self,
        output_name: str,
        element_list: Optional[List[str]] = None,
        num_type: Optional[int] = None,
        data_format: str = "atomic",
        compress=False,
    ):
        if element_list is not None:
            self.set_type_by_element(element_list)
        SaveSystem.write_data(
            output_name,
            self.box,
            self.data,
            element_list=element_list,
            num_type=num_type,
            data_format=data_format,
            compress=compress,
        )

    def write_dump(self, output_name: str, timestep: float = 0, compress=False):
        SaveSystem.write_dump(
            output_name, self.box, self.data, timestep=timestep, compress=compress
        )

    def build_neighbor(
        self,
        rc: float,
        max_neigh: Optional[int] = None,
    ) -> None:
        """
        Build neighbor list for the system.

        This method constructs a Verlet neighbor list within a cutoff radius,
        which is required for many analysis methods. For small systems, the
        box may be automatically replicated to ensure sufficient neighbors.

        Parameters
        ----------
        rc : float
            Cutoff radius for neighbor search (in simulation units).
        max_neigh : int, optional
            Maximum number of neighbors per atom. If None, automatically
            determined based on the actual neighbor number Increase this if you get warnings
            about insufficient neighbor slots.

        Attributes Set
        --------------
        verlet_list : np.ndarray
            2D array of shape (N, max_neigh) containing neighbor indices.
            Unfilled slots contain -1.
        distance_list : np.ndarray
            2D array of shape (N, max_neigh) containing distances to neighbors.
        neighbor_number : np.ndarray
            1D array of shape (N,) containing number of neighbors for each atom.
        rc : float
            The cutoff radius used.
        _enlarge_box : Box, optional
            replicated box for small system.
        _enlarge_data : pl.DataFrame, optional
            replicated atom dataframe for small system

        Examples
        --------
        >>> system.build_neighbor(rc=5.0)
        >>> print(system.verlet_list.shape)
        (1000, 50)  # 1000 atoms, up to 50 neighbors each
        >>> print(system.neighbor_number)
        [12 12 12 ... 12 12 12]  # FCC coordination

        See :class:`~mdapy.neighbor.Neighbor` for implementation details.
        """
        neigh = Neighbor(rc, self.box, self.data, max_neigh)
        neigh.compute()
        self.rc = rc
        if hasattr(neigh, "_enlarge_box"):
            self._enlarge_box: Box = neigh._enlarge_box
        if hasattr(neigh, "_enlarge_data"):
            self._enlarge_data: pl.DataFrame = neigh._enlarge_data
        self.verlet_list, self.distance_list, self.neighbor_number = (
            neigh.verlet_list,
            neigh.distance_list,
            neigh.neighbor_number,
        )

    def build_voronoi_neighbor(
        self, a_face_area_threshold: float = -1.0, r_face_area_threshold: float = -1.0
    ) -> None:
        """
        Calculate Voronoi neighbors and face properties for all atoms.

        This method performs Voronoi tessellation to identify nearest neighbors
        based on shared Voronoi cell faces. It can filter neighbors based on
        face area thresholds to exclude insignificant contacts.

        Parameters
        ----------
        a_face_area_threshold : float, optional
            Absolute face area threshold.
            Faces with area below this value are ignored as neighbors.
            Default is -1.0 (no filtering).
        r_face_area_threshold : float, optional
            Relative face area threshold (fraction of average face area).
            Faces with relative area below this value are ignored.
            Default is -1.0 (no filtering).

        Attributes Set
        --------------
        voro_verlet_list : np.ndarray
            2D array of shape (N, max_neigh) containing neighbor indices.
            Unfilled slots contain -1.
        voro_distance_list : np.ndarray
            2D array of shape (N, max_neigh) containing distances to neighbors.
        voro_neighbor_number : np.ndarray
            1D array of shape (N,) containing number of neighbors for each atom.
        voro_face_area : np.ndarray
            2D array of shape (N, max_neighbors) containing the area of
            the Voronoi face shared with each neighbor.
        _enlarge_box : Box, optional
            replicated box for small system.
        _enlarge_data : pl.DataFrame, optional
            replicated atom dataframe for small system

        Examples
        --------
        >>> system.build_voronoi_neighbor()
        >>> print(system.voro_verlet_list.shape)
        (1000, 50)  # 1000 atoms, up to 50 neighbors each

        See :class:`~mdapy.voronoi.Voronoi` for implementation details.
        """
        vor = Voronoi(self.box, self.data)
        (
            self.voro_verlet_list,
            self.voro_distance_list,
            self.voro_face_area,
            self.voro_neighbor_number,
        ) = vor.get_neighbor(a_face_area_threshold, r_face_area_threshold)
        if hasattr(vor, "_enlarge_box"):
            self._enlarge_box: Box = vor._enlarge_box
        if hasattr(vor, "_enlarge_data"):
            self._enlarge_data: pl.DataFrame = vor._enlarge_data

    def build_nearest_neighbor(self, k: int) -> None:
        """
        Calculate `k` nearest neighbor information.

        Parameters
        ----------
        k : int
            Maximum number of neighbors to find per atom.

        Attributes Set
        --------------
        verlet_list : np.ndarray
            2D array of shape (N, k) containing neighbor indices.
        distance_list : np.ndarray
            2D array of shape (N, k) containing distances to neighbors.
        neighbor_number : np.ndarray
            1D array of shape (N,) containing number of neighbors for each atom.
        _enlarge_box : Box, optional
            replicated box for small system.
        _enlarge_data : pl.DataFrame, optional
            replicated atom dataframe for small system

        Examples
        --------
        >>> system.build_nearest_neighbor(12)
        >>> print(system.verlet_list.shape)
        (1000, 12)  # 1000 atoms, up to 12 sorted neighbors each

        See :class:`~mdapy.knn.NearestNeighbor` for implementation details.
        """
        kdt = NearestNeighbor(self.data, self.box, k)
        kdt.compute()
        if hasattr(kdt, "_enlarge_box"):
            self._enlarge_box: Box = kdt._enlarge_box
        if hasattr(kdt, "_enlarge_data"):
            self._enlarge_data: pl.DataFrame = kdt._enlarge_data
        self.verlet_list, self.distance_list = kdt.indices_py, kdt.distances_py
        self.neighbor_number = np.full(self.verlet_list.shape[0], k, np.int32)

    def cal_identify_diamond_structure(self) -> None:
        """
        Identify diamond structure atoms. This will generate 'ids' column in self.data:

        - 0 Other
        - 1 Cubic diamond
        - 2 Cubic diamond (1st neighbor)
        - 3 Cubic diamond (2nd neighbor)
        - 4 Hexagonal diamond
        - 5 Hexagonal diamond (1st neighbor)
        - 6 Hexagonal diamond (2nd neighbor)

        Notes
        -----
        See :class:`~mdapy.identify_diamond_structure.IdentifyDiamondStructure`
        for implementation details.
        """
        verlet_list = None
        if self.N > 500:  # safe atom number
            if hasattr(self, "neighbor_number"):
                if self.neighbor_number.min() >= 4:
                    tool.sort_neighbor(
                        self.verlet_list, self.distance_list, self.neighbor_number, 4
                    )
                    verlet_list = self.verlet_list

        if hasattr(self, "_enlarge_data"):
            box = self._enlarge_box
            data = self._enlarge_data
        else:
            box = self.box
            data = self.__data

        ids = IdentifyDiamondStructure(data, box, verlet_list)
        ids.compute()
        self.update_data(self.__data.with_columns(ids=ids.pattern[: self.N]))

    def cal_common_neighbor_parameter(
        self, rc: float, max_neigh: Optional[int] = None
    ) -> None:
        """
        Calculate common neighbor parameter for structure analysis. This will save results to self.data['cnp'].

        Parameters
        ----------
        rc : float
            Cutoff radius for neighbor search.
        max_neigh : int, optional
            Maximum number of neighbors per atom.

        Notes
        -----
        See :class:`~mdapy.common_neighbor_parameter.CommonNeighborParameter`
        for implementation details.

        """
        has_neigh = False
        if hasattr(self, "rc"):
            if self.rc >= rc:
                has_neigh = True
        if not has_neigh:
            self.build_neighbor(rc, max_neigh)
        if hasattr(self, "_enlarge_data"):
            data = self._enlarge_data
            box = self._enlarge_box
        else:
            data = self.__data
            box = self.box

        cnp = CommonNeighborParameter(
            data, box, rc, self.verlet_list, self.distance_list, self.neighbor_number
        )
        cnp.compute()
        self.update_data(self.__data.with_columns(cnp=cnp.cnp[: self.N]))

    def cal_ackland_jones_analysis(self) -> None:
        """
        Perform Ackland-Jones structure analysis. This will save results to self.data['aja']:

        - 0 = Other (unknown coordination)
        - 1 = FCC (face-centered cubic)
        - 2 = HCP (hexagonal close-packed)
        - 3 = BCC (body-centered cubic)
        - 4 = ICO (icosahedral coordination)

        Notes
        -----
        See :class:`~mdapy.ackland_jones_analysis.AcklandJonesAnalysis`
        for implementation details.
        """
        N_neigh = 14
        if self.data.shape[0] < N_neigh and sum(self.box.boundary) == 0:
            self.update_data(self.__data.with_columns(pl.lit(0, pl.Int32).alias("aja")))
            return

        if hasattr(self, "neighbor_number") and self.neighbor_number.min() >= N_neigh:
            tool.sort_neighbor(
                self.verlet_list, self.distance_list, self.neighbor_number, N_neigh
            )
        else:
            self.build_nearest_neighbor(N_neigh)

        if hasattr(self, "_enlarge_data"):
            box = self._enlarge_box
            data = self._enlarge_data
        else:
            box = self.box
            data = self.__data

        aja = AcklandJonesAnalysis(data, box, self.verlet_list, self.distance_list)
        aja.compute()
        self.update_data(self.__data.with_columns(aja=aja.aja[: self.N]))

    def cal_warren_cowley_parameter(
        self, rc: float, max_neigh: Optional[int] = None
    ) -> WarrenCowleyParameter:
        """
        Calculate Warren-Cowley short-range order parameter.

        Parameters
        ----------
        rc : float
            Cutoff radius for neighbor search.
        max_neigh : int, optional
            Maximum number of neighbors per atom.

        Returns
        -------
        WarrenCowleyParameter
            Object containing Warren-Cowley parameters.

        Notes
        -----
        See :class:`~mdapy.warren_cowley_parameter.WarrenCowleyParameter`
        for implementation details and returned attributes.

        Examples
        --------
        >>> wcp = system.cal_warren_cowley_parameter(rc=3.0)
        >>> print(wcp.wcp)  # Access Warren-Cowley parameters
        """
        has_neigh = False
        if hasattr(self, "rc"):
            if self.rc >= rc:
                has_neigh = True
        if not has_neigh:
            self.build_neighbor(rc, max_neigh)
        if hasattr(self, "_enlarge_data"):
            data = self._enlarge_data
        else:
            data = self.__data

        wcp = WarrenCowleyParameter(self.verlet_list, self.neighbor_number, data)
        wcp.compute()
        return wcp

    def cal_atomic_temperature(
        self, rc: float, factor: float = 1.0, max_neigh: Optional[int] = None
    ) -> None:
        """
        Calculate the local atomic temperature for each atom by analyzing the velocity fluctuations of the atom and its neighbors.
        We assume velocity in units of Å/fs, you can use `factor` to convert your velocity.
        This will save results to self.data['atomic_temp'] in the unit of K.

        Parameters
        ----------
        rc : float
            Cutoff radius for local averaging.
        factor : float
            Scaling factor for velocities (e.g., for unit conversion).
            Default is 1.0.
        max_neigh : int, optional
            Maximum number of neighbors per atom.

        Notes
        -----
        See :class:`~mdapy.atomic_temperature.AtomicTemperature` for
        implementation details.

        """
        has_neigh = False
        if hasattr(self, "rc"):
            if self.rc >= rc:
                has_neigh = True
        if not has_neigh:
            self.build_neighbor(rc, max_neigh)
        if hasattr(self, "_enlarge_data"):
            data = self._enlarge_data
        else:
            data = self.__data

        atomTemp = AtomicTemperature(
            data, self.verlet_list, self.distance_list, rc, factor
        )
        atomTemp.compute()
        self.update_data(self.data.with_columns(atomic_temp=atomTemp.T[: self.N]))

    def cal_steinhardt_bond_orientation(
        self,
        llist: Union[np.ndarray, List[int]],
        use_voronoi: bool = False,
        nnn: int = 0,
        rc: float = -1.0,
        average: bool = False,
        use_weight: bool = False,
        weight: Optional[np.ndarray] = None,
        wl: bool = False,
        wlhat: bool = False,
        a_face_area_threshold: float = -1,
        r_face_area_threshold: float = -1,
        identify_liquid: bool = False,
        threshold: float = 0.7,
        n_bond: int = 7,
        max_neigh: Optional[int] = None,
    ) -> None:
        r"""Calculate Steinhardt bond orientation order parameters. One can also use it to identify the solid or liquid atom.

        Parameters
        ----------
        llist : np.ndarry or List[int]
            Degrees list, such as [4, 6, 8], should be positive, even integer.
        use_voronoi : bool
            If True, compute using Voronoi neighbor. Default is False.
        nnn : int
            If `use_voronoi` is False and this value is postive, compute using `nnn` nearest neighbors. Default is 0.
        rc : float
            If `use_voronoi` is False and `nnn` is zero, make sure this parameter is postive, and use it to build neighbor. Default is -1.0.
        average : bool
            If True, qlm will be averaged. Default is False.
        use_weight : bool
            If True, the neighbor will be weighed by `weight` array, otherwise the weight of each neighbor is 1.0. Default is False.
        weight : np.ndarray, optional
            If it is not None, it should be has the same shape with the `self.verlet_list`.
            If it is None and the `use_voronoi` is True, then the voronoi_face_area will be used as weight. Default is None.
        wl : bool
            If True, compute third-order invariant :math:`w_l` parameters. Default is False.
        wlhat : bool
            If True, compute normalized third-order invariant :math:`\hat{w}_l` parameters. Default is False.
        a_face_area_threshold : float
            Absolute face area threshold.
            Faces with area below this value are ignored as neighbors.
            Default is -1.0 (no filtering).
        r_face_area_threshold : float
            Relative face area threshold (fraction of average face area).
            Faces with relative area below this value are ignored.
            Default is -1.0 (no filtering).
        identify_liquid : bool
            Enable solid-liquid classification (requires :math:`l=6` in llist).
        threshold : float
            Threshold for solid-liquid identification (default: 0.7 for normalized :math:`q_6`).
        n_bond : int
            Minimum number of "solid-like" bonds for solid classification. Default to 7.
        max_neigh : int, optional
            Maximum number of neighbors per atom.

        Notes
        -----
        See :class:`~mdapy.steinhardt_bond_orientation.SteinhardtBondOrientation`
        for implementation details. The results will add to self.data['ql{l}'] for l in `llist`.
        If set `wl` is True, also will add self.data['wl{l}']. If set `wlhat` is True, also will add
        self.data['wlh{l}']. If set `identify_liquid`, it will add self.data['solidliquid'] and self.data['nbond'] results.
        """
        if use_voronoi:
            self.build_voronoi_neighbor(a_face_area_threshold, r_face_area_threshold)
            verlet_list, distance_list, neighbor_number = (
                self.voro_verlet_list,
                self.voro_distance_list,
                self.voro_neighbor_number,
            )
            if use_weight and weight is None:
                weight = self.voro_face_area
        else:
            if nnn > 0:
                has_sort_neigh = False
                if hasattr(self, "neighbor_number"):
                    if self.neighbor_number.min() >= nnn:
                        tool.sort_neighbor(
                            self.verlet_list,
                            self.distance_list,
                            self.neighbor_number,
                            nnn,
                        )
                        has_sort_neigh = True
                if not has_sort_neigh:
                    self.build_nearest_neighbor(nnn)
            else:
                assert rc > 0, (
                    "At least use voronoi, or set positive nnn, or positive rc."
                )

                if hasattr(self, "rc"):
                    if self.rc < rc:
                        self.build_neighbor(rc, max_neigh)
                else:
                    self.build_neighbor(rc, max_neigh)

            verlet_list, distance_list, neighbor_number = (
                self.verlet_list,
                self.distance_list,
                self.neighbor_number,
            )
        if hasattr(self, "_enlarge_data"):
            box = self._enlarge_box
            data = self._enlarge_data
        else:
            box = self.box
            data = self.__data
        SBO = SteinhardtBondOrientation(
            box,
            data,
            np.asarray(llist, int),
            nnn,
            rc,
            average,
            use_voronoi,
            use_weight,
            weight,
            verlet_list,
            distance_list,
            neighbor_number,
            wl,
            wlhat,
            identify_liquid,
            threshold,
            n_bond,
        )
        SBO.compute()
        if SBO.qnarray.shape[1] > 1:
            columns = []
            for i in llist:
                columns.append(f"ql{i}")
            if wl:
                for i in llist:
                    columns.append(f"wl{i}")
            if wlhat:
                for i in llist:
                    columns.append(f"wlh{i}")

            for i, name in enumerate(columns):
                self.__data = self.__data.with_columns(
                    pl.lit(SBO.qnarray[: self.N, i]).alias(name)
                )
        else:
            self.__data = self.__data.with_columns(
                pl.lit(SBO.qnarray.flatten()[: self.N]).alias(f"ql{llist[0]}")
            )
        if identify_liquid:
            self.__data = self.__data.with_columns(
                pl.lit(SBO.solidliquid[: self.N]).alias("solidliquid")
            )
            self.__data = self.__data.with_columns(
                pl.lit(SBO.nbond[: self.N]).alias("nbond")
            )

    def cal_polyhedral_template_matching(
        self,
        structure="fcc-hcp-bcc",
        rmsd_threshold=0.1,
        return_ordering=False,
        return_rmsd=False,
        return_atomic_distance=False,
        return_orientation=False,
        identify_fcc_planar_faults=False,
        identify_esf=True,
    ):
        """
        Identify crystal structures using polyhedral template matching, one can also further distiguish the planar defects in FCC structures.

        For PTM, structure results will add as self.data['ptm'] and self.data['ordering']

        - Structure type (integer, 0=Other, 1=FCC, 2=HCP, 3=BCC, 4=ICO, 5=SC, 6=DCUB, 7=DHEX, 8=Graphene)
        - Ordering type (interger, 0=Other, 1=L10, 2=L12 (A-site), 3=L12 (B-site), 4=B2, 5=zincblende / wurtzite)

        For planar defects, structure results will add as self.data['pft']

        - 0: Non-hcp atoms (e.g. perfect fcc or disordered)
        - 1: Indeterminate hcp-like (isolated hcp-like atoms, not forming a planar defect)
        - 2: Intrinsic stacking fault (ISF, two adjacent hcp-like layers)
        - 3: Coherent twin boundary (TB, one hcp-like layer)
        - 4: Multi-layer stacking fault (three or more adjacent hcp-like layers)
        - 5: Extrinsic Stacking Fault (ESF, if cal_esf=True)

        Parameters
        ----------
        structure : str
            String specifying the structure types to consider, separated by hyphens (Defaults is "fcc-hcp-bcc").
            Supported values: "fcc", "hcp", "bcc", "ico", "sc", "dcub", "dhex", "graphene".
            Special values: "all" (all types), "default" (fcc, hcp, bcc).
        rmsd_threshold : float
            Maximum RMSD for a valid structure match. Particles exceeding this are classified as "Other".
            Default is 0.1.
        return_ordering : bool
            If True, return the ordering type.
            Default is False.
        return_rmsd : bool
            If True, return the RMSD value.
            Default is False.
        return_atomic_distance : bool
            If True, return the interatomic distance.
            Default is False.
        return_orientation : bool
            If True, return the orientation value.
            Default is False.
        identify_fcc_planar_faults : bool
            If True, further detect the plannar defects.
            Default is False.
        identify_esf : bool
            If True, further defect the ESF.
            Default is True.

        Notes
        -----
        See :class:`~mdapy.polyhedral_template_matching.PolyhedralTemplateMatching` and class:`~mdapy.identify_fcc_planar_faults.IdentifyFccPlanarFaults`
        for implementation details.

        """

        verlet_list = None
        if self.N > 250:  # safe atom number
            if hasattr(self, "neighbor_number"):
                if self.neighbor_number.min() >= 18:
                    tool.sort_neighbor(
                        self.verlet_list, self.distance_list, self.neighbor_number, 18
                    )
                    verlet_list = self.verlet_list

        if hasattr(self, "_enlarge_data"):
            box = self._enlarge_box
            data = self._enlarge_data
        else:
            box = self.box
            data = self.__data

        ptm = PolyhedralTemplateMatching(
            structure, data, box, rmsd_threshold, verlet_list
        )
        ptm.compute()
        output = ptm.output[: self.N]
        data = self.__data.with_columns(pl.lit(output[:, 0], pl.Int32).alias("ptm"))
        if return_ordering:
            data = data.with_columns(pl.lit(output[:, 1]).alias("ordering"))
        if return_rmsd:
            data = data.with_columns(pl.lit(output[:, 2]).alias("rmsd"))
        if return_atomic_distance:
            data = data.with_columns(pl.lit(output[:, 3]).alias("interatomic_distance"))
        if return_orientation:
            data = data.with_columns(
                qx=output[:, 5],
                qy=output[:, 6],
                qz=output[:, 7],
                qw=output[:, 4],
            )

        if identify_fcc_planar_faults:
            structure_types = np.array(ptm.output[:, 0], np.int32)
            ptm_indices = np.ascontiguousarray(ptm.ptm_indices[:, 1:13])
            ifpt = IdentifyFccPlanarFaults(structure_types, ptm_indices, identify_esf)
            ifpt.compute()
            data = data.with_columns(pl.lit(ifpt.fault_types[: self.N]).alias("pft"))

        self.update_data(data)

    def cal_centro_symmetry_parameter(self, N: int):
        """
        Calculate centro-symmetry parameter for defect identification. The results will be added as self.data['csp'].

        Parameters
        ----------
        N : int
            Number of nearest neighbors to consider, should be a positive, even integer. 12 for FCC and 8 for BCC.

        Notes
        -----
        See :class:`~mdapy.centro_symmetry_parameter.CentroSymmetryParameter`
        for implementation details.
        """
        assert N % 2 == 0 and N > 0, f"N must be a positive even number: {N}."
        if self.N <= N and sum(self.box.boundary) == 0:
            res = np.full(self.N, 10000, float)
        else:
            has_verlet = False
            if hasattr(self, "neighbor_number"):
                if self.neighbor_number.min() >= N and hasattr(self, "rc"):
                    tool.sort_neighbor(
                        self.verlet_list, self.distance_list, self.neighbor_number, 18
                    )
                    has_verlet = True
            if not has_verlet:
                self.build_nearest_neighbor(N)
            if hasattr(self, "_enlarge_data"):
                box = self._enlarge_box
                data = self._enlarge_data
            else:
                box = self.box
                data = self.__data
            csp = CentroSymmetryParameter(data, box, N, self.verlet_list)
            csp.compute()
            res = csp.csp[: self.N]
        self.update_data(self.data.with_columns(csp=res))

    def cal_common_neighbor_analysis(
        self, rc: Optional[float] = None, max_neigh: Optional[int] = None
    ):
        """
        Perform common neighbor analysis for structure identification. This will generate 'cna' column in self.data:

        - 0 = Other (unknown coordination)
        - 1 = FCC (face-centered cubic)
        - 2 = HCP (hexagonal close-packed)
        - 3 = BCC (body-centered cubic)
        - 4 = ICO (icosahedral coordination)

        Parameters
        ----------
        rc : float, optional
            Cutoff radius. If not given, it will use adaptive mode to do CNA.
        max_neigh : int, optional
            Maximum number of neighbors per atom.

        Notes
        -----
        See :class:`~mdapy.common_neighbor_analysis.CommonNeighborAnalysis`
        for implementation details.

        """
        verlet_list = None
        neighbor_number = None
        if self.N > 500:  # safe atom number
            if hasattr(self, "rc"):
                if rc is None:
                    if self.neighbor_number.min() >= 14:
                        tool.sort_neighbor(
                            self.verlet_list,
                            self.distance_list,
                            self.neighbor_number,
                            14,
                        )
                        verlet_list = self.verlet_list
                else:
                    if self.rc < rc:
                        self.build_neighbor(rc, max_neigh)
                        verlet_list = self.verlet_list
                        neighbor_number = self.neighbor_number
            else:
                if rc is not None:
                    self.build_neighbor(rc, max_neigh)
                    verlet_list = self.verlet_list
                    neighbor_number = self.neighbor_number

        if hasattr(self, "_enlarge_data"):
            box = self._enlarge_box
            data = self._enlarge_data
        else:
            box = self.box
            data = self.__data

        cna = CommonNeighborAnalysis(data, box, verlet_list, neighbor_number, rc)
        cna.compute()
        self.update_data(self.__data.with_columns(cna=cna.pattern[: self.N]))

    def cal_structure_factor(
        self,
        k_min: float,
        k_max: float,
        nbins: int,
        cal_partial: bool = False,
        atomic_form_factors: bool = False,
        mode: str = "direct",
    ) -> StructureFactor:
        """
        Calculate static structure factor S(k).

        Parameters
        ----------
        k_min : float
            Minimum wave vector magnitude.
        k_max : float
            Maximum wave vector magnitude.
        nbins : int
            Number of bins for k-space averaging.
        cal_partial : bool, optional
            If True, calculate partial structure factors. Default is False.
        atomic_form_factors : bool, default=False
            If True, use atomic form factors f to weigh the atoms' individual contributions to S(k). Atomic form factors are taken from `TU Graz <https://lampz.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php>`_.
        mode : str, optional
            Calculation mode: 'direct' or 'debye'. Default is 'direct'.

        Returns
        -------
        StructureFactor
            Object containing structure factor data.

        Notes
        -----
        See :class:`~mdapy.structure_factor.StructureFactor` for
        implementation details and returned attributes.
        """
        sfc = StructureFactor(
            self.data,
            self.box,
            k_min,
            k_max,
            nbins,
            cal_partial,
            atomic_form_factors,
            mode,
        )
        sfc.compute()
        return sfc

    def cal_bond_analysis(
        self, rc: float, nbin: int, max_neigh: Optional[int] = None
    ) -> BondAnalysis:
        """
        Analyze bond length and angle distributions.

        Parameters
        ----------
        rc : float
            Cutoff radius for bond search.
        nbin : int
            Number of bins for histograms.
        max_neigh : int, optional
            Maximum number of neighbors per atom.

        Returns
        -------
        BondAnalysis
            Object containing bond distribution data.

        Notes
        -----
        See :class:`~mdapy.bond_analysis.BondAnalysis` for implementation
        details and returned attributes.

        Examples
        --------
        >>> ba = system.cal_bond_analysis(rc=5.0, nbin=100)
        >>> print(ba.bond_length_distribution)
        """
        has_neigh = False
        if hasattr(self, "rc"):
            if self.rc >= rc:
                has_neigh = True
        if not has_neigh:
            self.build_neighbor(rc, max_neigh)

        if hasattr(self, "_enlarge_box"):
            box = self._enlarge_box
            data = self._enlarge_data
        else:
            box = self.box
            data = self.data
        ba = BondAnalysis(
            data,
            box,
            rc,
            nbin,
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
        )
        ba.compute()
        return ba

    def cal_angular_distribution_function(
        self,
        rc_dict: Dict[str, List[float]],
        nbin: int,
        max_neigh: Optional[int] = None,
    ) -> AngularDistributionFunction:
        """
        Calculate angular distribution function (ADF).

        Parameters
        ----------
        rc_dict : dict of str to list of float
            Dictionary mapping triplet patterns to cutoff radii.
            Format: {'A-B-C': [rc_AB_min, rc_AB_max, rc_AC_min, rc_AC_max]}
            where A, B, C are element symbols and A is the central atom.
            Example: {'O-H-H': [0, 2.0, 0, 2.0]} for water molecules.
        nbin : int
            Number of angular bins.
        max_neigh : int, optional
            Maximum number of neighbors per atom.

        Returns
        -------
        AngularDistributionFunction
            Object containing ADF data.

        Notes
        -----
        See :class:`~mdapy.angular_distribution_function.AngularDistributionFunction`
        for implementation details and returned attributes.

        """
        assert "element" in self.data.columns
        rc = np.array(list(rc_dict.values())).max()

        has_neigh = False
        if hasattr(self, "rc"):
            if self.rc >= rc:
                has_neigh = True
        if not has_neigh:
            self.build_neighbor(rc, max_neigh)

        if hasattr(self, "_enlarge_box"):
            box = self._enlarge_box
            data = self._enlarge_data
        else:
            box = self.box
            data = self.data
        adf = AngularDistributionFunction(
            data,
            box,
            rc_dict,
            nbin,
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
        )
        adf.compute()
        return adf

    def cal_radial_distribution_function(
        self, rc: float, nbin: int = 100, max_neigh: Optional[int] = None
    ) -> RadialDistributionFunction:
        """
        Calculate radial distribution function g(r).

        Parameters
        ----------
        rc : float
            Maximum distance for RDF calculation.
        nbin : int, optional
            Number of distance bins. Default is 100.
        max_neigh : int, optional
            Maximum number of neighbors per atom.

        Returns
        -------
        RadialDistributionFunction
            Object containing RDF data and partial RDFs if applicable.

        Notes
        -----
        See :class:`~mdapy.radial_distribution_function.RadialDistributionFunction`
        for implementation details and returned attributes.
        """
        has_neigh = False
        if hasattr(self, "rc"):
            if self.rc >= rc:
                has_neigh = True
        if not has_neigh:
            self.build_neighbor(rc, max_neigh)

        if hasattr(self, "_enlarge_box"):
            box = self._enlarge_box
            data = self._enlarge_data
        else:
            box = self.box
            data = self.data

        if "element" in data.columns:
            ele2type = {j: i + 1 for i, j in enumerate(data["element"].unique().sort())}
            type_list = data.with_columns(
                pl.col("element").replace_strict(ele2type).alias("type")
            )["type"].to_numpy()
            self.__data = self.data.with_columns(type=type_list[: self.N])
        elif "type" in data.columns:
            type_list = data["type"].to_numpy()
        else:
            type_list = np.zeros(data.shape[0], np.int32)

        rdf = RadialDistributionFunction(
            rc,
            nbin,
            box,
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
            type_list,
        )
        rdf.compute()
        return rdf

    def average_by_neighbor(
        self,
        average_rc: float,
        property_name: str,
        include_self: bool = True,
        output_name: Optional[str] = None,
        max_neigh: Optional[int] = None,
    ) -> None:
        """
        Average a property over local neighborhoods.

        Parameters
        ----------
        average_rc : float
            Cutoff radius for local averaging.
        property_name : str
            Name of the column in data to average.
        include_self : bool
            Whether to include the central atom in averaging. Default is True.
        output_name : str, optional
            Name for the output column. If None, uses '{property_name}_ave'.
        max_neigh : int, optional
            Maximum number of neighbors per atom.

        Raises
        ------
        AssertionError
            If property_name is not a column in data.

        """
        assert property_name in self.__data.columns, f"{property_name} not in data."
        if hasattr(self, "rc"):
            if self.rc < average_rc:
                self.build_neighbor(average_rc, max_neigh)
        else:
            self.build_neighbor(average_rc, max_neigh)
        assert not hasattr(self, "_enlarge_data"), "Only supprot for big box."
        data = self.data
        if hasattr(self, "_enlarge_data"):
            data = self._enlarge_data
        data = tool.average_by_neighbor(
            average_rc,
            data,
            property_name,
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
            include_self,
            output_name,
        )[: self.N].rechunk()
        self.update_data(data)

    def cal_cluster_analysis(
        self,
        rc: Union[float, int, Dict[str, float]] = 5.0,
        max_neigh: Optional[int] = None,
    ) -> None:
        """
        Perform cluster analysis to identify connected atomic groups. The results will be added as self.data['cluster_id'].

        Parameters
        ----------
        rc : float, int, or dict, optional
            Cutoff radius for cluster connectivity. Can be:

            * **float/int**: Single cutoff for all atom pairs
            * **dict**: Type-specific cutoffs, e.g., {'1-1': 1.5, '1-2': 1.3}

            Default is 5.0.
        max_neigh : int, optional
            Maximum number of neighbors per atom.

        Notes
        -----
        See :class:`~mdapy.cluster_analysis.ClusterAnalysis` for
        implementation details.

        Examples
        --------
        Single cutoff:

        >>> system.cal_cluster_analysis(rc=3.0)

        Type-specific cutoffs:

        >>> rc_dict = {"1-1": 2.8, "1-2": 3.0, "2-2": 3.2}
        >>> system.cal_cluster_analysis(rc=rc_dict)
        >>> # cluster_id column added to system.data
        """
        if isinstance(rc, float) or isinstance(rc, int):
            max_rc = rc
        elif isinstance(rc, dict):
            max_rc = max([i for i in rc.values()])
        else:
            raise "rc should be a positive number, or a dict like {'1-1':1.5, '1-2':1.3}"
        if hasattr(self, "rc"):
            if self.rc < max_rc:
                self.build_neighbor(max_rc, max_neigh)
        else:
            self.build_neighbor(max_rc, max_neigh)

        type_list = None
        if isinstance(rc, dict):
            assert "type" in self.data.columns, (
                "Must have type for multi rc cluster calculation."
            )
            type_list = self.data["type"].to_numpy(zero_copy_only=True)
        ca = ClusterAnalysis(
            rc, self.verlet_list, self.distance_list, self.neighbor_number, type_list
        )
        ca.compute()
        self.__data = self.data.with_columns(cluster_id=ca.particleClusters[: self.N])

    def cal_structure_entropy(
        self,
        rc: float,
        sigma: float,
        use_local_density: bool = False,
        average_rc: float = 0.0,
        max_neigh: Optional[int] = None,
    ) -> None:
        """
        Calculate structural entropy for disorder quantification.

        Parameters
        ----------
        rc : float
            Cutoff radius for neighbor search.
        sigma : float
            Width parameter for Gaussian kernel.
        use_local_density : bool, optional
            If True, use local density normalization. Default is False.
        average_rc : float, optional
            Radius for local averaging of entropy. If 0, no averaging.
            Default is 0.0.
        max_neigh : int, optional
            Maximum number of neighbors per atom.

        Notes
        -----
        See :class:`~mdapy.structure_entropy.StructureEntropy` for
        implementation details.

        Examples
        --------
        >>> system.cal_structure_entropy(rc=5.0, sigma=0.5)
        >>> # entropy column added to system.data

        With local averaging:

        >>> system.cal_structure_entropy(rc=5.0, sigma=0.5, average_rc=3.0)
        >>> # entropy and entropy_ave columns added to system.data
        """
        if hasattr(self, "rc"):
            if self.rc < rc:
                self.build_neighbor(rc, max_neigh)
        else:
            self.build_neighbor(rc, max_neigh)

        if hasattr(self, "_enlarge_data"):
            box = self._enlarge_box
        else:
            box = self.box
        SE = StructureEntropy(
            box,
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
            rc,
            sigma,
            use_local_density,
            average_rc,
        )
        SE.compute()
        data = self.data.with_columns(entropy=SE.entropy[: self.N])
        if average_rc > 0:
            data = self.data.with_columns(entropy_ave=SE.entropy_ave[: self.N])
        self.update_data(data)

    def cal_voronoi_volume(self) -> None:
        """
        Calculate Voronoi cell volumes and related properties.

        This method computes the Voronoi tessellation and adds three columns
        to the data DataFrame:

        * **volume**: Atomic volume from Voronoi cell
        * **neighbor_number**: Coordination number from Voronoi neighbors
        * **cavity_radius**: This is the distance from the atom to the farthest vertex of its Voronoi cell, representing the radius of the largest empty sphere (containing no other atoms) that touches the atom.

        Notes
        -----
        See :class:`~mdapy.voronoi.Voronoi` for implementation details.

        Examples
        --------
        >>> system.cal_voronoi_volume()
        >>> # volume, neighbor_number, cavity_radius columns added to data
        >>> avg_volume = system.data["volume"].mean()
        """
        vor = Voronoi(self.box, self.data)
        volume, neighbor_number, cavity_radius = vor.get_volume()
        self.update_data(
            self.data.with_columns(
                volume=volume,
                neighbor_number=neighbor_number,
                cavity_radius=cavity_radius,
            )
        )


if __name__ == "__main__":
    pass
