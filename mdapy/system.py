# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import polars as pl
import numpy as np
import taichi as ti
import multiprocessing as mt
import re
from tqdm import tqdm


try:
    from tool_function import atomic_numbers, vdw_radii, atomic_masses
    from load_save_data import BuildSystem, SaveFile
    from tool_function import _wrap_pos, _partition_select_sort, _unwrap_pos
    from ackland_jones_analysis import AcklandJonesAnalysis
    from atomic_strain import AtomicStrain
    from bond_analysis import BondAnalysis
    from box import init_box
    from common_neighbor_analysis import CommonNeighborAnalysis
    from common_neighbor_parameter import CommonNeighborParameter
    from neighbor import Neighbor
    from temperature import AtomicTemperature
    from centro_symmetry_parameter import CentroSymmetryParameter
    from entropy import AtomicEntropy
    from identify_SFs_TBs import IdentifySFTBinFCC
    from identify_diamond_structure import IdentifyDiamondStructure
    from orthogonal_box import OrthogonalBox
    from pair_distribution import PairDistribution
    from polyhedral_template_matching import PolyhedralTemplateMatching
    from cluser_analysis import ClusterAnalysis
    from potential import EAM, NEP
    from void_distribution import VoidDistribution
    from warren_cowley_parameter import WarrenCowleyParameter
    from voronoi_analysis import VoronoiAnalysis
    from voronoi import _voronoi_analysis
    from minimizer import Minimizer
    from mean_squared_displacement import MeanSquaredDisplacement
    from lindemann_parameter import LindemannParameter
    from spatial_binning import SpatialBinning
    from steinhardt_bond_orientation import SteinhardtBondOrientation
    from replicate import Replicate
    from tool_function import _check_repeat_cutoff

except Exception:
    from .tool_function import atomic_numbers, vdw_radii, atomic_masses
    from .load_save_data import BuildSystem, SaveFile
    from .tool_function import _wrap_pos, _partition_select_sort, _unwrap_pos
    from .common_neighbor_analysis import CommonNeighborAnalysis
    from .ackland_jones_analysis import AcklandJonesAnalysis
    from .atomic_strain import AtomicStrain
    from .bond_analysis import BondAnalysis
    from .box import init_box
    from .common_neighbor_parameter import CommonNeighborParameter
    from .neighbor import Neighbor
    from .temperature import AtomicTemperature
    from .centro_symmetry_parameter import CentroSymmetryParameter
    from .entropy import AtomicEntropy
    from .identify_SFs_TBs import IdentifySFTBinFCC
    from .identify_diamond_structure import IdentifyDiamondStructure
    from .orthogonal_box import OrthogonalBox
    from .pair_distribution import PairDistribution
    from .polyhedral_template_matching import PolyhedralTemplateMatching
    from .cluser_analysis import ClusterAnalysis
    from .potential import EAM, NEP
    from .void_distribution import VoidDistribution
    from .warren_cowley_parameter import WarrenCowleyParameter
    from .voronoi_analysis import VoronoiAnalysis
    from .minimizer import Minimizer
    from .mean_squared_displacement import MeanSquaredDisplacement
    from .lindemann_parameter import LindemannParameter
    from .spatial_binning import SpatialBinning
    from .steinhardt_bond_orientation import SteinhardtBondOrientation
    from .replicate import Replicate
    from .tool_function import _check_repeat_cutoff
    import _voronoi_analysis


class System:
    """This class can generate a System class for rapidly accessing almost all the analysis
    method in mdapy.

    .. note::
      - mdapy now supports both rectangle and triclinic box from version 0.9.0.
      - mdapy only supports the simplest DATA format, atomic and charge, which means like bond information will cause an error.
      - We recommend you use DUMP as input file format or directly give particle positions and box.

    Args:
        filename (str, optional): DATA/DUMP filename. Defaults to None.
        fmt (str, optional): selected in ['data', 'lmp', 'dump', 'dump.gz', 'poscar', 'xyz', 'cif'], One can explicitly assign the file format or mdapy will handle it with the postsuffix of filename. Defaults to None.
        data (polars.Dataframe, optional): all particles information. Defaults to None.
        box (np.ndarray, optional): (:math:`4, 3` or :math:`3, 2`) system box. Defaults to None.
        pos (np.ndarray, optional): (:math:`N_p, 3`) particles positions. Defaults to None.
        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].
        vel (np.ndarray, optional): (:math:`N_p, 3`) particles velocities. Defaults to None.
        type_list (np.ndarray, optional): (:math:`N_p`) type per particles. Defaults to 1.
        type_name (list, optional): one can assign the type name per type, such as ['Al', 'C'], indicate type1 is Al and type2 is C. Defaults to None.
        sorted_id (bool, optional): whether sort system data by the particle id. Defaults to False.

    .. note::
      - Mdapy supports load/save `POSCAR format <https://www.vasp.at/wiki/index.php/POSCAR>`_ from version 0.9.6.
        We will convert the box vector to be compatiable with that `defined in lammps <https://docs.lammps.org/Howto_triclinic.html>`_.

    .. note::
      - Mdapy supports load/save `xyz file with classical and extended format <https://www.ovito.org/manual/reference/file_formats/input/xyz.html#file-formats-input-xyz-extended-format>`_ from version 0.9.6.

    Examples:

        There are two ways to create a System class.
        The first is directly reading from a DUMP/DATA file generated from LAMMPS.

        >>> import mdapy as mp

        >>> mp.init('cpu')

        >>> system = mp.System('example.dump')

        One can also create a System by giving pos, box manually.

        >>> import numpy as np

        >>> box = np.array([[0, 100], [0, 100], [0, 100.]])

        >>> pos = np.random.random((100, 3))*100

        >>> system = mp.System(box=box, pos=pos)

        Then one can access almost all the analysis method in mdapy with uniform API.

        >>> system.cal_atomic_entropy() # calculate the atomic entropy

        One can check the calculation results:

        >>> system.data

        And easily save it into disk with DUMP/DATA format.

        >>> system.write_dump()
    """

    def __init__(
        self,
        filename=None,
        fmt=None,
        data=None,
        box=None,
        pos=None,
        boundary=[1, 1, 1],
        vel=None,
        type_list=None,
        type_name=None,
        sorted_id=False,
    ) -> None:
        self.__filename = filename
        self.__fmt = fmt
        self.__timestep = 0
        self.__global_info = {}
        if (
            isinstance(data, pl.DataFrame)
            and isinstance(box, np.ndarray)
            and isinstance(boundary, list)
        ):
            self.__data, self.__box, self.__boundary = BuildSystem.fromdata(
                data, box, boundary
            )
        elif isinstance(self.__filename, str):
            self.__fmt = BuildSystem.getformat(self.__filename, fmt)
            if self.__fmt in ["dump", "dump.gz"]:
                (
                    self.__data,
                    self.__box,
                    self.__boundary,
                    self.__timestep,
                ) = BuildSystem.fromfile(self.__filename, self.__fmt)
            elif self.__fmt in ["xyz"]:
                self.__data, self.__box, self.__boundary, self.__global_info = (
                    BuildSystem.fromfile(self.__filename, self.__fmt)
                )
                if "time" in self.__global_info.keys():
                    self.__timestep = self.__global_info["time"]
            elif self.__fmt in ["data", "lmp", "poscar", "cif"]:
                self.__data, self.__box, self.__boundary = BuildSystem.fromfile(
                    self.__filename, self.__fmt
                )

        elif isinstance(pos, np.ndarray) and isinstance(boundary, list):
            self.__data, self.__box, self.__boundary = BuildSystem.fromarray(
                pos, box, boundary, vel, type_list
            )

        if type_name is not None:
            assert self.__data["type"].max() <= len(type_name)

            type_dict = {str(i): type_name[i - 1] for i in self.__data["type"].unique()}

            self.__data = self.__data.with_columns(
                pl.col("type")
                .cast(pl.Utf8)
                .replace_strict(type_dict)
                .alias("type_name")
            )

        if sorted_id:
            assert "id" in self.__data.columns
            self.__data = self.__data.sort("id")
        self.if_neigh = False
        self.__if_displayed = False
        self.update_pos()
        if "vx" in self.__data.columns:
            self.update_vel()

    @property
    def filename(self):
        """obtain filename.

        Returns:
            str: filename.
        """
        return self.__filename

    @property
    def fmt(self):
        """obtain file format.

        Returns:
            str: file format.
        """
        return self.__fmt

    @property
    def global_info(self):
        """obtaine global info, such as energy, virial and stress.

        Returns:
            dict: global information.
        """
        return self.__global_info

    @property
    def data(self) -> pl.DataFrame:
        """check particles information.

        Returns:
            polars.Dataframe: particles information.
        """
        return self.__data

    @property
    def box(self):
        """box information.

        Returns:
            np.ndarray: box information.
        """
        return self.__box

    @property
    def boundary(self):
        """boundary information.

        Returns:
            list: boundary information.
        """
        return self.__boundary

    def update_pos(self):
        """Call it only if you modify the positions information by modify the data."""
        assert "x" in self.__data.columns, "Must contains the position information."
        self.__pos = np.c_[self.__data["x"], self.__data["y"], self.__data["z"]]
        self.__pos.flags.writeable = False
        if self.__if_displayed == True:
            self.__if_displayed = False

    def update_vel(self):
        """Call it only if you modify the velocities information by modify the data."""
        assert "vx" in self.__data.columns, "Must contains the velocity information."
        self.__vel = np.c_[self.__data["vx"], self.__data["vy"], self.__data["vz"]]
        self.__vel.flags.writeable = False

    @property
    def pos(self):
        """particle position information. Do not change it directly.
        If you want to modify the positions, modify the data and call self.update_pos()

        Returns:
            np.ndarray: position information.
        """

        return self.__pos

    @property
    def vel(self):
        """particle velocity information. Do not change it directly.
        If you want to modify the velocities, modify the data and call self.update_vel()

        Returns:
            np.ndarray: velocity information.
        """
        if "vx" in self.__data.columns:
            return self.__vel
        else:
            raise "No Velocity found."

    @property
    def N(self):
        """particle number.

        Returns:
            int: particle number.
        """
        return self.__data.shape[0]

    @property
    def vol(self):
        """system volume.

        Returns:
            float: system volume.
        """
        return np.inner(self.__box[0], np.cross(self.__box[1], self.__box[2]))

    @property
    def rho(self):
        """system number density.

        Returns:
            float: system number density.
        """
        return self.N / self.vol

    @property
    def verlet_list(self):
        """verlet neighbor information. Each row indicates the neighbor atom's indice.

        Returns:
            np.ndarray: verlet information.
        """
        if self.if_neigh:
            return self.__verlet_list
        else:
            raise "No Neighbor Information found. Call build_neighbor() please."

    @property
    def distance_list(self):
        """distance neighbor information. Each row indicates the neighbor atom's distance.

        Returns:
            np.ndarray: distance information.
        """
        if self.if_neigh:
            return self.__distance_list
        else:
            raise "No Neighbor Information found. Call build_neighbor() please."

    @property
    def neighbor_number(self):
        """neighbor number information. Each row indicates the neighbor atom's number.

        Returns:
            np.ndarray: neighbor number.
        """
        if self.if_neigh:
            return self.__neighbor_number
        else:
            raise "No Neighbor Information found. Call build_neighbor() please."

    @property
    def rc(self):
        """current cutoff distance.

        Returns:
            float: cutoff distance.
        """
        if self.if_neigh:
            return self.__rc
        else:
            raise "No Neighbor Information found. Call build_neighbor() please."

    def change_filename(self, filename):
        """change the filename.

        Args:
            filename (str): the new filename.
        """
        assert isinstance(filename, str)
        self.__filename = filename

    def __repr__(self):
        if not self.__global_info:
            return f"Filename: {self.filename}\nAtom Number: {self.N}\nSimulation Box:\n{self.box}\nTimeStep: {self.__timestep}\nBoundary: {self.boundary}\nParticle Information:\n{self.__data}"
        else:
            info = f"Filename: {self.filename}\nAtom Number: {self.N}\nSimulation Box:\n{self.box}\nTimeStep: {self.__timestep}\nBoundary: {self.boundary}\n"
            for key in self.__global_info.keys():
                if key not in ["time", "lattice"]:
                    info += f"{key}: {self.__global_info[key]}\n"
            info += f"Particle Information:\n{self.__data}"
            return info

    def display(self):
        """Visualize the System."""
        try:
            import k3d
        except ModuleNotFoundError:
            raise "One should install k3d (https://github.com/K3D-tools/K3D-jupyter) to visualize the System. try: pip install k3d"
        try:
            from visualize import Visualize
        except:
            from .visualize import Visualize

        if not self.__if_displayed:
            self.view = Visualize(self.__data, self.__box)
            self.__if_displayed = True

        self.view.display()

    def atoms_colored_by(self, values, vmin=None, vmax=None, cmap="rainbow"):
        """Rendered atoms by the given values.

        Args:
            values (str): column names in data. Atoms will be colored base on it. A numpy.ndarray or polars.Series is also accept.
            vmin (float, optional): color range min, if not give, use the values.min(). Defaults to None.
            vmax (float, optional): color range max, if not give, use the values.max(). Defaults to None.
            cmap (str, optional): colormap name from matplotlib, check https://matplotlib.org/stable/users/explain/colors/colormaps.html. Defaults to "rainbow".
        """
        try:
            import k3d
        except ModuleNotFoundError:
            raise "One should install k3d (https://github.com/K3D-tools/K3D-jupyter) to visualize the System. try: pip install k3d"
        try:
            from visualize import Visualize
        except:
            from .visualize import Visualize
        if not self.__if_displayed:
            self.view = Visualize(self.__data, self.__box)
            self.view.atom_colored_by(values, vmin, vmax, cmap)
            self.view.display()
            self.__if_displayed = True
        else:
            self.view.data = self.__data
            self.view.atom_colored_by(values, vmin, vmax, cmap)

    def cal_atomic_strain(self, ref, rc=5.0, affi_map="off"):
        """This class is used to calculate the atomic shear strain. More details can be found here.
        https://www.ovito.org/docs/current/reference/pipelines/modifiers/atomic_strain.html

        Args:
            ref (mdapy.System): a reference system object.
            rc (float, optional): cutoff distance to determine the neighbor environments. Defaults to 5. A.
            affi_map (str, optional): selected in ['off', 'ref']. If use to 'ref', the current position will affine to the reference frame. Defaults to 'off'.

        Outputs:
            - **The result is added in self.data['shear_strain']**.
        """
        assert isinstance(ref, System), "ref must be a mdapy System object."
        assert affi_map in ["off", "ref"]
        assert self.boundary == ref.boundary
        assert self.N == ref.N
        if affi_map == "ref":
            assert self.boundary == [1, 1, 1]

        rebuild = True
        if ref.if_neigh:
            if ref.rc == rc:
                rebuild = False
        if rebuild:
            # print(f'rebuild neighbor with rc={rc}.')
            ref.build_neighbor(rc=rc)

        strain = AtomicStrain(
            ref.pos,
            ref.box,
            self.pos,
            self.box,
            ref.verlet_list,
            ref.neighbor_number,
            self.boundary,
            affi_map,
        )
        strain.compute()

        self.__data = self.__data.with_columns(
            pl.lit(strain.shear_strain).alias("shear_strain")
        )

    def minimize(
        self,
        elements_list,
        potential,
        fmax=1e-5,
        max_itre=200,
        volume_change=False,
        hydrostatic_strain=False,
    ):
        """This function use the fast inertial relaxation engine (FIRE) method to minimize the system, including optimizing position and box.

        Args:
            elements_list (list): element name, such as ['Al', 'C']
            potential (BasePotential): a BasePotential
            fmax (float, optional): maximum force per atom to consider as converged. Defaults to 1e-5.
            max_itre (int, optional): maximum iteration times. Defaults to 200.
            volume_change (bool, optional): whether change the box to optimize the pressure. Defaults to False.
            hydrostatic_strain (bool, optional): sonstrain the cell by only allowing hydrostatic deformation. Defaults to False.

        Returns:
            System: optimized system.
        """
        mini = Minimizer(
            self.data.select(["x", "y", "z"]).to_numpy(),
            self.box,
            self.boundary,
            potential,
            elements_list,
            self.data["type"].to_numpy(),
            fmax=fmax,
            max_itre=max_itre,
            volume_change=volume_change,
            hydrostatic_strain=hydrostatic_strain,
        )
        mini.compute()

        data = self.__data.with_columns(
            pl.lit(mini.pos[:, 0]).alias("x"),
            pl.lit(mini.pos[:, 1]).alias("y"),
            pl.lit(mini.pos[:, 2]).alias("z"),
        )
        return System(
            data=data,
            box=mini.box,
            boundary=self.__boundary,
            filename=self.__filename,
            fmt=self.__fmt,
        )

    def cell_opt(
        self,
        pair_parameter,
        elements_list,
        units="metal",
        atomic_style="atomic",
        extra_args=None,
        conversion_factor=None,
    ):
        """This function can be used to optimize box and position using lammps.

        Args:
            pair_parameter (str): including pair_style and pair_coeff, such as "pair_style eam/alloy\npair_coeff * * example/Al_DFT.eam.alloy Al".
            elements_list (list[str]): elements to be calculated, such as ['Al', 'Ni'].
            units (str, optional): lammps units, such as metal, real etc. Defaults to "metal".
            atomic_style (str, optional): atomic_style, such as atomic, charge etc. Defaults to "atomic".
            extra_args (str, optional): any lammps commond. Defaults to None.
            conversion_factor (float, optional): units conversion. Make sure converse the length units to A. Defaults to None.

        Returns:
            System: an optimized system.
        """
        try:
            from lammps import lammps
        except Exception:
            raise "One should install lammps-python interface to use this function. Chech the installation guide (https://docs.lammps.org/Python_install.html)."

        try:
            from cell_opt import CellOptimization
        except Exception:
            from .cell_opt import CellOptimization

        cpt = CellOptimization(
            self.pos,
            self.box,
            self.__data["type"].to_numpy(),
            elements_list,
            self.boundary,
            pair_parameter,
            units,
            atomic_style,
            extra_args,
            conversion_factor,
        )
        data, box = cpt.compute()
        if "type_name" in self.data.columns:
            type_dict = {str(i): j for i, j in enumerate(elements_list, start=1)}
            data = data.with_columns(
                pl.col("type")
                .cast(pl.Utf8)
                .replace_strict(type_dict)
                .alias("type_name")
            )
        return System(
            data=data,
            box=box,
            boundary=self.__boundary,
            filename=self.__filename,
            fmt=self.__fmt,
        )

    def select(self, data: pl.DataFrame):
        """Generate a subsystem.

        Args:
            data (polars.DataFrame): a new dataframe. Such as system.data.filter((pl.col('x')>50) & (pl.col('y')<100))

        Returns:
            System: a new subsystem.
        """
        assert isinstance(data, pl.DataFrame)
        subSystem = System(
            data=data,
            box=self.__box,
            boundary=self.__boundary,
            filename=self.__filename,
            fmt=self.__fmt,
        )
        return subSystem

    def update_data(self, data: pl.DataFrame, update_pos=False, update_vel=False):
        """Provide a interface to directly update the particle information. If you are not sure, do not use it.

        Args:
            data (pl.DataFrame): a new dataframe
            update_pos (bool, optional): if new data change the position, set it to True. Defaults to False.
            update_vel (bool, optional): if new data change the velocity, set it to True. Defaults to False.
        """
        assert isinstance(data, pl.DataFrame)
        self.__data = data
        if update_pos:
            self.update_pos()

        if update_vel:
            self.update_vel()

    def write_xyz(self, output_name=None, type_name=None, classical=False):
        """This function writes position into a XYZ file.
        Classical model only saves [type x y z] information and do not contain the box information.
        Extended model can includes any particles information, just like dump format.

        Args:
            output_name (str, optional): filename of generated XYZ file. Defaults to None.
            type_name (list, optional): assign the species name. Such as ['Al', 'Cu']. Defaults to None.
            classical (bool, optional): whether save with classical format. Defaults to False.
        """
        if output_name is None:
            if self.__filename is None:
                output_name = "output.xyz"
            else:
                output_name = ".".join(self.__filename.split(".")[:-1]) + ".output.xyz"
        data = self.__data
        if type_name is not None:
            assert len(type_name) >= self.__data["type"].max()

            type2name = {str(i + 1): j for i, j in enumerate(type_name)}

            data = self.__data.with_columns(
                pl.col("type")
                .cast(pl.Utf8)
                .replace_strict(type2name)
                .alias("type_name")
            )

        SaveFile.write_xyz(
            output_name,
            self.__box,
            data,
            self.__boundary,
            classical,
            **self.__global_info,
        )

    def write_dump(self, output_name=None, output_col=None, compress=False):
        """This function writes position into a DUMP file.

        Args:
            output_name (str, optional): filename of generated DUMP file.
            output_col (list, optional): which columns should be saved.
            compress (bool, optional): whether compress the DUMP file.
        """
        if output_name is None:
            if self.__filename is None:
                output_name = "output.dump"
            else:
                output_name = ".".join(self.__filename.split(".")[:-1]) + ".output.dump"
        if compress:
            if output_name.split(".")[-1] != "gz":
                output_name += ".gz"

        if output_col is None:
            data = self.__data
        else:
            data = self.__data.select(output_col)
        SaveFile.write_dump(
            output_name,
            self.__box,
            self.__boundary,
            data=data,
            pos=None,
            type_list=None,
            timestep=self.__timestep,
            compress=compress,
        )

    def write_cp2k(self, output_name=None, type_name=None):
        """This function writes particles information for cp2k calculation.

        Args:
            output_name (str, optional): output filename. Defaults to None.
            type_name (list, optional): species name. Such as ['Al', 'Fe'].
        """
        if output_name is None:
            if self.__filename is None:
                output_name = "output.cp2k"
            else:
                output_name = self.__filename + ".output.cp2k"

        if type_name is None:
            assert "type_name" in self.__data.columns, "One need to provide type_name."
            data = self.__data
        else:
            assert len(type_name) >= self.__data["type"].max()
            type_dict = {str(i + 1): j for i, j in enumerate(type_name)}

            data = self.__data.with_columns(
                pl.col("type")
                .cast(pl.Utf8)
                .replace_strict(type_dict)
                .alias("type_name")
            )

        SaveFile.write_cp2k(output_name, self.__box, self.__boundary, data)

    def write_cif(self, output_name=None, type_name=None):
        """This function writes particles information into a cif file.

        Args:
            output_name (str, optional): output filename. Defaults to None.
            type_name (list, optional): species name. Such as ['Al', 'Fe'].
        """

        if output_name is None:
            if self.__filename is None:
                output_name = "output.cif"
            else:
                output_name = self.__filename + ".output.cif"
        SaveFile.write_cif(output_name, self.__box, self.__data, type_name)

    def write_POSCAR(
        self,
        output_name=None,
        type_name=None,
        reduced_pos=False,
        selective_dynamics=False,
        save_velocity=False,
    ):
        """This function writes particles information into a POSCAR file.

        Args:
            output_name (str, optional): output filename. Defaults to None.
            type_name (list, optional): species name. Such as ['Al', 'Fe'].
            selective_dynamics (bool, optional): whether do selective dynamics. Defaults to False.
            reduced_pos (bool, optional): whether save directed coordination. Defaults to False.
            save_velocity (bool, optional): whether save velocities information. Defaults to False.
        """
        if output_name is None:
            if self.__filename is None:
                output_name = "output.POSCAR"
            else:
                output_name = self.__filename + ".output.POSCAR"
        SaveFile.write_POSCAR(
            output_name,
            self.__box,
            self.__data,
            type_name,
            reduced_pos,
            selective_dynamics,
            save_velocity,
        )

    def write_data(
        self, output_name=None, data_format="atomic", num_type=None, type_name=None
    ):
        """This function writes particles information into a DATA file.

        Args:
            output_name (str, optional): output filename. Defaults to None.
            data_format (str, optional): selected in ['atomic', 'charge']. Defaults to "atomic".
            num_type (int, optional): explictly assign a number of atom type. Defaults to None.
            type_name (list, optional): explictly assign elemantal name list, such as ['Al', 'C']. Defaults to None.
        """
        if output_name is None:
            if self.__filename is None:
                output_name = "output.data"
            else:
                output_name = ".".join(self.__filename.split(".")[:-1]) + ".output.data"

        SaveFile.write_data(
            output_name,
            self.__box,
            data=self.__data,
            pos=None,
            type_list=None,
            num_type=num_type,
            type_name=type_name,
            data_format=data_format,
        )

    def atom_distance(self, i, j):
        """Calculate the distance of atom :math:`i` and atom :math:`j` considering the periodic boundary.

        Args:
            i (int): atom :math:`i`.
            j (int): atom :math:`j`.

        Returns:
            float: distance between given two atoms.
        """
        box = self.__box[:-1]
        rij = self.pos[i] - self.pos[j]
        inverse_box = np.linalg.inv(box)
        n = rij @ inverse_box
        for i in range(3):
            if self.__boundary[i] == 1:
                if n[i] > 0.5:
                    n[i] -= 1
                elif n[i] < -0.5:
                    n[i] += 1
        return np.linalg.norm(n @ box)

    def wrap_pos(self):
        """Wrap atom position into box considering the periodic boundary."""

        self.__pos.flags.writeable = True
        _wrap_pos(
            self.__pos, self.box, np.array(self.boundary), np.linalg.inv(self.box[:-1])
        )
        self.__data = self.__data.with_columns(
            pl.lit(self.__pos[:, 0]).alias("x"),
            pl.lit(self.__pos[:, 1]).alias("y"),
            pl.lit(self.__pos[:, 2]).alias("z"),
        )
        self.__pos.flags.writeable = False

    def replicate(self, x=1, y=1, z=1):
        """Replicate the system.

        Args:
            x (int, optional): replication number (positive integer) along x axis. Defaults to 1.
            y (int, optional): replication number (positive integer) along y axis. Defaults to 1.
            z (int, optional): replication number (positive integer) along z axis. Defaults to 1.
        """
        assert x > 0 and isinstance(x, int), "x should be a positive integer."
        assert y > 0 and isinstance(y, int), "y should be a positive integer."
        assert z > 0 and isinstance(z, int), "z should be a positive integer."

        repli = Replicate(self.pos, self.box, x, y, z)
        repli.compute()
        num = x * y * z
        self.__data = pl.concat([self.__data] * num).with_columns(
            pl.lit(np.arange(1, repli.N + 1)).alias("id"),
            pl.lit(repli.pos[:, 0]).alias("x"),
            pl.lit(repli.pos[:, 1]).alias("y"),
            pl.lit(repli.pos[:, 2]).alias("z"),
        )
        self.update_pos()
        if "vx" in self.__data.columns:
            self.update_vel()
        self.__box = repli.box

    def build_neighbor_voronoi(self, ncore=-1):
        """Build voronoi neighbor. Only support rectangular and periodic boundary.
        After building, the system classs will have voro_neighbor_number, voro_verlet_list, voro_distance_list, voro_face_areas.
        This neighbor information is mainly used to calculate the steinhardt bond order.

        Args:
            ncore (int, optional): parallel CPU cores, -1 means use all cores. Defaults to -1.
        """
        # assert self.boundary == [1, 1, 1], "Only support all periodic boundary."
        ibox, _, rec = init_box(self.box)
        assert rec, "Only support rectangular box."
        box = np.zeros((3, 2))
        box[:, 0] = ibox[-1]
        box[:, 1] = np.array([ibox[0, 0], ibox[1, 1], ibox[2, 2]]) + box[:, 0]
        if ncore == -1:
            ncore = mt.cpu_count()
        else:
            assert ncore > 0, "ncore must be positive number."
            ncore = int(ncore)
        self.voro_neighbor_number = np.zeros(self.N, np.int32)
        self.voro_verlet_list, self.voro_distance_list, self.voro_face_areas = (
            _voronoi_analysis.get_voronoi_neighbor(
                self.pos, box, np.bool_(self.boundary), self.voro_neighbor_number, ncore
            )
        )

    def build_neighbor(self, rc=5.0, max_neigh=None):
        """Build neighbor withing a spherical distance based on the mdapy.Neighbor class.

        Args:
            rc (float, optional): cutoff distance. Defaults to 5.0.
            max_neigh (int, optional): maximum neighbor number (highly recommened to assign a number). If not given, will estimate atomatically. Default to None.

        Outputs:
            - **verlet_list** (np.ndarray) - (:math:`N_p, max\_neigh`) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1.
            - **distance_list** (np.ndarray) - (:math:`N_p, max\_neigh`) distance_list[i, j] means distance between i and j atom.
            - **neighbor_number** (np.ndarray) - (:math:`N_p`) neighbor atoms number.
            - **rc** (float) - cutoff distance.
        """
        Neigh = Neighbor(self.pos, self.box, rc, self.boundary, max_neigh=max_neigh)
        Neigh.compute()

        self.__verlet_list, self.__distance_list, self.__neighbor_number, self.__rc = (
            Neigh.verlet_list,
            Neigh.distance_list,
            Neigh.neighbor_number,
            rc,
        )
        self.if_neigh = True

    def cal_phono_dispersion(
        self,
        path,
        labels,
        potential,
        elements_list,
        symprec=1e-5,
        replicate=None,
        displacement=0.01,
        cutoff_radius=None,
    ):
        """This function can be used to calculate the phono dispersion based on Phonopy (https://phonopy.github.io/phonopy/). We support NEP and
        eam/alloy potential now.

        Args:
            path (str | np.ndarray| nested list): band path, such as '0. 0. 0. 0.5 0.5 0.5', indicating two points.
            labels (str | list): high symmetry points label, such as ["$\Gamma$", "K", "M", "$\Gamma$"].
            potential (BasePotential): base potential class defined in mdapy, which must including a compute method to calculate the energy, force, virial.
            elements_list (list[str]): element list, such as ['Al']
            pair_style (str, optional): pair style, selected in ['nep', 'eam/alloy']. Defaults to "eam/alloy".
            symprec (float): this is used to set geometric tolerance to find symmetry of crystal structure. Defaults to 1e-5.
            replicate (list, optional): replication to pos, such as [3, 3, 3]. If not given, we will replicate it exceeding 15 A per directions. Defaults to None.
            displacement (float, optional): displacement distance. Defaults to 0.01.
            cutoff_radius (float, optional): Set zero to force constants outside of cutoff distance. If not given, the force constant will consider the whole supercell. This parameter will not reduce the computation cost. Defaults to None.

        Outputs:
            **Phon** : a Phonon object, which can be used to plot phonon dispersion.
        """
        try:
            import phonopy
        except ModuleNotFoundError:
            raise "One should install phonopy (https://phonopy.github.io/phonopy/) to calculate the phono dispersion. try: pip install phonopy"
        try:
            from phonon import Phonon
        except Exception:
            from .phonon import Phonon

        self.Phon = Phonon(
            path,
            labels,
            potential,
            self.pos,
            self.box,
            elements_list,
            self.__data["type"].to_numpy(),
            symprec,
            replicate,
            displacement,
            cutoff_radius,
        )
        self.Phon.compute()

    def cal_species_number(self, element_list, search_species=None, check_most=10):
        """This function can recgnized the species based on the atom connectivity.
        For atom i and atom j, if rij <= (vdwr_i + vdwr_j) * 0.6, we think two atoms are connected.
        Similar method can be found in `OpenBabel <https://github.com/openbabel/openbabel>`_.

        Args:
            element_list (list): elemental name for your system. Such as ['C', 'H', 'O'].
            search_species (list, optional): the molecular formula you want to find. Such as ['H2O', 'CO2', 'Cl2', 'N2'].
            check_most (int, optional): if search_species is not given, we will give the N-most species.
        Returns:
            dict: search species and the corresponding number.
        """
        assert (
            len(element_list) == self.__data["type"].max()
        ), "The length of element_list must be equal to the atom type number."
        partial_cutoff = {}
        for type1 in range(len(element_list)):
            for type2 in range(type1, len(element_list)):
                partial_cutoff[f"{type1+1}-{type2+1}"] = (
                    vdw_radii[atomic_numbers[element_list[type1]]]
                    + vdw_radii[atomic_numbers[element_list[type2]]]
                ) * 0.6
        self.cal_cluster_analysis(partial_cutoff, max_neigh=None)
        if search_species is not None:
            pattern = r"([A-Z][a-z]?)(\d*)"
            trans_search_species = []
            for old_name in search_species:
                matches = re.findall(pattern, old_name)
                matches.sort()
                name = ""
                for i, j in matches:
                    if len(j):
                        name += i * int(j)
                    else:
                        name += i
                trans_search_species.append(name)
            res = (
                self.__data.with_columns(
                    pl.lit(
                        np.array(element_list)[(self.__data["type"] - 1).to_numpy()]
                    ).alias("type_name")
                )
                .group_by("cluster_id")
                .agg(pl.col("type_name"))
                .with_columns(pl.col("type_name").list.sort())
                .with_columns(pl.col("type_name").list.join(""))["type_name"]
                .value_counts()
            ).filter(pl.col("type_name").is_in(trans_search_species))
            res = dict(zip(res[:, 0], res[:, 1]))
            species = {}
            for i, j in zip(search_species, trans_search_species):
                if j not in res.keys():
                    species[i] = 0
                else:
                    species[i] = res[j]

        else:
            res = (
                (
                    self.__data.with_columns(
                        pl.lit(
                            np.array(element_list)[(self.__data["type"] - 1).to_numpy()]
                        ).alias("type_name")
                    )
                    .group_by("cluster_id")
                    .agg(pl.col("type_name"))
                    .with_columns(pl.col("type_name").list.sort())
                    .with_columns(pl.col("type_name").list.join(""))["type_name"]
                    .value_counts()
                )
                .sort("count", descending=True)
                .limit(check_most)
            )
            species = dict(zip(res[:, 0], res[:, 1]))
        return species

    def cal_atomic_temperature(
        self, amass=None, elemental_list=None, rc=5.0, units="metal", max_neigh=None
    ):
        """Calculate an average thermal temperature per atom, wchich is useful at shock
        simulations. The temperature of atom :math:`i` is given by:

        .. math:: T_i=\\sum^{N^i_{neigh}}_0 m^i_j(v_j^i -v_{COM}^i)^2/(3N^i_{neigh}k_B),

        where :math:`N^i_{neigh}` is neighbor atoms number of atom :math:`i`,
        :math:`m^i_j` and :math:`v^i_j` are the atomic mass and velocity of neighbor atom :math:`j` of atom :math:`i`,
        :math:`k_B` is the Boltzmann constant, :math:`v^i_{COM}` is
        the center of mass COM velocity of neighbor of atom :math:`i` and is given by:

        .. math:: v^i_{COM}=\\frac{\\sum _0^{N^i_{neigh}}m^i_jv_j^i}{\\sum_0^{N^i_{neigh}} m_j^i}.

        Here the neighbor of atom :math:`i` includes itself.

        Args:
            amass (np.ndarray, optional): (:math:`N_{type}`) atomic mass per species, such as np.array([1., 12.]).
            elemental_list (list, optional): (:math:`N_{type}`) elemental name, such as ['H', 'C'].
            rc (float, optional): cutoff distance. Defaults to 5.0.
            units (str, optional): units selected from ['metal', 'charge']. Defaults to "metal".
            max_neigh (int, optional): maximum neighbor number. If not given, will estimate atomatically. Default to None.

        Outputs:
            - **The result is added in self.data['atomic_temp']**.
        """

        Ntype = self.__data["type"].max()
        if amass is None:
            assert (
                elemental_list is not None
            ), "One must provide either amass or elemental_list!"
            assert (
                len(elemental_list) == Ntype
            ), f"length of elemental list should be equal to the atom type number: {Ntype}."
            amass = np.array([atomic_masses[atomic_numbers[i]] for i in elemental_list])
        else:
            amass = np.array(amass)
            assert (
                amass.shape[0] == Ntype
            ), f"length of amass should be equal to the atom type number: {Ntype}."

        assert units in ["metal", "charge"], "units must in ['metal', 'charge']."

        repeat = _check_repeat_cutoff(self.box, self.boundary, rc)
        verlet_list, distance_list = None, None
        if sum(repeat) == 3:
            if not self.if_neigh:
                self.build_neighbor(rc, max_neigh=max_neigh)
            if self.rc < rc:
                self.build_neighbor(rc, max_neigh=max_neigh)
            verlet_list, distance_list = (
                self.verlet_list,
                self.distance_list,
            )

        atype_list = self.__data["type"].to_numpy()  # to_numpy().astype(np.int32)

        assert (
            "vx" in self.__data.columns
        ), "Should contain velocity information for computing temperature."
        assert "vy" in self.__data.columns
        assert "vz" in self.__data.columns
        AtomicTemp = AtomicTemperature(
            amass,
            self.vel,
            atype_list,
            rc,
            verlet_list,
            distance_list,
            self.pos,
            self.box,
            self.boundary,
            units,
        )
        AtomicTemp.compute()
        self.__data = self.__data.with_columns(
            pl.lit(AtomicTemp.T).alias("atomic_temp")
        )

    def cal_ackland_jones_analysis(self):
        """Using Ackland Jones Analysis (AJA) method to identify the lattice structure.

        The AJA method can recgonize the following structure:

        1. Other
        2. FCC
        3. HCP
        4. BCC
        5. ICO

        .. note:: If you use this module in publication, you should also cite the original paper.
          `Ackland G J, Jones A P. Applications of local crystal structure measures in experiment and
          simulation[J]. Physical Review B, 2006, 73(5): 054104. <https://doi.org/10.1103/PhysRevB.73.054104>`_.

        .. hint:: The module uses the `legacy algorithm in LAMMPS <https://docs.lammps.org/compute_ackland_atom.html>`_.

        Outputs:
            - **The result is added in self.data['aja']**.
        """
        use_verlet = False
        if self.if_neigh:
            if self.neighbor_number.min() >= 14:
                _partition_select_sort(self.verlet_list, self.distance_list, 14)
                AcklandJonesAna = AcklandJonesAnalysis(
                    self.pos,
                    self.box,
                    self.boundary,
                    self.verlet_list,
                    self.distance_list,
                )
                AcklandJonesAna.compute()
                use_verlet = True

        if not use_verlet:
            AcklandJonesAna = AcklandJonesAnalysis(self.pos, self.box, self.boundary)
            AcklandJonesAna.compute()

        self.__data = self.__data.with_columns(pl.lit(AcklandJonesAna.aja).alias("aja"))

    def cal_steinhardt_bond_orientation(
        self,
        nnn=12,
        qlist=[6],
        rc=0.0,
        wlflag=False,
        wlhatflag=False,
        use_voronoi=False,
        use_weight=False,
        average=False,
        solidliquid=False,
        max_neigh=60,
        threshold=0.7,
        n_bond=7,
    ):
        """This function is used to calculate a set of bond-orientational order parameters :math:`Q_{\\ell}` to characterize the local orientational order in atomic structures. We first compute the local order parameters as averages of the spherical harmonics :math:`Y_{\ell m}` for each neighbor:

        .. math:: \\bar{Y}_{\\ell m} = \\frac{1}{nnn}\\sum_{j = 1}^{nnn} Y_{\\ell m}\\bigl( \\theta( {\\bf r}_{ij} ), \\phi( {\\bf r}_{ij} ) \\bigr),

        where the summation goes over the :math:`nnn` nearest neighbor and the :math:`\\theta` and the :math:`\\phi` are the azimuthal and polar
        angles. Then we can obtain a rotationally invariant non-negative amplitude by summing over all the components of degree :math:`l`:

        .. math:: Q_{\\ell}  = \\sqrt{\\frac{4 \\pi}{2 \\ell  + 1} \\sum_{m = -\\ell }^{m = \\ell } \\bar{Y}_{\\ell m} \\bar{Y}^*_{\\ell m}}.

        For a FCC lattice with :math:`nnn=12`, :math:`Q_4 = \\sqrt{\\frac{7}{192}} \\approx 0.19094`. More numerical values for commonly encountered high-symmetry structures are listed in Table 1 of `J. Chem. Phys. 138, 044501 (2013) <https://aip.scitation.org/doi/abs/10.1063/1.4774084>`_, and all data can be reproduced by this class.

        If :math:`wlflag` is True, this class will compute the third-order invariants :math:`W_{\\ell}` for the same degrees as for the :math:`Q_{\\ell}` parameters:

        .. math:: W_{\\ell} = \\sum \\limits_{m_1 + m_2 + m_3 = 0} \\begin{pmatrix}\\ell & \\ell & \\ell \\\m_1 & m_2 & m_3\\end{pmatrix}\\bar{Y}_{\\ell m_1} \\bar{Y}_{\\ell m_2} \\bar{Y}_{\\ell m_3}.

        For FCC lattice with :math:`nnn=12`, :math:`W_4 = -\\sqrt{\\frac{14}{143}} \\left(\\frac{49}{4096}\\right) \\pi^{-3/2} \\approx -0.0006722136`.

        If :math:`wlhatflag` is true, the normalized third-order invariants :math:`\\hat{W}_{\\ell}` will be computed:

        .. math:: \\hat{W}_{\\ell} = \\frac{\\sum \\limits_{m_1 + m_2 + m_3 = 0} \\begin{pmatrix}\\ell & \\ell & \\ell \\\m_1 & m_2 & m_3\\end{pmatrix}\\bar{Y}_{\\ell m_1} \\bar{Y}_{\\ell m_2} \\bar{Y}_{\\ell m_3}}{\\left(\\sum \\limits_{m=-l}^{l} |\\bar{Y}_{\ell m}|^2 \\right)^{3/2}}.

        For FCC lattice with :math:`nnn=12`, :math:`\\hat{W}_4 = -\\frac{7}{3} \\sqrt{\\frac{2}{429}} \\approx -0.159317`. More numerical values of :math:`\\hat{W}_{\\ell}` can be found in Table 1 of `Phys. Rev. B 28, 784 <https://doi.org/10.1103/PhysRevB.28.784>`_, and all data can be reproduced by this class.

        .. hint:: If you use this class in your publication, you should cite the original paper:

        `Steinhardt P J, Nelson D R, Ronchetti M. Bond-orientational order in liquids and glasses[J]. Physical Review B, 1983, 28(2): 784. <https://doi.org/10.1103/PhysRevB.28.784>`_

        .. note:: This class is translated from that in `LAMMPS <https://docs.lammps.org/compute_orientorder_atom.html>`_.

        We also further implement the bond order to identify the solid or liquid state for lattice structure. For FCC structure, one can compute the normalized cross product:

        .. math:: s_\\ell(i,j) = \\frac{4\\pi}{2\\ell + 1} \\frac{\\sum_{m=-\\ell}^{\\ell} \\bar{Y}_{\\ell m}(i) \\bar{Y}_{\\ell m}^*(j)}{Q_\\ell(i) Q_\\ell(j)}.

        According to `J. Chem. Phys. 133, 244115 (2010) <https://doi.org/10.1063/1.3506838>`_, when :math:`s_6(i, j)` is larger than a threshold value (typically 0.7), the bond is regarded as a solid bond. Id the number of solid bond is larger than a threshold (6-8), the atom is considered as solid phase.

        .. hint:: If you set `solidliquid` is True in your publication, you should cite the original paper:

        `Filion L, Hermes M, Ni R, et al. Crystal nucleation of hard spheres using molecular dynamics, umbrella sampling, and forward flux sampling: A comparison of simulation techniques[J]. The Journal of chemical physics, 2010, 133(24): 244115. <https://doi.org/10.1063/1.3506838>`_

        Args:
            nnn (int, optional): the number of nearest neighbors used to calculate :math:`Q_{\ell}`. If :math:`nnn > 0`, the :math:`rc` has no effects, otherwise the summation will go over all neighbors within :math:`rc`. Defaults to 12.
            qlist (list, optional): the list of order parameters to be computed, which should be a non-negative integer. Defaults to [6].
            rc (float, optional): cutoff distance to find neighbors. Defaults to 0.0.
            wlflag (bool, optional): whether calculate the third-order invariants :math:`W_{\ell}`. Defaults to False.
            wlhatflag (bool, optional): whether calculate the normalized third-order invariants :math:`\hat{W}_{\ell}`. If :math:`wlflag` is False, this parameter has no effect. Defaults to False.
            solidliquid (bool, optional): whether identify the solid/liquid phase. Defaults to False.
            max_neigh (int, optional): a given maximum neighbor number per atoms. Defaults to 60.
            threshold (float, optional): threshold value to determine the solid bond. Defaults to 0.7.
            n_bond (int, optional): threshold to determine the solid atoms. Defaults to 7.

        Outputs:
            - **The result is added in self.data[['ql', 'wl', 'whl', 'solidliquid']]**.
        """
        distance_list, verlet_list, neighbor_number = None, None, None
        weight = None
        if use_voronoi:
            if not hasattr(self, "voro_verlet_list"):
                self.build_neighbor_voronoi()
            distance_list, verlet_list, neighbor_number = (
                self.voro_distance_list,
                self.voro_verlet_list,
                self.voro_neighbor_number,
            )
            if use_weight:
                weight = self.voro_face_areas
        else:
            if nnn > 0:
                if self.if_neigh:
                    if self.neighbor_number.min() >= nnn:
                        _partition_select_sort(
                            self.verlet_list, self.distance_list, nnn
                        )
                        verlet_list, distance_list, neighbor_number = (
                            self.verlet_list,
                            self.distance_list,
                            self.neighbor_number,
                        )
            else:
                assert rc > 0
                repeat = _check_repeat_cutoff(self.box, self.boundary, rc)

                if sum(repeat) == 3:
                    if not self.if_neigh:
                        self.build_neighbor(rc=rc, max_neigh=max_neigh)
                    elif self.rc < rc:
                        self.build_neighbor(rc=rc, max_neigh=max_neigh)

                    verlet_list, distance_list, neighbor_number = (
                        self.verlet_list,
                        self.distance_list,
                        self.neighbor_number,
                    )

        SBO = SteinhardtBondOrientation(
            self.pos,
            self.box,
            self.boundary,
            verlet_list,
            distance_list,
            neighbor_number,
            rc,
            qlist,
            nnn,
            wlflag,
            wlhatflag,
            max_neigh,
            use_weight,
            weight,
            use_voronoi,
            average,
        )
        SBO.compute()
        if SBO.qnarray.shape[1] > 1:
            columns = []
            for i in qlist:
                columns.append(f"ql{i}")
            if wlflag:
                for i in qlist:
                    columns.append(f"wl{i}")
            if wlhatflag:
                for i in qlist:
                    columns.append(f"whl{i}")

            for i, name in enumerate(columns):
                self.__data = self.__data.with_columns(
                    pl.lit(SBO.qnarray[:, i]).alias(name)
                )
        else:
            self.__data = self.__data.with_columns(
                pl.lit(SBO.qnarray.flatten()).alias(f"ql{qlist[0]}")
            )
        if solidliquid:
            SBO.identifySolidLiquid(threshold, n_bond)
            self.__data = self.__data.with_columns(
                pl.lit(SBO.solidliquid).alias("solidliquid")
            )

    def cal_centro_symmetry_parameter(self, N=12):
        """Compute the CentroSymmetry Parameter (CSP),
        which is heluful to recgonize the structure in lattice, such as FCC and BCC.
        The  CSP is given by:

        .. math::

            p_{\mathrm{CSP}} = \sum_{i=1}^{N/2}{|\mathbf{r}_i + \mathbf{r}_{i+N/2}|^2},

        where :math:`r_i` and :math:`r_{i+N/2}` are two neighbor vectors from the central atom to a pair of opposite neighbor atoms.
        For ideal centrosymmetric crystal, the contributions of all neighbor pairs will be zero. Atomic sites within a defective
        crystal region, in contrast, typically have a positive CSP value.

        This parameter :math:`N` indicates the number of nearest neighbors that should be taken into account when computing
        the centrosymmetry value for an atom. Generally, it should be a positive, even integer. Note that larger number decreases the
        calculation speed. For FCC is 12 and BCC is 8.

        .. note:: If you use this module in publication, you should also cite the original paper.
        `Kelchner C L, Plimpton S J, Hamilton J C. Dislocation nucleation and defect
        structure during surface indentation[J]. Physical review B, 1998, 58(17): 11085. <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.58.11085>`_.

        .. hint:: The CSP is calculated by the `same algorithm as LAMMPS <https://docs.lammps.org/compute_centro_atom.html>`_.
        First calculate all :math:`N (N - 1) / 2` pairs of neighbor atoms, and the summation of the :math:`N/2` lowest weights
        is CSP values.

        Args:
            N (int, optional): neighbor atom number considered, should be a positive and even number. Defaults to 12.

        Outputs:
            - **The result is added in self.data['csp']**.
        """
        use_verlet = False
        if self.if_neigh:
            if self.neighbor_number.min() >= N:
                _partition_select_sort(self.verlet_list, self.distance_list, N)
                CentroSymmetryPara = CentroSymmetryParameter(
                    N, self.pos, self.box, self.boundary, self.verlet_list
                )
                CentroSymmetryPara.compute()
                use_verlet = True

        if not use_verlet:
            CentroSymmetryPara = CentroSymmetryParameter(
                N, self.pos, self.box, self.boundary
            )
            CentroSymmetryPara.compute()

        self.__data = self.__data.with_columns(
            pl.lit(CentroSymmetryPara.csp).alias("csp")
        )

    def cal_polyhedral_template_matching(
        self,
        structure="fcc-hcp-bcc",
        rmsd_threshold=0.1,
        return_rmsd=False,
        return_atomic_distance=False,
        return_wxyz=False,
    ):
        """This function identifies the local structural environment of particles using the Polyhedral Template Matching (PTM) method, which shows greater reliability than e.g. `Common Neighbor Analysis (CNA) <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.common_neighbor_analysis>`_. It can identify the following structure:

        1. other = 0
        2. fcc = 1
        3. hcp = 2
        4. bcc = 3
        5. ico (icosahedral) = 4
        6. sc (simple cubic) = 5
        7. dcub (diamond cubic) = 6
        8. dhex (diamond hexagonal) = 7
        9. graphene = 8

        .. hint:: If you use this class in publication, you should cite the original papar:

          `Larsen P M, Schmidt S, Schiøtz J. Robust structural identification via polyhedral template matching[J]. Modelling and Simulation in Materials Science and Engineering, 2016, 24(5): 055007. <10.1088/0965-0393/24/5/055007>`_

        .. note:: The present version is translated from that in `LAMMPS <https://docs.lammps.org/compute_ptm_atom.html>`_ and only can run serially, we will try to make it parallel.


        Args:
            structure (str, optional): the structure one want to identify, one can choose from ["fcc","hcp","bcc","ico","sc","dcub","dhex","graphene","all","default"], such as 'fcc-hcp-bcc'. 'default' represents 'fcc-hcp-bcc-ico'. Defaults to 'fcc-hcp-bcc'.
            rmsd_threshold (float, optional): rmsd threshold. Defaults to 0.1.
            return_rmsd (bool, optional): whether return rmsd. Defaults to False.
            return_atomic_distance (bool, optional): whether return interatomic distance. Defaults to False.
            return_wxyz (bool, optional): whether return local structure orientation. Defaults to False.

        Outputs:
            - **The result is added in self.data['ptm']**.
        """
        verlet_list = None
        if self.if_neigh:
            if self.neighbor_number.min() >= 18:
                _partition_select_sort(self.verlet_list, self.distance_list, 18)
                verlet_list = self.verlet_list

        ptm = PolyhedralTemplateMatching(
            self.pos,
            self.box,
            self.boundary,
            structure,
            rmsd_threshold,
            verlet_list,
            False,
        )
        ptm.compute()

        self.__data = self.__data.with_columns(
            pl.lit(np.array(ptm.output[:, 0], int)).alias("ptm")
        )
        if return_rmsd:
            self.__data = self.__data.with_columns(
                pl.lit(ptm.output[:, 1]).alias("rmsd")
            )
        if return_atomic_distance:
            self.__data = self.__data.with_columns(
                pl.lit(ptm.output[:, 2]).alias("interatomic_distance")
            )
        if return_wxyz:
            self.__data.hstack(
                pl.from_numpy(ptm.output[:, 3:], schema=["qw", "qx", "qy", "qz"]),
                in_place=True,
            )

    def cal_identify_SFs_TBs(
        self,
        rmsd_threshold=0.1,
    ):
        """This function is used to identify the stacking faults (SFs) and coherent twin boundaries (TBs) in FCC structure based on the `Polyhedral Template Matching (PTM) <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.polyhedral_template_matching>`_.
        It can identify the following structure:

        1. 0 = Non-hcp atoms (e.g. perfect fcc or disordered)
        2. 1 = Indeterminate hcp-like (isolated hcp-like atoms, not forming a planar defect)
        3. 2 = Intrinsic stacking fault (two adjacent hcp-like layers)
        4. 3 = Coherent twin boundary (one hcp-like layer)
        5. 4 = Multi-layer stacking fault (three or more adjacent hcp-like layers)
        6. 5 = Extrinsic stacking fault

        .. note:: This class is translated from that `implementation in Ovito <https://www.ovito.org/docs/current/reference/pipelines/modifiers/identify_fcc_planar_faults.html#modifiers-identify-fcc-planar-faults>`_ but optimized to be run parallely.
          And so-called multi-layer stacking faults maybe a combination of intrinsic stacking faults and/or twin boundary which are located on adjacent {111} plane. It can not be distiguished by the current method.


        Args:
            rmsd_threshold (float, optional): rmsd_threshold for ptm method. Defaults to 0.1.

        Outputs:
            - **The result is added in self.data['ptm']**.
            - **The result is added in self.data['fault_types']**.
        """
        verlet_list = None
        if self.if_neigh:
            if self.neighbor_number.min() >= 18:
                _partition_select_sort(self.verlet_list, self.distance_list, 18)
                verlet_list = self.verlet_list

        ptm = PolyhedralTemplateMatching(
            self.pos,
            self.box,
            self.boundary,
            "fcc-hcp-bcc",
            rmsd_threshold,
            verlet_list,
            True,
        )
        ptm.compute()
        structure_types = np.array(ptm.output[:, 0], int)
        SFTB = IdentifySFTBinFCC(structure_types, ptm.ptm_indices)
        SFTB.compute()
        self.__data = self.__data.with_columns(
            pl.lit(SFTB.structure_types).alias("ptm"),
            pl.lit(SFTB.fault_types).alias("fault_types"),
        )

    def cal_atomic_entropy(
        self,
        rc=5.0,
        sigma=0.2,
        use_local_density=False,
        compute_average=False,
        average_rc=None,
        max_neigh=80,
    ):
        """Calculate the entropy fingerprint, which is useful to distinguish
        between ordered and disordered environments, including liquid and solid-like environments,
        or glassy and crystalline-like environments. The potential application could identificate grain boundaries
        or a solid cluster emerging from the melt. One of the advantages of this parameter is that no a priori
        information of structure is required.

        This parameter for atom :math:`i` is computed using the following formula:

        .. math:: s_S^i=-2\\pi\\rho k_B \\int\\limits_0^{r_m} \\left [ g(r) \\ln g(r) - g(r) + 1 \\right ] r^2 dr,

        where :math:`r` is a distance, :math:`g(r)` is the radial distribution function,
        and :math:`\\rho` is the density of the system.
        The :math:`g(r)` computed for each atom :math:`i` can be noisy and therefore it can be smoothed using:

        .. math:: g_m^i(r) = \\frac{1}{4 \\pi \\rho r^2} \\sum\\limits_{j} \\frac{1}{\\sqrt{2 \\pi \\sigma^2}} e^{-(r-r_{ij})^2/(2\\sigma^2)},

        where the sum over :math:`j` goes through the neighbors of atom :math:`i` and :math:`\\sigma` is
        a parameter to control the smoothing. The average of the parameter over the neighbors of atom :math:`i`
        is calculated according to:

        .. math:: \\left< s_S^i \\right>  = \\frac{\\sum_j s_S^j + s_S^i}{N + 1},

        where the sum over :math:`j` goes over the neighbors of atom :math:`i` and :math:`N` is the number of neighbors.
        The average version always provides a sharper distinction between order and disorder environments.

        .. note:: If you use this module in publication, you should also cite the original paper.
        `Entropy based fingerprint for local crystalline order <https://doi.org/10.1063/1.4998408>`_

        .. note:: This class uses the `same algorithm with LAMMPS <https://docs.lammps.org/compute_entropy_atom.html>`_.

        .. tip:: Suggestions for FCC, the :math:`rc = 1.4a` and :math:`average\_rc = 0.9a` and
        for BCC, the :math:`rc = 1.8a` and :math:`average\_rc = 1.2a`, where the :math:`a`
        is the lattice constant.

        Args:
            rc (float, optional): cutoff distance. Defaults to 5.0.
            sigma (float, optional): smoothing parameter. Defaults to 0.2.
            use_local_density (bool, optional): whether use local atomic volume. Defaults to False.
            compute_average (bool, optional): whether compute the average version. Defaults to False.
            average_rc (_type_, optional): cutoff distance for averaging operation, if not given, it is equal to rc. This parameter should be lower than rc.
            max_neigh (int, optional): maximum number of atom neighbor number. Defaults to 80.

        Outputs:
            - **The entropy is added in self.data['atomic_entropy']**.
            - **The averaged entropy is added in self.data['ave_atomic_entropy']**.
        """

        repeat = _check_repeat_cutoff(self.box, self.boundary, rc, 4)
        verlet_list, distance_list = None, None
        if sum(repeat) == 3:
            if not self.if_neigh:
                self.build_neighbor(rc, max_neigh=max_neigh)
            if self.rc < rc:
                self.build_neighbor(rc, max_neigh=max_neigh)
            verlet_list, distance_list = (
                self.verlet_list,
                self.distance_list,
            )

        if average_rc is not None:
            assert average_rc <= rc, "Average rc should not be larger than rc!"

        AtomicEntro = AtomicEntropy(
            self.vol,
            verlet_list,
            distance_list,
            self.pos,
            self.box,
            self.boundary,
            rc,
            sigma,
            use_local_density,
            compute_average,
            average_rc,
        )
        AtomicEntro.compute()

        self.__data = self.__data.with_columns(
            pl.lit(AtomicEntro.entropy).alias("atomic_entropy")
        )
        if compute_average:
            self.__data = self.__data.with_columns(
                pl.lit(AtomicEntro.entropy_average).alias("ave_atomic_entropy")
            )

    def cal_pair_distribution(self, rc=5.0, nbin=200, max_neigh=80):
        """Calculate the radiul distribution function (RDF),which
        reflects the probability of finding an atom at distance r. The seperate pair-wise
        combinations of particle types can also be computed:

        .. math:: g(r) = c_{\\alpha}^2 g_{\\alpha \\alpha}(r) + 2 c_{\\alpha} c_{\\beta} g_{\\alpha \\beta}(r) + c_{\\beta}^2 g_{\\beta \\beta}(r),

        where :math:`c_{\\alpha}` and :math:`c_{\\beta}` denote the concentration of two atom types in system
        and :math:`g_{\\alpha \\beta}(r)=g_{\\beta \\alpha}(r)`.

        Args:
            rc (float, optional): cutoff distance. Defaults to 5.0.
            nbin (int, optional): number of bins. Defaults to 200.
            max_neigh (int, optional): maximum number of atom neighbor number. Defaults to 80.

        Outputs:
            - **The result adds a PairDistribution (mdapy.PairDistribution) class**

        .. tip:: One can check the results by:

          >>> system.PairDistribution.plot() # Plot gr.

          >>> system.PairDistribution.g # Check partial RDF.

          >>> system.PairDistribution.g_total # Check global RDF.
        """

        repeat = _check_repeat_cutoff(self.box, self.boundary, rc)
        verlet_list, distance_list, neighbor_number = None, None, None
        if sum(repeat) == 3:
            if not self.if_neigh:
                self.build_neighbor(rc, max_neigh=max_neigh)
            if self.rc < rc:
                self.build_neighbor(rc, max_neigh=max_neigh)
            verlet_list, distance_list, neighbor_number = (
                self.verlet_list,
                self.distance_list,
                self.neighbor_number,
            )

        self.PairDistribution = PairDistribution(
            rc,
            nbin,
            self.box,
            self.boundary,
            verlet_list,
            distance_list,
            neighbor_number,
            self.pos,
            self.__data["type"].to_numpy(),
        )
        self.PairDistribution.compute()

    def cal_cluster_analysis(self, rc=5.0, max_neigh=80):
        """Divide atoms connected within a given cutoff distance into a cluster.
        It is helpful to recognize the reaction products or fragments under shock loading.

        .. note:: This class use the `same method as in Ovito <https://www.ovito.org/docs/current/reference/pipelines/modifiers/cluster_analysis.html#particles-modifiers-cluster-analysis>`_.

        Args:
            rc (float | dict): cutoff distance. One can also assign multi cutoff for different elemental pair, such as {'1-1':1.5, '1-6':1.7}. The unassigned elemental pair will default use the maximum cutoff distance. Defaults to 5.0.
            max_neigh (int, optional): maximum number of atom neighbor number. Defaults to 80.

        Returns:
            int: cluster number.

        Outputs:
            - **The cluster id per atom is added in self.data['cluster_id']**
        """
        if isinstance(rc, float) or isinstance(rc, int):
            max_rc = rc
        elif isinstance(rc, dict):
            max_rc = max([i for i in rc.values()])
        else:
            raise "rc should be a positive number, or a dict like {'1-1':1.5, '1-2':1.3}"
        repeat = _check_repeat_cutoff(self.box, self.boundary, max_rc)
        verlet_list, distance_list, neighbor_number = None, None, None
        if sum(repeat) == 3:
            if not self.if_neigh:
                self.build_neighbor(max_rc, max_neigh=max_neigh)
            if self.rc < max_rc:
                self.build_neighbor(max_rc, max_neigh=max_neigh)
            verlet_list, distance_list, neighbor_number = (
                self.verlet_list,
                self.distance_list,
                self.neighbor_number,
            )

        ClusterAnalysi = ClusterAnalysis(
            rc,
            verlet_list,
            distance_list,
            neighbor_number,
            self.pos,
            self.box,
            self.boundary,
            self.__data["type"].to_numpy(),
        )
        ClusterAnalysi.compute()
        self.__data = self.__data.with_columns(
            pl.lit(ClusterAnalysi.particleClusters).alias("cluster_id")
        )
        return ClusterAnalysi.cluster_number

    def cal_common_neighbor_analysis(self, rc=None, max_neigh=30):
        """Use Common Neighbor Analysis (CNA) method to recgonize the lattice structure, based
        on which atoms can be divided into FCC, BCC, HCP and Other structure.

        .. note:: If one use this module in publication, one should also cite the original paper.
          `Stukowski, A. (2012). Structure identification methods for atomistic simulations of crystalline materials.
          Modelling and Simulation in Materials Science and Engineering, 20(4), 045021. <https://doi.org/10.1088/0965-0393/20/4/045021>`_.

        .. hint:: We use the `same algorithm as in OVITO <https://www.ovito.org/docs/current/reference/pipelines/modifiers/common_neighbor_analysis.html#particles-modifiers-common-neighbor-analysis>`_.

        CNA method is sensitive to the given cutoff distance. The suggesting cutoff can be obtained from the
        following formulas:

        .. math::

            r_{c}^{\mathrm{fcc}} = \\frac{1}{2} \\left(\\frac{\\sqrt{2}}{2} + 1\\right) a
            \\approx 0.8536 a,

        .. math::

            r_{c}^{\mathrm{bcc}} = \\frac{1}{2}(\\sqrt{2} + 1) a
            \\approx 1.207 a,

        .. math::

            r_{c}^{\mathrm{hcp}} = \\frac{1}{2}\\left(1+\\sqrt{\\frac{4+2x^{2}}{3}}\\right) a,

        where :math:`a` is the lattice constant and :math:`x=(c/a)/1.633` and 1.633 is the ideal ratio of :math:`c/a`
        in HCP structure.

        Prof. Alexander Stukowski has improved this method using adaptive cutoff distances based on the atomic neighbor environment, which is the default method
        in mdapy from version 0.11.1.

        The CNA method can recgonize the following structure:

        1. Other
        2. FCC
        3. HCP
        4. BCC
        5. ICO

        Args:
            rc (float, optional): cutoff distance, if not given, will using adaptive cutoff. Defaults to None.
            max_neigh (int, optional): maximum number of atom neighbor number. Defaults to 30.

        Outputs:
            - **The CNA pattern per atom is added in self.data['cna']**.
        """
        if rc is not None:
            repeat = _check_repeat_cutoff(self.box, self.boundary, rc, 4)

            if sum(repeat) == 3 and not self.if_neigh:
                self.build_neighbor(rc, max_neigh=max_neigh)
                CommonNeighborAnalysi = CommonNeighborAnalysis(
                    self.pos,
                    self.box,
                    self.boundary,
                    rc,
                    self.verlet_list,
                    self.neighbor_number,
                )
            else:
                CommonNeighborAnalysi = CommonNeighborAnalysis(
                    self.pos, self.box, self.boundary, rc
                )
            CommonNeighborAnalysi.compute()
        else:
            sort_neigh = False
            if self.if_neigh:
                if self.neighbor_number.min() >= 14:
                    _partition_select_sort(self.verlet_list, self.distance_list, 14)
                    sort_neigh = True
            if sort_neigh:
                CommonNeighborAnalysi = CommonNeighborAnalysis(
                    self.pos, self.box, self.boundary, rc, self.verlet_list
                )
            else:
                CommonNeighborAnalysi = CommonNeighborAnalysis(
                    self.pos,
                    self.box,
                    self.boundary,
                    rc,
                )
            CommonNeighborAnalysi.compute()

        self.__data = self.__data.with_columns(
            pl.lit(CommonNeighborAnalysi.pattern).alias("cna")
        )

    def cal_common_neighbor_parameter(self, rc=3.0, max_neigh=30):
        """Use Common Neighbor Parameter (CNP) method to recgonize the lattice structure.

        .. note:: If one use this module in publication, one should also cite the original paper.
          `Tsuzuki H, Branicio P S, Rino J P. Structural characterization of deformed crystals by analysis of common atomic neighborhood[J].
          Computer physics communications, 2007, 177(6): 518-523. <https://doi.org/10.1016/j.cpc.2007.05.018>`_.

        CNP method is sensitive to the given cutoff distance. The suggesting cutoff can be obtained from the
        following formulas:

        .. math::

            r_{c}^{\mathrm{fcc}} = \\frac{1}{2} \\left(\\frac{\\sqrt{2}}{2} + 1\\right) a
            \\approx 0.8536 a,

        .. math::

            r_{c}^{\mathrm{bcc}} = \\frac{1}{2}(\\sqrt{2} + 1) a
            \\approx 1.207 a,

        .. math::

            r_{c}^{\mathrm{hcp}} = \\frac{1}{2}\\left(1+\\sqrt{\\frac{4+2x^{2}}{3}}\\right) a,

        where :math:`a` is the lattice constant and :math:`x=(c/a)/1.633` and 1.633 is the ideal ratio of :math:`c/a`
        in HCP structure.

        Some typical CNP values:

        - FCC : 0.0
        - BCC : 0.0
        - HCP : 4.4
        - FCC (111) surface : 13.0
        - FCC (100) surface : 26.5
        - FCC dislocation core : 11.
        - Isolated atom : 1000. (manually assigned by mdapy)

        Args:
            rc (float, optional): cutoff distance. Defaults to 3.0.
            max_neigh (int, optional): maximum number of atom neighbor number. Defaults to 30.

        Outputs:
            - **The CNP pattern per atom is added in self.data['cnp']**.
        """
        repeat = _check_repeat_cutoff(self.box, self.boundary, rc)

        if sum(repeat) == 3:
            if not self.if_neigh:
                self.build_neighbor(rc, max_neigh=max_neigh)
            if self.rc < rc:
                self.build_neighbor(rc, max_neigh=max_neigh)
            CommonNeighborPar = CommonNeighborParameter(
                self.pos,
                self.box,
                self.boundary,
                rc,
                self.verlet_list,
                self.distance_list,
                self.neighbor_number,
            )
        else:
            CommonNeighborPar = CommonNeighborParameter(
                self.pos,
                self.box,
                self.boundary,
                rc,
            )
        CommonNeighborPar.compute()

        self.__data = self.__data.with_columns(
            pl.lit(CommonNeighborPar.cnp).alias("cnp")
        )

    def cal_identify_diamond_structure(
        self,
    ):
        """This class is used to identify the Diamond structure. The results and algorithm should be the same in Ovito.
        More details can be found in https://www.ovito.org/manual/reference/pipelines/modifiers/identify_diamond.html .

        Outputs:
            - **The pattern per atom is added in self.data['ids']**.

        The identified structures include:

        - 0 "other",
        - 1 "cubic_diamond",
        - 2 "cubic_diamond_1st_neighbor",
        - 3 "cubic_diamond_2st_neighbor",
        - 4 "hexagonal_diamond",
        - 5 "hexagonal_diamond_1st_neighbor",
        - 6 "hexagonal_diamond_2st_neighbor"

        """
        sort_neigh = False
        if self.if_neigh:
            if self.neighbor_number.min() >= 12:
                _partition_select_sort(self.verlet_list, self.distance_list, 12)
                sort_neigh = True
        if sort_neigh:
            IDS = IdentifyDiamondStructure(
                self.pos, self.box, self.boundary, self.verlet_list
            )
        else:
            IDS = IdentifyDiamondStructure(
                self.pos,
                self.box,
                self.boundary,
            )
        IDS.compute()
        self.__data = self.__data.with_columns(pl.lit(IDS.pattern).alias("ids"))

    def cal_energy_force_virial(self, potential, elements_list, centroid_stress=False):
        """Calculate the atomic energy and force based on the given potential.

        Args:
            potential (BasePotential): base potential class defined in mdapy, which must including a compute method to calculate the energy, force, virial.
            elements_list (list): elements to be calculated, such as ['Al', 'Ni'] indicates setting type 1 as 'Al' and type 2 as 'Ni'.
            centroid_stress (bool, optional): Only for LammpsPotential. If Ture, use compute stress/atomm. If False, use compute centroid/stress/atom. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: energy, force, virial.

        Units & Shape (Only for our originally supported EAM and NEP potential.):
            energy : eV (:math:`N_p`)
            force : eV/A (:math:`N_p, 3`). The order is fx, fy and fz.
            virial : eV*A^3 (:math:`N_p, 9`). The order is xx, yy, zz, xy, xz, yz, yx, zx, zy.
        """

        if isinstance(potential, EAM):
            repeat = _check_repeat_cutoff(self.box, self.boundary, potential.rc, 5)
            verlet_list, distance_list, neighbor_number = None, None, None
            if sum(repeat) == 3:
                if not self.if_neigh:
                    self.build_neighbor(rc=potential.rc)
                if self.rc < potential.rc:
                    self.build_neighbor(rc=potential.rc)
                verlet_list, distance_list, neighbor_number = (
                    self.verlet_list,
                    self.distance_list,
                    self.neighbor_number,
                )

            energy, force, virial = potential.compute(
                self.pos,
                self.box,
                elements_list,
                self.__data["type"].to_numpy(),
                self.boundary,
                verlet_list,
                distance_list,
                neighbor_number,
            )

        elif isinstance(potential, NEP):
            energy, force, virial = potential.compute(
                self.pos,
                self.box,
                elements_list,
                self.__data["type"].to_numpy(),
                self.boundary,
            )
        else:
            energy, force, virial = potential.compute(
                self.pos,
                self.box,
                elements_list,
                self.__data["type"].to_numpy(),
                self.boundary,
                centroid_stress,
            )

        return energy, force, virial

    def cal_void_distribution(self, cell_length, out_void=False, out_name="void.dump"):
        """This class is used to detect the void distribution in solid structure.
        First we divid particles into three-dimensional grid and check the its
        neighbors, if all neighbor grid is empty we treat this grid is void, otherwise it is
        a point defect. Then useing clustering all connected 'void' grid into an entire void.
        Excepting counting the number and volume of voids, this class can also output the
        spatial coordination of void for analyzing the void distribution.

        .. note:: The results are sensitive to the selection of :math:`cell\_length`, which should
          be illustrated if this class is used in your publication.

        Args:
            cell_length (float): length of cell, larger than lattice constant is okay.
            out_void (bool, optional): whether outputs void coordination. Defaults to False.
            out_name (str, optional): output filename. Defaults to void.dump.

        Returns:
            tuple: (void_number, void_volume), number and volume of voids.
        """

        void = VoidDistribution(
            self.pos,
            self.box,
            cell_length,
            self.boundary,
            out_void=out_void,
            out_name=out_name,
        )
        void.compute()

        return void.void_number, void.void_volume

    def cal_warren_cowley_parameter(self, rc=3.0, max_neigh=50):
        """This class is used to calculate the Warren Cowley parameter (WCP), which is useful to
        analyze the short-range order (SRO) in the 1st-nearest neighbor shell in alloy system and is given by:

        .. math:: WCP_{mn} = 1 - Z_{mn}/(X_n Z_{m}),

        where :math:`Z_{mn}` is the number of :math:`n`-type atoms around :math:`m`-type atoms,
        :math:`Z_m` is the total number of atoms around :math:`m`-type atoms, and :math:`X_n` is
        the atomic concentration of :math:`n`-type atoms in the alloy.

        .. note:: If you use this class in publication, you should also cite the original paper:
          `X‐Ray Measurement of Order in Single Crystals of Cu3Au <https://doi.org/10.1063/1.1699415>`_.

        Args:
            rc (float, optional): cutoff distance. Defaults to 3.0.
            max_neigh (int, optional): maximum number of atom neighbor number. Defaults to 50.

        Outputs:
            - **The result adds a WarrenCowleyParameter (mdapy.WarrenCowleyParameter) class**

        .. tip:: One can check the results by:

          >>> system.WarrenCowleyParameter.plot() # Plot WCP.

          >>> system.WarrenCowleyParameter.WCP # Check WCP.
        """
        repeat = _check_repeat_cutoff(self.box, self.boundary, rc, 4)
        verlet_list, neighbor_number = None, None
        if sum(repeat) == 3:
            if not self.if_neigh:
                self.build_neighbor(rc=rc, max_neigh=max_neigh)
                verlet_list, neighbor_number = (
                    self.verlet_list,
                    self.neighbor_number,
                )
            if self.rc != rc:
                neigh = Neighbor(self.pos, self.box, rc, self.boundary, max_neigh)
                neigh.compute()
                verlet_list, neighbor_number = (
                    neigh.verlet_list,
                    neigh.neighbor_number,
                )

        self.WarrenCowleyParameter = WarrenCowleyParameter(
            self.__data["type"].to_numpy(),
            verlet_list,
            neighbor_number,
            rc,
            self.pos,
            self.box,
            self.boundary,
        )

        self.WarrenCowleyParameter.compute()

    def cal_voronoi_volume(self, num_t=None):
        """This class is used to calculate the Voronoi polygon, which can be applied to
        estimate the atomic volume. The calculation is conducted by the `voro++ <https://math.lbl.gov/voro++/>`_ package and
        this class only provides a wrapper. From mdapy v0.8.6, we use extended parallel voro++ to improve the performance, the
        implementation can be found in `An extension to VORO++ for multithreaded computation of Voronoi cells <https://arxiv.org/abs/2209.11606>`_.

        Args:
            num_t (int, optional): threads number to generate Voronoi diagram. If not given, use all avilable threads.

        Outputs:
            - **The atomic Voronoi volume is added in self.data['voronoi_volume']**.
            - **The atomic Voronoi neighbor is added in self.data["voronoi_number"]**.
            - **The atomic Voronoi cavity radius is added in self.data["cavity_radius"]**.

        """
        if num_t is None:
            num_t = mt.cpu_count()
        else:
            assert num_t >= 1, "num_t should be a positive integer!"
            num_t = int(num_t)

        voro = VoronoiAnalysis(self.pos, self.box, self.boundary, num_t)
        voro.compute()

        self.__data = self.__data.with_columns(
            pl.lit(voro.vol).alias("voronoi_volume"),
            pl.lit(voro.neighbor_number).alias("voronoi_number"),
            pl.lit(voro.cavity_radius).alias("cavity_radius"),
        )

    def spatial_binning(self, direction, vbin, wbin=5.0, operation="mean"):
        """This class is used to divide particles into different bins and operating on each bin.
        One-dimensional to Three-dimensional binning are supported.

        Args:
            direction (str): binning direction, selected in ['x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'].
            vbin (str/list): values to be operated, such as 'x' or ['vx', 'pe'].
            wbin (float, optional): width of each bin. Defaults to 5.0.
            operation (str, optional): operation on each bin, selected from ['mean', 'sum', 'min', 'max']. Defaults to "mean".

        Outputs:
            - **The result adds a Binning (mdapy.SpatialBinning) class**

        .. tip:: One can check the results by:

          >>> system.Binning.plot() # Plot the binning results.

          >>> system.Binning.res # Check the binning results.

          >>> system.Binning.coor # Check the binning coordination.
        """

        vbin = np.ascontiguousarray(self.__data.select(vbin))
        self.Binning = SpatialBinning(self.pos, direction, vbin, wbin, operation)
        self.Binning.compute()

    def orthogonal_box(self, N=10):
        """This function try to change the box to rectangular.

        Args:
            N (int, optional): search limit. If you can't found rectangular box, increase N. Defaults to 10.

        Returns:
            System: a new system with reactangular box. The atoms number may be changed.
        """
        rec = OrthogonalBox(self.pos, self.box, self.data["type"].to_numpy())
        rec.compute(N)
        if "type_name" in self.data.columns:
            df = self.data.group_by("type").agg(pl.col("type_name"))
            type_dict = {str(df[i, 0]): df[i, 1][0] for i in range(df.shape[0])}
            system = System(
                pos=rec.rec_pos,
                box=rec.rec_box,
                type_list=rec.rec_type_list,
            )
            system.update_data(
                system.data.with_columns(
                    pl.col("type")
                    .cast(pl.Utf8)
                    .replace_strict(type_dict)
                    .alias("type_name")
                )
            )
            return system
        else:
            return System(
                pos=rec.rec_pos,
                box=rec.rec_box,
                type_list=rec.rec_type_list,
            )

    def cal_bond_analysis(self, rc, nbins=100, max_neigh=None):
        """This function calculates the distribution of bond length and angle based on a given cutoff distance.

        Args:
            rc (float): cutoff distance.
            nbins (int, optional): number of bins. Defaults to 100.
            max_neigh (int, optional): maximum number of atom neighbor number. Defaults to None.

        Outputs:
            - **The result adds a BA (mdapy.BondAnalysis) class.**

        .. tip:: One can check the results by:

          >>> system.BA.plot_bond_length_distribution() # Plot the bond length distribution.

          >>> system.BA.plot_bond_angle_distribution() # Plot the bond angle distribution.

          >>> system.BA.bond_length_distribution # Check the data.

          >>> system.BA.bond_angle_distribution # Check the data.
        """
        repeat = _check_repeat_cutoff(self.box, self.boundary, rc)

        if sum(repeat) == 3:
            if not self.if_neigh:
                self.build_neighbor(rc, max_neigh=max_neigh)
            if self.rc < rc:
                self.build_neighbor(rc, max_neigh=max_neigh)
            self.BA = BondAnalysis(
                self.pos,
                self.box,
                self.boundary,
                rc,
                nbins,
                self.verlet_list,
                self.distance_list,
                self.neighbor_number,
            )
        else:
            self.BA = BondAnalysis(self.pos, self.box, self.boundary, rc, nbins)
        self.BA.compute()


class MultiSystem(list):
    """
    This class is a collection of mdapy.System class and is helful to handle the atomic trajectory.

    Args:
        filename_list (list): ordered filename list, such as ['melt.0.dump', 'melt.100.dump'].
        unwrap (bool, optional): make atom positions do not wrap into box due to periotic boundary. Using minimum image criterion, see https://en.wikipedia.org/wiki/Periodic_boundary_conditions#Practical_implementation:_continuity_and_the_minimum_image_convention. Defaults to True.
        sorted_id (bool, optional): sort data by atoms id. Defaults to True.

    Outputs:
        - **pos_list** (np.ndarray): (:math:`N_f, N_p, 3`), :math:`N_f` frames particle position.
        - **Nframes** (int): number of frames.

    """

    def __init__(self, filename_list, unwrap=True, sorted_id=True):
        self.sorted_id = sorted_id
        self.unwrap = unwrap
        progress_bar = tqdm(filename_list)
        for filename in progress_bar:
            progress_bar.set_description(f"Reading {filename}")
            system = System(filename, sorted_id=self.sorted_id)
            self.append(system)

        pos_list, box_list, inverse_box_list = [], [], []
        for system in self:
            pos_list.append(system.pos)
            box_list.append(system.box)
            inverse_box_list.append(np.linalg.inv(system.box[:-1]))

        self.pos_list = np.array(pos_list)
        box_list = np.array(box_list)
        inverse_box_list = np.array(inverse_box_list)

        if self.unwrap:
            _unwrap_pos(self.pos_list, box_list, inverse_box_list, self[0].boundary)

            for i, system in enumerate(self):
                newdata = system.data.with_columns(
                    pl.lit(self.pos_list[i, :, 0]).alias("x"),
                    pl.lit(self.pos_list[i, :, 1]).alias("y"),
                    pl.lit(self.pos_list[i, :, 2]).alias("z"),
                )
                system.update_data(newdata, update_pos=True)

        self.Nframes = self.pos_list.shape[0]

    def write_dumps(self, output_col=None, compress=False):
        """Write all data to a series of DUMP files.

        Args:
            output_col (list, optional): columns to be saved, such as ['id', 'type', 'x', 'y', 'z'].
            compress (bool, optional): whether compress the DUMP file.
        """

        progress_bar = tqdm(self)
        for system in progress_bar:
            progress_bar.set_description(f"Saving {system.filename}")
            system.write_dump(output_col=output_col, compress=compress)

    def cal_mean_squared_displacement(self, mode="windows"):
        """Calculate the mean squared displacement MSD of system, which can be used to
        reflect the particle diffusion trend and describe the melting process. Generally speaking, MSD is an
        average displacement over all windows of length :math:`m` over the course of the simulation (so-called
        'windows' mode here) and defined by:

        .. math:: MSD(m) = \\frac{1}{N_{p}} \\sum_{i=1}^{N_{p}} \\frac{1}{N_{f}-m} \\sum_{k=0}^{N_{f}-m-1} (\\vec{r}_i(k+m) - \\vec{r}_i(k))^2,

        where :math:`r_i(t)` is the position of particle :math:`i` in frame :math:`t`. It is computationally extensive
        while using a fast Fourier transform can remarkably reduce the computation cost as described in `nMoldyn - Interfacing
        spectroscopic experiments, molecular dynamics simulations and models for time correlation functions
        <https://doi.org/10.1051/sfn/201112010>`_ and discussion in `StackOverflow <https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft>`_.

        .. note:: One can install `pyfftw <https://github.com/pyFFTW/pyFFTW>`_ to accelerate the calculation,
          otherwise mdapy will use `scipy.fft <https://docs.scipy.org/doc/scipy/reference/fft.html#module-scipy.fft>`_
          to do the Fourier transform.

        Sometimes one only need the following atomic displacement (so-called 'direct' mode here):

        .. math:: MSD(t) = \\dfrac{1}{N_p} \\sum_{i=1}^{N_p} (r_i(t) - r_i(0))^2.

        Args:
            mode (str, optional): 'windows' or 'direct'. Defaults to "windows".

        Outputs:
            - **The result adds a MSD (mdapy.MeanSquaredDisplacement) class**

        .. tip:: One can check the results by:

          >>> MS = mp.MultiSystem(filename_list) # Generate a MultiSystem class.

          >>> MS.cal_mean_squared_displacement() # Calculate MSD.

          >>> MS.MSD.plot() # Plot the MSD results.

          >>> MS[0].data['msd'] # Check MSD per particles.
        """

        self.MSD = MeanSquaredDisplacement(self.pos_list, mode=mode)
        self.MSD.compute()
        for frame in range(self.Nframes):
            self[frame].data.hstack(
                pl.from_numpy(self.MSD.particle_msd[frame], schema=["msd"]),
                in_place=True,
            )

    def cal_lindemann_parameter(self, only_global=False):
        """
        Calculate the `Lindemann index <https://en.wikipedia.org/wiki/Lindemann_index>`_,
        which is useful to distinguish the melt process and determine the melting points of nano-particles.
        The Lindemann index is defined as the root-mean-square bond-length fluctuation with following mathematical expression:

        .. math:: \\left\\langle\\sigma_{i}\\right\\rangle=\\frac{1}{N_{p}(N_{p}-1)} \\sum_{j \\neq i} \\frac{\\sqrt{\\left\\langle r_{i j}^{2}\\right\\rangle_t-\\left\\langle r_{i j}\\right\\rangle_t^{2}}}{\\left\\langle r_{i j}\\right\\rangle_t},

        where :math:`N_p` is the particle number, :math:`r_{ij}` is the distance between atom :math:`i` and :math:`j` and brackets :math:`\\left\\langle \\right\\rangle_t`
        represents an time average.

        .. note:: This class is partly referred to a `work <https://github.com/N720720/lindemann>`_ on calculating the Lindemann index.

        .. note:: This calculation is high memory requirement. One can estimate the memory by: :math:`2 * 8 * N_p^2 / 1024^3` GB.

        .. tip:: If only global lindemann index is needed, the class can be calculated in parallel.
          The local Lindemann index only run serially due to the dependencies between different frames.
          Here we use the `Welford method <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford>`_ to
          update the varience and mean of :math:`r_{ij}`.

        Args:
            only_global (bool, optional): whether only calculate the global index. Defaults to False.

        Outputs:
            - **The result adds a Lindemann (mdapy.LindemannParameter) class**.

        .. tip:: One can check the results by:

          >>> MS = mp.MultiSystem(filename_list) # Generate a MultiSystem class.

          >>> MS.cal_lindemann_parameter() # Calculate Lindemann index.

          >>> MS.Lindemann.plot() # Plot the Lindemann index.

          >>> MS[0].data['lindemann'] # Check Lindemann index per particles.

        """

        self.Lindemann = LindemannParameter(self.pos_list, only_global)
        self.Lindemann.compute()
        try:
            for frame in range(self.Nframes):
                self[frame].data.hstack(
                    pl.from_numpy(
                        self.Lindemann.lindemann_atom[frame], schema=["lindemann"]
                    ),
                    in_place=True,
                )
        except Exception:
            pass


if __name__ == "__main__":
    # system = System(r"ClO3.all.xyz")
    # print(system)

    # system.write_data('test.data')
    # system.write_dump('test.dump')
    # #system.write_POSCAR('test.poscar')
    # # system.write_cif('test-2.cif')
    # #system = System('test.poscar')
    # system = System('test.data')
    # print(system)
    # system = System('test.dump')
    # print(system)
    # system.write_xyz('test.xyz')
    # system.global_info['energy'] = 2.
    # system.global_info['Logo'] = 'Write by mdapy'
    # system.write_xyz('test_1.xyz')
    import taichi as ti
    import polars as pl
    import numpy as np
    from lattice_maker import LatticeMaker

    ti.init()
    # nep = NEP(r"D:\Study\Gra-Al\potential_test\validating\graphene\itre_45\nep.txt")
    # element_name, lattice_constant, lattice_type, potential = (
    #     "Al",
    #     4.1,
    #     "FCC",
    #     nep,
    # )
    # x, y, z = 5, 5, 5
    # fmax = 1e-5
    # max_itre = 200
    # lat = LatticeMaker(lattice_constant, lattice_type, x, y, z)
    # lat.compute()
    # noise = np.random.random((lat.N, 3))
    # system = System(pos=lat.pos + noise, box=lat.box)
    # relax_system = system.minimize(
    #     [element_name], potential, volume_change=True, hydrostatic_strain=True
    # )
    # print(relax_system)
    # relax_system.write_xyz("Al.xyz", type_name=["Al"])

    # lat = LatticeMaker(3.615, "FCC", 5, 5, 5)
    # lat.compute()
    import freud

    box, points = freud.data.make_random_system(30, 2000, seed=0)
    # print(box.to_matrix())
    # box.periodic_x = False
    neigh = freud.locality.Voronoi()
    neigh.compute((box, points))
    print(box.periodic)
    # neigh1 = freud.locality.AABBQuery(box, points)
    # nlist = neigh1.query(
    #     points, {"num_neighbors": 12, "exclude_ii": True}
    # ).toNeighborList()
    ql = freud.order.Steinhardt(l=6, average=True, weighted=True)
    ql.compute(
        (box, points), neigh.nlist
    )  # neigh.nlist) #{"num_neighbors": 6, "exclude_ii": True}

    print(ql.ql[:15])
    # print(box, points)
    system = System(box=box.Lx, pos=points.astype(np.float64))
    system.wrap_pos()
    # system.boundary[0] = 0
    # system.build_neighbor_voronoi()
    # print(system.voro_verlet_list[0])
    # print(neigh.nlist[neigh.nlist[:, 0] == 0])
    # system.build_neighbor(3.1, max_neigh=23)
    system.cal_steinhardt_bond_orientation(
        use_voronoi=True, average=True, use_weight=True
    )  # use_voronoi=True, use_weight=True)
    print(system.data["ql6"].to_numpy()[:15])
    x, y = ql.ql, system.data["ql6"].to_numpy()
    print(np.where(x - y > 1e-3))
    # print(neigh.nlist[neigh.nlist[:, 0] == 0])
    # print(system.voro_verlet_list[0])
    # # print(neigh.nlist.distances[neigh.nlist[:, 0] == 5])
    # # print(system.voro_distance_list[5])

    # for i in neigh.nlist[neigh.nlist[:, 0] == 0]:
    #     if i[1] not in system.voro_verlet_list[0]:
    #         print(i, system.atom_distance(i[0], i[1]))

    # n = 0
    # print("i j freud_dis, mdapy_dis")
    # for i in range(system.voro_neighbor_number[n]):
    #     j = system.voro_verlet_list[n, i]
    #     if j > -1:
    #         m_dis = system.voro_distance_list[n, i]
    #         f_dis = neigh.nlist.distances[
    #             (neigh.nlist[:, 0] == n) & (neigh.nlist[:, 1] == j)
    #         ][0]
    #         print(n, j, f_dis, m_dis)
    # print(system.voro_distance_list[5])
    # print(system.voro_neighbor_number[5])
    # print(system.voro_face_areas[5])
    # print(system.neighbor_number[5])
    # print(system.distance_list[5])
    # print(points[134] - points[5])
    # print(system.atom_distance(5, 134))
    # print(system.atom_distance(5, 510))
    # print(system.pos[5])
    # print(points[5])
    # print(system.voro_verlet_list[0])
    # print(system.voro_neighbor_number[0])
    # print(system.voro_distance_list[0])
    # print(system.voro_face_areas[0])
    # np.random.seed(1)
    # noise = np.random.random((lat.N, 3))
    # system = System(pos=lat.pos + noise, box=lat.box)
    # system.cal_steinhardt_bond_orientation(use_voronoi=True, use_weight=True)
    # print(system)

    # system = System(r'example/CoCuFeNiPd-4M.data')
    # system.cal_common_neighbor_analysis(rc=3.)
    # print(system.data.group_by(pl.col('cna')).count().sort(pl.col('cna')))

    # system = System(r"C:\Users\herrwu\Desktop\xyz\HexDiamond.xyz")
    # system.cal_identify_diamond_structure()
    # print(system.data.group_by(pl.col("ids")).count().sort(pl.col("ids")))
    # system = System(r"C:\Users\herrwu\Desktop\xyz\40.lmp")
    # system.cal_bond_analysis(2.5)
    # system.BA.plot_bond_angle_distribution()
    # system.BA.plot_bond_length_distribution()
    # system = System(r"D:\Study\Diamond\NEP\res\train.00000.xyz")
    # print(system)
    # pair_parameter = """
    # pair_style nep
    # pair_coeff * * D:\\Package\\MyPackage\\lammps_nep\\example\\C_2024_NEP4.txt C
    # """
    # elements_list = ["C"]
    # relax_gra = system.cell_opt(pair_parameter, elements_list)
    # print(relax_gra)
    # system = System(r"C:\Users\herrwu\Desktop\xyz\MoS2-H.xyz")
    # rec = system.orthogonal_box(10)
    # print(rec)
    # print("Rectangular box:")
    # print(rec.box)
    # print("Rectangular pos::")
    # print(rec.pos)

    # ref = System(r'D:\Study\Gra-Al\paper\Fig6\res\al_gra_deform_1e9_x\dump.0.xyz')
    # ref.build_neighbor(5., max_neigh=70)
    # cur = System(r'D:\Study\Gra-Al\paper\Fig6\res\al_gra_deform_1e9_x\dump.1000.xyz')
    # cur.cal_atomic_strain(ref, rc=5.)
    # print(cur)

    # system = System("test.xyz")
    # system.cal_pair_distribution(8., 200)

    # system.PairDistribution.plot()
    # system.PairDistribution.plot_partial(['Al', 'Cu'])

    # filename_list = [rf'C:\Users\herrwu\Desktop\wrap_test\dump.{i}.xyz' for i in range(100, 10100, 100)]
    # MS = MultiSystem(filename_list, sorted_id=False, unwrap=True)
    # MS.cal_mean_squared_displacement()
    # MS.MSD.plot()
    # MS.write_dumps()
    # for i in range(20):
    #     print(i, 'step', MS[i].pos[0])
    # # system.write_xyz("test.xyz")
    # system.write_POSCAR()
    # system.write_cif("test.cif")
    # system = System(r"D:\Study\Gra-Al\potential_test\phonon\aluminum\min.data")
    # system.write_cp2k("cp2k", type_name=["Al"])
    # relax_system = system.cell_opt(
    #     "pair_style eam/alloy\npair_coeff * * example/Al_DFT.eam.alloy Al", ["Al"]
    # )
    # print(relax_system)

    # ti.init()
    # from lattice_maker import LatticeMaker
    # from potential import LammpsPotential

    # lat = LatticeMaker(4.05, "FCC", 1, 1, 1)
    # lat.compute()
    # system = System(pos=lat.pos, box=lat.box)
    # potential = LammpsPotential(
    #     """pair_style eam/alloy
    #    pair_coeff * * example/Al_DFT.eam.alloy Al"""
    # )
    # system.cal_phono_dispersion(
    #     "0.0 0.0 0.0 0.5 0.0 0.5 0.625 0.25 0.625 0.375 0.375 0.75 0.0 0.0 0.0 0.5 0.5 0.5",
    #     "$\Gamma$ X U K $\Gamma$ L",
    #     potential=potential,
    #     elements_list=["Al"],
    # )
    # fig, ax = system.Phon.plot_dispersion(
    #     units="1/cm", merge_kpoints=[2, 3], color=(123, 204, 33), ylim=[0, 350]
    # )
    # system = System("example/solidliquid.data")
    # print(system)
    # system.write_data("test.data", type_name=["Al", "C", "Fe"])
    # system.cal_identify_SFs_TBs()
    # system.write_dump("test.dump")
    # system = System('E:/Al+SiC/compress/20/2km/shock.10000.dump')
    # system = System(r"E:/VORO_GRAPHENE/GPCu/GRAIN100/relax/GRA-Metal-FCC-100-1.data")
    # system.cal_polyhedral_template_matching(structure="all")
    # # print(system)
    # system.cal_polyhedral_template_matching("all")
    # print(system)
    # system.cal_voronoi_volume()
    # system = System("example/solidliquid.dump")
    # system.cal_atomic_temperature(elemental_list=['Mo'])
    # system.cal_atomic_temperature(amass=[95.94])
    # system.write_xyz()
    # system = System(r"E:\MyPackage\mdapy-tutorial\frame\Ti.data")
    # system.replicate(5, 5, 5)
    # system.cal_common_neighbor_analysis(rc=2.9357 * 1.207)
    # # print(system)
    # system.cal_ackland_jones_analysis()
    # # print(system)
    # system.cal_centro_symmetry_parameter()
    # # print(system)
    # system.cal_common_neighbor_parameter()
    # # print(system)
    # system.cal_steinhardt_bond_orientation()
    # # print(system)
    # print(system.data[:, 4:])

    # print(system.data["atomic_temp"].mean())
    # system.write_xyz(type_name=["Mo"])
    # system = System(r"D:\Package\MyPackage\mdapy-tutorial\frame\ap@al.dump")
    # species = system.cal_species_number(
    #     element_list=["H", "C", "N", "O", "F", "Al", "Cl"],
    #     search_species=["H2O", "Cl", "N2", "CO2", "HCl"],
    # )
    # print(species)
    # species = system.cal_species_number(
    #     element_list=["H", "C", "N", "O", "F", "Al", "Cl"], check_most=20
    # )
    # print(species)
    # system.write_xyz()
    # system.write_xyz(classical=True)
    # system.write_dump()
    # system.write_POSCAR()
    # system.write_data()
    # system = System(r"example\solidliquid.dump")
    # print(system)
    # system.wtite_POSCAR(output_name="POSCAR", save_velocity=True, type_name=["Mo"])
    # system = System('POSCAR')
    # print(system)
    # system = System(
    #     r"C:\Users\Administrator\Desktop\Fe\ML-DATA\VASP\examples\examples\POSCAR"
    # )
    # print(system)
    # system.wtite_POSCAR(output_name="c.POSCAR")
    # system.wtite_POSCAR(output_name="d.POSCAR", reduced_pos=True)
    # print(system)
    # system.replicate(3, 3, 3)
    # system.cal_ackland_jones_analysis()
    # system.cal_atomic_entropy()
    # print(system)
    # system.wrap_pos()
    # system.boundary[-1] = 0
    # system.write_dump()
    # system = System(r'E:\HEAShock\111\0.7km\shock-700m.10000.dump')
    # system.write_data()
    # system.write_dump()
    # system.write_dump(compress=True)
    # system.write_dump(output_name='test.dump.gz', compress=True)
    # from tool_function import _init_vel
    # system = System("./example/CoCuFeNiPd-4M.dump")
    # # print(system)
    # system.build_neighbor(max_neigh=60)
    # system.cal_pair_distribution()
    # # system.cal_warren_cowley_parameter()
    # system.cal_centro_symmetry_parameter()
    # system.cal_polyhedral_template_matching()
    # print(system)
    # system.write_data()
    # system.write_dump()
    # system.write_dump(compress=True)
    # system = System(r"./example/FCC.data")
    # # print(system)
    # # from time import time

    # # start = time()
    # # system.build_neighbor(5.0, 70)
    # # print(time() - start, "s")
    # # # system.cal_atomic_temperature(
    # # #     np.array([58.9332, 58.6934, 55.847, 26.981539, 63.546]), 5.0
    # # # )
    # # # print(system.data["atomic_temp"].mean())
    # # print(system.distance_list[0])
    # # print(system.verlet_list[0])
    # # print(system.neighbor_number.max())
    # # print(system.verlet_list.shape[1])
    # # system.replicate(4, 4, 4)
    # system.cal_energy_force("./example/CoNiFeAlCu.eam.alloy", ["Al"])
    # system.cal_ackland_jones_analysis()
    # system.cal_centro_symmetry_parameter()
    # system.cal_common_neighbor_analysis(4.05 * 0.86)
    # system.cal_common_neighbor_parameter(4.05 * 0.86)
    # system.cal_atomic_entropy(
    #     rc=4.05 * 1.4, average_rc=4.05 * 0.9, compute_average=True
    # )
    # system.cal_polyhedral_template_matching()
    # system.cal_identify_SFs_TBs()
    # print("Number of cluter:", system.cal_cluster_analysis(5.0))
    # # system.cal_pair_distribution()
    # # system.PairDistribution.plot()
    # system.cal_steinhardt_bond_orientation()
    # vel = _init_vel(system.N, 300.0, 1.0)
    # system.data[["vx", "vy", "vz"]] = vel
    # system.cal_atomic_temperature(np.array([1.0]))
    # system.cal_voronoi_volume()
    # system.cal_warren_cowley_parameter()
    # print(system.WarrenCowleyParameter.WCP)
    # print(system)
    # print(system.rc)
    # system = System(f"./benchmark/average_rdf/rdf.0.dump")
    # system.cal_pair_distribution(max_neigh=430)
    # print(system.verlet_list.shape[1])
    # print(system.neighbor_number.max())
    # system.PairDistribution.plot()
    # MS = MultiSystem([f"./benchmark/average_rdf/rdf.{i}.dump" for i in range(5)])
    # print(MS[0])
    # MS.cal_mean_squared_displacement()
    # MS.MSD.plot()
    # MS.cal_lindemann_parameter()
    # MS.Lindemann.plot()
    # MS.write_dumps()
