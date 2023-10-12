import pandas as pd
import numpy as np
import taichi as ti
import multiprocessing as mt

try:
    from load_save_data import BuildSystem, SaveFile
    from tool_function import _wrap_pos, _partition_select_sort, _unwrap_pos
    from ackland_jones_analysis import AcklandJonesAnalysis
    from common_neighbor_analysis import CommonNeighborAnalysis
    from common_neighbor_parameter import CommonNeighborParameter
    from neighbor import Neighbor
    from temperature import AtomicTemperature
    from centro_symmetry_parameter import CentroSymmetryParameter
    from entropy import AtomicEntropy
    from identify_SFs_TBs import IdentifySFTBinFCC
    from pair_distribution import PairDistribution
    from polyhedral_template_matching import PolyhedralTemplateMatching
    from cluser_analysis import ClusterAnalysis
    from potential import EAM
    from calculator import Calculator
    from void_distribution import VoidDistribution
    from warren_cowley_parameter import WarrenCowleyParameter
    from voronoi_analysis import VoronoiAnalysis
    from mean_squared_displacement import MeanSquaredDisplacement
    from lindemann_parameter import LindemannParameter
    from spatial_binning import SpatialBinning
    from steinhardt_bond_orientation import SteinhardtBondOrientation
    from replicate import Replicate
    from tool_function import _check_repeat_cutoff
except Exception:
    from .load_save_data import BuildSystem, SaveFile
    from .tool_function import _wrap_pos, _partition_select_sort, _unwrap_pos
    from .common_neighbor_analysis import CommonNeighborAnalysis
    from .ackland_jones_analysis import AcklandJonesAnalysis
    from .common_neighbor_parameter import CommonNeighborParameter
    from .neighbor import Neighbor
    from .temperature import AtomicTemperature
    from .centro_symmetry_parameter import CentroSymmetryParameter
    from .entropy import AtomicEntropy
    from .identify_SFs_TBs import IdentifySFTBinFCC
    from .pair_distribution import PairDistribution
    from .polyhedral_template_matching import PolyhedralTemplateMatching
    from .cluser_analysis import ClusterAnalysis
    from .potential import EAM
    from .calculator import Calculator
    from .void_distribution import VoidDistribution
    from .warren_cowley_parameter import WarrenCowleyParameter
    from .voronoi_analysis import VoronoiAnalysis
    from .mean_squared_displacement import MeanSquaredDisplacement
    from .lindemann_parameter import LindemannParameter
    from .spatial_binning import SpatialBinning
    from .steinhardt_bond_orientation import SteinhardtBondOrientation
    from .replicate import Replicate
    from .tool_function import _check_repeat_cutoff


class System:
    """This class can generate a System class for rapidly accessing almost all the analysis
    method in mdapy.

    .. note::
      - mdapy now supports both rectangle and triclinic box from version 0.9.0.
      - mdapy only supports the simplest DATA format, atomic and charge, which means like bond information will cause an error.
      - We recommend you use DUMP as input file format or directly give particle positions and box.

    Args:
        filename (str, optional): DATA/DUMP filename. Defaults to None.
        fmt (str, optional): selected in ['data', 'lmp', 'dump', 'dump.gz'], One can explicitly assign the file format or mdapy will handle it with the postsuffix of filename. Defaults to None.
        data (pd.Dataframe, optional): all particles information. Defaults to None.
        box (np.ndarray, optional): (:math:`4, 3` or :math:`3, 2`) system box. Defaults to None.
        pos (np.ndarray, optional): (:math:`N_p, 3`) particles positions. Defaults to None.
        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].
        vel (np.ndarray, optional): (:math:`N_p, 3`) particles velocities. Defaults to None.
        type_list (np.ndarray, optional): (:math:`N_p`) type per particles. Defaults to 1.
        sorted_id (bool, optional): whether sort system data by the particle id. Defaults to False.

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
        sorted_id=False,
    ) -> None:
        self.__filename = filename
        self.__fmt = fmt
        self.__timestep = 0
        if (
            isinstance(data, pd.DataFrame)
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
            else:
                self.__data, self.__box, self.__boundary = BuildSystem.fromfile(
                    self.__filename, self.__fmt
                )
        elif (
            isinstance(pos, np.ndarray)
            and isinstance(box, np.ndarray)
            and isinstance(boundary, list)
        ):
            self.__data, self.__box, self.__boundary = BuildSystem.fromarray(
                pos, box, boundary, vel, type_list
            )
        if sorted_id:
            assert "id" in self.__data.columns
            self.__data.sort_values("id", inplace=True)
        self.if_neigh = False

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
    def data(self):
        """check particles information.

        Returns:
            pd.Dataframe: particles information.
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

    @property
    def pos(self):
        """particle position information.

        Returns:
            np.ndarray: position information.
        """
        return self.__data[["x", "y", "z"]].values

    @property
    def vel(self):
        """particle velocity information.

        Returns:
            np.ndarray: velocity information.
        """
        if "vx" in self.__data.columns:
            return self.__data[["vx", "vy", "vz"]].values
        else:
            return "No Velocity found."

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
            return "No Neighbor Information found. Call build_neighbor() please."

    @property
    def distance_list(self):
        """distance neighbor information. Each row indicates the neighbor atom's distance.

        Returns:
            np.ndarray: distance information.
        """
        if self.if_neigh:
            return self.__distance_list
        else:
            return "No Neighbor Information found. Call build_neighbor() please."

    @property
    def neighbor_number(self):
        """neighbor number information. Each row indicates the neighbor atom's number.

        Returns:
            np.ndarray: neighbor number.
        """
        if self.if_neigh:
            return self.__neighbor_number
        else:
            return "No Neighbor Information found. Call build_neighbor() please."

    @property
    def rc(self):
        """current cutoff distance.

        Returns:
            float: cutoff distance.
        """
        if self.if_neigh:
            return self.__rc
        else:
            return "No Neighbor Information found. Call build_neighbor() please."

    def change_filename(self, filename):
        """change the filename.

        Args:
            filename (str): the new filename.
        """
        assert isinstance(filename, str)
        self.__filename = filename

    def __repr__(self):
        return f"Filename: {self.filename}\nAtom Number: {self.N}\nSimulation Box:\n{self.box}\nTimeStep: {self.__timestep}\nBoundary: {self.boundary}\nParticle Information:\n{self.data.head()}"

    def select(self, data: pd.DataFrame):
        """Generate a subsystem.

        Args:
            data (pd.DataFrame): a new dataframe. Such as system.data[system.data['x']>100]

        Returns:
            System: a new subsystem.
        """
        subSystem = System(
            data=data.reset_index(drop=True),
            box=self.__box,
            boundary=self.__boundary,
            filename=self.__filename,
            fmt=self.__fmt,
        )
        return subSystem

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
            data = self.__data.loc[:, output_col]
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

    def write_data(self, output_name=None, data_format="atomic"):
        """This function writes particles information into a DATA file.

        Args:
            output_name (str, optional): output filename. Defaults to None.
            data_format (str, optional): selected in ['atomic', 'charge']. Defaults to "atomic".
        """
        if output_name is None:
            if self.__filename is None:
                output_name = "output.data"
            else:
                output_name = ".".join(self.__filename.split(".")[:-1]) + ".output.data"

        SaveFile.write_data(
            output_name,
            self.__box,
            self.__boundary,
            data=self.__data,
            pos=None,
            type_list=None,
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
        box = self.__box
        rij = self.pos[i] - self.pos[j]
        nz = rij[2] / box[2][2]
        ny = (rij[1] - nz * box[2][1]) / box[1][1]
        nx = (rij[0] - ny * box[1][0] - nz * box[2][0]) / box[0][0]
        n = [nx, ny, nz]
        for i in range(3):
            if self.__boundary[i] == 1:
                if n[i] > 0.5:
                    n[i] -= 1
                elif n[i] < -0.5:
                    n[i] += 1
        return np.linalg.norm(n[0] * box[0] + n[1] * box[1] + n[2] * box[2])

    def wrap_pos(self):
        """Wrap atom position into box considering the periodic boundary."""
        pos = self.pos
        _wrap_pos(pos, self.box, np.array(self.boundary))
        self.__data[["x", "y", "z"]] = pos

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
        self.__data = pd.concat([self.__data] * num, ignore_index=True)
        self.__data["id"] = np.arange(1, self.N + 1)
        self.__data[["x", "y", "z"]] = repli.pos
        self.__box = repli.box

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

    def cal_atomic_temperature(self, amass, rc=5.0, units="metal", max_neigh=None):
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
            amass (np.ndarray): (:math:`N_{type}`) atomic mass per species.
            rc (float, optional): cutoff distance. Defaults to 5.0.
            units (str, optional): units selected from ['metal', 'charge']. Defaults to "metal".
            max_neigh (int, optional): maximum neighbor number. If not given, will estimate atomatically. Default to None.

        Outputs:
            - **The result is added in self.data['atomic_temp']**.
        """

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

        atype_list = self.data["type"].values.astype(np.int32)
        assert "vx" in self.__data.columns
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
        self.data["atomic_temp"] = AtomicTemp.T

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

        self.data["aja"] = AcklandJonesAna.aja

    def cal_steinhardt_bond_orientation(
        self,
        nnn=12,
        qlist=[6],
        rc=0.0,
        wlflag=False,
        wlhatflag=False,
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
        if nnn > 0:
            if self.if_neigh:
                if self.neighbor_number.min() >= nnn:
                    _partition_select_sort(self.verlet_list, self.distance_list, nnn)
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
        )
        SBO.compute()
        columns = []
        for i in qlist:
            columns.append(f"ql{i}")
        if wlflag:
            for i in qlist:
                columns.append(f"wl{i}")
        if wlhatflag:
            for i in qlist:
                columns.append(f"whl{i}")

        self.data[columns] = SBO.qnarray
        if solidliquid:
            SBO.identifySolidLiquid(threshold, n_bond)
            self.data["solidliquid"] = SBO.solidliquid

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

        self.data["csp"] = CentroSymmetryPara.csp

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
            - **The result is added in self.data['structure_types']**.
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
        self.data["structure_types"] = np.array(ptm.output[:, 0], int)
        if return_rmsd:
            self.data["rmsd"] = ptm.output[:, 1]
        if return_atomic_distance:
            self.data["interatomic_distance"] = ptm.output[:, 2]
        if return_wxyz:
            self.data[["qw", "qx", "qy", "qz"]] = ptm.output[:, 3:]

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
            - **The result is added in self.data['structure_types']**.
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
        self.data["structure_types"] = SFTB.structure_types
        self.data["fault_types"] = SFTB.fault_types

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
        self.data["atomic_entropy"] = AtomicEntro.entropy
        if compute_average:
            self.data["ave_atomic_entropy"] = AtomicEntro.entropy_average

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
            self.rho,
            verlet_list,
            distance_list,
            neighbor_number,
            self.pos,
            self.box,
            self.boundary,
            self.data["type"].values,
        )
        self.PairDistribution.compute()

    def cal_cluster_analysis(self, rc=5.0, max_neigh=80):
        """Divide atoms connected within a given cutoff distance into a cluster.
        It is helpful to recognize the reaction products or fragments under shock loading.

        .. note:: This class use the `same method as in Ovito <https://www.ovito.org/docs/current/reference/pipelines/modifiers/cluster_analysis.html#particles-modifiers-cluster-analysis>`_.

        Args:
            rc (float, optional): cutoff distance.. Defaults to 5.0.
            max_neigh (int, optional): maximum number of atom neighbor number. Defaults to 80.

        Returns:
            int: cluster number.

        Outputs:
            - **The cluster id per atom is added in self.data['cluster_id']**
        """
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

        ClusterAnalysi = ClusterAnalysis(
            rc, verlet_list, distance_list, self.pos, self.box, self.boundary
        )
        ClusterAnalysi.compute()
        self.data["cluster_id"] = ClusterAnalysi.particleClusters
        return ClusterAnalysi.cluster_number

    def cal_common_neighbor_analysis(self, rc=3.0, max_neigh=30):
        """Use Common Neighbor Analysis (CNA) method to recgonize the lattice structure, based
        on which atoms can be divided into FCC, BCC, HCP and Other structure.

        .. note:: If one use this module in publication, one should also cite the original paper.
          `Faken D, Jónsson H. Systematic analysis of local atomic structure combined with 3D computer graphics[J].
          Computational Materials Science, 1994, 2(2): 279-286. <https://doi.org/10.1016/0927-0256(94)90109-0>`_.

        .. hint:: We use the `same algorithm as in LAMMPS <https://docs.lammps.org/compute_cna_atom.html>`_.

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

        The CNA method can recgonize the following structure:

        1. Other
        2. FCC
        3. HCP
        4. BCC
        5. ICO

        Args:
            rc (float, optional): cutoff distance. Defaults to 3.0.
            max_neigh (int, optional): maximum number of atom neighbor number. Defaults to 30.

        Outputs:
            - **The CNA pattern per atom is added in self.data['cna']**.
        """
        repeat = _check_repeat_cutoff(self.box, self.boundary, rc, 4)

        if sum(repeat) == 3:
            if not self.if_neigh:
                self.build_neighbor(rc, max_neigh=max_neigh)
                CommonNeighborAnalysi = CommonNeighborAnalysis(
                    rc,
                    self.pos,
                    self.box,
                    self.boundary,
                    self.verlet_list,
                    self.neighbor_number,
                )
            if self.rc != rc:
                CommonNeighborAnalysi = CommonNeighborAnalysis(
                    rc,
                    self.pos,
                    self.box,
                    self.boundary,
                )
        else:
            CommonNeighborAnalysi = CommonNeighborAnalysis(
                rc,
                self.pos,
                self.box,
                self.boundary,
            )
        CommonNeighborAnalysi.compute()
        self.data["cna"] = CommonNeighborAnalysi.pattern

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
        self.data["cnp"] = CommonNeighborPar.cnp

    def cal_energy_force(self, filename, elements_list, max_neigh=None):
        """Calculate the atomic energy and force based on the given embedded atom method
        EAM potential. Multi-elements alloy is also supported.

        Args:
            filename (str): filename of eam.alloy potential file.
            elements_list (list): elements to be calculated, such as ['Al', 'Al', 'Ni'] indicates setting type 1 and 2 as 'Al' and type 3 as 'Ni'.
            max_neigh (int, optional): maximum number of atom neighbor number. Defaults to 120.

        Outputs:
            - **The energy per atom is added in self.data['pe']**.
            - **The force per atom is added in self.data[["afx", "afy", "afz"]]**.
        """

        potential = EAM(filename)
        repeat = _check_repeat_cutoff(self.box, self.boundary, potential.rc, 5)
        verlet_list, distance_list, neighbor_number = None, None, None
        if sum(repeat) == 3:
            if not self.if_neigh:
                self.build_neighbor(rc=potential.rc, max_neigh=max_neigh)
            if self.rc < potential.rc:
                self.build_neighbor(rc=potential.rc, max_neigh=max_neigh)
            verlet_list, distance_list, neighbor_number = (
                self.verlet_list,
                self.distance_list,
                self.neighbor_number,
            )

        Cal = Calculator(
            potential,
            self.pos,
            self.boundary,
            self.box,
            elements_list,
            self.data["type"].values,
            verlet_list,
            distance_list,
            neighbor_number,
        )
        Cal.compute()

        self.data["pe"] = Cal.energy
        self.data[["afx", "afy", "afz"]] = Cal.force

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
            self.data["type"].values,
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
        self.data["voronoi_volume"] = voro.vol
        self.data["voronoi_number"] = voro.neighbor_number
        self.data["cavity_radius"] = voro.cavity_radius

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

        vbin = self.data[vbin].values
        self.Binning = SpatialBinning(self.pos, direction, vbin, wbin, operation)
        self.Binning.compute()


class MultiSystem(list):
    """
    This class is a collection of mdapy.System class and is helful to handle the atomic trajectory.

    Args:
        filename_list (list): ordered filename list, such as ['melt.0.dump', 'melt.100.dump'].
        unwrap (bool, optional): make atom positions do not wrap into box due to periotic boundary. Defaults to True.
        sorted_id (bool, optional): sort data by atoms id. Defaults to True.
        image_p (np.ndarray, optional): (:math:`N_p, 3`), image_p help to unwrap positions, if don't provided, using minimum image criterion, see https://en.wikipedia.org/wiki/Periodic_boundary_conditions#Practical_implementation:_continuity_and_the_minimum_image_convention.

    Outputs:
        - **pos_list** (np.ndarray): (:math:`N_f, N_p, 3`), :math:`N_f` frames particle position.
        - **Nframes** (int): number of frames.

    """

    def __init__(self, filename_list, unwrap=True, sorted_id=True, image_p=None):
        self.sorted_id = sorted_id
        self.unwrap = unwrap
        self.image_p = image_p
        try:
            from tqdm import tqdm

            progress_bar = tqdm(filename_list)
            for filename in progress_bar:
                progress_bar.set_description(f"Reading {filename}")
                system = System(filename, sorted_id=self.sorted_id)
                self.append(system)
        except Exception:
            for filename in filename_list:
                print(f"\rReading {filename}", end="")
                system = System(filename, sorted_id=self.sorted_id)
                self.append(system)

        self.pos_list = np.array([system.pos for system in self])
        if self.unwrap:
            if self.image_p is None:
                try:
                    self.image_p = np.array(
                        [system.data[["ix", "iy", "iz"]].values for system in self]
                    )
                except Exception:
                    pass
            _unwrap_pos(self.pos_list, self[0].box, self[0].boundary, self.image_p)
            for i, system in enumerate(self):
                system.data[["x", "y", "z"]] = self.pos_list[i]

        self.Nframes = self.pos_list.shape[0]

    def write_dumps(self, output_col=None, compress=False):
        """Write all data to a series of DUMP files.

        Args:
            output_col (list, optional): columns to be saved, such as ['id', 'type', 'x', 'y', 'z'].
            compress (bool, optional): whether compress the DUMP file.
        """
        try:
            from tqdm import tqdm

            progress_bar = tqdm(self)
            for system in progress_bar:
                progress_bar.set_description(f"Saving {system.filename}")
                system.write_dump(output_col=output_col, compress=compress)
        except Exception:
            for system in self:
                print(f"\rSaving {system.filename}", end="")
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
            self[frame].data["msd"] = self.MSD.particle_msd[frame]

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
                self[frame].data["lindemann"] = self.Lindemann.lindemann_atom[frame]
        except Exception:
            pass


if __name__ == "__main__":
    ti.init()
    # system = System(r'E:\HEAShock\111\0.7km\shock-700m.10000.dump')
    # system.write_data()
    # system.write_dump()
    # system.write_dump(compress=True)
    # system.write_dump(output_name='test.dump.gz', compress=True)
    # from tool_function import _init_vel
    system = System("./example/CoCuFeNiPd-4M.dump")
    # print(system)
    system.build_neighbor(max_neigh=60)
    system.cal_pair_distribution()
    # system.cal_warren_cowley_parameter()
    system.cal_centro_symmetry_parameter()
    system.cal_polyhedral_template_matching()
    print(system)
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
