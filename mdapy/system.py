# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .common_neighbor_analysis import CommonNeighborAnalysis
except Exception:
    from common_neighbor_analysis import CommonNeighborAnalysis
try:
    from .neighbor import Neighbor
except Exception:
    from neighbor import Neighbor
try:
    from .temperature import AtomicTemperature
except Exception:
    from temperature import AtomicTemperature
try:
    from .centro_symmetry_parameter import CentroSymmetryParameter
except Exception:
    from centro_symmetry_parameter import CentroSymmetryParameter
try:
    from .entropy import AtomicEntropy
except Exception:
    from entropy import AtomicEntropy
try:
    from .pair_distribution import PairDistribution
except Exception:
    from pair_distribution import PairDistribution
try:
    from .cluser_analysis import ClusterAnalysis
except Exception:
    from cluser_analysis import ClusterAnalysis
try:
    from .potential import EAM
except Exception:
    from potential import EAM
try:
    from .calculator import Calculator
except Exception:
    from calculator import Calculator
try:
    from .void_distribution import VoidDistribution
except Exception:
    from void_distribution import VoidDistribution
try:
    from .warren_cowley_parameter import WarrenCowleyParameter
except Exception:
    from warren_cowley_parameter import WarrenCowleyParameter
try:
    from .voronoi_analysis import VoronoiAnalysis
except Exception:
    from voronoi_analysis import VoronoiAnalysis
try:
    from .mean_squared_displacement import MeanSquaredDisplacement
except Exception:
    from mean_squared_displacement import MeanSquaredDisplacement
try:
    from .lindemann_parameter import LindemannParameter
except Exception:
    from lindemann_parameter import LindemannParameter

try:
    from .spatial_binning import SpatialBinning
except Exception:
    from spatial_binning import SpatialBinning


@ti.kernel
def _wrap_pos(
    pos: ti.types.ndarray(), box: ti.types.ndarray(), boundary: ti.types.ndarray()
):
    """This function is used to wrap particle positions into box considering periodic boundarys.

    Args:
        pos (ti.types.ndarray): (Nx3) particle position.

        box (ti.types.ndarray): (3x2) system box.

        boundary (ti.types.ndarray): boundary conditions, 1 is periodic and 0 is free boundary.
    """
    boxlength = ti.Vector([box[j, 1] - box[j, 0] for j in range(3)])
    for i in range(pos.shape[0]):
        for j in ti.static(range(3)):
            if boundary[j] == 1:
                while pos[i, j] < box[j, 0]:
                    pos[i, j] += boxlength[j]
                while pos[i, j] >= box[j, 1]:
                    pos[i, j] -= boxlength[j]


@ti.kernel
def _unwrap_pos_with_image_p(
    pos_list: ti.types.ndarray(dtype=ti.math.vec3),
    box: ti.types.ndarray(),
    boundary: ti.types.vector(3, dtype=int),
    image_p: ti.types.ndarray(dtype=ti.types.vector(3, int)),
):
    """This function is used to unwrap particle positions
     into box considering periodic boundarys with help of image_p.

    Args:
        pos_list (ti.types.ndarray): (Nframes x Nparticles x 3) particle position.

        box (ti.types.ndarray): (3x2) system box.

        boundary (ti.types.vector): boundary conditions, 1 is periodic and 0 is free boundary.

        image_p (ti.types.ndarray): (Nframes x Nparticles x 3) image_p, such as 1 indicates plus a box distance and -2 means substract two box distances.
    """
    boxlength = ti.Vector([box[j, 1] - box[j, 0] for j in range(3)])
    for i, j in pos_list:
        for k in ti.static(range(3)):
            if boundary[k] == 1:
                pos_list[i, j][k] += image_p[i - 1, j][k] * boxlength[k]


@ti.kernel
def _unwrap_pos_without_image_p(
    pos_list: ti.types.ndarray(dtype=ti.math.vec3),
    box: ti.types.ndarray(),
    boundary: ti.types.vector(3, dtype=int),
    image_p: ti.types.ndarray(dtype=ti.types.vector(3, int)),
):
    """This function is used to unwrap particle positions
     into box considering periodic boundarys without help of image_p.

    Args:
        pos_list (ti.types.ndarray): (Nframes x Nparticles x 3) particle position.

        box (ti.types.ndarray): (3x2) system box.

        boundary (ti.types.vector): boundary conditions, 1 is periodic and 0 is free boundary.

        image_p (ti.types.ndarray): (Nframes x Nparticles x 3) fill with 0.
    """
    boxlength = ti.Vector([box[j, 1] - box[j, 0] for j in range(3)])
    ti.loop_config(serialize=True)
    for frame in range(1, pos_list.shape[0]):
        for i in range(pos_list.shape[1]):
            for j in ti.static(range(3)):
                if boundary[j] == 1:
                    pos_list[frame, i][j] += image_p[frame - 1, i][j] * boxlength[j]
            delta = pos_list[frame, i] - pos_list[frame - 1, i]
            for j in ti.static(range(3)):
                if boundary[j] == 1:
                    if delta[j] >= boxlength[j] / 2:
                        image_p[frame, i][j] -= 1
                        pos_list[frame, i][j] -= boxlength[j]
                    elif delta[j] <= -boxlength[j] / 2:
                        image_p[frame, i][j] += 1
                        pos_list[frame, i][j] += boxlength[j]


def _unwrap_pos(pos_list, box, boundary=[1, 1, 1], image_p=None):
    """This function is used to unwrap particle positions
     into box considering periodic boundarys.

    Args:
        pos_list (np.ndarray): (Nframes x Nparticles x 3) particle position.

        box (np.ndarray): (3x2) system box.

        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].

        image_p (_type_, optional): (Nframes x Nparticles x 3) image_p, such as 1 indicates plus a box distance and -2 means substract two box distances. Defaults to None.
    """
    if image_p is not None:
        boundary = ti.Vector(boundary)
        _unwrap_pos_with_image_p(pos_list, box, boundary, image_p)
    else:
        boundary = ti.Vector(boundary)
        image_p = np.zeros_like(pos_list, dtype=int)
        _unwrap_pos_without_image_p(pos_list, box, boundary, image_p)


class System:
    """This class can generate a System class for rapidly accessing almost all the analysis
    method in mdapy.

    .. note::
      - mdapy now only supports rectangle box and triclinic system will raise an error.
      - mdapy only supports the simplest DATA format, atomic and charge, which means like bond information will cause an error.
      - We recommend you use DUMP as input file format or directly give particle positions and box.

    Args:
        filename (str, optional): DATA/DUMP filename. Defaults to None.
        format (str, optional): 'data' or 'dump', One can explicitly assign the file format or mdapy will handle it with the postsuffix of filename. Defaults to None.
        box (np.ndarray, optional): (:math:`3, 2`) system box. Defaults to None.
        pos (np.ndarray, optional): (:math:`N_p, 3`) particles positions. Defaults to None.
        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].
        vel (np.ndarray, optional): (:math:`N_p, 3`) particles velocities. Defaults to None.
        type_list (np.ndarray, optional): (:math:`N_p`) type per particles. Defaults to 1.
        amass (np.ndarray, optional): (:math:`N_type`) atomic mass. Defaults to None.
        q (np.ndarray, optional): (:math:`N_p`) atomic charge. Defaults to 0.0.
        data_format (str, optional): `'atomic' or 'charge' <https://docs.lammps.org/read_data.html>`_ defined in lammps, format for DATA file. Defaults to None.
        sorted_id (bool, optional): whether sort system data by the particle id. Defaults to False.

    Outputs:
        - **data** (pd.DataFrame) - system data.


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
        format=None,
        box=None,
        pos=None,
        boundary=[1, 1, 1],
        vel=None,
        type_list=None,
        amass=None,
        q=None,
        data_format=None,
        sorted_id=False,
    ):
        self.dump_head = None
        self.data_head = None
        self.data_format = data_format
        self.amass = amass
        self.if_neigh = False
        self.filename = filename
        self.sorted_id = sorted_id
        if filename is None:

            self.format = format
            self.box = box
            self.pos = pos
            self.N = self.pos.shape[0]
            self.boundary = boundary
            self.vel = vel
            if type_list is None:
                self.type_list = np.ones(self.N)
            else:
                self.type_list = type_list
            if self.data_format == "charge":
                if q is None:
                    self.q = np.zeros(self.N)
                else:
                    self.q = q
            else:
                self.q = q

            self.data = pd.DataFrame(
                np.c_[np.arange(self.N) + 1, self.type_list, self.pos],
                columns=["id", "type", "x", "y", "z"],
            )
            if self.data_format == "charge":
                # self.data['q'] = self.q
                # self.data[["id", "type", "q", "x", "y", "z"]] = self.data[["id", "type", "x", "y", "z", "q"]]
                self.data.insert(2, "q", self.q)
            if not self.vel is None:
                self.data[["vx", "vy", "vz"]] = self.vel
            self.data[["id", "type"]] = self.data[["id", "type"]].astype(int)
            self.Ntype = len(np.unique(self.data["type"]))
        else:

            if format is None:
                self.format = self.filename.split(".")[-1]
            else:
                self.format = format
            assert self.format in [
                "data",
                "dump",
            ], "format only surppot dump and data file defined in lammps, and data file only surpport atomic and charge format."
            if self.format == "data":
                self._read_data()
            elif self.format == "dump":
                self._read_dump()
                self.data_format = None
            self.N = self.pos.shape[0]

        self.lx, self.ly, self.lz = self.box[:, 1] - self.box[:, 0]
        self.vol = self.lx * self.ly * self.lz
        self.rho = self.N / self.vol

    def _read_data(self):
        self.data_head = []
        self.box = np.zeros((3, 2))

        with open(self.filename) as op:
            file = op.readlines()

        row = 0
        mass_row = 0
        for line in file:

            self.data_head.append(line)
            content = line.split()
            if len(content):
                if content[-1] == "atoms":
                    self.N = int(content[0])
                if len(content) >= 2:
                    if content[1] == "bond":
                        raise "Do not support bond style."
                if len(content) >= 3:
                    if content[1] == "atom" and content[2] == "types":
                        self.Ntype = int(content[0])
                if content[-1] == "xhi":
                    self.box[0, :] = np.array([content[0], content[1]], dtype=float)
                if content[-1] == "yhi":
                    self.box[1, :] = np.array([content[0], content[1]], dtype=float)
                if content[-1] == "zhi":
                    self.box[2, :] = np.array([content[0], content[1]], dtype=float)
                if content[-1] in ["xy", "xz", "yz"]:
                    raise "Do not support triclinic box."
                if content[0] == "Masses":
                    mass_row = row + 1
                if content[0] == "Atoms":
                    break
            row += 1
        if mass_row > 0:
            self.amass = np.array(
                [
                    i.split()[:2]
                    for i in self.data_head[mass_row + 1 : mass_row + 1 + self.Ntype]
                ],
                dtype=float,
            )[:, 1]
        self.boundary = [1, 1, 1]

        if self.data_head[-1].split()[-1] == "atomic":
            self.data_format = "atomic"
            self.col_names = ["id", "type", "x", "y", "z"]
        elif self.data_head[-1].split()[-1] == "charge":
            self.data_format = "charge"
            self.col_names = ["id", "type", "q", "x", "y", "z"]
        else:
            if len(file[row + 2].split()) == 5:
                self.data_format = "atomic"
                self.col_names = ["id", "type", "x", "y", "z"]
            elif len(file[row + 2].split()) == 6:
                self.data_format = "charge"
                self.col_names = ["id", "type", "q", "x", "y", "z"]
            else:
                raise "Unrecgonized data format. Only support atomic and charge."

        data = np.array(
            [i.split() for i in file[row + 2 : row + 2 + self.N]], dtype=float
        )[:, : len(self.col_names)]
        row += 2 + self.N
        if_vel = False
        if row < len(file):
            for line in file[row:]:
                content = line.split()
                if len(content):
                    if content[0] == "Velocities":
                        if_vel = True
                        break
                row += 1
            if if_vel:
                vel = np.array(
                    [i.split() for i in file[row + 2 : row + 2 + self.N]], dtype=float
                )[:, 1:]
                self.col_names += ["vx", "vy", "vz"]
                self.data = pd.DataFrame(np.c_[data, vel], columns=self.col_names)
        else:
            self.data = pd.DataFrame(data, columns=self.col_names)

        if self.sorted_id:
            self.data.sort_values("id", inplace=True)
        self.data[["id", "type"]] = self.data[["id", "type"]].astype(int)
        self.pos = self.data[["x", "y", "z"]].values
        if if_vel:
            self.vel = self.data[["vx", "vy", "vz"]].values

    def _read_dump(self):
        self.dump_head = []
        with open(self.filename) as op:
            for _ in range(9):
                self.dump_head.append(op.readline())
        self.boundary = [1 if i == "pp" else 0 for i in self.dump_head[4].split()[-3:]]
        self.box = np.array([i.split()[:2] for i in self.dump_head[5:8]]).astype(float)
        self.col_names = self.dump_head[8].split()[2:]
        self.data = pd.read_csv(
            self.filename,
            skiprows=9,
            index_col=False,
            header=None,
            sep=" ",
            names=self.col_names,
        )

        if self.sorted_id:
            self.data.sort_values("id", inplace=True)
        self.pos = self.data[["x", "y", "z"]].values
        self.Ntype = len(np.unique(self.data["type"]))
        try:
            self.vel = self.data[["vx", "vy", "vz"]].values
        except Exception:
            pass

    def write_dump(self, output_name=None, output_col=None):
        """Write data to a DUMP file.

        Args:
            output_name (str, optional): filename of generated DUMP file.
            output_col (list, optional): which columns to be saved, which should be inclued in data columns, such as ['id', 'type', 'x', 'y', 'z'].
        """
        if output_col is None:
            data = self.data
        else:
            data = self.data.loc[:, output_col]

        if output_name is None:
            if self.filename is None:
                output_name = "output.dump"
            else:
                output_name = self.filename[:-4] + "output.dump"
        col_name = "ITEM: ATOMS "
        for i in data.columns:
            col_name += i
            col_name += " "
        col_name += "\n"
        with open(output_name, "w") as op:
            if self.dump_head is None:
                op.write("ITEM: TIMESTEP\n0\n")
                op.write("ITEM: NUMBER OF ATOMS\n")
                op.write(f"{self.N}\n")
                boundary = ["pp" if i == 1 else "ss" for i in self.boundary]
                op.write(
                    f"ITEM: BOX BOUNDS {boundary[0]} {boundary[1]} {boundary[2]}\n"
                )
                op.write(f"{self.box[0, 0]} {self.box[0, 1]}\n")
                op.write(f"{self.box[1, 0]} {self.box[1, 1]}\n")
                op.write(f"{self.box[2, 0]} {self.box[2, 1]}\n")
                op.write("".join(col_name))
            else:
                self.dump_head[3] = f"{data.shape[0]}\n"
                op.write("".join(self.dump_head[:-1]))
                op.write("".join(col_name))
        data.to_csv(
            output_name, header=None, index=False, sep=" ", mode="a", na_rep="nan"
        )

    def write_data(self, output_name=None, data_format=None):
        """Write data to a DATA file.

        Args:
            output_name (str, optional): filename of generated DATA file.
            data_format (str, optional): selected in ['atomic', 'charge'].
        """
        data = self.data
        if data_format is None:
            if self.data_format is None:
                data_format = "atomic"
            else:
                data_format = self.data_format
        else:
            assert data_format in [
                "atomic",
                "charge",
            ], "Unrecgonized data format. Only support atomic and charge."

        if output_name is None:
            if self.filename is None:
                output_name = "output.data"
            else:
                output_name = self.filename[:-4] + "output.data"

        with open(output_name, "w") as op:
            if self.data_head is None:
                op.write("# LAMMPS data file written by mdapy@HerrWu.\n\n")
                op.write(f"{data.shape[0]} atoms\n{self.Ntype} atom types\n\n")
                for i, j in zip(self.box, ["x", "y", "z"]):
                    op.write(f"{i[0]} {i[1]} {j}lo {j}hi\n")
                op.write("\n")
                if not self.amass is None:
                    op.write("Masses\n\n")
                    for i in range(self.Ntype):
                        op.write(f"{i+1} {self.amass[i]}\n")
                    op.write("\n")
                op.write(rf"Atoms # {data_format}")
                op.write("\n\n")
            else:
                self.data_head[-1] = f"Atoms # {data_format}\n"
                op.write("# LAMMPS data file written by mdapy@HerrWu.\n")
                op.write("".join(self.data_head[1:]))
                op.write("\n")
        if data_format == "atomic":
            self.data[["id", "type", "x", "y", "z"]].to_csv(
                output_name, header=None, index=False, sep=" ", mode="a", na_rep="nan"
            )
        elif data_format == "charge":
            if "q" not in self.data.columns:
                self.data.insert(2, "q", np.zeros(self.N))
                self.data[["id", "type", "q", "x", "y", "z"]].to_csv(
                    output_name,
                    header=None,
                    index=False,
                    sep=" ",
                    mode="a",
                    na_rep="nan",
                )
                self.data.drop("q", axis=1, inplace=True)
            else:
                self.data[["id", "type", "q", "x", "y", "z"]].to_csv(
                    output_name,
                    header=None,
                    index=False,
                    sep=" ",
                    mode="a",
                    na_rep="nan",
                )
        try:
            if len(self.vel):
                with open(output_name, "a") as op:
                    op.write("\nVelocities\n\n")
                data[["id", "vx", "vy", "vz"]].to_csv(
                    output_name,
                    header=None,
                    index=False,
                    sep=" ",
                    mode="a",
                    na_rep="nan",
                )
        except Exception:
            # print("No velocities provided!")
            pass

    def atom_distance(self, i, j):
        """Calculate the distance fo atom :math:`i` and atom :math:`j` considering the periodic boundary.

        Args:
            i (int): atom :math:`i`.
            j (int): atom :math:`j`.

        Returns:
            float: distance between given two atoms.
        """
        rij = self.pos[i] - self.pos[j]
        for i in range(3):
            if self.boundary[i] == 1:
                box_length = self.box[i][1] - self.box[i][0]
                rij[i] = rij[i] - box_length * np.round(rij[i] / box_length)
        return np.linalg.norm(rij)

    def wrap_pos(self):
        """Wrap atom position into box considering the periodic boundary."""
        pos = self.pos.copy()  # a deep copy can be modified
        _wrap_pos(pos, self.box, np.array(self.boundary))
        self.pos = pos
        self.data[["x", "y", "z"]] = self.pos

    def build_neighbor(self, rc=5.0, max_neigh=80, exclude=True):
        """Build neighbor withing a spherical distance based on the mdapy.Neighbor class.

        Args:
            rc (float, optional): cutoff distance. Defaults to 5.0.
            max_neigh (int, optional): maximum number of atom neighbor number. Defaults to 80.
            exclude (bool, optional): whether exclude atom itself. Defaults to True.

        Outputs:
            - **verlet_list** (np.ndarray) - (:math:`N_p, max\_neigh`) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1.
            - **distance_list** (np.ndarray) - (:math:`N_p, max\_neigh`) distance_list[i, j] means distance between i and j atom.
            - **neighbor_number** (np.ndarray) - (:math:`N_p`) neighbor atoms number.
        """
        Neigh = Neighbor(self.pos, self.box, rc, self.boundary, max_neigh, exclude)
        Neigh.compute()

        self.verlet_list, self.distance_list, self.neighbor_number, self.rc = (
            Neigh.verlet_list,
            Neigh.distance_list,
            Neigh.neighbor_number,
            rc,
        )
        self.if_neigh = True

    def cal_atomic_temperature(self, amass, rc=5.0, units="metal", max_neigh=80):
        """Calculate an average thermal temperature per atom, wchich is useful at shock
        simulations. The temperature of atom :math:`i` is given by:

        .. math:: T_i=\\sum^{N^i_{neigh}}_0 m^i_j(v_j^i -v_{COM}^i)^2/(3N_pk_B),

        where :math:`N^i_{neigh}` is neighbor atoms number of atom :math:`i`,
        :math:`m^i_j` and :math:`v^i_j` are the atomic mass and velocity of neighbor atom :math:`j` of atom :math:`i`,
        :math:`k_B` is the Boltzmann constant and :math:`N_p` is the number of particles in system, :math:`v^i_{COM}` is
        the center of mass COM velocity of neighbor of atom :math:`i` and is given by:

        .. math:: v^i_{COM}=\\frac{\\sum _0^{N^i_{neigh}}m^i_jv_j^i}{\\sum_0^{N^i_{neigh}} m_j^i}.

        Here the neighbor of atom :math:`i` includes itself.

        Args:
            amass (np.ndarray): (:math:`N_{type}`) atomic mass per species.
            rc (float, optional): cutoff distance. Defaults to 5.0.
            units (str, optional): units selected from ['metal', 'charge']. Defaults to "metal".
            max_neigh (int, optional): maximum number of atom neighbor number. Defaults to 80.

        Outputs:
            - **The result is added in self.data['atomic_temp']**.
        """

        if not self.if_neigh:
            self.build_neighbor(rc, max_neigh)
        elif self.rc < rc:
            self.build_neighbor(rc, max_neigh)
        atype_list = self.data["type"].values.astype(np.int32)
        AtomicTemp = AtomicTemperature(
            amass,
            self.vel,
            self.verlet_list,
            self.distance_list,
            atype_list,
            rc,
            units,
        )
        AtomicTemp.compute()
        self.data["atomic_temp"] = AtomicTemp.T

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

        try:
            CentroSymmetryPara = CentroSymmetryParameter(
                N, self.pos, self.box, self.boundary
            )
            CentroSymmetryPara.compute()
        except Exception:
            pos = self.pos.copy()  # a deep copy can be modified
            _wrap_pos(pos, self.box, np.array(self.boundary))
            CentroSymmetryPara = CentroSymmetryParameter(
                N, pos, self.box, self.boundary
            )
            CentroSymmetryPara.compute()
        self.data["csp"] = CentroSymmetryPara.csp

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

        if average_rc is not None:
            assert average_rc <= rc, "Average rc should not be larger than rc!"
        if not self.if_neigh:
            self.build_neighbor(rc=rc, max_neigh=max_neigh)
        elif self.rc < rc:
            self.build_neighbor(rc=rc, max_neigh=max_neigh)

        AtomicEntro = AtomicEntropy(
            self.vol,
            self.verlet_list,
            self.distance_list,
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
        if not self.if_neigh:
            self.build_neighbor(rc=rc, max_neigh=max_neigh)
        elif self.rc < rc:
            self.build_neighbor(rc=rc, max_neigh=max_neigh)
        rho = self.N / self.vol
        self.PairDistribution = PairDistribution(
            rc,
            nbin,
            rho,
            self.verlet_list,
            self.distance_list,
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
        if not self.if_neigh:
            self.build_neighbor(rc=rc, max_neigh=max_neigh)
        elif self.rc < rc:
            self.build_neighbor(rc=rc, max_neigh=max_neigh)

        ClusterAnalysi = ClusterAnalysis(rc, self.verlet_list, self.distance_list)
        ClusterAnalysi.compute()
        self.data["cluster_id"] = ClusterAnalysi.particleClusters
        return ClusterAnalysi.cluster_number

    def cal_common_neighbor_analysis(self, rc=3.0, max_neigh=30):
        """Sse Common Neighbor Analysis (CNA) method to recgonize the lattice structure, based
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

        if not self.if_neigh:
            Neigh = Neighbor(self.pos, self.box, rc, self.boundary, max_neigh)
            Neigh.compute()
            self.verlet_list, self.distance_list, self.neighbor_number, self.rc = (
                Neigh.verlet_list,
                Neigh.distance_list,
                Neigh.neighbor_number,
                rc,
            )
            self.if_neigh = True
        elif self.rc != rc:
            Neigh = Neighbor(self.pos, self.box, rc, self.boundary, max_neigh)
            Neigh.compute()

        CommonNeighborAnalysi = CommonNeighborAnalysis(
            rc,
            Neigh.verlet_list,
            Neigh.neighbor_number,
            self.pos,
            self.box,
            self.boundary,
        )
        CommonNeighborAnalysi.compute()
        self.data["cna"] = CommonNeighborAnalysi.pattern

    def cal_energy_force(self, filename, elements_list, max_neighbor=120):
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

        if not self.if_neigh:
            self.build_neighbor(rc=potential.rc, max_neigh=max_neighbor)
        if self.rc < potential.rc:
            self.build_neighbor(rc=potential.rc, max_neigh=max_neighbor)

        Cal = Calculator(
            potential,
            elements_list,
            self.data["type"].values,
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
            self.pos,
            self.boundary,
            self.box,
        )
        Cal.compute()

        self.data["pe"] = Cal.energy
        self.data[["afx", "afy", "afz"]] = Cal.force

    def cal_void_distribution(self, cell_length, out_void=False):
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

        Returns:
            tuple: (void_number, void_volume), number and volume of voids.
        """

        void = VoidDistribution(
            self.pos,
            self.box,
            cell_length,
            self.boundary,
            out_void=out_void,
            head=self.dump_head,
            out_name=self.filename[:-5] + ".void.dump",
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
        if not self.if_neigh:
            self.build_neighbor(rc=rc, max_neigh=max_neigh)
            self.WarrenCowleyParameter = WarrenCowleyParameter(
                self.verlet_list, self.neighbor_number, self.data["type"].values
            )
        elif self.rc != rc:
            neigh = Neighbor(self.pos, self.box, rc, self.boundary, max_neigh)
            neigh.compute()
            self.WarrenCowleyParameter = WarrenCowleyParameter(
                neigh.verlet_list, neigh.neighbor_number, self.data["type"].values
            )

        self.WarrenCowleyParameter.compute()

    def cal_voronoi_volume(self):
        """This class is used to calculate the Voronoi polygon, wchich can be applied to
        estimate the atomic volume. The calculation is conducted by the `voro++ <https://math.lbl.gov/voro++/>`_ package and
        this class only provides a wrapper.

        Outputs:
            - **The atomic Voronoi volume is added in self.data['voronoi_volume']**.
            - **The atomic Voronoi neighbor is added in self.data["voronoi_number"]**.
            - **The atomic Voronoi cavity radius is added in self.data["cavity_radius"]**.

        """
        voro = VoronoiAnalysis(self.pos, self.box, self.boundary)
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

        progress_bar = tqdm(filename_list)
        for filename in progress_bar:
            progress_bar.set_description(f"Reading {filename}")
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

    def write_dumps(self, output_col=None):
        """Write all data to a series of DUMP files.

        Args:
            output_col (list, optional): columns to be saved, such as ['id', 'type', 'x', 'y', 'z'].
        """
        progress_bar = tqdm(self)
        for system in progress_bar:
            try:
                progress_bar.set_description(f"Saving {system.filename}")
            except Exception:
                progress_bar.set_description(f"Saving file...")
            system.write_dump(output_col=output_col)

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
            self[frame].data["msd"] = self.MSD.partical_msd[frame]

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
    import taichi as ti

    ti.init()
    # system = System(filename=r"./example/CoCuFeNiPd-4M.data")
    system = System(filename=r"./example/CoCuFeNiPd-4M.dump")
    # box = np.array([[0, 10], [0, 10], [0, 10]])
    # pos = np.array([[0.0, 0.0, 0.0], [1.5, 6.5, 9.0]])
    # vel = np.array([[1.0, 0.0, 0.0], [2.5, 6.5, 9.0]])
    # q = np.array([1.0, 2.0])
    # system = System(
    #     box=box,
    #     pos=pos,
    #     type_list=[1, 2],
    #     amass=[2.3, 4.5],
    #     vel=vel,
    #     boundary=[1, 1, 0],
    #     data_format="charge",
    #     q=q,
    # )
    system.wrap_pos()
    print(system.data)
    print(system.Ntype, system.N, system.format)
    print(system.data_head)
    # system.write_data(data_format="charge")
    # system.write_dump()
