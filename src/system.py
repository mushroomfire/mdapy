# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np
import pandas as pd
from tqdm import tqdm

from .common_neighbor_analysis import CommonNeighborAnalysis
from .neighbor import Neighbor
from .temperature import AtomicTemperature
from .centro_symmetry_parameter import CentroSymmetryParameter
from .entropy import AtomicEntropy
from .pair_distribution import PairDistribution
from .cluser_analysis import ClusterAnalysis

from .potential import EAM
from .calculator import Calculator
from .void_distribution import VoidDistribution
from .warren_cowley_parameter import WarrenCowleyParameter
from .voronoi_analysis import VoronoiAnalysis

from .mean_squared_displacement import MeanSquaredDisplacement
from .lindemann_parameter import LindemannParameter

from .spatial_binning import SpatialBinning


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
    pos_list: ti.types.ndarray(element_dim=1),
    box: ti.types.ndarray(),
    boundary: ti.types.vector(3, dtype=int),
    image_p: ti.types.ndarray(element_dim=1),
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
    pos_list: ti.types.ndarray(element_dim=1),
    box: ti.types.ndarray(),
    boundary: ti.types.vector(3, dtype=int),
    image_p: ti.types.ndarray(element_dim=1),
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
                        image_p[frame, i][j] -= 1.0
                        pos_list[frame, i][j] -= boxlength[j]
                    elif delta[j] <= -boxlength[j] / 2:
                        image_p[frame, i][j] += 1.0
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
        image_p = np.zeros_like(pos_list)
        _unwrap_pos_without_image_p(pos_list, box, boundary, image_p)


class System:
    """
    This is the core class in mdapy project. One can see the usage at below.

        Args:
            filename (str, optional): DATA/DUMP filename. Defaults to None.
            format (str, optional): 'data' or 'dump', One can explicitly assign the file format or mdapy will handle it with the postsuffix of filename.. Defaults to None.
            box (np.ndarray, optional): (3 x 2) system box. Defaults to None.
            pos (np.ndarray, optional): (Nparticles x 3) particles positions. Defaults to None.
            boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].
            vel (np.ndarray, optional): (Nparticles x 3) particles velocities. Defaults to None.
            type_list (np.ndarray, optional): (Nparticles,) type per particles. Defaults to 1.
            amass (np.ndarray, optional): (Ntypes,) atomic mass. Defaults to None.
            q (np.ndarray, optional): (Nparticles,) atomic charge. Defaults to 0.0.
            data_format (str, optional): 'atomic' or 'charge', format for DATA file. Defaults to None.
            sorted_id (bool, optional): Whether sort system data by the particle id. Defaults to False.
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
            >>> system.cal_atomic_entropy()
            One can check the calculation results:
            >>> system.data
            And easily save it into disk with DUMP/DATA format.
            >>> system.write_dump()
        Note:
        mdapy now only support rectangle box and triclinic system will raise and error.
        mdapy only support the simplest DATA format, atomic and charge, which means like bond information will cause an error.
        We recommend you use DUMP as input file format or directly give particle positions and box.
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
                self.read_data()
            elif self.format == "dump":
                self.read_dump()
                self.data_format = None
            self.N = self.pos.shape[0]

        self.lx, self.ly, self.lz = self.box[:, 1] - self.box[:, 0]
        self.vol = self.lx * self.ly * self.lz
        self.rho = self.N / self.vol

    def read_data(self):
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

    def read_dump(self):
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

        """
        distance fo atom i and atom j considering periodic boundary
        """

        rij = self.pos[i] - self.pos[j]
        for i in range(3):
            if self.boundary[i] == 1:
                box_length = self.box[i][1] - self.box[i][0]
                rij[i] = rij[i] - box_length * np.round(rij[i] / box_length)
        return np.linalg.norm(rij)

    def wrap_pos(self):
        pos = self.pos.copy()  # a deep copy can be modified
        _wrap_pos(pos, self.box, np.array(self.boundary))
        self.pos = pos
        self.data[["x", "y", "z"]] = self.pos

    def build_neighbor(self, rc=5.0, max_neigh=80, exclude=True):
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

        """
        amass : (N_type)的array,N_type为原子类型的数目,按照体系的类型顺序写入相应原子类型的相对原子质量.
        units : metal or real.
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
        """
        N : int, 大于0的偶数,对于FCC结构是12,对于BCC是8. default : 12
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
        self, rc=5.0, sigma=0.25, use_local_density=False, max_neigh=80
    ):

        """
        sigma : 用来表征插值的精细程度类似于,不能太大, 默认使用0.25或者0.2就行.
        use_local_density : 是否使用局部密度,俺也不咋懂.
        """

        if not self.if_neigh:
            self.build_neighbor(rc=rc, max_neigh=max_neigh)
        elif self.rc < rc:
            self.build_neighbor(rc=rc, max_neigh=max_neigh)

        AtomicEntro = AtomicEntropy(
            self.vol,
            self.distance_list,
            rc,
            sigma,
            use_local_density,
        )
        AtomicEntro.compute()
        self.data["atomic_entropy"] = AtomicEntro.entropy

    def cal_pair_distribution(self, rc=5.0, nbin=200, max_neigh=80):
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
        if not self.if_neigh:
            self.build_neighbor(rc=rc, max_neigh=max_neigh)
        elif self.rc < rc:
            self.build_neighbor(rc=rc, max_neigh=max_neigh)

        ClusterAnalysi = ClusterAnalysis(rc, self.verlet_list, self.distance_list)
        ClusterAnalysi.compute()
        self.data["cluster_id"] = ClusterAnalysi.particleClusters
        return ClusterAnalysi.cluster_number

    def cal_common_neighbor_analysis(self, rc=3.0, max_neigh=30):
        """
        rc should be 0.854*a for FCC and 1.207*a for BCC metal.
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

    def cal_energy_force(self, filename, elements_list):
        """
        filename : eam.alloy 势函数文件名
        elements_list : 每一个原子type对应的元素, ['Al', 'Al', 'Ni']
        """

        potential = EAM(filename)

        if not self.if_neigh:
            self.build_neighbor(rc=potential.rc, max_neigh=120)
        if self.rc < potential.rc:
            self.build_neighbor(rc=potential.rc, max_neigh=120)

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
        """
        input:
        cell_length : 比晶胞长度大一点的数字
        out_void : 是否导出void的坐标文件
        output:
        void_number, void_volume
        """
        void = VoidDistribution(
            self.pos,
            self.box,
            cell_length,
            self.boundary,
            out_void=out_void,
            head=self.head,
            out_name=self.filename[:-5] + ".void.dump",
        )
        void.compute()

        return void.void_number, void.void_volume

    def cal_warren_cowley_parameter(self, rc=3.0, max_neigh=50):
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
        voro = VoronoiAnalysis(self.pos, self.box, self.boundary)
        voro.compute()
        self.data["voronoi_volume"] = voro.vol
        self.data["voronoi_number"] = voro.neighbor_number
        self.data["cavity_radius"] = voro.cavity_radius

    def spatial_binning(self, direction, vbin, wbin, operation="mean"):
        """
        input:
        pos: (Nx3) ndarray, spatial coordination and the order is x, y and z
        direction : str, binning direction
        1. 'x', 'y', 'z', One-dimensional binning
        2. 'xy', 'xz', 'yz', Two-dimensional binning
        3. 'xyz', Three-dimensional binning
        vbin: str or list
        wbin: float, width of each bin, default is 5.
        operation: str, ['mean', 'sum', 'min', 'max'], default is 'mean'
        output:
        Binning class
        res: ndarray
        coor: dict
        """
        vbin = self.data[vbin].values
        self.Binning = SpatialBinning(self.pos, direction, vbin, wbin, operation)
        self.Binning.compute()


class MultiSystem(list):

    """
    Generate a list of systems.
    input:
    filename_list : a list containing filename, such as ['melt.0.dump', 'melt.1.dump']
    unwrap : bool, make atom positions do not wrap into box due to periotic boundary
    sorted_id : bool, sort data by atoms id
    image_p : image_p help to unwrap positions, if don't provided, using minimum image criterion, see https://en.wikipedia.org/wiki/Periodic_boundary_conditions#Practical_implementation:_continuity_and_the_minimum_image_convention.
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
                    self.image_p = [system.data[["ix", "iy", "iz"]] for system in self]
                except Exception:
                    pass
            _unwrap_pos(self.pos_list, self[0].box, self[0].boundary, self.image_p)
            for i, system in enumerate(self):
                system.data[["x", "y", "z"]] = self.pos_list[i]

        self.Nframes = self.pos_list.shape[0]

    def write_dumps(self, output_col=None):
        progress_bar = tqdm(self)
        for system in progress_bar:
            try:
                progress_bar.set_description(f"Saving {system.filename}")
            except Exception:
                progress_bar.set_description(f"Saving file...")
            system.write_dump(output_col=output_col)

    def cal_mean_squared_displacement(self, mode="windows"):
        """
        Calculating MSD variation.
        input:
        mode : str, "windows" or "direct",
        see
        1. https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft
        2. https://freud.readthedocs.io/en/latest/modules/msd.html
        output:
        self.MSD.msd
        """

        self.MSD = MeanSquaredDisplacement(self.pos_list, mode=mode)
        self.MSD.compute()
        for frame in range(self.Nframes):
            self[frame].data["msd"] = self.MSD.partical_msd[frame]

    def cal_lindemann_parameter(self, only_global=False):
        """
        Need high memory!!!
        Using Welford method to updated the varience and mean of rij
        see
        1. https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford
        2. https://zhuanlan.zhihu.com/p/408474710

        Input:
        pos_list : np.ndarray (Nframes, Natoms, 3)
        only_global : bool, only calculate globle lindemann index, fast, parallel

        Output:
        lindemann_atom : np.ndarray (Nframes, Natoms)
        lindemann_frame : np.ndarray (Nframes)
        lindemann_trj : float
        """

        self.Lindemann = LindemannParameter(self.pos_list, only_global)
        self.Lindemann.compute()
        try:
            for frame in range(self.Nframes):
                self[frame].data["lindemann"] = self.Lindemann.lindemann_atom[frame]
        except Exception:
            pass


if __name__ == "__main__":

    # system = System(filename=r"./example/CoCuFeNiPd-4M.data")
    # system = System(filename=r"./example/CoCuFeNiPd-4M.dump")
    box = np.array([[0, 10], [0, 10], [0, 10]])
    pos = np.array([[0.0, 0.0, 0.0], [1.5, 6.5, 9.0]])
    vel = np.array([[1.0, 0.0, 0.0], [2.5, 6.5, 9.0]])
    q = np.array([1.0, 2.0])
    system = System(
        box=box,
        pos=pos,
        type_list=[1, 2],
        amass=[2.3, 4.5],
        vel=vel,
        boundary=[1, 1, 0],
        data_format="charge",
        q=q,
    )
    print(system.data)
    print(system.Ntype, system.N, system.format)
    print(system.data_head)
    system.write_data(data_format="charge")
    system.write_dump()
