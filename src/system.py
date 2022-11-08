import numpy as np
import pandas as pd

from .common_neighbor_analysis import CommonNeighborAnalysis
from .adaptive_common_neighbor_analysis import AdaptiveCommonNeighborAnalysis
from .neighbor import Neighbor
from .kdtree import kdtree
from .temperature import AtomicTemperature
from .centro_symmetry_parameter import CentroSymmetryParameter
from .entropy import AtomicEntropy
from .pair_distribution import PairDistribution
from .cluser_analysis import ClusterAnalysis

from .potential import EAM
from .calculator import Calculator
from .void_distribution import VoidDistribution
from .warren_cowley_parameter import WarrenCowleyParameter


class System:
    """
    生成一个System类,支持读取的文件格式为LAMMPS中的.dump格式,以此为基础来进行后处理,可以将结果保存为.data或者.dump格式.
    输入参数:
    filename : dump文件名称
    """

    def __init__(self, filename):
        self.filename = filename

        self.read_box()
        self.read_dump(self.col_names)
        self.lx, self.ly, self.lz = self.box[:, 1] - self.box[:, 0]
        self.vol = self.lx * self.ly * self.lz
        self.if_neigh = False

    def read_box(self):
        self.head = []
        with open(self.filename) as op:
            for _ in range(9):
                self.head.append(op.readline())
        self.boundary = [1 if i == "pp" else 0 for i in self.head[4].split()[-3:]]
        self.box = np.array([i.split()[:2] for i in self.head[5:8]]).astype(float)
        self.col_names = self.head[8].split()[2:]

    def read_dump(self, col_names):

        self.data = pd.read_csv(
            self.filename,
            skiprows=9,
            index_col=False,
            header=None,
            sep=" ",
            names=col_names,
        )
        self.N = self.data.shape[0]
        self.pos = self.data[["x", "y", "z"]].values

        try:
            self.vel = self.data[["vx", "vy", "vz"]].values
        except Exception:
            pass

    def write_dump(self, output_name=None, output_col=None):
        head, filename = self.head, self.filename
        if output_col is None:
            data = self.data
        else:
            data = self.data.loc[:, output_col]
        head[3] = f"{data.shape[0]}\n"
        # for dtype, name in zip(data.dtypes, data.columns):
        #     if dtype == "int64":
        #         data[name] = data[name].astype(np.int32)
        #     elif dtype == "float64":
        #         data[name] = data[name].astype(np.float32)
        if output_name is None:
            prefilename = filename.split(".")
            prefilename.insert(-1, "output")
            output_name = ".".join(prefilename)
        col_name = "ITEM: ATOMS "
        for i in data.columns:
            col_name += i
            col_name += " "
        col_name += "\n"
        with open(output_name, "w") as op:
            op.write("".join(head[:-1]))
            op.write("".join(col_name))
        data.to_csv(
            output_name, header=None, index=False, sep=" ", mode="a", na_rep="nan"
        )

    def write_data(self, output_name=None):
        data = self.data
        # for dtype, name in zip(data.dtypes, data.columns):
        #     if dtype == "int64":
        #         data[name] = data[name].astype(np.int32)
        #     elif dtype == "float64":
        #         data[name] = data[name].astype(np.float32)
        if output_name is None:
            output_name = self.filename[:-4] + "data"
        Ntype = len(np.unique(data["type"]))

        with open(output_name, "w") as op:
            op.write("# LAMMPS data file written by mdapy@HerrWu.\n\n")
            op.write(f"{data.shape[0]} atoms\n{Ntype} atom types\n\n")
            for i, j in zip(self.box, ["x", "y", "z"]):
                op.write(f"{i[0]} {i[1]} {j}lo {j}hi\n")
            op.write("\n")
            op.write(r"Atoms # atomic")
            op.write("\n\n")
        data[["id", "type", "x", "y", "z"]].to_csv(
            output_name, header=None, index=False, sep=" ", mode="a", na_rep="nan"
        )
        try:
            with open(output_name, "a") as op:
                op.write("\nVelocities\n\n")
            data[["id", "vx", "vy", "vz"]].to_csv(
                output_name, header=None, index=False, sep=" ", mode="a", na_rep="nan"
            )
        except Exception:
            pass

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

        CentroSymmetryPara = CentroSymmetryParameter(
            N, self.pos, self.box, self.boundary
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
            rc, nbin, rho, self.verlet_list, self.distance_list
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

    def cal_adaptive_common_neighbor_analysis(self):

        AdaptiveCommonNeighborAnalysi = AdaptiveCommonNeighborAnalysis(
            self.pos, self.box, self.boundary
        )
        AdaptiveCommonNeighborAnalysi.compute()
        self.data["acna"] = AdaptiveCommonNeighborAnalysi.pattern

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
