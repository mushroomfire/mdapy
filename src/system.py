import numpy as np
import pandas as pd
from neighbor import Neighbor


class System:
    def __init__(self, filename):
        self.filename = filename

        self.head, self.boundary, self.box, self.col_names = self.read_box()
        (
            self.data,
            self.N,
            self.pos,
            self.vel,
        ) = self.read_dump(self.col_names)
        self.if_neigh = False

    def read_box(self):
        head = []
        with open(self.filename) as op:
            for _ in range(9):
                head.append(op.readline())
        boundary = [1 if i == "pp" else 0 for i in head[4].split()[-3:]]
        box = np.array([i.split()[:2] for i in head[5:8]]).astype(float)
        col_names = head[8].split()[2:]
        return head, boundary, box, col_names

    def read_dump(self, col_names):

        data = pd.read_csv(
            self.filename,
            skiprows=9,
            index_col=False,
            header=None,
            sep=" ",
            names=col_names,
        )
        N = data.shape[0]

        pos = data[["x", "y", "z"]].values
        vel = data[["vx", "vy", "vz"]].values

        return data, N, pos, vel

    def write_dump(self, output_name=None, output_col=None):
        head, filename = self.head, self.filename
        if output_col is None:
            data = self.data
        else:
            data = self.data.loc[:, output_col]
        for dtype, name in zip(data.dtypes, data.columns):
            if dtype == "int64":
                data[name] = data[name].astype(np.int32)
            elif dtype == "float64":
                data[name] = data[name].astype(np.float32)
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
        for dtype, name in zip(data.dtypes, data.columns):
            if dtype == "int64":
                data[name] = data[name].astype(np.int32)
            elif dtype == "float64":
                data[name] = data[name].astype(np.float32)
        if output_name is None:
            output_name = self.filename[:-4] + "data"
        Ntype = len(np.unique(data["type"]))

        with open(output_name, "w") as op:
            op.write("# LAMMPS data file written by mdapy@HerrWu.\n\n")
            op.write(f"{self.N} atoms\n{Ntype} atom types\n\n")
            for i, j in zip(self.box, ["x", "y", "z"]):
                op.write(f"{i[0]} {i[1]} {j}lo {j}hi\n")
            op.write("\n")
            op.write(r"Atoms # atomic")
            op.write("\n\n")
        data[["id", "type", "x", "y", "z"]].to_csv(
            output_name, header=None, index=False, sep=" ", mode="a", na_rep="nan"
        )
        with open(output_name, "a") as op:
            op.write("\nVelocities\n\n")
        data[["id", "vx", "vy", "vz"]].to_csv(
            output_name, header=None, index=False, sep=" ", mode="a", na_rep="nan"
        )

    def build_neighbor(self, rc=5.0, max_neigh=50, exclude=True):
        self.Neighbor = Neighbor(
            self.pos, self.box, rc, self.boundary, max_neigh, exclude
        )
        self.Neighbor.compute()
        self.if_neigh = True
