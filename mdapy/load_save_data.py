# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.
import gzip
import numpy as np
import polars as pl
import os
import shutil
from tempfile import mkdtemp

try:
    from .pigz import compress_file
    from .tool_function import atomic_numbers, atomic_masses
except Exception:
    from pigz import compress_file
    from tool_function import atomic_numbers, atomic_masses


class SaveFile:
    @staticmethod
    def write_xyz(output_name, box, data, boundary, classical=False):
        assert isinstance(output_name, str)
        assert isinstance(box, np.ndarray)
        assert box.shape == (3, 2) or box.shape == (4, 3)
        if box.shape == (3, 2):
            new_box = np.zeros((4, 3), dtype=box.dtype)
            new_box[0, 0], new_box[1, 1], new_box[2, 2] = box[:, 1] - box[:, 0]
            new_box[-1] = box[:, 0]
        else:
            assert box[0, 1] == 0
            assert box[0, 2] == 0
            assert box[1, 2] == 0
            new_box = box
        assert isinstance(data, pl.DataFrame)
        for col in ["x", "y", "z"]:
            assert col in data.columns, f"data must contain {col}."
        assert (
            "type" in data.columns or "type_name" in data.columns
        ), f"data must contain type or type_name."

        if classical:
            with open(output_name, "wb") as op:
                op.write(f"{data.shape[0]}\n".encode())
                op.write("Classical XYZ file written by mdapy.\n".encode())
                if "type_name" in data.columns:
                    data.select("type_name", "x", "y", "z").write_csv(
                        op, separator=" ", include_header=False
                    )
                else:
                    data.select("type", "x", "y", "z").write_csv(
                        op, separator=" ", include_header=False
                    )
        else:
            properties = []
            for name, dtype in zip(data.columns, data.dtypes):
                if dtype in pl.INTEGER_DTYPES:
                    ptype = "I"
                elif dtype in pl.FLOAT_DTYPES:
                    ptype = "R"
                elif dtype == pl.Utf8:
                    ptype = "S"
                else:
                    raise f"Unrecognized data type {dtype}."
                if name == "type_name":
                    properties.append(f"species:{ptype}:1")
                else:
                    properties.append(f"{name}:{ptype}:1")
            properties_str = "Properties=" + ":".join(properties).replace(
                "x:R:1:y:R:1:z:R:1", "pos:R:3"
            )

            if "vx:R:1:vy:R:1:vz:R:1" in properties_str:
                properties_str = properties_str.replace(
                    "vx:R:1:vy:R:1:vz:R:1", "velo:R:3"
                )
            if "fx:R:1:fy:R:1:fz:R:1" in properties_str:
                properties_str = properties_str.replace(
                    "fx:R:1:fy:R:1:fz:R:1", "force:R:3"
                )

            lattice_str = (
                "Lattice="
                + '"'
                + " ".join(new_box[:-1].flatten().astype(str).tolist())
                + '"'
            )
            pbc_str = (
                "pbc="
                + '"'
                + " ".join(["T" if i == 1 else "F" for i in boundary])
                + '"'
            )
            origin_str = 'Origin="' + " ".join(new_box[-1].astype(str).tolist()) + '"'
            comments = (
                lattice_str + " " + properties_str + " " + pbc_str + " " + origin_str
            )
            with open(output_name, "wb") as op:
                op.write(f"{data.shape[0]}\n".encode())
                if "type_name" in data.columns and "type" in data.columns:
                    if "type:I:1:" in comments:
                        comments = comments.replace("type:I:1:", "")
                    if ":type:I:1" in comments:
                        comments = comments.replace(":type:I:1", "")
                    op.write(f"{comments}\n".encode())
                    data.select(pl.all().exclude("type")).write_csv(
                        op, separator=" ", include_header=False
                    )
                else:
                    op.write(f"{comments}\n".encode())
                    data.write_csv(op, separator=" ", include_header=False)

    @staticmethod
    def write_cif(output_name, box, data, type_name=None):
        assert isinstance(output_name, str)
        assert isinstance(box, np.ndarray)
        assert box.shape == (3, 2) or box.shape == (4, 3)
        old_box = box.copy()
        if box.shape == (3, 2):
            box = np.zeros((3, 3), dtype=old_box.dtype)
            box[0, 0], box[1, 1], box[2, 2] = old_box[:, 1] - old_box[:, 0]
        else:
            box = old_box[:-1]

        la, lb, lc = np.linalg.norm(box, axis=0)
        xy, xz, yz = box[1, 0], box[2, 0], box[2, 1]
        alpha = np.rad2deg(np.arccos((xy * xz + box[1, 1] * yz) / (lb * lc)))
        beta = np.rad2deg(np.arccos(xz / lc))
        gamma = np.rad2deg(np.arccos(xy / lb))

        assert isinstance(data, pl.DataFrame)
        for col in ["type", "x", "y", "z"]:
            assert col in data.columns, f"data must contain {col}."
        xlo = max(data["x"].min(), 0.0)
        ylo = max(data["y"].min(), 0.0)
        zlo = max(data["z"].min(), 0.0)
        data = data.sort("type").with_columns(
            pl.col("x") - xlo,
            pl.col("y") - ylo,
            pl.col("z") - zlo,
        )

        Ntype = data["type"].max()
        res = data.group_by("type", maintain_order=True).count()
        res = dict(zip(res[:, 0], res[:, 1]))
        index = []
        for i in res.keys():
            if i <= Ntype:
                index.append(np.arange(1, res[i] + 1))
        index = np.concatenate(index)
        if type_name is not None:
            assert (
                len(type_name) >= Ntype
            ), f"type list should contain more than {Ntype} elements."
            data = data.with_columns(
                pl.col("type")
                .replace(dict(enumerate(type_name, start=1)))
                .alias("type"),
                _a=index,
            ).with_columns((pl.col("type") + pl.col("_a").cast(str)).alias("index"))
            index = data["index"]
        else:
            if "type_name" in data.columns:
                index = data.with_columns(_a=index).with_columns(
                    (pl.col("type_name") + pl.col("_a").cast(str)).alias("index")
                )["index"]

        new_pos = np.dot(np.c_[data["x"], data["y"], data["z"]], np.linalg.pinv(box))
        data = data.with_columns(
            pl.lit(new_pos[:, 0]).alias("x"),
            pl.lit(new_pos[:, 1]).alias("y"),
            pl.lit(new_pos[:, 2]).alias("z"),
        ).select("type", index, "x", "y", "z")
        with open(output_name, "wb") as op:
            op.write("data_image0\n".encode())

            op.write(f"_cell_length_a {la}\n".encode())
            op.write(f"_cell_length_b {lb}\n".encode())
            op.write(f"_cell_length_c {lc}\n".encode())
            op.write(f"_cell_angle_alpha {alpha}\n".encode())
            op.write(f"_cell_angle_beta {beta}\n".encode())
            op.write(f"_cell_angle_gamma {gamma}\n\n".encode())

            op.write("loop_\n".encode())
            op.write("_atom_site_type_symbol\n".encode())
            op.write("_atom_site_label\n".encode())
            op.write("_atom_site_fract_x\n".encode())
            op.write("_atom_site_fract_y\n".encode())
            op.write("_atom_site_fract_z\n".encode())

            data.write_csv(op, separator=" ", include_header=False)

    @staticmethod
    def write_POSCAR(
        output_name,
        box,
        data,
        type_name=None,
        reduced_pos=False,
        selective_dynamics=False,
        save_velocity=False,
    ):
        assert isinstance(output_name, str)
        assert isinstance(box, np.ndarray)
        assert box.shape == (3, 2) or box.shape == (4, 3)
        if box.shape == (3, 2):
            new_box = np.zeros((3, 3), dtype=box.dtype)
            new_box[0, 0], new_box[1, 1], new_box[2, 2] = box[:, 1] - box[:, 0]
        else:
            assert box[0, 1] == 0
            assert box[0, 2] == 0
            assert box[1, 2] == 0
            new_box = box[:-1]
        assert isinstance(data, pl.DataFrame)
        for col in ["type", "x", "y", "z"]:
            assert col in data.columns, f"data must contain {col}."

        xlo = max(data["x"].min(), 0.0)
        ylo = max(data["y"].min(), 0.0)
        zlo = max(data["z"].min(), 0.0)
        data = data.sort("type").with_columns(
            pl.col("x") - xlo,
            pl.col("y") - ylo,
            pl.col("z") - zlo,
        )
        if reduced_pos:
            new_pos = np.dot(
                np.c_[data["x"], data["y"], data["z"]], np.linalg.pinv(new_box)
            )
            data = data.with_columns(
                pl.lit(new_pos[:, 0]).alias("x"),
                pl.lit(new_pos[:, 1]).alias("y"),
                pl.lit(new_pos[:, 2]).alias("z"),
            )

        type_list = data.group_by("type", maintain_order=True).count()["count"]
        if type_name is not None:
            assert len(type_name) == type_list.shape[0]
        else:
            if "type_name" in data.columns:
                type_name = data["type_name"].unique(maintain_order=True)
                assert len(type_name) == type_list.shape[0]

        if selective_dynamics:
            for col in ["sdx", "sdy", "sdz"]:
                assert (
                    col in data.columns
                ), f"data mush contain {col} if you set selective_dynamics to True."

        if save_velocity:
            for col in ["vx", "vy", "vz"]:
                assert (
                    col in data.columns
                ), f"data mush contain {col} if you set save_velocity to True."

        with open(output_name, "wb") as op:
            op.write("# VASP POSCAR file written by mdapy.\n".encode())
            op.write("1.0000000000\n".encode())
            op.write("{:.10f} {:.10f} {:.10f}\n".format(*new_box[0]).encode())
            op.write("{:.10f} {:.10f} {:.10f}\n".format(*new_box[1]).encode())
            op.write("{:.10f} {:.10f} {:.10f}\n".format(*new_box[2]).encode())
            if type_name is not None:
                for aname in type_name:
                    op.write(f"{aname} ".encode())
                op.write("\n".encode())
            for atype in type_list:
                op.write(f"{atype} ".encode())
            op.write("\n".encode())
            if selective_dynamics:
                op.write("Selective dynamics\n".encode())
            if reduced_pos:
                op.write("Direct\n".encode())
            else:
                op.write("Cartesian\n".encode())
            if selective_dynamics:
                data.select(["x", "y", "z", "sdx", "sdy", "sdz"]).write_csv(
                    op, separator=" ", include_header=False, float_precision=10
                )
            else:
                data.select(["x", "y", "z"]).write_csv(
                    op, separator=" ", include_header=False, float_precision=10
                )
            if save_velocity:
                op.write("Cartesian\n".encode())
                data.select(["vx", "vy", "vz"]).write_csv(
                    op, separator=" ", include_header=False, float_precision=10
                )

    @staticmethod
    def write_data(
        output_name,
        box,
        data=None,
        pos=None,
        type_list=None,
        num_type=None,
        type_name=None,
        data_format="atomic",
    ):
        assert isinstance(output_name, str)
        assert isinstance(box, np.ndarray)
        assert box.shape == (3, 2) or box.shape == (4, 3)
        if box.shape == (3, 2):
            new_box = np.zeros((4, 3), dtype=box.dtype)
            new_box[0, 0], new_box[1, 1], new_box[2, 2] = box[:, 1] - box[:, 0]
            new_box[-1] = box[:, 0]
        else:
            assert box[0, 1] == 0
            assert box[0, 2] == 0
            assert box[1, 2] == 0
            new_box = box

        assert data_format in [
            "atomic",
            "charge",
        ], "Unrecgonized data format. Only support atomic and charge."
        if isinstance(data, pl.DataFrame):
            for col in ["id", "type", "x", "y", "z"]:
                assert col in data.columns
        else:
            assert pos.shape[1] == 3
            if type_list is None:
                type_list = np.ones(pos.shape[0], int)
            else:
                assert len(type_list) == pos.shape[0]
                type_list = np.array(type_list, int)
            data = pl.DataFrame(
                {
                    "id": np.arange(pos.shape[0]) + 1,
                    "type": type_list,
                    "x": pos[:, 0],
                    "y": pos[:, 1],
                    "z": pos[:, 2],
                }
            )
        if num_type is None:
            num_type = data["type"].max()
        else:
            assert isinstance(num_type, int)
            assert (
                num_type >= data["type"].max()
            ), f"num_type should be >= {data['type'].max()}."

        if type_name is not None:
            assert (
                len(type_name) >= num_type
            ), f"type_name should at least contain {num_type} elements."
            num_type = len(type_name)
            for i in type_name:
                assert i in atomic_numbers.keys(), f"Unrecognized element name {i}."

        with open(output_name, "wb") as op:
            op.write("# LAMMPS data file written by mdapy.\n\n".encode())
            op.write(f"{data.shape[0]} atoms\n{num_type} atom types\n\n".encode())
            op.write(
                f"{new_box[-1, 0]} {new_box[-1, 0]+new_box[0, 0]} xlo xhi\n".encode()
            )
            op.write(
                f"{new_box[-1, 1]} {new_box[-1, 1]+new_box[1, 1]} ylo yhi\n".encode()
            )
            op.write(
                f"{new_box[-1, 2]} {new_box[-1, 2]+new_box[2, 2]} zlo zhi\n".encode()
            )
            xy, xz, yz = new_box[1, 0], new_box[2, 0], new_box[2, 1]
            if xy != 0 or xz != 0 or yz != 0:  # Triclinic box
                op.write(f"{xy} {xz} {yz} xy xz yz\n".encode())
            op.write("\n".encode())
            if type_name is not None:
                op.write("Masses\n\n".encode())
                for i, j in enumerate(type_name, start=1):
                    op.write(f"{i} {atomic_masses[atomic_numbers[j]]} # {j}\n".encode())
                op.write("\n".encode())

            op.write(rf"Atoms # {data_format}".encode())
            op.write("\n\n".encode())

            if data_format == "atomic":
                table = data.select(["id", "type", "x", "y", "z"])

            elif data_format == "charge":
                if "q" not in data.columns:
                    table = data.select(["id", "type", "x", "y", "z"])
                    table.insert_at_idx(2, pl.Series("q", np.zeros(data.shape[0])))
                else:
                    table = data.select(["id", "type", "q", "x", "y", "z"])
            table.write_csv(op, separator=" ", include_header=False)

            if "vx" in data.columns and "vy" in data.columns and "vz" in data.columns:
                op.write("\nVelocities\n\n".encode())
                table = data.select(["id", "vx", "vy", "vz"])
                table.write_csv(op, separator=" ", include_header=False)

    @staticmethod
    def write_dump(
        output_name,
        box,
        boundary,
        data=None,
        pos=None,
        type_list=None,
        timestep=0,
        compress=False,
    ):
        assert isinstance(output_name, str)
        assert isinstance(box, np.ndarray)
        assert len(boundary) == 3
        assert box.shape == (3, 2) or box.shape == (4, 3)
        if box.shape == (3, 2):
            new_box = np.zeros((4, 3), dtype=box.dtype)
            new_box[0, 0], new_box[1, 1], new_box[2, 2] = box[:, 1] - box[:, 0]
            new_box[-1] = box[:, 0]
        else:
            assert box[0, 1] == 0
            assert box[0, 2] == 0
            assert box[1, 2] == 0
            new_box = box
        if isinstance(data, pl.DataFrame):
            for col in ["id", "type", "x", "y", "z"]:
                assert col in data.columns
        else:
            assert pos.shape[1] == 3
            if type_list is None:
                type_list = np.ones(pos.shape[0], int)
            else:
                assert len(type_list) == pos.shape[0]
                type_list = np.array(type_list, int)
            data = pl.DataFrame(
                {
                    "id": np.arange(pos.shape[0]) + 1,
                    "type": type_list,
                    "x": pos[:, 0],
                    "y": pos[:, 1],
                    "z": pos[:, 2],
                }
            )
        data = data.select(pl.selectors.by_dtype(pl.NUMERIC_DTYPES))

        if compress:
            path, name = os.path.split(output_name)
            if name.split(".")[-1] == "gz":
                name = name[:-3]
            temp_dir = mkdtemp()
            output_name = os.path.join(temp_dir, name)
        boundary2str = ["pp" if i == 1 else "ss" for i in boundary]
        with open(output_name, "wb") as op:
            op.write(f"ITEM: TIMESTEP\n{timestep}\n".encode())
            op.write("ITEM: NUMBER OF ATOMS\n".encode())
            op.write(f"{data.shape[0]}\n".encode())
            xlo, ylo, zlo = new_box[3]
            xhi, yhi, zhi = (
                xlo + new_box[0, 0],
                ylo + new_box[1, 1],
                zlo + new_box[2, 2],
            )
            xy, xz, yz = new_box[1, 0], new_box[2, 0], new_box[2, 1]
            if xy != 0 or xz != 0 or yz != 0:  # Triclinic box
                xlo_bound = xlo + min(0.0, xy, xz, xy + xz)
                xhi_bound = xhi + max(0.0, xy, xz, xy + xz)
                ylo_bound = ylo + min(0.0, yz)
                yhi_bound = yhi + max(0.0, yz)
                zlo_bound = zlo
                zhi_bound = zhi
                op.write(
                    f"ITEM: BOX BOUNDS xy xz yz {boundary2str[0]} {boundary2str[1]} {boundary2str[2]}\n".encode()
                )
                op.write(f"{xlo_bound} {xhi_bound} {xy}\n".encode())
                op.write(f"{ylo_bound} {yhi_bound} {xz}\n".encode())
                op.write(f"{zlo_bound} {zhi_bound} {yz}\n".encode())
            else:
                op.write(
                    f"ITEM: BOX BOUNDS {boundary2str[0]} {boundary2str[1]} {boundary2str[2]}\n".encode()
                )
                op.write(f"{xlo} {xhi}\n".encode())
                op.write(f"{ylo} {yhi}\n".encode())
                op.write(f"{zlo} {zhi}\n".encode())
            col_name = "ITEM: ATOMS " + " ".join(data.columns) + " \n"
            op.write(col_name.encode())

            data.write_csv(op, separator=" ", include_header=False)
        if compress:
            compress_file(output_name, os.path.join(path, name + ".gz"))
            shutil.rmtree(temp_dir)


class BuildSystem:
    @staticmethod
    def getformat(filename, fmt=None):
        if fmt is None:
            postfix = os.path.split(filename)[-1].split(".")[-1]
            if postfix == "gz":
                assert (
                    filename.split(".")[-2] == "dump"
                ), "Only support compressed dump file."
                fmt = "dump.gz"
            else:
                fmt = postfix
        assert fmt in ["data", "lmp", "dump", "dump.gz", "POSCAR", "xyz", "cif"]
        return fmt

    @classmethod
    def fromfile(cls, filename, fmt):
        if fmt in ["dump", "dump.gz"]:
            return cls.read_dump(filename, fmt)
        elif fmt in ["data", "lmp"]:
            return cls.read_data(filename)
        elif fmt == "POSCAR":
            return cls.read_POSCAR(filename)
        elif fmt == "xyz":
            return cls.read_xyz(filename)
        elif fmt == "cif":
            return cls.read_cif(filename)

    @staticmethod
    def fromarray(pos, box, boundary, vel, type_list):
        assert pos.shape[1] == 3
        assert box.shape == (3, 2) or box.shape == (4, 3)
        assert len(boundary) == 3
        if box.shape == (3, 2):
            new_box = np.zeros((4, 3), dtype=box.dtype)
            new_box[0, 0], new_box[1, 1], new_box[2, 2] = box[:, 1] - box[:, 0]
            new_box[-1] = box[:, 0]
        else:
            new_box = box

        if type_list is None:
            type_list = np.ones(pos.shape[0], int)
        assert len(type_list) == pos.shape[0]
        assert type_list.dtype in [
            np.int32,
            np.int64,
        ], "type_list should be int32 or int64"

        data = pl.DataFrame(
            {
                "id": np.arange(pos.shape[0]) + 1,
                "type": type_list,
                "x": pos[:, 0],
                "y": pos[:, 1],
                "z": pos[:, 2],
            }
        )
        if vel is not None:
            assert vel.shape == pos.shape
            data = data.with_columns(
                pl.lit(vel[:, 0]).alias("vx"),
                pl.lit(vel[:, 1]).alias("vy"),
                pl.lit(vel[:, 2]).alias("vz"),
            )

        return data, box, boundary

    @staticmethod
    def fromdata(data, box, boundary):
        assert "x" in data.columns
        assert "y" in data.columns
        assert "z" in data.columns
        assert len(boundary) == 3
        assert box.shape == (3, 2) or box.shape == (4, 3)
        if box.shape == (3, 2):
            new_box = np.zeros((4, 3), dtype=box.dtype)
            new_box[0, 0], new_box[1, 1], new_box[2, 2] = box[:, 1] - box[:, 0]
            new_box[-1] = box[:, 0]
        else:
            new_box = box
        return data, new_box, boundary

    @staticmethod
    def read_xyz(filename):
        head = []
        with open(filename) as op:
            for i in range(3):
                head.append(op.readline())
        natom = int(head[0].split()[0])
        classical = True
        if "Lattice" in head[1] and "Properties" in head[1]:
            classical = False
            info = head[1]

            if "pbc=" in info:
                pindex = info.index("pbc=") + len("pbc=")
                boundary = [
                    1 if i == "T" or i == "1" else 0
                    for i in info[pindex + 1 : pindex + 6].split()
                ]
            else:
                boundary = [1, 1, 1]

            bindex = info.index("Lattice=") + len("Lattice=")
            try:
                box = np.array(info[bindex:].split("'")[1].split(), float).reshape(3, 3)
            except Exception:
                box = np.array(info[bindex:].split('"')[1].split(), float).reshape(3, 3)
            assert (
                box[0, 1] == box[0, 2] == box[1, 2] == 0
            ), "Only support lammps style box! box[0, 1]==box[0, 2]==box[1, 2]==0."
            if "Origin=" in info:
                oindex = info.index("Origin=") + len("Origin=")
                origin = np.expand_dims(
                    np.array(info[oindex:].split('"')[1].split(), float), axis=0
                )
                box = np.r_[box, origin]

            pindex = info.index("Properties=") + len("Properties=")
            content = info[pindex:].split()[0].split(":")
            i = 0
            columns = []
            schema = {}
            while i < len(content) - 2:
                n_col = int(content[i + 2])
                if content[i + 1] == "S":
                    dtype = pl.Utf8
                elif content[i + 1] == "R":
                    dtype = pl.Float64
                elif content[i + 1] == "I":
                    dtype = pl.Int64
                else:
                    raise f"Unrecognized type {content[i+1]}."

                if (
                    content[i] == "pos"
                    and content[i + 1] == "R"
                    and content[i + 2] == "3"
                ):
                    columns.extend(["x", "y", "z"])
                    schema["x"] = dtype
                    schema["y"] = dtype
                    schema["z"] = dtype
                elif (
                    content[i] in ["species", "type_name", "element"]
                    and content[i + 2] == "1"
                ):
                    columns.append("type_name")
                    schema["type_name"] = dtype
                elif (
                    content[i] == "velo"
                    and content[i + 1] == "R"
                    and content[i + 2] == "3"
                ):
                    columns.extend(["vx", "vy", "vz"])
                    schema["vx"] = dtype
                    schema["vy"] = dtype
                    schema["vz"] = dtype
                elif (
                    content[i] in ["force", "forces"]
                    and content[i + 1] == "R"
                    and content[i + 2] == "3"
                ):
                    columns.extend(["fx", "fy", "fz"])
                    schema["fx"] = dtype
                    schema["fy"] = dtype
                    schema["fz"] = dtype
                else:
                    if n_col > 1:
                        for j in range(n_col):
                            columns.append(content[i] + f"_{j}")
                            schema[content[i] + f"_{j}"] = dtype
                    else:
                        columns.append(content[i])
                        schema[content[i]] = dtype
                i += 3
        else:
            boundary = [0, 0, 0]
            columns = ["type_name", "x", "y", "z"]
            schema = {
                "type_name": pl.Utf8,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
            }

        multi_space = False
        if head[-1].count(" ") != len(columns) - 1:
            multi_space = True
        if classical:
            if multi_space:
                type_name, x, y, z = [], [], [], []
                with open(filename) as op:
                    op.readline()  # skip head
                    op.readline()
                    for i in range(natom):
                        content = op.readline().split()
                        type_name.append(content[0])
                        x.append(content[1])
                        y.append(content[2])
                        z.append(content[3])
                df = pl.DataFrame(
                    {"type_name": type_name, "x": x, "y": y, "z": z}, schema=schema
                )
            else:
                df = pl.read_csv(
                    filename,
                    separator=" ",
                    schema=schema,
                    skip_rows=2,
                    new_columns=columns,
                    columns=range(len(columns)),
                    has_header=False,
                    truncate_ragged_lines=True,
                )
            if df["type_name"][0].isdigit():
                df = df.with_columns(pl.col("type_name").cast(int)).rename(
                    {"type_name": "type"}
                )
            else:
                type_name2type = {
                    j: i + 1
                    for i, j in enumerate(df["type_name"].unique(maintain_order=True))
                }
                df = df.with_columns(
                    pl.col("type_name")
                    .replace(type_name2type, default=None)
                    .alias("type")
                )
            df = df.with_row_count("id", offset=1)
            coor = df.select("x", "y", "z")
            box = np.r_[
                np.eye(3) * (coor.max() - coor.min()).to_numpy(), coor.min().to_numpy()
            ]
        else:
            if multi_space:
                data = {}
                for i in columns:
                    data[i] = []
                with open(filename) as op:
                    op.readline()  # skip head
                    op.readline()
                    for i in range(natom):
                        for key, j in zip(columns, op.readline().split()):
                            data[key].append(j)
                df = pl.DataFrame(data, schema=schema)
            else:
                df = pl.read_csv(
                    filename,
                    separator=" ",
                    schema=schema,
                    skip_rows=2,
                    new_columns=columns,
                    columns=range(len(columns)),
                    has_header=False,
                    truncate_ragged_lines=True,
                )

            if "Origin=" not in info:
                box = np.r_[box, df.select("x", "y", "z").min().to_numpy()]
            if "id" not in df.columns:
                df = df.with_row_count("id", offset=1)
            if "type" not in df.columns:
                if "type_name" in df.columns:
                    if df["type_name"][0].isdigit():
                        df = df.with_columns(pl.col("type_name").cast(int)).rename(
                            {"type_name": "type"}
                        )
                    else:
                        name_list = df["type_name"].unique(maintain_order=True)
                        name2type = {j: i + 1 for i, j in enumerate(name_list)}
                        df = df.with_columns(
                            pl.col("type_name")
                            .replace(name2type, default=None)
                            .alias("type")
                        )
                else:
                    df = df.with_columns(pl.lit(1).alias("type"))
        return df, box, boundary

    @staticmethod
    def read_cif(filename):
        with open(filename) as op:
            file = op.readlines()

        a, b, c, alpha, beta, gamma = None, None, None, None, None, None
        head = []
        data = []
        for line in file:
            content = line.split()
            if len(content) > 0:
                if content[0] == "_cell_length_a":
                    a = float(content[1])
                if content[0] == "_cell_length_b":
                    b = float(content[1])
                if content[0] == "_cell_length_c":
                    c = float(content[1])
                if content[0] == "_cell_angle_alpha":
                    alpha = np.deg2rad(float(content[1]))
                if content[0] == "_cell_angle_beta":
                    beta = np.deg2rad(float(content[1]))
                if content[0] == "_cell_angle_gamma":
                    gamma = np.deg2rad(float(content[1]))
                if content[0].startswith("_atom_site"):
                    head.append(content[0])
                if len(content) >= 5 and len(content) == len(head):
                    data.append(content)

        for i in [a, b, c, alpha, beta, gamma]:
            assert (
                i is not None
            ), "Box information missing. Check file with a, b, c, alpha, beta, gamma."

        lx = a
        xy = b * np.cos(gamma)
        xz = c * np.cos(beta)
        ly = (b**2 - xy**2) ** 0.5
        yz = (b * c * np.cos(alpha) - xy * xz) / ly
        lz = (c**2 - xz**2 - yz**2) ** 0.5

        box = np.zeros((4, 3))
        box[0, 0] = lx
        box[1, 0] = xy
        box[1, 1] = ly
        box[2, 0] = xz
        box[2, 1] = xy
        box[2, 2] = lz

        box[np.abs(box) < 1e-6] = 0.0

        data = pl.from_numpy(np.array(data), schema=head)

        if "_atom_site_fract_x" in head:
            data = data.with_columns(
                pl.col("_atom_site_type_symbol").alias("type"),
                pl.col("_atom_site_fract_x").cast(pl.Float64).alias("x"),
                pl.col("_atom_site_fract_y").cast(pl.Float64).alias("y"),
                pl.col("_atom_site_fract_z").cast(pl.Float64).alias("z"),
            ).select("type", "x", "y", "z")
        else:
            data = data.with_columns(
                pl.col("_atom_site_type_symbol").alias("type"),
                pl.col("_atom_site_x").cast(pl.Float64).alias("x"),
                pl.col("_atom_site_y").cast(pl.Float64).alias("y"),
                pl.col("_atom_site_z").cast(pl.Float64).alias("z"),
            ).select("type", "x", "y", "z")

        if data["type"][0].isdigit():
            data = data.with_columns(pl.col("type").cast(int))
        else:
            res = data.group_by("type", maintain_order=True).count()["type"]
            species2type = dict([[j, i] for i, j in enumerate(res, start=1)])
            data = data.with_columns(pl.col("type").alias("type_name")).with_columns(
                pl.col("type_name")
                .replace(species2type, return_dtype=pl.Int64)
                .alias("type")
            )

        if "_atom_site_fract_x" in head:
            new_pos = data.select("x", "y", "z").to_numpy() @ box[:-1]
            data = data.with_columns(
                pl.lit(new_pos[:, 0]).alias("x"),
                pl.lit(new_pos[:, 1]).alias("y"),
                pl.lit(new_pos[:, 2]).alias("z"),
            )

        data = data.with_row_count("id", offset=1)

        return data, box, [1, 1, 1]

    @staticmethod
    def read_POSCAR(filename):
        with open(filename) as op:
            file = op.readlines()
        scale = float(file[1].strip())
        need_rotation = False
        box = np.array([i.split() for i in file[2:5]], float) * scale

        if box[0, 1] != 0 or box[0, 2] != 0 or box[1, 2] != 0:
            old_box = box.copy()
            ax = np.linalg.norm(box[0])
            bx = box[1] @ (box[0] / ax)
            by = np.sqrt(np.linalg.norm(box[1]) ** 2 - bx**2)
            cx = box[2] @ (box[0] / ax)
            cy = (box[1] @ box[2] - bx * cx) / by
            cz = np.sqrt(np.linalg.norm(box[2]) ** 2 - cx**2 - cy**2)
            box = np.array([[ax, bx, cx], [0, by, cy], [0, 0, cz]]).T
            need_rotation = True
            rotation = np.linalg.solve(old_box, box)

        row = 5
        type_list, type_name_list = [], []
        if file[5].strip()[0].isdigit():
            for atype, num in enumerate(file[5].split()):
                type_list.extend([atype + 1] * int(num))
            row += 1
        else:
            assert file[6].strip()[0].isdigit()
            content = file[5].split()
            name_dict = {}
            atype = 1
            for name in content:
                if name not in name_dict.keys():
                    name_dict[name] = atype
                    atype += 1
            for atype, num in enumerate(file[6].split()):
                t_name = content[atype]
                type_name_list.extend([t_name] * int(num))
                type_list.extend([name_dict[t_name]] * int(num))
            row += 2
        selective_dynamics = False
        if file[row].strip()[0] in ["S", "s"]:
            row += 1
            selective_dynamics = True
        natoms = len(type_list)
        pos, sd = [], []
        if selective_dynamics:
            for line in file[row + 1 : row + 1 + natoms]:
                content = line.split()
                pos.append(content[:3])
                sd.append(content[-3:])
            sd = np.array(sd)
        else:
            for line in file[row + 1 : row + 1 + natoms]:
                pos.append(line.split()[:3])
        pos = np.array(pos, float)
        if file[row][0] in ["C", "c", "K", "k"]:
            pos *= scale
            if need_rotation:
                pos = pos @ rotation
        else:
            pos = pos @ box

        row += natoms + 1
        vel = []
        if row <= len(file) - 1:
            if len(file[row].split()) > 0:
                if file[row].strip()[0] in ["L", "l"]:  # skip the lattice velocities
                    row += 8
        if row + 1 + natoms <= len(file):
            vel = np.array([i.split() for i in file[row + 1 : row + 1 + natoms]], float)
            if len(file[row].split()) > 0:
                if file[row].strip()[0] not in ["C", "c", "K", "k"]:
                    vel = vel @ box
                else:
                    if need_rotation:
                        vel = vel @ rotation

        data = {}
        data["id"] = np.arange(1, natoms + 1)
        data["type"] = type_list
        if len(type_name_list) > 0:
            data["type_name"] = type_name_list
        data["x"] = pos[:, 0]
        data["y"] = pos[:, 1]
        data["z"] = pos[:, 2]
        if len(sd) > 0:
            data["sdx"] = sd[:, 0]
            data["sdy"] = sd[:, 1]
            data["sdz"] = sd[:, 2]
        if len(vel) > 0:
            data["vx"] = vel[:, 0]
            data["vy"] = vel[:, 1]
            data["vz"] = vel[:, 2]
        data = pl.DataFrame(data)

        return data, np.r_[box, np.array([[0, 0, 0.0]])], [1, 1, 1]

    @staticmethod
    def read_data(filename):
        data_head = []
        box = np.zeros((4, 3))
        row = 0
        xy, xz, yz = 0, 0, 0
        with open(filename) as op:
            while True:
                line = op.readline()
                data_head.append(line)
                content = line.split()
                if len(content):
                    if content[-1] == "atoms":
                        N = int(content[0])
                    if len(content) >= 2:
                        if content[1] == "bond":
                            raise "Do not support bond style."
                    if content[-1] == "xhi":
                        xlo, xhi = float(content[0]), float(content[1])
                    if content[-1] == "yhi":
                        ylo, yhi = float(content[0]), float(content[1])
                    if content[-1] == "zhi":
                        zlo, zhi = float(content[0]), float(content[1])
                    if content[-1] == "yz":
                        xy, xz, yz = (
                            float(content[0]),
                            float(content[1]),
                            float(content[2]),
                        )
                    if content[0] == "Atoms":
                        line = op.readline()
                        data_head.append(line)
                        line = op.readline()
                        data_head.append(line)
                        row += 2  # Coordination part
                        break
                row += 1
        box = np.array(
            [
                [xhi - xlo, 0, 0],
                [xy, yhi - ylo, 0],
                [xz, yz, zhi - zlo],
                [xlo, ylo, zlo],
            ]
        )
        boundary = [1, 1, 1]

        if data_head[-3].split()[-1] == "atomic":
            col_names = ["id", "type", "x", "y", "z"]
        elif data_head[-3].split()[-1] == "charge":
            col_names = ["id", "type", "q", "x", "y", "z"]
        else:
            line = data_head[-1]
            if len(line.split()) == 5:
                col_names = ["id", "type", "x", "y", "z"]
            elif len(line.split()) == 6:
                col_names = ["id", "type", "q", "x", "y", "z"]
            else:
                raise "Unrecgonized data format. Only support atomic and charge."

        multi_space = False
        if data_head[-1].count(" ") != len(col_names) - 1:
            multi_space = True

        if multi_space:
            with open(filename) as op:
                file = op.readlines()

            data = np.array([i.split() for i in file[row : row + N]], float)
            data = pl.from_numpy(data, schema=col_names)
            data = data.with_columns(
                pl.col("id").cast(pl.Int64), pl.col("type").cast(pl.Int64)
            )

            try:
                row += N
                assert file[row + 1].split()[0].strip() == "Velocities"
                vel = np.array(
                    [i.split()[1:4] for i in file[row + 3 : row + 3 + N]], float
                )
                assert vel.shape[0] == data.shape[0]
                data = data.with_columns(vx=vel[:, 0], vy=vel[:, 1], vz=vel[:, 2])

            except Exception:
                pass
        else:
            data = pl.read_csv(
                filename,
                separator=" ",
                skip_rows=row,
                n_rows=N,
                new_columns=col_names,
                columns=range(len(col_names)),
                has_header=False,
                truncate_ragged_lines=True,
                ignore_errors=True,
            )

            row += N
            try:
                vel = pl.read_csv(
                    filename,
                    separator=" ",
                    skip_rows=row + 3,
                    new_columns=["vx", "vy", "vz"],
                    columns=range(1, 4),
                    has_header=False,
                    truncate_ragged_lines=True,
                    ignore_errors=True,
                )
                assert vel.shape[0] == data.shape[0]
                data = pl.concat([data, vel], how="horizontal")
            except Exception:
                pass

        return data, box, boundary

    @staticmethod
    def read_dump(filename, fmt):
        assert fmt in ["dump", "dump.gz"], "Only support dump or dump.gz format."
        dump_head = []
        if fmt == "dump":
            with open(filename) as op:
                for _ in range(9):
                    dump_head.append(op.readline())
        elif fmt == "dump.gz":
            with gzip.open(filename) as op:
                for _ in range(9):
                    dump_head.append(op.readline().decode())

        timestep = int(dump_head[1].strip())
        line = dump_head[4].split()
        boundary = [1 if i == "pp" else 0 for i in line[-3:]]
        if "xy" in line:
            xlo_bound, xhi_bound, xy = np.array(dump_head[5].split(), float)
            ylo_bound, yhi_bound, xz = np.array(dump_head[6].split(), float)
            zlo_bound, zhi_bound, yz = np.array(dump_head[7].split(), float)
            xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
            xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
            ylo = ylo_bound - min(0.0, yz)
            yhi = yhi_bound - max(0.0, yz)
            zlo = zlo_bound
            zhi = zhi_bound
            box = np.array(
                [
                    [xhi - xlo, 0, 0],
                    [xy, yhi - ylo, 0],
                    [xz, yz, zhi - zlo],
                    [xlo, ylo, zlo],
                ]
            )
        else:
            box = np.array([i.split()[:2] for i in dump_head[5:8]]).astype(float)
            xlo, xhi = np.array(dump_head[5].split(), float)
            ylo, yhi = np.array(dump_head[6].split(), float)
            zlo, zhi = np.array(dump_head[7].split(), float)
            box = np.array(
                [
                    [xhi - xlo, 0, 0],
                    [0, yhi - ylo, 0],
                    [0, 0, zhi - zlo],
                    [xlo, ylo, zlo],
                ]
            )
        col_names = dump_head[8].split()[2:]

        try:
            data = pl.read_csv(
                filename,
                separator=" ",
                skip_rows=9,
                new_columns=col_names,
                columns=range(len(col_names)),
                has_header=False,
                truncate_ragged_lines=True,
            )
        except Exception:
            data = pl.read_csv(
                filename,
                separator=" ",
                skip_rows=9,
                new_columns=col_names,
                columns=range(len(col_names)),
                has_header=False,
                truncate_ragged_lines=True,
                infer_schema_length=None,
            )

        if "xs" in data.columns:
            pos = data.select("xs", "ys", "zs").to_numpy() @ box[:-1]

            data = data.with_columns(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2]).select(
                pl.all().exclude("xs", "ys", "zs")
            )

        return data, box, boundary, timestep
