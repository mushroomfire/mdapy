# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.
import gzip
import numpy as np
import pandas as pd
import polars as pl
import os
import shutil
from tempfile import mkdtemp

try:
    from .pigz import compress_file
except Exception:
    from pigz import compress_file


class SaveFile:
    @staticmethod
    def write_data(
        output_name,
        box,
        boundary,
        data=None,
        pos=None,
        type_list=None,
        data_format="atomic",
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

        assert data_format in [
            "atomic",
            "charge",
        ], "Unrecgonized data format. Only support atomic and charge."
        if isinstance(data, pd.DataFrame):
            for col in ["id", "type", "x", "y", "z"]:
                assert col in data.columns
        else:
            assert pos.shape[1] == 3
            if type_list is None:
                type_list = np.ones(pos.shape[0], int)
            else:
                assert len(type_list) == pos.shape[0]
                type_list = np.array(type_list, int)
            data = pd.DataFrame(
                {
                    "id": np.arange(pos.shape[0]) + 1,
                    "type": type_list,
                    "x": pos[:, 0],
                    "y": pos[:, 1],
                    "z": pos[:, 2],
                }
            )

        with open(output_name, "wb") as op:
            op.write("# LAMMPS data file written by mdapy.\n\n".encode())
            op.write(
                f"{data.shape[0]} atoms\n{data['type'].max()} atom types\n\n".encode()
            )
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
            op.write(rf"Atoms # {data_format}".encode())
            op.write("\n\n".encode())

            if data_format == "atomic":
                table = pl.DataFrame(data[["id", "type", "x", "y", "z"]])
            elif data_format == "charge":
                if "q" not in data.columns:
                    table = pl.DataFrame(data[["id", "type", "x", "y", "z"]])
                    table.insert_at_idx(2, pl.Series("q", np.zeros(data.shape[0])))
                else:
                    table = pl.DataFrame(data[["id", "type", "q", "x", "y", "z"]])
            table.write_csv(op, separator=" ", has_header=False)

            if "vx" in data.columns:
                op.write("\nVelocities\n\n".encode())
                table = pl.DataFrame(data[["id", "vx", "vy", "vz"]])
                table.write_csv(op, separator=" ", has_header=False)

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
        if isinstance(data, pd.DataFrame):
            for col in ["id", "type", "x", "y", "z"]:
                assert col in data.columns
        else:
            assert pos.shape[1] == 3
            if type_list is None:
                type_list = np.ones(pos.shape[0], int)
            else:
                assert len(type_list) == pos.shape[0]
                type_list = np.array(type_list, int)
            data = pd.DataFrame(
                {
                    "id": np.arange(pos.shape[0]) + 1,
                    "type": type_list,
                    "x": pos[:, 0],
                    "y": pos[:, 1],
                    "z": pos[:, 2],
                }
            )

        if compress:
            path, name = os.path.split(output_name)
            if name.split(".")[-1] == "gz":
                name = name[:-3]
            temp_dir = mkdtemp()
            output_name = os.path.join(temp_dir, name)
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
                op.write(f"ITEM: BOX BOUNDS xy xz yz pp pp pp\n".encode())
                op.write(f"{xlo_bound} {xhi_bound} {xy}\n".encode())
                op.write(f"{ylo_bound} {yhi_bound} {xz}\n".encode())
                op.write(f"{zlo_bound} {zhi_bound} {yz}\n".encode())
            else:
                op.write(f"ITEM: BOX BOUNDS pp pp pp\n".encode())
                op.write(f"{xlo} {xhi}\n".encode())
                op.write(f"{ylo} {yhi}\n".encode())
                op.write(f"{zlo} {zhi}\n".encode())
            col_name = "ITEM: ATOMS " + " ".join(data.columns) + " \n"
            op.write(col_name.encode())

            pl.DataFrame(data).write_csv(op, separator=" ", has_header=False)
        if compress:
            compress_file(output_name, os.path.join(path, name + ".gz"))
            shutil.rmtree(temp_dir)


class BuildSystem:
    @staticmethod
    def getformat(filename, fmt=None):
        if fmt is None:
            postfix = filename.split(".")[-1]
            if postfix == "gz":
                assert (
                    filename.split(".")[-2] == "dump"
                ), "Only support compressed dump file."
                fmt = "dump.gz"
            else:
                fmt = postfix
        assert fmt in ["data", "lmp", "dump", "dump.gz"]
        return fmt

    @classmethod
    def fromfile(cls, filename, fmt):
        if fmt in ["dump", "dump.gz"]:
            return cls.read_dump(filename, fmt)
        elif fmt in ["data", "lmp"]:
            return cls.read_data(filename)

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
        data = pd.DataFrame(
            np.c_[np.arange(pos.shape[0]) + 1, type_list, pos],
            columns=["id", "type", "x", "y", "z"],
        )
        if vel is not None:
            assert vel.shape == pos.shape
            data[["vx", "vy", "vz"]] = vel
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
        return data, box, boundary

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

        row += 2  # Coordination part
        if data_head[-1].split()[-1] == "atomic":
            col_names = ["id", "type", "x", "y", "z"]
        elif data_head[-1].split()[-1] == "charge":
            col_names = ["id", "type", "q", "x", "y", "z"]
        else:
            with open(filename) as op:
                for _ in range(row):
                    op.readline()
                line = op.readline()
            if len(line.split()) == 5:
                col_names = ["id", "type", "x", "y", "z"]
            elif len(line.split()) == 6:
                col_names = ["id", "type", "q", "x", "y", "z"]
            else:
                raise "Unrecgonized data format. Only support atomic and charge."

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
        data = data.to_pandas()
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

        data = pl.read_csv(
            filename,
            separator=" ",
            skip_rows=9,
            new_columns=col_names,
            columns=range(len(col_names)),
            has_header=False,
            truncate_ragged_lines=True,
        ).to_pandas()

        return data, box, boundary, timestep
