# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from __future__ import annotations
from typing import Tuple, Union, Optional, Iterable, Dict, TYPE_CHECKING, Any, List
import numpy as np
import polars as pl
import os
import tempfile
from pathlib import Path
from mdapy.box import Box
from mdapy.data import atomic_numbers, atomic_masses
import re
import gzip
from mdapy.pigz import compress_file

if TYPE_CHECKING:
    from ase import Atoms
    from ovito.data import DataCollection


def _open_file(filename: str, mode: str = "r"):
    """
    Smart file opener that handles both regular and gzip files.

    Args:
        filename: Path to file
        mode: File mode ('r', 'rb', 'w', 'wb')

    Returns:
        File handle (automatically detects .gz)
    """
    if filename.endswith(".gz"):
        if "b" not in mode:
            mode = mode + "t"  # text mode for gzip
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)


def _save_with_compression(
    write_func, output_name: str, compress: bool = False, **kwargs
):
    """
    Helper function to save file with optional compression.

    Args:
        write_func: Function that writes the actual file
        output_name: Desired output filename
        compress: Whether to compress the output
        **kwargs: Arguments passed to write_func
    """
    if compress:
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=Path(output_name).suffix, delete=False
        ) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Write to temporary file
            write_func(tmp_path, **kwargs)

            # Compress the file
            if not output_name.endswith(".gz"):
                output_name = output_name + ".gz"
            compress_file(tmp_path, output_name)

        finally:
            # Clean up temporary file
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    else:
        # Direct write without compression
        write_func(output_name, **kwargs)


class BuildSystem:
    @classmethod
    def from_file(
        cls, filename: str, format: Optional[str] = None
    ) -> Tuple[pl.DataFrame, Box, Optional[Dict[str, Any]]]:
        """
        Load system from a file (supports .gz compression).

        Parameters
        ----------
        filename : str
            Path to the input file (can be .gz compressed).
        format : str, optional
            File format. If None, inferred from file extension.
            Supported formats: 'data', 'lmp', 'dump', 'poscar', 'xyz', self difined mp.
            Add '.gz' suffix for compressed files (e.g., 'data.gz', 'xyz.gz').

        Returns
        -------
        Tuple[pl.DataFrame, mdapy.box.Box, Optional[Dict[str, Any]]]
            DataFrame with atom data, simulation box, and optional global info.
        """
        if format is None:
            # Auto-detect format from filename
            parts = os.path.basename(filename).split(".")
            if parts[-1] == "gz":
                # Handle .gz files
                if len(parts) >= 2:
                    format = parts[-2]
                else:
                    raise ValueError("Cannot infer format from filename")
            else:
                format = parts[-1]

        format = format.lower()

        # Validate format
        supported_formats = ["data", "lmp", "dump", "poscar", "xyz", "mp"]
        if format not in supported_formats:
            raise ValueError(
                f"Format '{format}' not supported. "
                f"Supported formats: {supported_formats}"
            )

        # Route to appropriate reader
        if format in ["dump", "dump.gz"]:
            return cls.read_dump(filename)
        elif format in ["data", "lmp"]:
            return cls.read_data(filename)
        elif format == "poscar":
            return cls.read_poscar(filename)
        elif format == "xyz":
            return cls.read_xyz(filename)
        elif format == "mp":
            return cls.read_mp(filename)

    @staticmethod
    def from_ovito(atom: "DataCollection") -> Tuple[pl.DataFrame, Box, Dict[str, Any]]:
        """
        Load system from OVITO DataCollection.

        Parameters
        ----------
        atom : ovito.data.DataCollection
            OVITO data collection object.

        Returns
        -------
        Tuple[pl.DataFrame, mdapy.box.Box, Dict[str, Any]]
            DataFrame with atom data, simulation box, and global info.

        Raises
        ------
        ImportError
            If OVITO is not installed.
        """
        try:
            from ovito.data import DataCollection

            if not isinstance(atom, DataCollection):
                raise TypeError("Only accept an Ovito DataCollection object")
        except ImportError:
            raise ImportError(
                "You must install Ovito first. "
                "See https://www.ovito.org/manual/python/introduction/installation.html"
            )

        boundary = [1 if i else 0 for i in atom.cell.pbc]
        box = Box(np.array(atom.cell[...]).T, boundary)
        global_info = {}
        for i, j in atom.attributes.items():
            global_info[i] = j
        data = {}

        for i in atom.particles.keys():
            arr = np.array(atom.particles[i][...])
            if i in [
                "Position",
                "Particle Type",
                "Particle Identifier",
                "Velocity",
                "Velocity Magnitude",
                "Force",
            ]:
                if i == "Position":
                    data["x"] = arr[:, 0]
                    data["y"] = arr[:, 1]
                    data["z"] = arr[:, 2]
                if i == "Particle Type":
                    data["type"] = arr
                if i == "Particle Identifier":
                    data["id"] = arr
                if i == "Velocity":
                    data["vx"] = arr[:, 0]
                    data["vy"] = arr[:, 1]
                    data["vz"] = arr[:, 2]
                if i == "Velocity Magnitude":
                    pass
                if i == "Force":
                    data["fx"] = arr[:, 0]
                    data["fy"] = arr[:, 1]
                    data["fz"] = arr[:, 2]
            else:
                name = "".join(i.split())
                if arr.ndim == 1:
                    data[name] = arr
                else:
                    for j in range(arr.shape[1]):
                        data[f"{name}_{j}"] = arr[:, j]

        data = pl.DataFrame(data)
        type2element = {t.id: t.name for t in atom.particles.particle_type.types}
        if len(type2element[1]):
            data = data.with_columns(
                pl.col("type").replace_strict(type2element).alias("element")
            )
        return data.rechunk(), box, global_info

    @staticmethod
    def from_ase(atom: "Atoms") -> Tuple[pl.DataFrame, Box]:
        """
        Load system from ASE Atoms.

        Parameters
        ----------
        atom : ase.Atoms
            ASE Atoms object.

        Returns
        -------
        Tuple[pl.DataFrame, mdapy.box.Box]
            DataFrame with atom data and simulation box.

        Raises
        ------
        ImportError
            If ASE is not installed.
        """
        try:
            from ase import Atoms

            if not isinstance(atom, Atoms):
                raise TypeError("Only accept an ASE Atoms object")
        except ImportError:
            raise ImportError(
                "You must install ASE first. See https://ase-lib.org/install.html"
            )

        box = np.array(atom.get_cell())
        boundary = atom.get_pbc()
        box = Box(box, boundary)
        pos = atom.get_positions()
        element = np.array(atom.get_chemical_symbols())
        data = pl.from_numpy(
            pos, schema={"x": pl.Float64, "y": pl.Float64, "z": pl.Float64}
        ).with_columns(element=element)
        return data, box

    @staticmethod
    def from_array(
        pos: np.ndarray,
        box: Union[int, float, Iterable[float], np.ndarray, Box],
    ) -> Tuple[pl.DataFrame, Box]:
        """
        Create system from position array.

        Parameters
        ----------
        pos : np.ndarray
            Atom positions (N x 3).
        box : Union[int, float, Iterable[float], np.ndarray, mdapy.box.Box]
            Simulation box definition.

        Returns
        -------
        Tuple[pl.DataFrame, mdapy.box.Box]
            DataFrame with positions and simulation box.
        """
        if not isinstance(pos, np.ndarray):
            raise TypeError("pos must be numpy array")
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("pos must be N x 3 array")

        box = Box(box)
        data = pl.from_numpy(
            pos, schema={"x": pl.Float64, "y": pl.Float64, "z": pl.Float64}
        )
        return data, box

    @staticmethod
    def from_data(
        data: pl.DataFrame,
        box: Union[int, float, Iterable[float], np.ndarray, Box],
    ) -> Tuple[pl.DataFrame, Box]:
        """
        Create system from DataFrame.

        Parameters
        ----------
        data : pl.DataFrame
            DataFrame with at least 'x', 'y', 'z' columns.
        box : Union[int, float, Iterable[float], np.ndarray, mdapy.box.Box]
            Simulation box definition.

        Returns
        -------
        Tuple[pl.DataFrame, mdapy.box.Box]
            Input DataFrame and simulation box.
        """
        for i in ["x", "y", "z"]:
            if i not in data.columns:
                raise ValueError(f"Data must contain {i} column")
        data = data.with_columns(
            pl.col("x").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
            pl.col("z").cast(pl.Float64),
        )
        box = Box(box)
        return data, box

    @staticmethod
    def read_mp(filename: str) -> Tuple[pl.DataFrame, Box, Optional[Dict[str, Any]]]:
        """
        Read system from mp file.

        Parameters
        ----------
        filename : str
            Path to mp file, such as model.mp.

        Returns
        -------
        Tuple[pl.DataFrame, mdapy.box.Box, Optional[Dict[str, Any]]]
            DataFrame with atom data, simulation box, and global info.
        """
        meta_data = pl.read_parquet_metadata(filename)
        data = pl.read_parquet(filename, rechunk=True)
        if "box" in meta_data.keys():
            box = np.array(meta_data["box"].split(), float).reshape(3, 3)
        else:
            coor = data.select("x", "y", "z")
            box = np.eye(3) * (coor.max() - coor.min()).to_numpy()
        origin = None
        boundary = None
        if "origin" in meta_data.keys():
            origin = np.array(meta_data["origin"].split(), float)
        if "boundary" in meta_data.keys():
            boundary = np.array(meta_data["boundary"].split(), np.int32)
        box = Box(box, boundary, origin)

        global_info = {}
        for i in meta_data.keys():
            if i not in ["box", "origin", "boundary", "ARROW:schema"]:
                global_info[i] = meta_data[i]
        return data, box, global_info

    @staticmethod
    def read_xyz(filename: str) -> Tuple[pl.DataFrame, Box, Optional[Dict[str, Any]]]:
        """
        Read system from XYZ file (supports .gz compression).

        Parameters
        ----------
        filename : str
            Path to XYZ file (can be .xyz.gz).

        Returns
        -------
        Tuple[pl.DataFrame, mdapy.box.Box, Optional[Dict[str, Any]]]
            DataFrame with atom data, simulation box, and global info.
        """
        # Read header
        head = []
        with _open_file(filename, "r") as op:
            for i in range(3):
                head.append(op.readline())

        natom = int(head[0].strip())
        global_info = {}
        results = re.findall(r'(\w+)=(?:"([^"]+)"|([^ ]+))', head[1].replace("'", '"'))
        for match in results:
            key = match[0].lower()
            value = match[1] if match[1] else match[2]
            global_info[key] = value

        classical = "lattice" not in global_info

        if not classical:
            if "properties" not in global_info:
                raise ValueError("Extended XYZ must contain 'properties'")

            # Check boundary condition
            boundary = [1, 1, 1]
            if "pbc" in global_info:
                boundary = [
                    1 if i in ("T", "1") else 0 for i in global_info["pbc"].split()
                ]

            # Get box
            box = np.array(global_info["lattice"].split(), float).reshape(3, 3)

            # Check origin
            origin = np.zeros(3, float)
            if "origin" in global_info:
                origin = np.array(global_info["origin"].split(), float)
            box = np.r_[box, origin.reshape((1, 3))]

            # Parse properties
            content = global_info["properties"].strip().split(":")
            i = 0
            columns = []
            schema = {}
            while i < len(content) - 2:
                n_col = int(content[i + 2])

                # Determine dtype
                if content[i + 1] == "S":
                    dtype = pl.Utf8
                elif content[i + 1] == "R":
                    dtype = pl.Float64
                elif content[i + 1] == "I":
                    dtype = pl.Int32
                else:
                    raise ValueError(f"Unrecognized type {content[i + 1]}")

                # Handle special column names
                if content[i] == "pos" and content[i + 1] == "R" and n_col == 3:
                    columns.extend(["x", "y", "z"])
                    for coord in ["x", "y", "z"]:
                        schema[coord] = dtype
                elif (
                    content[i] in ["species", "element"]
                    and content[i + 1] == "S"
                    and n_col == 1
                ):
                    columns.append("element")
                    schema["element"] = dtype
                elif (
                    content[i] in ["velo", "vel"]
                    and content[i + 1] == "R"
                    and n_col == 3
                ):
                    columns.extend(["vx", "vy", "vz"])
                    for vel in ["vx", "vy", "vz"]:
                        schema[vel] = dtype
                elif (
                    content[i] in ["force", "forces"]
                    and content[i + 1] == "R"
                    and n_col == 3
                ):
                    columns.extend(["fx", "fy", "fz"])
                    for force in ["fx", "fy", "fz"]:
                        schema[force] = dtype
                else:
                    if n_col > 1:
                        for j in range(n_col):
                            col_name = f"{content[i]}_{j}"
                            columns.append(col_name)
                            schema[col_name] = dtype
                    else:
                        columns.append(content[i])
                        schema[content[i]] = dtype
                i += 3
        else:
            # Classical XYZ format
            boundary = [0, 0, 0]
            columns = ["element", "x", "y", "z"]
            schema = {
                "element": pl.Utf8,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
            }

        # Detect multi-space separator
        multi_space = head[-1].count(" ") != len(columns) - 1

        if multi_space:
            # Manual parsing for multi-space
            data = {col: [] for col in columns}
            with _open_file(filename, "r") as op:
                op.readline()  # skip natom
                op.readline()  # skip global_info
                for _ in range(natom):
                    for key, value in zip(columns, op.readline().split()):
                        data[key].append(value)

            df = pl.DataFrame(data).cast(schema)
        else:
            # Use polars CSV reader
            df = pl.read_csv(
                filename,
                separator=" ",
                schema=schema,
                skip_rows=2,
                has_header=False,
            )

        # For classical XYZ, infer box from coordinates
        if classical:
            coor = df.select("x", "y", "z")
            box = np.r_[
                np.eye(3) * (coor.max() - coor.min()).to_numpy(), coor.min().to_numpy()
            ]

        # Clean up global_info
        for key in ["pbc", "properties", "origin", "lattice"]:
            global_info.pop(key, None)

        return df.rechunk(), Box(box, boundary), global_info

    @staticmethod
    def read_poscar(
        filename: str,
    ) -> Tuple[pl.DataFrame, Box, Optional[Dict[str, Any]]]:
        """
        Read system from POSCAR file (supports .gz compression).

        Parameters
        ----------
        filename : str
            Path to POSCAR file (can be .gz).

        Returns
        -------
        Tuple[pl.DataFrame, mdapy.box.Box, Optional[Dict[str, Any]]]
            DataFrame with atom data, simulation box, and global info.
        """
        global_info = {}
        with _open_file(filename, "r") as op:
            file = op.readlines()

        global_info["comment"] = file[0].strip()
        scale = float(file[1].strip())

        box = np.array([i.split() for i in file[2:5]], float) * scale
        row = 5
        type_list, element_list = [], []

        if file[5].strip()[0].isdigit():
            # No species name
            for atype, num in enumerate(file[5].split(), start=1):
                type_list.extend([atype] * int(num))
            row += 1
        else:
            # Has species name
            if not file[6].strip()[0].isdigit():
                raise ValueError("Invalid POSCAR format")
            for atype, num in zip(file[5].split(), file[6].split()):
                element_list.extend([atype] * int(num))
            row += 2

        selective_dynamics = False
        if file[row].strip()[0] in ["S", "s"]:
            selective_dynamics = True
            row += 1
        global_info["selective_dynamics"] = selective_dynamics

        natoms = max(len(type_list), len(element_list))
        pos, sd = [], []

        if selective_dynamics:
            for line in file[row + 1 : row + 1 + natoms]:
                content = line.split()
                pos.append(content[:3])
                sd.append(content[3:6])
            sd = np.array(sd)
        else:
            for line in file[row + 1 : row + 1 + natoms]:
                pos.append(line.split()[:3])
        pos = np.array(pos, float)

        # Handle coordinate type
        if file[row][0] in ["C", "c", "K", "k"]:
            pos *= scale
        else:
            pos = pos @ box

        row += natoms + 1

        # Handle lattice velocities
        if row < len(file):
            if file[row][0] in ["L", "l"]:
                Lvel = np.array(
                    [line.split()[:3] for line in file[row + 2 : row + 8]], float
                )
                global_info["initialization_state"] = file[row + 1].strip()
                global_info["lattice_velocity"] = Lvel
                row += 8

        # Handle ion velocities
        vel = []
        if row + 1 + natoms <= len(file):
            vel = np.array([i.split() for i in file[row + 1 : row + 1 + natoms]], float)
            if len(file[row].split()) > 0:
                if file[row].strip()[0] not in ["C", "c", "K", "k"]:
                    vel = vel @ box

        # Build DataFrame
        data = {}
        schema = {}

        if len(element_list) == natoms:
            data["element"] = element_list
            schema["element"] = pl.Utf8
        else:
            if len(type_list) != natoms:
                raise ValueError("Atom count mismatch")
            data["type"] = type_list
            schema["type"] = pl.Int32

        data["x"] = pos[:, 0]
        data["y"] = pos[:, 1]
        data["z"] = pos[:, 2]
        for coord in ["x", "y", "z"]:
            schema[coord] = pl.Float64

        if len(sd) > 0:
            data["sdx"] = sd[:, 0]
            data["sdy"] = sd[:, 1]
            data["sdz"] = sd[:, 2]
            for sd_coord in ["sdx", "sdy", "sdz"]:
                schema[sd_coord] = pl.Utf8

        if len(vel) > 0:
            data["vx"] = vel[:, 0]
            data["vy"] = vel[:, 1]
            data["vz"] = vel[:, 2]
            for vel_coord in ["vx", "vy", "vz"]:
                schema[vel_coord] = pl.Float64

        return pl.DataFrame(data, schema).rechunk(), Box(box), global_info

    @staticmethod
    def read_data(filename: str) -> Tuple[pl.DataFrame, Box, Optional[Dict[str, Any]]]:
        """
        Read system from LAMMPS data file (supports .gz compression).

        Parameters
        ----------
        filename : str
            Path to data file (can be .gz).

        Returns
        -------
        Tuple[pl.DataFrame, mdapy.box.Box, Optional[Dict[str, Any]]]
            DataFrame with atom data, simulation box, and global info.
        """
        data_head = []
        box = np.zeros((4, 3))
        row = 0
        xy, xz, yz = 0.0, 0.0, 0.0
        N = 0
        xlo = xhi = ylo = yhi = zlo = zhi = 0.0

        with _open_file(filename, "r") as op:
            while True:
                line = op.readline()
                if not line:
                    break
                data_head.append(line)
                content = line.split()

                if len(content):
                    if content[-1] == "atoms":
                        N = int(content[0])

                    if len(content) >= 2:
                        if content[1] == "bond":
                            raise ValueError("Bond style not supported")

                    if content[-1] == "xhi":
                        xlo, xhi = float(content[0]), float(content[1])
                    if content[-1] == "yhi":
                        ylo, yhi = float(content[0]), float(content[1])
                    if content[-1] == "zhi":
                        zlo, zhi = float(content[0]), float(content[1])
                    if content[-1] == "yz":
                        xy = float(content[0])
                        xz = float(content[1])
                        yz = float(content[2])

                    if content[0] == "Atoms":
                        line = op.readline()
                        data_head.append(line)
                        line = op.readline()
                        data_head.append(line)
                        row += 2
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

        # Determine atom style
        if data_head[-3].split()[-1] == "atomic":
            col_names = ["id", "type", "x", "y", "z"]
        elif data_head[-3].split()[-1] == "charge":
            col_names = ["id", "type", "q", "x", "y", "z"]
        else:
            # Infer from first data line
            line = data_head[-1]
            n_cols = len(line.split())
            if n_cols == 5:
                col_names = ["id", "type", "x", "y", "z"]
            elif n_cols == 6:
                col_names = ["id", "type", "q", "x", "y", "z"]
            else:
                raise ValueError(
                    "Unrecognized data format. Only support atomic and charge"
                )

        schema = {}
        for i in col_names:
            if i in ["id", "type"]:
                schema[i] = pl.Int32
            else:
                schema[i] = pl.Float64

        # Detect multi-space separator
        multi_space = data_head[-1].count(" ") != len(col_names) - 1

        if multi_space:
            with _open_file(filename, "r") as op:
                file = op.readlines()

            data_array = np.array([i.split() for i in file[row : row + N]], float)
            data = pl.from_numpy(data_array[:, : len(col_names)], schema=schema)

            # Try to read velocities
            try:
                if row + N + 1 < len(file):
                    if file[row + N + 1].split()[0].strip() == "Velocities":
                        vel = np.array(
                            [
                                i.split()[1:4]
                                for i in file[row + N + 3 : row + N + 3 + N]
                            ],
                            float,
                        )
                        if vel.shape[0] == data.shape[0]:
                            data = data.with_columns(
                                vx=vel[:, 0], vy=vel[:, 1], vz=vel[:, 2]
                            )
            except (IndexError, ValueError):
                pass
        else:
            data = pl.read_csv(
                filename,
                separator=" ",
                skip_rows=row,
                n_rows=N,
                schema=schema,
                has_header=False,
                ignore_errors=True,
            )

            # Try to read velocities
            try:
                vel = pl.read_csv(
                    filename,
                    separator=" ",
                    skip_rows=row + N + 3,
                    schema={
                        "id": pl.Int32,
                        "vx": pl.Float64,
                        "vy": pl.Float64,
                        "vz": pl.Float64,
                    },
                    has_header=False,
                ).select(pl.all().exclude("id"))
                if vel.shape[0] == data.shape[0]:
                    data = pl.concat([data, vel], how="horizontal")
            except Exception:
                pass

        return data.rechunk(), Box(box, boundary), {}

    @staticmethod
    def read_dump(filename: str) -> Tuple[pl.DataFrame, Box, Dict[str, Any]]:
        """
        Read system from LAMMPS dump file.

        Parameters
        ----------
        filename : str
            Path to dump file.

        Returns
        -------
        Tuple[pl.DataFrame, mdapy.box.Box, Dict[str, Any]]
            DataFrame with atom data, simulation box, and global info.
        """

        dump_head = []
        with _open_file(filename, "r") as op:
            for _ in range(9):
                dump_head.append(op.readline())

        timestep = int(dump_head[1].strip())
        line = dump_head[4].split()
        boundary = [1 if i == "pp" else 0 for i in line[-3:]]

        if "xy" in line:
            # Triclinic box
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
            # Orthogonal box
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

        # Handle scaled coordinates
        if "xs" in data.columns:
            pos = data.select("xs", "ys", "zs").to_numpy() @ box[:-1]
            data = data.with_columns(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2]).select(
                pl.all().exclude("xs", "ys", "zs")
            )

        return data.rechunk(), Box(box, boundary), {"timestep": timestep}


class SaveSystem:
    @staticmethod
    def to_ase(data: pl.DataFrame, box: Box) -> "Atoms":
        """
        Convert system to ASE Atoms object.

        Parameters
        ----------
        data : pl.DataFrame
            Particle data with at least 'x', 'y', 'z', 'element' columns.
        box : mdapy.box.Box
            Simulation box.

        Returns
        -------
        ase.Atoms
            ASE Atoms object.

        Raises
        ------
        ImportError
            If ASE is not installed.
        ValueError
            If required columns are missing.

        """
        try:
            from ase import Atoms
        except ImportError:
            raise ImportError(
                "You must install ASE first. "
                "See https://wiki.fysik.dtu.dk/ase/install.html"
            )

        # Check required columns
        for col in ["x", "y", "z", "element"]:
            if col not in data.columns:
                raise ValueError(f"data must contain {col} column")

        # Get positions
        positions = data.select("x", "y", "z").to_numpy()
        symbols = data["element"].to_list()

        # Create ASE Atoms object
        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=box.box,
            pbc=[bool(p) for p in box.boundary],
        )

        # Add velocities if available
        if all(col in data.columns for col in ["vx", "vy", "vz"]):
            velocities = data.select("vx", "vy", "vz").to_numpy()
            atoms.set_velocities(velocities)

        return atoms

    @staticmethod
    def to_ovito(
        data: pl.DataFrame, box: Box, global_info: Optional[Dict[str, Any]] = None
    ) -> "DataCollection":
        """
        Save system to OVITO DataCollection.

        Parameters
        ----------
        data : pl.DataFrame
            Particle informations.
        box : Box
            Simulation cell.
        global_info : Optional[Dict[str, Any]], optional
            Global attributes to add.

        Returns
        -------
        ovito.data.DataCollection
            OVITO data collection object.

        Raises
        ------
        ImportError
            If OVITO is not installed.
        """
        try:
            from ovito.data import DataCollection
        except ImportError:
            raise ImportError(
                "You must install Ovito first. "
                "See https://www.ovito.org/manual/python/introduction/installation.html"
            )

        for i in ["x", "y", "z"]:
            if i not in data.columns:
                raise ValueError(f"data must contain {i} column")

        data_collection = DataCollection()
        cell = data_collection.create_cell(
            matrix=box.box.T, pbc=[bool(p) for p in box.boundary]
        )
        cell[:, 3] = box.origin
        particles = data_collection.create_particles(count=data.shape[0])
        particles.create_property(
            "Position", data=data.select("x", "y", "z").to_numpy()
        )

        if "element" in data.columns:
            types = particles.create_property("Particle Type")
            symbols = data["element"]
            with types as tarray:
                for i, sym in enumerate(symbols):
                    tarray[i] = types.add_type_name(sym, particles).id
        elif "type" in data.columns:
            particles.create_property("Particle Type", data=data["type"].to_numpy())
        else:
            particles.create_property(
                "Particle Type", data=np.ones(data.shape[0], np.int32)
            )

        # Create user-defined properties
        if all(col in data.columns for col in ["vx", "vy", "vz"]):
            particles.create_property(
                "Velocity", data=data.select("vx", "vy", "vz").to_numpy()
            )
        if all(col in data.columns for col in ["fx", "fy", "fz"]):
            particles.create_property(
                "Force", data=data.select("fx", "fy", "fz").to_numpy()
            )

        for name in data.columns:
            if name in [
                "x",
                "y",
                "z",
                "element",
                "type",
                "vx",
                "vy",
                "vz",
                "fx",
                "fy",
                "fz",
            ]:
                continue
            try:
                particles.create_property(name, data=data[name])
            except Exception:
                pass

        if global_info:
            for key, value in global_info.items():
                try:
                    data_collection.attributes[key] = value
                except Exception:
                    pass

        return data_collection

    def write_mp(
        output_name: str, data: pl.DataFrame, box: Box, global_info: Dict[str, str]
    ) -> None:
        """
        Write system to mp file.

        Parameters
        ----------
        output_name : str
            Output file path.
        data : pl.DataFrame
            Atom data with 'x', 'y', 'z', 'element'.
        box : Box
            Box information.
        global_info : Dict
            Global information.
        """
        metadata = {}
        metadata["box"] = " ".join(box.box.astype(str).flatten().tolist())
        metadata["origin"] = " ".join(box.origin.astype(str).tolist())
        metadata["boundary"] = " ".join(box.boundary.astype(str).tolist())
        for i in global_info.keys():
            if i not in ["box", "origin", "boundary"] and i in [
                "energy",
                "stress",
                "virial",
                "timestep",
            ]:
                metadata[str(i)] = str(global_info[i])
        data.write_parquet(output_name, metadata=metadata)

    @staticmethod
    def write_xyz(
        output_name: str,
        box: Box,
        data: pl.DataFrame,
        classical: bool = False,
        compress: bool = False,
        **kargs,
    ):
        """
        Write system to XYZ file (with optional compression).

        Parameters
        ----------
        output_name : str
            Output file path.
        box : mdapy.box.Box
            Simulation box.
        data : pl.DataFrame
            Atom data with 'x', 'y', 'z', 'element'.
        classical : bool, optional
            Use classical XYZ format.
        compress : bool, optional
            Compress output to .gz format.
        **kargs
            Additional key-value pairs for extended XYZ.
        """
        for col in ["x", "y", "z", "element"]:
            if col not in data.columns:
                raise ValueError(f"data must contain {col} column")

        def _write_xyz_internal(filename, box, data, classical, **kargs):
            if classical:
                with open(filename, "wb") as op:
                    op.write(f"{data.shape[0]}\n".encode())
                    op.write("Classical XYZ file written by mdapy.\n".encode())
                    data.select("element", "x", "y", "z").write_csv(
                        op, separator=" ", include_header=False
                    )
            else:
                # Build properties string
                properties = []
                for name, dtype in zip(data.columns, data.dtypes):
                    if dtype in pl.INTEGER_DTYPES:
                        ptype = "I"
                    elif dtype in pl.FLOAT_DTYPES:
                        ptype = "R"
                    elif dtype == pl.Utf8:
                        ptype = "S"
                    else:
                        raise ValueError(f"Unrecognized data type {dtype}")

                    if name == "element":
                        properties.append(f"species:{ptype}:1")
                    else:
                        properties.append(f"{name}:{ptype}:1")

                properties_str = "Properties=" + ":".join(properties).replace(
                    "x:R:1:y:R:1:z:R:1", "pos:R:3"
                )
                if "vx:R:1:vy:R:1:vz:R:1" in properties_str:
                    properties_str = properties_str.replace(
                        "vx:R:1:vy:R:1:vz:R:1", "vel:R:3"
                    )
                if "fx:R:1:fy:R:1:fz:R:1" in properties_str:
                    properties_str = properties_str.replace(
                        "fx:R:1:fy:R:1:fz:R:1", "force:R:3"
                    )

                # Handle group columns
                if "group_0" in data.columns:
                    numgroup = data.select("^group_.*$").shape[1]
                    groupstr = ":".join([f"group_{i}:I:1" for i in range(numgroup)])
                    properties_str = properties_str.replace(
                        groupstr, f"group:I:{numgroup}"
                    )

                # Build comment line
                lattice_str = (
                    'Lattice="' + " ".join(box.box.flatten().astype(str).tolist()) + '"'
                )
                pbc_str = (
                    'pbc="'
                    + " ".join(["T" if i == 1 else "F" for i in box.boundary])
                    + '"'
                )
                origin_str = (
                    'Origin="' + " ".join(box.origin.astype(str).tolist()) + '"'
                )
                comments = f"{lattice_str} {properties_str} {pbc_str} {origin_str}"

                for key, value in kargs.items():
                    if key.lower() not in ["lattice", "pbc", "properties", "origin"]:
                        try:
                            comments += f" {key}={value}"
                        except Exception:
                            pass

                with open(filename, "wb") as op:
                    op.write(f"{data.shape[0]}\n".encode())
                    op.write(f"{comments}\n".encode())
                    data.write_csv(op, separator=" ", include_header=False)

        _save_with_compression(
            _write_xyz_internal,
            output_name,
            compress,
            box=box,
            data=data,
            classical=classical,
            **kargs,
        )

    @staticmethod
    def write_poscar(
        output_name: str,
        box: Box,
        data: pl.DataFrame,
        reduced_pos: bool = False,
        selective_dynamics: bool = False,
        compress: bool = False,
    ):
        """
        Write system to POSCAR file (with optional compression).

        Parameters
        ----------
        output_name : str
            Output file path.
        box : mdapy.box.Box
            Simulation box.
        data : pl.DataFrame
            Atom data with 'element', 'x', 'y', 'z'.
        reduced_pos : bool, optional
            Use reduced coordinates.
        selective_dynamics : bool, optional
            Include selective dynamics.
        compress : bool, optional
            Compress output to .gz format.
        """
        for col in ["element", "x", "y", "z"]:
            if col not in data.columns:
                raise ValueError(f"data must contain {col} column")

        def _write_poscar_internal(
            filename, box: Box, data: pl.DataFrame, reduced_pos, selective_dynamics
        ):
            data = data.sort("element").with_columns(
                pl.col("x") - box.origin[0],
                pl.col("y") - box.origin[1],
                pl.col("z") - box.origin[2],
            )
            element_list = data["element"].unique().sort()
            element_number = [
                data.filter(pl.col("element") == i).shape[0] for i in element_list
            ]

            if reduced_pos:
                new_pos = np.dot(data.select("x", "y", "z").to_numpy(), box.inverse_box)
                data = data.with_columns(
                    pl.lit(new_pos[:, 0]).alias("x"),
                    pl.lit(new_pos[:, 1]).alias("y"),
                    pl.lit(new_pos[:, 2]).alias("z"),
                )

            if selective_dynamics:
                for col in ["sdx", "sdy", "sdz"]:
                    if col not in data.columns:
                        raise ValueError(
                            f"data must contain {col} if selective_dynamics=True"
                        )

            with open(filename, "wb") as op:
                op.write("# VASP POSCAR file written by mdapy.\n".encode())
                op.write("1.0000000000\n".encode())
                op.write("{:.10f} {:.10f} {:.10f}\n".format(*box.box[0]).encode())
                op.write("{:.10f} {:.10f} {:.10f}\n".format(*box.box[1]).encode())
                op.write("{:.10f} {:.10f} {:.10f}\n".format(*box.box[2]).encode())

                for aname in element_list:
                    op.write(f"{aname} ".encode())
                op.write("\n".encode())
                for atype in element_number:
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

        _save_with_compression(
            _write_poscar_internal,
            output_name,
            compress,
            box=box,
            data=data,
            reduced_pos=reduced_pos,
            selective_dynamics=selective_dynamics,
        )

    @staticmethod
    def write_data(
        output_name: str,
        box: Box,
        data: pl.DataFrame,
        element_list: Optional[List] = None,
        num_type: Optional[int] = None,
        data_format: str = "atomic",
        compress: bool = False,
    ):
        """
        Write system to LAMMPS data file (with optional compression).

        Parameters
        ----------
        output_name : str
            Output file path.
        box : mdapy.box.Box
            Simulation box.
        data : pl.DataFrame
            Atom data with 'type', 'x', 'y', 'z'.
        element_list : Optional[List], optional
            Element names for atom types.
        num_type : Optional[int], optional
            Number of atom types.
        data_format : str, optional
            'atomic' or 'charge'.
        compress : bool, optional
            Compress output to .gz format.
        """
        if not isinstance(data, pl.DataFrame):
            raise TypeError("data must be polars DataFrame")

        for col in ["type", "x", "y", "z"]:
            if col not in data.columns:
                raise ValueError(f"data must contain {col} column")

        def _write_data_internal(
            filename, box, data, element_list, num_type, data_format
        ):
            if "id" not in data.columns:
                data = data.with_row_index("id", offset=1)

            need_rotation = False
            if box.box[0, 1] != 0 or box.box[0, 2] != 0 or box.box[1, 2] != 0:
                need_rotation = True

            if need_rotation:
                pos = data.select("x", "y", "z").to_numpy(writable=True)
                box, rotate = box.align_to_lammps_box()
                pos -= box.origin
                box.origin[:] = 0
                pos = pos @ rotate
                data = data.with_columns(
                    pl.lit(pos[:, 0]).alias("x"),
                    pl.lit(pos[:, 1]).alias("y"),
                    pl.lit(pos[:, 2]).alias("z"),
                )

            if data_format not in ["atomic", "charge"]:
                raise ValueError(
                    f"Unrecognized data format: {data_format}. "
                    "Only support atomic and charge"
                )

            if num_type is None:
                num_type = data["type"].max()
            else:
                if num_type < data["type"].max():
                    raise ValueError(f"num_type should be >= {data['type'].max()}")

            if element_list is not None:
                if len(element_list) < num_type:
                    raise ValueError(
                        f"element_list should contain at least {num_type} elements"
                    )
                num_type = len(element_list)
                for i in element_list:
                    if i not in atomic_numbers:
                        raise ValueError(f"Unrecognized element name {i}")

            with open(filename, "wb") as op:
                op.write("# LAMMPS data file written by mdapy.\n\n".encode())
                op.write(f"{data.shape[0]} atoms\n{num_type} atom types\n\n".encode())
                op.write(
                    f"{box.origin[0]} {box.origin[0] + box.box[0, 0]} xlo xhi\n".encode()
                )
                op.write(
                    f"{box.origin[1]} {box.origin[1] + box.box[1, 1]} ylo yhi\n".encode()
                )
                op.write(
                    f"{box.origin[2]} {box.origin[2] + box.box[2, 2]} zlo zhi\n".encode()
                )
                xy, xz, yz = box.box[1, 0], box.box[2, 0], box.box[2, 1]
                if xy != 0 or xz != 0 or yz != 0:
                    op.write(f"{xy} {xz} {yz} xy xz yz\n".encode())
                op.write("\n".encode())

                if element_list is not None:
                    op.write("Masses\n\n".encode())
                    for i, j in enumerate(element_list, start=1):
                        mass = atomic_masses[atomic_numbers[j]]
                        op.write(f"{i} {mass} # {j}\n".encode())
                    op.write("\n".encode())

                op.write(rf"Atoms # {data_format}".encode())
                op.write("\n\n".encode())

                if data_format == "atomic":
                    table = data.select(["id", "type", "x", "y", "z"])
                elif data_format == "charge":
                    if "q" not in data.columns:
                        table = data.with_columns(pl.lit(0.0).alias("q")).select(
                            ["id", "type", "q", "x", "y", "z"]
                        )
                    else:
                        table = data.select(["id", "type", "q", "x", "y", "z"])
                table.write_csv(op, separator=" ", include_header=False)

                if all(col in data.columns for col in ["vx", "vy", "vz"]):
                    op.write("\nVelocities\n\n".encode())
                    table = data.select(["id", "vx", "vy", "vz"])
                    table.write_csv(op, separator=" ", include_header=False)

        _save_with_compression(
            _write_data_internal,
            output_name,
            compress,
            box=box,
            data=data,
            element_list=element_list,
            num_type=num_type,
            data_format=data_format,
        )

    @staticmethod
    def write_dump(
        output_name: str,
        box: Box,
        data: pl.DataFrame,
        timestep: float = 0.0,
        compress: bool = False,
    ):
        """
        Write system to LAMMPS dump file (with optional compression).

        Parameters
        ----------
        output_name : str
            Output file path.
        box : mdapy.box.Box
            Simulation box.
        data : pl.DataFrame
            Atom data with 'type', 'x', 'y', 'z'.
        timestep : float, optional
            Timestep value.
        compress : bool, optional
            Compress output to .gz format.
        """
        if not isinstance(data, pl.DataFrame):
            raise TypeError("data must be polars DataFrame")

        for col in ["type", "x", "y", "z"]:
            if col not in data.columns:
                raise ValueError(f"data must contain {col} column")

        def _write_dump_internal(filename, box, data, timestep):
            if "id" not in data.columns:
                data = data.with_row_index("id", offset=1)

            need_rotation = False
            if box.box[0, 1] != 0 or box.box[0, 2] != 0 or box.box[1, 2] != 0:
                need_rotation = True

            if need_rotation:
                pos = data.select("x", "y", "z").to_numpy(writable=True)
                box, rotate = box.align_to_lammps_box()
                pos -= box.origin
                box.origin[:] = 0
                pos = pos @ rotate
                data = data.with_columns(
                    pl.lit(pos[:, 0]).alias("x"),
                    pl.lit(pos[:, 1]).alias("y"),
                    pl.lit(pos[:, 2]).alias("z"),
                )

            data = data.select(pl.selectors.by_dtype(pl.NUMERIC_DTYPES))

            boundary2str = ["pp" if i == 1 else "ss" for i in box.boundary]
            with open(filename, "wb") as op:
                op.write(f"ITEM: TIMESTEP\n{timestep}\n".encode())
                op.write("ITEM: NUMBER OF ATOMS\n".encode())
                op.write(f"{data.shape[0]}\n".encode())

                xlo, ylo, zlo = box.origin
                xhi = xlo + box.box[0, 0]
                yhi = ylo + box.box[1, 1]
                zhi = zlo + box.box[2, 2]
                xy, xz, yz = box.box[1, 0], box.box[2, 0], box.box[2, 1]

                if xy != 0 or xz != 0 or yz != 0:
                    # Triclinic box
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
                    # Orthogonal box
                    op.write(
                        f"ITEM: BOX BOUNDS {boundary2str[0]} {boundary2str[1]} {boundary2str[2]}\n".encode()
                    )
                    op.write(f"{xlo} {xhi}\n".encode())
                    op.write(f"{ylo} {yhi}\n".encode())
                    op.write(f"{zlo} {zhi}\n".encode())

                col_name = "ITEM: ATOMS " + " ".join(data.columns) + " \n"
                op.write(col_name.encode())
                data.write_csv(op, separator=" ", include_header=False)

        _save_with_compression(
            _write_dump_internal,
            output_name,
            compress,
            box=box,
            data=data,
            timestep=timestep,
        )


if __name__ == "__main__":
    pass
