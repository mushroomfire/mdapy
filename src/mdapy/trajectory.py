# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from typing import List, Optional, Dict, Any, Union, Iterator, Tuple, TextIO
import numpy as np
import polars as pl
import re
from mdapy.system import System
from mdapy.box import Box

# Standard column names that have special meaning in XYZ format
_STANDARD_COLS = {"element", "x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz"}


class XYZTrajectory:
    """
    XYZ trajectory file reader and manager.

    This class provides functionality to read, manipulate, and write XYZ format
    trajectory files. It supports both classical XYZ format (element + coordinates)
    and extended XYZ format (with periodic boundary conditions and additional properties).

    Two reading modes are available:
    - Serial mode: Read frames sequentially (default)
    - Fast mode: Optimized batch reading assuming all frames have identical columns

    Parameters
    ----------
    filename : Optional[str]
        Path to XYZ trajectory file to load
    systems : Optional[List[System]]
        List of System objects to initialize trajectory from memory
    fast_mode : bool, default=False
        If True, use optimized reading assuming all frames have the same columns.
        This mode is significantly faster but requires all frames to have identical
        column structure.

    Raises
    ------
    ValueError
        If neither filename nor systems is provided

    Examples
    --------
    >>> # Load trajectory in serial mode
    >>> traj = XYZTrajectory("trajectory.xyz")

    >>> # Load trajectory in fast mode
    >>> traj = XYZTrajectory("trajectory.xyz", fast_mode=True)

    >>> # Create trajectory from existing System objects
    >>> traj = XYZTrajectory(systems=[system1, system2])

    >>> # Access frames
    >>> print(len(traj))  # Number of frames
    >>> frame = traj[0]  # Get first frame
    >>> sub_traj = traj[0:10]  # Slice trajectory

    >>> # Iterate over frames
    >>> for frame in traj:
    ...     print(frame.N)

    >>> # Save trajectory
    >>> traj.save("output.xyz")
    >>> traj.save("output.xyz", frames=[0, 1, 2])  # Save specific frames
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        systems: Optional[List[System]] = None,
        fast_mode: bool = False,
    ) -> None:
        self._systems: List[System] = []
        self._filename = filename
        self._fast_mode = fast_mode

        if systems is not None:
            self._systems = systems
        elif filename is not None:
            self._load()
        else:
            raise ValueError("At least has systems or filename.")

    def _load(self) -> None:
        """Load trajectory file using the specified reading mode."""
        if self._fast_mode:
            self._systems = self._read_xyz_fast(self._filename)
        else:
            self._systems = self._read_xyz_serial(self._filename)

    def _read_xyz_fast(self, filename: str) -> List[System]:
        """
        Fast trajectory reading mode.

        Assumes all frames have identical column structure, which allows for
        optimized batch processing using vectorized operations.

        Parameters
        ----------
        filename : str
            Path to XYZ file

        Returns
        -------
        List[System]
            List of System objects for each frame
        """
        df_raw = (
            pl.read_csv(
                filename,
                has_header=False,
                new_columns=["line"],
                separator="\n",
                truncate_ragged_lines=True,
            )
            .with_columns(pl.col("line").str.strip_chars())
            .with_row_index()
        )

        # Locate frame boundaries
        row = 0
        row_list = []
        while row < df_raw.shape[0]:
            row_list.append(row)
            N = int(df_raw.item(row, 1))
            row += N + 2
        row_list = np.array(row_list, np.int32)

        sele = np.r_[row_list, row_list + 1]
        Nframe = row_list.shape[0]

        # Detect format type from first frame
        first_comment_line = df_raw.item(row_list[0] + 1, 1)
        is_classical = "lattice" not in first_comment_line.lower()

        origin = np.zeros(3)
        if is_classical:
            # Classical XYZ format
            columns = ["element", "x", "y", "z"]
            schema = {
                "element": pl.Utf8,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
            }
            boundary = np.array([0, 0, 0], np.int32)
        else:
            # Extended XYZ format - parse metadata from comment lines
            target_keys = [
                "lattice",
                "properties",
                "pbc",
                "energy",
                "origin",
                "force",
                "virial",
                "stress",
            ]
            exprs = []
            for key in target_keys:
                pat = rf'(?i){key}=(?:"([^"]+)"|([^ \n]+))'
                expr = (
                    pl.col("line")
                    .str.extract_groups(pat)
                    .struct.field("1")
                    .fill_null(pl.col("line").str.extract_groups(pat).struct.field("2"))
                    .alias(key)
                )
                exprs.append(expr)

            comment_info = (
                df_raw.filter(pl.col("index").is_in(row_list + 1))
                .with_columns(pl.col("line").str.replace("'", '"'))
                .select(exprs)
            )
            columns, schema = self._parse_properties(comment_info["properties"].item(0))
            has_pbc = not comment_info["pbc"].has_nulls()
            has_origin = not comment_info["origin"].has_nulls()
            boundary = np.ones(3, np.int32)

        # Parse atom data for all frames
        rep = (
            df_raw.filter(pl.col("index").is_in(row_list))["line"]
            .cast(pl.Int32)
            .to_numpy()
        )
        frame = np.repeat(np.arange(Nframe), rep)

        # Get the first data line to determine the actual separator
        first_data_line: str = df_raw.item(row_list[0] + 2, 1)
        # Find the separator pattern between first and second field
        first_field_end = first_data_line.find(first_data_line.split()[1])
        separator = first_data_line[len(first_data_line.split()[0]) : first_field_end]

        all_data = (
            df_raw.filter(~pl.col("index").is_in(sele))
            .select(
                pl.col("line")
                .str.split_exact(separator, n=len(columns))
                .struct.rename_fields(columns)
                .alias("_tmp")
            )
            .unnest("_tmp")
            .cast(schema)
            .with_columns(frame=frame)
            .partition_by("frame", maintain_order=True, include_key=False)
        )

        # Build System objects for each frame
        systems = []
        for i in range(Nframe):
            if is_classical:
                coor = all_data[i].select("x", "y", "z")
                box = np.eye(3) * (coor.max() - coor.min()).to_numpy()
                new_box = Box(box=box, origin=origin, boundary=boundary)
                info = {}
            else:
                box = np.array(comment_info["lattice"].item(i).split(), float).reshape(
                    3, 3
                )

                if has_pbc:
                    boundary = np.array(
                        [
                            1 if j in ("T", "1") else 0
                            for j in comment_info["pbc"].item(i).split()
                        ],
                        np.int32,
                    )

                if has_origin:
                    origin = np.array(comment_info["origin"].item(i).split(), float)

                new_box = Box(box=box, origin=origin, boundary=boundary)

                info = {}
                for kk in ["energy", "force", "virial", "stress"]:
                    va = comment_info[kk].item(i)
                    if va is not None:
                        info[kk] = va

            systems.append(System(box=new_box, data=all_data[i], global_info=info))

        return systems

    def _read_xyz_serial(self, filename: str) -> List[System]:
        """
        Serial trajectory reading mode.

        Reads frames sequentially, parsing each frame's structure individually.
        This mode is slower but handles trajectories with varying column structures.

        Parameters
        ----------
        filename : str
            Path to XYZ file

        Returns
        -------
        List[System]
            List of System objects for each frame
        """
        systems = []
        with open(filename, "r") as f:
            while True:
                natom_line = f.readline()
                if not natom_line or not natom_line.strip():
                    break
                natom = int(natom_line.strip())

                info_line = f.readline()
                if not info_line:
                    break

                data_lines = []
                for _ in range(natom):
                    line = f.readline()
                    if not line:
                        break
                    data_lines.append(line)

                if len(data_lines) != natom:
                    break

                df, box, global_info = self._parse_frame(info_line, data_lines)
                systems.append(System(data=df, box=box, global_info=global_info))

        return systems

    def save(
        self,
        filename: str,
        frames: Optional[Union[List[int], int]] = None,
        mode: str = "w",
    ) -> None:
        """
        Save trajectory to XYZ file.

        Parameters
        ----------
        filename : str
            Output file path
        frames : Optional[Union[List[int], int]], default=None
            Frame indices to save. Can be:
            - None: save all frames
            - int: save single frame
            - List[int]: save specified frames
        mode : str, default='w'
            Writing mode can be:
            - 'w' : write mode
            - 'a' : append mode

        Examples
        --------
        >>> traj.save("output.xyz")  # Save all frames
        >>> traj.save("output.xyz", 0)  # Save first frame only
        >>> traj.save("output.xyz", [0, 5, 10])  # Save specific frames
        """
        if frames is None:
            systems_to_save = self._systems
        elif isinstance(frames, int):
            systems_to_save = [self._systems[frames]]
        else:
            systems_to_save = [self._systems[i] for i in frames]
        assert mode in ["w", "a"]
        with open(filename, mode) as f:
            for system in systems_to_save:
                self._write_single_frame(f, system)

    def append(self, system: System) -> None:
        """
        Append a frame to the trajectory.

        Parameters
        ----------
        system : System
            System object to append

        Raises
        ------
        TypeError
            If system is not a System object
        """
        if not isinstance(system, System):
            raise TypeError("Can only append System type objects")
        self._systems.append(system)

    def extend(self, systems: List[System]) -> None:
        """
        Extend trajectory with multiple frames.

        Parameters
        ----------
        systems : List[System]
            List of System objects to append

        Raises
        ------
        TypeError
            If any element is not a System object
        """
        for system in systems:
            if not isinstance(system, System):
                raise TypeError("Can only append System type objects")
        self._systems.extend(systems)

    def insert(self, index: int, system: System) -> None:
        """
        Insert a frame at specified position.

        Parameters
        ----------
        index : int
            Position to insert at
        system : System
            System object to insert

        Raises
        ------
        TypeError
            If system is not a System object
        """
        if not isinstance(system, System):
            raise TypeError("Can only insert System type objects")
        self._systems.insert(index, system)

    def pop(self, index: int = -1) -> System:
        """
        Remove and return frame at specified position.

        Parameters
        ----------
        index : int, default=-1
            Position to pop from (default is last frame)

        Returns
        -------
        System
            The removed System object
        """
        return self._systems.pop(index)

    def remove(self, indices: int) -> None:
        """
        Remove frame at specified index.

        Parameters
        ----------
        indices : int
            Frame index to remove
        """
        del self._systems[indices]

    def get_atoms_count(self) -> List[int]:
        """
        Get atom count for each frame.

        Returns
        -------
        List[int]
            List of atom counts
        """
        return [len(s.N) for s in self._systems]

    def concatenate(self, other: "XYZTrajectory") -> "XYZTrajectory":
        """
        Concatenate two trajectories.

        Parameters
        ----------
        other : XYZTrajectory
            Another trajectory to concatenate

        Returns
        -------
        XYZTrajectory
            New trajectory containing frames from both trajectories

        Examples
        --------
        >>> traj1 = XYZTrajectory("file1.xyz")
        >>> traj2 = XYZTrajectory("file2.xyz")
        >>> combined = traj1.concatenate(traj2)
        """
        return XYZTrajectory(systems=self._systems + other._systems)

    def __len__(self) -> int:
        """Return number of frames in trajectory."""
        return len(self._systems)

    def __getitem__(self, index: Union[int, slice]) -> Union[System, "XYZTrajectory"]:
        """
        Access frames by index or slice.

        Parameters
        ----------
        index : Union[int, slice]
            Frame index or slice

        Returns
        -------
        Union[System, XYZTrajectory]
            Single System if index is int, new XYZTrajectory if slice

        Examples
        --------
        >>> frame = traj[0]  # Get first frame
        >>> sub_traj = traj[0:10]  # Get first 10 frames
        >>> sub_traj = traj[::10]  # Get every 10th frame
        """
        if isinstance(index, slice):
            return XYZTrajectory(systems=self._systems[index])
        return self._systems[index]

    def __setitem__(self, index: int, system: System) -> None:
        """
        Set frame at specified index.

        Parameters
        ----------
        index : int
            Frame index
        system : System
            System object to set

        Raises
        ------
        TypeError
            If system is not a System object
        """
        if not isinstance(system, System):
            raise TypeError("Can only set System type objects")
        self._systems[index] = system

    def __iter__(self) -> Iterator[System]:
        """Iterate over all frames."""
        return iter(self._systems)

    def __repr__(self) -> str:
        """String representation of trajectory."""
        return (
            f"XYZTrajectory(frames={len(self._systems)}, "
            f"file='{self._filename if self._filename else 'memory'}')"
        )

    def _parse_frame(
        self, info_line: str, data_lines: List[str]
    ) -> Tuple[pl.DataFrame, Box, Optional[Dict[str, Any]]]:
        """
        Parse a single XYZ frame.

        Parses the comment line and data lines to extract atomic data,
        box information, and global metadata.

        Parameters
        ----------
        info_line : str
            Comment line containing metadata
        data_lines : List[str]
            Lines containing atomic data

        Returns
        -------
        Tuple[pl.DataFrame, Box, Optional[Dict[str, Any]]]
            DataFrame with atomic data, Box object, and global_info dictionary

        Raises
        ------
        ValueError
            If extended XYZ format is missing required 'properties' field
        """
        global_info = {}
        results = re.findall(
            r'(\w+)=(?:"([^"]+)"|([^ ]+))', info_line.replace("'", '"')
        )
        for match in results:
            key = match[0].lower()
            value = match[1] if match[1] else match[2]
            global_info[key] = value

        classical = "lattice" not in global_info

        if not classical:
            if "properties" not in global_info:
                raise ValueError("Extended XYZ must contain 'properties'")

            boundary = [1, 1, 1]
            if "pbc" in global_info:
                boundary = [
                    1 if i in ("T", "1") else 0 for i in global_info["pbc"].split()
                ]

            box_array = np.array(global_info["lattice"].split(), float).reshape(3, 3)

            origin = np.zeros(3, float)
            if "origin" in global_info:
                origin = np.array(global_info["origin"].split(), float)

            columns, schema = self._parse_properties(global_info["properties"])
        else:
            boundary = [0, 0, 0]
            columns = ["element", "x", "y", "z"]
            schema = {
                "element": pl.Utf8,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
            }
            origin = np.zeros(3, float)

        data = {col: [] for col in columns}
        for line in data_lines:
            values = line.split()
            for col, val in zip(columns, values):
                data[col].append(val)

        df = pl.DataFrame(data).cast(schema)

        if classical:
            coor = df.select("x", "y", "z")
            box_array = np.eye(3) * (coor.max() - coor.min()).to_numpy()

        for key in ["pbc", "properties", "origin", "lattice"]:
            global_info.pop(key, None)

        return df.rechunk(), Box(box_array, boundary, origin), global_info

    def _parse_properties(
        self, properties_str: str
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Parse extended XYZ properties string.

        Properties format: "name:type:count:name:type:count:..."
        where type is S (string), R (real/float), or I (integer).

        Parameters
        ----------
        properties_str : str
            Properties string (e.g., "species:S:1:pos:R:3:force:R:3")

        Returns
        -------
        Tuple[List[str], Dict[str, Any]]
            Column names and Polars schema dictionary

        Raises
        ------
        ValueError
            If property type is not recognized (not S, R, or I)

        Notes
        -----
        Special property names are mapped to standard column names:
        - "pos" -> ["x", "y", "z"]
        - "species"/"element" -> "element"
        - "vel"/"velo" -> ["vx", "vy", "vz"]
        - "force"/"forces" -> ["fx", "fy", "fz"]
        """
        content = properties_str.strip().split(":")
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
                dtype = pl.Int32
            else:
                raise ValueError(f"Unrecognized type {content[i + 1]}")

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
            elif content[i] in ["velo", "vel"] and content[i + 1] == "R" and n_col == 3:
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

        return columns, schema

    def _write_single_frame(self, f: TextIO, system: System) -> None:
        """
        Write a single frame to XYZ file.

        Writes in either classical XYZ format (no periodic boundaries)
        or extended XYZ format (with lattice, pbc, and properties).

        Parameters
        ----------
        f : TextIO
            File handle opened for writing
        system : System
            System object containing frame data to write

        Notes
        -----
        Classical format is used when system.box.boundary.sum() == 0,
        otherwise extended format is used.
        """
        df = system.data
        natom = len(df)

        f.write(f"{natom}\n")

        info_parts = []

        # Determine format based on boundary conditions
        is_extended = system.box.boundary.sum() > 0

        if is_extended:
            # Extended XYZ format
            lattice = system.box.box.flatten()
            lattice_str = " ".join(f"{x:.10f}" for x in lattice)
            info_parts.append(f'Lattice="{lattice_str}"')

            pbc = " ".join("T" if b else "F" for b in system.box.boundary)
            info_parts.append(f'pbc="{pbc}"')

            if hasattr(system.box, "origin") and np.any(system.box.origin != 0):
                origin_str = " ".join(f"{x:.10f}" for x in system.box.origin)
                info_parts.append(f'Origin="{origin_str}"')

            properties = []
            if "element" in df.columns:
                properties.append("species:S:1")
            properties.append("pos:R:3")

            if all(c in df.columns for c in ["vx", "vy", "vz"]):
                properties.append("vel:R:3")

            if all(c in df.columns for c in ["fx", "fy", "fz"]):
                properties.append("force:R:3")

            # Add other columns to properties
            for col in df.columns:
                if col not in _STANDARD_COLS:
                    dtype = df.schema[col]
                    if dtype == pl.Utf8:
                        properties.append(f"{col}:S:1")
                    elif dtype in [pl.Float32, pl.Float64]:
                        properties.append(f"{col}:R:1")
                    elif dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
                        properties.append(f"{col}:I:1")

            info_parts.append(f"Properties={':'.join(properties)}")

        # Add global_info metadata
        if system.global_info:
            for key, value in system.global_info.items():
                try:
                    value_str = str(value)
                    if not value_str.startswith("<") and not value_str.startswith("["):
                        if "energy" in key:
                            info_parts.append(f"{key}={value_str}")
                        else:
                            info_parts.append(f'{key}="{value_str}"')
                except Exception:
                    continue

        # Write comment line
        if info_parts:
            f.write(" ".join(info_parts) + "\n")
        else:
            f.write("\n")

        # Determine column write order
        write_columns = []
        if "element" in df.columns:
            write_columns.append("element")
        write_columns.extend(["x", "y", "z"])

        if is_extended:
            # Extended format: write additional columns
            if all(c in df.columns for c in ["vx", "vy", "vz"]):
                write_columns.extend(["vx", "vy", "vz"])
            if all(c in df.columns for c in ["fx", "fy", "fz"]):
                write_columns.extend(["fx", "fy", "fz"])

            # Add remaining custom columns
            for col in df.columns:
                if col not in _STANDARD_COLS:
                    write_columns.append(col)

        # Write data rows
        for row in df.select(write_columns).iter_rows():
            line_data = []
            for val in row:
                if isinstance(val, float):
                    line_data.append(f"{val:.10f}")
                else:
                    line_data.append(str(val))
            f.write(" ".join(line_data) + "\n")


if __name__ == "__main__":
    from time import time

    start = time()
    systems = XYZTrajectory(r"C:\Users\HerrW\Desktop\test.xyz")
    print(f"serial time: {time() - start} s.")
    systems.save("t.xyz", 0)

    # start = time()
    # systems = XYZTrajectory(r"C:\Users\HerrW\Desktop\test.xyz", fast_mode=True)
    # print(f"fast mode: {time() - start} s.")

    # systems.save('test2.xyz', [0, 10, 20])
