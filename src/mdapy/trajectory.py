# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from typing import List, Optional, Dict, Any, Union, Iterator, Tuple, TextIO
import numpy as np
import polars as pl
import re
from mdapy.system import System
from mdapy.box import Box

# Standard column names that have special meaning in XYZ format
_STANDARD_COLS = (
    "element",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "fx",
    "fy",
    "fz",
    "bec_0",
    "bec_1",
    "bec_2",
    "bec_3",
    "bec_4",
    "bec_5",
    "bec_6",
    "bec_7",
    "bec_8",
)


class _TrajectoryListBase:
    """Common list-like API shared by :class:`XYZTrajectory` and
    :class:`Trajectory`. Subclasses are expected to own a
    ``self._systems: list[System]`` attribute and a ``save`` method.

    Splitting the list-API into a mixin keeps the per-format readers
    (which still live on :class:`XYZTrajectory`) free of bookkeeping
    code, and eliminates the duplicated list-method implementations
    that used to live on both classes.
    """

    _systems: List[System]

    def __len__(self) -> int:
        return len(self._systems)

    def __getitem__(self, idx: Union[int, slice]):
        if isinstance(idx, slice):
            # Wrap the slice in the same concrete subclass so users
            # who do `traj[:5]` get back the same class they started
            # with (XYZTrajectory or Trajectory).
            return type(self)(systems=self._systems[idx])
        return self._systems[idx]

    def __setitem__(self, idx: int, system: System) -> None:
        if not isinstance(system, System):
            raise TypeError("can only assign System instances")
        self._systems[idx] = system

    def __iter__(self) -> Iterator[System]:
        return iter(self._systems)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {len(self)} frame(s)>"

    def append(self, system: System) -> None:
        if not isinstance(system, System):
            raise TypeError("only System instances can be appended")
        self._systems.append(system)

    def extend(self, systems: List[System]) -> None:
        for s in systems:
            self.append(s)

    def insert(self, index: int, system: System) -> None:
        if not isinstance(system, System):
            raise TypeError("only System instances can be inserted")
        self._systems.insert(index, system)

    def pop(self, index: int = -1) -> System:
        return self._systems.pop(index)

    def remove(self, indices: Union[int, List[int]]) -> None:
        if isinstance(indices, int):
            indices = [indices]
        for i in sorted(indices, reverse=True):
            self._systems.pop(i)

    def get_atoms_count(self) -> List[int]:
        """Per-frame atom counts."""
        return [s.N for s in self._systems]

    def concatenate(self, other: "_TrajectoryListBase") -> "_TrajectoryListBase":
        """Return a new container holding ``self`` followed by ``other``."""
        return type(self)(systems=self._systems + other._systems)


class XYZTrajectory(_TrajectoryListBase):
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

        # Get the first data line to determine the actual separator. We
        # match the whitespace run immediately after the first
        # non-whitespace token rather than using `str.find` on the
        # second token — `find` returns the FIRST occurrence and if the
        # second token happens to equal the first (e.g. `0.5 0.5 ...`)
        # it would mis-identify the separator as the empty string.
        first_data_line: str = df_raw.item(row_list[0] + 2, 1)
        sep_match = re.match(r"\S+(\s+)", first_data_line)
        separator = sep_match.group(1) if sep_match else " "

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

    def _read_xyz_serial(self, filename: str, verbose: bool = False
                          ) -> List[System]:
        """
        Serial trajectory reading mode. Slower than the fast path but
        tolerant of multi-space separators and per-frame schema drift.

        Parameters
        ----------
        filename : str
            Path to XYZ file.
        verbose : bool, default=False
            Print progress every 200 frames during the read.
        """
        systems = []
        frame_idx = 0
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
                frame_idx += 1
                if verbose and frame_idx % 200 == 0:
                    print(f"  [xyz.serial] frame {frame_idx} read", flush=True)

        if verbose:
            print(f"  [xyz.serial] done — {frame_idx} frames", flush=True)
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

    # NOTE: list-like API (__len__, __getitem__, __setitem__, __iter__,
    # __repr__, append, extend, insert, pop, remove, get_atoms_count,
    # concatenate) is inherited from `_TrajectoryListBase` so it stays
    # in sync between XYZTrajectory and Trajectory.

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

            # Magic-name mapping is applied only for the *first*
            # occurrence; if a second property in the same Properties
            # string would alias to the same canonical names (e.g. both
            # `force:R:3` and `forces:R:3` → fx/fy/fz), the alias is
            # disabled for the second one and it falls through to the
            # generic `<name>_<j>` path. This keeps every column unique
            # so the data-row split lines up with the column list.
            if (
                content[i] == "pos"
                and content[i + 1] == "R"
                and n_col == 3
                and "x" not in schema
            ):
                columns.extend(["x", "y", "z"])
                for coord in ["x", "y", "z"]:
                    schema[coord] = dtype
            elif (
                content[i] in ["species", "element"]
                and content[i + 1] == "S"
                and n_col == 1
                and "element" not in schema
            ):
                columns.append("element")
                schema["element"] = dtype
            elif (
                content[i] in ["velo", "vel"]
                and content[i + 1] == "R"
                and n_col == 3
                and "vx" not in schema
            ):
                columns.extend(["vx", "vy", "vz"])
                for vel in ["vx", "vy", "vz"]:
                    schema[vel] = dtype
            elif (
                content[i] in ["force", "forces"]
                and content[i + 1] == "R"
                and n_col == 3
                and "fx" not in schema
            ):
                columns.extend(["fx", "fy", "fz"])
                for force in ["fx", "fy", "fz"]:
                    schema[force] = dtype
            else:
                if n_col > 1:
                    for j in range(n_col):
                        col_name = f"{content[i]}_{j}"
                        # Defensive: if the generic name *also*
                        # collides (extremely rare but possible when
                        # two properties share a base name), tack on
                        # extra suffixes until unique.
                        suffix = 0
                        unique_name = col_name
                        while unique_name in schema:
                            suffix += 1
                            unique_name = f"{col_name}__{suffix}"
                        columns.append(unique_name)
                        schema[unique_name] = dtype
                else:
                    base = content[i]
                    suffix = 0
                    unique_name = base
                    while unique_name in schema:
                        suffix += 1
                        unique_name = f"{base}__{suffix}"
                    columns.append(unique_name)
                    schema[unique_name] = dtype
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

            if all(
                c in df.columns
                for c in [
                    "bec_0",
                    "bec_1",
                    "bec_2",
                    "bec_3",
                    "bec_4",
                    "bec_5",
                    "bec_6",
                    "bec_7",
                    "bec_8",
                ]
            ):
                properties.append("bec:R:9")

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
            if all(
                c in df.columns
                for c in [
                    "bec_0",
                    "bec_1",
                    "bec_2",
                    "bec_3",
                    "bec_4",
                    "bec_5",
                    "bec_6",
                    "bec_7",
                    "bec_8",
                ]
            ):
                write_columns.extend(
                    [
                        "bec_0",
                        "bec_1",
                        "bec_2",
                        "bec_3",
                        "bec_4",
                        "bec_5",
                        "bec_6",
                        "bec_7",
                        "bec_8",
                    ]
                )

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


# ===========================================================================
#  Module-level XYZ writers (used by both XYZTrajectory and Trajectory)
# ===========================================================================


def _write_xyz_frame_to(f: TextIO, system: System) -> None:
    """Write one XYZ frame to an open text file. Same logic as
    :meth:`XYZTrajectory._write_single_frame` but as a free function so
    :class:`Trajectory` can call it without instantiating XYZTrajectory."""
    df = system.data
    natom = len(df)
    f.write(f"{natom}\n")
    info_parts = []
    is_extended = system.box.boundary.sum() > 0
    if is_extended:
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
        bec_cols = [f"bec_{i}" for i in range(9)]
        if all(c in df.columns for c in bec_cols):
            properties.append("bec:R:9")
        for col in df.columns:
            if col not in _STANDARD_COLS:
                dtype = df.schema[col]
                if dtype == pl.Utf8:
                    properties.append(f"{col}:S:1")
                elif dtype in (pl.Float32, pl.Float64):
                    properties.append(f"{col}:R:1")
                elif dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
                    properties.append(f"{col}:I:1")
        info_parts.append(f"Properties={':'.join(properties)}")

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

    if info_parts:
        f.write(" ".join(info_parts) + "\n")
    else:
        f.write("\n")

    write_columns = []
    if "element" in df.columns:
        write_columns.append("element")
    write_columns.extend(["x", "y", "z"])
    if is_extended:
        if all(c in df.columns for c in ["vx", "vy", "vz"]):
            write_columns.extend(["vx", "vy", "vz"])
        if all(c in df.columns for c in ["fx", "fy", "fz"]):
            write_columns.extend(["fx", "fy", "fz"])
        bec_cols = [f"bec_{i}" for i in range(9)]
        if all(c in df.columns for c in bec_cols):
            write_columns.extend(bec_cols)
        for col in df.columns:
            if col not in _STANDARD_COLS:
                write_columns.append(col)

    for row in df.select(write_columns).iter_rows():
        line_data = []
        for val in row:
            if isinstance(val, float):
                line_data.append(f"{val:.10f}")
            else:
                line_data.append(str(val))
        f.write(" ".join(line_data) + "\n")


def _write_multi_xyz(filename: str, systems: List[System], mode: str = "w") -> None:
    """Write a list of System frames to an XYZ file."""
    if mode not in ("w", "a"):
        raise ValueError(f"mode must be 'w' or 'a', got {mode!r}")
    with open(filename, mode) as f:
        for s in systems:
            _write_xyz_frame_to(f, s)


# ===========================================================================
#  Unified multi-frame trajectory: XYZ or LAMMPS dump (read + write)
# ===========================================================================


def _infer_trajectory_format(filename: str) -> str:
    """Infer 'xyz' vs 'dump' from the filename, accepting `.gz` suffix."""
    f = filename.lower()
    if f.endswith(".gz"):
        f = f[:-3]
    if f.endswith(".xyz"):
        return "xyz"
    if f.endswith(".dump") or f.endswith(".lammpstrj"):
        return "dump"
    raise ValueError(
        f"Cannot infer trajectory format from {filename!r}; "
        f"pass format='xyz' or format='dump' explicitly."
    )


def _progress(stream_name: str, current: int, total: int, every: int = 200) -> None:
    """Tiny progress printer used by the verbose=True trajectory loaders.
    Prints one update every `every` frames plus a final tick at completion."""
    if current == total or (current and current % every == 0):
        pct = 100.0 * current / max(total, 1)
        print(f"  [{stream_name}] frame {current}/{total} ({pct:.0f}%)",
              flush=True)


def _read_multi_dump_serial(filename: str, verbose: bool = False) -> List[System]:
    """Split a LAMMPS dump file into frames at every ITEM: TIMESTEP and
    parse each one with `BuildSystem.parse_dump_frame`. Robust to
    multi-space separators and per-frame schema variation; significantly
    slower than :func:`_read_multi_dump_fast` on regular dumps.
    """
    from mdapy.load_save import _open_file, BuildSystem

    with _open_file(filename, "r") as fp:
        lines = fp.readlines()

    ts_idx = [i for i, l in enumerate(lines) if l.strip().startswith("ITEM: TIMESTEP")]
    if not ts_idx:
        raise ValueError(f"{filename}: no ITEM: TIMESTEP header found")

    boundaries = ts_idx + [len(lines)]
    n_frames = len(ts_idx)
    systems: List[System] = []
    for i in range(n_frames):
        frame = lines[boundaries[i] : boundaries[i + 1]]
        df, box, info = BuildSystem.parse_dump_frame(
            frame, source=f"{filename}[frame {i}]"
        )
        systems.append(System(data=df, box=box, global_info=info))
        if verbose:
            _progress("dump.serial", i + 1, n_frames)
    return systems


# Back-compat alias.
_read_multi_dump = _read_multi_dump_serial


def _read_multi_dump_fast(filename: str, verbose: bool = False) -> List[System]:
    """Vectorised LAMMPS dump reader.

    Assumes:
      * every frame uses the **same** ``ITEM: ATOMS`` column layout;
      * every frame uses the **same** ``ITEM: BOX BOUNDS`` geometry
        keywords (orthogonal vs. ``xy xz yz`` triclinic vs. ``abc origin``
        general triclinic);
      * the per-atom data block is whitespace-separated by single
        characters (no tabs, no runs of multiple spaces).

    Atom data is bulk-parsed via a single ``pl.read_csv`` over a
    StringIO of all frames concatenated, then partitioned by frame.
    Per-frame box / timestep parsing stays in Python — that's O(n_frames)
    and never the bottleneck.

    Falls back gracefully with a ``ValueError`` whose message names the
    failing frame when an assumption is violated.
    """
    from mdapy.load_save import _open_file
    import io as _io

    with _open_file(filename, "r") as fp:
        lines = fp.readlines()

    ts_idx = [i for i, l in enumerate(lines) if l.strip().startswith("ITEM: TIMESTEP")]
    if not ts_idx:
        raise ValueError(f"{filename}: no ITEM: TIMESTEP header found")
    n_frames = len(ts_idx)
    boundaries = ts_idx + [len(lines)]

    # ----- Parse each frame's 9-line header in Python ------------------
    timesteps: List[int] = []
    n_atoms_list: List[int] = []
    box_per_frame: List[Tuple[np.ndarray, List[int]]] = []
    atom_blocks: List[List[str]] = []  # list of (variable-length) line lists

    expected_cols: Optional[List[str]] = None
    expected_geom: Optional[Tuple[bool, bool]] = None  # (is_abc, is_triclinic)

    for f_idx in range(n_frames):
        frame = lines[boundaries[f_idx] : boundaries[f_idx + 1]]
        if len(frame) < 9:
            raise ValueError(
                f"{filename}[frame {f_idx}]: dump frame has only "
                f"{len(frame)} lines (<9); use fast_mode=False for "
                "tolerant parsing."
            )
        try:
            timesteps.append(int(frame[1].strip()))
            n_atoms_list.append(int(frame[3].strip()))
        except ValueError as e:
            raise ValueError(
                f"{filename}[frame {f_idx}]: malformed TIMESTEP / "
                f"NUMBER OF ATOMS — {e}"
            ) from None

        bb_line = frame[4].strip()
        if not bb_line.startswith("ITEM: BOX BOUNDS"):
            raise ValueError(
                f"{filename}[frame {f_idx}]: expected 'ITEM: BOX BOUNDS' "
                "on header line 5."
            )
        bb_tokens = bb_line.split()[3:]
        if bb_tokens and all(t in {"pp", "ff", "ss", "mm"} for t in bb_tokens[-3:]):
            boundary = [1 if t == "pp" else 0 for t in bb_tokens[-3:]]
            geom_tok = bb_tokens[:-3]
        else:
            boundary = [1, 1, 1]
            geom_tok = bb_tokens
        is_abc = "abc" in geom_tok and "origin" in geom_tok
        is_triclinic = ("xy" in geom_tok and "xz" in geom_tok and "yz" in geom_tok)
        if expected_geom is None:
            expected_geom = (is_abc, is_triclinic)
        elif expected_geom != (is_abc, is_triclinic):
            raise ValueError(
                f"{filename}[frame {f_idx}]: BOX BOUNDS geometry differs "
                f"from frame 0 — fast_mode requires identical geometry "
                "across frames; use fast_mode=False."
            )

        bb_rows = [frame[5].split(), frame[6].split(), frame[7].split()]
        if is_abc:
            avec = np.array(bb_rows[0][:3], dtype=np.float64)
            bvec = np.array(bb_rows[1][:3], dtype=np.float64)
            cvec = np.array(bb_rows[2][:3], dtype=np.float64)
            origin = np.array([bb_rows[0][3], bb_rows[1][3], bb_rows[2][3]],
                              dtype=np.float64)
            box = np.vstack([avec, bvec, cvec, origin])
        elif is_triclinic:
            xlo_b, xhi_b, xy = (float(bb_rows[0][0]), float(bb_rows[0][1]),
                                 float(bb_rows[0][2]))
            ylo_b, yhi_b, xz = (float(bb_rows[1][0]), float(bb_rows[1][1]),
                                 float(bb_rows[1][2]))
            zlo_b, zhi_b, yz = (float(bb_rows[2][0]), float(bb_rows[2][1]),
                                 float(bb_rows[2][2]))
            xlo = xlo_b - min(0.0, xy, xz, xy + xz)
            xhi = xhi_b - max(0.0, xy, xz, xy + xz)
            ylo = ylo_b - min(0.0, yz)
            yhi = yhi_b - max(0.0, yz)
            zlo, zhi = zlo_b, zhi_b
            box = np.array([
                [xhi - xlo, 0,         0        ],
                [xy,        yhi - ylo, 0        ],
                [xz,        yz,        zhi - zlo],
                [xlo,       ylo,       zlo      ],
            ])
        else:
            xlo, xhi = float(bb_rows[0][0]), float(bb_rows[0][1])
            ylo, yhi = float(bb_rows[1][0]), float(bb_rows[1][1])
            zlo, zhi = float(bb_rows[2][0]), float(bb_rows[2][1])
            box = np.array([
                [xhi - xlo, 0,         0        ],
                [0,         yhi - ylo, 0        ],
                [0,         0,         zhi - zlo],
                [xlo,       ylo,       zlo      ],
            ])
        box_per_frame.append((box, boundary))

        atoms_header = frame[8].rstrip()
        if not atoms_header.startswith("ITEM: ATOMS"):
            raise ValueError(
                f"{filename}[frame {f_idx}]: expected 'ITEM: ATOMS' "
                "on header line 9."
            )
        cols = atoms_header.split()[2:]
        if expected_cols is None:
            expected_cols = cols
        elif cols != expected_cols:
            raise ValueError(
                f"{filename}[frame {f_idx}]: ATOMS column layout "
                f"{cols} differs from frame 0 {expected_cols} — "
                "fast_mode requires identical schema; use fast_mode=False."
            )

        body = frame[9 : 9 + n_atoms_list[-1]]
        if len(body) != n_atoms_list[-1]:
            raise ValueError(
                f"{filename}[frame {f_idx}]: expected "
                f"{n_atoms_list[-1]} atom rows, got {len(body)}."
            )
        atom_blocks.append(body)

    assert expected_cols is not None  # n_frames >= 1 already enforced

    # ----- Bulk-parse the concatenated atom blocks ---------------------
    INT_COLS = {"id", "type", "ix", "iy", "iz", "mol", "proc", "procp1"}
    STR_COLS = {"element", "typelabel"}
    schema: Dict[str, "pl.DataType"] = {}
    for name in expected_cols:
        if name in INT_COLS:
            schema[name] = pl.Int32
        elif name in STR_COLS:
            schema[name] = pl.Utf8
        else:
            schema[name] = pl.Float64

    big_buf = _io.StringIO("".join(line for block in atom_blocks for line in block))
    try:
        all_data = pl.read_csv(big_buf, separator=" ", schema=schema,
                               has_header=False)
    except Exception as e:
        # Most likely: the file has multi-space separators or tabs, which
        # the fast path cannot handle. Surface a clear, actionable error.
        raise ValueError(
            f"{filename}: fast dump reader could not parse the per-atom "
            f"block. The data is probably whitespace-irregular (multiple "
            f"spaces, tabs) or a column has unexpected dtype. Re-run with "
            f"fast_mode=False. Underlying error: {e}"
        ) from None

    if all_data.shape[0] != sum(n_atoms_list):
        raise ValueError(
            f"{filename}: bulk parser returned {all_data.shape[0]} rows; "
            f"expected {sum(n_atoms_list)} from the per-frame headers."
        )

    # ----- Slice into per-frame DataFrames + apply coord normalisation -
    systems: List[System] = []
    cursor = 0
    cols_set = set(expected_cols)
    has_xs = {"xs", "ys", "zs"}.issubset(cols_set)
    has_xsu = {"xsu", "ysu", "zsu"}.issubset(cols_set)
    has_xu = {"xu", "yu", "zu"}.issubset(cols_set)
    for f_idx in range(n_frames):
        n = n_atoms_list[f_idx]
        df = all_data.slice(cursor, n)
        cursor += n
        box, boundary = box_per_frame[f_idx]
        if has_xs or has_xsu:
            tag = "xs" if has_xs else "xsu"
            ty, tz = tag.replace("x", "y"), tag.replace("x", "z")
            scaled = df.select(tag, ty, tz).to_numpy()
            absolute = box[3] + scaled @ box[:3]
            df = df.with_columns(
                x=pl.Series(absolute[:, 0]),
                y=pl.Series(absolute[:, 1]),
                z=pl.Series(absolute[:, 2]),
            ).select(pl.all().exclude(tag, ty, tz))
        elif has_xu:
            df = df.rename({"xu": "x", "yu": "y", "zu": "z"})
        systems.append(System(
            data=df.rechunk(),
            box=Box(box, boundary),
            global_info={"timestep": timesteps[f_idx]},
        ))
        if verbose:
            _progress("dump.fast", f_idx + 1, n_frames)
    return systems


def _write_multi_dump(filename: str, systems: List[System], mode: str) -> None:
    from mdapy.load_save import SaveSystem

    if mode not in ("w", "a"):
        raise ValueError(f"mode must be 'w' or 'a', got {mode!r}")
    with open(filename, mode + "b") as fp:
        for sys_obj in systems:
            ts = sys_obj.global_info.get("timestep", 0) if sys_obj.global_info else 0
            try:
                ts = int(ts)
            except (TypeError, ValueError):
                ts = 0
            SaveSystem.write_dump_frame_to(fp, sys_obj.box, sys_obj.data, ts)


class Trajectory(_TrajectoryListBase):
    """Multi-frame trajectory container — supports XYZ and LAMMPS dump
    (read + write).

    Parameters
    ----------
    filename : str, optional
        Path to load (``.xyz``, ``.dump``, ``.lammpstrj``, optionally
        ``.gz``).
    systems : list of mdapy.System, optional
        Pre-built list of frames.
    format : {'xyz', 'dump'}, optional
        Override file-format detection. Defaults to inferring from the
        filename extension.
    fast_mode : bool, default=False
        Use a vectorised reader (`_read_xyz_fast` for XYZ,
        `_read_multi_dump_fast` for LAMMPS dumps). Requires the file to
        be regular: identical schema across frames AND single-character
        whitespace separators in the per-atom block. Significantly
        faster on large LAMMPS-written files; raises ``ValueError`` with
        a clear message if the file does not satisfy the assumptions.
    verbose : bool, default=False
        Print one ``frame i/N`` progress line every 200 frames during
        loading. Useful for very large trajectories so the user can see
        the read is making progress.

    Examples
    --------
    >>> traj = mp.Trajectory("dump.lammpstrj")
    >>> for frame in traj: print(frame.N)
    >>> traj.save("subset.xyz", frames=[0, 2, 4])
    >>> traj.append(other_system)
    >>> # large LAMMPS dump — fast path, with progress
    >>> traj = mp.Trajectory("big.dump", fast_mode=True, verbose=True)
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        systems: Optional[List[System]] = None,
        format: Optional[str] = None,
        fast_mode: bool = False,
        verbose: bool = False,
    ) -> None:
        self._systems: List[System] = []
        self._filename = filename
        self._format = format
        self._fast_mode = fast_mode
        self._verbose = verbose
        if systems is not None:
            self._systems = list(systems)
        elif filename is not None:
            self._load()
        else:
            raise ValueError("Trajectory needs either filename= or systems=")

    def _load(self) -> None:
        fmt = self._format or _infer_trajectory_format(self._filename)
        if fmt == "xyz":
            # XYZTrajectory owns the XYZ parser pipeline; we instantiate it
            # purely as a parser (no list-API surface; XYZTrajectory now
            # inherits from Trajectory itself, see below).
            xt = XYZTrajectory.__new__(XYZTrajectory)
            xt._filename = self._filename
            xt._fast_mode = self._fast_mode
            xt._verbose = self._verbose
            if self._fast_mode:
                xt._systems = xt._read_xyz_fast(self._filename)
            else:
                xt._systems = xt._read_xyz_serial(self._filename,
                                                  verbose=self._verbose)
            self._systems = xt._systems
        elif fmt == "dump":
            if self._fast_mode:
                self._systems = _read_multi_dump_fast(self._filename,
                                                     verbose=self._verbose)
            else:
                self._systems = _read_multi_dump_serial(self._filename,
                                                       verbose=self._verbose)
        else:
            raise ValueError(f"Unsupported trajectory format: {fmt!r}")

    def save(
        self,
        filename: str,
        frames: Optional[Union[int, List[int]]] = None,
        mode: str = "w",
        format: Optional[str] = None,
    ) -> None:
        """Write the trajectory to disk.

        Parameters
        ----------
        filename : str
            Output path.  Format is auto-detected from the suffix.
        frames : int or list of int, optional
            Subset of frame indices to save.  Default: all frames.
        mode : {'w', 'a'}, default 'w'
            Open mode (write/truncate vs. append).
        format : {'xyz', 'dump'}, optional
            Override format detection.
        """
        if frames is None:
            sel = self._systems
        elif isinstance(frames, int):
            sel = [self._systems[frames]]
        else:
            sel = [self._systems[i] for i in frames]
        if mode not in ("w", "a"):
            raise ValueError(f"mode must be 'w' or 'a', got {mode!r}")

        fmt = format or _infer_trajectory_format(filename)
        if fmt == "xyz":
            _write_multi_xyz(filename, sel, mode)
        elif fmt == "dump":
            _write_multi_dump(filename, sel, mode)
        else:
            raise ValueError(f"Unsupported trajectory format: {fmt!r}")

    # NOTE: list-like API (__len__, __getitem__, __setitem__, __iter__,
    # __repr__, append, extend, insert, pop, remove, get_atoms_count,
    # concatenate) is inherited from `_TrajectoryListBase`.


if __name__ == "__main__":
    traj = XYZTrajectory(
        "/u/22/wuy33/unix/Desktop/GAP_CN/gap_cn_training_dataset.xyz"
    )
    print(traj)
