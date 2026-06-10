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

    def __getitem__(self, idx):
        """Index by ``int`` (single frame), ``slice``, ``list``/``tuple`` or
        a 1-D ``numpy.ndarray`` (integer or boolean).

        - ``traj[3]`` → :class:`System` (the frame at index 3)
        - ``traj[1:4]`` → same-type container holding 3 frames
        - ``traj[[0, 5, 7]]`` → same-type container holding 3 frames
        - ``traj[np.array([True, False, True, ...])]`` → frames at the
          ``True`` positions (the mask must have length ``len(traj)``)

        Boolean masks support filtering on derived per-frame quantities,
        e.g. ``traj[traj.get_atoms_count() > 100]``.
        """
        if isinstance(idx, slice):
            # Wrap the slice in the same concrete subclass so users
            # who do `traj[:5]` get back the same class they started
            # with (XYZTrajectory or Trajectory).
            return type(self)(systems=self._systems[idx])

        # Fancy indexing: bool mask or integer index array / list / tuple.
        if isinstance(idx, (list, tuple, np.ndarray)):
            arr = np.asarray(idx)
            if arr.dtype == bool:
                if arr.shape != (len(self._systems),):
                    raise IndexError(
                        f"boolean mask must have length {len(self._systems)} "
                        f"to index a {len(self._systems)}-frame trajectory; "
                        f"got length {arr.shape[0] if arr.ndim else 'scalar'}."
                    )
                picked = [self._systems[i] for i in np.flatnonzero(arr)]
            elif np.issubdtype(arr.dtype, np.integer):
                # Allow negative indices the same way numpy does.
                n = len(self._systems)
                norm = [int(i) + n if int(i) < 0 else int(i) for i in arr]
                for i in norm:
                    if i < 0 or i >= n:
                        raise IndexError(
                            f"frame index {i} out of bounds for "
                            f"{n}-frame trajectory."
                        )
                picked = [self._systems[i] for i in norm]
            else:
                raise TypeError(
                    f"trajectory index array must be bool or integer; "
                    f"got dtype {arr.dtype}."
                )
            return type(self)(systems=picked)

        # Plain int — return the underlying System.
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

    def get_atoms_count(self) -> np.ndarray:
        """Per-frame atom counts as a 1-D ``int64`` numpy array.

        Returning a numpy array (rather than a Python list) lets users
        write boolean filters directly:

        >>> hot_frames = traj[traj.get_atoms_count() > 100]
        """
        return np.array([s.N for s in self._systems], dtype=np.int64)

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
        verbose: bool = True,
    ) -> None:
        self._systems: List[System] = []
        self._filename = filename
        self._fast_mode = fast_mode
        self._verbose = verbose

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
            self._systems = self._read_xyz_serial(self._filename, verbose=self._verbose)

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

    def _read_xyz_serial(self, filename: str, verbose: bool = False) -> List[System]:
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
                    # XYZ is a streaming format — we don't know the total
                    # frame count without a full pre-scan, so print just
                    # the running counter and refresh the same line.
                    print(
                        f"\r  [xyz.serial] frame {frame_idx} read   ",
                        end="",
                        flush=True,
                    )

        if verbose:
            print(f"\r  [xyz.serial] done — {frame_idx} frames        ", flush=True)
        return systems

    def save(
        self,
        filename: str,
        frames: Optional[Union[List[int], int]] = None,
        mode: str = "w",
        vacuum: float = 0.0,
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
        vacuum : float, default=0.0
            When ``> 0``, every non-periodic axis of every saved frame
            is padded by ``vacuum`` Å (atoms shifted by ``vacuum / 2``
            so they sit centred in the new cell, boundary flipped to
            periodic). Useful for auto-boxing classical-XYZ frames so
            downstream MD code sees a well-defined supercell. The
            in-memory trajectory is not mutated — padding is applied
            to a per-frame copy at write time.

        Examples
        --------
        >>> traj.save("output.xyz")              # save all frames
        >>> traj.save("output.xyz", 0)           # save first frame only
        >>> traj.save("output.xyz", [0, 5, 10])  # save specific frames
        >>> traj.save("output.xyz", vacuum=200)  # auto-box FFF frames
        """
        if vacuum < 0:
            raise ValueError(f"vacuum must be >= 0, got {vacuum}.")
        if frames is None:
            systems_to_save = self._systems
        elif isinstance(frames, int):
            systems_to_save = [self._systems[frames]]
        else:
            systems_to_save = [self._systems[i] for i in frames]
        assert mode in ["w", "a"]
        with open(filename, mode) as f:
            for system in systems_to_save:
                if vacuum > 0:
                    system = _pad_with_vacuum(system, vacuum)
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

        # "classical" here means *no explicit cell* — the box is derived
        # from the coordinate extents rather than read from a Lattice=
        # field. It is NOT the same as "has no per-column Properties":
        # ASE/GPUMD write non-periodic frames (e.g. isolated monomers,
        # pbc="F F F") that omit Lattice but still carry a full
        # Properties string declaring force/energy columns. So the
        # column schema must come from Properties whenever it is present,
        # independent of whether a cell exists — otherwise those frames
        # silently lose every column past x/y/z (forces, etc.).
        classical = "lattice" not in global_info

        if "properties" in global_info:
            columns, schema = self._parse_properties(global_info["properties"])
        elif not classical:
            raise ValueError("Extended XYZ must contain 'properties'")
        else:
            columns = ["element", "x", "y", "z"]
            schema = {
                "element": pl.Utf8,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
            }

        if "pbc" in global_info:
            boundary = [
                1 if i in ("T", "1") else 0 for i in global_info["pbc"].split()
            ]
        else:
            boundary = [0, 0, 0] if classical else [1, 1, 1]

        origin = np.zeros(3, float)
        if "origin" in global_info:
            origin = np.array(global_info["origin"].split(), float)

        if not classical:
            box_array = np.array(global_info["lattice"].split(), float).reshape(3, 3)

        # Per-frame parsing strategy (chosen empirically; see the
        # benchmarks in tests/_generate_fixtures/README or the release
        # notes for the numbers):
        #   * uniform single-space block → `pl.read_csv` (fastest;
        #     ~0.4 ms / 1k atoms when the text really is single-space).
        #   * any irregular whitespace   → numpy `str.split` + per-column
        #     `pl.Series(.astype(...))`. Hot-cache head-to-head on a
        #     2616×514 multi-space training set gave numpy 2.2 s vs
        #     pure-Python dict 2.6 s; on a 7343×~150 13-column file
        #     it was numpy 4.3 s vs dict 5.7 s. `np.loadtxt` was the
        #     slowest (3+ ms / 1k atoms) — Python-side parser overhead.
        # Bulk-vectorised reading remains available via `fast_mode=True`
        # which amortises ALL per-frame work (regex on the comment
        # line, `_parse_properties`, Box construction) into one pass.
        from mdapy.load_save import _is_uniform_single_space
        import io as _io

        if _is_uniform_single_space(data_lines, len(columns)):
            buf = _io.StringIO("".join(data_lines))
            df = pl.read_csv(buf, separator=" ", schema=schema, has_header=False)
        else:
            cells = np.array([row.split()[: len(columns)] for row in data_lines])
            df_cols = {}
            for j, c in enumerate(columns):
                col = cells[:, j]
                if schema[c] == pl.Int32:
                    df_cols[c] = pl.Series(c, col.astype(np.int32), dtype=pl.Int32)
                elif schema[c] == pl.Utf8:
                    df_cols[c] = pl.Series(c, col.tolist(), dtype=pl.Utf8)
                else:
                    df_cols[c] = pl.Series(c, col.astype(np.float64), dtype=pl.Float64)
            df = pl.DataFrame(df_cols)

        if classical:
            coor = df.select("x", "y", "z")
            extents = (coor.max() - coor.min()).to_numpy().flatten()
            # Pad zero extents (e.g. a single atom or a planar config)
            # so Box() can still invert the cell matrix.
            extents = np.where(extents > 0, extents, 1e-9)
            box_array = np.diag(extents)

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
                # GPUMD writes "unwrapped_position:R:3" for trajectories
                # already unwrapped at the simulator side. Map it to the
                # LAMMPS-style ``xu/yu/zu`` triplet so downstream code
                # (``unwrap_trajectory``, MSD, etc.) sees a uniform column
                # name regardless of source.
                content[i] in ("unwrapped_position", "unwrapped_pos")
                and content[i + 1] == "R"
                and n_col == 3
                and "xu" not in schema
            ):
                columns.extend(["xu", "yu", "zu"])
                for coord in ["xu", "yu", "zu"]:
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


def _pad_with_vacuum(system: System, vacuum: float) -> System:
    """Return a new :class:`System` with a vacuum buffer added along
    every non-periodic axis.

    For each axis ``i`` whose boundary is open (``box.boundary[i] == 0``)
    the cell is extended by ``vacuum`` Å, the atoms are shifted by
    ``vacuum / 2`` along ``i`` so the original cluster sits centred in
    the padded box, and the boundary on those axes flips to periodic.
    Periodic axes are left untouched. ``vacuum == 0`` is a no-op and
    returns the input system unchanged.

    Used by the trajectory writers to auto-box training-set frames that
    came in as classical XYZ (PBC = FFF) so downstream MD code sees a
    well-defined supercell.
    """
    if vacuum < 0:
        raise ValueError(f"vacuum must be >= 0, got {vacuum}.")
    if vacuum == 0:
        return system
    boundary = list(system.box.boundary)
    if all(b == 1 for b in boundary):
        return system  # nothing to pad

    new_box_mat = np.asarray(system.box.box, dtype=float).copy()
    new_origin = np.asarray(system.box.origin, dtype=float).copy()
    new_boundary = list(boundary)
    shift = np.zeros(3, dtype=float)
    for i in range(3):
        if boundary[i] == 0:
            new_box_mat[i, i] += vacuum
            shift[i] = vacuum / 2.0
            new_boundary[i] = 1

    new_data = system.data.with_columns(
        pl.col("x") + shift[0],
        pl.col("y") + shift[1],
        pl.col("z") + shift[2],
    )
    return System(
        data=new_data,
        box=Box(new_box_mat, new_boundary, new_origin),
        global_info=system.global_info,
    )


def _write_xyz_frame_to(f: TextIO, system: System,
                        vacuum: float = 0.0) -> None:
    """Write one XYZ frame to an open text file. Same logic as
    :meth:`XYZTrajectory._write_single_frame` but as a free function so
    :class:`Trajectory` can call it without instantiating XYZTrajectory.

    When ``vacuum > 0`` and the system has any non-periodic axis, the
    written frame is the result of :func:`_pad_with_vacuum` — a padded
    copy of the input. The original ``system`` object is not mutated.
    """
    if vacuum > 0:
        system = _pad_with_vacuum(system, vacuum)
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


def _write_multi_xyz(filename: str, systems: List[System], mode: str = "w",
                     vacuum: float = 0.0) -> None:
    """Write a list of System frames to an XYZ file.

    ``vacuum`` is forwarded to :func:`_write_xyz_frame_to` per frame,
    so each frame's open axes get padded independently — useful when
    the trajectory is a mix of classical and extended frames.
    """
    if mode not in ("w", "a"):
        raise ValueError(f"mode must be 'w' or 'a', got {mode!r}")
    with open(filename, mode) as f:
        for s in systems:
            _write_xyz_frame_to(f, s, vacuum=vacuum)


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
    """In-place progress bar used by the verbose=True trajectory loaders.

    Refreshes a single line via carriage-return so the terminal doesn't
    fill up with hundreds of progress messages. The line stays visible
    after completion (we end with ``\\n`` on the final tick) so the user
    can still tell what the file was.

    Layout::

        [xyz.serial] [#####.....]  500/1000 (50%)\\r
    """
    if not (current == total or (current and current % every == 0)):
        return
    pct = 100.0 * current / max(total, 1)
    bar_w = 30
    filled = int(round(pct / 100.0 * bar_w))
    bar = "#" * filled + "." * (bar_w - filled)
    end = "\n" if current == total else ""
    # Trailing space pads over any leftover characters from a wider line.
    print(
        f"  [{stream_name}] [{bar}]  {current}/{total} ({pct:.0f}%)   ",
        end=end + ("\r" if not end else ""),
        flush=True,
    )


def _read_multi_dump_serial(filename: str, verbose: bool = False) -> List[System]:
    """Split a LAMMPS dump file into frames at every ITEM: TIMESTEP and
    parse each one with `BuildSystem.parse_dump_frame`. Each per-frame
    parser already uses ``pl.read_csv`` on uniform-space blocks (see
    ``_parse_dump_frame_impl``), so the dump path is already
    vectorised — there's no separate `fast_mode` path that would add
    measurable speedup, by design.
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
        Use the vectorised XYZ reader (`_read_xyz_fast`). Amortises
        per-frame overhead (regex-parsing the comment line, building
        the column schema, constructing :class:`Box`) over a single
        bulk pass — typically 5–7× faster on a long, regular XYZ
        trajectory. Requires identical column schema across frames AND
        single-character whitespace separators in the per-atom block;
        raises a ``ValueError`` naming the offending frame otherwise.
        ``fast_mode=True`` is **not supported for LAMMPS dump** — the
        dump serial reader already vectorises each frame internally
        (via ``pl.read_csv``), so a separate "bulk" path adds
        complexity without measurable speedup. Pass ``fast_mode=True``
        on a dump file and you get a ``ValueError`` saying so.
    verbose : bool, default=True
        Print a one-line ``frame i/N`` progress update every 200
        frames during loading. Pass ``verbose=False`` to silence the
        output (useful for tests / scripts).

    Examples
    --------
    >>> traj = mp.Trajectory("dump.lammpstrj")
    >>> for frame in traj: print(frame.N)
    >>> traj.save("subset.xyz", frames=[0, 2, 4])
    >>> traj.append(other_system)
    >>> # boolean / integer-array indexing for filtering frames
    >>> hot = traj[traj.get_atoms_count() > 100]      # bool mask
    >>> first_few = traj[[0, 5, 7]]                    # integer mask
    >>> # silence progress output
    >>> traj = mp.Trajectory("big.xyz", fast_mode=True, verbose=False)
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        systems: Optional[List[System]] = None,
        format: Optional[str] = None,
        fast_mode: bool = False,
        verbose: bool = True,
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
                xt._systems = xt._read_xyz_serial(self._filename, verbose=self._verbose)
            self._systems = xt._systems
        elif fmt == "dump":
            if self._fast_mode:
                raise ValueError(
                    "fast_mode is not supported for LAMMPS dump format. "
                    "The dump serial reader already vectorises each "
                    "frame internally via pl.read_csv, so a separate "
                    "bulk path would add complexity without measurable "
                    "speedup. Pass fast_mode=False (the default)."
                )
            self._systems = _read_multi_dump_serial(
                self._filename, verbose=self._verbose
            )
        else:
            raise ValueError(f"Unsupported trajectory format: {fmt!r}")

    def save(
        self,
        filename: str,
        frames: Optional[Union[int, List[int]]] = None,
        mode: str = "w",
        format: Optional[str] = None,
        vacuum: float = 0.0,
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
        vacuum : float, default=0.0
            **XYZ output only.** When ``> 0``, every non-periodic axis
            of every saved frame is padded by ``vacuum`` Å (atoms
            shifted by ``vacuum / 2`` so they sit centred in the new
            cell, boundary flipped to periodic). Useful for
            auto-boxing training-set frames that came in as classical
            XYZ (PBC = FFF) so downstream MD code sees a well-defined
            supercell. The in-memory trajectory is **not** mutated —
            padding is applied to a per-frame copy at write time.
            Ignored for the dump format (LAMMPS dumps already require
            an explicit box).
        """
        if vacuum < 0:
            raise ValueError(f"vacuum must be >= 0, got {vacuum}.")
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
            _write_multi_xyz(filename, sel, mode, vacuum=vacuum)
        elif fmt == "dump":
            if vacuum > 0:
                import warnings
                warnings.warn(
                    "vacuum>0 is ignored for LAMMPS dump output (dumps "
                    "already require a fully defined box).",
                    UserWarning, stacklevel=2,
                )
            _write_multi_dump(filename, sel, mode)
        else:
            raise ValueError(f"Unsupported trajectory format: {fmt!r}")

    # NOTE: list-like API (__len__, __getitem__, __setitem__, __iter__,
    # __repr__, append, extend, insert, pop, remove, get_atoms_count,
    # concatenate) is inherited from `_TrajectoryListBase`.

    def unwrap(self) -> "Trajectory":
        """Return a new :class:`Trajectory` with continuous (unwrapped)
        particle positions. See :func:`mdapy.unwrap_trajectory` for the
        full algorithm description and edge cases."""
        from mdapy.unwrap_trajectory import unwrap_trajectory

        return unwrap_trajectory(self)


if __name__ == "__main__":
    traj = Trajectory("/u/22/wuy33/unix/Desktop/GAP_CN/gap_cn_training_dataset.xyz")
    print(traj)
