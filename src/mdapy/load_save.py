# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
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
import io
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


def _is_uniform_single_space(lines: List[str], expected_ncols: int,
                             sample: int = 32) -> bool:
    """Return True iff every sampled line is exactly single-space-separated
    with `expected_ncols` fields (no leading/trailing whitespace, no tabs,
    no runs of multiple spaces).

    Used as a cheap pre-flight before falling back to `pl.read_csv` (fast
    path). When this returns False, callers should parse the block in
    Python with `str.split()` (which handles arbitrary whitespace).
    """
    if not lines:
        return False
    n = min(len(lines), sample)
    for i in range(n):
        raw = lines[i].rstrip("\r\n")
        toks = raw.split()
        if len(toks) != expected_ncols:
            return False
        # Reject tab, leading/trailing space, runs of multi-space.
        if raw != " ".join(toks):
            return False
    return True


def _parse_dump_frame_impl(lines: List[str], source: str
                           ) -> Tuple[pl.DataFrame, Box, Dict[str, Any]]:
    """Parse one LAMMPS dump frame from `lines` (the 9 header lines + N
    atom rows). Pulled out of `BuildSystem.read_dump` so the multi-frame
    Trajectory reader can reuse the exact same parsing logic.
    """
    if len(lines) < 9:
        raise ValueError(f"{source}: dump frame has only {len(lines)} lines (<9)")

    try:
        timestep = int(lines[1].strip())
    except (IndexError, ValueError):
        raise ValueError(f"{source}: malformed ITEM: TIMESTEP value")
    try:
        n_atoms = int(lines[3].strip())
    except (IndexError, ValueError):
        raise ValueError(f"{source}: malformed ITEM: NUMBER OF ATOMS value")

    bb_line = lines[4].strip()
    if not bb_line.startswith("ITEM: BOX BOUNDS"):
        raise ValueError(f"{source}: expected 'ITEM: BOX BOUNDS' on line 5")
    bb_tokens = bb_line.split()[3:]

    if bb_tokens and all(t in {"pp", "ff", "ss", "mm"} for t in bb_tokens[-3:]):
        boundary = [1 if t == "pp" else 0 for t in bb_tokens[-3:]]
        geometry_tokens = bb_tokens[:-3]
    else:
        boundary = [1, 1, 1]
        geometry_tokens = bb_tokens

    bb_rows = [lines[5].split(), lines[6].split(), lines[7].split()]

    if "abc" in geometry_tokens and "origin" in geometry_tokens:
        avec = np.array(bb_rows[0][:3], dtype=np.float64)
        bvec = np.array(bb_rows[1][:3], dtype=np.float64)
        cvec = np.array(bb_rows[2][:3], dtype=np.float64)
        origin = np.array(
            [bb_rows[0][3], bb_rows[1][3], bb_rows[2][3]],
            dtype=np.float64,
        )
        box = np.vstack([avec, bvec, cvec, origin])
    elif "xy" in geometry_tokens and "xz" in geometry_tokens and "yz" in geometry_tokens:
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

    atoms_header = lines[8].rstrip()
    if not atoms_header.startswith("ITEM: ATOMS"):
        raise ValueError(f"{source}: expected 'ITEM: ATOMS' on line 9")
    col_names = atoms_header.split()[2:]

    data_rows = lines[9 : 9 + n_atoms]
    if len(data_rows) != n_atoms:
        raise ValueError(
            f"{source}: expected {n_atoms} atom rows, got {len(data_rows)}"
        )

    INT_COLS = {"id", "type", "ix", "iy", "iz", "mol", "proc", "procp1"}
    STR_COLS = {"element", "typelabel"}

    schema: Dict[str, "pl.DataType"] = {}
    for name in col_names:
        if name in INT_COLS:
            schema[name] = pl.Int32
        elif name in STR_COLS:
            schema[name] = pl.Utf8
        else:
            schema[name] = pl.Float64

    if _is_uniform_single_space(data_rows, len(col_names)):
        buf = io.StringIO("".join(data_rows))
        data = pl.read_csv(buf, separator=" ", schema=schema, has_header=False)
    else:
        cells = np.array([row.split()[: len(col_names)] for row in data_rows])
        df_cols = {}
        for j, name in enumerate(col_names):
            col = cells[:, j]
            if schema[name] == pl.Int32:
                df_cols[name] = pl.Series(name, col.astype(np.int32),
                                          dtype=pl.Int32)
            elif schema[name] == pl.Utf8:
                df_cols[name] = pl.Series(name, col.tolist(), dtype=pl.Utf8)
            else:
                df_cols[name] = pl.Series(name, col.astype(np.float64),
                                          dtype=pl.Float64)
        data = pl.DataFrame(df_cols)

    # Coordinate-variant normalisation. Preference order: an explicit
    # ``x y z`` triplet wins over scaled/unwrapped variants; the other
    # columns are kept under their original names so the user can still
    # access them. Only when ``x y z`` is missing do we promote
    # ``xs ys zs``/``xsu ysu zsu``/``xu yu zu`` into ``x y z``.
    cols = set(data.columns)
    has_xyz = {"x", "y", "z"}.issubset(cols)
    if not has_xyz:
        if {"xs", "ys", "zs"}.issubset(cols) or {"xsu", "ysu", "zsu"}.issubset(cols):
            tag = "xs" if {"xs", "ys", "zs"}.issubset(cols) else "xsu"
            tag_y, tag_z = tag.replace("x", "y"), tag.replace("x", "z")
            scaled = data.select(tag, tag_y, tag_z).to_numpy()
            absolute = box[3] + scaled @ box[:3]
            data = data.with_columns(
                x=pl.Series(absolute[:, 0]),
                y=pl.Series(absolute[:, 1]),
                z=pl.Series(absolute[:, 2]),
            ).select(pl.all().exclude(tag, tag_y, tag_z))
        elif {"xu", "yu", "zu"}.issubset(cols):
            data = data.rename({"xu": "x", "yu": "y", "zu": "z"})

    return data.rechunk(), Box(box, boundary), {"timestep": timestep}


def _build_xyz_properties(columns: List[str],
                          dtypes: List["pl.DataType"]) -> str:
    """Assemble the Properties=… token of an extended-XYZ comment line by
    walking through the column list. Recognises the canonical 3-vector
    aliases pos/vel/forces and folds *consecutive* {x,y,z}/{vx,vy,vz}/
    {fx,fy,fz} runs into a single token. Multi-column user families with
    a `<base>_0`, `<base>_1`, … `<base>_{N-1}` naming pattern are folded
    into `<base>:T:N`. Everything else becomes a plain `<name>:T:1` token.

    The output is independent of column order: a non-canonical layout
    (e.g. element first, then a custom column, then x/y/z) still produces
    a valid extended-XYZ Properties string because the folding only
    triggers on contiguous runs of the canonical names.
    """
    def _ptype(dt):
        if dt.is_integer():
            return "I"
        if dt.is_float():
            return "R"
        if dt == pl.Utf8:
            return "S"
        raise ValueError(f"Unrecognised dtype {dt} in extended XYZ output")

    THREE_ALIASES = [
        (("x",  "y",  "z"),  "pos",                "R"),
        (("xu", "yu", "zu"), "unwrapped_position", "R"),
        (("vx", "vy", "vz"), "vel",                "R"),
        (("fx", "fy", "fz"), "forces",             "R"),
    ]

    tokens: List[str] = []
    i = 0
    while i < len(columns):
        # Try the canonical 3-vector aliases first.
        consumed = False
        for triple, alias, want in THREE_ALIASES:
            if columns[i : i + 3] == list(triple) and \
               all(_ptype(dtypes[i + j]) == want for j in range(3)):
                tokens.append(f"{alias}:{want}:3")
                i += 3
                consumed = True
                break
        if consumed:
            continue

        # element → species:S:1 (the conventional extended-XYZ name).
        if columns[i] == "element":
            tokens.append(f"species:{_ptype(dtypes[i])}:1")
            i += 1
            continue

        # Detect a consecutive `<base>_0 _1 _2 ...` family of same dtype.
        cur = columns[i]
        base, sep, idx = cur.rpartition("_")
        if sep and base and idx.isdigit() and int(idx) == 0:
            run = 1
            same = dtypes[i]
            while i + run < len(columns):
                nb, ns, ni = columns[i + run].rpartition("_")
                if not ns or nb != base or not ni.isdigit() or int(ni) != run:
                    break
                if dtypes[i + run] != same:
                    break
                run += 1
            if run >= 2:
                tokens.append(f"{base}:{_ptype(same)}:{run}")
                i += run
                continue

        tokens.append(f"{cur}:{_ptype(dtypes[i])}:1")
        i += 1

    return "Properties=" + ":".join(tokens)


def _parse_masses_to_elements(lines: List[str], n_types: int) -> Optional[Dict[int, str]]:
    """Parse a LAMMPS Masses block. Each row is `<type> <mass> [# <element>]`.
    Return type→element mapping when EVERY row has a parseable element
    comment; otherwise return None (the caller leaves out the `element`
    column).

    Used by ``read_data`` to round-trip through `write_data(..., element_list=...)`.
    """
    mapping: Dict[int, str] = {}
    parsed = 0
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" not in line:
            continue
        head, comment = line.split("#", 1)
        head_toks = head.split()
        if len(head_toks) < 2:
            continue
        try:
            t = int(head_toks[0])
        except ValueError:
            continue
        elem = comment.strip().split()[0] if comment.strip() else ""
        if not elem:
            continue
        mapping[t] = elem
        parsed += 1
        if parsed >= n_types:
            break
    if parsed != n_types:
        return None
    return mapping


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
        # Place the temporary file in the destination directory rather
        # than the system temp dir. This avoids:
        #   * cross-device rename failures inside compress_file when the
        #     destination volume differs from /tmp
        #   * Windows file-handle release races when reopening a tmp file
        #     in "wb" right after closing it in text mode
        out_path = Path(output_name)
        out_dir = out_path.parent if str(out_path.parent) else Path(".")
        suffix = out_path.suffix or ".tmp"
        # mkstemp returns (fd, path); close the fd immediately — we'll
        # reopen via write_func in its own mode.
        fd, tmp_path = tempfile.mkstemp(suffix=suffix,
                                        prefix=".mdapy_tmp_",
                                        dir=str(out_dir))
        os.close(fd)

        try:
            write_func(tmp_path, **kwargs)
            if not output_name.endswith(".gz"):
                output_name = output_name + ".gz"
            compress_file(tmp_path, output_name)
        finally:
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
        # `particle_type` is only present when the source defined per-atom
        # types (LAMMPS dumps, XYZ with species, etc.). A System built from
        # raw positions has no type table — guard accordingly.
        pt = getattr(atom.particles, "particle_type", None)
        if pt is not None and "type" in data.columns:
            type2element = {t.id: t.name for t in pt.types}
            # Only attach an element column if every type has a real, non-empty
            # name in the table (OVITO leaves "" for unnamed types).
            if type2element and all(
                isinstance(name, str) and len(name) > 0
                for name in type2element.values()
            ):
                data = data.with_columns(
                    pl.col("type")
                    .replace_strict(type2element)
                    .rechunk()
                    .alias("element")
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
        if "box" in meta_data:
            box = np.array(meta_data["box"].split(), float).reshape(3, 3)
        else:
            # No stored box → infer the bounding box of the atom cloud.
            # Pad zero-extent axes (e.g. 2D layer) to keep the cell
            # invertible — Box() rejects singular matrices.
            coor = data.select("x", "y", "z")
            extents = (coor.max() - coor.min()).to_numpy().flatten()
            extents = np.where(extents > 0, extents, 1e-9)
            box = np.diag(extents)
        origin = None
        boundary = None
        if "origin" in meta_data:
            origin = np.array(meta_data["origin"].split(), float)
        if "boundary" in meta_data:
            boundary = np.array(meta_data["boundary"].split(), np.int32)
        box = Box(box, boundary, origin)

        # Pass through user-set global_info keys; skip parquet/arrow internals.
        SKIP = {"box", "origin", "boundary"}
        global_info = {
            k: v for k, v in meta_data.items()
            if k not in SKIP and not k.startswith("ARROW")
        }
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
        (pl.DataFrame, mdapy.box.Box, dict)
            Per-atom data, simulation cell, and the trailing key=value
            metadata from the comment line as a dict (with `lattice`,
            `properties`, `pbc`, `origin` removed since they're already
            represented in the box / column schema).

        Notes
        -----
        Both classical (legacy) and extended XYZ are supported. The
        extended-XYZ ``properties=`` string is parsed token by token,
        supporting the conventional aliases:

        * ``pos:R:3``           → ``x y z``
        * ``species:S:1`` / ``element:S:1`` → ``element``
        * ``vel:R:3`` / ``velo:R:3`` → ``vx vy vz``
        * ``forces:R:3`` / ``force:R:3`` → ``fx fy fz``

        Multi-column user-defined fields (e.g. ``custom:R:5``) become
        ``custom_0`` … ``custom_4``.

        Performance: when the per-atom rows are uniformly single-space
        separated, parsing dispatches to ``pl.read_csv`` (fast, vectorised);
        otherwise it falls back to a Python ``str.split()`` pass that
        tolerates tabs, runs of spaces and trailing whitespace.
        """
        # ------------------------------------------------------------------
        # 1. Slurp the entire file. XYZ files for a single frame are
        #    typically small; the savings of streaming aren't worth the
        #    extra branching.
        # ------------------------------------------------------------------
        with _open_file(filename, "r") as op:
            lines = op.readlines()
        if len(lines) < 2:
            raise ValueError(f"{filename}: too short to be an XYZ file")

        natom = int(lines[0].strip())
        if natom < 0:
            raise ValueError(f"{filename}: negative atom count {natom}")
        if len(lines) < 2 + natom:
            raise ValueError(
                f"{filename}: header says {natom} atoms but only "
                f"{len(lines) - 2} body lines present"
            )

        # Strip the trailing newline (and CRLF on Windows) before regex
        # parsing, otherwise unquoted tokens at end-of-line capture the
        # newline and downstream string compares break.
        comment = lines[1].rstrip("\r\n")
        global_info: Dict[str, Any] = {}
        # Capture both quoted and unquoted key=value pairs.
        for match in re.findall(
            r'(\w+)=(?:"([^"]+)"|([^ ]+))', comment.replace("'", '"')
        ):
            key = match[0].lower()
            global_info[key] = match[1] if match[1] else match[2]

        # "classical" means *no explicit cell* — the box is derived from
        # the coordinate extents rather than read from a Lattice= field.
        # It does NOT imply "no Properties": ASE/GPUMD write non-periodic
        # frames (isolated monomers, pbc="F F F") that omit Lattice but
        # still declare a full Properties string with force/energy
        # columns. So the column schema must come from Properties whenever
        # it is present, regardless of whether a cell exists — otherwise
        # those frames silently lose every column past x/y/z.
        classical = "lattice" not in global_info

        # ------------------------------------------------------------------
        # 2. Build column list + per-column dtype.
        # ------------------------------------------------------------------
        if "properties" in global_info:
            content = global_info["properties"].strip().split(":")
            columns, schema = [], {}
            DTYPE_MAP = {"S": pl.Utf8, "R": pl.Float64, "I": pl.Int32}
            i = 0
            while i + 2 < len(content):
                name, ptype, n_col_str = content[i], content[i + 1], content[i + 2]
                if ptype not in DTYPE_MAP:
                    raise ValueError(f"{filename}: unrecognised XYZ type {ptype!r}")
                dtype = DTYPE_MAP[ptype]
                n_col = int(n_col_str)

                # Convention aliases (well-known per ASE / OVITO).
                # If the canonical alias names are already taken (the
                # Properties string mentions e.g. both `force:R:3` and
                # `forces:R:3`), the *second* occurrence falls through
                # to the generic ``<name>_<j>`` path so every column
                # stays unique and the data row split lines up.
                if (name == "pos" and ptype == "R" and n_col == 3
                        and "x" not in schema):
                    sub = ["x", "y", "z"]
                elif (name in ("unwrapped_position", "unwrapped_pos")
                        and ptype == "R" and n_col == 3
                        and "xu" not in schema):
                    # GPUMD writes "unwrapped_position:R:3"; map to the
                    # LAMMPS-style xu/yu/zu so the rest of the toolchain
                    # (unwrap_trajectory, MSD, ...) sees a uniform name.
                    sub = ["xu", "yu", "zu"]
                elif (name in ("species", "element") and ptype == "S"
                        and n_col == 1 and "element" not in schema):
                    sub = ["element"]
                elif (name in ("vel", "velo") and ptype == "R" and n_col == 3
                        and "vx" not in schema):
                    sub = ["vx", "vy", "vz"]
                elif (name in ("force", "forces") and ptype == "R" and n_col == 3
                        and "fx" not in schema):
                    sub = ["fx", "fy", "fz"]
                else:
                    sub = [name] if n_col == 1 else [f"{name}_{j}" for j in range(n_col)]
                for c in sub:
                    # Defensive uniqueness: in the rare case the
                    # generic name collides too, append a suffix.
                    base = c
                    suffix = 0
                    while c in schema:
                        suffix += 1
                        c = f"{base}__{suffix}"
                    columns.append(c)
                    schema[c] = dtype
                i += 3
        elif not classical:
            raise ValueError(
                f"{filename}: extended XYZ must contain a 'properties=' field"
            )
        else:
            columns = ["element", "x", "y", "z"]
            schema = {
                "element": pl.Utf8,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
            }

        # Boundary: pbc="T T F" / pbc="1 1 0" / etc. Without an explicit
        # pbc field a celled frame defaults to fully periodic and a
        # cell-less frame to fully open.
        if "pbc" in global_info:
            boundary = [
                1 if t in ("T", "1") else 0
                for t in global_info["pbc"].split()
            ]
        else:
            boundary = [0, 0, 0] if classical else [1, 1, 1]

        origin = np.zeros(3, float)
        if "origin" in global_info:
            origin = np.array(global_info["origin"].split(), float)

        # Box from Lattice= (9 reals = a/b/c stacked rows) when present;
        # the cell-less case is filled in from the coordinate extents
        # after the per-atom block is parsed (see below).
        if not classical:
            box_mat = np.array(global_info["lattice"].split(), float).reshape(3, 3)
            box = np.vstack([box_mat, origin.reshape(1, 3)])

        # ------------------------------------------------------------------
        # 3. Parse the per-atom block.
        #    Fast path: clean single-space separator → pl.read_csv on a
        #    StringIO of the body. Slow path: numpy split, robust to runs
        #    of whitespace, tabs, CRLF.
        # ------------------------------------------------------------------
        body = lines[2 : 2 + natom]
        if _is_uniform_single_space(body, len(columns)):
            buf = io.StringIO("".join(body))
            df = pl.read_csv(
                buf,
                separator=" ",
                schema=schema,
                has_header=False,
            )
        else:
            cells = np.array([row.split()[: len(columns)] for row in body])
            df_cols = {}
            for j, c in enumerate(columns):
                col = cells[:, j]
                if schema[c] == pl.Int32:
                    df_cols[c] = pl.Series(c, col.astype(np.int32), dtype=pl.Int32)
                elif schema[c] == pl.Utf8:
                    df_cols[c] = pl.Series(c, col.tolist(), dtype=pl.Utf8)
                else:
                    df_cols[c] = pl.Series(c, col.astype(np.float64),
                                           dtype=pl.Float64)
            df = pl.DataFrame(df_cols)

        # Classical XYZ: infer the bounding box from the data. If any axis
        # has zero extent (e.g. 2D layer at z=const), pad it slightly so the
        # cell matrix stays invertible — Box() rejects singular matrices.
        if classical:
            coor = df.select("x", "y", "z")
            mins = coor.min().to_numpy().flatten()
            extents = (coor.max() - coor.min()).to_numpy().flatten()
            extents = np.where(extents > 0, extents, 1e-9)
            box = np.vstack([np.diag(extents), mins])

        # Strip metadata keys we've already consumed.
        for key in ("pbc", "properties", "origin", "lattice"):
            global_info.pop(key, None)

        return df.rechunk(), Box(box, boundary), global_info

    @staticmethod
    def read_poscar(
        filename: str,
    ) -> Tuple[pl.DataFrame, Box, Optional[Dict[str, Any]]]:
        """
        Read a VASP POSCAR file (supports ``.gz`` compression).

        Format (line indices below are 0-based):

        - line 0: free-form comment.
        - line 1: universal scale factor (multiplies both the lattice
          and the Cartesian atom positions).
        - lines 2–4: three lattice vectors ``a``, ``b``, ``c`` (one per
          row).
        - line 5: element-symbol line (e.g. ``"Al Cu"``) *or* the
          per-species count line (e.g. ``"1 1"``) — distinguished by
          whether the first non-whitespace char is alphabetic vs.
          numeric.
        - line 6: per-species count line (only when line 5 is symbols).
        - optional ``Selective dynamics`` header (first non-blank char
          ``S`` / ``s``).
        - coordinate-type line: first non-blank char in
          ``{D, d}`` → Direct, ``{C, c, K, k}`` → Cartesian / K-point.
        - ``N`` atom rows (3 floats each, plus 3 T/F flags when SD is
          on).
        - optional lattice-velocity / ion-velocity blocks.

        Returns
        -------
        Tuple[pl.DataFrame, mdapy.box.Box, dict]
        """
        global_info: Dict[str, Any] = {}
        with _open_file(filename, "r") as op:
            file = op.readlines()
        if len(file) < 7:
            raise ValueError(f"{filename}: too short to be a POSCAR")

        # ------------------------------------------------------------------
        # Lines 0–4: comment + scale + lattice vectors.
        # ------------------------------------------------------------------
        global_info["comment"] = file[0].strip()
        scale = float(file[1].strip())
        box = np.array([file[2].split(), file[3].split(), file[4].split()],
                       dtype=float) * scale

        # ------------------------------------------------------------------
        # Lines 5–6: optional element symbols + per-species counts.
        # ------------------------------------------------------------------
        line5 = file[5].split()
        if line5 and line5[0][0].isalpha():
            symbols = line5
            counts = [int(c) for c in file[6].split()]
            row = 7
        else:
            symbols = None
            counts = [int(c) for c in line5]
            row = 6

        natoms = sum(counts)
        if symbols is not None:
            if len(symbols) != len(counts):
                raise ValueError(
                    f"{filename}: {len(symbols)} symbols but {len(counts)} counts"
                )
            element_list = [s for s, n in zip(symbols, counts) for _ in range(n)]
            type_list = []
        else:
            type_list = [t for t, n in enumerate(counts, start=1) for _ in range(n)]
            element_list = []

        # ------------------------------------------------------------------
        # Optional "Selective dynamics" header (first non-blank char S/s).
        # ------------------------------------------------------------------
        selective_dynamics = False
        if file[row].lstrip().startswith(("S", "s")):
            selective_dynamics = True
            row += 1
        global_info["selective_dynamics"] = selective_dynamics

        # ------------------------------------------------------------------
        # Coordinate-type line. Use lstrip() — real-world POSCARs often
        # have leading whitespace before "Cartesian"/"Direct".
        # ------------------------------------------------------------------
        coord_char = file[row].lstrip()[:1]
        is_cartesian = coord_char in ("C", "c", "K", "k")
        row += 1

        # ------------------------------------------------------------------
        # N atom rows.
        # ------------------------------------------------------------------
        atom_rows = file[row : row + natoms]
        if len(atom_rows) != natoms:
            raise ValueError(
                f"{filename}: expected {natoms} atom rows, got {len(atom_rows)}"
            )
        if selective_dynamics:
            pos = np.array([r.split()[:3] for r in atom_rows], dtype=float)
            sd = np.array([r.split()[3:6] for r in atom_rows])
        else:
            pos = np.array([r.split()[:3] for r in atom_rows], dtype=float)
            sd = np.empty((0, 3))
        row += natoms

        # Convert positions to absolute Cartesian. Per the VASP spec the
        # universal scale factor multiplies BOTH the lattice and the
        # Cartesian coordinates.
        if is_cartesian:
            pos = pos * scale
        else:
            pos = pos @ box

        # ------------------------------------------------------------------
        # Optional lattice-velocity block (6 rows after a header line).
        # ------------------------------------------------------------------
        if row < len(file) and file[row].lstrip().startswith(("L", "l")):
            global_info["initialization_state"] = file[row + 1].strip()
            global_info["lattice_velocity"] = np.array(
                [line.split()[:3] for line in file[row + 2 : row + 8]],
                dtype=float,
            )
            row += 8

        # ------------------------------------------------------------------
        # Optional ion-velocity block. Header is the coord-type line
        # (Cartesian/Direct), then N rows.
        # ------------------------------------------------------------------
        vel = []
        if row + 1 + natoms <= len(file):
            vel = np.array(
                [r.split() for r in file[row + 1 : row + 1 + natoms]],
                dtype=float,
            )
            if vel.shape[0] == natoms:
                vel_char = file[row].lstrip()[:1] if file[row].strip() else "C"
                if vel_char not in ("C", "c", "K", "k"):
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
        Read a single-frame LAMMPS data file (supports .gz compression).

        Only ``atom_style atomic`` and ``atom_style charge`` are supported.
        Other styles (bond, molecular, full, …) raise NotImplementedError.

        The optional image-flag columns ``nx ny nz`` are preserved as
        ``ix iy iz`` int columns alongside the wrapped ``x y z``.

        Parameters
        ----------
        filename : str
            Path to data file (can be ``.gz``).

        Returns
        -------
        (pl.DataFrame, mdapy.box.Box, dict)
            Per-atom data, simulation cell, and an empty global-info dict.
        """
        # ------------------------------------------------------------------
        # 1. Slurp the whole file into a list of lines. Data files are
        #    typically small enough that this is fine, and a single in-memory
        #    pass is far simpler than two-pass with offsets.
        # ------------------------------------------------------------------
        with _open_file(filename, "r") as op:
            lines = op.readlines()

        # Body section keywords we care about. Any line whose first token is
        # in this set marks the start of a section. (Sections we don't read
        # we still need to know about so we can skip them.)
        BODY_SECTIONS = {
            "Atoms", "Velocities", "Masses",
            # Forbidden / unsupported sections — presence means the file
            # uses a richer atom_style than atomic/charge.
            "Bonds", "Angles", "Dihedrals", "Impropers",
            "Bond", "Angle", "Dihedral", "Improper",        # *Coeffs prefixes
            "Pair", "PairIJ",
            "Ellipsoids", "Lines", "Triangles", "Bodies",
            "Atom", "Bond",                                  # Type Labels
        }
        FORBIDDEN_HINTS = ("bond", "angle", "dihedral", "improper",
                           "molecular", "ellipsoid")

        # ------------------------------------------------------------------
        # 2. Find the first section header (= end of the header block).
        #    Scan looking for a line whose stripped first token starts with
        #    a capital and is in BODY_SECTIONS.
        # ------------------------------------------------------------------
        section_starts = {}      # name -> line index of the header
        for i, raw in enumerate(lines):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            first_tok = stripped.split()[0]
            if first_tok in BODY_SECTIONS:
                section_starts.setdefault(first_tok, i)

        if "Atoms" not in section_starts:
            raise ValueError(f"{filename}: no 'Atoms' section found")

        header_lines = lines[: section_starts["Atoms"]]

        # ------------------------------------------------------------------
        # 3. Parse the header for N, n_types, box, atom_style hints.
        # ------------------------------------------------------------------
        N = 0
        n_types = 0
        xlo = xhi = ylo = yhi = zlo = zhi = 0.0
        xy = xz = yz = 0.0
        for raw in header_lines:
            tokens = raw.split()
            if not tokens:
                continue
            tail = tokens[-1]
            try:
                if tail == "atoms":
                    N = int(tokens[0])
                elif tail == "types" and len(tokens) >= 2 and tokens[1] == "atom":
                    n_types = int(tokens[0])
                elif tail == "xhi":
                    xlo, xhi = float(tokens[0]), float(tokens[1])
                elif tail == "yhi":
                    ylo, yhi = float(tokens[0]), float(tokens[1])
                elif tail == "zhi":
                    zlo, zhi = float(tokens[0]), float(tokens[1])
                elif tail == "yz":
                    xy, xz, yz = (float(tokens[0]), float(tokens[1]),
                                  float(tokens[2]))
            except (IndexError, ValueError):
                # Header lines with the wrong shape are skipped silently —
                # `Atoms` is the real anchor.
                pass

            # Guard against unsupported atom_style hints (header keywords
            # like '5 bond types' / '... bonds' / '... angles' / etc.).
            if len(tokens) >= 2 and any(h in tokens[1].lower() for h in FORBIDDEN_HINTS):
                raise NotImplementedError(
                    f"{filename}: only atom_style atomic / charge are supported "
                    f"(found header keyword '{tokens[1]}')"
                )

        box = np.array([
            [xhi - xlo, 0,        0       ],
            [xy,        yhi - ylo, 0      ],
            [xz,        yz,        zhi - zlo],
            [xlo,       ylo,       zlo    ],
        ])

        # ------------------------------------------------------------------
        # 4. Locate the data block under `Atoms`. The header line may carry
        #    a trailing comment ("Atoms # atomic" / "Atoms" / "Atoms # foo").
        #    Detect atom_style robustly:
        #
        #      * explicit "# atomic" / "# charge"  → use directly
        #      * unrecognised comment              → infer from column count
        #      * no comment                        → infer from column count
        #
        #    Unsupported styles named explicitly (bond/full/molecular/…)
        #    raise NotImplementedError.
        # ------------------------------------------------------------------
        atoms_header_idx = section_starts["Atoms"]
        atoms_header = lines[atoms_header_idx]
        atom_style_hint = None
        if "#" in atoms_header:
            atom_style_hint = atoms_header.split("#", 1)[1].strip().split()[0].lower()
        if atom_style_hint and atom_style_hint not in ("atomic", "charge"):
            raise NotImplementedError(
                f"{filename}: atom_style '{atom_style_hint}' not supported "
                f"(only atomic / charge)"
            )

        # First data line: skip blanks/comments after the `Atoms` header.
        first_data_idx = atoms_header_idx + 1
        while first_data_idx < len(lines):
            stripped = lines[first_data_idx].strip()
            if stripped and not stripped.startswith("#"):
                break
            first_data_idx += 1
        if first_data_idx >= len(lines):
            raise ValueError(f"{filename}: 'Atoms' section is empty")

        first_cols = lines[first_data_idx].split()
        n_cols = len(first_cols)

        # Match column count to atom style. Image flags `nx ny nz` add 3.
        # Allowed shapes:
        #   atomic: 5  (id type x y z)             or 8  (+ ix iy iz)
        #   charge: 6  (id type q x y z)           or 9  (+ ix iy iz)
        if atom_style_hint == "atomic":
            valid = (5, 8)
        elif atom_style_hint == "charge":
            valid = (6, 9)
        else:
            # Infer from column count.
            valid = None

        if valid is None:
            if n_cols in (5, 8):
                style = "atomic"
            elif n_cols in (6, 9):
                style = "charge"
            else:
                raise NotImplementedError(
                    f"{filename}: cannot infer atom_style from {n_cols}-column "
                    f"Atoms data (only atomic / charge are supported)"
                )
        else:
            if n_cols not in valid:
                raise ValueError(
                    f"{filename}: atom_style {atom_style_hint!r} expects "
                    f"{valid} columns, got {n_cols}"
                )
            style = atom_style_hint

        if style == "atomic":
            col_names = ["id", "type", "x", "y", "z"]
        else:  # charge
            col_names = ["id", "type", "q", "x", "y", "z"]
        has_image_flags = (n_cols == len(col_names) + 3)
        if has_image_flags:
            col_names += ["ix", "iy", "iz"]

        # ------------------------------------------------------------------
        # 5. Parse the per-atom block.
        #
        #    Fast path: when every row is a clean single-space-separated
        #    record with the expected column count, use polars' CSV reader
        #    (fast, vectorised). Otherwise fall back to a numpy split-based
        #    parser (robust to runs of whitespace, mixed tabs, comments).
        # ------------------------------------------------------------------
        data_block = lines[first_data_idx : first_data_idx + N]
        if len(data_block) != N:
            raise ValueError(
                f"{filename}: expected {N} atom rows, found {len(data_block)}"
            )

        # Per-column dtype for atomic / charge.
        schema: Dict[str, "pl.DataType"] = {}
        for c in col_names:
            if c in ("id", "type", "ix", "iy", "iz"):
                schema[c] = pl.Int32
            else:
                schema[c] = pl.Float64

        if _is_uniform_single_space(data_block, len(col_names)):
            # Fast path — feed just the data block to polars so its CSV
            # reader can't accidentally scan past it (n_rows alone doesn't
            # cap the file's-end-of-block detection on every Polars version).
            buf = io.StringIO("".join(data_block))
            data = pl.read_csv(
                buf,
                separator=" ",
                schema=schema,
                has_header=False,
            )
        else:
            # Slow path — Python split is robust to multi-space / tab.
            atoms_arr = np.array([row.split()[: len(col_names)] for row in data_block])
            cols = {}
            for i, c in enumerate(col_names):
                col = atoms_arr[:, i]
                if schema[c] == pl.Int32:
                    cols[c] = pl.Series(c, col.astype(np.int32), dtype=pl.Int32)
                else:
                    cols[c] = pl.Series(c, col.astype(np.float64), dtype=pl.Float64)
            data = pl.DataFrame(cols)

        # ------------------------------------------------------------------
        # 6. If a `Velocities` section exists, parse it and join.
        # ------------------------------------------------------------------
        if "Velocities" in section_starts:
            vh = section_starts["Velocities"]
            first_vel_idx = vh + 1
            while first_vel_idx < len(lines):
                stripped = lines[first_vel_idx].strip()
                if stripped and not stripped.startswith("#"):
                    break
                first_vel_idx += 1

            vel_block = lines[first_vel_idx : first_vel_idx + N]
            if len(vel_block) == N:
                # Velocity rows have 4 cols: atom-id vx vy vz.
                if _is_uniform_single_space(vel_block, 4):
                    buf = io.StringIO("".join(vel_block))
                    vdf = pl.read_csv(
                        buf,
                        separator=" ",
                        schema={"id": pl.Int32, "vx": pl.Float64,
                                "vy": pl.Float64, "vz": pl.Float64},
                        has_header=False,
                    )
                    vel_ids = vdf["id"].to_numpy()
                    vel_arr = vdf.select("vx", "vy", "vz").to_numpy()
                else:
                    vel_arr = np.array([row.split()[1:4] for row in vel_block],
                                       dtype=np.float64)
                    vel_ids = np.array([int(row.split()[0]) for row in vel_block])

                id_to_vel = dict(zip(vel_ids, vel_arr))
                ordered = np.array([id_to_vel[a] for a in data["id"].to_list()],
                                   dtype=np.float64)
                data = data.with_columns(
                    vx=ordered[:, 0], vy=ordered[:, 1], vz=ordered[:, 2],
                )

        # ------------------------------------------------------------------
        # 7. If the Masses section carries `# Element` comments for every
        #    atom type, attach an `element` column. This round-trips with
        #    `write_data(..., element_list=...)`.
        # ------------------------------------------------------------------
        if "Masses" in section_starts and n_types > 0:
            mh = section_starts["Masses"]
            first_mass_idx = mh + 1
            while first_mass_idx < len(lines):
                stripped = lines[first_mass_idx].strip()
                if stripped and not stripped.startswith("#"):
                    break
                first_mass_idx += 1
            mass_block = lines[first_mass_idx : first_mass_idx + n_types]
            type2elem = _parse_masses_to_elements(mass_block, n_types)
            if type2elem is not None:
                # Use polars' replace_strict for vectorised mapping.
                data = data.with_columns(
                    pl.col("type").replace_strict(type2elem).alias("element")
                )

        return data.rechunk(), Box(box, [1, 1, 1]), {}

    @staticmethod
    def parse_dump_frame(lines: List[str],
                         source: str = "<dump>"
                         ) -> Tuple[pl.DataFrame, Box, Dict[str, Any]]:
        """
        Parse one frame's worth of LAMMPS dump lines (the 9 header lines
        + N atom rows). Used by both :py:meth:`read_dump` (single-frame)
        and :class:`mdapy.Trajectory` (multi-frame).
        """
        return _parse_dump_frame_impl(lines, source)

    @staticmethod
    def read_dump(filename: str) -> Tuple[pl.DataFrame, Box, Dict[str, Any]]:
        """
        Read a single-frame LAMMPS dump file (text style, supports .gz).

        Multi-frame dumps are rejected with a clear error — use
        :class:`mdapy.Trajectory` for those.

        Coordinate column variants understood (see LAMMPS dump custom docs):

        * ``x  y  z``   wrapped Cartesian            → kept as ``x y z``
        * ``xs ys zs``  scaled (fractional)          → unscaled to Cartesian
        * ``xu yu zu``  unwrapped Cartesian          → kept as ``x y z``
        * ``xsu ysu zsu`` scaled-unwrapped           → unscaled to Cartesian

        Image flags ``ix iy iz`` and the optional ``element`` column from
        ``dump_modify ... element ...`` are preserved verbatim.

        BOX BOUNDS header forms recognised:

        * ``ITEM: BOX BOUNDS xx yy zz``                       — orthogonal
        * ``ITEM: BOX BOUNDS xy xz yz xx yy zz``              — restricted triclinic
        * ``ITEM: BOX BOUNDS abc origin``                     — general triclinic (newer LAMMPS)
        """
        with _open_file(filename, "r") as op:
            lines = op.readlines()

        # Locate every "ITEM: TIMESTEP" — there must be exactly one.
        ts_indices = [i for i, l in enumerate(lines)
                      if l.strip().startswith("ITEM: TIMESTEP")]
        if not ts_indices:
            raise ValueError(f"{filename}: no 'ITEM: TIMESTEP' header found")
        if len(ts_indices) > 1:
            raise ValueError(
                f"{filename}: multi-frame dump file (found {len(ts_indices)} "
                f"frames). Use mdapy.Trajectory or split the file first."
            )

        return _parse_dump_frame_impl(lines, source=filename)

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
                        op, separator=" ", include_header=False,
                        float_precision=10,
                    )
            else:
                # Token-by-token Properties string assembly. Independent of
                # column order — folding only triggers on contiguous runs
                # of canonical names (x/y/z, vx/vy/vz, fx/fy/fz, base_0/1/…).
                properties_str = _build_xyz_properties(data.columns,
                                                        list(data.dtypes))

                # Build the comment line with Lattice / Properties /
                # pbc / Origin (the four well-known extended-XYZ keys).
                lattice_str = (
                    'Lattice="' + " ".join(box.box.flatten().astype(str).tolist())
                    + '"'
                )
                pbc_str = (
                    'pbc="'
                    + " ".join("T" if i == 1 else "F" for i in box.boundary)
                    + '"'
                )
                origin_str = (
                    'Origin="' + " ".join(box.origin.astype(str).tolist()) + '"'
                )
                comments = f"{lattice_str} {properties_str} {pbc_str} {origin_str}"

                for key, value in kargs.items():
                    if key.lower() not in ("lattice", "pbc", "properties", "origin"):
                        try:
                            if key.lower() in ("virial", "bec", "stress"):
                                comments += f' {key}="{value}"'
                            else:
                                comments += f" {key}={value}"
                        except Exception:
                            pass

                with open(filename, "wb") as op:
                    op.write(f"{data.shape[0]}\n".encode())
                    op.write(f"{comments}\n".encode())
                    data.write_csv(op, separator=" ", include_header=False,
                                   float_precision=10)

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

            type_max = int(data["type"].max())
            if num_type is None:
                num_type = type_max
            else:
                if num_type < type_max:
                    raise ValueError(
                        f"num_type ({num_type}) is less than the largest type "
                        f"in data ({type_max})"
                    )

            if element_list is not None:
                if len(element_list) < num_type:
                    raise ValueError(
                        f"element_list has {len(element_list)} entries but "
                        f"num_type is {num_type}; need at least {num_type}"
                    )
                # Honour the user's explicit num_type; just warn when the
                # element_list is longer (extra entries are unused but
                # listed in Masses for forward compatibility).
                if len(element_list) > num_type:
                    import warnings
                    warnings.warn(
                        f"element_list has {len(element_list)} entries but "
                        f"num_type={num_type}; writing only the first "
                        f"{num_type} elements.",
                        stacklevel=2,
                    )
                    element_list = list(element_list)[:num_type]
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
                # float_precision=10 keeps positions / velocities round-
                # trippable to ~9 significant digits without writing huge
                # default-precision strings.
                table.write_csv(op, separator=" ", include_header=False,
                                float_precision=10)

                if all(col in data.columns for col in ["vx", "vy", "vz"]):
                    op.write("\nVelocities\n\n".encode())
                    table = data.select(["id", "vx", "vy", "vz"])
                    table.write_csv(op, separator=" ", include_header=False,
                                    float_precision=10)

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
            with open(filename, "wb") as op:
                SaveSystem.write_dump_frame_to(op, box, data, timestep)

        _save_with_compression(
            _write_dump_internal,
            output_name,
            compress,
            box=box,
            data=data,
            timestep=timestep,
        )

    @staticmethod
    def write_dump_frame_to(op, box: Box, data: pl.DataFrame,
                            timestep: float = 0.0) -> None:
        """Write one LAMMPS dump frame to an already-open binary file
        handle ``op``. Used by both :py:meth:`write_dump` (single-frame)
        and :class:`mdapy.Trajectory` (multi-frame).
        """
        if "id" not in data.columns:
            data = data.with_row_index("id", offset=1)

        # Triclinic dumps require lammps-aligned axes; rotate if needed.
        if box.box[0, 1] != 0 or box.box[0, 2] != 0 or box.box[1, 2] != 0:
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

        # Keep numeric columns + the optional `element` string column
        # (newer LAMMPS dumps support `element` via `dump_modify ...
        # element X Y Z`). Other string columns are dropped.
        keep = [c for c, dt in zip(data.columns, data.dtypes)
                if dt.is_numeric() or c == "element"]
        data = data.select(keep)

        boundary2str = ["pp" if i == 1 else "ss" for i in box.boundary]
        op.write(f"ITEM: TIMESTEP\n{timestep}\n".encode())
        op.write(f"ITEM: NUMBER OF ATOMS\n{data.shape[0]}\n".encode())

        xlo, ylo, zlo = box.origin
        xhi = xlo + box.box[0, 0]
        yhi = ylo + box.box[1, 1]
        zhi = zlo + box.box[2, 2]
        xy, xz, yz = box.box[1, 0], box.box[2, 0], box.box[2, 1]

        if xy != 0 or xz != 0 or yz != 0:
            xlo_bound = xlo + min(0.0, xy, xz, xy + xz)
            xhi_bound = xhi + max(0.0, xy, xz, xy + xz)
            ylo_bound = ylo + min(0.0, yz)
            yhi_bound = yhi + max(0.0, yz)
            op.write(
                f"ITEM: BOX BOUNDS xy xz yz {boundary2str[0]} "
                f"{boundary2str[1]} {boundary2str[2]}\n".encode()
            )
            op.write(f"{xlo_bound} {xhi_bound} {xy}\n".encode())
            op.write(f"{ylo_bound} {yhi_bound} {xz}\n".encode())
            op.write(f"{zlo} {zhi} {yz}\n".encode())
        else:
            op.write(
                f"ITEM: BOX BOUNDS {boundary2str[0]} "
                f"{boundary2str[1]} {boundary2str[2]}\n".encode()
            )
            op.write(f"{xlo} {xhi}\n".encode())
            op.write(f"{ylo} {yhi}\n".encode())
            op.write(f"{zlo} {zhi}\n".encode())

        op.write(("ITEM: ATOMS " + " ".join(data.columns) + " \n").encode())
        data.write_csv(op, separator=" ", include_header=False,
                       float_precision=10)


if __name__ == "__main__":
    pass
