# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from mdapy import _neighbor, _repeat_cell, _split
from typing import Optional, Tuple
import numpy as np
import polars as pl
from mdapy.box import Box
from mdapy.data import atomic_masses, atomic_numbers
import os


def average_by_neighbor(
    average_rc: float,
    data: pl.DataFrame,
    property_name: str,
    verlet_list: np.ndarray,
    distance_list: np.ndarray,
    neighbor_number: np.ndarray,
    include_self: bool = True,
    output_name: Optional[str] = None,
) -> pl.DataFrame:
    """
    Compute the averaged property of each atom based on its neighbors within a given cutoff radius.

    This function calculates the local average of a specified property (e.g., temperature, stress)
    for each atom by averaging over its neighboring atoms within the cutoff ``average_rc``.
    The neighbor information is provided through the Verlet list and corresponding distances.

    Parameters
    ----------
    average_rc : float
        The cutoff radius used for averaging the property.
    data : pl.DataFrame
        The input atomic data containing the property to be averaged.
    property_name : str
        The name of the column in ``data`` to average.
    verlet_list : np.ndarray
        A 2D array of neighbor indices. Each row corresponds to one atom and lists its neighbors.
    distance_list : np.ndarray
        A 2D array of distances corresponding to the ``verlet_list`` entries.
    neighbor_number : np.ndarray
        A 1D array indicating the actual number of neighbors for each atom.
    include_self : bool, optional, default=True
        Whether to include the atom itself when computing the average.
    output_name : str, optional
        The name of the output column to store the averaged property.
        If not provided, defaults to ``f"{property_name}_ave"``.

    Returns
    -------
    pl.DataFrame
        A new DataFrame containing the original data plus an additional column
        with the averaged property values.

    """
    assert property_name in data.columns, f"{property_name} not in data."
    property_ave = np.zeros(data.shape[0])
    _neighbor.average_by_neighbor(
        average_rc,
        verlet_list,
        distance_list,
        neighbor_number,
        data[property_name].to_numpy(allow_copy=False),
        property_ave,
        include_self,
    )
    if output_name is None:
        output_name = f"{property_name}_ave"
    return data.with_columns(pl.lit(property_ave).alias(output_name))


def sort_neighbor(
    verlet_list: np.ndarray,
    distance_list: np.ndarray,
    neighbor_number: np.ndarray,
    k: int,
):
    """
    Sort the first ``k`` neighbors of each atom by ascending distance.

    This function reorders the first ``k`` neighbor entries in the Verlet list and distance list
    so that the nearest neighbors appear first.

    Parameters
    ----------
    verlet_list : np.ndarray
        A 2D array of neighbor indices. Each row corresponds to one atom and lists its neighbors.
    distance_list : np.ndarray
        A 2D array of distances corresponding to the ``verlet_list`` entries.
    neighbor_number : np.ndarray
        A 1D array indicating the actual number of neighbors for each atom.
    k : int
        The number of nearest neighbors to sort. Must be less than or equal to
        the minimum value in ``neighbor_number``.

    Raises
    ------
    AssertionError
        If any atom has fewer than ``k`` neighbors.

    Examples
    --------
    >>> verlet = np.array([[3, 1, 2], [0, 2, 3]])
    >>> distance = np.array([[0.3, 0.1, 0.2], [0.5, 0.2, 0.4]])
    >>> neighbor_num = np.array([3, 3])
    >>> sort_neighbor(verlet, distance, neighbor_num, k=2)
    >>> verlet
    array([[1, 2, 3],
           [2, 3, 0]])
    >>> distance
    array([[0.1, 0.2, 0.3],
           [0.2, 0.4, 0.5]])
    """
    minNumber = neighbor_number.min()
    assert minNumber >= k, f"The min neighbor number {minNumber} is lower than k {k}."
    _neighbor.sort_verlet_by_distance(verlet_list, distance_list, k)


def wrap_pos(data: pl.DataFrame, box: Box) -> pl.DataFrame:
    """Wrap position into box based on PBC.

    Args:
        data (pl.DataFrame): atom information.
        box (Box): box information

    Returns:
        pl.DataFrame: DataFrame with new wraped position.
    """
    x, y, z = (
        data["x"].to_numpy(writable=True),
        data["y"].to_numpy(writable=True),
        data["z"].to_numpy(writable=True),
    )
    _neighbor.wrap_positions(x, y, z, box.box, box.origin, box.boundary)
    return data.with_columns(x=x, y=y, z=z)


def replicate(
    data: pl.DataFrame, box: Box, nx: int, ny: int, nz: int
) -> Tuple[pl.DataFrame, Box]:
    """Replicates atomic data along the x, y, and z directions.

    This function creates a supercell by replicating the input atomic positions
    and associated data `nx` times along x, `ny` times along y, and `nz` times
    along z. The input DataFrame is expected to
    contain at least 'x', 'y', and 'z' columns for atomic coordinates.

    Args:
        data: Atomic information, including positions in 'x', 'y', 'z' columns.
        box: Simulation box information to be scaled.
        nx: Number of replications along the x direction.
        ny: Number of replications along the y direction.
        nz: Number of replications along the z direction.

    Returns:
        A tuple containing the replicated DataFrame (with updated positions and
        optionally renumbered 'id' column) and the scaled Box object.
    """
    old_box = box.box
    old_pos = data.select("x", "y", "z").to_numpy()
    n_old = old_pos.shape[0]
    total = n_old * nx * ny * nz * 3
    new_pos = np.zeros(total, dtype=np.float64)
    _repeat_cell.repeat_cell(new_pos, old_box, old_pos, nx, ny, nz)
    new_pos = new_pos.reshape((-1, 3))
    new_box = old_box * np.array([nx, ny, nz]).reshape((3, 1))
    new_data = pl.concat([data] * nx * ny * nz).with_columns(
        x=new_pos[:, 0], y=new_pos[:, 1], z=new_pos[:, 2]
    )
    if "id" in new_data.columns:
        new_data = new_data.select(pl.all().exclude("id")).with_row_index(
            "id", offset=1
        )
    return new_data.rechunk(), Box(new_box, box.boundary, box.origin)


def _replicate_pos(
    data: pl.DataFrame, box: Box, nx: int, ny: int, nz: int
) -> Tuple[pl.DataFrame, Box]:
    old_box = box.box
    old_pos = data.select("x", "y", "z").to_numpy()
    n_old = old_pos.shape[0]
    total = n_old * nx * ny * nz * 3
    new_pos = np.zeros(total, dtype=np.float64)
    _repeat_cell.repeat_cell(new_pos, old_box, old_pos, nx, ny, nz)
    new_pos = new_pos.reshape((-1, 3))
    new_box = old_box * np.array([nx, ny, nz]).reshape((3, 1))
    new_data = pl.from_numpy(new_pos, schema=["x", "y", "z"])
    return new_data.rechunk(), Box(new_box, box.boundary, box.origin)


def _set_pka(
    data: pl.DataFrame,
    box: Box,
    energy: float,
    direction: np.ndarray,
    index: int = None,
    element: str = None,
) -> pl.DataFrame:
    required_cols = ["x", "y", "z", "element", "vx", "vy", "vz"]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Must include '{col}' column in data.")

    direction = np.array(direction, dtype=float)
    if direction.shape != (3,):
        raise ValueError("Direction must be a 3D vector.")

    element_unique = data["element"].unique()
    has_amass = True
    if "amass" not in data.columns:
        ele2mass = {}
        for ele in element_unique:
            if ele not in atomic_numbers.keys():
                raise ValueError(f"Unknown element '{ele}' in atomic_numbers.")
            ele2mass[ele] = atomic_masses[atomic_numbers[ele]]
        data = data.with_columns(
            pl.col("element").replace_strict(ele2mass).alias("amass")
        )
        has_amass = False

    totalmass = data["amass"].sum()
    if index is None:
        cx, cy, cz = box.box @ np.array([0.5, 0.5, 0.5]) + box.origin
        dist_expr: pl.Expr = (
            (pl.col("x") - cx) ** 2 + (pl.col("y") - cy) ** 2 + (pl.col("z") - cz) ** 2
        )
        if element is None:
            index = data.select(dist_expr.arg_min()).item()
        else:
            if element not in element_unique:
                raise ValueError(f"Element '{element}' not in data.")
            ele_filtered = data.with_row_index("row_idx").filter(
                pl.col("element") == element
            )
            local_min = ele_filtered.select(dist_expr.arg_min()).item()
            index = ele_filtered["row_idx"][local_min]
    else:
        if index < 0 or index >= len(data):
            raise ValueError(f"Index {index} out of bounds.")
        if element is not None and data.select(pl.col("element"))[index, 0] != element:
            raise ValueError(f"Element at index {index} is not '{element}'.")

    atom_mass = data["amass"][index]
    speed = np.sqrt(2 * energy / atom_mass)
    norm_dir = direction / np.linalg.norm(direction)
    # We make sure the input velocity in the units of A/fs.
    newv = speed * norm_dir / 10.18051  # eV/amu to A/fs

    data[index, "vx"] = newv[0]
    data[index, "vy"] = newv[1]
    data[index, "vz"] = newv[2]

    com_vx = data.select((pl.col("amass") * pl.col("vx")).sum() / totalmass).item()
    com_vy = data.select((pl.col("amass") * pl.col("vy")).sum() / totalmass).item()
    com_vz = data.select((pl.col("amass") * pl.col("vz")).sum() / totalmass).item()

    data = data.with_columns(
        pl.col("vx") - com_vx,
        pl.col("vy") - com_vy,
        pl.col("vz") - com_vz,
    )
    if not has_amass:
        data = data.drop("amass")

    return data


def split_xyz(input_file, output_dir="res", output_prefix=None, in_memory=True):
    """
    Split a multi-frame XYZ file into individual frame files.

    Parameters
    ----------
    input_file : str
        Path to the input XYZ file containing multiple frames.
        The file should follow standard XYZ format:
            - Line 1: Number of atoms (integer)
            - Line 2: Comment line
            - Lines 3 to N+2: Atomic coordinates (element x y z ...)
            - Repeat for each frame

    output_dir : str, optional
        Directory where individual frame files will be saved.
        The directory will be created if it doesn't exist.
        Default is "res".

    output_prefix : str, optional
        Prefix for output filenames. If None, uses the input filename
        (without extension) as prefix.
        Output files are named as: {output_prefix}.{frame:0Nd}.xyz
        where N is the number of digits needed (e.g., 5 digits for 10000 frames).
        Default is None.

    in_memory : bool, optional
        If input file is too big to load into memory, set this parameter to False.
        Default is True.

    Returns
    -------
    None

    Examples
    --------
    >>> # Split trajectory.xyz into res/trajectory.00000.xyz, res/trajectory.00001.xyz, ...
    >>> split_xyz("trajectory.xyz")

    >>> # Specify custom output directory and prefix
    >>> split_xyz("trajectory.xyz", output_dir="frames", output_prefix="md")
    >>> # Output: frames/md.00000.xyz, frames/md.00001.xyz, ...

    >>> # For a file with 100000 frames, output will use 6 digits
    >>> split_xyz("large_traj.xyz")
    >>> # Output: res/large_traj.000000.xyz, ..., res/large_traj.099999.xyz

    Raises
    ------
    RuntimeError
        If the input file cannot be opened or contains no valid frames.
    OSError
        If the output directory cannot be created.

    """
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(input_file))[0]
    if in_memory:
        _split.split_xyz(input_file, output_dir, output_prefix)
    else:
        with open(input_file, "r") as file:
            frame = 0
            while True:
                line = file.readline()
                if not line:
                    break
                n = int(line.strip())
                output_file = os.path.join(
                    output_dir, f"{output_prefix}.{frame:0>6d}.xyz"
                )
                with open(output_file, "w") as out_file:
                    out_file.write(line)
                    for _ in range(n + 1):
                        out_file.write(file.readline())
                frame += 1


def generate_velocity(N, mass, temperature, remove_com=True, seed=None):
    """
    Generate velocities from Maxwell-Boltzmann distribution at given temperature.

    Parameters
    ----------
    N : int
        Number of atoms.
    mass : float or array_like
        Atomic mass in g/mol. Can be a scalar (all atoms same mass)
        or array of length N (different masses).
    temperature : float
        Target temperature in Kelvin.
    remove_com : bool, optional
        If True, remove center-of-mass velocity to ensure zero total momentum.
        Default is True.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    vel : ndarray, shape (N, 3)
        Velocities in Å/fs.

    Notes
    -----
    The velocity is generated from Maxwell-Boltzmann distribution:
        P(v) ∝ exp(-m*v²/(2*kb*T))

    The standard deviation of velocity component is:
        σ_v = sqrt(kb*T/m)

    Units:
        - Temperature: K
        - Mass: g/mol
        - Velocity: Å/fs
        - kb = 1.380649e-23 J/K
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert mass to array if scalar
    mass = np.atleast_1d(mass)
    if mass.size == 1:
        mass = np.full(N, mass[0])
    elif mass.size != N:
        raise ValueError(f"Mass array size {mass.size} doesn't match N={N}")

    # Physical constants
    kb = 1.380649e-23  # Boltzmann constant (J/K)
    afu = 6.022140857e23  # Avogadro's number (1/mol)

    # Convert mass from g/mol to kg
    mass_kg = mass / (afu * 1000.0)  # kg

    # Standard deviation of velocity (m/s)
    # sigma_v = sqrt(kb*T/m)
    sigma_v = np.sqrt(kb * temperature / mass_kg)  # m/s

    # Convert to Å/fs: 1 m/s = 1e-10 m / 1e-15 s = 1e-5 Å/fs
    sigma_v_Aps = sigma_v * 1e-5  # Å/fs

    # Generate velocities from normal distribution
    # Each component (vx, vy, vz) follows N(0, sigma_v²)
    vel = np.random.normal(0, sigma_v_Aps[:, np.newaxis], size=(N, 3))

    # Remove center-of-mass velocity
    if remove_com:
        total_momentum = np.sum(vel * mass[:, np.newaxis], axis=0)
        total_mass = np.sum(mass)
        v_com = total_momentum / total_mass
        vel -= v_com

    return vel
