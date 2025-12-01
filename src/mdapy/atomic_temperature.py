# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from mdapy import _atomtemp
import polars as pl
import numpy as np
from mdapy.data import atomic_masses, atomic_numbers


class AtomicTemperature:
    """
    Calculate atomic temperature from velocity fluctuations, in the units of A/fs.

    This class computes the local atomic temperature for each atom by analyzing
    the velocity fluctuations of the atom and its neighbors. The temperature is
    calculated from the kinetic energy distribution.

    Parameters
    ----------
    data : pl.DataFrame
        Atomic data containing at least 'vx', 'vy', 'vz' velocity columns and
        either 'amass' (atomic mass) or 'element' columns.
    verlet_list : np.ndarray
        Neighbor list array of shape (N, max_neigh).
    distance_list : np.ndarray
        Distance list array of shape (N, max_neigh).
    rc : float
        Cutoff radius for neighbor consideration.
    factor : float, default=1.0
        Scaling factor for velocities (e.g., for unit conversion).

    Attributes
    ----------
    data : pl.DataFrame
        Input atomic data.
    verlet_list : np.ndarray
        Neighbor indices.
    distance_list : np.ndarray
        Neighbor distances.
    rc : float
        Cutoff radius.
    factor : float
        Velocity scaling factor.
    T : np.ndarray
        Computed atomic temperatures (K) after calling compute().

    Notes
    -----
    The atomic temperature represents the local kinetic temperature based on
    velocity fluctuations. It differs from the thermodynamic temperature and
    is useful for identifying hot spots, shock fronts, or temperature gradients.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        verlet_list: np.ndarray,
        distance_list: np.ndarray,
        rc: float,
        factor: float = 1.0,
    ) -> None:
        self.data = data
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.rc = rc
        self.factor = factor

    def compute(self) -> None:
        """
        Compute atomic temperatures.

        This method calculates the atomic temperature for each atom based on
        velocity fluctuations and stores the result in the ``T`` attribute.

        Raises
        ------
        AssertionError
            If velocity columns ('vx', 'vy', 'vz') are not present in data.
        ValueError
            If neither 'amass' nor 'element' columns are present, or if an
            unknown element symbol is encountered.
        """
        for i in ["vx", "vy", "vz"]:
            assert i in self.data.columns, "No velocity information."
        if "amass" in self.data.columns:
            amass = self.data["amass"].to_numpy(allow_copy=False)
        elif "element" in self.data.columns:
            ele2mass = {}
            element_unique = self.data["element"].unique().sort()
            for ele in element_unique:
                if ele not in atomic_numbers.keys():
                    raise ValueError(f"Unknown element '{ele}' in atomic_numbers.")
                ele2mass[ele] = atomic_masses[atomic_numbers[ele]]
            amass = self.data.with_columns(
                pl.col("element").replace_strict(ele2mass).alias("amass")
            )["amass"].to_numpy(allow_copy=False)
        else:
            raise ValueError("No atomic mass.")

        self.T = np.zeros(self.data.shape[0], float)
        # cpp part we use A/ps us units, so we change the A/fs to A/ps first.
        _atomtemp.compute_temp(
            self.verlet_list,
            self.distance_list,
            self.data["vx"].to_numpy(allow_copy=False) * 1e3 * self.factor,
            self.data["vy"].to_numpy(allow_copy=False) * 1e3 * self.factor,
            self.data["vz"].to_numpy(allow_copy=False) * 1e3 * self.factor,
            amass,
            self.T,
            self.rc,
        )


if __name__ == "__main__":
    pass
