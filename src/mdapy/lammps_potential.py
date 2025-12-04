# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

try:
    from lammps import lammps
except ImportError:
    raise ImportError(
        "One can install lammps python package: https://docs.lammps.org/Python_install.html"
    )
from mdapy.calculator import CalculatorMP
from mdapy.box import Box

import numpy as np
import polars as pl
from typing import List, Any


class LammpsPotential(CalculatorMP):
    """
    LAMMPS-based calculator that runs a single-point evaluation to obtain
    per-atom energies, forces, virials and the global stress.

    Parameters
    ----------
    pair_parameter : str
        The LAMMPS pair style / pair coeff commands as a single string.
        This string is passed directly to LAMMPS with `commands_string`.
    element_list : List[str]
        List of element names supported by this potential. The index in this
        list defines the corresponding LAMMPS atom type (1-based).
    units : str, optional
        Units for LAMMPS (default ``"metal"``). Currently the code asserts
        that units == "metal".
    centroid_stress : bool, optional
        If True, uses `compute centroid/stress/atom NULL` in LAMMPS;
        otherwise uses `compute stress/atom NULL`.
    """

    def __init__(
        self,
        pair_parameter: str,
        element_list: List[str],
        units: str = "metal",
        centroid_stress: bool = False,
    ) -> None:
        self.pair_parameter = pair_parameter
        self.element_list = element_list
        self.units = units
        assert units == "metal", "Only support metal units now."
        self.centroid_stress = centroid_stress

    def calculate(self, data: pl.DataFrame, box: Box) -> None:
        """
        Run LAMMPS to calculate per-atom energies, forces and virials and
        compute global stress.

        This function validates inputs, constructs a triclinic LAMMPS box,
        creates atoms, sets up computes, runs `run 0`, extracts LAMMPS
        computed quantities, converts units, reorders virials, and stores
        results in ``self.results``.

        Parameters
        ----------
        data : polars.DataFrame
            Polars DataFrame with required columns: "x", "y", "z", "element".
        box : Box
            Box object from mdapy.

        Notes
        -----
        - The method relies on `lammps` Python bindings to exist and provide:
          - `lammps(cmdargs=...)`, `commands_string`, `create_atoms`,
          - `numpy.extract_atom(...)`, `numpy.extract_compute(...)`,
          - `numpy.extract_atom("f")`, and `.close()`.
        - Virial unit conversion: `virial = virial / 1e4 / 160.21766208`
          (converts LAMMPS reported units to eV).
        - The final global stress is computed as:
          stress = -(virial_tensor + virial_tensor.T) / (2 * box.volume)
          and returned in Voigt order [xx, yy, zz, yz, xz, xy].
        """

        for i in ["x", "y", "z", "element"]:
            assert i in data.columns, f"data does not have {i} information."
        for i in data["element"].unique():
            assert i in self.element_list, f"element_list dose not have {i} element."
        boundary = " ".join(["p" if i == 1 else "s" for i in box.boundary])
        N_atom = data.shape[0]
        lmp = lammps(cmdargs=["-echo", "none", "-log", "none", "-screen", "none"])

        try:
            lmp.commands_string(f"units {self.units}")
            lmp.commands_string(f"boundary {boundary}")
            lmp.commands_string("atom_style atomic")

            num_type = len(self.element_list)
            create_box = f"""lattice custom 1.0 a1 {box.box[0, 0]} {box.box[0, 1]} {box.box[0, 2]} a2 {box.box[1, 0]} {box.box[1, 1]} {box.box[1, 2]} a3 {box.box[2, 0]} {box.box[2, 1]} {box.box[2, 2]} basis 0.0 0.0 0.0 triclinic/general
            create_box {num_type} NULL 0 1 0 1 0 1"""
            lmp.commands_string(create_box)

            ele2type = {j: i + 1 for i, j in enumerate(self.element_list)}
            type_list = data.select(
                pl.col("element")
                .replace_strict(ele2type, return_dtype=pl.Int32)
                .rechunk()
                .alias("type")
            )["type"].to_numpy(allow_copy=False)
            id_list = data.with_row_index("id", offset=1)["id"].to_numpy(
                allow_copy=False
            )
            x_list = (
                data.select(
                    pl.col("x") - box.origin[0],
                    pl.col("y") - box.origin[1],
                    pl.col("z") - box.origin[2],
                )
                .to_numpy()
                .flatten()
            )

            N_lmp = lmp.create_atoms(N_atom, id_list, type_list, x_list)
            assert N_atom == N_lmp, "Create atoms incorrectly."
            for i in range(num_type):
                lmp.commands_string(f"mass {i + 1} 1.0")
            if self.centroid_stress:
                lmp.commands_string("compute 1 all centroid/stress/atom NULL")
            else:
                lmp.commands_string("compute 1 all stress/atom NULL")
            lmp.commands_string("compute 2 all pe/atom")

            lmp.commands_string(self.pair_parameter)
            lmp.commands_string("run 0")
            sort_index = np.argsort(lmp.numpy.extract_atom("id")[:N_atom])
            energy = np.asarray(
                lmp.numpy.extract_compute("2", 1, 1)[:N_atom][sort_index]
            )
            force = np.asarray(lmp.numpy.extract_atom("f")[:N_atom][sort_index])
            # xx, yy, zz, xy, xz, yz, yx, zx, zy.
            # xx, yy, zz, xy, xz, yz
            virial = -np.asarray(
                lmp.numpy.extract_compute("1", 1, 2)[:N_atom][sort_index]
            )
        except Exception as e:
            raise e
        finally:
            lmp.close()

        virial = virial / 1e4 / 160.21766208  # bar to eV
        # v_xx, v_xy, v_xz, v_yx, v_yy, v_yz, v_zx, v_zy, v_zz
        virial = self._reorder_virial(virial)
        self.results["energies"] = energy
        self.results["forces"] = force
        self.results["virials"] = virial
        # Calculate stress tensor from virials
        v = virial.sum(axis=0)  # Sum virials over all atoms
        # Reshape to 3×3 matrix: v_xx, v_xy, v_xz, v_yx, v_yy, v_yz, v_zx, v_zy, v_zz
        v = v.reshape(3, 3)
        # Stress = -(virial + virial^T) / (2 * volume)
        stress = (-0.5 * (v + v.T) / box.volume).ravel()
        # Convert to Voigt notation: [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]
        stress = stress[[0, 4, 8, 5, 2, 1]]
        self.results["stress"] = stress

    def _reorder_virial(self, v: np.ndarray) -> np.ndarray:
        """
        Reorder virial array into a 9-component per-atom format.

        Parameters
        ----------
        v : numpy.ndarray
            Input virial array with shape (N, 9) or (N, 6). The code expects
            LAMMPS-style ordering. If shape is (N,9), the function unpacks as:
            xx, yy, zz, xy, xz, yz, yx, zx, zy.
            If shape is (N,6), it's treated as symmetric: xx, yy, zz, xy, xz, yz.

        Returns
        -------
        numpy.ndarray
            Array of shape (N, 9) with columns ordered as:
            [xx, xy, xz, yx, yy, yz, zx, zy, zz]
        """
        if v.shape[1] == 9:
            xx, yy, zz, xy, xz, yz, yx, zx, zy = v.T

        elif v.shape[1] == 6:
            # symmetric case
            xx, yy, zz, xy, xz, yz = v.T
            yx, zx, zy = xy, xz, yz
        else:
            raise ValueError("Input must have shape (N,9) or (N,6)")

        out = np.column_stack([xx, xy, xz, yx, yy, yz, zx, zy, zz])
        return out

    def get_energies(self, data: pl.DataFrame, box: Box) -> Any:
        """
        Return per-atom energies. If not already computed, triggers calculate().

        Parameters
        ----------
        data : polars.DataFrame
        box : Box

        Returns
        -------
        Any
            Stored per-atom energies (as placed into self.results["energies"]).
        """
        if "energies" not in self.results.keys():
            self.calculate(data, box)
        return self.results["energies"]

    def get_energy(self, data: pl.DataFrame, box: Box) -> Any:
        """
        Return total energy (sum of per-atom energies).

        Parameters
        ----------
        data : polars.DataFrame
        box : Box

        Returns
        -------
        Any
            Sum of per-atom energies.
        """
        return self.get_energies(data, box).sum()

    def get_forces(self, data: pl.DataFrame, box: Box) -> Any:
        """
        Return per-atom forces; compute if necessary.
        """
        if "forces" not in self.results.keys():
            self.calculate(data, box)
        return self.results["forces"]

    def get_stress(self, data: pl.DataFrame, box: Box) -> Any:
        """
        Return global stress in Voigt order [xx, yy, zz, yz, xz, xy]; compute if necessary.
        """
        if "stress" not in self.results.keys():
            self.calculate(data, box)
        return self.results["stress"]

    def get_virials(self, data: pl.DataFrame, box: Box) -> Any:
        """
        Return per-atom virials (9 components) and compute if necessary.
        """
        if "virials" not in self.results.keys():
            self.calculate(data, box)
        return self.results["virials"]


if __name__ == "__main__":
    pass
