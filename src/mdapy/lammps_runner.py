# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

try:
    from lammps import lammps
except ImportError:
    raise ImportError(
        "One can install lammps python package: https://docs.lammps.org/Python_install.html"
    )

import ctypes

import numpy as np
import polars as pl
from typing import List, Optional, Tuple, Union

from mdapy.box import Box
from mdapy.data import atomic_masses, atomic_numbers
from mdapy.lammps_potential import silence
from mdapy.system import System

_DEFAULT_DUMP_KEYWORDS = "id type element x y z"
_DEFAULT_THERMO_KEYWORDS = "step temp pe ke etotal press vol density lx ly lz xy xz yz"


class LammpsRunner:
    """
    Persistent LAMMPS runner driving common simulation tasks
    (minimization with/without box relaxation, NVE/NVT/NPT MD)
    on top of an mdapy :class:`System`.

    Parameters
    ----------
    system : System
        Initial mdapy System. Must contain ``x``, ``y``, ``z`` and
        ``element`` columns. If ``vx``, ``vy``, ``vz`` columns exist they
        are pushed into LAMMPS as initial velocities.
    pair_parameter : str
        Multi-line LAMMPS string for pair_style + pair_coeff (and any
        related commands). Passed to ``commands_string``.
    element_list : list of str
        Element ordering. Defines the LAMMPS atom-type index (1-based).
        Every ``element`` in ``system.data`` must appear here.
    units : str, optional
        LAMMPS units. Only ``"metal"`` is supported.
    cmdargs : list of str, optional
        Extra args appended to ``lammps(cmdargs=...)`` after the default
        ``["-echo", "none", "-log", "none", "-screen", "none"]``. Use this
        for accelerator packages (Kokkos, OMP, GPU).
    extra_commands : str, optional
        Extra LAMMPS commands executed once after the box/atoms are set up
        and *before* ``pair_parameter`` (e.g. ``"newton on"`` or
        ``"package omp 8"``).
    masses : dict[str, float], optional
        Override masses (in g/mol). Keys are element symbols. Any element
        not in the dict falls back to ``data.atomic_masses``.
    log_file : str, optional
        If given, LAMMPS writes its log to this file (``log <file>``);
        otherwise no log file is created.
    silence_lammps : bool, optional
        If True (default), redirect LAMMPS stdout/stderr to /dev/null
        across all calls. Set to False to see LAMMPS output for debugging.
    """

    def __init__(
        self,
        system: System,
        pair_parameter: str,
        element_list: List[str],
        units: str = "metal",
        cmdargs: Optional[List[str]] = None,
        extra_commands: Optional[str] = None,
        masses: Optional[dict] = None,
        log_file: Optional[str] = None,
        silence_lammps: bool = True,
    ) -> None:
        assert units == "metal", "Only support metal units now."
        for col in ("x", "y", "z", "element"):
            assert col in system.data.columns, f"system data missing column {col!r}."
        for ele in system.data["element"].unique():
            assert ele in element_list, f"element_list missing element {ele!r}."

        self.units = units
        self.element_list = list(element_list)
        self.pair_parameter = pair_parameter
        self.extra_commands = extra_commands
        self.masses = dict(masses) if masses else {}
        self._closed = False
        self._rotate: Optional[np.ndarray] = None  # input frame -> lammps frame
        self._boundary = np.asarray(system.box.boundary, dtype=np.int32).copy()
        self.silence_lammps = silence_lammps

        base_cmdargs = ["-echo", "none"]
        if silence_lammps:
            base_cmdargs += ["-screen", "none"]
        if log_file:
            base_cmdargs += ["-log", log_file]
        else:
            base_cmdargs += ["-log", "none"]
        if cmdargs:
            base_cmdargs += list(cmdargs)

        with silence(self.silence_lammps):
            self.lmp = lammps(cmdargs=base_cmdargs)
            try:
                self._setup(system)
            except Exception:
                self.lmp.close()
                raise

    # ------------------------------------------------------------------ setup

    def _setup(self, system: System) -> None:
        box = system.box
        data = system.data
        N_atom = data.shape[0]

        boundary = " ".join(["p" if i == 1 else "s" for i in box.boundary])
        self.lmp.commands_string(f"units {self.units}")
        self.lmp.commands_string(f"boundary {boundary}")
        self.lmp.commands_string("atom_style atomic")
        self.lmp.commands_string("atom_modify map array")

        num_type = len(self.element_list)
        create_box = (
            f"lattice custom 1.0 "
            f"a1 {box.box[0, 0]} {box.box[0, 1]} {box.box[0, 2]} "
            f"a2 {box.box[1, 0]} {box.box[1, 1]} {box.box[1, 2]} "
            f"a3 {box.box[2, 0]} {box.box[2, 1]} {box.box[2, 2]} "
            f"basis 0.0 0.0 0.0 triclinic/general\n"
            f"create_box {num_type} NULL 0 1 0 1 0 1"
        )
        self.lmp.commands_string(create_box)

        ele2type = {j: i + 1 for i, j in enumerate(self.element_list)}
        type_list = data.select(
            pl.col("element")
            .replace_strict(ele2type, return_dtype=pl.Int32)
            .rechunk()
            .alias("type")
        )["type"].to_numpy(allow_copy=False)
        id_list = np.arange(1, N_atom + 1)

        if box.is_general_box():
            box_lmp, rotate = box.align_to_lammps_box()
            self._rotate = rotate
            x_list = (
                (data.select("x", "y", "z").to_numpy() - box.origin) @ rotate
            ).flatten()
        else:
            self._rotate = None
            x_list = (
                data.select(
                    pl.col("x") - box.origin[0],
                    pl.col("y") - box.origin[1],
                    pl.col("z") - box.origin[2],
                )
                .to_numpy()
                .flatten()
            )

        N_lmp = self.lmp.create_atoms(N_atom, id_list, type_list, x_list)
        assert N_atom == N_lmp, "Create atoms incorrectly."

        for i, ele in enumerate(self.element_list):
            mass = self.masses.get(ele, atomic_masses[atomic_numbers[ele]])
            self.lmp.commands_string(f"mass {i + 1} {mass}")

        if self.extra_commands:
            self.lmp.commands_string(self.extra_commands)

        self.lmp.commands_string(self.pair_parameter)

        if all(c in data.columns for c in ("vx", "vy", "vz")):
            v = data.select("vx", "vy", "vz").to_numpy()
            if self._rotate is not None:
                v = v @ self._rotate
            v_flat = np.ascontiguousarray(v.flatten(), dtype=np.float64)
            buf = (ctypes.c_double * v_flat.size)(*v_flat)
            self.lmp.scatter_atoms("v", 1, 3, buf)

    # ----------------------------------------------------------------- ensure

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("LammpsRunner has been closed.")

    # ------------------------------------------------------------- minimizers

    def minimize(
        self,
        etol: float = 1e-8,
        ftol: float = 1e-8,
        maxiter: int = 10000,
        maxeval: int = 100000,
        style: str = "cg",
    ) -> None:
        """
        Energy minimization at fixed cell.

        Parameters
        ----------
        etol, ftol : float
            Energy / force tolerances (LAMMPS ``minimize`` args).
        maxiter, maxeval : int
            Max iterations and force evaluations.
        style : str
            Minimization style passed to ``min_style`` (cg, sd, fire, ...).
        """
        self._ensure_open()
        with silence(self.silence_lammps):
            self.lmp.commands_string(f"min_style {style}")
            self.lmp.commands_string(f"minimize {etol} {ftol} {maxiter} {maxeval}")

    def minimize_box(
        self,
        etol: float = 1e-8,
        ftol: float = 1e-8,
        maxiter: int = 10000,
        maxeval: int = 100000,
        style: str = "cg",
        pressure_bar: float = 0.0,
        relax_style: str = "iso",
    ) -> None:
        """
        Energy minimization with simultaneous box relaxation
        (``fix box/relax``).

        Parameters
        ----------
        pressure_bar : float
            Target external pressure (bar in metal units).
        relax_style : str
            One of ``"iso"``, ``"aniso"``, ``"tri"``. For ``"tri"`` all six
            components are relaxed independently.
        """
        self._ensure_open()
        if relax_style == "tri":
            relax_cmd = f"fix __boxrelax all box/relax tri {pressure_bar}"
        elif relax_style in ("iso", "aniso"):
            relax_cmd = f"fix __boxrelax all box/relax {relax_style} {pressure_bar}"
        else:
            raise ValueError(f"relax_style must be iso/aniso/tri, got {relax_style!r}")

        with silence(self.silence_lammps):
            self.lmp.commands_string(f"min_style {style}")
            self.lmp.commands_string(relax_cmd)
            self.lmp.commands_string(f"minimize {etol} {ftol} {maxiter} {maxeval}")
            self.lmp.commands_string("unfix __boxrelax")

    # ------------------------------------------------------------------ MD

    def run_md(
        self,
        ensemble: str = "nvt",
        nsteps: int = 1000,
        timestep: float = 0.001,
        temp: Union[float, Tuple[float, float]] = 300.0,
        tdamp: float = 0.1,
        pressure_bar: Union[float, Tuple[float, float]] = 0.0,
        pdamp: float = 1.0,
        press_style: str = "iso",
        init_velocity_temp: Optional[float] = None,
        velocity_seed: int = 12345,
        thermo_freq: int = 100,
        thermo_keywords: Optional[str] = None,
        dump_file: Optional[str] = None,
        dump_freq: int = 100,
        dump_keywords: Optional[str] = None,
    ) -> None:
        """
        Run molecular dynamics.

        Parameters
        ----------
        ensemble : str
            One of ``"nve"``, ``"nvt"``, ``"npt"``.
        nsteps : int
            Number of integration steps.
        timestep : float
            MD timestep (ps in metal units).
        temp : float or (float, float)
            Target temperature for nvt/npt. A tuple ``(T_start, T_stop)``
            gives a linear ramp; a scalar uses it for both.
        tdamp : float
            Temperature damping (ps).
        pressure_bar : float or (float, float)
            Target pressure (bar) for npt; tuple gives a ramp.
        pdamp : float
            Pressure damping (ps).
        press_style : str
            One of ``"iso"``, ``"aniso"``, ``"tri"``. Controls how the
            barostat couples cell components.
        init_velocity_temp : float, optional
            If set, ``velocity all create <T> <seed> dist gaussian`` is
            issued before the run. If ``None`` and the system already had
            vx/vy/vz, those persist; otherwise atoms start at rest.
        velocity_seed : int
            RNG seed for ``velocity create``.
        thermo_freq : int
            ``thermo`` output frequency. Set to 0 to keep the LAMMPS
            default behaviour.
        thermo_keywords : str, optional
            Custom ``thermo_style custom <...>`` columns. Defaults to a
            common set including step/temp/pe/press/vol/cell.
        dump_file : str, optional
            If set, a ``custom`` dump is created for the run.
        dump_freq : int
            Dump frequency (every N steps). Ignored if ``dump_file`` is None.
        dump_keywords : str, optional
            Columns for the dump (LAMMPS ``custom`` syntax). Defaults to
            ``"id type element x y z"``.
        """
        self._ensure_open()
        ensemble = ensemble.lower()
        if ensemble not in ("nve", "nvt", "npt"):
            raise ValueError(f"ensemble must be nve/nvt/npt, got {ensemble!r}")

        t_start, t_stop = (temp, temp) if np.isscalar(temp) else temp
        p_start, p_stop = (
            (pressure_bar, pressure_bar) if np.isscalar(pressure_bar) else pressure_bar
        )

        with silence(self.silence_lammps):
            self.lmp.commands_string(f"timestep {timestep}")

            if init_velocity_temp is not None:
                self.lmp.commands_string(
                    f"velocity all create {init_velocity_temp} {velocity_seed} "
                    "dist gaussian mom yes rot yes"
                )

            if thermo_freq > 0:
                self.lmp.commands_string(f"thermo {thermo_freq}")
                self.lmp.commands_string(
                    f"thermo_style custom {thermo_keywords or _DEFAULT_THERMO_KEYWORDS}"
                )

            dump_id = "__mdapy_dump"
            if dump_file is not None:
                kw = dump_keywords or _DEFAULT_DUMP_KEYWORDS
                self.lmp.commands_string(
                    f"dump {dump_id} all custom {dump_freq} {dump_file} {kw}"
                )
                if "element" in kw:
                    self.lmp.commands_string(
                        f"dump_modify {dump_id} format float %.15g sort id element "
                        + " ".join(self.element_list)
                    )
                else:
                    self.lmp.commands_string(
                        f"dump_modify {dump_id} format float %.15g sort id"
                    )

            fix_id = "__mdapy_md"
            if ensemble == "nve":
                self.lmp.commands_string(f"fix {fix_id} all nve")
            elif ensemble == "nvt":
                self.lmp.commands_string(
                    f"fix {fix_id} all nvt temp {t_start} {t_stop} {tdamp}"
                )
            else:  # npt
                self.lmp.commands_string(
                    f"fix {fix_id} all npt temp {t_start} {t_stop} {tdamp} "
                    f"{press_style} {p_start} {p_stop} {pdamp}"
                )

            try:
                self.lmp.commands_string(f"run {nsteps}")
            finally:
                self.lmp.commands_string(f"unfix {fix_id}")
                if dump_file is not None:
                    self.lmp.commands_string(f"undump {dump_id}")

    # --------------------------------------------------------------- extract

    def get_system(self) -> System:
        """
        Build a fresh mdapy :class:`System` from the current LAMMPS state.

        Notes
        -----
        Positions/velocities are returned in the *current* LAMMPS frame.
        For a non-orthogonal input box that required rotation at setup,
        the returned System lives in the rotated (LAMMPS) frame — call
        ``System.write_data`` and reload if you need the original frame.
        """
        self._ensure_open()
        N = self.lmp.get_natoms()
        with silence(self.silence_lammps):
            ids = np.asarray(self.lmp.numpy.extract_atom("id")[:N])
            order = np.argsort(ids)
            types = np.asarray(self.lmp.numpy.extract_atom("type")[:N])[order]
            xyz = np.asarray(self.lmp.numpy.extract_atom("x")[:N])[order]
            v_arr = self.lmp.numpy.extract_atom("v")
            vxyz = np.asarray(v_arr[:N])[order] if v_arr is not None else None
            boxlo, boxhi, xy, yz, xz, _periodicity, _box_change = self.lmp.extract_box()

        boxlo = np.asarray(boxlo, dtype=float)
        boxhi = np.asarray(boxhi, dtype=float)
        a = np.array([boxhi[0] - boxlo[0], 0.0, 0.0])
        b = np.array([xy, boxhi[1] - boxlo[1], 0.0])
        c = np.array([xz, yz, boxhi[2] - boxlo[2]])
        new_box = Box(np.vstack([a, b, c, boxlo]), boundary=self._boundary)

        type2ele = {i + 1: ele for i, ele in enumerate(self.element_list)}
        cols = {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "element": [type2ele[int(t)] for t in types],
        }
        if vxyz is not None:
            cols["vx"] = vxyz[:, 0]
            cols["vy"] = vxyz[:, 1]
            cols["vz"] = vxyz[:, 2]
        df = pl.DataFrame(cols)
        return System(data=df, box=new_box)

    # ---------------------------------------------------------------- close

    def close(self) -> None:
        if not self._closed:
            try:
                self.lmp.close()
            finally:
                self._closed = True

    def __enter__(self) -> "LammpsRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    import os
    import sys

    if sys.platform == "darwin":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    from mdapy import build_crystal

    path = "md_elastic_run"
    system = build_crystal("Al", "fcc", 4.05, nx=3, ny=3, nz=3)
    pair_parameter = """
pair_style nep/kk
pair_coeff * * tests/input_files/UNEP-v1.txt Al
"""
    element_list = ["Al"]
    with LammpsRunner(
        system,
        pair_parameter,
        element_list,
        cmdargs=[
            "-k",
            "on",
            "-sf",
            "kk",
            "-pk",
            "kokkos",
            "newton",
            "on",
            "neigh",
            "half",
        ],
        log_file=f"{path}/log.lammps",
        silence_lammps=False,
    ) as r:
        # r.minimize(etol=1e-8, ftol=1e-8)  # fixed cell
        r.minimize_box(pressure_bar=0.0, relax_style="iso")  # fix box/relax
        relax = r.get_system()
        relax.write_data(f"{path}/mini.data", element_list=element_list)
        r.lmp.commands_string("reset_timestep 0")
        r.lmp.commands_string("compute tt all temp")
        r.lmp.commands_string("compute 1 all centroid/stress/atom tt")
        r.run_md(
            ensemble="npt",
            nsteps=10000,
            timestep=0.001,
            temp=(50, 300),
            tdamp=0.1,
            pressure_bar=0.0,
            pdamp=1.0,
            press_style="iso",
            init_velocity_temp=50,
            velocity_seed=42,
            thermo_freq=100,
            thermo_keywords="step temp pe press vol",
            dump_file=f"{path}/traj.dump",
            dump_freq=500,
            dump_keywords="id type element x y z vx vy vz c_1[*]",
        )
        new_system = r.get_system()  # mdapy System with updated box/pos/velocity
        new_system.write_data(f"{path}/relax.data", element_list=element_list)
