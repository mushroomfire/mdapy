# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

"""
Finite-temperature elastic constants from MD stress-strain
==========================================================

Implements the Aidan Thompson finite-T recipe (LAMMPS examples/ELASTIC/T):

1. (Optional) NPT pre-relax to equilibrium volume V(T) at target pressure.
2. Equilibrate the cell at T (NVT, langevin); average reference stress sigma_0.
3. For each Voigt direction d in {1..6} and each sign s in {-1,+1}:
   - Reset the cell from a saved restart.
   - change_box delta = s*delta*L0 in direction d (with proper xy/xz/yz coupling).
   - NVT (isothermal) or NVE (adiabatic) for n_equil + n_run steps.
   - Time-average the pressure tensor -> sigma_pm.
4. C_id = -(sigma_pos[i] - sigma_neg[i]) / (2*delta) for i,d in 1..6.
5. Symmetrize: C_ij = (C_ij + C_ji) / 2.

Each LAMMPS instance is a single subprocess; the 12 deformations within one
temperature run on a `ProcessPoolExecutor` so a serial-Kokkos LAMMPS build
parallelises to N cores. ``MDElastic.scan_parallel`` further runs all
temperatures concurrently (1 NPT + 12 deformations per T, 13*N tasks total).

Driver uses the LAMMPS Python module (``from lammps import lammps``).
``pair_style`` / ``pair_coeff`` are passed by the caller — for serial CPU
runs use ``-k on -sf kk`` only (no ``g 1`` and no ``-pk kokkos`` GPU args).
"""

from __future__ import annotations

import os
import tempfile
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np
import polars as pl

from mdapy.system import System

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

try:
    from lammps import lammps as _lammps  # noqa: F401  (used in subprocess fns)
except ImportError:
    raise ImportError(
        "lammps Python module is required for MDElastic. "
        "Install via conda: `conda install -c conda-forge lammps`, or build "
        "LAMMPS with -DBUILD_SHARED_LIBS=ON -DPKG_PYTHON=ON."
    )


_EAA_PER_BAR = 1.0e-4   # 1 bar = 1e-4 GPa


# ============================================================
# Result container
# ============================================================


class MDElasticResult:
    """Output of one MDElastic.run() call.

    Attributes
    ----------
    cij_voigt : (6,6) ndarray
        Full 6x6 elastic stiffness tensor in GPa (already symmetrized).
    V_eq : float
        Equilibrium cell volume at T (Angstrom^3).
    T_actual : float
        Time-averaged temperature during reference run (K).
    stress_ref : (6,) ndarray
        Time-averaged reference stress in GPa (should be ~0 if relax_volume).
    ensemble : str
        "isothermal" (NVT) or "adiabatic" (NVE after equilibration).
    temperature : float
        Target temperature (K).
    """

    def __init__(
        self,
        cij_voigt: np.ndarray,
        V_eq: float,
        T_actual: float,
        stress_ref: np.ndarray,
        ensemble: str,
        temperature: float,
    ) -> None:
        self.cij_voigt = np.asarray(cij_voigt, dtype=float)
        self.V_eq = float(V_eq)
        self.T_actual = float(T_actual)
        self.stress_ref = np.asarray(stress_ref, dtype=float)
        self.ensemble = str(ensemble)
        self.temperature = float(temperature)

    def cubic_average(self) -> Tuple[float, float, float]:
        """Cubic-symmetry-averaged (C11, C12, C44) in GPa.

        SQS / defected cells are not strictly cubic; this is the standard
        averaging convention (used by the LAMMPS reference script).
        """
        C = self.cij_voigt
        C11 = (C[0, 0] + C[1, 1] + C[2, 2]) / 3.0
        C12 = (C[0, 1] + C[0, 2] + C[1, 2]) / 3.0
        C44 = (C[3, 3] + C[4, 4] + C[5, 5]) / 3.0
        return C11, C12, C44

    def vrh(self) -> dict:
        """Voigt-Reuss-Hill polycrystalline averages computed on the full 6x6."""
        from mdapy.elastic import ElasticTensor

        return ElasticTensor(self.cij_voigt).vrh()

    def born_stable_cubic(self) -> bool:
        """Born stability for cubic (uses cubic_average)."""
        c11, c12, c44 = self.cubic_average()
        return (c11 > abs(c12)) and (c44 > 0.0) and (c11 + 2.0 * c12 > 0.0)

    def print(self) -> None:
        c11, c12, c44 = self.cubic_average()
        K, G = (c11 + 2 * c12) / 3.0, (c11 - c12 + 3 * c44) / 5.0
        E = 9 * K * G / (3 * K + G) if (3 * K + G) > 0 else 0.0
        print(f"MDElastic @ T={self.temperature:.0f} K ({self.ensemble}):")
        print(f"  V_eq           = {self.V_eq:.3f} A^3")
        print(f"  T_actual       = {self.T_actual:.1f} K")
        print(f"  reference sigma= {np.array2string(self.stress_ref, precision=3)} GPa")
        print(f"  C11 (cubic avg)= {c11:.2f} GPa")
        print(f"  C12 (cubic avg)= {c12:.2f} GPa")
        print(f"  C44 (cubic avg)= {c44:.2f} GPa")
        print(f"  K (V)          = {K:.2f} GPa")
        print(f"  G (V)          = {G:.2f} GPa")
        print(f"  E              = {E:.2f} GPa")


# ============================================================
# Module-level segment functions (subprocess-callable)
# ============================================================
#
# These run inside ProcessPoolExecutor workers. Each takes a plain dict
# of args (picklable) and returns a plain dict of results (picklable).
# A fresh LAMMPS instance is created and torn down per call.


def _spawn_lammps(args: dict):
    """Build a LAMMPS instance from cmdargs in ``args``.

    If ``args['n_threads'] > 1`` we set ``OMP_NUM_THREADS`` accordingly and
    prepend ``-k on t N -sf kk`` to the cmdargs. ``n_threads = 1`` prepends
    ``-k on -sf kk``.

    The stage-level LAMMPS log goes to ``args['stage_log']`` if given;
    otherwise log output is suppressed.
    """
    from lammps import lammps  # local import: subprocess-safe

    n_threads = int(args.get("n_threads", 1))
    if n_threads > 1:
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
    user_args = list(args.get("lammps_cmdargs") or [])
    if n_threads > 1:
        kk_prefix = ["-k", "on", "t", str(n_threads), "-sf", "kk"]
    else:
        kk_prefix = ["-k", "on", "-sf", "kk"]
    cmdargs = kk_prefix + user_args
    if "-screen" not in cmdargs:
        cmdargs += ["-screen", "none"]
    if "-log" not in cmdargs:
        cmdargs += ["-log", args.get("stage_log") or "none"]
    return lammps(cmdargs=cmdargs)


def _apply_thermostat(lmp, T, thermostat, tdamp, seed):
    """Add fix integ (+ optional langevin) for the equilibration phase."""
    if thermostat == "nose-hoover":
        lmp.command(f"fix integ all nvt temp {T} {T} {tdamp}")
    else:  # langevin
        lmp.command("fix integ all nve")
        lmp.command(f"fix thermo all langevin {T} {T} {tdamp} {seed}")


def _switch_to_nve(lmp, thermostat):
    """Adiabatic averaging: drop the thermostat, keep an integrator."""
    if thermostat == "nose-hoover":
        lmp.command("unfix integ")
        lmp.command("fix integ all nve")
    else:  # langevin
        lmp.command("unfix thermo")


def _stop_thermostat(lmp, thermostat, ensemble):
    """Remove integ + thermo fixes (call before clear/restart)."""
    lmp.command("unfix integ")
    if thermostat == "langevin" and ensemble != "adiabatic":
        lmp.command("unfix thermo")


def _read_avp(lmp) -> np.ndarray:
    """Read fix avp -> 6-vector pressure in bar in elastic Voigt order
    [pxx, pyy, pzz, pyz, pxz, pxy]."""
    comps = np.array(
        [lmp.extract_fix("avp", 0, 1, i, 0) for i in range(6)], dtype=float
    )
    return np.array(
        [comps[0], comps[1], comps[2], comps[5], comps[4], comps[3]], dtype=float
    )


def _run_reference_segment(args: dict) -> dict:
    """Single-T reference: NPT pre-relax (optional) + reference NVT/NVE.

    Saves a restart file at args['restart_path']. If args['npt_log'] is
    provided, the NPT phase writes step / temp / press / lx ly lz / vol /
    pe ke etotal / pxx-pyz to that log file (every 100 steps).
    """
    lmp = _spawn_lammps(args)
    try:
        lmp.commands_string(f"""
units metal
dimension 3
boundary p p p
atom_modify map array
read_data {args['data_path']}
change_box all triclinic
""")
        lmp.command(f"pair_style {args['pair_style']}")
        lmp.command(f"pair_coeff {args['pair_coeff']}")
        lmp.command(f"timestep {args['timestep']}")
        lmp.command(
            f"velocity all create {args['T']} {args['seed']} loop geom"
        )

        # Configure thermo style (used for the NPT log + console output).
        lmp.commands_string("""
thermo_style custom step temp press lx ly lz vol pe ke etotal pxx pyy pzz pxy pxz pyz
thermo_modify norm no flush yes
thermo 100
""")

        T = args["T"]
        # Stage 1a: optional NPT pre-relax. Thermo + LAMMPS log already
        # carry full output via the -log flag set in _spawn_lammps, so the
        # whole stage 1 (NPT + ref NVT + averaging) ends up in one file
        # ``ref_T{T}.log``.
        if args["relax_volume"]:
            lmp.command(
                f"fix npt_relax all npt temp {T} {T} {args['tdamp']} "
                f"{args['pressure_coupling']} "
                f"{args['P']} {args['P']} {args['pdamp']}"
            )
            lmp.command(f"run {args['n_relax']}")
            lmp.command("unfix npt_relax")

        # Stage 1: reference NVT equilibration + averaging.
        _apply_thermostat(lmp, T, args["thermostat"], args["tdamp"], args["seed"])
        lmp.command(f"run {args['n_equil']}")
        if args["ensemble"] == "adiabatic":
            _switch_to_nve(lmp, args["thermostat"])

        # `ave running` accumulates samples across the whole n_run window,
        # so the final extracted value is the time-average over the entire
        # averaging phase (not just the last nfreq steps).
        lmp.command(
            f"fix avp all ave/time {args['nevery']} {args['nrepeat']} "
            f"{args['nfreq']} c_thermo_press mode vector ave running"
        )
        lmp.command(
            f"fix avT all ave/time {args['nevery']} {args['nrepeat']} "
            f"{args['nfreq']} c_thermo_temp ave running"
        )
        lmp.command(f"run {args['n_run']}")
        sigma_ref = _read_avp(lmp)
        T_actual = float(lmp.extract_fix("avT", 0, 0, 0, 0))
        V_eq = float(lmp.get_thermo("vol"))
        lx0 = float(lmp.get_thermo("lx"))
        ly0 = float(lmp.get_thermo("ly"))
        lz0 = float(lmp.get_thermo("lz"))

        lmp.command("unfix avp")
        lmp.command("unfix avT")
        _stop_thermostat(lmp, args["thermostat"], args["ensemble"])
        lmp.command(f"write_restart {args['restart_path']}")
    finally:
        lmp.close()

    return dict(
        sigma_ref_bar=sigma_ref.tolist(),
        T_actual=T_actual,
        V_eq=V_eq,
        len0=(lx0, ly0, lz0),
        restart_path=args["restart_path"],
        T=args["T"],
    )


def _run_deform_segment(args: dict) -> dict:
    """One deformation segment (single direction, single sign)."""
    lmp = _spawn_lammps(args)
    try:
        lmp.commands_string("""
units metal
dimension 3
boundary p p p
atom_modify map array
box tilt large
""")
        lmp.command(f"read_restart {args['restart_path']}")
        lmp.command(f"pair_style {args['pair_style']}")
        lmp.command(f"pair_coeff {args['pair_coeff']}")
        lmp.command(f"timestep {args['timestep']}")
        lmp.commands_string("""
thermo_style custom step temp press lx ly lz vol pe ke etotal pxx pyy pzz pxy pxz pyz
thermo_modify norm no flush yes
thermo 100
""")

        d = args["direction"]
        signed_delta = args["signed_delta"]
        len0 = args["len0"]
        # Reference length per direction (matches LAMMPS displace.mod):
        # dir 1: lx0; dir 2: ly0; dirs 3, 4, 5: lz0; dir 6: ly0.
        ref_len = {1: len0[0], 2: len0[1], 3: len0[2],
                   4: len0[2], 5: len0[2], 6: len0[1]}[d]
        d_abs = signed_delta * ref_len
        xy = float(lmp.get_thermo("xy"))
        xz = float(lmp.get_thermo("xz"))
        yz = float(lmp.get_thermo("yz"))
        dxy = signed_delta * xy
        dxz = signed_delta * xz
        dyz = signed_delta * yz
        if d == 1:
            lmp.command(
                f"change_box all x delta 0 {d_abs} "
                f"xy delta {dxy} xz delta {dxz} remap units box"
            )
        elif d == 2:
            lmp.command(
                f"change_box all y delta 0 {d_abs} yz delta {dyz} remap units box"
            )
        elif d == 3:
            lmp.command(
                f"change_box all z delta 0 {d_abs} remap units box"
            )
        elif d == 4:
            lmp.command(f"change_box all yz delta {d_abs} remap units box")
        elif d == 5:
            lmp.command(f"change_box all xz delta {d_abs} remap units box")
        elif d == 6:
            lmp.command(f"change_box all xy delta {d_abs} remap units box")
        else:
            raise ValueError(f"bad direction {d}")

        T = args["T"]
        _apply_thermostat(lmp, T, args["thermostat"], args["tdamp"], args["seed"])
        lmp.command(f"run {args['n_equil']}")
        if args["ensemble"] == "adiabatic":
            _switch_to_nve(lmp, args["thermostat"])
        lmp.command(
            f"fix avp all ave/time {args['nevery']} {args['nrepeat']} "
            f"{args['nfreq']} c_thermo_press mode vector"
        )
        lmp.command(f"run {args['n_run']}")
        sigma = _read_avp(lmp)
    finally:
        lmp.close()

    return dict(
        direction=int(d),
        sign=int(np.sign(signed_delta)),
        sigma_bar=sigma.tolist(),
    )


# ============================================================
# Main class
# ============================================================


class MDElastic:
    """
    Finite-T elastic constants via stress-strain MD (LAMMPS).

    Parameters
    ----------
    system : System
        mdapy System with element labels; will be written to a LAMMPS data
        file internally.
    pair_style : str
        LAMMPS ``pair_style`` argument, e.g. ``"eam/alloy"``, ``"nep"``,
        ``"nep/kk"``, ``"meam"``.
    pair_coeff : str
        Full ``pair_coeff`` line *after* ``pair_coeff``, e.g.
        ``"* * NiCoCr.lammps.eam Ni Co Cr"``. Element order here must match
        ``elements`` below.
    elements : list of str
        Element symbol order used to map ``system.data["element"]`` to LAMMPS
        atom types (1..N). Must match the species list in ``pair_coeff``.
    temperature : float
        Target temperature (K).
    pressure : float, optional
        Target pressure for the optional NPT relax stage (bar). Default 0.
    pressure_coupling : {"iso", "aniso"}, optional
        ``fix npt`` coupling. ``iso`` keeps the cell shape isotropic (rescales
        all three lattice vectors together — appropriate for nominally cubic
        SQS / chemically-disordered cells). ``aniso`` lets each axis relax
        independently (use for non-cubic structures). Default ``iso``.
    delta : float, optional
        Voigt strain magnitude. LAMMPS reference uses 2e-2; 5e-3..2e-2 is
        the useful range. Default 0.01.
    timestep : float, optional
        MD timestep (ps). Default 0.001.
    n_equil : int, optional
        Equilibration steps per stage. Default 10000.
    n_run : int, optional
        Averaging steps per stage. Default 5000.
    nevery, nrepeat : int, optional
        Parameters of LAMMPS ``fix ave/time``. The averaging window is
        ``nevery * nrepeat``. Defaults 10 and 10.
    ensemble : {"isothermal", "adiabatic"}, optional
        Integration during the *averaging* run. Default "isothermal".
    thermostat : {"nose-hoover", "langevin"}, optional
        Default "nose-hoover" (Nose-Hoover NVT, lower stress noise).
    thermostat_damp : float, optional
        Thermostat tau in ps. Default ``100 * timestep``.
    seed : int, optional
        RNG seed. Default 87287.
    relax_volume : bool, optional
        Run NPT pre-relax. Default True.
    n_relax : int, optional
        NPT pre-relax steps. Default 50000.
    pdamp : float, optional
        Barostat tau in ps. Default ``1000 * timestep``.
    n_workers : int, optional
        Number of *deformation* worker processes. The 12 deformations are
        scheduled across this pool. Default ``min(12, os.cpu_count())``.
        Set to 1 to run sequentially in the main process.
    n_threads_ref : int, optional
        OpenMP threads used by the *reference* (NPT + ref-NVT) LAMMPS
        instance. With Kokkos OpenMP backend this becomes ``-k on t N -sf
        kk``. Default 1. Increase when the reference phase is the
        bottleneck (e.g. n_relax >> n_run + n_equil).
    n_threads_def : int, optional
        OpenMP threads per deformation worker. Default 1 (deformation
        phase usually wants more processes than threads).
    log_dir : str or Path, optional
        Per-stage log files written here. The reference (NPT + ref NVT)
        log is ``log_dir/ref_T{T}.log`` (one per temperature); each
        deformation gets ``log_dir/def_T{T}_d{dir}_{sign}.log``. Each log
        carries thermo every 100 steps with columns
        ``step temp press lx ly lz vol pe ke etotal pxx pyy pzz pxy pxz pyz``.
    lammps_cmdargs : list of str, optional
        Extra args for ``lammps(cmdargs=...)``. ``-k on [t N] -sf kk`` is
        injected automatically based on ``n_threads_*``.
    work_dir : str, optional
        Directory for data + restart files. Default a tmpdir cleaned up
        on completion.
    quiet : bool
        Suppress mdapy progress prints.
    """

    _STRAIN_DIRS = (1, 2, 3, 4, 5, 6)

    def __init__(
        self,
        system: System,
        pair_style: str,
        pair_coeff: str,
        elements: Sequence[str],
        temperature: float,
        pressure: float = 0.0,
        pressure_coupling: str = "iso",
        delta: float = 0.01,
        timestep: float = 1e-3,
        n_equil: int = 10000,
        n_run: int = 5000,
        nevery: int = 10,
        nrepeat: int = 10,
        ensemble: str = "isothermal",
        thermostat: str = "nose-hoover",
        thermostat_damp: Optional[float] = None,
        seed: int = 87287,
        relax_volume: bool = True,
        n_relax: int = 50000,
        pdamp: Optional[float] = None,
        n_workers: Optional[int] = None,
        n_threads_ref: int = 1,
        n_threads_def: int = 1,
        log_dir: Optional[Union[str, os.PathLike]] = None,
        lammps_cmdargs: Optional[List[str]] = None,
        work_dir: Optional[str] = None,
        quiet: bool = False,
    ) -> None:
        assert "element" in system.data.columns, (
            "system must contain element information."
        )
        if ensemble not in ("isothermal", "adiabatic"):
            raise ValueError(
                f"ensemble must be 'isothermal' or 'adiabatic', got {ensemble!r}"
            )
        if thermostat not in ("nose-hoover", "langevin"):
            raise ValueError(
                f"thermostat must be 'nose-hoover' or 'langevin', got {thermostat!r}"
            )
        if pressure_coupling not in ("iso", "aniso"):
            raise ValueError(
                f"pressure_coupling must be 'iso' or 'aniso', got {pressure_coupling!r}"
            )
        self.system = system
        self.pair_style = pair_style
        self.pair_coeff = pair_coeff
        self.elements = list(elements)
        self.temperature = float(temperature)
        self.pressure = float(pressure)
        self.pressure_coupling = pressure_coupling
        self.delta = float(delta)
        self.timestep = float(timestep)
        self.n_equil = int(n_equil)
        self.n_run = int(n_run)
        self.nevery = int(nevery)
        self.nrepeat = int(nrepeat)
        self.nfreq = self.nevery * self.nrepeat
        if self.n_run < self.nfreq:
            raise ValueError(
                f"n_run ({self.n_run}) must be >= nevery*nrepeat ({self.nfreq})."
            )
        self.ensemble = ensemble
        self.thermostat = thermostat
        self.thermostat_damp = (
            float(thermostat_damp) if thermostat_damp is not None
            else 100.0 * self.timestep
        )
        self.seed = int(seed)
        self.relax_volume = bool(relax_volume)
        self.n_relax = int(n_relax)
        self.pdamp = (
            float(pdamp) if pdamp is not None else 1000.0 * self.timestep
        )
        self.n_workers = (
            int(n_workers) if n_workers is not None
            else min(12, os.cpu_count() or 1)
        )
        self.n_threads_ref = int(n_threads_ref)
        self.n_threads_def = int(n_threads_def)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lammps_cmdargs = list(lammps_cmdargs) if lammps_cmdargs else []
        self.quiet = bool(quiet)

        self._user_work_dir = work_dir
        self._owns_work_dir = work_dir is None

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def run(self) -> MDElasticResult:
        """Execute the protocol for a single temperature.

        Reference NPT+NVT runs in the main process; the 12 deformations
        run on a ``ProcessPoolExecutor`` of size ``min(n_workers, 12)``.
        """
        if self._owns_work_dir:
            self._work_dir = Path(tempfile.mkdtemp(prefix="md_elastic_"))
        else:
            self._work_dir = Path(self._user_work_dir)
            self._work_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._write_initial_data()
            return self._execute_single()
        finally:
            if self._owns_work_dir and self._work_dir.exists():
                import shutil

                shutil.rmtree(self._work_dir, ignore_errors=True)

    @staticmethod
    def scan_parallel(
        system: System,
        temperatures: Sequence[float],
        work_dir: Union[str, os.PathLike],
        n_workers_ref: Optional[int] = None,
        n_workers_def: Optional[int] = None,
        n_workers: Optional[int] = None,        # alias: same for both phases
        log_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ) -> pl.DataFrame:
        """Run MDElastic at multiple temperatures with two-phase parallelism.

        Phase 1: each temperature's reference (NPT + NVT) runs in its own
        process; up to ``n_workers_ref`` simultaneously. Each process can
        use ``n_threads_ref`` OpenMP threads (passed via kwargs).
        Phase 2: each (T, direction, sign) deformation runs in its own
        process; up to ``n_workers_def`` simultaneously, each typically
        single-threaded (``n_threads_def`` via kwargs).

        Typical 16-core CPU + 4 temperatures::

            scan_parallel(
                ..., n_workers_ref=4, n_threads_ref=4,    # 4*4 = 16 cores
                     n_workers_def=12, n_threads_def=1,   # 12 procs single-thread
            )

        Parameters
        ----------
        system, temperatures : as named.
        work_dir : str or Path
            Persistent directory for LAMMPS data + per-T restart files.
        n_workers_ref : int, optional
            Number of reference processes. Default ``len(temperatures)``.
        n_workers_def : int, optional
            Number of deformation processes. Default
            ``min(12 * len(temperatures), os.cpu_count())``.
        n_workers : int, optional
            Convenience alias: sets both ``n_workers_ref`` and
            ``n_workers_def``. Ignored if either is set explicitly.
        log_dir : str or Path, optional
            Directory for per-stage LAMMPS log files. Each T's stage 1
            (NPT + ref NVT + averaging) goes to ``log_dir/ref_T{T}.log``;
            each deformation goes to ``log_dir/def_T{T}_d{dir}_{sign}.log``.
            Useful for checking convergence + debugging.
        **kwargs : forwarded to MDElastic.__init__ for each T (including
            ``n_threads_ref`` / ``n_threads_def`` / ``thermostat`` / etc.).

        Returns
        -------
        pl.DataFrame with one row per T, columns:
            T, V_eq, T_actual, C11, C12, C44 (cubic averages),
            K_VRH, G_VRH, E_VRH, nu_VRH, stable,
            stress_ref_max, c{ij} (full 6x6 entries).
        """
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

        if n_workers_ref is None:
            n_workers_ref = n_workers if n_workers is not None else len(temperatures)
        if n_workers_def is None:
            n_workers_def = (
                n_workers if n_workers is not None
                else min(12 * len(temperatures), os.cpu_count() or 1)
            )

        # Build a per-T MDElastic stub (gives us config dict construction);
        # we don't call .run() on it because we use the segment functions
        # directly across the shared pool.
        stubs = []
        data_paths = []
        for i, T in enumerate(temperatures):
            sub_dir = work_dir / f"T_{int(T)}"
            sub_dir.mkdir(exist_ok=True)
            data_path = sub_dir / "init.data"
            system.write_data(str(data_path), element_list=kwargs["elements"])
            data_paths.append(data_path)
            # Drop kwargs that we set explicitly so the user can still pass
            # `quiet=...` via kwargs without TypeError.
            kw = {k: v for k, v in kwargs.items()
                  if k not in ("temperature", "log_dir", "work_dir")}
            stub = MDElastic(
                system,
                temperature=float(T),
                log_dir=log_dir,
                work_dir=str(sub_dir),
                **kw,
            )
            # Manually fix up data path so segment functions see it.
            stub._data_path = data_path
            stub._work_dir = sub_dir
            stubs.append(stub)

        ref_tasks = [s._build_reference_args() for s in stubs]

        quiet = kwargs.get("quiet", False)
        n_threads_ref = kwargs.get("n_threads_ref", 1)
        n_threads_def = kwargs.get("n_threads_def", 1)
        if not quiet:
            print(
                f"[MDElastic.scan_parallel] phase 1: {len(ref_tasks)} ref runs "
                f"on {n_workers_ref} workers x {n_threads_ref} threads ..."
            )
        ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers_ref, mp_context=ctx) as ex:
            ref_results = list(ex.map(_run_reference_segment, ref_tasks))
        # Sort ref_results by temperature in case of out-of-order completion.
        ref_results.sort(key=lambda r: r["T"])

        deform_tasks = []
        for i, ref in enumerate(ref_results):
            for d in (1, 2, 3, 4, 5, 6):
                for sign in (+1, -1):
                    deform_tasks.append(stubs[i]._build_deform_args(d, sign, ref))

        if not quiet:
            print(
                f"[MDElastic.scan_parallel] phase 2: {len(deform_tasks)} deformation "
                f"runs on {n_workers_def} workers x {n_threads_def} threads ..."
            )
        with ProcessPoolExecutor(max_workers=n_workers_def, mp_context=ctx) as ex:
            def_results = list(ex.map(_run_deform_segment, deform_tasks))

        # Aggregate per-T C_ij.
        rows = []
        per_T_def = {}  # T -> list of deform-result dicts
        # Group deform_tasks by T (preserved by order: 12 per T in the order built).
        for i, T in enumerate(temperatures):
            chunk = def_results[i * 12:(i + 1) * 12]
            per_T_def[float(T)] = chunk

        for i, T in enumerate(temperatures):
            ref = ref_results[i]
            sigma_pos = np.zeros((6, 6))
            sigma_neg = np.zeros((6, 6))
            for r in per_T_def[float(T)]:
                d = r["direction"] - 1
                sb = np.array(r["sigma_bar"])
                if r["sign"] > 0:
                    sigma_pos[d] = sb
                else:
                    sigma_neg[d] = sb
            cij = np.zeros((6, 6))
            delta = stubs[i].delta
            for d in range(6):
                for k in range(6):
                    cij[k, d] = (
                        -(sigma_pos[d, k] - sigma_neg[d, k])
                        / (2.0 * delta) * _EAA_PER_BAR
                    )
            cij = 0.5 * (cij + cij.T)
            res = MDElasticResult(
                cij_voigt=cij,
                V_eq=ref["V_eq"],
                T_actual=ref["T_actual"],
                stress_ref=np.array(ref["sigma_ref_bar"]) * _EAA_PER_BAR,
                ensemble=stubs[i].ensemble,
                temperature=float(T),
            )
            c11, c12, c44 = res.cubic_average()
            v = res.vrh()
            row = {
                "T": float(T), "V_eq": res.V_eq, "T_actual": res.T_actual,
                "C11": c11, "C12": c12, "C44": c44,
                "K_VRH": v["K_H"], "G_VRH": v["G_H"], "E_VRH": v["E"],
                "nu_VRH": v["nu"], "stable": res.born_stable_cubic(),
                "stress_ref_max": float(np.abs(res.stress_ref).max()),
            }
            for ii in range(6):
                for jj in range(6):
                    row[f"c{ii + 1}{jj + 1}"] = res.cij_voigt[ii, jj]
            rows.append(row)
        return pl.DataFrame(rows)

    # ----------------------------------------------------------
    # Lattice constant + thermal expansion helpers
    # ----------------------------------------------------------

    @staticmethod
    def thermal_expansion(
        df: pl.DataFrame,
        n_unit_cells: int,
    ) -> pl.DataFrame:
        """Compute lattice constant and thermal expansion from a scan_parallel
        result DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            Output of ``MDElastic.scan_parallel``. Must contain ``T`` and
            ``V_eq`` columns.
        n_unit_cells : int
            Number of *conventional* unit cells in the simulation cell. For a
            3x3x3-of-conventional-FCC SQS (108 atoms with 4 atoms/unit cell),
            pass ``n_unit_cells=27``.

        Returns
        -------
        pl.DataFrame
            Original columns plus ``a`` (lattice constant in Å, equal to
            ``(V / n_unit_cells)**(1/3)``), ``alpha_V`` (volumetric thermal
            expansion in 1/K, centred finite-difference of ``V(T)``) and
            ``alpha_L`` (linear thermal expansion, ``alpha_V / 3`` under the
            cubic-symmetry assumption).
        """
        assert "T" in df.columns and "V_eq" in df.columns, (
            "thermal_expansion() needs T and V_eq columns "
            "(use MDElastic.scan_parallel output)."
        )
        out = df.sort("T")
        T = out["T"].to_numpy()
        V = out["V_eq"].to_numpy()
        n = len(T)
        alpha_V = np.zeros(n)
        for i in range(n):
            if n == 1:
                break
            if i == 0:
                alpha_V[i] = (V[1] - V[0]) / (T[1] - T[0]) / V[0]
            elif i == n - 1:
                alpha_V[i] = (V[i] - V[i - 1]) / (T[i] - T[i - 1]) / V[i]
            else:
                alpha_V[i] = (V[i + 1] - V[i - 1]) / (T[i + 1] - T[i - 1]) / V[i]
        a = (V / float(n_unit_cells)) ** (1.0 / 3.0)
        return out.with_columns(
            pl.Series("a", a),
            pl.Series("alpha_V", alpha_V),
            pl.Series("alpha_L", alpha_V / 3.0),
        )

    @staticmethod
    def parse_log(
        log_path: Union[str, os.PathLike],
        skip_fraction: float = 0.5,
    ) -> dict:
        """Parse a per-stage LAMMPS log written by MDElastic and return
        time-averages of the thermo columns over the *equilibrated* portion.

        Reads the ``thermo_style custom step temp press lx ly lz vol pe ke
        etotal pxx pyy pzz pxy pxz pyz`` block emitted by stage 1 / stage 2
        runs. The first ``skip_fraction`` of rows is discarded as
        equilibration; the rest is averaged.

        Parameters
        ----------
        log_path : path to ``ref_T{T}.log`` or similar.
        skip_fraction : float in [0, 1), default 0.5.

        Returns
        -------
        dict with keys ``T_avg``, ``press_avg``, ``a_avg`` (Å, from lx ly lz),
        ``V_avg`` (Å^3), ``n_samples``, plus ``stress_voigt`` 6-vector
        ``[pxx, pyy, pzz, pyz, pxz, pxy]`` in bar.
        """
        cols = ("step", "temp", "press", "lx", "ly", "lz", "vol",
                "pe", "ke", "etotal", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz")
        n_cols = len(cols)
        rows: List[List[float]] = []
        in_block = False
        with open(log_path) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("Step"):
                    in_block = True
                    continue
                if in_block:
                    toks = stripped.split()
                    if len(toks) == n_cols:
                        try:
                            rows.append([float(t) for t in toks])
                        except ValueError:
                            in_block = False
                    else:
                        in_block = False
        if not rows:
            raise ValueError(f"no thermo data parsed from {log_path}")
        arr = np.asarray(rows)
        n_skip = int(skip_fraction * len(arr))
        eq = arr[n_skip:]
        i = {name: idx for idx, name in enumerate(cols)}
        return dict(
            n_samples=len(eq),
            T_avg=float(eq[:, i["temp"]].mean()),
            press_avg=float(eq[:, i["press"]].mean()),
            V_avg=float(eq[:, i["vol"]].mean()),
            a_avg=float(((eq[:, i["lx"]].mean() *
                          eq[:, i["ly"]].mean() *
                          eq[:, i["lz"]].mean()) ** (1.0 / 3.0))),
            stress_voigt=np.array([
                eq[:, i["pxx"]].mean(),
                eq[:, i["pyy"]].mean(),
                eq[:, i["pzz"]].mean(),
                eq[:, i["pyz"]].mean(),
                eq[:, i["pxz"]].mean(),
                eq[:, i["pxy"]].mean(),
            ]),
        )

    # ----------------------------------------------------------
    # Internals
    # ----------------------------------------------------------

    def _write_initial_data(self) -> None:
        self._data_path = self._work_dir / "init.data"
        self.system.write_data(
            str(self._data_path), element_list=self.elements
        )

    def _build_reference_args(self) -> dict:
        ref_log = None
        if self.log_dir is not None:
            ref_log = str(self.log_dir / f"ref_T{int(self.temperature)}.log")
        return dict(
            lammps_cmdargs=self.lammps_cmdargs,
            stage_log=ref_log,
            n_threads=self.n_threads_ref,
            data_path=str(self._data_path),
            pair_style=self.pair_style,
            pair_coeff=self.pair_coeff,
            T=self.temperature, P=self.pressure,
            pressure_coupling=self.pressure_coupling,
            timestep=self.timestep,
            n_relax=self.n_relax,
            n_equil=self.n_equil, n_run=self.n_run,
            nevery=self.nevery, nrepeat=self.nrepeat, nfreq=self.nfreq,
            ensemble=self.ensemble, thermostat=self.thermostat,
            tdamp=self.thermostat_damp, pdamp=self.pdamp,
            seed=self.seed,
            relax_volume=self.relax_volume,
            restart_path=str(self._work_dir / "ref.restart"),
        )

    def _build_deform_args(self, direction: int, sign: int, ref: dict) -> dict:
        def_log = None
        if self.log_dir is not None:
            sgn = "+" if sign > 0 else "-"
            def_log = str(
                self.log_dir
                / f"def_T{int(self.temperature)}_d{int(direction)}_{sgn}.log"
            )
        return dict(
            lammps_cmdargs=self.lammps_cmdargs,
            stage_log=def_log,
            n_threads=self.n_threads_def,
            restart_path=ref["restart_path"],
            pair_style=self.pair_style,
            pair_coeff=self.pair_coeff,
            T=self.temperature,
            timestep=self.timestep,
            direction=int(direction),
            signed_delta=sign * self.delta,
            len0=ref["len0"],
            n_equil=self.n_equil, n_run=self.n_run,
            nevery=self.nevery, nrepeat=self.nrepeat, nfreq=self.nfreq,
            ensemble=self.ensemble, thermostat=self.thermostat,
            tdamp=self.thermostat_damp, seed=self.seed,
        )

    def _execute_single(self) -> MDElasticResult:
        # Phase 1: reference NPT+NVT (this process).
        if not self.quiet:
            print(
                f"[MDElastic] T={self.temperature} K reference "
                f"({self.thermostat}, ensemble={self.ensemble}) ..."
            )
        ref = _run_reference_segment(self._build_reference_args())

        # Phase 2: 12 deformations across n_workers processes.
        deform_args = [
            self._build_deform_args(d, sign, ref)
            for d in self._STRAIN_DIRS
            for sign in (+1, -1)
        ]
        n_w = min(self.n_workers, len(deform_args))
        if not self.quiet:
            print(
                f"[MDElastic] T={self.temperature} K: "
                f"{len(deform_args)} deformations on {n_w} workers ..."
            )
        if n_w > 1:
            ctx = _mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_w, mp_context=ctx) as ex:
                results = list(ex.map(_run_deform_segment, deform_args))
        else:
            results = [_run_deform_segment(a) for a in deform_args]

        # Aggregate.
        sigma_pos = np.zeros((6, 6))
        sigma_neg = np.zeros((6, 6))
        for r in results:
            d = r["direction"] - 1
            sb = np.array(r["sigma_bar"])
            if r["sign"] > 0:
                sigma_pos[d] = sb
            else:
                sigma_neg[d] = sb
        cij = np.zeros((6, 6))
        for d in range(6):
            for i in range(6):
                cij[i, d] = (
                    -(sigma_pos[d, i] - sigma_neg[d, i])
                    / (2.0 * self.delta) * _EAA_PER_BAR
                )
        cij = 0.5 * (cij + cij.T)

        return MDElasticResult(
            cij_voigt=cij,
            V_eq=ref["V_eq"],
            T_actual=ref["T_actual"],
            stress_ref=np.array(ref["sigma_ref_bar"]) * _EAA_PER_BAR,
            ensemble=self.ensemble,
            temperature=self.temperature,
        )
