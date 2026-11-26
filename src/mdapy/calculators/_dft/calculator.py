# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""MDAPY's plane-wave DFT as a calculator: energies, forces and virials of
periodic cells from norm-conserving, ultrasoft or PAW pseudopotentials on a
GPU."""

import inspect
import math
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mdapy import units
from mdapy.calculators._base import WITHOUT_VIRIAL, Calculator, CalculatorResult
from mdapy.calculators._dft import loader
from mdapy.core.system import System
from mdapy.errors import CalculationFailed, InvalidParameter, MissingProperty

HARTREE = 27.211386245988  # eV, CODATA 2018
BOHR = 0.529177210903  # Å, CODATA 2018

# The settings of scf.run with a default, by their units: those in hartree or
# bohr, with the factor to eV or Å, and those without, so that the record of a
# computation gives every one in MDAPY's units (a test holds every setting to
# one of the two)
SCF_UNITS = {
    "sigma": HARTREE,
    "kerker": 1.0 / BOHR,
    "energy_tolerance": HARTREE,
    "density_tolerance": HARTREE,
    "magnetic_kerker": 1.0 / BOHR,
    "first_tolerance": HARTREE,
}
# hartree: the residual of the bands of the first step of a computation from a
# ground state of a file, as the SCF takes it from the state of the one before
_STARTED_TOLERANCE = 1e-4

SCF_WITHOUT_UNITS = frozenset({
    "kgrid", "offset", "smearing", "beta", "minimum", "history", "metric_weight",
    "max_iterations", "davidson_iterations", "davidson_basis", "seed", "magnetic_beta",
    "stress", "correct_forces", "atomic_start", "kpoint_threads", "conditioning",
    "output_energies", "adaptive_damping", "first_held", "preconditioner", "precision",
    "kernels",
})  # fmt: skip

_SMEARINGS = (
    "fixed",
    "gaussian",
    "fermi-dirac",
    "methfessel-paxton",
    "marzari-vanderbilt",
)


def _find_pseudopotential(folder: Path, element: str) -> Path:
    """The UPF of an element in a folder: {element}.upf, or the one file
    {element}_*.upf or {element}.*.upf, the element and the extension in
    either case (GBRV writes ni_pbe_v1.4.uspp.F.UPF)."""
    for name in (f"{element}.upf", f"{element}.UPF"):
        if (folder / name).is_file():
            return folder / name
    low = element.lower()
    found = sorted(
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".upf"
        and p.name.lower().startswith((low + "_", low + "."))
    )
    if len(found) != 1:
        what = "none" if not found else ", ".join(p.name for p in found)
        raise InvalidParameter(
            f"one pseudopotential of {element} in {folder} is needed, found {what}; "
            "give them by element"
        )
    return found[0]


def _description(elements, species, mesh, setup) -> str:
    """The line of a computation in the log: its atoms, k-points, bands and
    grid."""
    formula = " ".join(f"{e} {elements.count(e)}" for e in species)
    text = (
        f"{len(elements)} atoms, {formula}; mesh {' '.join(map(str, mesh))}, "
        f"{setup['kpoints']} k-points, {setup['operations']} operations, "
        f"{setup['bands']} bands, grid {' '.join(map(str, setup['grid']))}"
    )
    if tuple(setup["smooth_grid"]) != tuple(setup["grid"]):
        text += f", bands on {' '.join(map(str, setup['smooth_grid']))}"
    if setup["spins"] == 2:
        text += ", collinear spins"
    if setup["gamma"]:
        text += ", real bands at Gamma"
    return text


class DFT(Calculator):
    """Energies, forces and virials from Kohn-Sham density functional theory
    in plane waves, computed by MDAPY on a GPU (the module ``_dft`` of the
    package ``mdapy-cuda12`` with NVIDIA's cuBLAS, cuFFT and cuSOLVER, or of
    ``mdapy-rocm6``), for periodic cells.

    Parameters
    ----------
    encut : float64
        The kinetic energy cutoff of the plane waves in eV, as VASP's ENCUT.
    ecutrho : float64, optional
        The cutoff of the density in eV, as QE's ecutrho. Default 4 encut,
        8 encut with ultrasoft or PAW pseudopotentials, whose augmentation charges
        need more.
    pseudopotentials : mapping or str or path-like, optional
        Norm-conserving, ultrasoft or PAW pseudopotentials in UPF, by element, or a folder
        that holds ``{element}.upf`` or one ``{element}_*.upf`` of each
        element. Default the environment variable ``MDAPY_PSEUDO_DIR``.
    kpoints : sequence of int32, optional
        A Gamma-centered mesh of k-points, three numbers. Default from
        ``kspacing``.
    kspacing : float64, optional
        The largest spacing of the mesh of k-points in 1/Å, 2 pi included,
        as VASP's KSPACING. Default 0.25.
    smearing : str, optional
        ``"gaussian"`` (default), ``"fermi-dirac"``, ``"methfessel-paxton"``
        (first order), ``"marzari-vanderbilt"``, or ``"fixed"`` for
        insulators.
    sigma : float64, optional
        The width of the smearing in eV. Default 0.1.
    magmoms : mapping or sequence of float64, optional
        The first magnetic moments in Bohr magnetons, by element or one per
        atom; with them the spins are collinear. Default none: no spin.
    hubbard : mapping of str to (str, float64), optional
        DFT+U of Dudarev: for an element the label of one of the
        pseudo-orbitals of its file (such as ``"3d"``) and U in eV, on
        Quantum ESPRESSO's atomic projectors (the pseudo-orbitals, neither
        orthogonalized nor normalized). U belongs to its projectors: values
        fitted with those of other codes, such as the spheres of VASP's PAW,
        do not carry over. Norm-conserving pseudopotentials only. Default
        none.
    functional : str, optional
        ``"pbe"``, ``"pbe_pw"`` (PBE on the PW92 correlation of the paper, as
        Quantum ESPRESSO's PBE), ``"pbesol"``, ``"pbesol_pw"`` (likewise),
        ``"lda_pw"`` or ``"lda_pw_mod"``. Default that of the
        pseudopotentials as Quantum ESPRESSO takes it: ``"pbe_pw"``,
        ``"pbesol_pw"`` or ``"lda_pw"``.
    tolerance : float64, optional
        The self-consistent field stops when the energy changes by less than
        this between two steps and the Hartree energy of the change of the
        density is below it, in eV per cell. Default 1e-6.
    max_iterations : int32, optional
        The steps of the self-consistent field before it fails. Default 100.
    mixing : float64, optional
        The share of the residual of the density taken at each step. Default
        0.7 without spin, as Quantum ESPRESSO, and 0.4 with it: of the
        magnetic cells of the robustness study, iron packed at random as a
        liquid converges in 69 steps where 0.7 does not in 100, iron with a
        vacancy in 23 against 45.
    magnetic_mixing : float64, optional
        With spin, the share of the residual of the magnetization taken at
        each step, as VASP's AMIX_MAG. Default 1.0: 1.6 overshoots, and fcc
        nickel of GBRV started at 0.6 Bohr magnetons ends without moment, 0.3
        eV above its ferromagnetic state.
    kerker : float64, optional
        The wave vector of Kerker's preconditioner of the mixing in 1/Å, 0
        for none. Default 1.5: 32 atoms of aluminium converge in half the
        steps, 64 of silicon in as many.
    preconditioner : str, optional
        Of the mixing of the total density: ``"kerker"``, or ``"ldos"``, from
        the local density of states at the Fermi level of each step (Herbst
        and Levitt, 2021), which screens metal and not vacuum or insulator in
        one cell: slabs and surfaces, and magnetic alloys, whose moments
        Kerker's lets wander or fall. Default ``"kerker"``.
    bands : int32, optional
        Bands per k-point and spin. Default as Quantum ESPRESSO: four or 10 %
        more than the occupied ones for fixed occupations, else the larger
        of 1.2 N/2 and N/2 + 4 for N electrons.
    seed : int32, optional
        Seed of the random first bands. Default 0.
    gpu : int32, optional
        The GPU of this process, by its index among those it sees. Default
        the first (CUDA_VISIBLE_DEVICES chooses which). Processes on a node
        of several GPUs, each computing its own configurations, take one each.
    restart : bool or str, optional
        Start each computation from the bands and the density of the last,
        for the steps of a relaxation or of molecular dynamics. True, or
        "atomic": the density of the last computation less its free atoms,
        plus the free atoms where they are now. "second_order": that
        difference extrapolated from the last three computations (Arias,
        Payne and Joannopoulos), which saves iterations along a smooth path
        such as molecular dynamics. Default False: each computation starts
        cold, as the labels of a training set should.
    symmetry : bool, optional
        Use the symmetry of the crystal, found by spglib, to keep the
        irreducible k-points alone and make the density, the forces and the
        stress symmetric. Default None: when spglib is installed.
    moment_radius : float64, optional
        With spin, the radius in Å of the spheres within which the magnetic
        moment of each atom is integrated. Default none: half the shortest
        distance between two atoms of the cell, their periodic images
        included, the same for all atoms, so that the spheres do not overlap.
    precision : str, optional
        ``"double"``, or ``"mixed"``: the first steps of the self-consistent
        field in single precision, the last in double, for GPUs of little
        double precision; the label is that of double precision.
        ``"auto"``: mixed on GPUs whose double
        precision runs at a small part of the rate of single (cards for games
        and workstations), double on the others (V100, A100, H100, GH200).
        Default ``"double"``.
    kernels : str, optional
        Whose kernels run on the GPU. ``"mdapy"``: those of MDAPY (the fast
        Fourier transforms, the dense linear algebra, the Hermitian
        eigensolver and the Cholesky factorization: all the DFT takes from
        them), without the libraries of the vendor, which only the FFTs of a
        grid given whose sizes are not products of 2, 3, 5 and 7 still take.
        ``"vendor"``: the GPU libraries of the vendor throughout (cuFFT,
        cuBLAS and cuSOLVER on NVIDIA GPUs, which
        ``pip install "mdapy[cuda12,dft]"`` installs; hipFFT, hipBLAS and
        hipSOLVER on AMD GPUs). The results agree to rounding. Default
        ``"mdapy"``.
    comm : mpi4py communicator, optional
        Processes that compute each configuration together, one GPU each,
        sharing its k-points and spins: every process calls ``compute`` with
        the same system and gets the whole result. Without ``gpu``, each takes
        the GPU of its rank among the processes of its node. Default none: one
        process.
    log : str or path-like or stream, optional
        Where every step of the self-consistent field of each computation is
        written, with the parameters the computation took: a file, written
        anew by the first computation of this calculator and added to by the
        next, or a stream such as ``sys.stdout``. Under MPI the first process
        writes it. Default none.
    start : str or path-like, optional
        A ground state that ``save`` wrote: the first computation starts from
        its density (and the occupations of PAW or DFT+U) in place of those
        of the free atoms. It must be of the same elements, cell, positions,
        pseudopotentials and grid. Default none.

    Attributes
    ----------
    cutoff : float64
        0: the calculator needs no neighbor list.
    last : dict
        Of the last computation: ``iterations``, ``operations`` (of the
        symmetry used, 1 without), ``davidson`` (the
        iterations of the eigensolver over all steps, k-points and spins),
        ``restarted`` (from the one before), ``fermi`` (eV),
        ``magnetization`` and ``absolute_magnetization`` (Bohr magnetons, with
        spin), ``magnetic_moments`` ((N,) float64 Bohr magnetons, with spin:
        the magnetization density, spin up minus spin down of the density on
        the grid, integrated within a sphere of ``moment_radius`` around each
        atom, in the order of the atoms) and ``moment_radius`` (Å, the radius
        used), with DFT+U ``hubbard_energy`` (eV) and ``hubbard_occupations``
        (the occupations of the orbitals of each Hubbard atom, (spins, 2l + 1,
        2l + 1) float64, a spin channel each), ``net_force`` (eV/Å, taken out
        of the forces), ``kpoints`` (the mesh), ``grid`` (the FFT grid),
        ``single_steps`` (the steps in single precision), ``gpu_memory``
        (MiB, float64, the largest at the end of the steps: ``in_use`` by the
        arrays of the DFT and ``reserved`` by its memory pool, of the whole
        process, the runs of its threads together, and ``device`` in use on
        the GPU, by the CUDA context, the libraries and other processes too),
        ``provenance``: what makes the labels of a set comparable, the SHA-256
        of each pseudopotential, the cutoff, the mesh, the smearing and its
        width, the functional, the spins, the grid, the bands and U,
        ``calculation`` (its number among the computations of this
        calculator, as the log counts them) and ``parameters``: every
        parameter the computation took, defaults and ``"auto"`` resolved, as
        the result's ``info`` keeps it under ``dft`` (see the guide).
    log : str or path-like or stream or None
        The log.
    start : str or path-like or None
        The ground state the first computation starts from.

    Notes
    -----
    The energy is the free energy of the smearing, VASP's TOTEN. The virial
    is the derivative at a fixed set of plane waves: it lacks the Pulay
    stress of an incomplete basis, which a higher cutoff makes small.

    Raises
    ------
    InvalidParameter
        If a parameter is out of range, if there is no GPU part with the DFT,
        or if the libraries of the DFT are missing.
    """

    implemented = frozenset({"energy", "forces", "virial"})
    device = "cuda"

    def __init__(
        self,
        encut: float,
        *,
        ecutrho: float | None = None,
        pseudopotentials: Mapping[str, str | os.PathLike]
        | str
        | os.PathLike
        | None = None,
        kpoints: Sequence[int] | None = None,
        kspacing: float = 0.25,
        smearing: str = "gaussian",
        sigma: float = 0.1,
        magmoms: Mapping[str, float] | Sequence[float] | None = None,
        hubbard: Mapping[str, tuple[str, float]] | None = None,
        functional: str | None = None,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        mixing: float | None = None,
        magnetic_mixing: float = 1.0,
        kerker: float = 1.5,
        preconditioner: str = "kerker",
        bands: int | None = None,
        seed: int = 0,
        gpu: int | None = None,
        restart: bool | str = False,
        symmetry: bool | None = None,
        moment_radius: float | None = None,
        precision: str = "double",
        kernels: str = "mdapy",
        comm: Any = None,
        log: Any = None,
        start: str | os.PathLike | None = None,
    ):
        from mdapy.calculators._dft import log as logs

        if not encut > 0:
            raise InvalidParameter("encut must be positive")
        if smearing not in _SMEARINGS:
            raise InvalidParameter(f"smearing must be one of {list(_SMEARINGS)}")
        if smearing != "fixed" and not sigma > 0:
            raise InvalidParameter("sigma must be positive")
        if (
            not tolerance > 0
            or max_iterations < 1
            or not (mixing is None or 0 < mixing <= 1)
            or kerker < 0
        ):
            raise InvalidParameter(
                "tolerance > 0, max_iterations >= 1, 0 < mixing <= 1 and kerker >= 0"
            )
        self.encut = float(encut)
        if ecutrho is not None and not ecutrho >= 4 * encut:
            raise InvalidParameter("ecutrho must be at least 4 encut")
        self.ecutrho = None if ecutrho is None else float(ecutrho)
        self.kpoints = None
        if kpoints is not None:
            mesh = np.asarray(kpoints)
            if (
                mesh.shape != (3,)
                or not np.issubdtype(mesh.dtype, np.integer)
                or (mesh < 1).any()
            ):
                raise InvalidParameter("kpoints must be three integers of at least 1")
            self.kpoints = tuple(int(k) for k in mesh)
        if not kspacing > 0:
            raise InvalidParameter("kspacing must be positive")
        self.kspacing = float(kspacing)
        if pseudopotentials is None:
            found = os.environ.get("MDAPY_PSEUDO_DIR")
            if found is None:
                raise InvalidParameter(
                    "give pseudopotentials, or set MDAPY_PSEUDO_DIR to their folder"
                )
            pseudopotentials = found
        if isinstance(pseudopotentials, Mapping):
            self.pseudopotentials = {
                str(k): Path(v) for k, v in pseudopotentials.items()
            }
            self.__folder = None
        else:
            self.pseudopotentials = {}
            self.__folder = Path(pseudopotentials)
            if not self.__folder.is_dir():
                raise InvalidParameter(f"{self.__folder} is not a folder")
        self.smearing = smearing
        self.sigma = float(sigma)
        self.magmoms = magmoms
        self.hubbard = {}
        for element, value in (hubbard or {}).items():
            if not (
                isinstance(value, (tuple, list))
                and len(value) == 2
                and isinstance(value[0], str)
                and float(value[1]) >= 0
            ):
                raise InvalidParameter(
                    "hubbard maps an element to the label of a pseudo-orbital and U >= 0 in eV"
                )
            self.hubbard[str(element)] = (value[0], float(value[1]))
        self.functional = functional
        self.tolerance = float(tolerance)
        self.symmetry = symmetry
        if moment_radius is not None and not moment_radius > 0:
            raise InvalidParameter("moment_radius must be positive")
        self.moment_radius = None if moment_radius is None else float(moment_radius)
        if precision not in ("double", "mixed", "auto"):
            raise InvalidParameter("precision must be 'double', 'mixed' or 'auto'")
        self.precision = precision
        if kernels not in ("mdapy", "vendor"):
            raise InvalidParameter("kernels must be 'mdapy' or 'vendor'")
        self.kernels = kernels
        self.max_iterations = int(max_iterations)
        self.mixing = None if mixing is None else float(mixing)
        self.kerker = float(kerker)
        if preconditioner not in ("kerker", "ldos"):
            raise InvalidParameter("preconditioner must be 'kerker' or 'ldos'")
        self.preconditioner = preconditioner
        if not magnetic_mixing > 0:
            raise InvalidParameter("magnetic_mixing must be positive")
        self.magnetic_mixing = float(magnetic_mixing)
        self.bands = bands
        self.seed = int(seed)
        if restart not in (False, True, "atomic", "second_order"):
            raise InvalidParameter(
                "restart must be False, True, 'atomic' or 'second_order'"
            )
        self.restart = restart
        if gpu is not None and int(gpu) < 0:
            raise InvalidParameter("gpu must be at least 0")
        self.gpu = None if gpu is None else int(gpu)
        if comm is not None and not all(
            callable(getattr(comm, name, None))
            for name in ("Get_size", "Get_rank", "Allreduce")
        ):
            raise InvalidParameter(
                f"comm must be a communicator of mpi4py, not {comm!r}"
            )
        self.comm = comm
        if comm is not None and gpu is None and comm.Get_size() > 1:
            from mpi4py import MPI

            self.gpu = comm.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
        logs.check_target(log)
        self.log = log
        from mdapy.calculators._dft import ground

        self.start = start
        self.__start = None if start is None else ground.read(start)
        self.__ground: dict[str, Any] | None = None
        loader.module()  # the GPU part with the DFT and its libraries, loaded
        self.__writer: Any = None
        self.__calculations = 0
        self.__state = None
        self.cutoff = 0.0
        self.elements = None
        self.last: dict[str, Any] = {}
        self.__read: dict[str, Any] = {}
        self.__files: dict[str, str] = {}

    def __pseudopotential(self, element: str):
        from mdapy.calculators._dft.upf import read_upf

        if element not in self.__read:
            if element in self.pseudopotentials:
                path = self.pseudopotentials[element]
            elif self.__folder is not None:
                path = _find_pseudopotential(self.__folder, element)
            else:
                raise InvalidParameter(f"no pseudopotential of {element}")
            pp = read_upf(path)
            if pp.element != element:
                raise InvalidParameter(
                    f"{path} is a pseudopotential of {pp.element}, not {element}"
                )
            self.__read[element] = pp
            self.__files[element] = Path(path).name
        return self.__read[element]

    def save(self, path: str | os.PathLike) -> None:
        """Writes the ground state of the last computation.

        Parameters
        ----------
        path : str or path-like
            The file, written whole or not at all, in NumPy's ``.npz``: the
            parameters, the structure, the energy, the forces, the density,
            the occupations of PAW or DFT+U, the k-points and the bands
            (see the guide). Under MPI the first process writes it.

        Raises
        ------
        InvalidParameter
            Before a computation, or if the file cannot be written.
        """
        from mdapy.calculators._dft import ground

        if self.__ground is None:
            raise InvalidParameter("no computation yet: nothing to save")
        if self.comm is None or self.comm.Get_rank() == 0:
            ground.write(path, self.__ground)

    def __log_writer(self):
        """The writer of the log, made at the first computation, on the GPU
        it runs on."""
        from mdapy.calculators._dft import log as logs

        if self.__writer is None:
            active = self.comm is None or self.comm.Get_rank() == 0
            if self.log is None or not active:
                self.__writer = logs.Writer(None, [])
            else:
                from mdapy.simulation._log import _machine

                _dft = loader.module()
                gpu = _dft.device_name()
                if _dft.api != "HIP":
                    major, minor = _dft.device_capability()
                    gpu += f", compute capability {major}.{minor}"
                ranks = 1 if self.comm is None else self.comm.Get_size()
                if ranks > 1:
                    gpu += f"; {ranks} processes, one GPU each"
                self.__writer = logs.Writer(
                    self.log, [f"engine: MDAPY on {gpu}", f"machine: {_machine()}"]
                )
        return self.__writer

    def __record(self, pseudos, species, mesh, ecutrho, mixing, radius, setup, options):
        """Every parameter a computation took, defaults and "auto" resolved:
        the arguments of DFT that give it again, the pseudopotentials by
        SHA-256, the settings of scf.run that DFT does not set, and the
        variables MDAPY_DFT_* of the environment."""
        from mdapy import __version__
        from mdapy.calculators._dft import scf
        from mdapy.calculators._dft import symmetry as symmetries

        magmoms = self.magmoms
        if isinstance(magmoms, Mapping):
            magmoms = {str(e): float(m) for e, m in magmoms.items()}
        elif magmoms is not None:
            magmoms = [float(m) for m in magmoms]
        arguments = {
            "encut": self.encut,
            "ecutrho": ecutrho,
            "kpoints": list(mesh),
            "kspacing": self.kspacing,
            "smearing": self.smearing,
            "sigma": self.sigma,
            "magmoms": magmoms,
            "hubbard": {e: [label, u] for e, (label, u) in self.hubbard.items()},
            "functional": self.functional or scf.functional_of(pseudos),
            "tolerance": self.tolerance,
            "max_iterations": self.max_iterations,
            "mixing": mixing,
            "magnetic_mixing": self.magnetic_mixing,
            "kerker": self.kerker,
            "preconditioner": self.preconditioner,
            "bands": int(setup["bands"]),
            "seed": self.seed,
            "restart": self.restart,
            "symmetry": symmetries.available()
            if self.symmetry is None
            else bool(self.symmetry),
            "moment_radius": None if radius is None else float(radius),
            "precision": "mixed" if setup["mixed"] else "double",
            "kernels": setup["kernels"],
        }
        internal = {}
        for name, p in inspect.signature(scf.run).parameters.items():
            # first_tolerance is set for a start from a file, a setting of the
            # field still
            if (name in options and name != "first_tolerance") or not isinstance(
                p.default, (bool, int, float, str, tuple)
            ):
                continue
            value = options.get(name, p.default)
            if isinstance(value, tuple):
                value = list(value)
            elif name in SCF_UNITS:
                value = value * SCF_UNITS[name]
            internal[name] = value
        _dft = loader.module()
        return {
            "mdapy": __version__,
            "arguments": arguments,
            "pseudopotentials": {
                e: {"file": self.__files[e], "sha256": pp.digest}
                for e, pp in zip(species, pseudos)
            },
            "internal": internal,
            "environment": {
                k: v
                for k, v in sorted(os.environ.items())
                if k.startswith("MDAPY_DFT_")
            },
            "gpu": _dft.device_name(),
            "processes": 1 if self.comm is None else int(self.comm.Get_size()),
        }

    def __restart_state(self):
        if not self.restart:
            return None
        if self.__state is None:
            self.__state = loader.module().State()
            self.__state.steps = 3 if self.restart == "second_order" else 1
        return self.__state

    def __mesh(self, cell: np.ndarray) -> tuple[int, int, int]:
        if self.kpoints is not None:
            return self.kpoints
        # |b_i| / kspacing up, b with 2 pi, in 1/Å
        b = 2 * math.pi * np.linalg.inv(cell).T
        return tuple(
            max(1, math.ceil(float(np.linalg.norm(v)) / self.kspacing - 1e-8))
            for v in b
        )

    def __moments(self, elements: list[str]) -> np.ndarray | None:
        if self.magmoms is None:
            return None
        if isinstance(self.magmoms, Mapping):
            return np.array([float(self.magmoms.get(e, 0.0)) for e in elements])
        moments = np.asarray(self.magmoms, dtype=float)
        if moments.shape != (len(elements),):
            raise InvalidParameter(
                f"magmoms holds {moments.size} moments for {len(elements)} atoms"
            )
        return moments

    def compute(
        self,
        system: System,
        properties: Iterable[str] = ("energy", "forces"),
        *,
        neighbors: Any = None,
    ) -> CalculatorResult:
        """Computes the energy, the forces and the virial of a periodic cell.

        Parameters
        ----------
        system : System
            Periodic along a, b and c, with an ``element`` column.
        properties : iterable of str, optional
            Of ``energy``, ``forces`` and ``virial``; the energy and the
            forces always come.
        neighbors : NeighborList, optional
            Not used.

        Returns
        -------
        CalculatorResult
            The energy in eV, the forces in eV/Å, and the virial in eV when
            asked.

        Raises
        ------
        InvalidParameter
            If the box is not periodic along a, b and c, or a
            pseudopotential is missing.
        MissingProperty
            If the system lacks the element column.
        CalculationFailed
            If the self-consistent field does not converge.
        """
        from mdapy.calculators._dft import ground, scf
        from mdapy.calculators._dft import moments as moments_of

        properties = tuple(properties)
        wanted = self._check_properties(properties)
        # the stress costs a pass over the bands; dynamics asks without it on
        # the steps that need no pressure
        stress = "virial" in wanted and WITHOUT_VIRIAL not in properties
        if self.gpu is not None:
            _dft = loader.module()
            count = _dft.device_count()
            _dft.set_device(
                self.gpu % count if self.comm is not None and count > 0 else self.gpu
            )
        if "element" not in system.table.columns:
            raise MissingProperty("DFT needs the element column")
        if not all(system.box.pbc):
            raise InvalidParameter("DFT needs a box periodic along a, b and c")
        elements = system.table["element"].cast(str).to_list()
        species = list(dict.fromkeys(elements))
        pseudos = [self.__pseudopotential(e) for e in species]
        kinds = [species.index(e) for e in elements]
        cell = system.box.matrix
        fractional = system.box.to_fractional(system["positions"]) % 1.0
        mesh = self.__mesh(cell)
        moments = self.__moments(elements)
        mixing = self.mixing
        if mixing is None:
            mixing = 0.7 if moments is None else 0.4
        radius = None
        if moments is not None:
            radius = self.moment_radius
            if radius is None:
                radius = moments_of.default_radius(cell, fractional @ cell)
        ecutrho = self.ecutrho
        if ecutrho is None:
            # ultrasoft augmentation charges need more of the density
            ecutrho = (8.0 if any(pp.ultrasoft for pp in pseudos) else 4.0) * self.encut
        options = dict(
            density_cutoff=ecutrho / HARTREE,
            kgrid=mesh, smearing=self.smearing, sigma=self.sigma / HARTREE,
            bands=self.bands, beta=mixing, kerker=self.kerker * BOHR, preconditioner=self.preconditioner,
            magnetic_beta=self.magnetic_mixing,
            energy_tolerance=self.tolerance / HARTREE,
            density_tolerance=self.tolerance / HARTREE,
            max_iterations=self.max_iterations, seed=self.seed,
            functional=self.functional, magmoms=moments,
            hubbard={species.index(e): (label, u / HARTREE)
                     for e, (label, u) in self.hubbard.items() if e in species} or None,
            stress=stress, state=self.__restart_state(),
            symmetry=self.symmetry, comm=self.comm, precision=self.precision,
            kernels=self.kernels,
        )  # fmt: skip
        started = self.__start
        if started is not None:
            ground.check(started, self.start, elements, cell, system["positions"],
                         {e: pp.digest for e, pp in zip(species, pseudos)})  # fmt: skip
            options["start_density"] = started["density"] * BOHR**3
            if moments is None:
                options["start_density"] = options["start_density"][0]
            if "becsum" in started:
                options["start_becsum"] = started["becsum"]
            # the bands of the first step solved as tightly as from the state
            # of the computation before: from the free atoms' orbitals in a
            # converged potential, loose bands would undo the density (Al4 3
            # steps instead of 12, PAW Si2 4 instead of 15)
            options["first_tolerance"] = _STARTED_TOLERANCE
            self.__start = None
        self.__calculations += 1
        number = self.__calculations
        writer = self.__log_writer()
        record: dict[str, Any] = {}

        def on_start(setup: dict) -> None:
            if started is not None:
                ground.check_grid(started, self.start, setup["spins"], setup["grid"])
            record.update(
                self.__record(pseudos, species, mesh, ecutrho, mixing, radius, setup, options)
            )  # fmt: skip
            if writer.on:
                writer.start(
                    number, _description(elements, species, mesh, setup), record
                )

        def on_step(
            step, energy, accuracy, magnetization, fermi, davidson, single, accepted
        ):
            writer.step(step, energy * HARTREE, accuracy * HARTREE, magnetization,
                        fermi * HARTREE, davidson, single, accepted)  # fmt: skip

        began = time.perf_counter()
        try:
            r = scf.run(
                cell / BOHR, fractional, kinds, pseudos, self.encut / HARTREE, **options,
                on_start=on_start, on_step=on_step if writer.on else None,
            )  # fmt: skip
        except BaseException as error:
            writer.end(f"stopped: {type(error).__name__}: {error}")
            raise
        elapsed = time.perf_counter() - began
        if not r["converged"]:
            steps = len(r["energies"])
            writer.end(f"not converged in {steps} steps, {elapsed:.2f} s")
            tail = ", ".join(
                f"{e * HARTREE:.8f} eV ({a * HARTREE:.1e})"
                for e, a in zip(r["energies"][-4:], r["accuracy"][-4:])
            )
            raise CalculationFailed(
                f"the self-consistent field did not converge in {steps} steps; the last "
                f"energies (and the Hartree energies of the change of the density): {tail}"
            )
        energy = float(r["energy"]) * HARTREE
        forces = r["forces"] * HARTREE / BOHR
        virial = None
        if stress:
            virial = -r["strain_derivative"] * HARTREE
        if writer.on:
            largest = float(np.linalg.norm(forces, axis=1).max())
            summary = (
                f"converged in {int(r['iterations'])} steps, {elapsed:.2f} s"
                + ("; from the last ground state" if r["restarted"] else "")
                + (
                    f"; from the ground state of {os.fspath(self.start)}"
                    if started is not None
                    else ""
                )
                + f"; energy {energy:.6f} eV, largest force {largest:.6f} eV/Å"
            )
            if virial is not None:
                pressure = np.trace(virial) / (3 * system.box.volume)
                summary += f", pressure {pressure * units.EV_PER_A3_TO_GPA:.4f} GPa"
            writer.end(summary)
        self.last = {
            "calculation": number,
            "iterations": int(r["iterations"]),
            "davidson": int(np.sum(r["davidson"])),
            "restarted": bool(r["restarted"]),
            "operations": int(r["operations"]),
            "fermi": float(r["fermi"]) * HARTREE,
            "net_force": r["net_force"] * HARTREE / BOHR,
            "kpoints": mesh,
            "grid": tuple(r["grid"]),
            "provenance": {
                "pseudopotentials": {e: pp.digest for e, pp in zip(species, pseudos)},
                "encut": self.encut,
                "ecutrho": self.ecutrho,
                "kpoints": mesh,
                "smearing": self.smearing,
                "sigma": self.sigma,
                "functional": self.functional or scf.functional_of(pseudos),
                "spins": int(r["spins"]),
                "grid": tuple(r["grid"]),
                "bands": int(r["bands"]),
                "hubbard": dict(self.hubbard),
                "kernels": r["kernels"],
            },
            "parameters": record,
            "single_steps": int(np.sum(r.get("single", 0))),
            "gpu_memory": {
                name: float(np.max(r["memory"][key])) / 2**20
                for name, key in (
                    ("in_use", "in_use"),
                    ("reserved", "reserved"),
                    ("device", "device_used"),
                )
            },
        }
        if "hubbard_energy" in r:
            self.last["hubbard_energy"] = float(r["hubbard_energy"]) * HARTREE
            self.last["hubbard_occupations"] = r["hubbard_occupations"]
        if r["spins"] == 2:
            self.last["magnetization"] = float(r["magnetization"])
            self.last["absolute_magnetization"] = float(r["absolute_magnetization"])
            # the density is (up, down) in electrons per bohr^3 on the grid of
            # the atoms' cell; the grid sums to r["magnetization"]
            m = (r["density"][0] - r["density"][1]) / BOHR**3
            self.last["magnetic_moments"] = moments_of.magnetic_moments(
                m, cell, fractional @ cell, radius
            )
            self.last["moment_radius"] = float(radius)
        spin_axis = (lambda a: a) if r["spins"] == 2 else (lambda a: a[None])
        self.__ground = {
            "parameters": record,
            "elements": elements,
            "cell": np.array(cell, dtype=float),
            "positions": np.array(system["positions"], dtype=float),
            "energy": energy,
            "fermi": self.last["fermi"],
            "forces": forces,
            "density": spin_axis(r["density"]) / BOHR**3,
            "kpoints": np.asarray(r["kpoints"], dtype=float),
            "weights": np.asarray(r["weights"], dtype=float),
            "eigenvalues": spin_axis(r["eigenvalues"]) * HARTREE,
            "occupations": spin_axis(r["occupations"]),
        }
        if "becsum" in r:
            self.__ground["becsum"] = np.asarray(r["becsum"], dtype=float)
        if "magnetic_moments" in self.last:
            self.__ground["magnetic_moments"] = self.last["magnetic_moments"]
        return CalculatorResult(
            energy=energy,
            forces=forces,
            virial=virial,
            volume=system.box.volume,
            info={"dft": record},
        )
