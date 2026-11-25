# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""MDAPY's plane-wave DFT as a calculator: energies, forces and virials of
periodic cells from norm-conserving, ultrasoft or PAW pseudopotentials on an
NVIDIA GPU."""

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mdapy.calculators._base import WITHOUT_VIRIAL, Calculator, CalculatorResult
from mdapy.core.system import System
from mdapy.errors import CalculationFailed, InvalidParameter, MissingProperty

HARTREE = 27.211386245988  # eV, CODATA 2018
BOHR = 0.529177210903  # Å, CODATA 2018

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


class DFT(Calculator):
    """Energies, forces and virials from Kohn-Sham density functional theory
    in plane waves, computed by MDAPY on an NVIDIA GPU (the package
    ``mdapy-dft-cuda12``), for periodic cells.

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
        Whose kernels run on the GPU. ``"mdapy"``: those of MDAPY where it
        has its own (today the fast Fourier transforms), the GPU libraries of
        the vendor for the rest. ``"vendor"``: the GPU libraries of the vendor
        throughout (cuFFT, cuBLAS and cuSOLVER on NVIDIA GPUs; hipFFT, hipBLAS
        and hipSOLVER on AMD GPUs). The results agree to rounding. Default
        ``"mdapy"``.
    comm : mpi4py communicator, optional
        Processes that compute each configuration together, one GPU each,
        sharing its k-points and spins: every process calls ``compute`` with
        the same system and gets the whole result. Without ``gpu``, each takes
        the GPU of its rank among the processes of its node. Default none: one
        process.

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
        ``single_steps`` (the steps in single precision) and
        ``provenance``: what makes the labels of a set comparable, the SHA-256
        of each pseudopotential, the cutoff, the mesh, the smearing and its
        width, the functional, the spins, the grid, the bands and U.

    Notes
    -----
    The energy is the free energy of the smearing, VASP's TOTEN. The virial
    is the derivative at a fixed set of plane waves: it lacks the Pulay
    stress of an incomplete basis, which a higher cutoff makes small.
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
    ):
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
        self.__state = None
        self.cutoff = 0.0
        self.elements = None
        self.last: dict[str, Any] = {}
        self.__read: dict[str, Any] = {}

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
        return self.__read[element]

    def __restart_state(self):
        if not self.restart:
            return None
        if self.__state is None:
            from mdapy_dft_cuda12 import _dft

            self.__state = _dft.State()
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
        from mdapy.calculators._dft import moments as moments_of
        from mdapy.calculators._dft import scf

        properties = tuple(properties)
        wanted = self._check_properties(properties)
        # the stress costs a pass over the bands; dynamics asks without it on
        # the steps that need no pressure
        stress = "virial" in wanted and WITHOUT_VIRIAL not in properties
        if self.gpu is not None:
            from mdapy_dft_cuda12 import _dft

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
        r = scf.run(
            cell / BOHR, fractional, kinds, pseudos, self.encut / HARTREE,
            density_cutoff=None if self.ecutrho is None else self.ecutrho / HARTREE,
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
        if not r["converged"]:
            steps = len(r["energies"])
            tail = ", ".join(
                f"{e * HARTREE:.8f} eV ({a * HARTREE:.1e})"
                for e, a in zip(r["energies"][-4:], r["accuracy"][-4:])
            )
            raise CalculationFailed(
                f"the self-consistent field did not converge in {steps} steps; the last "
                f"energies (and the Hartree energies of the change of the density): {tail}"
            )
        self.last = {
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
                "kernels": self.kernels,
            },
            "single_steps": int(np.sum(r.get("single", 0))),
        }
        if "hubbard_energy" in r:
            self.last["hubbard_energy"] = float(r["hubbard_energy"]) * HARTREE
            self.last["hubbard_occupations"] = r["hubbard_occupations"]
        if r["spins"] == 2:
            self.last["magnetization"] = float(r["magnetization"])
            self.last["absolute_magnetization"] = float(r["absolute_magnetization"])
            # the density is (up, down) in electrons per bohr^3 on the grid of
            # the atoms' cell; the grid sums to r["magnetization"]
            radius = self.moment_radius
            if radius is None:
                radius = moments_of.default_radius(cell, fractional @ cell)
            m = (r["density"][0] - r["density"][1]) / BOHR**3
            self.last["magnetic_moments"] = moments_of.magnetic_moments(
                m, cell, fractional @ cell, radius
            )
            self.last["moment_radius"] = float(radius)
        virial = None
        if stress:
            virial = -r["strain_derivative"] * HARTREE
        return CalculatorResult(
            energy=float(r["energy"]) * HARTREE,
            forces=r["forces"] * HARTREE / BOHR,
            virial=virial,
            volume=system.box.volume,
        )
