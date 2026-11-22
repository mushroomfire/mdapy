# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""MDAPY's plane-wave DFT as a calculator: energies, forces and virials of
periodic cells from norm-conserving pseudopotentials on an NVIDIA GPU."""

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mdapy.calculators._base import Calculator, CalculatorResult
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
    {element}_*.upf or {element}.*.upf, the extension in either case."""
    for name in (f"{element}.upf", f"{element}.UPF"):
        if (folder / name).is_file():
            return folder / name
    patterns = (f"{element}_*.[uU][pP][fF]", f"{element}.*.[uU][pP][fF]")
    found = sorted({p for pattern in patterns for p in folder.glob(pattern)})
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
        8 encut with ultrasoft pseudopotentials, whose augmentation charges
        need more.
    pseudopotentials : mapping or str or path-like, optional
        Norm-conserving pseudopotentials in UPF, by element, or a folder
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
    functional : str, optional
        ``"pbe"``, ``"pbe_pw"`` (PBE on the PW92 correlation of the paper, as
        Quantum ESPRESSO's PBE), ``"lda_pw"`` or ``"lda_pw_mod"``. Default
        that of the pseudopotentials.
    tolerance : float64, optional
        The self-consistent field stops when the energy changes by less than
        this between two steps and the Hartree energy of the change of the
        density is below it, in eV per cell. Default 1e-6.
    max_iterations : int32, optional
        The steps of the self-consistent field before it fails. Default 100.
    mixing : float64, optional
        The share of the residual of the density taken at each step. Default
        0.7.
    magnetic_mixing : float64, optional
        With spin, the share of the residual of the magnetization taken at
        each step, as VASP's AMIX_MAG. Default 1.6: 16 atoms of iron converge
        in 27 steps, against 62 with 0.4.
    kerker : float64, optional
        The wave vector of Kerker's preconditioner of the mixing in 1/Å, 0
        for none. Default 1.5: 32 atoms of aluminium converge in half the
        steps, 64 of silicon in as many.
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
    restart : bool, optional
        Start each computation from the bands and the density of the last,
        for the steps of a relaxation or of molecular dynamics. Default
        False: each computation starts cold, as the labels of a training set
        should.

    Attributes
    ----------
    cutoff : float64
        0: the calculator needs no neighbor list.
    last : dict
        Of the last computation: ``iterations``, ``restarted`` (from the
        one before), ``fermi`` (eV),
        ``magnetization`` and ``absolute_magnetization`` (Bohr magnetons, with
        spin), ``net_force`` (eV/Å, taken out of the forces), ``kpoints``
        (the mesh), ``grid`` (the FFT grid) and ``provenance``: what makes
        the labels of a set comparable, the SHA-256 of each pseudopotential,
        the cutoff, the mesh, the smearing and its width, the functional, the
        spins, the grid and the bands.

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
        functional: str | None = None,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        mixing: float = 0.7,
        magnetic_mixing: float = 1.6,
        kerker: float = 1.5,
        bands: int | None = None,
        seed: int = 0,
        gpu: int | None = None,
        restart: bool = False,
    ):
        if not encut > 0:
            raise InvalidParameter("encut must be positive")
        if smearing not in _SMEARINGS:
            raise InvalidParameter(f"smearing must be one of {list(_SMEARINGS)}")
        if smearing != "fixed" and not sigma > 0:
            raise InvalidParameter("sigma must be positive")
        if not tolerance > 0 or max_iterations < 1 or not 0 < mixing <= 1 or kerker < 0:
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
        self.functional = functional
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.mixing = float(mixing)
        self.kerker = float(kerker)
        if not magnetic_mixing > 0:
            raise InvalidParameter("magnetic_mixing must be positive")
        self.magnetic_mixing = float(magnetic_mixing)
        self.bands = bands
        self.seed = int(seed)
        self.restart = bool(restart)
        if gpu is not None and int(gpu) < 0:
            raise InvalidParameter("gpu must be at least 0")
        self.gpu = None if gpu is None else int(gpu)
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
        from mdapy.calculators._dft import scf

        wanted = self._check_properties(properties)
        if self.gpu is not None:
            from mdapy_dft_cuda12 import _dft

            _dft.set_device(self.gpu)
        if "element" not in system.table.columns:
            raise MissingProperty("DFT needs the element column")
        if not all(system.box.pbc):
            raise InvalidParameter("DFT needs a box periodic along a, b and c")
        elements = system.table["element"].cast(str).to_list()
        species = list(dict.fromkeys(elements))
        pseudos = [self.__pseudopotential(e) for e in species]
        if "virial" in wanted and any(pp.ultrasoft for pp in pseudos):
            raise InvalidParameter(
                "the virial of ultrasoft pseudopotentials is not computed yet"
            )
        kinds = [species.index(e) for e in elements]
        cell = system.box.matrix
        fractional = system.box.to_fractional(system["positions"]) % 1.0
        mesh = self.__mesh(cell)
        r = scf.run(
            cell / BOHR, fractional, kinds, pseudos, self.encut / HARTREE,
            density_cutoff=None if self.ecutrho is None else self.ecutrho / HARTREE,
            kgrid=mesh, smearing=self.smearing, sigma=self.sigma / HARTREE,
            bands=self.bands, beta=self.mixing, kerker=self.kerker * BOHR,
            magnetic_beta=self.magnetic_mixing,
            energy_tolerance=self.tolerance / HARTREE,
            density_tolerance=self.tolerance / HARTREE,
            max_iterations=self.max_iterations, seed=self.seed,
            functional=self.functional, magmoms=self.__moments(elements),
            stress="virial" in wanted, state=self.__restart_state(),
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
            "restarted": bool(r["restarted"]),
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
            },
        }
        if r["spins"] == 2:
            self.last["magnetization"] = float(r["magnetization"])
            self.last["absolute_magnetization"] = float(r["absolute_magnetization"])
        virial = None
        if "virial" in wanted:
            virial = -r["strain_derivative"] * HARTREE
        return CalculatorResult(
            energy=float(r["energy"]) * HARTREE,
            forces=r["forces"] * HARTREE / BOHR,
            virial=virial,
            volume=system.box.volume,
        )
