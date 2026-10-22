# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Any interaction of LAMMPS as a calculator, through the LAMMPS Python module."""

import atexit
import ctypes
import weakref
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from mdapy.calculators._base import Calculator, CalculatorResult
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.elements import SYMBOLS, atomic_masses
from mdapy.errors import InvalidParameter, MissingProperty

_ARGS = ["-screen", "none", "-log", "none", "-nocite"]

# Instances running Kokkos, of which one must shut Kokkos down at exit.
_kokkos: list[Any] = []


def _finalize_kokkos() -> None:
    if _kokkos:
        _kokkos[-1].finalize()


def _lammps(cmdargs: Sequence[str]) -> Any:
    try:
        from lammps import lammps
    except ImportError:
        raise ImportError(
            "the LAMMPS calculator needs the LAMMPS Python module: "
            "https://docs.lammps.org/Python_install.html"
        ) from None
    args = list(cmdargs)
    instance = lammps(cmdargs=_ARGS + args)
    if any(a in ("-k", "-kokkos") for a in args) and not _kokkos:
        atexit.register(_finalize_kokkos)
    if any(a in ("-k", "-kokkos") for a in args):
        _kokkos[:] = [instance]
    return instance


def _enclosing(box: Box, positions: np.ndarray, margin: float = 5.0) -> Box:
    """box stretched along non-periodic directions to hold the positions and
    margin Å more on each side."""
    if box.pbc.all():
        return box
    f = box.to_fractional(positions) if len(positions) else np.zeros((1, 3))
    matrix, origin = box.matrix.copy(), box.origin.copy()
    for d in np.flatnonzero(~box.pbc):
        pad = margin / box.thickness[d]
        low, high = f[:, d].min() - pad, f[:, d].max() + pad
        origin = origin + low * box.matrix[d]
        matrix[d] = box.matrix[d] * (high - low)
    return Box(matrix, origin=origin, pbc=box.pbc)


def _close(holder: list) -> None:
    while holder:
        holder.pop().close()


class LAMMPS(Calculator):
    """An interaction of LAMMPS, defined by its commands, as a calculator.

    Parameters
    ----------
    pair : str
        The LAMMPS commands of the interaction, such as ``pair_style`` and
        ``pair_coeff``, in metal units.
    elements : sequence of str
        The element of each LAMMPS atom type, from type 1.
    commands : str, optional
        LAMMPS commands run before ``pair``, such as ``newton on``.
    cmdargs : sequence of str, optional
        Command-line arguments of LAMMPS, such as ``["-k", "on", "t", "4",
        "-sf", "kk"]`` for Kokkos with four threads.

    Attributes
    ----------
    elements : tuple of str
        The element of each atom type.
    cutoff : float
        The largest cutoff of the interaction in Å, from LAMMPS.

    Raises
    ------
    InvalidParameter
        If LAMMPS rejects the commands.
    ImportError
        If the LAMMPS Python module is not installed.
    """

    def __init__(
        self,
        pair: str,
        elements: Sequence[str],
        *,
        commands: str = "",
        cmdargs: Sequence[str] = (),
    ):
        self.pair = pair
        self.elements = tuple(elements)
        unknown = [e for e in self.elements if e not in SYMBOLS]
        if not self.elements or unknown:
            raise InvalidParameter(f"elements must be chemical symbols, not {unknown}")
        self.commands = commands
        self.cmdargs = tuple(cmdargs)
        self._holder: list = []
        self._key: tuple | None = None
        self._box = Box(10.0)
        self._finalizer = weakref.finalize(self, _close, self._holder)
        # the cutoff, from a box without atoms
        lmp = _lammps(self.cmdargs)
        try:
            self._setup(lmp, Box(10.0))
            lmp.command("run 0 post no")
            self.cutoff = float(lmp.extract_global("neigh_cutmax"))
        except Exception as error:
            raise InvalidParameter(f"LAMMPS: {error}") from None
        finally:
            lmp.close()

    def _setup(self, lmp: Any, box: Box) -> None:
        lammps_box = box.to_lammps()
        pbc = box.pbc[[0, 2, 1]] if np.linalg.det(box.matrix) < 0 else box.pbc
        boundary = " ".join("p" if p else "f" for p in pbc)
        lx, ly, lz = (float(v) for v in lammps_box.hi - lammps_box.lo)
        xy, xz, yz = (float(v) for v in lammps_box.tilt)
        masses = [float(m) for m in atomic_masses(self.elements)]
        lmp.commands_string(
            f"""units metal
atom_style atomic
atom_modify map array sort 0 0.0
boundary {boundary}
region box prism 0 {lx!r} 0 {ly!r} 0 {lz!r} {xy!r} {xz!r} {yz!r}
create_box {len(self.elements)} box
neighbor 0.0 bin
neigh_modify every 1 delay 0 check yes
"""
            + "".join(f"mass {k + 1} {m!r}\n" for k, m in enumerate(masses))
            + self.commands
            + "\n"
            + self.pair
            + "\ncompute mdapy_pe all pe/atom"
            + "\ncompute mdapy_stress all stress/atom NULL virial"
            + "\ncompute mdapy_virial all pressure NULL virial"
            + "\nthermo_style custom step pe vol"
        )

    def _frame(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # positions in the LAMMPS box, and their fractional coordinates
        box = self._box
        wrapped, _ = box.wrap(positions)
        lammps_box = box.to_lammps()
        return wrapped @ lammps_box.rotation - lammps_box.lo, box.to_fractional(wrapped)

    def _inside(self, positions: np.ndarray) -> bool:
        # whether the atoms are well inside the box along fixed boundaries
        _, f = self._frame(positions)
        free = ~self._box.pbc
        return bool(((f[:, free] > 1e-6) & (f[:, free] < 1 - 1e-6)).all())

    def close(self) -> None:
        """Closes the LAMMPS instance; the next computation starts a new one."""
        _close(self._holder)
        self._key = None

    def compute(
        self,
        system: System,
        properties: Iterable[str] = ("energy", "forces"),
        *,
        neighbors: Any = None,
    ) -> CalculatorResult:
        """Computes properties of a system with LAMMPS.

        Parameters
        ----------
        system : System
            The atoms, with an ``element`` column.
        properties : iterable of str, optional
            The properties to compute; all are computed.
        neighbors : NeighborList, optional
            Not used: LAMMPS searches its own neighbors.

        Returns
        -------
        CalculatorResult
            Every property.
        """
        self._check_properties(properties)
        if "element" not in system.table.columns:
            raise MissingProperty("the LAMMPS calculator needs the element column")
        numbers = system.table["element"].to_physical().to_numpy()
        type_of = np.zeros(len(SYMBOLS), dtype=np.int32)
        for k, element in reversed(list(enumerate(self.elements))):
            type_of[SYMBOLS.index(element)] = k + 1
        types = type_of[numbers]
        if (types == 0).any():
            missing = sorted({SYMBOLS[z] for z in numbers[types == 0]})
            raise InvalidParameter(
                f"no atom type has the elements {missing}; the types are {list(self.elements)}"
            )
        n = len(system)
        if n == 0:
            return CalculatorResult(
                energy=0.0,
                atomic_energies=np.zeros(0),
                forces=np.zeros((0, 3)),
                virial=np.zeros((3, 3)),
                atomic_virials=np.zeros((0, 3, 3)),
            )
        key = (n, types.tobytes(), system.box)
        if key != self._key or not self._inside(system.positions):
            self.close()
            self._box = _enclosing(system.box, system.positions)
            lmp = _lammps(self.cmdargs)
            self._holder.append(lmp)
            self._setup(lmp, self._box)
            x, _ = self._frame(system.positions)
            lmp.create_atoms(
                n, list(range(1, n + 1)), types.tolist(), x.ravel().tolist()
            )
            self._key = key
        else:
            lmp = self._holder[0]
            x, _ = self._frame(system.positions)
            lmp.scatter_atoms("x", 1, 3, (ctypes.c_double * (3 * n))(*x.ravel()))
        rotation = self._box.to_lammps().rotation
        lmp.command("run 0 post no")
        nktv2p = lmp.extract_global("nktv2p")
        energies = np.ctypeslib.as_array(lmp.gather("c_mdapy_pe", 1, 1)).copy()
        forces = np.ctypeslib.as_array(lmp.gather("f", 1, 3)).reshape(n, 3) @ rotation.T
        stress = np.ctypeslib.as_array(lmp.gather("c_mdapy_stress", 1, 6)).reshape(n, 6)
        s = -stress / nktv2p  # xx yy zz xy xz yz of the virial of each atom
        atomic = np.stack([s[:, [0, 3, 4]], s[:, [3, 1, 5]], s[:, [4, 5, 2]]], axis=1)
        p = np.array(lmp.numpy.extract_compute("mdapy_virial", 0, 1))
        virial = np.array([[p[0], p[3], p[4]], [p[3], p[1], p[5]], [p[4], p[5], p[2]]])
        virial *= lmp.get_thermo("vol") / nktv2p
        return CalculatorResult(
            energy=float(lmp.get_thermo("pe")),
            atomic_energies=energies,
            forces=forces,
            virial=rotation @ virial @ rotation.T,
            atomic_virials=rotation @ atomic @ rotation.T,
        )

    def __repr__(self) -> str:
        return f"LAMMPS({self.pair!r}, elements={self.elements}, cutoff={self.cutoff})"
