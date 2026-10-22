# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""VASP as a calculator: single points run by a command, as on a cluster."""

import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mdapy.calculators._base import Calculator, CalculatorResult
from mdapy.core.system import System
from mdapy.errors import CalculationFailed, InvalidParameter, MissingProperty

# INCAR tags that make every run a single point with the stress
_SINGLE_POINT = {"NSW": 0, "IBRION": -1, "ISIF": 2}


def _incar_value(value: Any) -> str:
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(_incar_value(v) for v in value)
    return str(value)


class VASP(Calculator):
    """Energies, forces and virials from VASP: each call writes POSCAR,
    INCAR, KPOINTS and POTCAR, runs VASP with a command and reads its
    OUTCAR.

    Parameters
    ----------
    incar : mapping, optional
        INCAR tags, such as ``{"ENCUT": 520, "EDIFF": 1e-6, "ISMEAR": 1,
        "SIGMA": 0.1}``. NSW, IBRION and ISIF are set for a single point
        with the stress and may not be given.
    kpoints : sequence of int, optional
        A Gamma-centered mesh of k-points, three numbers. Default none: VASP
        then takes KSPACING from the INCAR.
    potcars : str or path-like, optional
        The folder of the POTCARs, one folder per element or setup, such as
        ``potpaw_PBE``. Default the environment variable ``VASP_PP_PATH``,
        or its ``potpaw_PBE`` folder.
    setups : mapping, optional
        The POTCAR of an element when not its symbol, such as
        ``{"Ti": "Ti_sv"}``.
    command : str, optional
        The shell command that runs VASP in the directory, as in a job
        script, such as
        ``"mpirun -np 128 vasp_std"`` or ``"srun vasp_std"``. Default the
        environment variable ``MDAPY_VASP_COMMAND``, else
        ``ASE_VASP_COMMAND``, else ``vasp_std``.
    directory : str or path-like, optional
        Where VASP runs, made when missing. Default ``vasp``.
    restart : bool, optional
        Keep WAVECAR and CHGCAR between calls, so that VASP starts from
        the last wave functions, as in a minimization. Default True.

    Attributes
    ----------
    cutoff : float
        0: VASP needs no neighbor list.
    """

    def __init__(
        self,
        incar: Mapping[str, Any] | None = None,
        *,
        kpoints: Sequence[int] | None = None,
        potcars: str | os.PathLike | None = None,
        setups: Mapping[str, str] | None = None,
        command: str | None = None,
        directory: str | os.PathLike = "vasp",
        restart: bool = True,
    ):
        self.incar = {str(k).upper(): v for k, v in dict(incar or {}).items()}
        fixed = sorted(set(self.incar) & set(_SINGLE_POINT))
        if fixed:
            raise InvalidParameter(
                f"{', '.join(fixed)} make each call a single point and may not be given"
            )
        self.kpoints = None
        if kpoints is not None:
            mesh = np.asarray(kpoints)
            if mesh.shape != (3,) or not np.issubdtype(mesh.dtype, np.integer):
                raise InvalidParameter("kpoints must be three integers")
            if (mesh < 1).any():
                raise InvalidParameter("kpoints must be at least 1")
            self.kpoints = tuple(int(k) for k in mesh)
        if potcars is None:
            found = os.environ.get("VASP_PP_PATH")
            potcars = None if found is None else Path(found)
            if potcars is not None and (potcars / "potpaw_PBE").is_dir():
                potcars = potcars / "potpaw_PBE"
        self.potcars = None if potcars is None else Path(potcars)
        self.setups = dict(setups or {})
        self.command = (
            command
            or os.environ.get("MDAPY_VASP_COMMAND")
            or os.environ.get("ASE_VASP_COMMAND")
            or "vasp_std"
        )
        self.directory = Path(directory).absolute()
        self.restart = bool(restart)
        self.cutoff = 0.0
        self.elements = None
        self.implemented = frozenset({"energy", "forces", "virial"})

    def compute(
        self,
        system: System,
        properties: Iterable[str] = ("energy", "forces"),
        *,
        neighbors: Any = None,
    ) -> CalculatorResult:
        """Runs VASP on a system.

        Parameters
        ----------
        system : System
            The atoms, with an ``element`` column.
        properties : iterable of str, optional
            The properties to return: energy (the free energy TOTEN),
            forces and virial; VASP computes them all.
        neighbors : NeighborList, optional
            Not used.

        Returns
        -------
        CalculatorResult
            The energy, forces and virial of the system, its atoms in their
            order.

        Raises
        ------
        CalculationFailed
            If VASP stops with an error or leaves no converged result.
        """
        from mdapy.io import write
        from mdapy.mlip import read_outcar

        self._check_properties(properties)
        if "element" not in system.table.columns:
            raise MissingProperty("VASP needs the element column")
        elements = system.table["element"].cast(str).to_list()
        # VASP holds the atoms of an element together, in order of appearance
        species = list(dict.fromkeys(elements))
        rank = {e: k for k, e in enumerate(species)}
        order = np.argsort([rank[e] for e in elements], kind="stable")
        here = self.directory
        here.mkdir(parents=True, exist_ok=True)
        write(here / "POSCAR", system.select(order), format="poscar")
        self._write_incar(here / "INCAR")
        if self.kpoints is not None:
            mesh = " ".join(str(k) for k in self.kpoints)
            (here / "KPOINTS").write_text(f"mdapy\n0\nGamma\n{mesh}\n0 0 0\n")
        else:
            (here / "KPOINTS").unlink(missing_ok=True)
        self._write_potcar(here / "POTCAR", species)
        if not self.restart:
            for name in ("WAVECAR", "CHGCAR"):
                (here / name).unlink(missing_ok=True)
        (here / "OUTCAR").unlink(missing_ok=True)
        # through the shell, as in a job script
        run = subprocess.run(
            self.command, shell=True, cwd=here, capture_output=True, text=True
        )
        frames = read_outcar(here / "OUTCAR") if (here / "OUTCAR").exists() else []
        if run.returncode != 0 or not frames:
            told = (run.stdout + run.stderr).strip().splitlines()[-20:]
            reason = f"exit status {run.returncode}" if run.returncode else "no result"
            raise CalculationFailed(f"VASP in {here} gave {reason}: " + "\n".join(told))
        frame = frames[-1]
        forces = np.empty((len(system), 3))
        forces[order] = frame.table["forces"].to_numpy()
        return CalculatorResult(
            energy=float(frame.info["energy"]),
            forces=forces,
            virial=np.asarray(frame.info.get("virial"), dtype=float)
            if "virial" in frame.info
            else None,
        )

    def _write_incar(self, path: Path) -> None:
        tags = {**self.incar, **_SINGLE_POINT}
        path.write_text(
            "".join(f"{tag} = {_incar_value(value)}\n" for tag, value in tags.items())
        )

    def _write_potcar(self, path: Path, species: list[str]) -> None:
        if self.potcars is None:
            raise InvalidParameter(
                "no POTCARs: give potcars= or set the environment variable VASP_PP_PATH"
            )
        parts = []
        for element in species:
            setup = self.setups.get(element, element)
            source = self.potcars / setup / "POTCAR"
            if not source.is_file():
                raise InvalidParameter(f"no POTCAR for {element} at {source}")
            parts.append(source.read_text())
        path.write_text("".join(parts))

    def __repr__(self) -> str:
        return f"VASP(command={self.command!r}, directory={str(self.directory)!r})"
