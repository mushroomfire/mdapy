# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Training sets of machine-learned potentials: the predictions of a model
against the labels of the frames, the spread of a committee of models,
how far frames lie from others, and frames that look wrong."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.elements import covalent_radii
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList

# GPUMD's components of a virial, and its mark of a missing reference
_GPUMD_VIRIAL = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))
_GPUMD_MISSING = -1e5


@dataclass(frozen=True)
class Predictions(Result):
    """The energies, forces and virials of frames: the labels and the
    predictions of a model.

    Attributes
    ----------
    atoms : numpy.ndarray
        (F,) int64 the number of atoms of each frame.
    energy : numpy.ndarray
        (F, 2) the label and the prediction of each frame, in eV per atom;
        NaN for a frame without a label.
    forces : numpy.ndarray
        (A, 2, 3) the label and the prediction of the force on each atom of
        all frames one after the other, in eV/Å; NaN without labels.
    virial : numpy.ndarray
        (F, 2, 3, 3) the label and the prediction of the virial of each
        frame, in eV per atom; NaN where either is missing.
    """

    atoms: np.ndarray
    energy: np.ndarray
    forces: np.ndarray
    virial: np.ndarray

    @property
    def frame_of_atom(self) -> np.ndarray:
        """(A,) the frame of each atom of ``forces``."""
        return np.repeat(np.arange(len(self.atoms)), self.atoms)

    def table(self) -> pl.DataFrame:
        """The errors of each frame: of the energy per atom in eV, the root
        mean square and the largest error of the force components in eV/Å,
        and the root mean square error of the virial per atom in eV.

        Returns
        -------
        polars.DataFrame
            One row per frame: frame, atoms, energy, energy_error,
            force_rmse, force_max_error, virial_rmse.
        """
        frame = self.frame_of_atom
        count = len(self.atoms)
        error = (self.forces[:, 1] - self.forces[:, 0]) ** 2
        with np.errstate(invalid="ignore"):
            sums = np.bincount(frame, error.sum(axis=1), minlength=count)
            largest = np.full(count, np.nan)
            np.fmax.at(largest, frame, np.sqrt(error).max(axis=1))
            force_rmse = np.sqrt(sums / (3 * np.maximum(self.atoms, 1)))
            virial = np.sqrt(
                ((self.virial[:, 1] - self.virial[:, 0]) ** 2).mean(axis=(1, 2))
            )
        return pl.DataFrame(
            {
                "frame": np.arange(count),
                "atoms": self.atoms,
                "energy": self.energy[:, 0],
                "energy_error": self.energy[:, 1] - self.energy[:, 0],
                "force_rmse": force_rmse,
                "force_max_error": largest,
                "virial_rmse": virial,
            }
        )

    def rmse(self) -> dict[str, float]:
        """The root mean square errors over all frames with labels.

        Returns
        -------
        dict
            ``energy`` in eV per atom, ``forces`` in eV/Å over the force
            components, ``virial`` in eV per atom over the components.
        """
        energy = self.energy[:, 1] - self.energy[:, 0]
        forces = (self.forces[:, 1] - self.forces[:, 0]).ravel()
        virial = (self.virial[:, 1] - self.virial[:, 0]).ravel()
        return {
            name: float(np.sqrt(np.mean(v[np.isfinite(v)] ** 2)))
            if np.isfinite(v).any()
            else float("nan")
            for name, v in (("energy", energy), ("forces", forces), ("virial", virial))
        }

    def worst(self, count: int, by: str = "force_rmse") -> np.ndarray:
        """The frames of the largest errors.

        Parameters
        ----------
        count : int
            How many.
        by : str, optional
            A column of :meth:`table`: energy_error (by its size),
            force_rmse, force_max_error or virial_rmse. Default force_rmse.

        Returns
        -------
        numpy.ndarray
            int64 the frames, the largest error first.
        """
        columns = ("energy_error", "force_rmse", "force_max_error", "virial_rmse")
        if by not in columns:
            raise InvalidParameter(f"by must be one of {columns}, not {by!r}")
        values = np.abs(self.table()[by].to_numpy())
        values = np.where(np.isfinite(values), values, -np.inf)
        order = np.argsort(-values, kind="stable")
        return order[: max(int(count), 0)]


def _labels(frame: System) -> tuple[float, np.ndarray, np.ndarray]:
    """The energy, forces and virial of a frame per atom, NaN where it has
    none."""
    n = len(frame)
    energy = frame.info.get("energy")
    energy = float(energy) / n if energy is not None and n else np.nan
    forces = (
        np.asarray(frame["forces"], dtype=np.float64)
        if "forces" in frame.table.columns
        else np.full((n, 3), np.nan)
    )
    virial = frame.info.get("virial")
    virial = (
        np.asarray(virial, dtype=np.float64) / n
        if virial is not None and n
        else np.full((3, 3), np.nan)
    )
    return energy, forces, virial


def predict(frames: Sequence[System], calculator: Any) -> Predictions:
    """The predictions of a model against the labels of frames, such as
    those of a training set.

    Parameters
    ----------
    frames : sequence of System
        The frames, their labels the info ``energy`` (eV) and ``virial``
        (eV) and the column ``forces`` (eV/Å), as extended XYZ files of
        training sets hold them.
    calculator : Calculator
        The model: any mdapy calculator, such as NEP, or an ASE calculator
        through :func:`mdapy.interop.from_ase_calculator`.

    Returns
    -------
    Predictions
        The labels and the predictions, per atom.
    """
    wanted = ["energy", "forces"]
    if "virial" in calculator.implemented:
        wanted.append("virial")
    atoms, energy, forces, virial = [], [], [], []
    for frame in frames:
        n = len(frame)
        label_energy, label_forces, label_virial = _labels(frame)
        result = calculator.compute(frame, wanted)
        predicted_virial = (
            np.asarray(result.virial) / n
            if result.virial is not None and n
            else np.full((3, 3), np.nan)
        )
        atoms.append(n)
        energy.append([label_energy, result.energy / n if n else np.nan])
        forces.append(np.stack([label_forces, np.asarray(result.forces)], axis=1))
        virial.append([label_virial, predicted_virial])
    if not atoms:
        raise InvalidParameter("there are no frames")
    return Predictions(
        atoms=np.array(atoms, dtype=np.int64),
        energy=np.array(energy, dtype=np.float64),
        forces=np.concatenate(forces).reshape(-1, 2, 3),
        virial=np.array(virial, dtype=np.float64),
    )


def _atom_counts(path: Path) -> np.ndarray:
    """The number of atoms of each frame of an XYZ file."""
    counts = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            n = int(line)
            counts.append(n)
            for _ in range(n + 1):
                next(file)
    return np.array(counts, dtype=np.int64)


def read_gpumd(
    folder: str | Path, kind: str = "train", *, xyz: str | Path | None = None
) -> Predictions:
    """The predictions of a NEP model against its training or test set,
    from the files GPUMD or torchnep write: energy, force and virial.

    Parameters
    ----------
    folder : str or Path
        The folder of ``energy_train.out``, ``force_train.out`` and
        ``virial_train.out`` (or ``_test``).
    kind : str, optional
        ``"train"`` or ``"test"``. Default train.
    xyz : str or Path, optional
        The extended XYZ file of the frames, whose frames give the number of
        atoms of each. Default ``train.xyz`` (or ``test.xyz``) in the folder.

    Returns
    -------
    Predictions
        The labels and the predictions, the frames in the order of the
        training set; virials NaN where it has none.
    """
    if kind not in ("train", "test"):
        raise InvalidParameter(f"kind must be train or test, not {kind!r}")
    folder = Path(folder)
    try:
        energy = np.loadtxt(folder / f"energy_{kind}.out", ndmin=2)
        forces = np.loadtxt(folder / f"force_{kind}.out", ndmin=2)
        written = np.loadtxt(folder / f"virial_{kind}.out", ndmin=2)
        xyz = Path(xyz) if xyz is not None else folder / f"{kind}.xyz"
        atoms = _atom_counts(xyz)
    except (OSError, ValueError, StopIteration) as error:
        raise InvalidParameter(
            f"cannot read the {kind} output of GPUMD: {error}"
        ) from None
    if energy.shape[1] != 2 or forces.shape[1] != 6 or written.shape[1] != 12:
        raise InvalidParameter("the GPUMD files do not have 2, 6 and 12 columns")
    if not len(written) == len(energy) == len(atoms) or atoms.sum() != len(forces):
        raise InvalidParameter(f"the GPUMD files do not hold the frames of {xyz.name}")
    virial = np.full((len(energy), 2, 3, 3), np.nan)
    for k, (i, j) in enumerate(_GPUMD_VIRIAL):
        for side, column in ((1, k), (0, 6 + k)):
            value = written[:, column]
            if side == 0:
                value = np.where(value > _GPUMD_MISSING, value, np.nan)
            virial[:, side, i, j] = virial[:, side, j, i] = value
    return Predictions(
        atoms=atoms,
        energy=energy[:, ::-1].copy(),
        forces=np.stack([forces[:, 3:], forces[:, :3]], axis=1),
        virial=virial,
    )


@dataclass(frozen=True)
class CommitteeResult(Result):
    """How much the predictions of several models differ.

    Attributes
    ----------
    atoms : numpy.ndarray
        (F,) int64 the number of atoms of each frame.
    energy : numpy.ndarray
        (F,) the standard deviation of the energy per atom of each frame in
        eV.
    forces : numpy.ndarray
        (A,) of each atom of all frames one after the other, the root mean
        square over the models of the distance of its force from their mean
        force, in eV/Å.
    max_force : numpy.ndarray
        (F,) the largest of ``forces`` in each frame.
    """

    atoms: np.ndarray
    energy: np.ndarray
    forces: np.ndarray
    max_force: np.ndarray


def committee(frames: Sequence[System], calculators: Sequence[Any]) -> CommitteeResult:
    """The spread of the predictions of several models of frames: large
    where they disagree, as where none of them was trained.

    Parameters
    ----------
    frames : sequence of System
        The frames.
    calculators : sequence of Calculator
        Two or more models, such as NEP models trained with other seeds,
        or models of other kinds.

    Returns
    -------
    CommitteeResult
        The spread of the energies and the forces.
    """
    if len(calculators) < 2:
        raise InvalidParameter("a committee needs two or more models")
    atoms, energy, forces, largest = [], [], [], []
    for frame in frames:
        n = len(frame)
        results = [c.compute(frame, ("energy", "forces")) for c in calculators]
        e = np.array([r.energy for r in results]) / max(n, 1)
        f = np.stack([np.asarray(r.forces) for r in results])
        spread = np.sqrt(((f - f.mean(axis=0)) ** 2).sum(axis=2).mean(axis=0))
        atoms.append(n)
        energy.append(e.std())
        forces.append(spread)
        largest.append(spread.max() if n else 0.0)
    if not atoms:
        raise InvalidParameter("there are no frames")
    return CommitteeResult(
        atoms=np.array(atoms, dtype=np.int64),
        energy=np.array(energy),
        forces=np.concatenate(forces),
        max_force=np.array(largest),
    )


def novelty(points: Any, reference: Any = None) -> np.ndarray:
    """How far each point lies from the nearest of others, such as the
    descriptors of candidate frames from those of a training set.

    Parameters
    ----------
    points : array_like
        (M, D) the points.
    reference : array_like, optional
        (K, D) the points to measure from. Default the other points of
        ``points``, whose nearly zero distances find repeated frames.

    Returns
    -------
    numpy.ndarray
        (M,) the distance of each point from the nearest reference point.
    """
    x = np.asarray(points, dtype=np.float64)
    alone = reference is None
    y = x if alone else np.asarray(reference, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise InvalidParameter("points and reference must be (M, D) and (K, D)")
    if len(y) < (2 if alone else 1):
        raise InvalidParameter("there are no other points to measure from")
    norms = (y**2).sum(axis=1)
    found = np.empty(len(x))
    block = max(1, (1 << 24) // max(len(y), 1))
    for start in range(0, len(x), block):
        part = x[start : start + block]
        squared = (part**2).sum(axis=1)[:, None] - 2 * part @ y.T + norms[None, :]
        if alone:
            squared[np.arange(len(part)), np.arange(start, start + len(part))] = np.inf
        found[start : start + len(part)] = np.sqrt(np.maximum(squared.min(axis=1), 0.0))
    return found


def check_frames(frames: Sequence[System], *, close: float = 0.5) -> pl.DataFrame:
    """What may be wrong with each frame of a training set: atoms too close,
    a net force, missing labels.

    Parameters
    ----------
    frames : sequence of System
        The frames, with an ``element`` column.
    close : float, optional
        Two atoms closer than this fraction of the sum of their covalent
        radii are too close. Default 0.5.

    Returns
    -------
    polars.DataFrame
        One row per frame: frame, atoms; closest, the smallest distance of
        two atoms over the sum of their covalent radii where it is below the
        fraction, else inf; too_close; net_force, the length of the sum of
        the forces in eV/Å (NaN without forces); and has_energy,
        has_forces, has_virial.
    """
    if not close > 0:
        raise InvalidParameter(f"close must be positive, not {close!r}")
    rows = []
    for k, frame in enumerate(frames):
        radii = covalent_radii(frame["element"]) if len(frame) else np.zeros(0)
        closest = np.inf
        if len(frame) > 1:
            pairs = NeighborList.build(frame, close * 2 * float(radii.max()))
            i = np.repeat(np.arange(len(frame)), np.diff(pairs.offsets))
            ratio = pairs.distances / (radii[i] + radii[pairs.j])
            if len(ratio) and ratio.min() < close:
                closest = float(ratio.min())
        has_forces = "forces" in frame.table.columns
        rows.append(
            {
                "frame": k,
                "atoms": len(frame),
                "closest": closest,
                "too_close": closest < close,
                "net_force": float(
                    np.linalg.norm(np.asarray(frame["forces"]).sum(axis=0))
                )
                if has_forces
                else float("nan"),
                "has_energy": frame.info.get("energy") is not None,
                "has_forces": has_forces,
                "has_virial": frame.info.get("virial") is not None,
            }
        )
    return pl.DataFrame(rows)
