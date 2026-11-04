# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Free energies of solids and liquids by nonequilibrium thermodynamic
integration in LAMMPS."""

import math
import tempfile
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from mdapy import units
from mdapy.calculators._lammps import LAMMPS
from mdapy.core.box import Box
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import CancelToken
from mdapy.elements import atomic_masses
from mdapy.errors import Cancelled, InvalidParameter, MissingProperty
from mdapy.simulation._dynamics import MolecularDynamics

_PHASES = ("solid", "liquid")
# the Uhlenbeck-Ford model of the liquid path: its strength over kT, its length
# and its cutoff in Å, where the interaction is about 1e-5 kT
_UF_STRENGTH = 50.0
_UF_LENGTH = 1.5
_UF_CUTOFF = 6.0
# how many times longer than the others the switching of the model alone is
_MODEL_SWITCHING = 4
# steps between the samples of the energies along a switching
_EVERY = 10
# damping time of the Langevin thermostat, in ps
_DAMPING = 0.1
# the enthalpy there and back differ by this part of kT when a phase has changed
_HYSTERESIS = 0.5
# times a scan is run again, short of where the phase changed
_RESCANS = 3
# below this mean square displacement in Å², over the time of the equilibration,
# the atoms of a liquid have not moved from their neighbors
_DIFFUSED = 1.0


@dataclass(frozen=True)
class FreeEnergyResult(Result):
    """The free energy of a phase, per atom.

    Attributes
    ----------
    temperatures : numpy.ndarray
        (T,) float64, the temperatures in K.
    pressure : float64 or None
        The pressure in GPa, None at constant volume.
    free_energy : numpy.ndarray
        (T,) float64, the Gibbs energy in eV, or at constant volume the Helmholtz
        energy.
    enthalpy : numpy.ndarray
        (T,) float64, the enthalpy in eV, or at constant volume the energy.
    entropy : numpy.ndarray
        (T,) float64, the entropy in eV/K.
    volume : numpy.ndarray
        (T,) float64, the volume in Å³.
    phase : str
        "solid" or "liquid".
    error : float64 or None
        The standard error in eV of the free energy at the first temperature,
        from the repeats; None for one repeat.
    dissipation : numpy.ndarray
        (T,) float64 half the difference in eV of the work there and back: near zero
        when the switching and the scan are slow enough and the phase stays
        as it is.
    system : System
        The atoms in the box of the first temperature.
    """

    temperatures: np.ndarray
    pressure: float | None
    free_energy: np.ndarray
    enthalpy: np.ndarray
    entropy: np.ndarray
    volume: np.ndarray
    phase: str
    error: float | None
    dissipation: np.ndarray
    system: System

    def crossing(self, other: "FreeEnergyResult") -> "PhaseCoexistence":
        """Where this phase and another have the same free energy.

        Parameters
        ----------
        other : FreeEnergyResult
            The other phase, at the same pressure, with temperatures that
            overlap these.

        Returns
        -------
        PhaseCoexistence
            The temperature, and the latent heat and the changes of entropy
            and volume from this phase to the other.

        Raises
        ------
        InvalidParameter
            If the pressures differ, the temperatures do not overlap, or the
            free energies do not cross between them.
        """
        if self.pressure != other.pressure:
            raise InvalidParameter("the two phases must be at the same pressure")
        low = max(self.temperatures.min(), other.temperatures.min())
        high = min(self.temperatures.max(), other.temperatures.max())
        if not low < high:
            raise InvalidParameter("the temperatures of the two phases do not overlap")
        grid = np.linspace(low, high, 2001)

        def on_grid(result: "FreeEnergyResult", values: np.ndarray) -> np.ndarray:
            order = np.argsort(result.temperatures)
            return np.interp(grid, result.temperatures[order], values[order])

        difference = on_grid(other, other.free_energy) - on_grid(self, self.free_energy)
        changes = np.flatnonzero(np.diff(np.sign(difference)))
        if len(changes) == 0:
            raise InvalidParameter(
                f"the free energies do not cross between {low:g} and {high:g} K"
            )
        k = changes[0]
        share = difference[k] / (difference[k] - difference[k + 1])
        temperature = grid[k] + share * (grid[k + 1] - grid[k])

        def jump(name: str) -> float:
            values = on_grid(other, getattr(other, name)) - on_grid(
                self, getattr(self, name)
            )
            return float(values[k] + share * (values[k + 1] - values[k]))

        return PhaseCoexistence(
            temperature=float(temperature),
            latent_heat=jump("enthalpy"),
            entropy=jump("entropy"),
            volume=jump("volume"),
        )


@dataclass(frozen=True)
class PhaseCoexistence(Result):
    """Two phases of the same free energy, per atom.

    Attributes
    ----------
    temperature : float64
        The temperature in K.
    latent_heat : float64
        The change of enthalpy in eV.
    entropy : float64
        The change of entropy in eV/K.
    volume : float64
        The change of volume in Å³.
    """

    temperature: float
    latent_heat: float
    entropy: float
    volume: float


class _Progress:
    """Counts the steps of a whole calculation for progress and cancel."""

    def __init__(self, total: int, progress, cancel):
        self.total, self.done = total, 0
        self.progress, self.cancel = progress, cancel

    def add(self, steps: int) -> None:
        self.done += steps
        if self.progress is not None:
            self.progress(self.done, self.total)

    def check(self) -> None:
        if self.cancel is not None and self.cancel.cancelled:
            raise Cancelled("the free energy calculation was cancelled")


def _run(md: MolecularDynamics, steps: int, counter: _Progress) -> None:
    """Runs steps in pieces, for progress and cancel, after setting up the
    fixes that changed since the last run."""
    left, first = steps, True
    while left > 0 or first:
        counter.check()
        piece = min(left, 2000)
        md.command(f"run {piece} post no" if first else f"run {piece} pre no post no")
        counter.add(piece)
        left -= piece
        first = False


def _printing(path: str) -> str:
    """The command that prints the energies being integrated, and λ, to a file."""
    return (
        f'fix mdapy_print all print {_EVERY} "${{mdapy_de}} ${{mdapy_l}}" '
        f'file "{path}" screen no'
    )


def _cumulative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The trapezoidal integral of y from the first x to each x."""
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])


def _switch(step: int, steps: int) -> str:
    """The commonly used switching function of the fraction of the steps done,
    with zero slope at both ends, as a formula of LAMMPS."""
    t = f"((step-{step})/{steps})"
    return f"{t}^5*(70*{t}^4-315*{t}^3+540*{t}^2-420*{t}+126)"


def _integral(path: Path, sign: int = 1) -> float:
    """The integral of the first column over the second of a printed file."""
    data = np.loadtxt(path, comments="#", ndmin=2)
    return sign * float(np.trapezoid(data[:, 0], data[:, 1]))


def _masses(system: System) -> np.ndarray:
    return atomic_masses(system.table["element"].cast(str).to_list())


def _elements(system: System) -> tuple[list[str], np.ndarray]:
    """The elements of the system, and the element index of each atom."""
    names = system.table["element"].cast(str).to_numpy()
    kinds, index = np.unique(names, return_inverse=True)
    return [str(k) for k in kinds], index


def _ideal_gas(system: System, temperature: float) -> float:
    """The free energy in eV per atom of the atoms as an ideal gas in their box."""
    kt = units.BOLTZMANN * temperature
    _, index = _elements(system)
    n = len(system)
    masses = _masses(system)
    wavelength = units.PLANCK / np.sqrt(
        2 * np.pi * masses * units.AMU_A2_PER_PS2_TO_EV * kt
    )
    counts = np.bincount(index)
    states = n * np.log(system.box.volume) - sum(math.lgamma(c + 1) for c in counts)
    return kt * (3 * np.log(wavelength).sum() - states) / n


def _einstein(
    system: System, temperature: float, springs: np.ndarray
) -> tuple[float, float]:
    """The free energy in eV per atom of independent springs, of (N,) constants
    in eV/Å², and the correction for the fixed center of mass."""
    kt = units.BOLTZMANN * temperature
    n = len(system)
    masses = _masses(system)
    omega = np.sqrt(springs / (masses * units.AMU_A2_PER_PS2_TO_EV))
    hbar = units.PLANCK / (2 * np.pi)
    einstein = 3 * kt * np.log(hbar * omega / kt).mean()
    weights = masses / masses.sum()
    variance = (weights**2 * kt / springs).sum()  # of the center of mass
    fixed = kt / n * np.log(n / system.box.volume * (2 * np.pi * variance) ** 1.5)
    return float(einstein), float(fixed)


def _mean_box(frames: Sequence[System]) -> Box:
    """The box averaged over the second half of the frames."""
    half = frames[len(frames) // 2 :]
    matrix = np.mean([frame.box.matrix for frame in half], axis=0)
    return Box(matrix, origin=half[-1].box.origin, pbc=half[-1].box.pbc)


def _thermal_means(rows, pressure: float | None, n: int) -> tuple[float, float]:
    """The enthalpy, or energy, and the volume per atom over the second half
    of rows of the thermodynamic state."""
    half = rows[len(rows) // 2 :]
    energy = half["total_energy"].to_numpy()
    volume = half["volume"].to_numpy()
    if pressure is not None:
        energy = energy + pressure * units.GPA_TO_EV_PER_A3 * volume
    return float(energy.mean()) / n, float(volume.mean()) / n


def _equilibrate(
    md: MolecularDynamics,
    steps: int,
    temperature,
    pressure: float | None,
    barostat: str,
    counter: _Progress,
    frames: int | None = None,
) -> None:
    done = md.step
    if pressure is None:
        md.run(
            steps,
            ensemble="nvt",
            temperature=temperature,
            thermo=max(steps // 200, 1),
            frames=frames,
            cancel=counter.cancel,
        )
    else:
        md.run(
            steps,
            ensemble="npt",
            temperature=temperature,
            pressure=pressure,
            barostat=barostat,
            thermo=max(steps // 200, 1),
            frames=frames,
            cancel=counter.cancel,
        )
    counter.add(md.step - done)


def _spring_constants(
    crystal: System, potential: LAMMPS, temperature, timestep, steps, seed, counter
) -> np.ndarray:
    """(N,) spring constants in eV/Å² that give each element the mean square
    displacement it has in the crystal at this temperature."""
    with MolecularDynamics(crystal, potential, timestep=timestep) as md:
        md.velocities(2 * temperature, seed=seed)
        md.command(
            f"fix mdapy_thermostat all langevin {temperature!r} {temperature!r} "
            f"{_DAMPING} {seed} zero yes"
        )
        before = md.step
        md.run(
            steps,
            ensemble="nve",
            thermo=steps,
            frames=max(steps // 50, 1),
            cancel=counter.cancel,
        )
        counter.add(md.step - before)
        frames = md.frames[len(md.frames) // 2 :]
    _, index = _elements(crystal)
    start = crystal["positions"]
    fractions = np.array(
        [crystal.box.to_fractional(frame["positions"]) for frame in frames]
    ) - crystal.box.to_fractional(start)
    shifts = (fractions - np.round(fractions)) @ crystal.box.matrix
    shifts -= shifts.mean(axis=1, keepdims=True)
    squares = (shifts**2).sum(axis=2).mean(axis=0)
    per_element = np.bincount(index, squares) / np.bincount(index)
    return 3 * units.BOLTZMANN * temperature / per_element[index]


def _frenkel_ladd(
    crystal: System,
    potential: LAMMPS,
    temperature: float,
    springs: np.ndarray,
    timestep: float,
    equilibration: int,
    switching: int,
    seed: int,
    counter: _Progress,
) -> tuple[float, float]:
    """The free energy per atom of a crystal at constant volume, by switching
    between it and independent springs, and half the difference of the works."""
    n = len(crystal)
    kinds, index = _elements(crystal)
    with (
        tempfile.TemporaryDirectory() as folder,
        MolecularDynamics(crystal, potential, timestep=timestep) as md,
    ):
        forward = Path(folder, "forward").as_posix()
        backward = Path(folder, "backward").as_posix()
        md.velocities(2 * temperature, seed=seed)
        lines = ["fix mdapy_integrate all nve"]
        fixes = []
        for k, kind in enumerate(kinds):
            types = [
                t + 1 for t, element in enumerate(potential.elements) if element == kind
            ]
            lines.append(f"group mdapy_{k} type {' '.join(map(str, types))}")
            lines.append(
                f"fix mdapy_spring{k} mdapy_{k} ti/spring "
                f"{float(springs[index == k][0])!r} {switching} {equilibration} "
                "function 2"
            )
            fixes.append(f"f_mdapy_spring{k}")
        # ti/spring rescales every force set before it: the thermostat comes
        # after it, or its forces fade with the crystal's
        lines.append(
            f"fix mdapy_thermostat all langevin {temperature!r} {temperature!r} "
            f"{_DAMPING} {seed} zero yes"
        )
        lines.append(f"variable mdapy_de equal pe-{'-'.join(fixes)}")
        lines.append("variable mdapy_l equal f_mdapy_spring0[1]")
        md.command("\n".join(lines))
        _run(md, equilibration, counter)
        md.command(_printing(forward))
        _run(md, switching, counter)
        md.command("unfix mdapy_print")
        _run(md, equilibration, counter)
        md.command(_printing(backward))
        _run(md, switching, counter)
        md.command("unfix mdapy_print")
        there = _integral(Path(forward)) / n  # solid minus springs, dλ from 0 to 1
        back = _integral(Path(backward)) / n  # the same, from 1 to 0
    einstein, fixed = _einstein(crystal, temperature, springs)
    return einstein + fixed + 0.5 * (there - back), -0.5 * (there + back)


def _scaled_pair(potential: LAMMPS, cutoff: float, epsilon: float) -> tuple[str, str]:
    """The commands of the interaction scaled by v_mdapy_a, plus the
    Uhlenbeck-Ford model scaled by v_mdapy_b, and the name of its style."""
    style, lines = None, []
    for line in potential.pair.splitlines():
        words = line.split()
        if words and words[0] == "pair_style":
            if words[1].startswith("hybrid"):
                raise InvalidParameter(
                    "the free energy of a liquid needs one pair style, not a hybrid"
                )
            style = words[1]
            arguments = " ".join(words[2:])
            lines.append(
                f"pair_style hybrid/scaled v_mdapy_a {style} {arguments} "
                f"v_mdapy_b ufm {cutoff}"
            )
        elif words and words[0] == "pair_coeff":
            lines.append(
                f"pair_coeff {words[1]} {words[2]} {style} {' '.join(words[3:])}"
            )
        else:
            lines.append(line)
    lines.append(f"pair_coeff * * ufm {epsilon!r} {_UF_LENGTH}")
    return "\n".join(lines), style


def _liquid_reference(
    liquid: System,
    potential: LAMMPS,
    temperature: float,
    timestep: float,
    equilibration: int,
    switching: int,
    seed: int,
    counter: _Progress,
) -> tuple[float, float]:
    """The free energy per atom of a liquid at constant volume, by switching
    between it and the Uhlenbeck-Ford model and from that to the ideal gas, and
    half the difference of the works."""
    n = len(liquid)
    kt = units.BOLTZMANN * temperature
    pair, style = _scaled_pair(potential, _UF_CUTOFF, _UF_STRENGTH * kt)
    with (
        tempfile.TemporaryDirectory() as folder,
        MolecularDynamics(liquid, potential, timestep=timestep) as md,
    ):
        paths = {
            name: Path(folder, name).as_posix() for name in ("to_model", "from_model")
        }

        def weights(formula: str) -> None:
            md.command(
                "\n".join(
                    [
                        "variable mdapy_l delete",
                        "variable mdapy_a delete",
                        "variable mdapy_b delete",
                        f"variable mdapy_l equal {formula}",
                        "variable mdapy_a equal 1-v_mdapy_l",
                        "variable mdapy_b equal v_mdapy_l",
                    ]
                )
            )

        md.velocities(temperature, seed=seed)
        md.command(
            "\n".join(
                [
                    "variable mdapy_l equal 0",
                    "variable mdapy_a equal 1-v_mdapy_l",
                    "variable mdapy_b equal v_mdapy_l",
                    pair,
                    f"compute mdapy_u all pair {style}",
                    "compute mdapy_v all pair ufm",
                    "variable mdapy_de equal c_mdapy_u-c_mdapy_v",
                    "fix mdapy_integrate all nve",
                    f"fix mdapy_thermostat all langevin {temperature!r} "
                    f"{temperature!r} {_DAMPING} {seed}",
                ]
            )
        )
        _run(md, equilibration, counter)
        weights(_switch(md.step, switching))
        md.command(_printing(paths["to_model"]))
        _run(md, switching, counter)
        md.command("unfix mdapy_print")
        weights("1")
        _run(md, equilibration, counter)
        weights(f"1-({_switch(md.step, switching)})")
        md.command(_printing(paths["from_model"]))
        _run(md, switching, counter)
        md.command("unfix mdapy_print")
        # the interaction minus the model, dλ from 0 to 1 and from 1 to 0
        there = _integral(Path(paths["to_model"])) / n
        back = _integral(Path(paths["from_model"])) / n
    # the model alone is cheap, and its dissipation the larger: four times as long
    model, spread = _model_excess(
        liquid,
        potential,
        temperature,
        timestep,
        equilibration,
        _MODEL_SWITCHING * switching,
        seed,
        counter,
    )
    return (
        _ideal_gas(liquid, temperature) + model + 0.5 * (there - back),
        -0.5 * (there + back) + spread,
    )


def _model_excess(
    liquid: System,
    potential: LAMMPS,
    temperature: float,
    timestep: float,
    equilibration: int,
    switching: int,
    seed: int,
    counter: _Progress,
) -> tuple[float, float]:
    """The free energy per atom of the Uhlenbeck-Ford model at the density of a
    liquid over that of the ideal gas, by switching it on from nothing, and
    half the difference of the works."""
    n = len(liquid)
    kt = units.BOLTZMANN * temperature
    with (
        tempfile.TemporaryDirectory() as folder,
        MolecularDynamics(liquid, potential, timestep=timestep) as md,
    ):
        up = Path(folder, "up").as_posix()
        down = Path(folder, "down").as_posix()

        def weight(formula: str) -> None:
            md.command(
                "\n".join(
                    [
                        "variable mdapy_l delete",
                        f"variable mdapy_l equal {formula}",
                    ]
                )
            )

        md.velocities(temperature, seed=seed + 1)
        md.command(
            "\n".join(
                [
                    "variable mdapy_l equal 1",
                    f"pair_style hybrid/scaled v_mdapy_l ufm {_UF_CUTOFF}",
                    f"pair_coeff * * ufm {_UF_STRENGTH * kt!r} {_UF_LENGTH}",
                    "compute mdapy_v all pair ufm",
                    "variable mdapy_de equal c_mdapy_v",
                    "fix mdapy_integrate all nve",
                    f"fix mdapy_thermostat all langevin {temperature!r} "
                    f"{temperature!r} {_DAMPING} {seed + 1}",
                ]
            )
        )
        _run(md, equilibration, counter)
        weight(f"1-({_switch(md.step, switching)})")
        md.command(_printing(down))
        _run(md, switching, counter)
        md.command("unfix mdapy_print")
        weight("0")
        _run(md, equilibration, counter)
        weight(_switch(md.step, switching))
        md.command(_printing(up))
        _run(md, switching, counter)
        md.command("unfix mdapy_print")
        rise = _integral(Path(up)) / n
        fall = _integral(Path(down)) / n
    return 0.5 * (rise - fall), 0.5 * (rise + fall)


def _melted(
    system: System,
    potential: LAMMPS,
    temperature: float,
    melt_temperature: float,
    pressure: float | None,
    timestep: float,
    steps: int,
    seed: int,
    counter: _Progress,
) -> tuple[System, float, float]:
    """The atoms as a liquid at this temperature, in its mean box; its enthalpy
    and volume per atom."""
    n = len(system)
    with MolecularDynamics(system, potential, timestep=timestep) as md:
        md.velocities(2 * melt_temperature, seed=seed)
        _equilibrate(md, steps, melt_temperature, pressure, "iso", counter)
        _equilibrate(
            md, steps, (melt_temperature, temperature), pressure, "iso", counter
        )
        start = md.system
        recorded = len(md.thermo)
        _equilibrate(
            md,
            steps,
            temperature,
            pressure,
            "iso",
            counter,
            frames=max(steps // 50, 1),
        )
        box = start.box if pressure is None else _mean_box(md.frames)
        enthalpy, volume = _thermal_means(md.thermo[recorded:], pressure, n)
        last = md.system
    # how far the atoms went while the temperature held, by their images
    lengths = start.box.matrix
    travel = (last["positions"] - start["positions"]) + (
        (last["image"] - start["image"]) @ lengths
    )
    travel -= travel.mean(axis=0)
    if (travel**2).sum(axis=1).mean() < _DIFFUSED:
        raise InvalidParameter(
            f"the atoms do not move as in a liquid at {temperature:g} K: raise "
            "melt_temperature or the temperature"
        )
    moved = last.with_box(box, scale_positions=True)
    return moved.drop("velocities", "image"), enthalpy, volume


def _scan(
    start: System,
    potential: LAMMPS,
    first: float,
    last: float,
    pressure: float | None,
    barostat: str,
    start_temperature: float,
    timestep: float,
    equilibration: int,
    steps: int,
    seed: int,
    counter: _Progress,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The temperatures, the enthalpy, or energy, there and back, and the
    volume, per atom, from runs with the temperature changing from first to
    last and back."""
    n = len(start)
    every = max(steps // 2000, 1)
    steps -= steps % every

    def enthalpy(rows) -> np.ndarray:
        energy = rows["total_energy"].to_numpy()
        if pressure is not None:
            energy = (
                energy + pressure * units.GPA_TO_EV_PER_A3 * rows["volume"].to_numpy()
            )
        return energy / n

    options = (
        dict(ensemble="nvt")
        if pressure is None
        else dict(ensemble="npt", pressure=pressure, barostat=barostat)
    )
    with MolecularDynamics(start, potential, timestep=timestep) as md:
        md.velocities(start_temperature, seed=seed)
        before = md.step
        md.run(equilibration, temperature=first, thermo=equilibration, **options)
        md.run(
            steps,
            temperature=(first, last),
            thermo=every,
            cancel=counter.cancel,
            **options,
        )
        rows = md.thermo
        there = rows[-(steps // every + 1) :]
        md.run(equilibration, temperature=last, thermo=equilibration, **options)
        md.run(
            steps,
            temperature=(last, first),
            thermo=every,
            cancel=counter.cancel,
            **options,
        )
        rows = md.thermo
        back = rows[-(steps // every + 1) :]
        counter.add(md.step - before)
    grid = np.linspace(first, last, len(there))
    volume = 0.5 * (there["volume"].to_numpy() + back["volume"].to_numpy()[::-1]) / n
    return grid, enthalpy(there), enthalpy(back)[::-1], volume


def _windowed(values: np.ndarray, width: int) -> np.ndarray:
    """values averaged over a window, shorter at the ends."""
    kernel = np.ones(width) / width
    weights = np.convolve(np.ones_like(values), kernel, mode="same")
    return np.convolve(values, kernel, mode="same") / weights


def _phase_change(grid, there, back) -> float | None:
    """The temperature at which a phase changed on the way there, or None. It
    did if the enthalpy there and back differ by a good part of kT at some
    temperature; where, the enthalpy there leaves the line it started on."""
    width = max(len(grid) // 20, 1)
    limit = _HYSTERESIS * units.BOLTZMANN * grid
    if (np.abs(_windowed(there - back, width)) < limit).all():
        return None
    smooth = _windowed(there, width)
    part = max(len(grid) // 4, 2)
    scaled = (grid - grid[0]) / (grid[-1] - grid[0])
    line = np.polynomial.polynomial.polyfit(scaled[:part], smooth[:part], 1)
    gap = np.abs(smooth - np.polynomial.polynomial.polyval(scaled, line))
    away = np.flatnonzero(gap[part:] > limit[part:])
    return float(grid[part + away[0]]) if len(away) else float(grid[-1])


def _smooth(x: np.ndarray, y: np.ndarray, degree: int = 5) -> np.ndarray:
    """y through a polynomial of x, for the noise of the instant values."""
    scale = (x - x.mean()) / max(np.ptp(x), 1e-12)
    return np.polynomial.polynomial.polyval(
        scale, np.polynomial.polynomial.polyfit(scale, y, min(degree, len(x) - 1))
    )


def free_energy(
    system: System,
    potential: LAMMPS,
    temperatures: float | ArrayLike,
    *,
    phase: str = "solid",
    pressure: float | None = 0.0,
    timestep: float = 0.001,
    equilibration: float = 10.0,
    switching: float = 50.0,
    scan: float = 100.0,
    repeats: int = 1,
    mixing: bool = False,
    melt_temperature: float | None = None,
    barostat: str | None = None,
    seed: int = 1,
    progress: Callable[[int, int], None] | None = None,
    cancel: CancelToken | None = None,
) -> FreeEnergyResult:
    """The free energy of a solid or a liquid, per atom, from molecular
    dynamics in LAMMPS, with classical atoms.

    Parameters
    ----------
    system : System
        The atoms, with an ``element`` column, in a box periodic in every
        direction: for a solid at their places in the crystal (relax them
        first), for a liquid anywhere, as it is melted first.
    potential : LAMMPS
        The interaction. A liquid needs one pair style, not a hybrid.
    temperatures : float64 or array_like
        Temperatures in K, positive. The free energy is calculated at the
        first, and followed from there to the others, which must lie on one
        side of it, in order.
    phase : {"solid", "liquid"}, optional
        The phase. Default "solid".
    pressure : float64 or None, optional
        Pressure in GPa for the Gibbs energy; None for the Helmholtz energy
        in the box of system. Default 0.
    timestep : float64, optional
        Time step in ps. Default 0.001.
    equilibration : float64, optional
        Time in ps of each equilibration. Default 10.
    switching : float64, optional
        Time in ps of each switching between two interactions. Default 50.
    scan : float64, optional
        Time in ps of each run between the first and the last temperature.
        Default 100.
    repeats : int32, optional
        Independent calculations at the first temperature. Default 1.
    mixing : bool, optional
        Add the ideal entropy of mixing of a random solid solution, whose
        atoms do not exchange places in the dynamics. A liquid has it by
        itself. Default False.
    melt_temperature : float64, optional
        Temperature in K at which a liquid is melted. Default twice the first
        temperature.
    barostat : {"iso", "aniso", "tri"}, optional
        How the box of a solid changes, as in
        ``MolecularDynamics.run``. Default "aniso", and "iso" for a liquid.
    seed : int32, optional
        Seed of the random numbers, positive. Default 1.
    progress : callable, optional
        Called as ``progress(done, total)`` with the steps run.
    cancel : CancelToken, optional
        Stops the calculation, which raises ``mdapy.errors.Cancelled``.

    Returns
    -------
    FreeEnergyResult
        The free energy, enthalpy, entropy and volume per atom at each
        temperature. A phase that changes on the way, as a crystal melts when
        it is too hot, leaves out the temperatures from there on, with a
        warning.

    Raises
    ------
    InvalidParameter
        For invalid options, a box not periodic in every direction, a hybrid
        pair style for a liquid, a liquid that does not move as one, a phase
        that changes in every scan, or if LAMMPS fails.
    MissingProperty
        If the system has no ``element`` column.
    Cancelled
        If cancelled.
    """
    if phase not in _PHASES:
        raise InvalidParameter(f"phase must be one of {_PHASES}, not {phase!r}")
    if not isinstance(potential, LAMMPS):
        raise InvalidParameter("potential must be a mdapy.calculators.LAMMPS")
    if "element" not in system.table.columns:
        raise MissingProperty("the free energy needs the element column")
    if not system.box.pbc.all():
        raise InvalidParameter(
            "the free energy needs a box periodic in every direction"
        )
    wanted = np.atleast_1d(np.asarray(temperatures, dtype=float))
    if wanted.ndim != 1 or not (wanted > 0).all():
        raise InvalidParameter("temperatures must be positive")
    steps_between = np.diff(wanted)
    if not ((steps_between > 0).all() or (steps_between < 0).all() or len(wanted) == 1):
        raise InvalidParameter("temperatures must go in one direction from the first")
    if not min(equilibration, switching, scan) > 0 or not timestep > 0:
        raise InvalidParameter(
            "timestep, equilibration, switching and scan must be positive"
        )
    if int(repeats) < 1 or int(seed) < 1:
        raise InvalidParameter("repeats and seed must be positive")
    if phase == "liquid" and mixing:
        raise InvalidParameter("a liquid has its entropy of mixing by itself")
    if melt_temperature is not None and not melt_temperature > 0:
        raise InvalidParameter("melt_temperature must be positive")
    if barostat is None:
        barostat = "aniso" if phase == "solid" else "iso"
    if phase == "liquid":
        _scaled_pair(potential, _UF_CUTOFF, 1.0)  # refuses a hybrid at once
    n = len(system)
    first, repeats, seed = float(wanted[0]), int(repeats), int(seed)
    equilibrium = max(int(round(equilibration / timestep)), 1)
    switch = max(int(round(switching / timestep)), 1)
    scanning = max(int(round(scan / timestep)), 1)
    moved = len(wanted) > 1
    if phase == "solid":
        total = (pressure is not None) * equilibrium + equilibrium
        total += repeats * 2 * (equilibrium + switch)
    else:
        total = 3 * equilibrium + repeats * (
            4 * equilibrium + (2 + 2 * _MODEL_SWITCHING) * switch
        )
    if moved:
        total += 2 * (equilibrium + scanning)  # more if the phase changes on the way
    counter = _Progress(total, progress, cancel)

    if phase == "solid":
        crystal, enthalpy, volume = _solid_state(
            system,
            potential,
            first,
            pressure,
            barostat,
            timestep,
            equilibrium,
            seed,
            counter,
        )
        springs = _spring_constants(
            crystal, potential, first, timestep, equilibrium, seed, counter
        )
        outcomes = [
            _frenkel_ladd(
                crystal,
                potential,
                first,
                springs,
                timestep,
                equilibrium,
                switch,
                seed + 1 + k,
                counter,
            )
            for k in range(repeats)
        ]
    else:
        crystal, enthalpy, volume = _melted(
            system,
            potential,
            first,
            2 * first if melt_temperature is None else float(melt_temperature),
            pressure,
            timestep,
            equilibrium,
            seed,
            counter,
        )
        outcomes = [
            _liquid_reference(
                crystal,
                potential,
                first,
                timestep,
                equilibrium,
                switch,
                seed + 1 + 2 * k,
                counter,
            )
            for k in range(repeats)
        ]
    values = np.array([value for value, _ in outcomes])
    reference = float(values.mean())
    if pressure is not None:
        reference += pressure * units.GPA_TO_EV_PER_A3 * crystal.box.volume / n
    if mixing:
        _, index = _elements(system)
        fractions = np.bincount(index) / n
        reference += (
            units.BOLTZMANN * first * float((fractions * np.log(fractions)).sum())
        )
    error = float(values.std(ddof=1) / np.sqrt(repeats)) if repeats > 1 else None
    dissipation0 = float(np.mean([abs(d) for _, d in outcomes]))

    if not moved:
        free = np.array([reference])
        return FreeEnergyResult(
            temperatures=wanted,
            pressure=None if pressure is None else float(pressure),
            free_energy=free,
            enthalpy=np.array([enthalpy]),
            entropy=(np.array([enthalpy]) - free) / wanted,
            volume=np.array([volume]),
            phase=phase,
            error=error,
            dissipation=np.array([dissipation0]),
            system=crystal,
        )

    last = float(wanted[-1])
    for attempt in range(_RESCANS + 1):
        grid, there, back, volumes = _scan(
            crystal,
            potential,
            first,
            last,
            pressure,
            barostat,
            2 * first if phase == "solid" else first,
            timestep,
            equilibrium,
            scanning,
            seed + 2 * repeats + 3 + attempt,
            counter,
        )
        changed = _phase_change(grid, there, back)
        if changed is None:
            break
        if attempt == _RESCANS:
            raise InvalidParameter(
                f"the {phase} changes phase near {changed:.0f} K in every scan from "
                f"{first:g} K: choose other temperatures"
            )
        # short of it by a tenth of the way
        last = changed - 0.1 * (changed - first)
    inside = (wanted - first) * (last - first) >= 0
    inside &= np.abs(wanted - first) <= abs(last - first)
    if not inside.all():
        warnings.warn(
            f"the {phase} changes phase beyond {last:.0f} K in a scan of {scan:g} ps: "
            f"the temperatures beyond are left out",
            stacklevel=2,
        )
        wanted = wanted[inside]
    # the free energy over the temperature changes by the integral of the
    # enthalpy over the temperature squared
    mean = 0.5 * (there + back)
    ratio = reference / first - _cumulative(mean / grid**2, grid)
    wasted = np.abs(_cumulative(0.5 * (there - back) / grid**2, grid)) * grid
    smooth = _smooth(grid, mean)
    order = np.argsort(grid)

    def at(values: np.ndarray) -> np.ndarray:
        return np.interp(wanted, grid[order], values[order])

    free = at(ratio) * wanted
    heat = at(smooth)
    return FreeEnergyResult(
        temperatures=wanted,
        pressure=None if pressure is None else float(pressure),
        free_energy=free,
        enthalpy=heat,
        entropy=(heat - free) / wanted,
        volume=at(_smooth(grid, volumes)),
        phase=phase,
        error=error,
        dissipation=dissipation0 + at(wasted),
        system=crystal,
    )


def _solid_state(
    system: System,
    potential: LAMMPS,
    temperature: float,
    pressure: float | None,
    barostat: str,
    timestep: float,
    steps: int,
    seed: int,
    counter: _Progress,
) -> tuple[System, float, float]:
    """The crystal scaled into its mean box at this temperature and pressure,
    with its atoms at their places; its enthalpy and volume per atom."""
    n = len(system)
    if pressure is None:
        with MolecularDynamics(system, potential, timestep=timestep) as md:
            md.velocities(2 * temperature, seed=seed)
            _equilibrate(md, steps, temperature, None, barostat, counter)
            enthalpy, volume = _thermal_means(md.thermo, None, n)
        return system, enthalpy, volume
    with MolecularDynamics(system, potential, timestep=timestep) as md:
        md.velocities(2 * temperature, seed=seed)
        _equilibrate(
            md,
            steps,
            temperature,
            pressure,
            barostat,
            counter,
            frames=max(steps // 50, 1),
        )
        box = _mean_box(md.frames)
        enthalpy, volume = _thermal_means(md.thermo, pressure, n)
    return system.with_box(box, scale_positions=True), enthalpy, volume
