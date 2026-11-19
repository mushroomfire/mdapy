# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Controls of a molecular dynamics run: what is done to groups of atoms,
each a boolean mask over the atoms of the system, during one run."""

from typing import Any

import numpy as np

from mdapy.errors import InvalidParameter


def _mask(mask: Any) -> np.ndarray:
    out = np.asarray(mask)
    if out.dtype != bool or out.ndim != 1:
        raise InvalidParameter("a group must be an (N,) bool mask over the atoms")
    return out.copy()


def _vector(value: Any, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if out.shape != (3,) or not np.isfinite(out).all():
        raise InvalidParameter(f"{name} must be three finite numbers, not {value!r}")
    return out


def _name(name: Any) -> str | None:
    if name is None:
        return None
    if not isinstance(name, str) or not name.isidentifier():
        raise InvalidParameter(f"name must be a word, not {name!r}")
    return name


class Hold:
    """Atoms kept still along some directions for a run: their velocities
    there are zero and their forces do nothing, as a ``movable`` column that
    is False there.

    Parameters
    ----------
    mask : numpy.ndarray
        (N,) bool, the atoms held.
    directions : str, optional
        The axes of the system along which they are held, any of "x", "y"
        and "z". Default "xyz", still.
    """

    def __init__(self, mask: Any, *, directions: str = "xyz"):
        self.mask = _mask(mask)
        if (
            not isinstance(directions, str)
            or not directions
            or any(c not in "xyz" for c in directions)
            or len(set(directions)) != len(directions)
        ):
            raise InvalidParameter(
                f"directions must be some of x, y and z, not {directions!r}"
            )
        self.directions = directions

    @property
    def axes(self) -> np.ndarray:
        """(3,) bool, the axes held."""
        return np.array([c in self.directions for c in "xyz"])

    def __repr__(self) -> str:
        return f"Hold({int(self.mask.sum())} atoms, directions={self.directions!r})"


class Move:
    """Atoms moved together at a velocity for a run, whatever the forces on
    them: a rigid part that drives the rest, such as a wall that shears a
    crystal or a piston. The force the rest of the atoms exert on the group
    is recorded.

    Parameters
    ----------
    mask : numpy.ndarray
        (N,) bool, the atoms moved.
    velocity : (float64, float64, float64)
        Their velocity in Å/ps, in the axes of the system.
    name : str, optional
        With a name, ``thermo`` records ``<name>_force``, the force of the
        potential on the group in eV/Å. Default None, recording none.
    """

    def __init__(self, mask: Any, velocity: Any, *, name: str | None = None):
        self.mask = _mask(mask)
        self.velocity = _vector(velocity, "velocity")
        self.name = _name(name)

    def __repr__(self) -> str:
        return (
            f"Move({int(self.mask.sum())} atoms, velocity={self.velocity.tolist()!r}, "
            f"name={self.name!r})"
        )


class AddForce:
    """A force added to each atom of a group for a run, as LAMMPS's fix
    addforce: a load on a part of the system.

    Parameters
    ----------
    mask : numpy.ndarray
        (N,) bool, the atoms pushed.
    force : (float64, float64, float64)
        The force on each in eV/Å, in the axes of the system.
    """

    def __init__(self, mask: Any, force: Any):
        self.mask = _mask(mask)
        self.force = _vector(force, "force")

    def __repr__(self) -> str:
        return f"AddForce({int(self.mask.sum())} atoms, force={self.force.tolist()!r})"


class Langevin:
    """A Langevin thermostat on a group of atoms for a run, friction and
    random forces on them alone: a heat bath, such as the source and the
    sink of heat that flows through a crystal, or the borders that take up
    the heat of a cascade. With a name, ``thermo`` records the heat it gave
    the group since the run began.

    Parameters
    ----------
    mask : numpy.ndarray
        (N,) bool, the atoms of the bath.
    temperature : float64 or (float64, float64)
        Its temperature in K, positive, or its values at the start and the
        end of the run.
    damping : float64, optional
        Its damping time in ps, positive. Default 100 time steps.
    seed : int32, optional
        Seed of its random forces, positive; unpredictable when None.
    name : str, optional
        With a name, ``thermo`` records ``<name>_energy``, the heat in eV
        the bath gave the group since the run began, negative when it took
        heat away. Default None, recording none.
    """

    def __init__(
        self,
        mask: Any,
        temperature: Any,
        *,
        damping: float | None = None,
        seed: int | None = None,
        name: str | None = None,
    ):
        self.mask = _mask(mask)
        values = np.atleast_1d(np.asarray(temperature, dtype=float))
        if values.size == 1:
            values = np.repeat(values, 2)
        if (
            values.shape != (2,)
            or not (values > 0).all()
            or not np.isfinite(values).all()
        ):
            raise InvalidParameter(
                f"temperature must be positive, or (start, stop), not {temperature!r}"
            )
        self.temperature = (float(values[0]), float(values[1]))
        if damping is not None and not (np.isfinite(damping) and damping > 0):
            raise InvalidParameter(f"damping must be positive, not {damping!r}")
        self.damping = None if damping is None else float(damping)
        if seed is not None and (
            isinstance(seed, bool) or int(seed) != seed or seed < 1
        ):
            raise InvalidParameter(f"seed must be a positive integer, not {seed!r}")
        self.seed = None if seed is None else int(seed)
        self.name = _name(name)

    def __repr__(self) -> str:
        first, last = self.temperature
        heat = f"{first!r} K" if first == last else f"{first!r} to {last!r} K"
        return (
            f"Langevin({int(self.mask.sum())} atoms, {heat}, damping={self.damping!r}, "
            f"seed={self.seed!r}, name={self.name!r})"
        )


class Deform:
    """The box deformed at constant rates for a run: stretched, compressed or
    sheared, the atoms carried with it, as LAMMPS's fix deform with erate
    or trate, remap x. With a barostat, the barostat holds the pressure of
    the lengths the deformation leaves: a crystal stretched along x at zero
    pressure along y and z.

    Parameters
    ----------
    xx, yy, zz : float64, optional
        Strain rates of the lengths of the box along x, y and z, in 1/ps;
        positive stretches. Default 0, the length left.
    yz, xz, xy : float64, optional
        Shear strain rates in 1/ps: the tilt yz grows by the rate times the
        length along z at the start, every ps, xz the same, and xy by the
        rate times the length along y. Default 0.
    kind : {"engineering", "true"}, optional
        Of the lengths: growing by the rate times their length at the start
        every ps (LAMMPS erate), or by the rate times their length now (trate,
        exponentially). Default "engineering".
    """

    def __init__(
        self,
        *,
        xx: float = 0.0,
        yy: float = 0.0,
        zz: float = 0.0,
        yz: float = 0.0,
        xz: float = 0.0,
        xy: float = 0.0,
        kind: str = "engineering",
    ):
        rates = np.array([xx, yy, zz, yz, xz, xy], dtype=float)
        if not np.isfinite(rates).all():
            raise InvalidParameter("the strain rates must be finite")
        if not rates.any():
            raise InvalidParameter("a deformation needs a strain rate that is not zero")
        if kind not in ("engineering", "true"):
            raise InvalidParameter(f"kind must be engineering or true, not {kind!r}")
        self.rates = rates
        self.kind = kind

    def _periodic(self, pbc: np.ndarray) -> None:
        """Refuses rates on lengths that are not periodic, and tilts across
        them (yz and xz across z, xy across y)."""
        needs = np.array([pbc[0], pbc[1], pbc[2], pbc[2], pbc[2], pbc[1]])
        if (self.rates[~needs] != 0.0).any():
            raise InvalidParameter(
                "a deformation changes periodic lengths only, and tilts across them"
            )

    def __repr__(self) -> str:
        names = ("xx", "yy", "zz", "yz", "xz", "xy")
        rates = ", ".join(f"{n}={r!r}" for n, r in zip(names, self.rates.tolist()) if r)
        return f"Deform({rates}, kind={self.kind!r})"


class AdaptiveTimestep:
    """The time step of a run set by the fastest atom, as LAMMPS's fix
    dt/reset: at the start of the run and every ``every`` steps, the largest
    step in which no atom moves more than ``max_distance``, by its velocity
    and its force, within the bounds; for the collisions of a cascade, whose
    atoms start fast and slow down.

    Parameters
    ----------
    max_distance : float64
        The largest distance an atom moves in a step, in Å, positive.
    min_timestep : float64, optional
        The smallest time step in ps, positive. Default None, no bound.
    max_timestep : float64, optional
        The largest time step in ps, positive. Default None, the time step
        of ``MolecularDynamics``.
    every : int32, optional
        Steps between the settings of the time step, positive. Default 1.
    """

    def __init__(
        self,
        max_distance: float,
        *,
        min_timestep: float | None = None,
        max_timestep: float | None = None,
        every: int = 1,
    ):
        def positive(value: Any, name: str) -> float:
            if isinstance(value, bool) or not (np.isfinite(value) and float(value) > 0):
                raise InvalidParameter(f"{name} must be positive, not {value!r}")
            return float(value)

        self.max_distance = positive(max_distance, "max_distance")
        self.min_timestep = (
            None if min_timestep is None else positive(min_timestep, "min_timestep")
        )
        self.max_timestep = (
            None if max_timestep is None else positive(max_timestep, "max_timestep")
        )
        if (
            self.min_timestep is not None
            and self.max_timestep is not None
            and self.min_timestep >= self.max_timestep
        ):
            raise InvalidParameter("min_timestep must be less than max_timestep")
        if isinstance(every, bool) or int(every) != every or every < 1:
            raise InvalidParameter(f"every must be a positive integer, not {every!r}")
        self.every = int(every)

    def __repr__(self) -> str:
        return (
            f"AdaptiveTimestep(max_distance={self.max_distance!r}, "
            f"min_timestep={self.min_timestep!r}, "
            f"max_timestep={self.max_timestep!r}, every={self.every!r})"
        )


def smoothstep(t: float) -> float:
    """The switching function of LAMMPS's fix ti/spring with function 2,
    from 0 at t = 0 to 1 at t = 1 with zero slopes at both ends."""
    t2 = t * t
    return (
        (70.0 * t2 * t2 - 315.0 * t2 * t + 540.0 * t2 - 420.0 * t + 126.0) * t2 * t2 * t
    )


class Switch:
    """The potential of a run turned into another along the run, for
    thermodynamic integration: the forces and the energy of (1 − λ) times
    the potential plus λ times the target, λ going from ``start`` to
    ``stop`` by the switching function of LAMMPS's fix ti/spring, whose
    slope is zero at both ends, or staying at ``start`` when the two are
    equal. With a name, ``thermo`` records ``<name>_lambda`` and
    ``<name>_difference``, the energy of the target minus that of the
    potential, whose integral over λ is the difference of their free
    energies.

    Parameters
    ----------
    target : Calculator or None
        The other interaction, such as ``Springs`` or ``UhlenbeckFord``;
        None for none, the ideal gas.
    start : float64, optional
        λ at the start of the run, from 0 to 1. Default 0.
    stop : float64, optional
        λ at the end of the run, from 0 to 1. Default 1.
    name : str, optional
        With a name, ``thermo`` records λ and the difference of the
        energies. Default None, recording none.
    """

    def __init__(
        self,
        target: Any,
        *,
        start: float = 0.0,
        stop: float = 1.0,
        name: str | None = None,
    ):
        from mdapy.calculators._base import Calculator

        if target is not None and not isinstance(target, Calculator):
            raise InvalidParameter(
                f"target must be a calculator or None, not {target!r}"
            )
        for value, what in ((start, "start"), (stop, "stop")):
            if isinstance(value, bool) or not (0.0 <= float(value) <= 1.0):
                raise InvalidParameter(f"{what} must be from 0 to 1, not {value!r}")
        self.target = target
        self.start = float(start)
        self.stop = float(stop)
        self.name = _name(name)

    def weight(self, fraction: float) -> float:
        """λ when this fraction of the run is done."""
        if self.start == self.stop:
            return self.start
        return self.start + (self.stop - self.start) * smoothstep(fraction)

    def __repr__(self) -> str:
        return (
            f"Switch({self.target!r}, start={self.start!r}, stop={self.stop!r}, "
            f"name={self.name!r})"
        )


CONTROLS = (Hold, Move, AddForce, Langevin, Deform, AdaptiveTimestep, Switch)


def checked(controls: Any, n: int) -> tuple:
    """The controls of a run, checked for n atoms: groups of n atoms, no atom
    both held and moved or moved by two, names that differ."""
    if controls is None:
        return ()
    if isinstance(controls, CONTROLS):
        controls = (controls,)
    out = tuple(controls)
    names = set()
    moved = np.zeros(n, dtype=bool)
    held = np.zeros(n, dtype=bool)
    bathed = np.zeros(n, dtype=bool)
    deforms = adaptive = switches = 0
    for control in out:
        if not isinstance(control, CONTROLS):
            raise InvalidParameter(
                "a control must be one of Hold, Move, AddForce, Langevin, Deform, "
                f"AdaptiveTimestep or Switch, not {control!r}"
            )
        if isinstance(control, Deform):
            if deforms:
                raise InvalidParameter("a run takes one Deform")
            deforms += 1
            continue
        if isinstance(control, AdaptiveTimestep):
            if adaptive:
                raise InvalidParameter("a run takes one AdaptiveTimestep")
            adaptive += 1
            continue
        if isinstance(control, Switch):
            if switches:
                raise InvalidParameter("a run takes one Switch")
            switches += 1
            if control.name is not None:
                if control.name in names:
                    raise InvalidParameter(f"two controls are named {control.name!r}")
                names.add(control.name)
            continue
        if not isinstance(control, Deform) and control.mask.shape != (n,):
            raise InvalidParameter(
                f"the group of {control!r} has {control.mask.size} values, not one "
                f"for each of the {n} atoms"
            )
        if isinstance(control, Move):
            if (moved & control.mask).any() or (held & control.mask).any():
                raise InvalidParameter(
                    "an atom moved by one control is held or moved by another"
                )
            moved |= control.mask
        if isinstance(control, Hold):
            if (moved & control.mask).any():
                raise InvalidParameter(
                    "an atom moved by one control is held or moved by another"
                )
            held |= control.mask
        if isinstance(control, Langevin):
            if (bathed & control.mask).any():
                raise InvalidParameter("an atom is in two Langevin baths")
            bathed |= control.mask
        name = getattr(control, "name", None)
        if name is not None:
            if name in names:
                raise InvalidParameter(f"two controls are named {name!r}")
            names.add(name)
    if (moved & bathed).any():
        raise InvalidParameter("an atom moved by a control is in a Langevin bath")
    if deforms and (moved.any() or held.any()):
        raise InvalidParameter(
            "a deformed box with held or moved atoms is not supported"
        )
    if adaptive and deforms:
        raise InvalidParameter("an adaptive time step with a Deform is not supported")
    if adaptive and moved.any():
        raise InvalidParameter(
            "an adaptive time step with moved atoms is not supported"
        )
    return out
