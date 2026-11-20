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


class Wall:
    """A flat wall across an axis that reflects the atoms that pass it, still
    or moving at a constant velocity: a momentum mirror that a crystal hits,
    or a piston that drives a shock into it, as LAMMPS's fix wall/reflect
    and fix wall/piston. After the atoms move in a step, an atom past the
    wall is put back by as much and its velocity across the wall becomes
    2 u - v, u that of the wall.

    Parameters
    ----------
    axis : {"x", "y", "z"}, optional
        The axis of the system the wall is across; the periodic vectors of
        the box must be normal to it. Default "z".
    side : {"low", "high"}, optional
        The atoms are kept above the wall ("low") or below it ("high").
        Default "low".
    position : float64, optional
        The coordinate of the wall along the axis at the start of the run in
        Å. Default None, the face of the box on that side at the start of
        the run.
    velocity : float64, optional
        The velocity of the wall along the axis in Å/ps; positive moves it
        along the axis. Default 0, a still wall. In LAMMPS a wall moves on
        the low side of z only, up z.
    """

    def __init__(
        self,
        axis: str = "z",
        *,
        side: str = "low",
        position: float | None = None,
        velocity: float = 0.0,
    ):
        if not isinstance(axis, str) or axis not in ("x", "y", "z"):
            raise InvalidParameter(f"axis must be x, y or z, not {axis!r}")
        if side not in ("low", "high"):
            raise InvalidParameter(f"side must be low or high, not {side!r}")
        if position is not None and not np.isfinite(position):
            raise InvalidParameter(f"position must be finite, not {position!r}")
        if isinstance(velocity, bool) or not np.isfinite(velocity):
            raise InvalidParameter(f"velocity must be finite, not {velocity!r}")
        self.axis = axis
        self.side = side
        self.position = None if position is None else float(position)
        self.velocity = float(velocity)

    @property
    def _sign(self) -> float:
        """+1 for a wall that keeps the atoms above it, -1 below."""
        return 1.0 if self.side == "low" else -1.0

    @property
    def _normal(self) -> np.ndarray:
        """(3,) the unit vector into the side the atoms are kept on."""
        out = np.zeros(3)
        out["xyz".index(self.axis)] = self._sign
        return out

    def _start(self, matrix: np.ndarray, origin: np.ndarray, pbc: Any) -> float:
        """The coordinate of the wall at the start in a box of the system:
        its position, or the face of the box on its side. Refuses a box with
        a periodic vector that is not normal to the axis, across which the
        wall would cut its images."""
        k = "xyz".index(self.axis)
        for vector, periodic in zip(matrix, pbc):
            if periodic and abs(vector[k]) > 1e-9 * np.linalg.norm(vector):
                raise InvalidParameter(
                    f"a wall across {self.axis} needs a box whose periodic vectors "
                    f"are normal to {self.axis}"
                )
        if self.position is not None:
            return self.position
        cube = np.array([[i, j, m] for i in (0, 1) for j in (0, 1) for m in (0, 1)])
        corners = origin + cube @ matrix
        return float(corners[:, k].min() if self.side == "low" else corners[:, k].max())

    def __repr__(self) -> str:
        return (
            f"Wall(axis={self.axis!r}, side={self.side!r}, position={self.position!r}, "
            f"velocity={self.velocity!r})"
        )


class Indenter:
    """A virtual indenter that pushes the atoms out of it, as LAMMPS's fix
    indent: an atom at a distance r inside a sphere or a cylinder of radius R
    feels K (R - r)^2 away from it, and one past a plane K d^2 back across
    it, d how far past. Still or moving at a constant velocity; with a name,
    ``thermo`` records the force of the atoms on it and its energy.

    Parameters
    ----------
    shape : {"sphere", "cylinder", "plane"}, optional
        A sphere, a cylinder along ``axis``, or the region beyond a plane
        across ``axis``. Default "sphere".
    center : (3,) float64, optional
        The center of the sphere, or a point on the axis of the cylinder, at
        the start of the run in Å; for a sphere and a cylinder.
    radius : float64, optional
        The radius of the sphere or of the cylinder in Å, not negative; for
        a sphere and a cylinder.
    axis : {"x", "y", "z"}, optional
        The axis of the cylinder, or that across the plane.
    position : float64, optional
        The coordinate of the plane along its axis at the start of the run in
        Å; for a plane.
    side : str, optional
        For a sphere or a cylinder "out", pushing the atoms out of it, or
        "in", keeping them inside (default "out"); for a plane the side it
        fills, "low" or "high".
    stiffness : float64, optional
        K in eV/Å³, not negative. Default 10.
    velocity : (3,) float64 or float64, optional
        The velocity of the center in Å/ps, or of the plane along its axis.
        Default None, still.
    name : str, optional
        Records ``<name>_force``, (3,) the force of the atoms on the
        indenter in eV/Å, and ``<name>_energy`` in eV.
    """

    def __init__(
        self,
        shape: str = "sphere",
        *,
        center: Any = None,
        radius: float | None = None,
        axis: str | None = None,
        position: float | None = None,
        side: str | None = None,
        stiffness: float = 10.0,
        velocity: Any = None,
        name: str | None = None,
    ):
        if shape not in ("sphere", "cylinder", "plane"):
            raise InvalidParameter(
                f"shape must be sphere, cylinder or plane, not {shape!r}"
            )
        if isinstance(stiffness, bool) or not (
            np.isfinite(stiffness) and stiffness >= 0
        ):
            raise InvalidParameter(
                f"stiffness must be a number not negative, not {stiffness!r}"
            )
        self.shape = shape
        self.stiffness = float(stiffness)
        self.name = _name(name)
        if shape == "plane":
            if axis not in ("x", "y", "z"):
                raise InvalidParameter(f"a plane needs an axis x, y or z, not {axis!r}")
            if position is None or not np.isfinite(position):
                raise InvalidParameter(
                    f"a plane needs a finite position, not {position!r}"
                )
            if side not in ("low", "high"):
                raise InvalidParameter(
                    f"a plane needs a side low or high, not {side!r}"
                )
            if center is not None or radius is not None:
                raise InvalidParameter(
                    "a plane takes a position, not a center or radius"
                )
            velocity = 0.0 if velocity is None else velocity
            if (
                isinstance(velocity, bool)
                or np.ndim(velocity)
                or not np.isfinite(velocity)
            ):
                raise InvalidParameter(
                    f"the velocity of a plane is one number, not {velocity!r}"
                )
            self.axis, self.position, self.side = axis, float(position), side
            self.center, self.radius = None, None
            self.velocity = float(velocity)
            return
        if shape == "cylinder" and axis not in ("x", "y", "z"):
            raise InvalidParameter(f"a cylinder needs an axis x, y or z, not {axis!r}")
        if shape == "sphere" and axis is not None:
            raise InvalidParameter("a sphere takes no axis")
        if position is not None:
            raise InvalidParameter(f"a {shape} takes a center, not a position")
        if center is None:
            raise InvalidParameter(f"a {shape} needs a center")
        if (
            radius is None
            or isinstance(radius, bool)
            or not (np.isfinite(radius) and radius >= 0)
        ):
            raise InvalidParameter(
                f"radius must be a number not negative, not {radius!r}"
            )
        side = "out" if side is None else side
        if side not in ("out", "in"):
            raise InvalidParameter(f"side must be out or in, not {side!r}")
        self.axis, self.position, self.side = axis, None, side
        self.center = _vector(center, "center")
        self.radius = float(radius)
        still = velocity is None
        self.velocity = _vector((0.0, 0.0, 0.0) if still else velocity, "velocity")

    @property
    def moving(self) -> bool:
        return bool(np.any(self.velocity != 0.0))

    def _at(self, time: float) -> Any:
        """The center, or the position of the plane, after a time in ps."""
        if self.shape == "plane":
            return self.position + self.velocity * time
        return self.center + self.velocity * time

    def __repr__(self) -> str:
        where = (
            f"axis={self.axis!r}, position={self.position!r}"
            if self.shape == "plane"
            else f"center={self.center.tolist()!r}, radius={self.radius!r}"
            + (f", axis={self.axis!r}" if self.axis else "")
        )
        velocity = self.velocity if self.shape == "plane" else self.velocity.tolist()
        return (
            f"Indenter({self.shape!r}, {where}, side={self.side!r}, "
            f"stiffness={self.stiffness!r}, velocity={velocity!r}, name={self.name!r})"
        )


class _HeatField:
    """The driving field of homogeneous nonequilibrium molecular dynamics
    (HNEMD) for a run: each atom pushed by the field through its virial, the
    pushes less their mean, so that heat flows along the field; ``thermo``
    records ``heat_flux``, the potential part of the heat flux times the
    volume. The field is in 1/Å, in the axes of the system. convective: each
    atom pushed by the field times its energy less their mean too, and the
    energy each atom carries in the flux (Evans 1982), for atoms of one
    mass."""

    def __init__(self, field: Any, convective: bool = False):
        self.field = _vector(field, "field")
        if not self.field.any():
            raise InvalidParameter("the field must not be zero")
        self.convective = bool(convective)


CONTROLS = (
    Hold,
    Move,
    AddForce,
    Langevin,
    Deform,
    AdaptiveTimestep,
    Switch,
    Wall,
    Indenter,
    _HeatField,
)


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
    deforms = adaptive = switches = fields = 0
    walls: set = set()
    moving_walls = False
    indenters = moving_indenters = 0
    for control in out:
        if not isinstance(control, CONTROLS):
            raise InvalidParameter(
                "a control must be one of Hold, Move, AddForce, Langevin, Deform, "
                f"AdaptiveTimestep, Switch, Wall or Indenter, not {control!r}"
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
        if isinstance(control, Wall):
            if (control.axis, control.side) in walls:
                raise InvalidParameter(
                    f"two walls on the {control.side} side of {control.axis}"
                )
            walls.add((control.axis, control.side))
            moving_walls = moving_walls or control.velocity != 0.0
            continue
        if isinstance(control, Indenter):
            indenters += 1
            moving_indenters += control.moving
            if control.name is not None:
                if control.name in names:
                    raise InvalidParameter(f"two controls are named {control.name!r}")
                names.add(control.name)
            continue
        if isinstance(control, _HeatField):
            if fields:
                raise InvalidParameter("a run takes one heat field")
            fields += 1
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
    if fields and (switches or deforms or moved.any()):
        raise InvalidParameter(
            "a heat field with a Switch, a Deform or moved atoms is not supported"
        )
    if walls and deforms:
        raise InvalidParameter("a wall with a Deform is not supported")
    if indenters and deforms:
        raise InvalidParameter("an indenter with a Deform is not supported")
    if adaptive and moving_indenters:
        raise InvalidParameter(
            "an adaptive time step with a moving indenter is not supported"
        )
    if adaptive and moving_walls:
        raise InvalidParameter(
            "an adaptive time step with a moving wall is not supported"
        )
    if adaptive and deforms:
        raise InvalidParameter("an adaptive time step with a Deform is not supported")
    if adaptive and moved.any():
        raise InvalidParameter(
            "an adaptive time step with moved atoms is not supported"
        )
    return out
