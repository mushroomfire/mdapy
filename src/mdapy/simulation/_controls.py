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


CONTROLS = (Hold, Move, AddForce, Langevin)


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
    for control in out:
        if not isinstance(control, CONTROLS):
            raise InvalidParameter(
                f"a control must be one of Hold, Move, AddForce or Langevin, not {control!r}"
            )
        if control.mask.shape != (n,):
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
    return out
