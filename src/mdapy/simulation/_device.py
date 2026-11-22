# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The atoms of MDAPY's dynamics on an NVIDIA GPU: positions, velocities and
forces held on the device, where mdapy._cuda does the steps of the
integrator on every atom; each step copies back only the few numbers the
thermostats and the barostat need, and the atoms come back when asked.
Under MPI a subclass holds the atoms of one process on its GPU, with its
ghosts (simulation/_domain_device.py): the sums over the atoms then go
through _total and the arrays of all the atoms through _rows_of."""

import math
import secrets
from typing import Any

import numpy as np
import polars as pl

from mdapy import units
from mdapy.calculators import _gpu
from mdapy.calculators._base import Calculator
from mdapy.calculators._springs import Springs
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import CalculationFailed, InvalidParameter

_MVV2E = units.AMU_A2_PER_PS2_TO_EV
_NOT_FINITE = "the energy or the forces are not finite: atoms may overlap"


class DeviceAtoms:
    """The interface of _HostAtoms of simulation/_integrator.py, on the GPU."""

    def __init__(
        self,
        calculator: Calculator,
        system: System,
        x: np.ndarray,
        v: np.ndarray,
        masses: np.ndarray,
        skin: float,
        movable: np.ndarray | None = None,
        rotation: np.ndarray | None = None,
    ):
        """movable, (N, 3) bool in the axes of the system, holds atoms along
        the directions where it is False; the vectors of the system are those
        of x and v times rotation transposed."""
        on = getattr(calculator, "_on_device", None)
        if on is None:
            raise InvalidParameter(
                f"{type(calculator).__name__} has no molecular dynamics on the GPU"
            )
        self._module = _gpu.cuda()
        self._system = system
        # of all the atoms
        self._species = np.ascontiguousarray(calculator._species(system))
        self._masses = np.asarray(masses, dtype=np.float64)
        self._count = len(system)
        self._rotation = np.ascontiguousarray(
            np.eye(3) if rotation is None else rotation, dtype=np.float64
        )
        self._movable = movable  # the atoms the system itself holds
        self._free: np.ndarray | None = movable  # those of the run
        self._prescribed: np.ndarray | None = None
        self._added: np.ndarray | None = None
        self._watched = 0  # groups whose forces are summed: bits 0 to this
        self._baths: list = []  # (bit, seed, [counter]) of each bath
        self._masks: list = []  # the masks of the groups of the controls
        # the electronic stopping of a run: its table on the device, and
        # (least, needed, cutoff, bit of its group or -1, last) as set_stopping
        # makes them
        self._stopper: Any = None
        self._stopping: Any = None
        self._stopping_mask: np.ndarray | None = None
        self.electronic_power = 0.0
        self.sums: list = []
        self._cutoff, self._skin = on.cutoff, skin
        self._neighbors = self._module.Neighbors(on.cutoff, skin)
        self._potential = on.potential
        self._h = np.zeros((3, 3))  # the box, for the images of the wrapped atoms
        self._seed = 0
        self._counter = 0
        # a Switch: the other interaction (False for none, None for the ideal
        # gas), its potential on the device and its species, and λ
        self._target: Any = False
        self._other: Any = None
        self._other_species: np.ndarray | None = None
        self._sites: tuple | None = None  # (sites, constants) of Springs
        self.weight = 0.0
        self.parts = (math.nan, math.nan)
        self._field: np.ndarray | None = None  # of HNEMD, in the frame
        self._convective, self._kick = False, 0.0  # as on the processor
        self._atoms: Any = None
        self._new_atoms(x, v)

    def _new_atoms(self, x: np.ndarray, v: np.ndarray) -> None:
        """The atoms on the device, with their species and masses."""
        atoms = self._module.Atoms(len(x))
        atoms.set_species(self._rows_of(self._species))
        atoms.set_masses(np.ascontiguousarray(self._rows_of(self._masses)))
        atoms.set_positions(np.ascontiguousarray(x, dtype=np.float64))
        atoms.set_velocities(np.ascontiguousarray(v, dtype=np.float64))
        self._atoms = atoms
        self._set_free(self._free)

    # -- what a subclass of one process of MPI overrides ----------------------

    def _rows_of(self, values: np.ndarray) -> np.ndarray:
        """The rows of the atoms on the device of an array of all the atoms."""
        return np.ascontiguousarray(values)

    def _total(self, value: Any) -> Any:
        """The sum over the processes."""
        return value

    def _agreed(self, value: Any) -> Any:
        """What the main process has, on every process."""
        return value

    def _update(self, box: Box, h: np.ndarray, lo: np.ndarray, pbc: Any) -> tuple:
        """The neighbor list built again when the atoms moved enough; the box
        the potentials compute in."""
        try:
            self._neighbors.update(self._atoms, h, lo, pbc)
        except ValueError as error:
            raise InvalidParameter(str(error)) from None
        except self._module.NotFinite:
            raise CalculationFailed(_NOT_FINITE) from None
        return h, lo, pbc

    def _finish_forces(self) -> None:
        """After a potential computed: the forces on the ghosts sent back."""

    # the module, the atoms and the neighbor list on the device, the
    # potentials there, and the system: given again or built again, not kept
    # in a restart, whose system holds the positions and the velocities
    _TRANSIENT = (
        "_module",
        "_atoms",
        "_system",
        "_neighbors",
        "_potential",
        "_other",
        "_target",
        "_stopper",
    )

    def state(self) -> dict[str, Any]:
        """What a restart keeps of the atoms on the device, besides their
        velocities: the forces of the last step too, those of the atoms
        before a deformation carried them, which the next step uses, and the
        positions unwrapped, for the boxes each atom crossed, which the
        wrapped positions of the system do not tell."""
        kept = {k: v for k, v in vars(self).items() if k not in self._TRANSIENT}
        kept["forces"] = self.forces
        kept["positions"] = self.positions
        return kept

    def restore(self, state: dict[str, Any]) -> None:
        """The counters, baths and controls as a restart kept them, the
        forces and the unwrapped positions, which the device wraps at its
        next build; the controls of a run are set on the device by its
        start."""
        state = dict(state)
        forces = state.pop("forces")
        positions = state.pop("positions", None)
        vars(self).update({k: v for k, v in state.items() if k not in self._TRANSIENT})
        if positions is not None:
            self._atoms.set_positions(
                np.ascontiguousarray(self._rows_of(positions), dtype=np.float64)
            )
        self._atoms.set_forces(
            np.ascontiguousarray(self._rows_of(forces), dtype=np.float64)
        )

    @property
    def positions(self) -> np.ndarray:
        """Unwrapped: the GPU wraps the atoms into the box when it builds a
        neighbor list and counts the boxes it moved them by."""
        return self._atoms.positions() + self._atoms.images() @ self._h

    @property
    def velocities(self) -> np.ndarray:
        return self._atoms.velocities()

    @velocities.setter
    def velocities(self, v: np.ndarray) -> None:
        self._atoms.set_velocities(np.ascontiguousarray(v, dtype=np.float64))

    @property
    def forces(self) -> np.ndarray:
        return self._atoms.forces()

    def twice_kinetic(self) -> float:
        return float(np.trace(self._total(self._atoms.kinetic()))) * _MVV2E

    def step_limit(self, distance: float) -> float:
        """As on the processor, from the velocities and forces copied back."""
        from mdapy.simulation._integrator import _step_limit

        return _step_limit(self.velocities, self.forces, self._masses, distance)

    def extent(self, lo: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """As on the processor, from the positions copied back."""
        s = (self.positions - lo) @ np.linalg.inv(h)
        return s.min(axis=0), s.max(axis=0)

    def mvv(self) -> np.ndarray:
        return self._total(self._atoms.kinetic()) * _MVV2E

    def compute(
        self, table: pl.DataFrame, box: Box, energy: bool, virial: bool
    ) -> tuple[float, np.ndarray]:
        """The forces, and the energy and the virial when asked; the
        neighbor list built again when the atoms have moved enough."""
        h, lo, pbc = _gpu.box_arguments(box)
        self._h = h
        sub = self._update(box, h, lo, pbc)
        driven = self._field is not None
        # with the field, the energies and virials of the heat flux
        self._evaluate(self._potential, sub, energy, virial, driven)
        self._finish_forces()
        if driven:
            self._drive(self._field)
            if self._convective:
                self._convect(self._field, self._kick)
        if self._target is not False:
            e, w = self._switched(sub, energy, virial)
        if self._watched:
            self.sums = list(
                np.asarray(self._total(self._atoms.group_forces(self._watched)))
            )
        self.indented = []
        for (kind, axis, sign, k, radius), at in zip(self.indenters, self.indenter_at):
            where = np.zeros(3)
            where[:] = at if np.ndim(at) else (float(at), 0.0, 0.0)
            out = self._total(
                np.asarray(
                    self._atoms.indent(
                        kind, axis, sign, k, radius, where, np.ascontiguousarray(h), pbc
                    ),
                    dtype=float,
                )
            )
            self.indented.append((float(out[0]), np.asarray(out[1:], dtype=float)))
        if self._stopping is not None:
            least, needed, cutoff, bit, last = self._stopping
            power, beyond = self._stopper.apply(
                self._atoms, self._neighbors, *sub, least, needed, cutoff, bit, _MVV2E
            )
            power, beyond = self._total(np.array([power, beyond], dtype=float))
            if beyond:
                raise CalculationFailed(
                    "an atom is faster than the stopping table reaches: its kinetic "
                    f"energy is beyond {last!r} eV, the last energy of the table"
                )
            self.electronic_power = float(power)
        self._atoms.add_forces()
        if self._free is not None:
            self._atoms.restrict_forces()
        if self._target is False:
            e = self._total(self._atoms.total_energy()) if energy else math.nan
            w = (
                self._total(self._atoms.total_virial())
                if virial
                else np.full((3, 3), np.nan)
            )
        if energy and not math.isfinite(e):
            raise CalculationFailed(_NOT_FINITE)
        return e, w

    def _evaluate(
        self, potential: Any, sub: tuple, energy: bool, virial: bool, heat: bool
    ) -> None:
        """A potential on the device computes the atoms."""
        if heat:
            potential.compute_heat(self._atoms, self._neighbors, *sub)
        else:
            potential.compute(self._atoms, self._neighbors, *sub, energy, virial, False)

    def _drive(self, field: np.ndarray) -> None:
        self._atoms.drive(field)

    def _convect(self, field: np.ndarray, kick: float) -> None:
        self._atoms.convect(field, kick, _MVV2E)

    def _switched(
        self, sub: tuple, energy: bool, virial: bool
    ) -> tuple[float, np.ndarray]:
        """(1 - λ) times the potential plus λ times the other interaction of a
        Switch: the energy and the virial, the forces mixed on the device; the
        energies of the two kept."""
        atoms, weight = self._atoms, self.weight
        first = self._total(atoms.total_energy()) if energy else math.nan
        work = self._total(atoms.total_virial()) if virial else np.full((3, 3), np.nan)
        other, push = 0.0, np.zeros((3, 3))
        if self._target is None:  # the ideal gas
            atoms.mix_forces(weight, False)
        elif isinstance(self._target, Springs):
            atoms.stash_forces()
            other = self._total(atoms.springs(np.ascontiguousarray(self._h)))
            atoms.mix_forces(weight, True)
        else:
            atoms.stash_forces()
            if self._other_species is not None:
                atoms.set_species(self._rows_of(self._other_species))
            with_virial = virial and "virial" in self._target.implemented
            self._evaluate(self._other.potential, sub, energy, with_virial, False)
            self._finish_forces()
            other = self._total(atoms.total_energy()) if energy else math.nan
            if with_virial:
                push = self._total(atoms.total_virial())
            if self._other_species is not None:
                atoms.set_species(self._rows_of(self._species))
            atoms.mix_forces(weight, True)
        self.parts = (first, other)
        return (1.0 - weight) * first + weight * other, (
            1.0 - weight
        ) * work + weight * push

    def set_switch(self, target: Any) -> None:
        """The other interaction of a Switch, None for none, or False for no
        switch; a calculator on the GPU, or Springs."""
        if isinstance(self._target, Springs):
            self._atoms.clear_springs()
        self._sites = None
        if self._other is not None and self._other.cutoff > self._cutoff:
            self._set_neighbors(self._cutoff)
        self._target, self._other, self._other_species = target, None, None
        self.weight = 0.0
        self.parts = (math.nan, math.nan)
        if target is False or target is None:
            return
        if isinstance(target, Springs):
            self._sites = (
                np.asarray(target.sites, dtype=np.float64),
                np.asarray(target.constants, dtype=np.float64),
            )
            self._set_springs()
            return
        on = getattr(target, "_on_device", None)
        if on is None:
            self._target = False
            raise InvalidParameter(
                "the target of a Switch on the GPU must compute on the GPU too: "
                f'{type(target).__name__} with device="cuda"'
            )
        self._other = on
        species = np.ascontiguousarray(target._species(self._system))
        if not np.array_equal(species, self._species):
            self._other_species = species
        if on.cutoff > self._cutoff:  # one list for both, as each sorts the atoms
            self._set_neighbors(on.cutoff)

    def _set_neighbors(self, cutoff: float) -> None:
        """A neighbor list of another cutoff, built at the next computation."""
        self._neighbors = self._module.Neighbors(cutoff, self._skin)

    def _set_springs(self) -> None:
        sites, constants = self._sites
        self._atoms.set_springs(
            np.ascontiguousarray(self._rows_of(sites)),
            np.ascontiguousarray(self._rows_of(constants)),
        )

    def set_field(
        self, field: np.ndarray | None, convective: bool = False, kick: float = 0.0
    ) -> None:
        """The driving field of HNEMD in the frame, or None for none;
        convective: with the energies each atom carries, at the velocities
        kick times the force over the mass on."""
        self._field = (
            None if field is None else np.ascontiguousarray(field, dtype=np.float64)
        )
        self._convective, self._kick = bool(convective), float(kick)

    def heat_flux(self) -> np.ndarray:
        """(3,) the heat flux times the volume in the frame, of the energies
        and virials of the last computation and the velocities now: its
        potential part, and with the energies each atom carries that too."""
        flux = np.asarray(self._atoms.heat_flux(), dtype=float)
        if self._convective:
            flux = flux + np.asarray(self._atoms.energy_flux(_MVV2E), dtype=float)
        return self._total(flux)

    def kick(self, c: float) -> None:
        self._atoms.kick(c)

    def drift(self, dt: float) -> None:
        self._atoms.drift(dt)

    def kick_drift(self, c: float, dt: float) -> None:
        self._atoms.kick_drift(c, dt)

    indenters: tuple = ()  # (kind, axis, sign, k, radius) of each indenter
    indenter_at: tuple = ()  # their centers or planes at the step
    indented: list = []  # (energy, force on it) of each, of the last computation

    def speed_squares(self) -> float:
        """The sum of the squares of the velocities."""
        return float(self._total(self._atoms.speed_squares()))

    def msst_kick(
        self,
        half: float,
        base: float,
        axial: float,
        axis: int,
        trial: bool,
        ftm2v: float,
    ) -> float:
        """The velocities of fix msst half a step on, kept unless trial; the
        sum of their squares."""
        return float(
            self._total(
                self._atoms.msst_kick(
                    float(half),
                    float(base),
                    float(axial),
                    int(axis),
                    bool(trial),
                    ftm2v,
                )
            )
        )

    def reflect(self, normal: np.ndarray, offset: float, speed: float) -> None:
        """A flat wall: each atom with x . normal < offset put back by as
        much, its velocity along the normal made 2 speed - v . normal."""
        self._atoms.reflect(
            np.ascontiguousarray(normal, dtype=np.float64), float(offset), float(speed)
        )

    def scale(self, factor: Any) -> None:
        f = np.broadcast_to(np.asarray(factor, dtype=np.float64), (3,))
        self._atoms.scale(float(f[0]), float(f[1]), float(f[2]))

    def shear(self, half: float, w: np.ndarray) -> None:
        self._atoms.shear(half, float(w[3]), float(w[4]), float(w[5]))

    def restrict(self) -> None:
        self._atoms.restrict_velocities()

    def _set_free(self, free: np.ndarray | None) -> None:
        self._free = free
        if free is not None:
            self._atoms.set_movable(
                np.ascontiguousarray(self._rows_of(free), dtype=bool), self._rotation
            )
        else:
            self._atoms.clear_movable()

    def set_controls(
        self,
        free: np.ndarray | None,
        prescribed: np.ndarray | None,
        added: np.ndarray | None,
        watched: tuple,
        baths: tuple,
    ) -> None:
        """The controls of a run, as _HostAtoms.set_controls takes them: the
        groups whose forces are summed and the baths as bits of a word of
        each atom."""
        self._prescribed, self._added = prescribed, added
        self._masks = [*watched, *(mask for mask, _ in baths)]
        if len(self._masks) > 31:
            raise InvalidParameter("the GPU takes at most 31 groups moved or bathed")
        self._watched = len(watched)
        self._baths = [
            (len(watched) + k, int(seed) % (1 << 63), [0])
            for k, (_, seed) in enumerate(baths)
        ]
        self._set_free(free)
        self._set_run_arrays()
        self.sums = []

    def _set_run_arrays(self) -> None:
        """The prescribed velocities, the forces added and the groups of the
        run on the device."""
        atoms = self._atoms
        if self._prescribed is None:
            atoms.clear_prescribed()
        else:
            atoms.set_prescribed(
                np.ascontiguousarray(self._rows_of(self._prescribed), dtype=np.float64)
            )
        if self._added is None:
            atoms.clear_added()
        else:
            atoms.set_added(
                np.ascontiguousarray(self._rows_of(self._added), dtype=np.float64)
            )
        self._set_groups()

    def _set_groups(self) -> None:
        """The bits of the groups of the controls on the device, and after
        them that of the atoms an electronic stopping may slow."""
        masks = list(self._masks)
        if self._stopping is not None and self._stopping[3] >= 0:
            masks.append(self._stopping_mask)
        if masks:
            bits = np.zeros(len(masks[0]), dtype=np.int32)
            for k, mask in enumerate(masks):
                bits[mask] |= np.int32(1 << k)
            self._atoms.set_groups(self._rows_of(bits))
        else:
            self._atoms.clear_groups()

    def set_stopping(self, spec: Any) -> None:
        """The electronic stopping of a run, as _HostAtoms.set_stopping takes
        it, or None: the table of each species of the potential that of the
        element of its atoms."""
        had = self._stopping is not None
        self._stopper = self._stopping = None
        self.electronic_power = 0.0
        if spec is None:
            if had:
                self._set_groups()
            return
        energies, values, kinds, least, needed, mask, cutoff = spec
        species = self._species
        table = np.zeros((int(species.max(initial=0)) + 1, len(energies)))
        feel = np.ones(len(species), dtype=bool) if mask is None else mask
        # MDAPY's potentials give each element its own species
        first = np.unique(species[feel], return_index=True)
        table[first[0]] = values[kinds[np.flatnonzero(feel)[first[1]]]]
        self._stopper = self._module.Stopping(
            np.ascontiguousarray(energies, dtype=np.float64),
            np.ascontiguousarray(table),
        )
        if mask is not None and len(self._masks) > 30:
            raise InvalidParameter(
                "the GPU takes at most 31 groups moved, bathed or stopped"
            )
        self._stopping_mask = mask
        bit = -1 if mask is None else len(self._masks)
        self._stopping = (float(least), int(needed), float(cutoff), bit,
                          float(energies[-1]))  # fmt: skip
        self._set_groups()

    def clear_controls(self) -> None:
        self._prescribed = self._added = None
        self._masks = []
        self._watched = 0
        self._baths = []
        self.sums = []
        self._set_free(self._movable)
        self._set_run_arrays()

    def twice_kinetic_free(self) -> float:
        return float(self._total(self._atoms.kinetic_free())) * _MVV2E

    def bath(self, k: int, c1: float, c2: float, kt: float) -> float:
        """Bath k for half a step: the kinetic energy it added, in eV."""
        bit, seed, counter = self._baths[k]
        heat = self._atoms.bath(bit, c1, c2, kt, seed, counter[0])
        counter[0] += 1
        return float(self._total(heat)) * _MVV2E

    def remap(
        self, old_lo: np.ndarray, old: np.ndarray, lo: np.ndarray, new: np.ndarray
    ) -> None:
        self._atoms.remap(
            np.ascontiguousarray(old_lo, dtype=np.float64),
            np.ascontiguousarray(np.linalg.inv(old)),
            np.ascontiguousarray(lo, dtype=np.float64),
            np.ascontiguousarray(new, dtype=np.float64),
        )
        self._h = np.array(new, dtype=np.float64)

    def seed(self, seed: int | None) -> None:
        self._seed = self._agreed(secrets.randbits(63) if seed is None else int(seed))
        self._counter = 0

    def langevin(self, c1: float, c2: float, kt: float, keep: bool = False) -> None:
        """v = c1 v + c2 sqrt(kt / m) g, g Gaussian of the seed and a counter,
        with the noise of held directions removed; with keep, less an equal
        share of the momentum of the noise."""
        self._atoms.langevin(c1, c2, kt, self._seed, self._counter, keep)
        self._counter += 1
        if self._free is not None:
            self._atoms.restrict_velocities()
