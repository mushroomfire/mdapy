"""A stand-in for mdapy._cuda that computes on the processor with NumPy and
the kernels of mdapy._core: what calls the GPU module can be tested where
there is no GPU. tests/cuda compares the GPU itself with the processor."""

import types

import numpy as np

from mdapy import _core
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.neighbor import NeighborList


class NotFinite(RuntimeError):
    pass


class Atoms:
    """The atoms, as mdapy._cuda.Atoms holds them, in NumPy arrays."""

    def __init__(self, n):
        self.size = n
        self.species = self.x = self.v = self.masses = self.result = None

    def set_species(self, species):
        self.species = np.array(species)

    def set_masses(self, masses):
        self.masses = np.array(masses)

    def set_positions(self, positions):
        self.x = np.array(positions)

    def set_velocities(self, velocities):
        self.v = np.array(velocities)

    def positions(self):
        return self.x.copy()

    def images(self):  # the stand-in never wraps
        return np.zeros((self.size, 3), dtype=np.int32)

    def velocities(self):
        return self.v.copy()

    def forces(self):
        return self.result[1]

    def energies(self):
        return self.result[0]

    def virials(self):
        return self.result[2]

    def total_energy(self):
        return float(self.result[0].sum())

    def total_virial(self):
        return self.result[2].sum(axis=0)

    def kick(self, c):
        self.v += c * self.result[1] / self.masses[:, None]

    def drift(self, dt):
        self.x += dt * self.v

    def kick_drift(self, c, dt):
        self.kick(c)
        self.drift(dt)

    def scale(self, fx, fy, fz):
        self.v *= np.array([fx, fy, fz])

    def shear(self, half, yz, xz, xy):
        v = self.v
        v[:, 0] -= half * (v[:, 1] * xy + v[:, 2] * xz)
        v[:, 1] -= half * v[:, 2] * yz

    def remap(self, old_lo, old_inverse, lo, h):
        self.x = lo + np.einsum(
            "ij,jk->ik", np.einsum("ij,jk->ik", self.x - old_lo, old_inverse), h
        )

    def langevin(self, c1, c2, kt, seed, counter):
        g = np.random.default_rng([seed, counter]).normal(size=self.v.shape)
        self.v = c1 * self.v + c2 * np.sqrt(kt / self.masses)[:, None] * g

    def kinetic(self):
        return np.einsum("i,ij,ik->jk", self.masses, self.v, self.v)

    def set_movable(self, movable, rotation):
        self.movable, self.rotation = np.array(movable), np.array(rotation)

    def __restricted(self, vectors):
        r = self.rotation
        return ((vectors @ r.T) * self.movable) @ r

    def restrict_forces(self):
        e, f, w = self.result
        self.result = (e, self.__restricted(f), w)

    def restrict_velocities(self):
        self.v = self.__restricted(self.v)


class Neighbors:
    def __init__(self, cutoff, skin):
        self.cutoff, self.skin = cutoff, skin

    def update(self, atoms, h, lo, pbc):
        if not np.isfinite(atoms.x).all():
            raise NotFinite("the positions are not finite")
        thick = np.abs(np.linalg.det(h)) / np.linalg.norm(
            np.cross(h[[1, 2, 0]], h[[2, 0, 1]]), axis=1
        )
        if (np.array(pbc) & (thick <= 2 * (self.cutoff + self.skin))).any():
            raise ValueError("the box is too thin: thicker than twice the cutoff")
        return True


class Potential:
    """A potential of the stand-in: the kernels of the processor."""

    def __init__(self, made, compute):
        self.native, self.compute_native = made, compute
        self.asked = []

    def compute(self, atoms, neighbors, h, lo, pbc, energy, virial, atomic):
        self.asked.append((energy, virial, atomic))
        system = System(atoms.x, Box(h, origin=lo, pbc=pbc))
        nl = NeighborList.build(system, neighbors.cutoff)
        atoms.result = self.compute_native(
            self.native, atoms.species, np.ascontiguousarray(atoms.x), h,
            nl.offsets, nl.j, nl.shifts, None, None,
        )  # fmt: skip


def stand_in(found=True):
    """A module with the names of mdapy._cuda."""
    native = _core.calculators

    def make_eam(embedding, density, pair, drho, dr, cutoff, rho_max, precision):
        made = native.make_eam(
            embedding, density, pair, drho, dr, cutoff, rho_max=rho_max
        )
        return Potential(made, native.compute_eam)

    def make_adp(
        embedding, density, pair, dipole, quadrupole, drho, dr, cutoff, precision
    ):
        made = native.make_adp(
            embedding, density, pair, dipole, quadrupole, drho, dr, cutoff
        )
        return Potential(made, native.compute_adp)

    def make_stillinger_weber(elements, rows, precision):
        made = native.make_stillinger_weber(elements, rows)
        return Potential(made, native.compute_stillinger_weber)

    def make_tersoff(elements, rows, precision):
        made = native.make_tersoff(elements, rows)
        return Potential(made, native.compute_tersoff)

    def make_meam(element, ebound, cmin, cmax, tables, options, numbers, precision):
        made = native.make_meam(element, ebound, cmin, cmax, tables, options, numbers)
        return Potential(made, native.compute_meam)

    def make_nep(*arguments):
        if max(arguments[0][1:5]) + 1 > 20:  # functions of each kind
            raise ValueError("on the GPU, NEP takes at most 20 functions of each kind")
        made = native.make_nep(*arguments[:-1])
        return Potential(made, native.compute_nep)

    def make_pair(style, species, cutsq, offset, coefficients, precision):
        made = native.make_pair(style, cutsq, offset, coefficients)
        return Potential(made, native.compute_pair)

    def device_name():
        if not found:
            raise RuntimeError("no CUDA device: none was found")
        return "stand-in"

    return types.SimpleNamespace(
        Atoms=Atoms, Neighbors=Neighbors, make_eam=make_eam, make_pair=make_pair,
        make_adp=make_adp, make_stillinger_weber=make_stillinger_weber,
        make_tersoff=make_tersoff, make_meam=make_meam, make_nep=make_nep,
        device_name=device_name, NotFinite=NotFinite,
    )  # fmt: skip
