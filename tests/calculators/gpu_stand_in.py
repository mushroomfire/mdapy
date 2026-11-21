"""A stand-in for mdapy._cuda that computes on the processor with NumPy and
the kernels of mdapy._core: what calls the GPU module can be tested where
there is no GPU. tests/cuda compares the GPU itself with the processor. Its
noise is that of the device and of the processor, of the counter-based
generator by the index of each atom (core/philox.h); under MPI the steps and
the sums go through the first owned atoms, those of the process."""

import types

import numpy as np

from mdapy import _core
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.neighbor import NeighborList


class NotFinite(RuntimeError):
    pass


# what each atom carries from process to process, as the device packs it
_CARRIED = np.dtype([("x", "f8", 3), ("species", "i4"), ("image", "i4", 3),
                     ("mass", "f8"), ("id", "i8"), ("v", "f8", 3), ("f", "f8", 3)])  # fmt: skip


class Atoms:
    """The atoms, as mdapy._cuda.Atoms holds them, in NumPy arrays."""

    def __init__(self, n):
        self.size = n
        self.species = self.x = self.v = self.masses = self.result = None
        self.m = n  # the atoms of the process, the first
        self.index = np.arange(n, dtype=np.int64)
        self.image = np.zeros((n, 3), dtype=np.int32)  # the boxes wrap moved them by

    def set_owned(self, owned):
        self.m = owned if owned > 0 else self.size

    @property
    def owned(self):
        return self.m

    def set_ids(self, ids):
        self.index = np.array(ids, dtype=np.int64)

    def clear_ids(self):
        self.index = np.arange(self.size, dtype=np.int64)

    def set_species(self, species):
        self.species = np.array(species)

    def set_masses(self, masses):
        self.masses = np.array(masses)

    def set_positions(self, positions):
        self.x = np.array(positions)
        self.image = np.zeros((self.size, 3), dtype=np.int32)

    def set_velocities(self, velocities):
        self.v = np.array(velocities)

    def positions(self):
        return self.x.copy()

    def images(self):  # the stand-in wraps only under MPI, at a build
        return self.image.copy()

    # -- a build of the bricks under MPI, rows the caller's order ------------

    def ids(self):
        return self.index.copy()

    def settle(self):
        pass

    def __renumbered(self, n, forces):
        self.size = self.m = n
        self.movable = self.prescribed = self.added = self.groups = None
        self.sites = self.constants = None
        self.result = (np.zeros(n), forces, np.zeros((n, 3, 3)))

    def keep_rows(self, rows):
        forces = self.forces()[rows]
        self.x, self.v = self.x[rows], self.v[rows]
        self.species, self.masses = self.species[rows], self.masses[rows]
        self.index, self.image = self.index[rows], self.image[rows]
        self.__renumbered(len(rows), forces)

    @staticmethod
    def row_bytes():
        return _CARRIED.itemsize

    def pack_rows(self, rows):
        out = np.empty(len(rows), dtype=_CARRIED)
        out["x"], out["species"], out["image"] = self.x[rows], self.species[rows], self.image[rows]
        out["mass"], out["id"], out["v"] = self.masses[rows], self.index[rows], self.v[rows]
        out["f"] = self.forces()[rows]
        return out.view(np.uint8)

    def __appended(self, x, species, image, masses, ids, v, f):
        self.x = np.concatenate([self.x, x])
        self.species = np.concatenate([self.species, species])
        self.image = np.concatenate([self.image, image])
        self.masses = np.concatenate([self.masses, masses])
        self.index = np.concatenate([self.index, ids])
        self.v = np.concatenate([self.v, v])
        self.__renumbered(len(self.x), np.concatenate([self.forces(), f]))

    def append_rows(self, data, count):
        rows = np.asarray(data, dtype=np.uint8).view(_CARRIED)
        assert len(rows) == count
        self.__appended(rows["x"], rows["species"], rows["image"], rows["mass"],
                        rows["id"], rows["v"], rows["f"])  # fmt: skip

    def append_ghosts(self, positions, species, ids):
        n = len(positions)
        zeros = np.zeros((n, 3))
        self.__appended(np.array(positions), np.array(species), np.zeros((n, 3), np.int32),
                        np.zeros(n), np.array(ids), zeros, zeros)  # fmt: skip

    def wrap(self, h, lo, pbc):
        m = self.m
        s = (self.x[:m] - lo) @ np.linalg.inv(h)
        whole = np.where(np.array(pbc) & np.isfinite(s), np.floor(s), 0.0)
        self.x[:m] -= whole @ h
        self.image[:m] += whole.astype(np.int32)

    def fractions(self, d, h, lo, begin, end):
        return ((self.x[begin:end] - lo) @ np.linalg.inv(h))[:, d]

    def gather_positions(self, rows, shift):
        return self.x[rows] + shift

    def velocities(self):
        return self.v.copy()

    def forces(self):
        if self.result is None:  # zero on a new device, as there
            return np.zeros((self.size, 3))
        return self.result[1]

    def set_forces(self, forces):
        if self.result is None:
            self.result = (np.zeros(self.size), None, np.zeros((self.size, 3, 3)))
        energies, _, virials = self.result
        self.result = (energies, np.array(forces, dtype=float), virials)

    def energies(self):
        return self.result[0]

    def virials(self):
        return self.result[2]

    def total_energy(self):
        return float(self.result[0][: self.m].sum())

    def total_virial(self):
        return self.result[2][: self.m].sum(axis=0)

    def kick(self, c):
        m = self.m
        self.v[:m] += c * self.result[1][:m] / self.masses[:m, None]

    def drift(self, dt):
        m = self.m
        self.x[:m] += dt * self.v[:m]

    def kick_drift(self, c, dt):
        self.kick(c)
        self.drift(dt)

    def indent(self, kind, axis, sign, k, radius, at, h, pbc):
        from mdapy.simulation._integrator import _indented

        where = at[0] if kind == 2 else np.asarray(at)
        push, energy, on = _indented(
            self.x[: self.m], np.asarray(h), pbc, (kind, axis, sign, k, radius), where
        )
        energies, forces, virials = self.result
        forces = np.array(forces)
        forces[: self.m] += push
        self.result = (energies, forces, virials)
        return np.array([energy, *on])

    def speed_squares(self):
        v = self.v[: self.m]
        return float((v * v).sum())

    def msst_kick(self, half, base, axial, axis, trial, ftm2v):
        n = self.m
        m = self.masses[:n, None]
        c = self.result[1][:n] * ftm2v / m
        d = np.repeat(base / m, 3, axis=1)
        d[:, axis] -= axial
        v = self.v[:n]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            e = np.exp(d * half)
            exact = e * (c + d * v - c / e) / d
        small = v + (c + d * v) * half + 0.5 * (d * d * v + c * d) * half * half
        new = np.where(np.abs(half * d) > 1.0e-06, exact, small)
        if not trial:
            self.v[:n] = new
        return float((new * new).sum())

    def reflect(self, normal, offset, speed):
        m = self.m
        x, v = self.x[:m], self.v[:m]
        depth = x @ normal - offset
        past = depth < 0.0
        x[past] -= 2.0 * depth[past, None] * normal
        v[past] += 2.0 * (speed - v[past] @ normal)[:, None] * normal

    def scale(self, fx, fy, fz):
        self.v[: self.m] *= np.array([fx, fy, fz])

    def shear(self, half, yz, xz, xy):
        v = self.v[: self.m]
        v[:, 0] -= half * (v[:, 1] * xy + v[:, 2] * xz)
        v[:, 1] -= half * v[:, 2] * yz

    def remap(self, old_lo, old_inverse, lo, h):
        m = self.m
        self.x[:m] = lo + np.einsum(
            "ij,jk->ik", np.einsum("ij,jk->ik", self.x[:m] - old_lo, old_inverse), h
        )

    def __noise(self, c2, kt, seed, counter):
        m = self.m
        g = _core.core.normals(self.index[:m], seed, 0, counter)
        return c2 * np.sqrt(kt / self.masses[:m])[:, None] * g

    def langevin(self, c1, c2, kt, seed, counter, keep=False):
        m = self.m
        noise = self.__noise(c2, kt, seed, counter)
        if keep:  # less an equal share of the momentum of the noise
            noise -= (self.masses[:m] @ noise) / m / self.masses[:m, None]
        self.v[:m] = c1 * self.v[:m] + noise

    def draw_noise(self, c2, kt, seed, counter):
        self.noise = self.__noise(c2, kt, seed, counter)
        return self.masses[: self.m] @ self.noise

    def keep_noise(self, c1, share):
        m = self.m
        self.v[:m] = c1 * self.v[:m] + self.noise - share / self.masses[:m, None]

    stash = sites = constants = None

    def stash_forces(self):
        self.stash = self.result[1].copy()

    def mix_forces(self, weight, other):
        energies, forces, virials = self.result
        mixed = (
            (1 - weight) * self.stash + weight * forces
            if other
            else (1 - weight) * forces
        )
        self.result = (energies, mixed, virials)

    def set_springs(self, sites, constants):
        self.sites, self.constants = np.array(sites), np.array(constants)

    def clear_springs(self):
        self.sites = self.constants = None

    def springs(self, h):
        d = self.x - self.sites  # the stand-in never wraps
        energies = 0.5 * self.constants * (d * d).sum(axis=1)
        self.result = (energies, -self.constants[:, None] * d, self.result[2])
        return float(energies[: self.m].sum())

    def push(self, field):
        self.pushes = np.einsum("a,iab->ib", field, self.result[2][: self.m])
        return self.pushes.sum(axis=0)

    def pushed(self, mean):
        energies, forces, virials = self.result
        forces = np.array(forces)
        forces[: self.m] += self.pushes - mean
        self.result = (energies, forces, virials)

    def drive(self, field):
        self.pushed(self.push(field) / self.m)

    def heat_flux(self):
        m = self.m
        return np.einsum("iab,ib->a", self.result[2][:m], self.v[:m])

    def __energies_with(self, v, mvv2e):
        m = self.m
        return self.result[0][:m] + 0.5 * self.masses[:m] * (v * v).sum(axis=1) * mvv2e

    def carried(self, kick, mvv2e):
        m = self.m
        ahead = self.v[:m] + kick * self.result[1][:m] / self.masses[:m, None]
        self.energies_carried = self.__energies_with(ahead, mvv2e)
        return float(self.energies_carried.sum())

    def convected(self, field, mean):
        energies, forces, virials = self.result
        forces = np.array(forces)
        forces[: self.m] += np.outer(self.energies_carried - mean, field)
        self.result = (energies, forces, virials)

    def convect(self, field, kick, mvv2e):
        self.convected(field, self.carried(kick, mvv2e) / self.m)

    def energy_flux(self, mvv2e):
        v = self.v[: self.m]
        return self.__energies_with(v, mvv2e) @ v

    def kinetic(self):
        m = self.m
        return np.einsum("i,ij,ik->jk", self.masses[:m], self.v[:m], self.v[:m])

    movable = rotation = prescribed = added = groups = None

    def set_movable(self, movable, rotation):
        self.movable, self.rotation = np.array(movable), np.array(rotation)

    def clear_movable(self):
        self.movable = None

    def __restricted(self, vectors):
        """Those of the first atoms, of the process."""
        if self.movable is None:
            return vectors
        r = self.rotation
        return ((vectors @ r.T) * self.movable[: len(vectors)]) @ r

    def restrict_forces(self):
        e, f, w = self.result
        f = np.array(f)
        f[: self.m] = self.__restricted(f[: self.m])
        self.result = (e, f, w)

    def restrict_velocities(self):
        m = self.m
        self.v[:m] = self.__restricted(self.v[:m])
        if self.prescribed is not None:
            self.v[:m] = self.v[:m] + self.prescribed[:m]

    def set_prescribed(self, velocities):
        self.prescribed = np.array(velocities)

    def clear_prescribed(self):
        self.prescribed = None

    def set_added(self, forces):
        self.added = np.array(forces)

    def clear_added(self):
        self.added = None

    def add_forces(self):
        if self.added is not None:
            e, f, w = self.result
            self.result = (e, f + self.added, w)

    def set_groups(self, bits):
        self.groups = np.array(bits)

    def clear_groups(self):
        self.groups = None

    def group_forces(self, count):
        f, groups = self.result[1][: self.m], self.groups[: self.m]
        return np.array([f[(groups >> g) & 1 == 1].sum(axis=0) for g in range(count)])

    def kinetic_free(self):
        m = self.m
        v = self.__restricted(self.v[:m])
        return float(np.einsum("i,ij,ij->", self.masses[:m], v, v))

    def bath(self, bit, c1, c2, kt, seed, counter):
        n = self.m
        v = self.v[:n]
        mask = (self.groups[:n] >> bit) & 1 == 1
        g = np.zeros_like(v)
        g[mask] = _core.core.normals(self.index[:n][mask], seed, 1, counter)
        g = self.__restricted(g)
        m = self.masses[:n][mask]
        before = np.einsum("i,ij,ij->", m, v[mask], v[mask])
        v[mask] = c1 * v[mask] + c2 * np.sqrt(kt / m)[:, None] * g[mask]
        return 0.5 * float(np.einsum("i,ij,ij->", m, v[mask], v[mask]) - before)


class Neighbors:
    def __init__(self, cutoff, skin):
        self.cutoff, self.skin = cutoff, skin
        self.built = None  # the positions of the last build

    def update(self, atoms, h, lo, pbc):
        self.build(atoms, h, lo, pbc)
        return True

    def stale(self, atoms, h, lo, pbc):
        if not np.isfinite(atoms.x).all():
            raise NotFinite("the positions are not finite")
        if self.built is None or len(self.built) != atoms.size:
            return True
        moved = np.sqrt(((atoms.x - self.built) ** 2).sum(axis=1).max(initial=0.0))
        return bool(2.0 * moved > self.skin)

    def build(self, atoms, h, lo, pbc):
        if not np.isfinite(atoms.x).all():
            raise NotFinite("the positions are not finite")
        thick = np.abs(np.linalg.det(h)) / np.linalg.norm(
            np.cross(h[[1, 2, 0]], h[[2, 0, 1]]), axis=1
        )
        if (np.array(pbc) & (thick <= 2 * (self.cutoff + self.skin))).any():
            raise ValueError("the box is too thin: thicker than twice the cutoff")
        self.built = atoms.x.copy()


class Exchange:
    """The messages of the borders of the bricks, as the device's: the
    stand-in keeps its atoms in the caller's order."""

    def __init__(self):
        self.messages = {}

    def set(self, atoms, k, rows, start, received):
        self.messages[k] = (np.array(rows, dtype=np.int64), int(start), int(received))

    def clear(self):
        self.messages = {}

    def sent(self, k):
        return len(self.messages[k][0])

    def received(self, k):
        return self.messages[k][2]

    def pack_positions(self, atoms, k, shift):
        return atoms.x[self.messages[k][0]] + shift

    def unpack_positions(self, atoms, k, positions):
        _, start, count = self.messages[k]
        atoms.x[start : start + count] = positions

    def pack_forces(self, atoms, k):
        _, start, count = self.messages[k]
        return np.array(atoms.result[1][start : start + count])

    def add_forces(self, atoms, k, forces):
        e, f, w = atoms.result
        f = np.array(f)
        f[self.messages[k][0]] += forces
        atoms.result = (e, f, w)


class Stopping:
    """Electronic stopping: atom by atom, as the kernel of the device."""

    def __init__(self, energies, values):
        self.energies, self.values = np.array(energies), np.array(values)

    def apply(self, atoms, neighbors, h, lo, pbc, least, needed, cutoff, bit, mvv2e):
        system = System(atoms.x, Box(h, origin=lo, pbc=pbc))
        nl = NeighborList.build(system, cutoff)
        energies, forces, virials = atoms.result
        forces = np.array(forces)
        power = beyond = 0.0
        table = self.energies
        for i in range(atoms.m):
            if bit >= 0 and not (atoms.groups[i] >> bit) & 1:
                continue
            v = atoms.v[i]
            e = 0.5 * mvv2e * atoms.masses[i] * float(v @ v)
            if e < least or e < table[0] or nl.counts[i] < needed:
                continue
            beyond += e > table[-1]
            up = min(max(int(np.searchsorted(table, e)), 1), len(table) - 1)
            low, high = self.values[atoms.species[i], [up - 1, up]]
            drag = low + (high - low) * (e - table[up - 1]) / (
                table[up] - table[up - 1]
            )
            speed = float(np.sqrt(v @ v))
            forces[i] -= drag * v / speed
            power += drag * speed
        atoms.result = (energies, forces, virials)
        return np.array([power, beyond])


class Potential:
    """A potential of the stand-in: the kernels of the processor."""

    def __init__(self, made, compute):
        self.native, self.compute_native = made, compute
        self.asked = []

    def compute(self, atoms, neighbors, h, lo, pbc, energy, virial, atomic, heat=False):
        self.asked.append((energy, virial, atomic))
        system = System(atoms.x, Box(h, origin=lo, pbc=pbc))
        nl = NeighborList.build(system, neighbors.cutoff)
        # the kernels of three-body terms take heat, the others need none
        extra = (heat,) if heat else ()
        atoms.result = self.compute_native(
            self.native, atoms.species, np.ascontiguousarray(atoms.x), h,
            nl.offsets, nl.j, nl.shifts, None, None, *extra,
        )  # fmt: skip

    def compute_heat(self, atoms, neighbors, h, lo, pbc):
        """The energies and virials of the heat flux: those of the kernels
        with heat for EAM and the three-body terms, as the device's
        compute_heat."""
        heat = self.compute_native in (
            _core.calculators.compute_stillinger_weber,
            _core.calculators.compute_tersoff,
            _core.calculators.compute_eam,
        )
        self.compute(atoms, neighbors, h, lo, pbc, True, True, True, heat)


class Overlay:
    """Potentials of the stand-in added together, as the device's."""

    def __init__(self, parts):
        self.parts = list(parts)

    def __run(self, atoms, compute):
        total = None
        for part in self.parts:
            compute(part)
            if total is None:
                total = atoms.result
            else:
                total = tuple(
                    None if a is None or b is None else a + b
                    for a, b in zip(total, atoms.result)
                )
        atoms.result = total

    def compute(self, atoms, neighbors, h, lo, pbc, energy, virial, atomic, heat=False):
        self.__run(atoms, lambda p: p.compute(atoms, neighbors, h, lo, pbc, energy,
                                              virial, atomic))  # fmt: skip

    def compute_heat(self, atoms, neighbors, h, lo, pbc):
        self.__run(atoms, lambda p: p.compute_heat(atoms, neighbors, h, lo, pbc))


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

    def device_properties():
        return {
            "name": device_name(),
            "memory": 12 * 2**30,
            "compute_capability": (8, 6),
            "multiprocessors": 26,
        }

    return types.SimpleNamespace(
        Atoms=Atoms, Neighbors=Neighbors, Stopping=Stopping, Exchange=Exchange,
        make_eam=make_eam,
        make_overlay=Overlay,
        make_pair=make_pair,
        make_adp=make_adp, make_stillinger_weber=make_stillinger_weber,
        make_tersoff=make_tersoff, make_meam=make_meam, make_nep=make_nep,
        device_name=device_name, device_properties=device_properties,
        NotFinite=NotFinite, device_count=lambda: 1, use_device=lambda device: None,
    )  # fmt: skip
