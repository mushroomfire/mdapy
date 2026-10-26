# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Conversion between System and the data collections of OVITO."""

import re
from typing import Any

import numpy as np
import polars as pl

from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.core.topology import KINDS, Topology
from mdapy.elements import SYMBOLS
from mdapy.io._lammps_data import _atom_types

# Columns of a System and the standard particle properties of OVITO that
# hold the same values.
_STANDARD = {
    "velocities": "Velocity",
    "forces": "Force",
    "mass": "Mass",
    "charge": "Charge",
    "id": "Particle Identifier",
    "molecule": "Molecule Identifier",
    "image": "Periodic Image",
    "atomic_energies": "Potential Energy",
    "structure": "Structure Type",
    "csp": "Centrosymmetry",
    "coordination": "Coordination",
    "cluster": "Cluster",
    "orientation": "Orientation",
    "rmsd": "RMSD",
    "interatomic_distance": "Interatomic Distance",
    "shear_strain": "Shear Strain",
    "volumetric_strain": "Volumetric Strain",
    "d2min": "Nonaffine Squared Displacement",
    "voronoi_volume": "Atomic Volume",
}
_COLUMN_OF = {ovito: name for name, ovito in _STANDARD.items()}
# The containers of the topology terms in OVITO and their type properties.
_TERMS = {
    "bonds": ("bonds", "create_bonds", "Bond Type"),
    "angles": ("angles", "create_angles", "Angle Type"),
    "dihedrals": ("dihedrals", "create_dihedrals", "Dihedral Type"),
    "impropers": ("impropers", "create_impropers", "Improper Type"),
}


def _ovito():
    try:
        from ovito.data import BondType, DataCollection, ElementType, ParticleType
    except ImportError:  # pragma: no cover - OVITO is installed for the tests
        raise ImportError("converting to and from OVITO needs OVITO") from None
    # OVITO has a class of types for bonds only; other terms take the base
    term_types = {
        "bonds": BondType,
        "angles": ElementType,
        "dihedrals": ElementType,
        "impropers": ElementType,
    }
    return DataCollection, ParticleType, term_types


def _column_name(name: str) -> str:
    """The column of a property of OVITO that is not a standard column."""
    return _COLUMN_OF.get(name) or re.sub(r"\W+", "_", name).strip("_").lower()


def _type_names(table: pl.DataFrame, types: np.ndarray) -> dict[int, str]:
    """The type label of each type that has a single one."""
    if "type_label" not in table.columns:
        return {}
    labels = table["type_label"].to_numpy()
    names = {}
    for t in np.unique(types):
        found = {str(s) for s in labels[types == t] if s is not None}
        if len(found) == 1:
            names[int(t)] = found.pop()
    return names


def _bond_images(system: System, atoms: np.ndarray) -> np.ndarray:
    """The periodic image of the second atom of each bond, as OVITO keeps it:
    the nearest image of the second atom."""
    box = system.box
    fractional = box.to_fractional(system["positions"])
    delta = fractional[atoms[:, 1]] - fractional[atoms[:, 0]]
    return np.where(box.pbc, -np.round(delta), 0.0).astype(np.int32)


def to_ovito(system: System) -> Any:
    """The system as an OVITO data collection.

    Parameters
    ----------
    system : System
        The atoms. The box becomes the simulation cell; elements, or the
        ``type`` column with the elements or type labels as names, become
        the particle types; columns with a standard property of OVITO, such
        as ``velocities``, ``id`` or ``structure``, become that property and
        other numeric columns user properties of the same name; the
        topology becomes bonds, angles, dihedrals and impropers; ``info``
        becomes the global attributes.

    Returns
    -------
    ovito.data.DataCollection
        A new data collection; nothing is shared with the system.
    """
    DataCollection, ParticleType, term_types = _ovito()
    box, table, n = system.box, system.table, len(system)
    data = DataCollection()
    cell = np.zeros((3, 4))
    cell[:, :3] = box.matrix.T  # OVITO keeps the cell vectors as columns
    cell[:, 3] = box.origin
    data.create_cell(cell, pbc=tuple(bool(p) for p in box.pbc))
    particles = data.create_particles(count=n)
    particles.create_property("Position", data=system["positions"])
    if "type" in table.columns or "element" in table.columns:
        types, named = _atom_types(table, None)
        named = _type_names(table, types) | named
        prop = particles.create_property("Particle Type", data=types.astype(np.int32))
        for t in np.unique(types):
            particle_type = ParticleType(id=int(t), name=named.get(int(t), ""))
            if int(t) in named:
                particle_type.load_defaults()
            prop.types.append(particle_type)
    for name in table.columns:
        if name in ("positions", "element", "type", "type_label"):
            continue
        values = table[name].to_numpy()
        if values.dtype == bool:
            values = values.astype(np.int8)
        if values.dtype.kind in "iuf":
            particles.create_property(_STANDARD.get(name, name), data=values.copy())
    if system.topology is not None:
        for kind, df in system.topology.tables().items():
            if not len(df):
                continue
            _, create, type_name = _TERMS[kind]
            terms = getattr(particles, create)(count=len(df))
            atoms = df["atoms"].to_numpy().reshape(len(df), KINDS[kind])
            terms.create_property("Topology", data=atoms)
            types = df["type"].to_numpy()
            prop = terms.create_property(type_name, data=types)
            names = _type_names(df, types)
            for t in np.unique(types):
                prop.types.append(
                    term_types[kind](id=int(t), name=names.get(int(t), ""))
                )
            if kind == "bonds":
                terms.create_property(
                    "Periodic Image", data=_bond_images(system, atoms)
                )
    for key, value in system.info.items():
        data.attributes[key] = (
            np.array(value) if isinstance(value, np.ndarray) else value
        )
    return data


def _types(prop: Any) -> tuple[np.ndarray, dict[int, str]]:
    """The type ids of a typed property and the names of the named types."""
    return np.asarray(prop).astype(np.int64), {
        t.id: t.name for t in prop.types if t.name
    }


def _terms(container: Any, kind: str) -> pl.DataFrame | None:
    _, _, type_name = _TERMS[kind]
    if container is None or not container.count:
        return None
    width = KINDS[kind]
    atoms = np.asarray(container["Topology"]).astype(np.int64).reshape(-1, width)
    columns = {"atoms": pl.Series(atoms, dtype=pl.Array(pl.Int64, width))}
    if type_name in container:
        types, names = _types(container[type_name])
        columns["type"] = pl.Series(types, dtype=pl.Int32)
        if names:
            columns["type_label"] = pl.Series(
                [names.get(t) for t in types], dtype=pl.String
            )
    return pl.DataFrame(columns)


def from_ovito(data: Any) -> System:
    """An OVITO data collection as a System.

    Parameters
    ----------
    data : ovito.data.DataCollection
        The particles, for example the output of a pipeline,
        ``pipeline.compute()``. The simulation cell becomes the box (a box
        enclosing the particles when there is none); particle types become
        the ``type`` column, with ``element`` where their names are chemical
        symbols and ``type_label`` where they have other names; standard
        properties with a column of mdapy, such as ``Velocity``,
        ``Particle Identifier`` or ``Structure Type``, become that column,
        and other properties columns of their names in lowercase with
        underscores; bonds, angles, dihedrals and impropers become the
        topology; global attributes become ``info``.

    Returns
    -------
    System
        A new system; nothing is shared with the data collection.
    """
    particles = data.particles
    positions = np.array(particles.positions, dtype=np.float64)
    box = None
    if data.cell is not None:
        cell = np.array(data.cell)
        box = Box(cell[:, :3].T, origin=cell[:, 3], pbc=data.cell.pbc)
    columns: dict[str, Any] = {}
    for name in particles.keys():
        prop = particles[name]
        if name == "Position" or prop.dtype.kind not in "iuf":
            continue
        if name == "Particle Type":
            types, names = _types(prop)
            columns["type"] = types
            symbols = [names.get(t) for t in types]
            if names and all(s in SYMBOLS for s in symbols):
                columns["element"] = pl.Series(symbols, dtype=pl.String)
            elif names:
                columns["type_label"] = pl.Series(symbols, dtype=pl.String)
            continue
        column = _column_name(name)
        values = np.array(prop)
        columns[column] = values.astype(bool) if column == "movable" else values
    terms = {kind: _terms(getattr(particles, kind), kind) for kind in KINDS}
    topology = Topology(**terms) if any(t is not None for t in terms.values()) else None
    info = dict(data.attributes)
    system = System(positions, box, info=info, topology=topology)
    return system.with_columns(**columns) if columns else system
