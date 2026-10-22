# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Molecular topology: bonds, angles, dihedrals and impropers."""

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from mdapy.errors import InvalidParameter

__all__ = ["Topology"]

KINDS = {"bonds": 2, "angles": 3, "dihedrals": 4, "impropers": 4}
"""Number of atoms in each kind of term."""


def _empty(width: int) -> pl.DataFrame:
    return pl.DataFrame(schema={"atoms": pl.Array(pl.Int64, width), "type": pl.Int32})


def _normalize(kind: str, terms: pl.DataFrame | ArrayLike | None) -> pl.DataFrame:
    width = KINDS[kind]
    if terms is None:
        return _empty(width)
    if isinstance(terms, pl.DataFrame):
        df = terms
        if "atoms" not in df.columns:
            raise InvalidParameter(f"{kind} table needs an 'atoms' column")
        if "type" not in df.columns:
            df = df.with_columns(pl.lit(1, dtype=pl.Int32).alias("type"))
    else:
        atoms = np.asarray(terms)
        if atoms.size == 0:
            atoms = atoms.reshape(0, width)
        if atoms.ndim != 2 or atoms.shape[1] != width:
            raise InvalidParameter(
                f"{kind} must be an (M, {width}) array of atom indices, "
                f"got shape {atoms.shape}"
            )
        df = pl.DataFrame(
            {
                "atoms": pl.Series(
                    atoms.astype(np.int64), dtype=pl.Array(pl.Int64, width)
                ),
                "type": pl.Series(np.ones(len(atoms), dtype=np.int32)),
            }
        )
    try:
        df = df.with_columns(
            pl.col("atoms").cast(pl.Array(pl.Int64, width)),
            pl.col("type").cast(pl.Int32),
        )
    except pl.exceptions.PolarsError as exc:
        raise InvalidParameter(
            f"{kind}: 'atoms' must hold {width} integers and 'type' integers ({exc})"
        ) from None
    if "type_label" in df.columns:
        df = df.with_columns(pl.col("type_label").cast(pl.String))
    if df["atoms"].null_count() or df["type"].null_count():
        raise InvalidParameter(f"{kind} must not contain missing values")
    atoms = df["atoms"].to_numpy()
    if atoms.size and atoms.min() < 0:
        raise InvalidParameter(f"{kind} contain negative atom indices")
    if len(atoms) and np.any(np.diff(np.sort(atoms, axis=1), axis=1) == 0):
        raise InvalidParameter(f"{kind} contain a term that repeats an atom")
    if len(df) and df["type"].min() < 1:
        raise InvalidParameter(f"{kind} types must be at least 1")
    return df.rechunk()


class Topology:
    """Bonds, angles, dihedrals and impropers of a molecular system.

    Each kind of term is a polars DataFrame with an ``atoms`` column of
    0-based atom indices (rows of the system's table), a ``type`` column of
    integer types starting at 1, an optional ``type_label`` column, and any
    further columns.

    Parameters
    ----------
    bonds, angles, dihedrals, impropers : polars.DataFrame or array_like, optional
        Terms with 2, 3, 4 and 4 atoms. An (M, k) integer array gives atom
        indices with all types equal to 1. Atom order follows LAMMPS: the
        central atom of an angle is the second one, dihedral atoms follow the
        chain, and improper order depends on the improper style.

    Raises
    ------
    InvalidParameter
        If a table has the wrong shape or types, negative indices, a term
        with a repeated atom, missing values, or a type below 1.
    """

    __slots__ = tuple(f"_{kind}" for kind in KINDS)

    def __init__(
        self,
        bonds: pl.DataFrame | ArrayLike | None = None,
        angles: pl.DataFrame | ArrayLike | None = None,
        dihedrals: pl.DataFrame | ArrayLike | None = None,
        impropers: pl.DataFrame | ArrayLike | None = None,
    ) -> None:
        self._bonds = _normalize("bonds", bonds)
        self._angles = _normalize("angles", angles)
        self._dihedrals = _normalize("dihedrals", dihedrals)
        self._impropers = _normalize("impropers", impropers)

    @property
    def bonds(self) -> pl.DataFrame:
        """Bonds: ``atoms`` holds 2 atom indices."""
        return self._bonds.clone()

    @property
    def angles(self) -> pl.DataFrame:
        """Angles: ``atoms`` holds 3 atom indices, the vertex in the middle."""
        return self._angles.clone()

    @property
    def dihedrals(self) -> pl.DataFrame:
        """Dihedrals: ``atoms`` holds 4 atom indices along the chain."""
        return self._dihedrals.clone()

    @property
    def impropers(self) -> pl.DataFrame:
        """Impropers: ``atoms`` holds 4 atom indices."""
        return self._impropers.clone()

    def tables(self) -> dict[str, pl.DataFrame]:
        """All four tables, keyed by kind."""
        return {kind: getattr(self, f"_{kind}").clone() for kind in KINDS}

    def max_atom_index(self) -> int:
        """Largest atom index used by any term, or -1 if there are none."""
        largest = -1
        for df in self.tables().values():
            if len(df):
                largest = max(largest, int(df["atoms"].to_numpy().max()))
        return largest

    def remap(self, mapping: np.ndarray) -> "Topology":
        """Topology after atoms are renumbered or removed.

        Parameters
        ----------
        mapping : numpy.ndarray
            For each old atom index, its new index, or -1 if the atom is
            removed. Terms that contain a removed atom are dropped.

        Returns
        -------
        Topology
            New topology with renumbered atoms.
        """
        mapping = np.asarray(mapping, dtype=np.int64)
        result = {}
        for kind, df in self.tables().items():
            if len(df) == 0:
                result[kind] = df
                continue
            atoms = df["atoms"].to_numpy()
            if atoms.max() >= len(mapping):
                raise InvalidParameter(f"{kind} refer to atoms beyond the mapping")
            new = mapping[atoms]
            keep = np.all(new >= 0, axis=1)
            result[kind] = df.filter(pl.Series(keep)).with_columns(
                pl.Series("atoms", new[keep], dtype=pl.Array(pl.Int64, KINDS[kind]))
            )
        return Topology(**result)

    def __len__(self) -> int:
        """Total number of terms of all kinds."""
        return sum(len(df) for df in self.tables().values())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Topology):
            return NotImplemented
        return all(
            a.equals(b) for a, b in zip(self.tables().values(), other.tables().values())
        )

    __hash__ = None  # compares by content

    def __repr__(self) -> str:
        counts = ", ".join(f"{kind}={len(df)}" for kind, df in self.tables().items())
        return f"Topology({counts})"
