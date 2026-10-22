# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""The calculator interface: energies, forces and virials of systems."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from mdapy import units
from mdapy.core.system import System
from mdapy.errors import InvalidParameter

#: Every property a calculator may compute.
PROPERTIES = frozenset(
    {"energy", "atomic_energies", "forces", "virial", "atomic_virials"}
)


@dataclass(frozen=True)
class CalculatorResult:
    """What a calculator computed for a system; unrequested values are None.

    Attributes
    ----------
    energy : float or None
        Total energy in eV.
    atomic_energies : numpy.ndarray or None
        (N,) energies in eV; they sum to ``energy``.
    forces : numpy.ndarray or None
        (N, 3) forces in eV/Å.
    virial : numpy.ndarray or None
        (3, 3) virial in eV, positive under compression.
    atomic_virials : numpy.ndarray or None
        (N, 3, 3) virials in eV; they sum to ``virial``.
    """

    energy: float | None = None
    atomic_energies: np.ndarray | None = None
    forces: np.ndarray | None = None
    virial: np.ndarray | None = None
    atomic_virials: np.ndarray | None = None

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                value.flags.writeable = False

    def stress(self, system: System) -> np.ndarray:
        """(3, 3) stress in GPa, positive under tension: -virial / volume."""
        if self.virial is None:
            raise InvalidParameter("the virial was not computed")
        return -self.virial / system.box.volume * units.EV_PER_A3_TO_GPA

    def attach(self, system: System) -> System:
        """The system with these values as columns and information.

        ``forces``, ``atomic_energies`` and ``atomic_virials`` become
        columns; ``energy`` and ``virial`` become entries of ``info``.
        """
        columns = {
            name: getattr(self, name)
            for name in ("forces", "atomic_energies", "atomic_virials")
            if getattr(self, name) is not None
        }
        info = {
            name: getattr(self, name)
            for name in ("energy", "virial")
            if getattr(self, name) is not None
        }
        return system.with_columns(**columns).with_info(**info)


class Calculator(ABC):
    """Computes energies, forces and virials of systems.

    Implementations set ``cutoff``, ``elements`` and ``implemented`` and
    define :meth:`compute`. :func:`mdapy.calculators.testing.check_calculator`
    tests any implementation.

    Attributes
    ----------
    cutoff : float
        Interaction cutoff in Å; a neighbor list for :meth:`compute` must have
        at least this cutoff.
    elements : tuple of str or None
        Elements the calculator handles, or None for any.
    implemented : frozenset of str
        The properties :meth:`compute` can return, a subset of
        ``energy``, ``atomic_energies``, ``forces``, ``virial`` and
        ``atomic_virials``.
    """

    cutoff: float
    elements: tuple[str, ...] | None = None
    implemented: frozenset[str] = PROPERTIES

    @abstractmethod
    def compute(
        self,
        system: System,
        properties: Iterable[str] = ("energy", "forces"),
        *,
        neighbors: Any = None,
    ) -> CalculatorResult:
        """Computes properties of a system.

        Parameters
        ----------
        system : System
            The atoms, with the columns the calculator needs (usually
            ``element``).
        properties : iterable of str, optional
            The properties to compute; others may be computed too.
        neighbors : NeighborList, optional
            A neighbor list of this system with at least ``cutoff``; built
            when not given.

        Returns
        -------
        CalculatorResult
            At least the requested properties.

        Raises
        ------
        InvalidParameter
            If a property is not implemented, or the neighbor list does not
            fit the system.
        MissingProperty
            If the system lacks a column the calculator needs.
        """

    def _check_properties(self, properties: Iterable[str]) -> frozenset[str]:
        wanted = frozenset(properties)
        unknown = wanted - PROPERTIES
        if unknown:
            raise InvalidParameter(f"unknown properties {sorted(unknown)}")
        missing = wanted - self.implemented
        if missing:
            raise InvalidParameter(
                f"{type(self).__name__} does not compute {sorted(missing)}"
            )
        return wanted
