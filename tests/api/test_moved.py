# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The names that moved in 2.2 keep working at their old places, with a
warning that names the new one."""

import importlib

import pytest

import mdapy
from mdapy._moved import GOES_IN, MOVED

OLD = [(module, name) for module, names in MOVED.items() for name in names]


@pytest.mark.parametrize("module, name", OLD)
def test_an_old_name_gives_the_new_object_with_a_warning(module, name):
    new = MOVED[module][name]
    with pytest.warns(
        DeprecationWarning, match=rf"{module}\.{name} moved to {new}\.{name}"
    ):
        old = getattr(importlib.import_module(module), name)
    assert old is getattr(importlib.import_module(new), name)
    assert name in importlib.import_module(new).__all__


def test_from_an_old_module_import_works():
    with pytest.warns(DeprecationWarning, match="moved to mdapy.properties.phonons"):
        from mdapy.lattice_dynamics import phonons
    with pytest.warns(DeprecationWarning, match="moved to mdapy.properties.defects"):
        from mdapy.simulation import surface_energy
    assert phonons is mdapy.properties.phonons.phonons
    assert surface_energy is mdapy.properties.defects.surface_energy
    with pytest.warns(DeprecationWarning):
        assert (
            mdapy.lattice_dynamics.quasi_harmonic
            is mdapy.properties.thermodynamics.quasi_harmonic
        )


def test_what_stays_gives_no_warning(recwarn):
    from mdapy.simulation import MolecularDynamics, minimize, thermal_velocities  # noqa: F401
    from mdapy.simulation.controls import Hold  # noqa: F401

    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


@pytest.mark.parametrize("module", ["mdapy.lattice_dynamics", "mdapy.simulation"])
def test_an_unknown_name_is_an_attribute_error(module):
    with pytest.raises(AttributeError, match="has no attribute 'nothing'"):
        getattr(importlib.import_module(module), "nothing")
    assert not hasattr(
        importlib.import_module(module), "hnemd"
    )  # new in 2.2, never here


def test_the_old_imports_go_when_they_said():
    """Two minor releases after a move its old imports go: then remove
    mdapy/_moved.py, mdapy/lattice_dynamics/, its import in mdapy/__init__.py,
    the __getattr__ of mdapy/simulation/__init__.py and this file."""
    release = tuple(int(part) for part in mdapy.__version__.split(".")[:2])
    goes = tuple(int(part) for part in GOES_IN.split("."))
    assert release < goes, f"the imports that moved go in {GOES_IN}: remove them"
