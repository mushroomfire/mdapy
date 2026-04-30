# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
Core `System` class tests.

Covers construction paths, public mutators, getters, the calculator hook,
the in-place neighbor invalidation invariants, and a handful of regression
tests for bugs found during the 2026-04 audit.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import mdapy as mp
from mdapy.box import Box


# ===========================================================================
# Helpers
# ===========================================================================

def _make_simple(n=5, with_velocities=False, with_element=False):
    """Tiny system inside a 10 Å cubic PBC box."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.5, 9.5, size=(n, 3))
    s = mp.System(pos=pos, box=10.0)
    cols = {"type": pl.Series([1] * n, dtype=pl.Int32)}
    if with_velocities:
        cols.update({"vx": rng.normal(size=n), "vy": rng.normal(size=n),
                     "vz": rng.normal(size=n)})
    if with_element:
        cols["element"] = pl.Series(["Cu"] * n)
    s.update_data(s.data.with_columns(**cols))
    return s


# ===========================================================================
# Construction
# ===========================================================================

def test_init_pos_box():
    pos = np.zeros((4, 3))
    s = mp.System(pos=pos, box=5.0)
    assert s.N == 4
    assert tuple(s.data.columns[:3]) == ("x", "y", "z")
    np.testing.assert_allclose(s.box.box, np.eye(3) * 5.0)


def test_init_data_box():
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]})
    s = mp.System(data=df, box=[2, 2, 2])
    assert s.N == 2


def test_init_no_args_raises():
    with pytest.raises(RuntimeError):
        mp.System()


def test_init_global_info_does_not_leak_between_instances():
    """Regression: __init__ used `global_info: Dict = {}` as default,
    so all instances shared one dict. Mutating one leaked to the other."""
    a = mp.System(pos=np.zeros((1, 3)), box=1.0)
    b = mp.System(pos=np.zeros((1, 3)), box=1.0)
    a.global_info["pollute"] = 42
    assert "pollute" not in b.global_info, (
        "global_info must be per-instance, not shared via mutable default."
    )


def test_init_global_info_explicit_passes_through():
    s = mp.System(pos=np.zeros((1, 3)), box=1.0,
                  global_info={"timestep": 7, "energy": -1.5})
    assert s.global_info["timestep"] == 7
    assert s.global_info["energy"] == -1.5


def test_init_data_is_single_chunk():
    """The constructor must rechunk so downstream `to_numpy(allow_copy=False)`
    is safe (multi-chunk frames break that)."""
    df = pl.concat(
        [pl.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]}) for _ in range(3)]
    )
    assert df.n_chunks() > 1  # sanity — concat leaves multiple chunks
    s = mp.System(data=df, box=1.0)
    assert s.data.n_chunks() == 1


# ===========================================================================
# repr / N / data property
# ===========================================================================

def test_repr_has_atom_count():
    s = _make_simple(3)
    text = repr(s)
    assert "Atom Number: 3" in text


def test_repr_with_global_info():
    s = mp.System(pos=np.zeros((1, 3)), box=1.0, global_info={"timestep": 10})
    assert "timestep: 10" in repr(s)


# ===========================================================================
# set_element / set_type_by_element
# ===========================================================================

def test_set_element_scalar():
    s = _make_simple(4)
    s.set_element("Cu")
    assert s.data["element"].to_list() == ["Cu"] * 4


def test_set_element_list_replaces_existing():
    s = _make_simple(3, with_element=True)
    s.set_element(["Fe", "Ni", "Co"])
    assert s.data["element"].to_list() == ["Fe", "Ni", "Co"]


def test_set_element_wrong_length():
    s = _make_simple(3)
    with pytest.raises(AssertionError):
        s.set_element(["Fe", "Ni"])


def test_set_type_by_element_orders_from_list():
    s = _make_simple(3)
    s.set_element(["Cu", "Al", "Cu"])
    s.set_type_by_element(["Cu", "Al"])
    assert s.data["type"].to_list() == [1, 2, 1]


def test_set_type_by_element_missing_element_in_list():
    s = _make_simple(2)
    s.set_element(["Cu", "Au"])
    with pytest.raises(AssertionError):
        s.set_type_by_element(["Cu", "Al"])  # Au is missing


# ===========================================================================
# get_positions / get_velocities
# ===========================================================================

def test_get_positions_cartesian():
    s = _make_simple(3)
    cart = s.get_positions(reduced=False)
    assert cart.columns == ["x", "y", "z"]


def test_get_positions_reduced_roundtrip():
    """frac = pos @ inv_box; pos = frac @ box. Round-trip must be identity."""
    s = _make_simple(5)
    frac = s.get_positions(reduced=True).to_numpy()
    cart = frac @ s.box.box
    np.testing.assert_allclose(cart, s.data.select("x", "y", "z").to_numpy(),
                               atol=1e-10)


def test_get_velocities_missing_raises():
    s = _make_simple(2, with_velocities=False)
    with pytest.raises(AssertionError):
        s.get_velocities()


def test_get_velocities_present():
    s = _make_simple(3, with_velocities=True)
    v = s.get_velocities()
    assert v.shape == (3, 3)


# ===========================================================================
# update_data / update_box
# ===========================================================================

def test_update_data_rechunks():
    s = _make_simple(2)
    df = pl.concat([s.data[:1], s.data[1:]])
    assert df.n_chunks() > 1
    s.update_data(df)
    assert s.data.n_chunks() == 1


def test_update_data_reset_neighbor_clears_attrs():
    s = _make_simple(20)
    s.build_neighbor(rc=3.0)
    assert hasattr(s, "verlet_list")
    s.update_data(s.data, reset_neighbor=True)
    assert not hasattr(s, "verlet_list")
    assert not hasattr(s, "rc")


def test_update_box_scale_pos_preserves_fractional():
    s = _make_simple(4)
    frac_before = s.get_positions(reduced=True).to_numpy()
    s.update_box([20, 20, 20], scale_pos=True)
    frac_after = s.get_positions(reduced=True).to_numpy()
    np.testing.assert_allclose(frac_before, frac_after, atol=1e-10)


def test_update_box_scale_pos_rejects_non_pbc():
    """scale_pos requires the *new* box to be fully periodic — the affine
    map otherwise has no well-defined behaviour at open boundaries."""
    s = mp.System(pos=np.zeros((1, 3)), box=10.0)
    with pytest.raises(AssertionError):
        s.update_box(Box(np.eye(3) * 6, boundary=[1, 1, 0]), scale_pos=True)


def test_update_box_invalidates_neighbors():
    s = _make_simple(20)
    s.build_neighbor(rc=3.0)
    s.update_box([20, 20, 20])
    assert not hasattr(s, "verlet_list")


# ===========================================================================
# wrap_pos / replicate
# ===========================================================================

def test_wrap_pos_into_box():
    pos = np.array([[10.5, -0.1, 5.0]])
    s = mp.System(pos=pos, box=10.0)
    s.wrap_pos()
    p = s.data.select("x", "y", "z").to_numpy()
    assert (p >= 0).all() and (p < 10).all()


def test_replicate_scales_count_and_box():
    s = _make_simple(2)
    s.replicate(2, 3, 1)
    assert s.N == 2 * 6
    np.testing.assert_allclose(s.box.box[0, 0], 20.0)
    np.testing.assert_allclose(s.box.box[1, 1], 30.0)


# ===========================================================================
# calc setter validation
# ===========================================================================

def test_calc_setter_rejects_non_calculator():
    s = _make_simple(1)
    with pytest.raises(TypeError):
        s.calc = "not a calculator"


def test_get_energy_without_calc_raises():
    s = _make_simple(1)
    with pytest.raises(AssertionError):
        s.get_energy()


# ===========================================================================
# Bridges: ASE / OVITO round-trip
# ===========================================================================

def test_ase_roundtrip():
    pytest.importorskip("ase")
    s = _make_simple(4, with_element=True)
    atoms = s.to_ase()
    s2 = mp.System(ase_atom=atoms)
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        s2.data.select("x", "y", "z").to_numpy(),
        atol=1e-10,
    )


def test_ovito_roundtrip():
    pytest.importorskip("ovito")
    # Element is needed: the OVITO reader looks up `type2element[1]`, which
    # only exists when the source system populated particle_type.types.
    s = _make_simple(4, with_element=True)
    dc = s.to_ovito()
    s2 = mp.System(ovito_atom=dc)
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        s2.data.select("x", "y", "z").to_numpy(),
        atol=1e-10,
    )


# ===========================================================================
# Regression tests for audit bugs
# ===========================================================================

def test_write_data_does_not_mutate_self_data(tmp_path):
    """Regression: System.write_data used to call set_type_by_element on self,
    silently overwriting the in-memory `type` column as a side effect of
    saving. Writing must be a pure read."""
    s = _make_simple(3, with_element=True)
    # Element list with a different ordering than what `type` was at construction.
    original_type = s.data["type"].to_list()
    s.write_data(str(tmp_path / "out.data"), element_list=["Au", "Cu"])
    assert s.data["type"].to_list() == original_type, (
        "write_data must not mutate self.data"
    )


def test_cal_cluster_analysis_invalid_rc_raises_typeerror():
    """Regression: previously the code did `raise "..."`, which itself
    raises TypeError but with the wrong intent. Now it must raise TypeError
    with a useful message."""
    s = _make_simple(5)
    with pytest.raises(TypeError, match="rc should be"):
        s.cal_cluster_analysis(rc="not a number")


def test_cal_structure_entropy_keeps_both_columns_when_average_rc_set():
    """Regression: cal_structure_entropy used to drop the `entropy` column
    when `average_rc>0` because the second `data = self.data.with_columns(...)`
    re-started from self.data instead of chaining."""
    s = _make_simple(40, with_element=True)
    s.cal_structure_entropy(rc=4.0, sigma=0.5, average_rc=2.5)
    assert "entropy" in s.data.columns
    assert "entropy_ave" in s.data.columns


def test_cal_centro_symmetry_parameter_respects_N():
    """Regression: cal_centro_symmetry_parameter used to call
    sort_neighbor with hardcoded 18 instead of the user's N. With N=8 (BCC),
    that previously gave wrong results. Smoke-check that csp is finite."""
    # Build a small BCC-ish cube — exact correctness isn't checked here
    # (PTM tests cover that), only that the call completes for N=8.
    rng = np.random.default_rng(1)
    pos = rng.uniform(0.5, 9.5, size=(50, 3))
    s = mp.System(pos=pos, box=10.0)
    s.build_neighbor(rc=4.0)  # ensure existing neighbors > 8
    s.cal_centro_symmetry_parameter(N=8)
    assert "csp" in s.data.columns
    assert np.all(np.isfinite(s.data["csp"].to_numpy()))


def test_average_by_neighbor_message_is_grammatical():
    """Regression: the assert message used to read 'Only supprot for big box.'
    Make sure it's spelled correctly now."""
    s = _make_simple(5)
    s.update_data(s.data.with_columns(prop=pl.lit(1.0)))
    s.build_neighbor(rc=3.0)
    if hasattr(s, "_enlarge_data"):
        with pytest.raises(AssertionError, match="support"):
            s.average_by_neighbor(3.0, "prop")


def test_box_property_setter_invalidates_neighbors():
    """Direct ``system.box = new_box`` must clear neighbor caches just like
    ``update_box`` does — otherwise verlet_list refers to stale indices."""
    s = _make_simple(20)
    s.build_neighbor(rc=3.0)
    assert hasattr(s, "verlet_list")
    s.box = Box(np.eye(3) * 20)
    assert not hasattr(s, "verlet_list")
    assert not hasattr(s, "rc")


def test_box_property_accepts_raw_array():
    s = _make_simple(2)
    s.box = np.eye(3) * 7.5
    np.testing.assert_allclose(s.box.box, np.eye(3) * 7.5)


def test_cal_radial_distribution_function_does_not_overwrite_type():
    """Regression: previously RDF derived `type` from `element` and wrote it
    back into self.data, silently overwriting any user-supplied `type`."""
    s = _make_simple(40, with_element=True)
    # Give the user-side `type` column a deliberately weird ordering that
    # does NOT match alphabetical element ordering.
    s.update_data(s.data.with_columns(type=pl.lit(5, dtype=pl.Int32)))
    types_before = s.data["type"].to_list()
    s.cal_radial_distribution_function(rc=4.0, nbin=20)
    types_after = s.data["type"].to_list()
    assert types_before == types_after, (
        "RDF must not overwrite a user-provided `type` column."
    )


def test_cal_steinhardt_keeps_data_single_chunk():
    """Regression: cal_steinhardt_bond_orientation used to assign self.__data
    directly without rechunking, which downstream `to_numpy(allow_copy=False)`
    callers cannot tolerate."""
    s = _make_simple(40)
    s.cal_steinhardt_bond_orientation([6], rc=3.0)
    assert s.data.n_chunks() == 1
    assert "ql6" in s.data.columns


def test_cal_cluster_analysis_keeps_data_single_chunk():
    s = _make_simple(40)
    s.cal_cluster_analysis(rc=3.0)
    assert s.data.n_chunks() == 1
    assert "cluster_id" in s.data.columns


def test_set_element_length_mismatch_message_uses_N():
    """Regression: the assert previously read shape[0] in code, fine, but the
    message must clearly state the actual atom count."""
    s = _make_simple(3)
    with pytest.raises(AssertionError, match="3"):
        s.set_element(["Fe", "Ni"])


def test_from_ovito_no_particle_type_table():
    """Regression: from_ovito previously did `type2element[1]` unconditionally,
    crashing when the source had no particle_type table (e.g. a System built
    from raw positions only)."""
    pytest.importorskip("ovito")
    s = mp.System(pos=np.zeros((3, 3)), box=5.0)
    dc = s.to_ovito()
    s2 = mp.System(ovito_atom=dc)
    assert s2.N == 3


# ===========================================================================
# Deprecated kwarg
# ===========================================================================

def test_update_data_reset_calcolator_deprecated_alias():
    """The misspelled `reset_calcolator` kwarg still works for back-compat
    but emits a DeprecationWarning."""
    s = _make_simple(3)
    with pytest.warns(DeprecationWarning, match="misspelling"):
        s.update_data(s.data, reset_calcolator=True)


# ===========================================================================
# set_pka
# ===========================================================================

def test_set_pka_assigns_velocity_to_target_index():
    """With a >>thermal PKA energy, the targeted atom must end up with the
    largest |v| even after COM-momentum subtraction."""
    s = _make_simple(10, with_velocities=True, with_element=True)
    s.set_pka(energy=200_000.0, direction=np.array([1.0, 0.0, 0.0]), index=3)
    speeds = np.linalg.norm(
        s.data.select("vx", "vy", "vz").to_numpy(), axis=1
    )
    assert int(np.argmax(speeds)) == 3


def test_set_pka_unknown_element_raises():
    s = _make_simple(4, with_velocities=True, with_element=True)
    with pytest.raises(ValueError, match="not in data"):
        s.set_pka(energy=100.0, direction=np.array([1, 0, 0]), element="Au")


def test_set_pka_index_out_of_bounds_raises():
    s = _make_simple(4, with_velocities=True, with_element=True)
    with pytest.raises(ValueError, match="out of bounds"):
        s.set_pka(energy=100.0, direction=np.array([1, 0, 0]), index=999)


# ===========================================================================
# replicate / wrap_pos preserve data columns
# ===========================================================================

def test_replicate_preserves_extra_columns():
    s = _make_simple(2, with_element=True)
    s.update_data(s.data.with_columns(charge=pl.lit(0.5)))
    s.replicate(2, 1, 1)
    assert "charge" in s.data.columns
    assert s.data["charge"].to_list() == [0.5] * 4


def test_wrap_pos_preserves_extra_columns():
    s = _make_simple(3)
    s.update_data(s.data.with_columns(temperature=pl.lit(300.0)))
    # Push one atom outside the box.
    s.update_data(s.data.with_columns(x=s.data["x"] + 100.0))
    s.wrap_pos()
    assert "temperature" in s.data.columns


# ===========================================================================
# cal_chemical_species smoke
# ===========================================================================

def test_cal_chemical_species_search():
    """Synthetic 'water' configuration: 2 H atoms close to each O, far from
    the next molecule. cal_chemical_species(['H2O']) should count 2."""
    pos = np.array([
        [0.0, 0.0, 0.0],   # O1
        [0.9, 0.0, 0.0],   # H
        [-0.9, 0.0, 0.0],  # H
        [10.0, 10.0, 10.0],  # O2
        [10.9, 10.0, 10.0],  # H
        [9.1, 10.0, 10.0],   # H
    ])
    s = mp.System(pos=pos, box=Box(np.eye(3) * 30, boundary=[1, 1, 1]))
    s.set_element(["O", "H", "H", "O", "H", "H"])
    counts = s.cal_chemical_species(search_species=["H2O"], scale=0.6)
    assert counts == {"H2O": 2}
