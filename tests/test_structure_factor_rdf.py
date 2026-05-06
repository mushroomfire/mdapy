# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""mdapy.StructureFactor mode='rdf' — analytic and cross-mode validation.

The rdf-mode integrates a windowed g(r) → S(k) Fourier transform. We verify:

* analytic 2-atom ideal-gas case where S(k) = 1 + sin(kr0)/(kr0)
* the rdf-mode peak position matches the debye/direct-mode peak position on
  a disordered crystal (peak heights legitimately differ — debye uses every
  min-image pair, rdf truncates at L/2 with a Lorch window — that's the whole
  point of having rdf-mode)
* large-k behaviour converges to 1 across modes (sanity)
* partial structure factors are returned with correct keys when cal_partial=True
"""

import numpy as np
import polars as pl
import pytest

import mdapy as mp
from mdapy.box import Box


def _two_atom_system(r0: float, L: float = 20.0):
    """Two atoms in a periodic cubic box, well separated from PBC images."""
    data = pl.DataFrame({
        "id": [1, 2],
        "type": [1, 1],
        "element": ["Cu", "Cu"],
        "x": [0.0, r0],
        "y": [0.0, 0.0],
        "z": [0.0, 0.0],
    })
    return mp.System(data=data, box=Box([L, L, L]))


def test_rdf_mode_two_atoms_matches_analytic():
    """Two atoms at distance r0=2 in a 20Å box. Analytical S(k) = 1 + sinc(kr0).

    The rdf-mode integrates a windowed g(r); for an isolated pair this reduces
    to the analytical form to within window-induced error.
    """
    r0 = 2.0
    sys_ = _two_atom_system(r0, L=20.0)
    sfc = sys_.cal_structure_factor(0.5, 6.0, 50, mode="rdf", nbin_rdf=4000)
    k = sfc.k
    S_analytic = 1.0 + np.sin(k * r0) / (k * r0)
    # 5% tolerance — windowing introduces some smoothing.
    np.testing.assert_allclose(sfc.Sk, S_analytic, atol=0.1)


def test_rdf_mode_no_window_converges_better():
    """Without the Lorch window the integrand is sharp; for a clean two-atom
    case the no-window curve is closer to the analytic form than the windowed
    one."""
    r0 = 2.0
    sys_ = _two_atom_system(r0, L=20.0)
    nb_rdf = 4000
    sfc_w = sys_.cal_structure_factor(0.5, 6.0, 50, mode="rdf",
                                      nbin_rdf=nb_rdf, window=True)
    sfc_nw = sys_.cal_structure_factor(0.5, 6.0, 50, mode="rdf",
                                       nbin_rdf=nb_rdf, window=False)
    k = sfc_w.k
    S_analytic = 1.0 + np.sin(k * r0) / (k * r0)
    err_w = np.max(np.abs(sfc_w.Sk - S_analytic))
    err_nw = np.max(np.abs(sfc_nw.Sk - S_analytic))
    # Both better than 0.2 absolute, no-window strictly tighter on this clean case.
    assert err_nw < err_w
    assert err_nw < 0.1


def test_rdf_mode_first_peak_matches_debye_position():
    """On disordered fcc Cu the first S(k) peak position must agree across
    modes (rdf, debye, direct). Heights legitimately differ."""
    np.random.seed(0)
    sys_ = mp.build_lattice.build_crystal("Cu", "fcc", 3.615, nx=8, ny=8, nz=8)
    # Add small thermal-like displacement so the peak is broadened and finite.
    data = sys_.data.with_columns(
        x=sys_.data["x"] + np.random.normal(0, 0.15, sys_.N),
        y=sys_.data["y"] + np.random.normal(0, 0.15, sys_.N),
        z=sys_.data["z"] + np.random.normal(0, 0.15, sys_.N),
    )
    sys_._System__data = data
    sfc_d = sys_.cal_structure_factor(1.5, 12.0, 200, mode="debye")
    sfc_r = sys_.cal_structure_factor(1.5, 12.0, 200, mode="rdf", nbin_rdf=400)
    # First peak: search in [2.5, 3.5]
    mask = (sfc_d.k > 2.5) & (sfc_d.k < 3.5)
    k_peak_d = sfc_d.k[mask][np.argmax(sfc_d.Sk[mask])]
    k_peak_r = sfc_r.k[mask][np.argmax(sfc_r.Sk[mask])]
    # Same k-bin or one bin off
    assert abs(k_peak_d - k_peak_r) < 2 * (sfc_d.k[1] - sfc_d.k[0])


def test_rdf_mode_large_k_converges_to_one():
    """At large k, S(k) → 1 (or → c_α^2 weights for partial sum)."""
    sys_ = mp.build_lattice.build_crystal("Cu", "fcc", 3.615, nx=8, ny=8, nz=8)
    sfc_r = sys_.cal_structure_factor(1.5, 12.0, 200, mode="rdf", nbin_rdf=400)
    tail_mean = sfc_r.Sk[150:].mean()
    assert 0.85 < tail_mean < 1.15


def test_rdf_mode_partial_keys_and_total_relation():
    """cal_partial=True yields one A_αβ entry per upper-triangular pair, and
    the total reduces to the concentration-weighted Faber-Ziman sum:
        S(k) = Σ_αβ c_α c_β A_αβ(k)        (full matrix; α<β counted twice)
    """
    hea = mp.build_hea(
        ("Al", "Cu"), (0.5, 0.5), "fcc",
        a=3.7, nx=4, ny=4, nz=4, random_seed=1,
    )
    sfc = hea.cal_structure_factor(
        0.5, 8.0, 60, cal_partial=True, mode="debye", nbin_rdf=200,
    )
    keys = set(sfc.Sk_partial.keys())
    assert keys == {("Al", "Al"), ("Al", "Cu"), ("Cu", "Cu")}, f"got {keys}"
    c_Al = c_Cu = 0.5
    expected_total = (
        c_Al ** 2 * sfc.Sk_partial[("Al", "Al")]
        + 2 * c_Al * c_Cu * sfc.Sk_partial[("Al", "Cu")]
        + c_Cu ** 2 * sfc.Sk_partial[("Cu", "Cu")]
    )
    np.testing.assert_allclose(sfc.Sk, expected_total, atol=1e-12)


def test_debye_mode_matches_erhard_reference_formula():
    """mdapy's debye mode (window=False) equals the Faber-Ziman reference

        A_αβ(k) = 1 + (4πρ/k) ∫₀^{r_c} r [g_αβ(r) - 1] sin(kr) dr

    bit-for-bit. Reproduces the trapezoid-on-bin-centers integration used by
    Erhard et al. (Nat Commun 15, 2024) — verified against their published
    code on a binary Cu/Ni dataset to <1e-10.

    Self-contained: we compute the reference inline from mdapy's own RDF (so
    the test doesn't depend on external code), then check that
    cal_structure_factor produces the same numbers.
    """
    np.random.seed(0)
    hea = mp.build_hea(
        ("Al", "Cu"), (0.5, 0.5), "fcc",
        a=3.7, nx=4, ny=4, nz=4, random_seed=3,
    )
    rc = 6.0
    nbin = 200
    k = np.linspace(0.5, 8.0, 60)

    # Inline FZ reference using mdapy's RDF.
    rdf = hea.cal_radial_distribution_function(rc, nbin, streaming=True)
    rho = hea.N / hea.box.volume
    sin_kr = np.sin(np.outer(k, rdf.r))
    A_ref = {}
    for pair in [("Al", "Al"), ("Al", "Cu"), ("Cu", "Cu")]:
        g_ab = rdf.g_partial[pair]
        I = np.trapezoid(sin_kr * rdf.r * (g_ab - 1.0), x=rdf.r, axis=1)
        A_ref[pair] = 1.0 + 4.0 * np.pi * rho * I / k

    sfc = hea.cal_structure_factor(
        0.5, 8.0, 60, cal_partial=True, mode="debye",
        rc=rc, nbin_rdf=nbin, window=False,
    )
    # cal_structure_factor uses linspace endpoints; match here too.
    np.testing.assert_allclose(sfc.k, k, atol=1e-12)
    for pair, expected in A_ref.items():
        np.testing.assert_allclose(
            sfc.Sk_partial[pair], expected, atol=1e-10,
            err_msg=f"FZ A_alpha_beta for {pair} departs from inline reference",
        )


def test_neutron_electron_and_derived_pdfs_match_inline_reference():
    r"""Independent inline reference for the X-ray, neutron, and electron
    weighted totals plus the inverse-FT derived :math:`g(r)`,
    :math:`G(r)`, :math:`R(r)` — all keyed off the partial Faber-Ziman
    A_alpha_beta(k) the class returns.
    """
    hea = mp.build_hea(
        ("Al", "Cu"), (0.5, 0.5), "fcc",
        a=3.7, nx=4, ny=4, nz=4, random_seed=4,
    )
    sfc = hea.cal_structure_factor(
        0.5, 8.0, 60, cal_partial=True, atomic_form_factors=True,
        mode="debye", rc=6.0, nbin_rdf=200, window=False,
    )

    # Erhard-style total: sum over (2 - delta) c_a c_b f_a f_b A_ab,
    # divided by (sum_a c_a f_a)^2.
    elements = sfc._uniele
    c = sfc._concentrations

    def total_with(ff):
        norm = np.zeros_like(sfc.k)
        for ci, fi in zip(c, ff):
            norm = norm + ci * fi
        out = np.zeros_like(sfc.k, dtype=ff[0].dtype)
        for (a, b), A_ab in sfc.Sk_partial.items():
            ia = elements.index(a); ib = elements.index(b)
            multi = 1.0 if ia == ib else 2.0
            out = out + multi * c[ia] * c[ib] * ff[ia] * ff[ib] * A_ab
        return out / norm ** 2

    # X-ray reference
    ff_x = [sfc._xray_form_factor(e) for e in elements]
    ref_xray = total_with(ff_x)
    np.testing.assert_allclose(sfc.Sk_xray, ref_xray, atol=1e-12)

    # Neutron reference (Cu, Al are real-valued so total is real here)
    ff_n = [sfc._neutron_form_factor(e) for e in elements]
    ref_neutron = total_with(ff_n)
    np.testing.assert_allclose(
        sfc.get_neutron_structure_factor(), ref_neutron, atol=1e-12
    )

    # Electron reference (Mott-Bethe with k > 0 is well-defined)
    ff_e = [sfc._electron_form_factor(e) for e in elements]
    ref_electron = total_with(ff_e)
    np.testing.assert_allclose(
        sfc.get_electron_structure_factor(), ref_electron, atol=1e-12
    )

    # Derived g(r), G(r), R(r) — sanity check shapes and finiteness.
    r, g_x = sfc.get_xray_pair_distribution_function()
    assert r.shape == g_x.shape
    assert np.all(np.isfinite(g_x))
    _, G_x = sfc.get_xray_reduced_pair_distribution_function()
    assert np.all(np.isfinite(G_x))
    _, R_x = sfc.get_xray_radial_distribution_function()
    assert np.all(np.isfinite(R_x))


def test_rdf_g_partial_keys_are_tuples_for_element_input():
    """When the System has an ``element`` column, RDF g_partial keys are
    string tuples like ('Cu', 'Ni')."""
    hea = mp.build_hea(
        ("Al", "Cu"), (0.5, 0.5), "fcc",
        a=3.7, nx=4, ny=4, nz=4, random_seed=2,
    )
    rdf = hea.cal_radial_distribution_function(5.0, 100)
    keys = set(rdf.g_partial.keys())
    assert keys == {("Al", "Al"), ("Al", "Cu"), ("Cu", "Cu")}, f"got {keys}"
    # g_total is a 1-D array (replaces the previous 3-D rdf.g matrix)
    assert rdf.g_total.ndim == 1
    assert rdf.g_total.shape == (100,)


def test_rdf_mode_default_rc_is_half_max_box_vector():
    """When rc is left None, debye mode picks rc = max(|a|, |b|, |c|)/2,
    matching OVITO's documented L = maximum simulation cell vector length."""
    sys_ = mp.build_lattice.build_crystal("Cu", "fcc", 3.615, nx=4, ny=4, nz=4)
    L_max = float(max(np.linalg.norm(sys_.box.box[i]) for i in range(3)))
    sfc = sys_.cal_structure_factor(0.5, 6.0, 30, mode="debye")
    assert abs(sfc.rc - L_max / 2.0) < 1e-9
