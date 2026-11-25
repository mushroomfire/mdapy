# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
"""The two kernels of the DFT: MDAPY's own kernels (the FFTs and the linear
algebra) against the GPU libraries of the vendor. The transforms agree to
rounding on the grids of the DFT, the products and the triangular solves on
the shapes of the solver and the projectors, and so do whole calculations:
norm-conserving, ultrasoft and PAW pseudopotentials, k-points and the real
bands of Gamma, spin, mixed precision.
"""

import numpy as np
import pytest

import mdapy
from mdapy.calculators import DFT
from mdapy.calculators._dft import scf
from mdapy.calculators._dft.upf import read_upf
from mdapy.errors import InvalidParameter

from dft import espresso, gpu

_dft = gpu.module()
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

BOHR, HARTREE = 0.529177210903, 27.211386245988
KERNELS = ("mdapy", "vendor")


def sphere_of(cell, cutoff, grid, half=False):
    miller, _, box, partner = _dft.sphere(
        cell, [0.0, 0.0, 0.0], cutoff, list(grid), half
    )
    return miller, box, partner


# sizes of every radix, odd and even, a plane that fits a block and one that
# does not; 22 = 2 x 11 is not MDAPY's to transform and goes to the vendor's
@pytest.mark.parametrize(
    "grid",
    [
        (27, 27, 27),
        (36, 30, 25),
        (20, 21, 35),
        (45, 48, 80),
        (64, 64, 64),
        (90, 90, 96),
        (22, 24, 20),
    ],
)
def test_the_transforms_agree(grid):
    rng = np.random.default_rng(1)
    cell = np.diag([9.0, 9.5, 10.0])
    _, box, _ = sphere_of(cell, 3.0, grid)
    c = rng.normal(size=(3, len(box))) + 1j * rng.normal(size=(3, len(box)))
    values = [_dft.to_real_space(c, box, list(grid), kernels=b) for b in KERNELS]
    scale = np.abs(values[1]).max()
    assert np.abs(values[0] - values[1]).max() < 1e-13 * scale
    # two programs differ in the last bits; one program gives the same bits
    assert np.array_equal(values[0], values[1]) == (grid == (22, 24, 20))
    noise = rng.normal(size=(2, *grid)) + 1j * rng.normal(size=(2, *grid))
    back = [_dft.to_coefficients(noise, box, kernels=b) for b in KERNELS]
    assert np.abs(back[0] - back[1]).max() < 1e-13 * np.abs(back[1]).max()
    # the round trip of MDAPY's own
    again = _dft.to_coefficients(values[0], box, kernels="mdapy")
    assert np.abs(again - c).max() < 1e-13 * np.abs(c).max()


@pytest.mark.parametrize(
    "grid", [(27, 27, 27), (36, 30, 25), (45, 48, 80), (90, 90, 96)]
)
def test_the_transforms_of_the_planes_of_a_sphere(grid):
    """The transforms of boxes of which only the planes of the sphere matter:
    the values of the sphere are those of the whole boxes to the last bit (the
    planes beyond it are never read), and its coefficients from noise those of
    the whole boxes to rounding (the axes in another order); with a plane
    larger than a block, the whole boxes."""
    rng = np.random.default_rng(2)
    cell = np.diag([9.0, 9.5, 10.0])
    miller, box, _ = sphere_of(cell, 3.0, grid)
    window = int(np.abs(np.asarray(miller)[:, 0]).max())
    assert 2 * window + 1 < grid[0]
    c = rng.normal(size=(3, len(box))) + 1j * rng.normal(size=(3, len(box)))
    for kernels in KERNELS:
        whole = _dft.to_real_space(c, box, list(grid), kernels=kernels)
        planes = _dft.to_real_space(c, box, list(grid), kernels=kernels, window=window)
        np.testing.assert_array_equal(planes, whole)
        noise = rng.normal(size=(2, *grid)) + 1j * rng.normal(size=(2, *grid))
        whole = _dft.to_coefficients(noise, box, kernels=kernels)
        planes = _dft.to_coefficients(noise, box, kernels=kernels, window=window)
        assert np.abs(planes - whole).max() < 1e-13 * np.abs(whole).max()


@pytest.mark.parametrize(
    "grid",
    [
        (27, 27, 27),
        (36, 30, 24),
        (20, 21, 35),
        (45, 48, 80),
        (48, 45, 75),
        (90, 90, 96),
    ],
)
def test_the_real_transforms_agree(grid):
    rng = np.random.default_rng(2)
    cell = np.diag([9.0, 9.5, 10.0])
    _, box, partner = sphere_of(cell, 3.0, grid, half=True)
    c = rng.normal(size=(3, len(box))) + 1j * rng.normal(size=(3, len(box)))
    c[:, 0] = c[:, 0].real  # G = 0, the first of the half sphere, is real
    values = [
        _dft.to_real_space_gamma(c, box, partner, list(grid), kernels=b)
        for b in KERNELS
    ]
    assert np.abs(values[0] - values[1]).max() < 1e-13 * np.abs(values[1]).max()
    noise = rng.normal(size=(2, *grid))
    back = [_dft.to_coefficients_gamma(noise, box, kernels=b) for b in KERNELS]
    assert np.abs(back[0] - back[1]).max() < 1e-13 * np.abs(back[1]).max()


def columns(rng, rows, cols, dtype):
    """A matrix of noise stored by columns."""
    a = rng.uniform(-1, 1, (rows, cols))
    if np.issubdtype(dtype, np.complexfloating):
        a = a + 1j * rng.uniform(-1, 1, (rows, cols))
    return np.asfortranarray(a, dtype=dtype)


def op(a, t):
    return a if t == "N" else a.T if t == "T" else a.conj().T


TYPES = [np.float64, np.float32, np.complex128, np.complex64]


# (m, n, k) of c (m, n) = op(a) (m, k) op(b) (k, n): sizes off the tiles of 32,
# the rotations of the bands (many rows), the inner products (long sums, split
# along the sum) and the products with the projectors
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1),
        (33, 31, 17),
        (100, 70, 130),
        (6000, 40, 90),
        (45, 37, 9000),
        (130, 7, 20011),
        (64, 64, 0),
    ],
)
def test_the_products_agree(dtype, shape):
    m, n, k = shape
    rng = np.random.default_rng(5)
    single = dtype in (np.float32, np.complex64)
    limit = 2e-5 if single else 5e-14
    for ta in "NTC":
        for tb in "NTC":
            a = columns(rng, *((m, k) if ta == "N" else (k, m)), dtype)
            b = columns(rng, *((k, n) if tb == "N" else (n, k)), dtype)
            own, vendor = (_dft.product(a, b, ta, tb, kernels=x) for x in KERNELS)
            assert own.shape == (m, n) and own.dtype == dtype
            wide = (
                np.complex128
                if np.issubdtype(dtype, np.complexfloating)
                else np.float64
            )
            exact = op(a.astype(wide), ta) @ op(b.astype(wide), tb)
            scale = max(np.abs(exact).max(), 1.0)
            assert np.abs(own - vendor).max() < limit * scale
            assert np.abs(own - exact).max() < limit * scale
    # the long sums are split, another order than the library's: other bits
    if k > 5000:
        assert not np.array_equal(own, vendor)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 1), (33, 31, 17), (100, 70, 130), (6000, 40, 90), (154, 154, 17640)],
)
def test_the_small_blocks_of_the_products(dtype, shape):
    """The small blocks of the portable products of double precision, which
    MDAPY takes where its large ones would stand idle on a small result, give
    the products of the large ones to rounding (GPUs with tensor cores use
    neither)."""
    m, n, k = shape
    rng = np.random.default_rng(7)
    for ta in "NTC":
        for tb in "NTC":
            a = columns(rng, *((m, k) if ta == "N" else (k, m)), dtype)
            b = columns(rng, *((k, n) if tb == "N" else (n, k)), dtype)
            small, large = (_dft.product(a, b, ta, tb, small_tiles=s) for s in (1, 0))
            exact = op(a, ta) @ op(b, tb)
            scale = max(np.abs(exact).max(), 1.0)
            assert np.abs(small - exact).max() < 5e-14 * scale
            assert np.abs(small - large).max() < 5e-14 * scale


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("shape", [(1, 1), (7, 5), (1000, 17), (3000, 100), (50, 333)])
def test_the_triangular_solves_agree(dtype, shape):
    """b r^-1 with r upper triangular and well conditioned, as the Cholesky
    factor of the overlap of the bands is; the lower part of r is not read."""
    m, n = shape
    rng = np.random.default_rng(6)
    single = dtype in (np.float32, np.complex64)
    limit = 1e-4 if single else 1e-12
    b = columns(rng, m, n, dtype)
    r = columns(rng, n, n, dtype) * dtype(0.5 / np.sqrt(n))
    r[np.diag_indices(n)] = 1.0 + 0.3 * np.sin(np.arange(n))
    own, vendor = (_dft.solve_upper(b, r, kernels=x) for x in KERNELS)
    wide = np.complex128 if np.issubdtype(dtype, np.complexfloating) else np.float64
    upper = np.triu(r.astype(wide))
    scale = max(np.abs(vendor).max(), 1.0)
    assert np.abs(own - vendor).max() < limit * scale
    assert np.abs(own.astype(wide) @ upper - b).max() < limit * scale


def hermitian(rng, n, dtype, values=None):
    """A Hermitian matrix of noise stored by columns, or the one of the
    eigenvalues given in a basis of noise."""
    a = columns(
        rng,
        n,
        n,
        np.complex128 if np.issubdtype(dtype, np.complexfloating) else np.float64,
    )
    a = a + a.conj().T
    if values is not None:
        _, q = np.linalg.eigh(a)
        a = (q * np.asarray(values, dtype=float)) @ q.conj().T
        a = (a + a.conj().T) / 2
    return np.asfortranarray(a, dtype=dtype)


# the eigenvalues of each case, by its size: noise, all equal, a few values
# many times each, close pairs, and a wide range
SPECTRA = {
    "noise": lambda n: None,
    "equal": lambda n: np.full(n, 2.5),
    "repeated": lambda n: np.arange(n) % 3 - 1.0,
    "clustered": lambda n: (
        np.repeat(np.arange((n + 1) // 2), 2)[:n] + 1e-9 * np.arange(n)
    ),
    "graded": lambda n: np.logspace(-8, 2, n),
}


# (n, the largest matrix MDAPY solves on the host, -1 for its default of 64):
# on the host, and on the GPU, where 33 and 65 are just beyond a leaf of the
# divide and conquer and a block of reflectors
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("spectrum", list(SPECTRA))
@pytest.mark.parametrize(
    "n, host", [(1, -1), (2, -1), (3, -1), (20, -1), (65, -1), (200, -1), (1, 0), (2, 0), (3, 0), (33, 0), (65, 0),
                 (200, 0), (300, -1), (515, -1)]
)  # fmt: skip
def test_the_eigenpairs_agree(dtype, spectrum, n, host):
    """Eigenvalues against the library of the vendor and NumPy; eigenvectors by
    their residuals and their orthonormality, which degenerate eigenvalues
    leave the only test. MDAPY's kernels both ways, as on a GPU of full double
    precision (the scalar work of the merges on the GPU, a launch a step of
    the tridiagonalization) and as on one without, and with the
    tridiagonalization in panels, which MDAPY takes for the large matrices."""
    rng = np.random.default_rng(n)
    single = dtype in (np.float32, np.complex64)
    limit = 2e-5 if single else 1e-12
    a = hermitian(rng, n, dtype, SPECTRA[spectrum](n))
    garbage = np.asfortranarray(
        np.triu(a) + np.tril(columns(rng, n, n, dtype), -1)
    )  # the lower part is not read
    wide = np.complex128 if np.issubdtype(dtype, np.complexfloating) else np.float64
    exact = np.linalg.eigvalsh(a.astype(wide))
    scale = max(np.abs(exact).max(), 1.0)
    for kernels, full, panels in (
        ("mdapy", 0, -1),
        ("mdapy", 1, -1),
        ("mdapy", 1, 0),
        ("vendor", -1, -1),
    ):
        w, v = _dft.eigh(
            garbage,
            kernels=kernels,
            host_limit=host,
            full_double=full,
            panels_from=panels,
        )
        assert v.dtype == dtype and w.dtype == np.float64
        # ascending; the library of the vendor within rounding in a cluster
        assert np.all(np.diff(w) >= (0 if kernels == "mdapy" else -limit * scale))
        assert np.abs(w - exact).max() < limit * scale
        v = v.astype(wide)
        assert np.abs(a.astype(wide) @ v - v * w).max() < 20 * limit * scale
        assert np.abs(v.conj().T @ v - np.eye(n)).max() < 20 * limit


@pytest.mark.parametrize("dtype, n", [(np.complex128, 1100), (np.float64, 1600)])
def test_the_eigenpairs_of_large_matrices(dtype, n):
    """What MDAPY takes for the large matrices on a GPU of full double
    precision: the tridiagonalization in panels (from 1024 complex numbers and
    1536 real ones), leaves of 16 rows and blocks of 64 reflectors."""
    rng = np.random.default_rng(n)
    a = hermitian(rng, n, dtype, SPECTRA["clustered"](n))
    exact = np.linalg.eigvalsh(a)
    scale = max(np.abs(exact).max(), 1.0)
    w, v = _dft.eigh(a, full_double=1)
    assert np.all(np.diff(w) >= 0)
    assert np.abs(w - exact).max() < 1e-12 * scale
    assert np.abs(a @ v - v * w).max() < 1e-11 * scale
    assert np.abs(v.conj().T @ v - np.eye(n)).max() < 1e-11


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize(
    "n, host",
    [
        (1, -1),
        (2, -1),
        (17, -1),
        (64, -1),
        (200, -1),
        (1, 0),
        (2, 0),
        (64, 0),
        (65, 0),
        (200, 0),
        (400, -1),
    ],
)
def test_the_cholesky_factors_agree(dtype, n, host):
    """a = r^H r with r upper triangular, the lower part of a not read; and the
    pivot named when a is not positive definite. On the host, and on the GPU
    (the limit 0) in blocks of 64 columns."""
    rng = np.random.default_rng(n + 1)
    single = dtype in (np.float32, np.complex64)
    limit = 1e-4 if single else 1e-12
    b = columns(rng, n + 3, n, dtype)
    wide = np.complex128 if np.issubdtype(dtype, np.complexfloating) else np.float64
    a = np.asfortranarray(
        b.astype(wide).conj().T @ b.astype(wide) + 0.1 * np.eye(n), dtype=dtype
    )
    lower = np.tril(columns(rng, n, n, dtype), -1)
    given = np.asfortranarray(np.triu(a) + lower)
    factors = []
    for kernels in KERNELS:
        info, r = _dft.cholesky(given, kernels=kernels, host_limit=host)
        assert info == 0 and r.dtype == dtype
        u = np.triu(r).astype(wide)
        assert np.all(np.diagonal(u).real > 0)
        assert np.abs(u.conj().T @ u - a).max() < limit * np.abs(a).max()
        factors.append(u)
    assert np.abs(factors[0] - factors[1]).max() < limit * np.abs(factors[1]).max()
    if n > 2:
        bad = given.copy(order="F")
        bad[n // 2, n // 2] = -1.0
        assert [_dft.cholesky(bad, kernels=k, host_limit=host)[0] for k in KERNELS] == [
            n // 2 + 1
        ] * 2


class Device:
    """A GPU part of an API and a compute capability, for the choices by the GPU."""

    def __init__(self, api, capability):
        self.api = api
        self.capability = capability

    def device_capability(self):
        return self.capability


@pytest.mark.parametrize(
    "api, capability, full",
    [
        ("CUDA", (7, 0), True),  # V100
        ("CUDA", (8, 0), True),  # A100
        ("CUDA", (9, 0), True),  # H100, GH200
        ("CUDA", (8, 6), False),  # RTX 30 and the A-series
        ("CUDA", (8, 9), False),  # RTX 40
        ("CUDA", (7, 5), False),  # T4
        ("HIP", (9, 0), True),  # the Instinct series, whatever HIP reports
        ("HIP", (8, 6), True),
    ],
)
def test_the_kernels_by_the_gpu(monkeypatch, api, capability, full):
    """kernels="auto" is the vendor's on GPUs of full double precision and
    MDAPY's on the others, as precision="auto" is double and mixed; a choice
    given is kept."""
    monkeypatch.setattr(scf.loader, "module", lambda: Device(api, capability))
    assert scf.full_double() == full
    assert scf.kernels_of("auto") == ("vendor" if full else "mdapy")
    assert scf.mixed_precision("auto") == (not full)
    assert scf.kernels_of("mdapy") == "mdapy" and scf.kernels_of("vendor") == "vendor"
    with pytest.raises(ValueError, match="kernels"):
        scf.kernels_of("other")


def test_a_backend_unknown_is_refused():
    cell = np.diag([9.0, 9.5, 10.0])
    _, box, _ = sphere_of(cell, 3.0, (20, 20, 20))
    with pytest.raises(ValueError, match="kernels"):
        _dft.to_real_space(
            np.zeros((1, len(box)), dtype=complex), box, [20, 20, 20], kernels="other"
        )
    with pytest.raises(ValueError, match="kernels"):
        scf.run(np.eye(3) * 8.0, np.zeros((1, 3)), [0], [None], 10.0, kernels="other")
    with pytest.raises(InvalidParameter, match="kernels"):
        DFT(300.0, kernels="other")


def both(cell, positions, kinds, pseudos, cutoff, **options):
    """The same calculation with each kernels, converged tightly."""
    runs = [
        scf.run(cell, positions, kinds, pseudos, cutoff, energy_tolerance=1e-12, density_tolerance=1e-14,
                symmetry=False, kernels=kernels, **options)
        for kernels in KERNELS
    ]  # fmt: skip
    assert all(r["converged"] for r in runs)
    return runs


def agree(own, vendor, atoms, energy=1e-10, forces=1e-9, strain=1e-9, levels=1e-9):
    """Energies (hartree), forces (hartree/bohr) and strain derivatives of the
    two kernels within the tolerances, the first and the last per atom."""
    assert own["grid"] == vendor["grid"] and own["bands"] == vendor["bands"]
    assert own["energy"] == pytest.approx(vendor["energy"], abs=energy * atoms)
    np.testing.assert_allclose(own["forces"], vendor["forces"], rtol=0, atol=forces)
    np.testing.assert_allclose(
        own["strain_derivative"],
        vendor["strain_derivative"],
        rtol=0,
        atol=strain * atoms,
    )
    held = vendor["occupations"] > 1e-8
    np.testing.assert_allclose(
        own["eigenvalues"][held], vendor["eigenvalues"][held], rtol=0, atol=levels
    )


def cubic_silicon():
    a = 5.431 / BOHR
    base = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                     [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]])  # fmt: skip
    rng = np.random.default_rng(3)
    return np.eye(3) * a, base + rng.normal(0.0, 0.004, base.shape)


@pytest.mark.parametrize("gamma", [False, True])
def test_norm_conserving_silicon_at_gamma(gamma):
    """Complex bands, and the real bands of Gamma through the real transforms."""
    cell, positions = cubic_silicon()
    pp = read_upf(espresso.pseudo("Si_std.upf"))
    own, vendor = both(
        cell, positions, [0] * 8, [pp], 12.0, functional="pbe_pw", gamma=gamma
    )
    assert own["gamma"] == gamma
    agree(own, vendor, 8)


def test_norm_conserving_silicon_with_k_points():
    cell, positions = cubic_silicon()
    pp = read_upf(espresso.pseudo("Si_std.upf"))
    own, vendor = both(
        cell, positions, [0] * 8, [pp], 12.0, functional="pbe_pw", kgrid=(2, 2, 2)
    )
    agree(own, vendor, 8)


def test_ultrasoft_aluminium_with_smearing():
    """Two grids: the bands on the smooth one, the density on the dense one."""
    a = 4.05 / BOHR
    positions = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) + [[0.01, 0, 0], [0, 0, 0], [0, 0, 0],
                                                                                 [0, -0.01, 0.005]]  # fmt: skip
    pp = read_upf(espresso.pseudo("pslibrary/Al.pbe-n-rrkjus_psl.1.0.0.UPF"))
    own, vendor = both(np.eye(3) * a, positions, [0] * 4, [pp], 12.0, density_cutoff=96.0, smearing="gaussian",
                       sigma=0.02, kgrid=(2, 2, 2))  # fmt: skip
    agree(own, vendor, 4)


@pytest.mark.parametrize("gamma", [False, True])
def test_paw_iron_with_spin(gamma):
    h = 2.87 / BOHR
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.49, 0.5]])
    pp = read_upf(espresso.pseudo("pslibrary/Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"))
    own, vendor = both(np.eye(3) * h, positions, [0, 0], [pp], 15.0, density_cutoff=120.0, smearing="gaussian",
                       sigma=0.02, magmoms=[2.5, 2.5], gamma=gamma)  # fmt: skip
    agree(own, vendor, 2)
    assert own["magnetization"] == pytest.approx(vendor["magnetization"], abs=1e-8)


@pytest.mark.parametrize("gamma", [False, True])
def test_mixed_precision(gamma):
    """The steps in single precision take the single transforms of each
    kernels; the run ends in double precision, at the same result."""
    cell, positions = cubic_silicon()
    pp = read_upf(espresso.pseudo("Si_std.upf"))
    own, vendor = both(
        cell,
        positions,
        [0] * 8,
        [pp],
        12.0,
        functional="pbe_pw",
        gamma=gamma,
        precision="mixed",
    )
    assert np.sum(own["single"]) > 0 and np.sum(vendor["single"]) > 0
    agree(own, vendor, 8, energy=1e-9, forces=1e-7, strain=1e-7, levels=1e-7)


def test_the_calculator_keeps_its_backend():
    """DFT(kernels=...): "mdapy" by default, kept in the provenance; the
    energy, the forces and the virial of the two agree."""
    a = 4.05
    cell = np.diag([a, a, a])
    fractions = np.array([[0.01, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    system = mdapy.System(fractions @ cell, mdapy.Box(cell), elements=["Al"] * 4)
    common = dict(pseudopotentials={"Al": espresso.pseudo("Al_std.upf")}, kpoints=(2, 2, 2), sigma=0.2,
                  functional="pbe_pw", tolerance=1e-10, symmetry=False)  # fmt: skip
    own, vendor = (
        DFT(300.0, kernels="mdapy", **common),
        DFT(300.0, kernels="vendor", **common),
    )
    assert own.kernels == "mdapy" and vendor.kernels == "vendor"
    # by default the kernels of the GPU, which the provenance names
    auto = DFT(300.0, **common)
    assert auto.kernels == "auto"
    auto.compute(system, ("energy",))
    assert auto.last["provenance"]["kernels"] == scf.kernels_of("auto")
    first = own.compute(system, ("energy", "forces", "virial"))
    second = vendor.compute(system, ("energy", "forces", "virial"))
    assert own.last["provenance"]["kernels"] == "mdapy"
    assert vendor.last["provenance"]["kernels"] == "vendor"
    # two runs stopped at 1e-10 eV are not at the same density: their forces
    # are within about 1e-6 eV/Å of the converged ones, and of each other (the
    # runs converged tightly, above, agree to 1e-9 hartree/bohr)
    assert first.energy == pytest.approx(second.energy, abs=1e-8)
    np.testing.assert_allclose(first.forces, second.forces, atol=5e-6)
    np.testing.assert_allclose(first.virial, second.virial, atol=2e-5)
