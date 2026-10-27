# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import struct
import zlib

import numpy as np
import pytest
from mdapy._core import visualization as _native

import mdapy
from mdapy.analysis import DislocationAnalysis, SurfaceMesh, VoronoiAnalysis
from mdapy.build import build_crystal, dislocation_quadrupole
from mdapy.errors import Cancelled, InvalidParameter
from mdapy.identify import CommonNeighborAnalysis
from mdapy.visualization import (
    REPRESENTATIONS,
    STYLES,
    VIEWS,
    Scene,
    atom_colors,
    colormap,
    covalent_bonds,
    element_colors,
    element_radii,
    gpu_available,
    render,
    write_png,
)


def read_png(path):
    """The RGBA pixels of a PNG file written by write_png."""
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    position, stream = 8, b""
    while position < len(data):
        (length,) = struct.unpack(">I", data[position : position + 4])
        tag = data[position + 4 : position + 8]
        body = data[position + 8 : position + 8 + length]
        (crc,) = struct.unpack(
            ">I", data[position + 8 + length : position + 12 + length]
        )
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF
        if tag == b"IHDR":
            width, height = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            stream += body
        position += 12 + length
    raw = np.frombuffer(zlib.decompress(stream), np.uint8).reshape(
        height, 1 + 4 * width
    )
    assert (raw[:, 0] == 0).all()
    return raw[:, 1:].reshape(height, width, 4)


def srgb(linear):
    linear = np.asarray(linear, dtype=float)
    return np.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055
    )


# ------------------------------------------------------------ ray tracing


def random_scene(seed=0):
    rng = np.random.default_rng(seed)
    scene = Scene()
    scene.add_spheres(rng.random((30, 3)) * 10, rng.random(30) * 0.8 + 0.2, (1, 0, 0))
    a = rng.random((20, 3)) * 10
    scene.add_cylinders(a, a + rng.normal(size=(20, 3)), 0.3, (0, 1, 0))
    corner = rng.random((15, 3)) * 10
    triangles = corner[:, None, :] + rng.normal(size=(15, 3, 3))
    scene.add_triangles(triangles, (0, 0, 1))
    return scene


def brute_force(scene, origins, directions):
    """The nearest hit of each ray by testing every primitive."""
    arrays = scene._arrays()
    hits = []
    for o, d in zip(origins, directions):
        d = d / np.linalg.norm(d)
        best = (np.inf, -1, -1)
        for k, (c, r) in enumerate(
            zip(arrays["sphere_center"], arrays["sphere_radius"])
        ):
            oc = o - c
            b, cc = oc @ d, oc @ oc - r * r
            disc = b * b - cc
            if disc >= 0:
                for t in (-b - np.sqrt(disc), -b + np.sqrt(disc)):
                    if t > 0:
                        best = min(best, (t, 0, k))
                        break
        for k, (ends, r) in enumerate(
            zip(arrays["cylinder_end"], arrays["cylinder_radius"])
        ):
            p0, axis = ends[:3], ends[3:] - ends[:3]
            length = np.linalg.norm(axis)
            u = axis / length
            dp, op = d - (d @ u) * u, (o - p0) - ((o - p0) @ u) * u
            qa, qb, qc = dp @ dp, dp @ op, op @ op - r * r
            disc = qb * qb - qa * qc
            if qa > 1e-12 and disc >= 0:
                for t in ((-qb - np.sqrt(disc)) / qa, (-qb + np.sqrt(disc)) / qa):
                    along = (o + t * d - p0) @ u
                    if t > 0 and 0 <= along <= length:
                        best = min(best, (t, 1, k))
                        break
        for k, v in enumerate(arrays["triangle_vertex"].reshape(-1, 3, 3)):
            e1, e2 = v[1] - v[0], v[2] - v[0]
            matrix = np.array([-d, e1, e2]).T
            if abs(np.linalg.det(matrix)) > 1e-12:
                t, u, w = np.linalg.solve(matrix, o - v[0])
                if t > 0 and u >= 0 and w >= 0 and u + w <= 1:
                    best = min(best, (t, 2, k))
        hits.append(best)
    return hits


def test_trace_matches_brute_force():
    scene = random_scene()
    rng = np.random.default_rng(1)
    origins = rng.random((400, 3)) * 30 - 10
    directions = rng.random((400, 3)) * 10 - origins  # toward the primitives
    arrays = scene._arrays()
    distance, kind, index = _native.trace(
        **arrays,
        origins=origins.astype(np.float32),
        directions=directions.astype(np.float32),
    )
    expected = brute_force(
        scene,
        origins.astype(np.float32).astype(float),
        directions.astype(np.float32).astype(float),
    )
    hits = 0
    for d, k, i, (t, kk, ii) in zip(distance, kind, index, expected):
        if np.isinf(t):
            assert k == -1 and np.isinf(d)
        else:
            hits += 1
            assert d == pytest.approx(t, rel=1e-4, abs=1e-4)
            assert (k, i) == (kk, ii)
    assert hits > 150
    with pytest.raises(ValueError, match="differ in length"):
        _native.trace(
            **arrays,
            origins=origins[:2].astype(np.float32),
            directions=directions[:3].astype(np.float32),
        )


# ------------------------------------------------------------ pictures


def one_sphere(color=(0.2, 0.6, 0.9), **options):
    scene = Scene()
    scene.add_spheres([[0.0, 0, 0]], 1.0, color)
    return scene.render(width=200, height=200, view="top", **options)


def test_silhouette_and_flat_color():
    image = one_sphere(style="flat", outline_width=0.0, background=None, zoom=0.5)
    # framed with 4 % to spare and zoomed out twice: the sphere spans a
    # quarter of the picture's width
    covered = (image[..., 3] > 127).sum()
    radius_pixels = 200 / (2 * 2.0 * 1.04)
    assert covered == pytest.approx(np.pi * radius_pixels**2, rel=0.03)
    assert (image[0, 0] == 0).all()  # transparent outside
    center = image[100, 100]
    np.testing.assert_allclose(
        center[:3], np.round(255 * srgb(srgb_to_linear([0.2, 0.6, 0.9]))), atol=1
    )
    assert center[3] == 255


def srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def test_front_sphere_hides_the_back_one():
    scene = Scene()
    scene.add_spheres([[0.0, 0, 0], [0.0, 0, -3]], 1.0, [[1.0, 0, 0], [0, 0, 1.0]])
    image = scene.render(
        width=100, height=100, view="top", style="flat", outline_width=0
    )
    red, green, blue = image[50, 50, :3]
    assert red > 200 and blue < 50 and green < 50


def test_transparent_triangles():
    scene = Scene()
    scene.add_spheres([[0.0, 0, 0]], 1.0, (0, 0, 1.0))
    scene.add_triangles(
        [[[-5, -5, 2.0], [5, -5, 2], [0, 5, 2]]], (1.0, 0, 0), alpha=0.5
    )
    image = scene.render(
        width=100, height=100, view="top", style="flat", outline_width=0
    )
    r, g, b = image[50, 50, :3].astype(float) / 255
    expected = srgb(0.5 * srgb_to_linear([1, 0, 0]) + 0.5 * srgb_to_linear([0, 0, 1]))
    np.testing.assert_allclose([r, g, b], expected, atol=0.01)


def test_outlines():
    plain = one_sphere(style="flat", outline_width=0.0)
    lined = one_sphere(style="flat", outline_width=3.0)
    dark = lined[..., :3].sum(axis=2) < 60
    assert dark.sum() > 0 and (plain[..., :3].sum(axis=2) < 60).sum() == 0
    # the line circles the sphere, just outside its silhouette
    y, x = np.nonzero(dark)
    distance = np.hypot(x - 99.5, y - 99.5)
    radius = 100 / 1.04
    assert np.abs(distance - radius).max() < 5


def test_lighting_shading_and_threads():
    image = one_sphere(style="studio")
    lit = image[..., :3].astype(float).sum(axis=2)
    # the key light comes from the upper left
    assert lit[60, 60] > lit[140, 140]
    with mdapy.num_threads(3):
        again = one_sphere(style="studio")
    np.testing.assert_array_equal(image, again)
    for style in STYLES:
        assert one_sphere(style=style).shape == (200, 200, 4)


def test_views_and_perspective():
    scene = Scene()
    scene.add_spheres([[1.0, 2, 3]], 0.5, (1.0, 0.5, 0))
    for view in list(VIEWS) + [(1.0, 2.0, -0.5)]:
        image = scene.render(
            width=120, height=80, view=view, background=None, style="preview"
        )
        y, x = np.nonzero(image[..., 3] > 127)
        assert abs(x.mean() - 59.5) < 1.5 and abs(y.mean() - 39.5) < 1.5
    # along the axis up: another up is taken
    assert scene.render(
        width=40, height=40, view=(0, 0, -1), up=(0, 0, 1), style="preview"
    ).shape == (40, 40, 4)
    two = Scene()
    two.add_spheres([[-3.0, 0, 0], [3.0, 0, -12]], 1.0, (0.5, 0.5, 0.5), ids=[0, 1])
    image = two.render(
        width=200,
        height=100,
        view="top",
        perspective=True,
        background=None,
        style="preview",
    )
    near = (image[:, :100, 3] > 127).sum()
    far = (image[:, 100:, 3] > 127).sum()
    assert near > 1.2 * far > 0


def test_focus_and_span_frame_a_given_region():
    """A span of four radii shows a sphere at the focus half as tall as the
    picture, also in perspective; a focus one radius off moves it a quarter
    of the picture the other way."""
    scene = Scene()
    scene.add_spheres([[1.0, 2.0, 3.0]], 1.0, (1.0, 0.5, 0.0))
    options = dict(
        width=200,
        height=200,
        view="top",
        background=None,
        style="flat",
        outline_width=0.0,
    )
    image = scene.render(focus=(1.0, 2.0, 3.0), span=4.0, **options)
    shown = image[..., 3] > 127
    assert shown.sum() == pytest.approx(np.pi * 50**2, rel=0.03)
    y, x = np.nonzero(shown)
    assert abs(x.mean() - 99.5) < 1 and abs(y.mean() - 99.5) < 1
    moved = scene.render(focus=(2.0, 2.0, 3.0), span=4.0, **options)
    y, x = np.nonzero(moved[..., 3] > 127)
    assert abs(x.mean() - 49.5) < 1.5 and abs(y.mean() - 99.5) < 1
    seen = scene.render(focus=(1.0, 2.0, 3.0), span=4.0, perspective=True, **options)
    assert (seen[..., 3] > 127).sum() == pytest.approx(np.pi * 50**2, rel=0.08)
    with pytest.raises(InvalidParameter, match="span must be positive"):
        scene.render(span=0.0, **options)
    with pytest.raises(InvalidParameter, match="focus must be a point"):
        scene.render(focus=(1.0, 2.0), **options)


def test_style_settings():
    from mdapy.visualization import STYLES, style_settings

    goodsell = style_settings("goodsell")
    assert goodsell["flat"] is True and goodsell["outline_width"] == 1.2
    assert goodsell["supersample"] == 2 and goodsell["ao_distance"] is None
    assert all(set(style_settings(name)) == set(goodsell) for name in STYLES)
    # a copy: changing it changes no style
    goodsell["flat"] = False
    assert style_settings("goodsell")["flat"] is True
    # the settings render the same picture as the style
    scene = Scene()
    scene.add_spheres([[0.0, 0, 0]], 1.0, (0.2, 0.6, 0.9))
    options = dict(width=40, height=40, view="top")
    np.testing.assert_array_equal(
        scene.render(style="preview", **options),
        scene.render(**style_settings("preview"), **options),
    )
    with pytest.raises(InvalidParameter, match="style must be one of"):
        style_settings("pastel")


def test_png_round_trip(tmp_path):
    image = one_sphere(style="studio")
    write_png(tmp_path / "a.png", image)
    np.testing.assert_array_equal(read_png(tmp_path / "a.png"), image)
    scene = Scene()
    scene.add_spheres([[0.0, 0, 0]], 1.0, (1, 1, 1))
    written = scene.render(tmp_path / "b.png", width=30, height=20)
    np.testing.assert_array_equal(read_png(tmp_path / "b.png"), written)
    with pytest.raises(InvalidParameter, match="RGBA"):
        write_png(tmp_path / "c.png", image[..., :3])


def test_progress_cancel_and_errors():
    scene = random_scene()
    reports = []
    scene.render(
        width=64, height=64, progress=lambda done, total: reports.append((done, total))
    )
    assert reports and reports[-1][0] == reports[-1][1]
    token = mdapy.CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        scene.render(width=64, height=64, cancel=token)
    for options, message in [
        ({"style": "oil"}, "style must be one of"),
        ({"view": "under"}, "view must be one of"),
        ({"shine": 3}, "unknown options"),
        ({"width": 0}, "width and height"),
        ({"background": "mauve"}, "unknown color"),
        ({"background": (1, 2)}, "three numbers"),
    ]:
        with pytest.raises(InvalidParameter, match=message):
            scene.render(**options)
    empty = Scene().render(width=10, height=10)
    assert (empty == 255).all()
    with pytest.raises(InvalidParameter, match="radii must have shape"):
        Scene().add_spheres([[0.0, 0, 0]], [1.0, 2.0], (1, 1, 1))
    with pytest.raises(InvalidParameter, match="same shape"):
        Scene().add_cylinders([[0.0, 0, 0]], [[1.0, 0, 0], [2.0, 0, 0]], 0.1, (1, 1, 1))
    with pytest.raises(InvalidParameter, match="alpha must be"):
        Scene().add_triangles(np.zeros((1, 3, 3)), (1, 1, 1), alpha=2.0)
    with pytest.raises(InvalidParameter, match="fewer than three"):
        Scene().add_polygons(np.zeros((2, 3)), [0, 2], [0, 1], (1, 1, 1))
    with pytest.raises(InvalidParameter, match="ids must be"):
        Scene().add_spheres([[0.0, 0, 0]], 1.0, (1, 1, 1), ids=[-1])
    with pytest.raises(ValueError, match="positive size"):
        _native.render(**Scene()._arrays(), **_raw_camera(), width=0, height=1)


def _raw_camera():
    f = np.float32
    return dict(
        center=np.zeros(3, f), forward=np.array([0, 0, -1], f), up=np.array([0, 1, 0], f),
        half_width=1.0, half_height=1.0, perspective=False, aperture=0.0, focal_distance=1.0,
        ambient=0.3, diffuse=0.7,
        specular=0.2, shininess=30.0, key=np.ones(3, f), fill=np.ones(3, f), fill_intensity=0.2,
        ao_samples=0, ao_distance=1.0, ao_strength=0.5, shadows=False, shadow_strength=0.5,
        flat=False, toon_levels=0, outline_width=0.0, outline_depth=0.1,
        outline_color=np.zeros(3, f), fog=0.0, background=np.ones(4, f), supersample=1,
    )  # fmt: skip


# ------------------------------------------------------------ colors and systems


def test_colors():
    np.testing.assert_allclose(
        element_colors(["Cu", "Qq"]), [[200 / 255, 128 / 255, 51 / 255], [1, 1, 1]]
    )
    np.testing.assert_allclose(element_radii(["Cu", "Fe", "Qq"]), [1.28, 1.26, 1.0])
    ramp = colormap([0.0, 0.5, 1.0])
    np.testing.assert_allclose(ramp[0], [0x44 / 255, 0x01 / 255, 0x54 / 255])
    np.testing.assert_allclose(ramp[-1], [0xFD / 255, 0xE7 / 255, 0x25 / 255])
    np.testing.assert_allclose(
        colormap([5.0, -1.0], "gray", (0, 1)), [[1, 1, 1], [0, 0, 0]]
    )
    np.testing.assert_allclose(
        colormap([np.nan, np.nan], "gray"), [[0, 0, 0], [0, 0, 0]]
    )
    with pytest.raises(InvalidParameter, match="colormap must be"):
        colormap([1.0], "rainbow")
    system = build_crystal("fcc", 3.6, elements="Cu", repeat=(2, 2, 2)).with_columns(
        value=np.arange(32.0)
    )
    assert atom_colors(system).shape == (32, 3)
    structures = CommonNeighborAnalysis().compute(system)
    np.testing.assert_allclose(
        atom_colors(system, structures), np.broadcast_to(structures.colors[1], (32, 3))
    )
    np.testing.assert_allclose(atom_colors(system, "value")[0], colormap([0.0])[0])
    np.testing.assert_allclose(
        atom_colors(system, np.arange(32.0), value_range=(0, 31))[-1],
        colormap([0.0, 1.0])[1],
    )
    np.testing.assert_allclose(atom_colors(system, "#ff0000")[5], [1, 0, 0])
    np.testing.assert_allclose(atom_colors(system, np.ones((32, 3)))[0], [1, 1, 1])
    bare = mdapy.System(system["positions"], system.box)
    np.testing.assert_allclose(atom_colors(bare)[0], [1, 1, 1])


def test_render_systems(tmp_path):
    system = build_crystal("bcc", 2.87, elements="Fe", repeat=(3, 3, 3))
    image = render(system, tmp_path / "fe.png", width=160, height=120, style="preview")
    assert image.shape == (120, 160, 4) and (tmp_path / "fe.png").exists()
    half = np.arange(len(system)) < len(system) // 2
    shown = render(
        system,
        atoms=half,
        width=160,
        height=120,
        style="preview",
        box=False,
        background=None,
    )
    some = render(
        system,
        atoms=np.flatnonzero(half),
        width=160,
        height=120,
        style="preview",
        box=False,
        background=None,
    )
    np.testing.assert_array_equal(shown, some)
    with pytest.raises(InvalidParameter, match="mask of atoms"):
        render(system, atoms=np.ones(3, bool))
    bonded = render(
        system, bonds=2.6, radius=0.4, width=160, height=120, style="preview"
    )
    nl = mdapy.NeighborList.build(system, 2.6)
    np.testing.assert_array_equal(
        bonded,
        render(system, bonds=nl, radius=0.4, width=160, height=120, style="preview"),
    )
    cells = VoronoiAnalysis(polyhedra=True).compute(system).polyhedra
    drawn = render(
        system,
        atoms=[0],
        polyhedra=cells,
        width=160,
        height=120,
        style="preview",
        box=False,
    )
    assert drawn.shape == (120, 160, 4)
    by_edges = render(
        system,
        atoms=[0],
        polyhedra=cells,
        polyhedra_color="edges",
        polyhedra_edges=0.0,
        style="preview",
        width=60,
        height=60,
    )
    assert by_edges.shape == (60, 60, 4)
    with pytest.raises(InvalidParameter, match="polyhedra_color"):
        render(system, atoms=[0], polyhedra=cells, polyhedra_color="area")
    nothing = render(
        system,
        atoms=np.zeros(len(system), bool),
        polyhedra=cells,
        box=False,
        width=20,
        height=20,
    )
    assert (nothing == 255).all()


def test_bonds_across_the_boundary_stay_in_the_box():
    """A bond to a periodic image is drawn as the half next to each atom,
    so nothing lies beyond the box."""
    chain = mdapy.System(
        [[0.5, 2.0, 2.0], [3.5, 2.0, 2.0]], mdapy.Box(4.0), elements=["Cu", "Cu"]
    )
    scene_options = dict(view="top", width=200, height=200, background=None)
    bonded = render(
        chain,
        bonds=1.6,
        radius=0.3,
        bond_radius=0.1,
        box=True,
        style="preview",
        **scene_options,
    )
    boxed = render(chain, radius=0.3, box=True, style="preview", **scene_options)
    # the same framing: the bonds add nothing outside the box
    covered = bonded[..., 3] > 127
    inside = (boxed[..., 3] > 127).any(axis=0)
    assert not (covered.any(axis=0) & ~inside).any()
    row = covered[100]
    assert row[:15].any() and row[-15:].any()  # the halves reach both faces


def test_a_surface_across_the_boundary_stays_in_the_box():
    system = build_crystal("fcc", 3.615, elements="Cu", repeat=(6, 6, 6))
    corner = np.linalg.norm(system.box.minimum_image(system["positions"]), axis=1)
    holed = system.filter(corner > 5.0)  # a void around a corner of the box
    surface = SurfaceMesh().compute(holed).mesh
    hidden = np.zeros(len(holed), dtype=bool)
    options = dict(view="top", width=120, height=120, background=None, style="preview")
    drawn = render(holed, atoms=hidden, mesh=surface, **options)
    boxed = render(holed, atoms=hidden, **options)
    covered = drawn[..., 3] > 127
    inside = boxed[..., 3] > 127
    rows, columns = (
        np.flatnonzero(inside.any(axis=1)),
        np.flatnonzero(inside.any(axis=0)),
    )
    assert not covered[: rows[0]].any() and not covered[rows[-1] + 1 :].any()
    assert (
        not covered[:, : columns[0]].any() and not covered[:, columns[-1] + 1 :].any()
    )
    # a quarter of the void in each corner of the picture
    for r, c in [(rows[0] + 5, columns[0] + 5), (rows[-1] - 5, columns[-1] - 5)]:
        assert covered[r, c]
    colors = np.tile([1.0, 0.0, 0.0], (len(surface), 1))
    red = render(holed, atoms=hidden, mesh=surface, mesh_color=colors, **options)
    shade = red[covered & ~inside][:, :3].astype(int)  # off the edges of the box
    assert (shade[:, 0] >= shade[:, 1]).all() and (
        shade[:, 0] > shade[:, 1]
    ).mean() > 0.9
    faint = render(holed, atoms=hidden, mesh=surface, mesh_alpha=0.3, **options)
    assert not np.array_equal(faint, drawn)
    with pytest.raises(InvalidParameter, match="mesh_color"):
        render(holed, mesh=surface, mesh_color=np.zeros((2, 3)))


def test_dislocation_lines_stay_in_the_box():
    """Lines through the periodic box are cut at its faces, the pieces
    moved in; tubes in the color of their family."""
    from mdapy.visualization._render import _inside

    box = mdapy.Box(10.0)
    starts, ends = _inside(np.array([[8.0, 5, 5], [13.0, 5, 5], [13.0, 7, 5]]), box)
    assert ((starts >= 0) & (starts <= 10) & (ends >= 0) & (ends <= 10)).all()
    lengths = np.linalg.norm(ends - starts, axis=1)
    assert lengths.sum() == pytest.approx(7.0) and len(lengths) == 3
    system = dislocation_quadrupole(
        "bcc screw", 2.865, (243.3, 145.0, 116.1), (6, 10, 5), elements="Fe"
    )
    result = DislocationAnalysis("bcc").compute(system)
    hidden = np.zeros(len(system), dtype=bool)
    options = dict(
        view="front", width=160, height=160, background=None, style="preview"
    )
    drawn = render(system, atoms=hidden, dislocations=result, **options)
    boxed = render(system, atoms=hidden, **options)
    covered = drawn[..., 3] > 127
    inside = boxed[..., 3] > 127
    rows = np.flatnonzero(inside.any(axis=1))
    assert not covered[: rows[0]].any() and not covered[rows[-1] + 1 :].any()
    green = drawn[covered & ~inside][:, :3].astype(int)  # 1/2<111>, off the box edges
    assert len(green) and (green[:, 1] > green[:, 0] + 50).mean() > 0.9
    thin = render(
        system, atoms=hidden, dislocations=result, dislocation_radius=0.2, **options
    )
    assert (thin[..., 3] > 127).sum() < covered.sum()
    with pytest.raises(InvalidParameter, match="dislocation_radius"):
        render(system, dislocations=result, dislocation_radius=0)


def test_burgers_vector_arrows():
    """An arrow from the middle of each line along its Burgers vector, in
    one color or in that of the line."""
    system = dislocation_quadrupole(
        "fcc edge", 3.615, (168.4, 121.4, 75.4), (14, 8, 3), elements="Cu"
    )
    result = DislocationAnalysis("fcc").compute(system)
    hidden = np.zeros(len(system), dtype=bool)
    # along the lines: the edge Burgers vectors stick out across them; the
    # box keeps the framing
    options = dict(view="top", width=160, height=160, background=None, style="preview")
    lines = render(system, atoms=hidden, dislocations=result, **options)
    arrows = render(
        system,
        atoms=hidden,
        dislocations=result,
        burgers_vectors=True,
        burgers_color=(1.0, 0.0, 0.0),
        **options,
    )
    added = (arrows[..., 3] > 127) & ~(lines[..., 3] > 127)
    assert added.sum() > 20
    red = arrows[added][:, :3].astype(int)
    red = red[red.max(axis=1) > 100]  # lit, not the outlines
    assert len(red) > 20 and (red[:, 0] > red[:, 1] + 50).mean() > 0.9
    family = render(
        system,
        atoms=hidden,
        dislocations=result,
        burgers_vectors=True,
        burgers_color="family",
        burgers_scale=5.0,
        burgers_radius=0.5,
        **options,
    )
    wider = (family[..., 3] > 127) & ~(lines[..., 3] > 127)
    assert wider.sum() > added.sum()
    blue = family[wider][:, :3].astype(int)  # 1/2<110>
    blue = blue[blue.max(axis=1) > 100]
    assert len(blue) > 20 and (blue[:, 2] > blue[:, 0] + 50).mean() > 0.9
    for name, value in [
        ("burgers_scale", 0),
        ("burgers_radius", -1),
        ("burgers_color", "plaid"),
        ("burgers_color", (1, 0)),
    ]:
        with pytest.raises(InvalidParameter, match=name):
            render(system, dislocations=result, burgers_vectors=True, **{name: value})


# ------------------------------------------------------------ global illumination


def test_a_lone_sphere_lit_by_the_sky_and_the_key_light():
    # every bounce off a lone sphere leaves for the sky: no noise at all
    color = np.array([0.2, 0.6, 0.9])
    settings = {"ambient": 0.45, "diffuse": 0.8, "specular": 0.2, "shininess": 60.0}
    image = one_sphere(
        color, style="photo", light_size=0.0, supersample=1, samples=4, **settings
    )
    key = np.array([-0.45, 0.6, 0.66]) / np.linalg.norm([-0.45, 0.6, 0.66])
    normal = np.array([0.0, 0, 1])  # at the middle, toward the camera
    halfway = (key + normal) / np.linalg.norm(key + normal)
    light = (
        srgb_to_linear(color)
        * (settings["ambient"] + settings["diffuse"] * key @ normal)
        + settings["specular"] * (normal @ halfway) ** settings["shininess"]
    )
    np.testing.assert_allclose(
        image[100, 100, :3], np.round(255 * srgb(np.minimum(light, 1))), atol=2
    )
    with mdapy.num_threads(3):
        again = one_sphere(color, style="photo", samples=4)
    np.testing.assert_array_equal(again, one_sphere(color, style="photo", samples=4))
    with pytest.raises(InvalidParameter, match="samples must be"):
        one_sphere(style="photo", samples=0)
    with pytest.raises(InvalidParameter, match="bounces must be"):
        one_sphere(bounces=-1)


def floor_and_ball(ball_color, center, size=20.0):
    scene = Scene()
    corners = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]) * size
    scene.add_triangles(corners[[[0, 1, 2], [0, 2, 3]]], (0.6, 0.6, 0.6))
    scene.add_spheres([center], 1.0, ball_color)
    return scene


def test_soft_shadows():
    # a small ball high above the floor, toward the key light from the middle
    key = np.array([-0.45, 0.6, 0.66]) / np.linalg.norm([-0.45, 0.6, 0.66])
    scene = floor_and_ball("green", 15 * key)
    shadow = {}
    for size in (0.0, 0.4):
        image = scene.render(
            width=120,
            height=120,
            view="top",
            style="photo",
            light_size=size,
            samples=16,
        )
        rgb = image[..., :3].astype(float)
        floor = rgb[..., 0] > rgb[..., 1] - 5  # gray, not the green ball
        lit = np.median(rgb[..., 0][floor])
        shadow[size] = (rgb[..., 0][floor] < 0.85 * lit).sum()
    # a point light casts a dark shadow; a light wider than the ball, none
    assert shadow[0.0] > 20 and shadow[0.4] == 0


def test_soft_shadows_in_direct_light():
    # the ball subtends 0.067 rad: a light 0.1 rad wide leaves no full shadow,
    # only a wider half shadow
    key = np.array([-0.45, 0.6, 0.66]) / np.linalg.norm([-0.45, 0.6, 0.66])
    scene = floor_and_ball("green", 15 * key)
    deep, half = {}, {}
    for size in (0.0, 0.1):
        rgb = scene.render(
            width=120, height=120, view="top", light_size=size, samples=16
        )[..., :3].astype(float)
        floor = rgb[..., 0] > rgb[..., 1] - 5
        red = rgb[..., 0][floor]
        lit = np.median(red)
        deep[size] = (red < lit - 10).sum()
        half[size] = ((red < lit - 3) & (red >= lit - 10)).sum()
    assert deep[0.0] > 20 and deep[0.1] < 5
    assert half[0.1] > 2 * half[0.0]


def test_samples_in_every_style():
    # rays anywhere in the subpixels cover a disk closer to its area
    picture = {
        samples: one_sphere(
            style="flat",
            outline_width=0.0,
            background=None,
            zoom=0.5,
            supersample=1,
            samples=samples,
        )
        for samples in (1, 16)
    }
    radius_pixels = 200 / (2 * 2.0 * 1.04)
    area = picture[16][..., 3].astype(float).sum() / 255
    assert area == pytest.approx(np.pi * radius_pixels**2, rel=0.005)
    assert not np.array_equal(picture[1], picture[16])


def test_depth_of_field():
    # a sphere at the focus and one 20 Å behind it, seen from the front
    scene = Scene()
    scene.add_spheres([[-2.5, 0, 0], [2.5, 20, 0]], 1.0, (0.8, 0.3, 0.2))
    options = {
        "width": 160,
        "height": 160,
        "view": "front",
        "perspective": True,
        "focus": [0, 0, 0],
        "span": 12,
        "background": None,
        "style": "flat",
        "outline_width": 0.0,
        "samples": 16,
    }

    def blurred_edges(image):
        # pixels partly covered, about the near and the far sphere
        alpha = image[..., 3].astype(int)
        soft = (alpha > 12) & (alpha < 243)
        return soft[:, :80].sum(), soft[:, 80:].sum()

    near, far = blurred_edges(scene.render(**options))
    # sharp at the focus by default: the far sphere blurs
    near_dof, far_dof = blurred_edges(scene.render(aperture=0.4, **options))
    assert near_dof <= 1.1 * near and far_dof > 3 * far
    # sharp at the far sphere: the near one blurs
    distance = 6 / np.tan(np.radians(15))  # from the camera to the focus
    near_dof, far_dof = blurred_edges(
        scene.render(aperture=0.4, focal_distance=distance + 20, **options)
    )
    assert near_dof > 3 * near and far_dof <= 1.1 * far
    np.testing.assert_array_equal(
        scene.render(aperture=0.0, focal_distance=5.0, **options),
        scene.render(**options),
    )
    with mdapy.num_threads(3):
        again = scene.render(aperture=0.4, **options)
    np.testing.assert_array_equal(again, scene.render(aperture=0.4, **options))
    for settings, message in [
        ({"aperture": -1.0}, "aperture must be"),
        ({"aperture": float("nan")}, "aperture must be"),
        ({"focal_distance": 0.0}, "focal_distance must be"),
        ({"aperture": 0.4, "perspective": False}, "needs a perspective camera"),
    ]:
        with pytest.raises(InvalidParameter, match=message):
            scene.render(**(options | settings))


def test_light_reflected_in_color():
    scene = floor_and_ball("red", [0.0, 0, 1.0], size=4.0)
    options = {"width": 120, "height": 120, "view": "iso"}
    flat = scene.render(style="flat", outline_width=0, **options)[..., :3].astype(int)
    ball = flat[..., 0] > flat[..., 1] + 50
    near = np.zeros_like(ball)  # the ball and the pixels next to it
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            near |= np.roll(np.roll(ball, dy, axis=0), dx, axis=1)
    floor = ~near & (flat[..., 1] > 30)
    for style, tinted in (("studio", False), ("photo", True)):
        rgb = scene.render(style=style, **options)[..., :3].astype(int)
        red = (rgb[..., 0] - rgb[..., 1])[floor]
        # light off the red ball turns the floor next to it red
        share = (red >= 3).mean()
        assert (share > 0.01) == tinted and (share > 0) == tinted, (style, share)
        assert (red >= 0).all()


# ------------------------------------------------------------ molecules


def benzene():
    """Benzene in the xy plane: C-C 1.39 Å, C-H 1.09 Å."""
    angles = np.arange(6) * np.pi / 3
    ring = np.stack([np.cos(angles), np.sin(angles), np.zeros(6)], axis=1)
    positions = np.vstack([1.39 * ring, (1.39 + 1.09) * ring])
    return mdapy.System(positions, elements=["C"] * 6 + ["H"] * 6)


def test_covalent_bonds_and_radii():
    molecule = benzene()
    bonds = covalent_bonds(molecule)
    pairs = {tuple(sorted(p)) for p in zip(bonds.i.tolist(), bonds.j.tolist())}
    ring = {tuple(sorted((k, (k + 1) % 6))) for k in range(6)}
    assert pairs == ring | {(k, k + 6) for k in range(6)}  # no H-H, no C-H across
    assert len(bonds) == 24  # each bond both ways
    np.testing.assert_allclose(
        element_radii(["H", "C", "Og"], "covalent"), [0.31, 0.76, 1.5]
    )
    np.testing.assert_allclose(element_radii(["H", "C", "Og"], "vdw"), [1.2, 1.77, 2.0])
    np.testing.assert_allclose(element_radii(["H", "Zz"]), [0.46, 1.0])
    with pytest.raises(InvalidParameter, match="kind must be"):
        element_radii(["H"], "ionic")
    with pytest.raises(InvalidParameter, match="element column"):
        covalent_bonds(mdapy.System(molecule["positions"]))


def test_representations():
    molecule = benzene()
    options = {
        "width": 120,
        "height": 120,
        "view": "top",
        "box": False,
        "style": "flat",
    }
    pictures = {
        r: render(molecule, representation=r, **options) for r in REPRESENTATIONS
    }
    covered = {r: (p[..., :3].sum(axis=2) < 3 * 250).sum() for r, p in pictures.items()}
    # van der Waals spheres cover most, sticks the least
    assert covered["space_filling"] > covered["spheres"] > covered["licorice"]
    assert covered["ball_and_stick"] > covered["licorice"]
    # the parts can still be given by hand
    np.testing.assert_array_equal(
        render(molecule, representation="ball_and_stick", **options),
        render(
            molecule,
            radius=("vdw", 0.25),
            bonds="covalent",
            bond_radius=0.15,
            **options,
        ),
    )
    assert render(molecule, radius="covalent", **options).shape == (120, 120, 4)
    with pytest.raises(InvalidParameter, match="representation must be"):
        render(molecule, representation="cartoon")
    with pytest.raises(InvalidParameter, match="bonds must be"):
        render(molecule, bonds="hydrogen")


# ------------------------------------------------------------ on the GPU

needs_gpu = pytest.mark.skipif(not gpu_available(), reason="no GPU")


@needs_gpu
@pytest.mark.parametrize(
    "options",
    [
        {"style": "studio"},
        {"style": "goodsell"},
        {"style": "toon", "perspective": True, "fog": 0.5},
        {"style": "photo", "samples": 4},
        {"style": "glossy", "background": None},
        {"style": "studio", "light_size": 0.2, "samples": 4},
        {"style": "studio", "perspective": True, "aperture": 0.5, "samples": 4},
    ],
)
def test_the_gpu_draws_what_the_processor_draws(options):
    scene = random_scene()
    scene.add_triangles([[[-3, -3, 0.5], [6, -3, 0.5], [0, 6, 0.5]]], "red", alpha=0.4)
    cpu = scene.render(width=96, height=80, **options).astype(int)
    gpu = scene.render(width=96, height=80, device="gpu", **options).astype(int)
    difference = np.abs(cpu - gpu).max(axis=2)
    # rounding moves a few edges by a subpixel, nothing else
    assert (difference > 2).mean() < 0.01 and np.abs(cpu - gpu).mean() < 0.5
    np.testing.assert_array_equal(
        scene.render(width=96, height=80, device="auto", **options), gpu
    )


@needs_gpu
def test_the_gpu_can_stop_and_report():
    scene = random_scene()
    reports = []
    scene.render(
        width=64, height=64, device="gpu", progress=lambda d, t: reports.append((d, t))
    )
    assert reports[-1][0] == reports[-1][1]
    token = mdapy.CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        scene.render(width=64, height=64, device="gpu", cancel=token)

    def stop(done, total):
        raise RuntimeError("enough")

    with pytest.raises(RuntimeError, match="enough"):
        scene.render(width=64, height=64, device="gpu", progress=stop)


def test_devices(monkeypatch):
    scene = random_scene()
    with pytest.raises(InvalidParameter, match="device must be"):
        scene.render(width=8, height=8, device="tpu")
    monkeypatch.setattr("mdapy.visualization._scene.gpu_available", lambda: False)
    with pytest.raises(InvalidParameter, match="no GPU"):
        scene.render(width=8, height=8, device="gpu")
    np.testing.assert_array_equal(
        scene.render(width=8, height=8, device="auto"), scene.render(width=8, height=8)
    )


def test_the_color_of_the_box():
    system = mdapy.System(np.zeros((1, 3)), mdapy.Box(10.0), elements="Cu")
    options = {
        "atoms": np.zeros(1, dtype=bool),
        "style": "flat",
        "outline_width": 0.0,
        "width": 120,
        "height": 120,
        "box_radius": 0.3,
    }
    # the middle of the lines, where little green is left of the white
    rgb = render(system, box_color="red", **options)[..., :3].astype(int)
    lines = rgb[..., 1] < 120
    assert lines.sum() > 100 and (rgb[lines, 0] > 200).all()
    black = render(system, **options)[..., :3].astype(int)
    lines = black[..., 1] < 120
    assert lines.sum() > 100 and (black[lines, 0] < 120).all()
    with pytest.raises(InvalidParameter, match="color"):
        render(system, box_color="plaid", **options)
