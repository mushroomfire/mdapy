# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import os
import re
import shutil
import struct
import subprocess
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import pytest
from mdapy._core import visualization as _native

import mdapy
from mdapy.analysis import DislocationAnalysis, SurfaceMesh, VoronoiAnalysis
from mdapy.build import build_crystal, dislocation_quadrupole
from mdapy.errors import Cancelled, InvalidParameter
from mdapy.identify import CommonNeighborAnalysis
from mdapy.visualization import (
    LABEL_COLORS,
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
    show_image,
    write_png,
)
from mdapy.visualization import _render as render_module


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
            # the disks of the ends
            if d @ u != 0:
                for end in (0, length):
                    t = (end - (o - p0) @ u) / (d @ u)
                    across = o + t * d - p0 - end * u
                    if t > 0 and across @ across <= r * r:
                        best = min(best, (t, 1, k))
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


@pytest.mark.parametrize(
    "size, options",
    [
        ({"width": 8193, "height": 1}, {}),
        ({"width": 1, "height": 8193}, {}),
        ({"width": 8192, "height": 1}, {"supersample": 3}),
        ({"width": 5000, "height": 6000}, {"supersample": 3}),
    ],
)
def test_a_picture_that_could_use_up_the_memory_is_refused(size, options):
    with pytest.raises(InvalidParameter, match="at most 8192|more than 16384"):
        Scene().render(**size, style="flat", **options)


def test_the_largest_picture_is_drawn():
    scene = Scene()
    scene.add_spheres([[0.0, 0, 0]], 1.0, (0.2, 0.6, 0.9))
    image = scene.render(width=8192, height=2, style="flat", supersample=2)
    assert image.shape == (2, 8192, 4)


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
    image = one_sphere(style="matte")
    lit = image[..., :3].astype(float).sum(axis=2)
    # the key light comes from the upper left
    assert lit[60, 60] > lit[140, 140]
    with mdapy.num_threads(3):
        again = one_sphere(style="matte")
    np.testing.assert_array_equal(image, again)
    for style in STYLES:
        assert one_sphere(style=style).shape == (200, 200, 4)


def sphere_pixel(normal):
    """The pixel of one_sphere, seen from the top, where the surface has the
    normal: the sphere is 2 / 1.04 of the picture wide."""
    radius = 100 / 1.04
    return int(round(100 - normal[1] * radius - 0.5)), int(
        round(100 + normal[0] * radius - 0.5)
    )


def test_fast_lights_as_the_viewports_of_ovito():
    # in sRGB: 0.4 ambient, 0.6 from the viewer, a highlight of 0.25 with the
    # exponent 6 from the upper left and a little behind
    color = np.array([0.2, 0.6, 0.9])
    image = one_sphere(color, style="fast").astype(float) / 255
    viewer = np.array([0.0, 0, 1])
    light = np.array([-1.8, 1.5, -0.2]) / np.linalg.norm([-1.8, 1.5, -0.2])
    peak = (light + viewer) / np.linalg.norm(light + viewer)  # mirrors it to the viewer
    side = np.array([0.6, -0.3, 0.0])
    side[2] = np.sqrt(1 - side @ side)
    for normal, highlight in ((viewer, 0.0), (peak, 0.25), (side, 0.0)):
        expected = np.minimum(color * (0.4 + 0.6 * normal[2]) + highlight, 1)
        np.testing.assert_allclose(image[sphere_pixel(normal)][:3], expected, atol=0.02)


def brightest(image):
    """The row and column of the brightest pixel of the sphere."""
    light = image[..., :3].astype(int).sum(axis=2) * (image[..., 3] == 255)
    return np.unravel_index(light.argmax(), light.shape)


def test_highlights_and_metals():
    options = {"style": "fast", "background": None}
    row, column = brightest(one_sphere((0.5, 0.5, 0.5), **options))
    assert row < 100 and column < 100  # from the upper left
    moved = one_sphere((0.5, 0.5, 0.5), highlight=(1.8, -1.5, 0.2), **options)
    row, column = brightest(moved)
    assert row > 100 and column > 100
    # a metal: 40 % of the diffuse light, highlights in its own color
    color = np.array([0.9, 0.5, 0.1])
    metal = one_sphere(color, style="fast", metalness=1.0).astype(float) / 255
    light = np.array([-1.8, 1.5, -0.2]) / np.linalg.norm([-1.8, 1.5, -0.2])
    peak = (light + np.array([0, 0, 1])) / np.linalg.norm(light + np.array([0, 0, 1]))
    np.testing.assert_allclose(metal[100, 100, :3], 0.4 * color, atol=0.02)
    np.testing.assert_allclose(
        metal[sphere_pixel(peak)][:3],
        0.4 * color * (0.4 + 0.6 * peak[2]) + 0.25 * color,
        atol=0.02,
    )


def test_base_light_and_srgb():
    color = np.array([0.2, 0.3, 0.4])
    options = {"style": "fast", "specular": 0.0, "ambient": 0.2, "diffuse": 0.3}
    # in sRGB, the lights scale the color as given
    lit = one_sphere(color, base_light=0.5, **options).astype(float) / 255
    np.testing.assert_allclose(lit[100, 100, :3], color * 1.0, atol=0.01)
    # in linear light, they scale its linear value
    linear = one_sphere(color, srgb=False, **options).astype(float) / 255
    np.testing.assert_allclose(
        linear[100, 100, :3], srgb(0.5 * srgb_to_linear(color)), atol=0.01
    )


def test_views_and_perspective():
    scene = Scene()
    scene.add_spheres([[1.0, 2, 3]], 0.5, (1.0, 0.5, 0))
    for view in list(VIEWS) + [(1.0, 2.0, -0.5)]:
        image = scene.render(
            width=120, height=80, view=view, background=None, style="fast"
        )
        y, x = np.nonzero(image[..., 3] > 127)
        assert abs(x.mean() - 59.5) < 1.5 and abs(y.mean() - 39.5) < 1.5
    # along the axis up: another up is taken
    assert scene.render(
        width=40, height=40, view=(0, 0, -1), up=(0, 0, 1), style="fast"
    ).shape == (40, 40, 4)
    two = Scene()
    two.add_spheres([[-3.0, 0, 0], [3.0, 0, -12]], 1.0, (0.5, 0.5, 0.5), ids=[0, 1])
    image = two.render(
        width=200,
        height=100,
        view="top",
        perspective=True,
        background=None,
        style="fast",
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

    assert list(STYLES) == ["fast", "matte", "shiny", "flat"]
    flat = style_settings("flat")
    assert flat["flat"] is True and flat["outline_width"] == 1.0
    assert flat["supersample"] == 2 and flat["ao_distance"] is None
    assert flat["highlight"] is None and flat["metalness"] == 0.0
    assert style_settings() == style_settings("matte")
    assert all(set(style_settings(name)) == set(flat) for name in STYLES)
    # a copy: changing it changes no style
    flat["flat"] = False
    assert style_settings("flat")["flat"] is True
    # the settings render the same picture as the style
    scene = Scene()
    scene.add_spheres([[0.0, 0, 0]], 1.0, (0.2, 0.6, 0.9))
    options = dict(width=40, height=40, view="top")
    np.testing.assert_array_equal(
        scene.render(style="fast", **options),
        scene.render(**style_settings("fast"), **options),
    )
    with pytest.raises(InvalidParameter, match="style must be one of"):
        style_settings("pastel")


def test_png_round_trip(tmp_path):
    image = one_sphere(style="matte")
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
        half_width=1.0, half_height=1.0, perspective=False,
        ambient=0.3, diffuse=0.7,
        specular=0.2, shininess=30.0, metalness=0.0, key=np.ones(3, f), highlight=np.ones(3, f),
        fill=np.ones(3, f), fill_intensity=0.2,
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
    # the ends of the table of viridis
    ramp = colormap([0.0, 0.5, 1.0])
    np.testing.assert_allclose(ramp[0], [0.267004, 0.004874, 0.329415])
    np.testing.assert_allclose(ramp[-1], [0.993248, 0.906157, 0.143936])
    np.testing.assert_allclose(
        colormap([5.0, -1.0], "gray", (0, 1)), [[1, 1, 1], [0, 0, 0]]
    )
    np.testing.assert_allclose(
        colormap([np.nan, np.nan], "gray"), [[0, 0, 0], [0, 0, 0]]
    )
    with pytest.raises(InvalidParameter, match="colormap must be"):
        colormap([1.0], "sunset")
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
    image = render(system, tmp_path / "fe.png", width=160, height=120, style="fast")
    assert image.shape == (120, 160, 4) and (tmp_path / "fe.png").exists()
    half = np.arange(len(system)) < len(system) // 2
    shown = render(
        system,
        atoms=half,
        width=160,
        height=120,
        style="fast",
        box=False,
        background=None,
    )
    some = render(
        system,
        atoms=np.flatnonzero(half),
        width=160,
        height=120,
        style="fast",
        box=False,
        background=None,
    )
    np.testing.assert_array_equal(shown, some)
    with pytest.raises(InvalidParameter, match="mask of atoms"):
        render(system, atoms=np.ones(3, bool))
    bonded = render(system, bonds=2.6, radius=0.4, width=160, height=120, style="fast")
    nl = mdapy.NeighborList.build(system, 2.6)
    np.testing.assert_array_equal(
        bonded,
        render(system, bonds=nl, radius=0.4, width=160, height=120, style="fast"),
    )
    cells = VoronoiAnalysis(polyhedra=True).compute(system).polyhedra
    drawn = render(
        system,
        atoms=[0],
        polyhedra=cells,
        width=160,
        height=120,
        style="fast",
        box=False,
    )
    assert drawn.shape == (120, 160, 4)
    by_edges = render(
        system,
        atoms=[0],
        polyhedra=cells,
        polyhedra_color="edges",
        polyhedra_edges=0.0,
        style="fast",
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
        style="fast",
        **scene_options,
    )
    boxed = render(chain, radius=0.3, box=True, style="fast", **scene_options)
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
    options = dict(view="top", width=120, height=120, background=None, style="fast")
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
    options = dict(view="front", width=160, height=160, background=None, style="fast")
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
    options = dict(view="top", width=160, height=160, background=None, style="fast")
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


def test_vector_arrows():
    """An arrow for each atom shown with a vector, from a column or an
    array; none for a zero vector; in one color or that of each atom."""
    system = build_crystal("fcc", 3.615, elements="Cu", repeat=(2, 2, 2))
    vectors = np.zeros((len(system), 3))
    vectors[:4] = [0.0, 0.0, 6.0]
    moved = system.with_columns(displacement=vectors)
    hidden = np.zeros(len(system), dtype=bool)
    options = dict(view="front", width=120, height=120, background=None, style="fast")
    bare = render(moved, atoms=hidden, **options)
    column = render(moved, atoms=hidden, vectors="displacement", **options)
    np.testing.assert_array_equal(column, bare)  # no atom shown, no arrow
    shown = np.arange(len(system)) < 8
    bare = render(moved, atoms=shown, radius=0.3, **options)
    column = render(moved, atoms=shown, radius=0.3, vectors="displacement", **options)
    array = render(moved, atoms=shown, radius=0.3, vectors=vectors, **options)
    np.testing.assert_array_equal(column, array)
    added = (column[..., 3] > 127) & ~(bare[..., 3] > 127)
    yellow = column[added][:, :3].astype(int)
    yellow = yellow[yellow.max(axis=1) > 100]
    assert len(yellow) > 20 and (yellow[:, 2] < yellow[:, 0] - 50).mean() > 0.9
    tinted = render(
        moved,
        atoms=shown,
        radius=0.3,
        color=(0.0, 0.0, 1.0),
        vectors=vectors,
        vector_color="atom",
        **options,
    )
    blue = tinted[added][:, :3].astype(int)
    blue = blue[blue.max(axis=1) > 100]
    assert len(blue) > 20 and (blue[:, 2] > blue[:, 0] + 50).mean() > 0.9
    # reversed, the arrows point down from the same atoms
    down = render(
        moved, atoms=shown, radius=0.3, vectors=vectors, vector_scale=-1, **options
    )
    assert not np.array_equal(down, column)
    for name, value in [
        ("vectors", np.zeros((3, 3))),
        ("vector_scale", 0),
        ("vector_radius", 0),
        ("vector_alignment", "tail"),
        ("vector_color", "plaid"),
    ]:
        with pytest.raises(InvalidParameter, match=name):
            render(moved, **{"vectors": vectors, name: value})


def test_lines():
    """Polylines as tubes; cut at the periodic faces and moved in with
    wrap_lines, so that nothing leaves the box."""
    box = mdapy.Box(10.0)
    system = mdapy.System([[5.0, 5, 5]], box, elements="Ar")
    path = np.array([[[2.0, 5, 5], [6.0, 5, 6], [13.0, 5, 6], [16.0, 7, 6]]])
    hidden = np.zeros(1, dtype=bool)
    # the same camera for all: the box in the middle, room on either side
    options = dict(
        view="front",
        focus=[5.0, 5, 5],
        span=24.0,
        width=160,
        height=160,
        background=None,
        style="fast",
    )
    bare = render(system, atoms=hidden, **options)
    out = render(system, atoms=hidden, lines=path, line_color="red", **options)
    inside = render(system, atoms=hidden, lines=path, wrap_lines=True, **options)
    framed = np.flatnonzero((bare[..., 3] > 127).any(axis=0))  # the box
    for picture, leaves in ((out, True), (inside, False)):
        covered = (picture[..., 3] > 127) & ~(bare[..., 3] > 127)
        assert covered.sum() > 50
        beyond = covered[:, framed[-1] + 1 :].any() or covered[:, : framed[0]].any()
        assert beyond == leaves
    red = out[(out[..., 3] > 127) & ~(bare[..., 3] > 127)][:, :3].astype(int)
    red = red[red.max(axis=1) > 100]
    assert (red[:, 0] > red[:, 1] + 50).mean() > 0.9
    thin = render(system, atoms=hidden, lines=path, line_radius=0.05, **options)
    assert (thin[..., 3] > 127).sum() < (out[..., 3] > 127).sum()
    for name, value in [("line_radius", 0), ("line_color", "plaid")]:
        with pytest.raises(InvalidParameter, match=name):
            render(system, lines=path, **{name: value})
    # no lines, nothing drawn
    np.testing.assert_array_equal(
        render(system, atoms=hidden, lines=[], **options), bare
    )
    # a color for each line
    two = [path[0], path[0][:2] + [0.0, 0.0, 3.0]]
    both = render(
        system, atoms=hidden, lines=two, line_color=["red", (0, 0, 1)], **options
    )
    drawn = both[(both[..., 3] > 127) & ~(bare[..., 3] > 127)][:, :3].astype(int)
    drawn = drawn[drawn.max(axis=1) > 100]
    assert (drawn[:, 0] > drawn[:, 2] + 50).any() and (
        drawn[:, 2] > drawn[:, 0] + 50
    ).any()
    same = render(system, atoms=hidden, lines=path, line_color=[(1.0, 0, 0)], **options)
    np.testing.assert_array_equal(
        same,
        render(system, atoms=hidden, lines=path, line_color=(1.0, 0, 0), **options),
    )
    for wrong in (["red"], ["red", "blue", "red"], [["red"], "plaid"]):
        with pytest.raises(InvalidParameter, match="one color or 2, one for each line"):
            render(system, lines=two, line_color=wrong)


# ------------------------------------------------------------ global illumination


def test_a_lone_sphere_lit_by_the_sky_and_the_key_light():
    # every bounce off a lone sphere leaves for the sky: no noise at all
    color = np.array([0.2, 0.6, 0.9])
    settings = {
        "ambient": 0.45,
        "diffuse": 0.8,
        "specular": 0.2,
        "shininess": 6.0,
        "metalness": 0.0,
        "key": (-0.45, 0.6, 0.66),
    }
    image = one_sphere(
        color, style="shiny", light_size=0.0, supersample=1, samples=4, **settings
    )
    key = np.array([-0.45, 0.6, 0.66]) / np.linalg.norm([-0.45, 0.6, 0.66])
    normal = np.array([0.0, 0, 1])  # at the middle, toward the camera
    mirrored = 2 * (normal @ key) * normal - key  # the key light, reflected
    light = (
        srgb_to_linear(color)
        * (settings["ambient"] + settings["diffuse"] * key @ normal)
        + settings["specular"] * (mirrored @ normal) ** settings["shininess"]
    )
    np.testing.assert_allclose(
        image[100, 100, :3], np.round(255 * srgb(np.minimum(light, 1))), atol=2
    )
    with mdapy.num_threads(3):
        again = one_sphere(color, style="shiny", samples=4)
    np.testing.assert_array_equal(again, one_sphere(color, style="shiny", samples=4))
    with pytest.raises(InvalidParameter, match="samples must be"):
        one_sphere(style="shiny", samples=0)
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
            style="shiny",
            key=(-0.45, 0.6, 0.66),
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
            width=120,
            height=120,
            view="top",
            key=(-0.45, 0.6, 0.66),
            light_size=size,
            samples=16,
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
    for style, tinted in (("matte", False), ("shiny", True)):
        rgb = scene.render(style=style, **options)[..., :3].astype(int)
        red = (rgb[..., 0] - rgb[..., 1])[floor]
        # light off the red ball turns the floor next to it red
        share = (red >= 3).mean()
        assert (share > 0.002) == tinted and (share > 0) == tinted, (style, share)
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


def test_bonds_of_the_topology():
    """The bonds of the topology drawn as the same bonds found by a cutoff,
    across the boundaries too."""
    from mdapy import transform
    from mdapy.build import build_crystal

    silicon = build_crystal("diamond", 5.431, elements="Si", repeat=(2, 2, 2))
    bonded = transform.create_bonds(silicon, 2.6)
    options = dict(radius=0.5, bond_radius=0.2, width=120, height=120, view="iso")
    np.testing.assert_array_equal(
        render(bonded, bonds="topology", **options),
        render(silicon, bonds=2.6, **options),
    )
    with pytest.raises(mdapy.errors.MissingProperty, match="topology"):
        render(silicon, bonds="topology", **options)


def closed_bond_pictures(device):
    """A thick bond between two small atoms seen along its axis, and leaning
    towards the eye by 45 degrees: its end, 8 pixels to the Å."""
    box = mdapy.Box(20.0, pbc=False)
    options = dict(
        bonds=4.0,
        bond_radius=1.0,
        radius=0.05,
        box=False,
        view="top",
        focus=(10, 10, 10),
        span=8.0,
        width=64,
        height=64,
        background=None,
        style="fast",
        device=device,
    )
    along = mdapy.System([[10.0, 10, 8], [10.0, 10, 11]], box, elements="Cu")
    leaning = mdapy.System([[8.0, 10, 8], [10.0, 10, 10]], box, elements="Cu")
    return render(along, **options), render(leaning, **options)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_bonds_are_closed_at_their_ends(device):
    """Seen along the axis the end is a disk of the radius; leaning, it is
    one flat color about the end of the axis, not the inside of a tube."""
    if device == "gpu" and not gpu_available():
        pytest.skip("no GPU")
    along, leaning = closed_bond_pictures(device)
    disk = (along[..., 3] > 127).sum()
    assert abs(disk - np.pi * 8**2) < 0.05 * np.pi * 8**2
    # the end of the axis at the middle of the picture, the disk about it an
    # ellipse 8 pixels across and 5.7 along the x axis; the atom hides its
    # center
    rows, columns = np.mgrid[29:35, 29:35]
    ring = (rows - 31.5) ** 2 + (columns - 31.5) ** 2 > 2**2
    ring &= (rows - 31.5) ** 2 + (columns - 31.5) ** 2 < 4**2
    seen = leaning[rows[ring], columns[ring], :3].astype(int)
    assert (leaning[rows[ring], columns[ring], 3] > 250).all()
    assert (seen.max(axis=0) - seen.min(axis=0)).max() <= 2


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_a_camera_far_away_sees_atoms_and_bonds_whole(device):
    """From 3400 Å through a narrow view, an atom and a bond look as they do
    without perspective: their intersections keep their digits however far
    the rays come from, without rings of wrong hits."""
    if device == "gpu" and not gpu_available():
        pytest.skip("no GPU")
    box = mdapy.Box(10.0, pbc=False)
    pair = mdapy.System([[-0.9, 0.0, 0.0], [0.9, 0.0, 0.0]], box, elements="Cu")
    options = dict(width=96, height=64, view=(0.3, 0.2, -1.0), focus=(0, 0, 0), span=3.0,
                   box=False, style="matte", bonds=2.0, device=device)  # fmt: skip
    near = render(pair, perspective=False, **options).astype(int)
    far = render(pair, perspective=True, fov=0.05, **options).astype(int)
    difference = np.abs(near - far).max(axis=2)
    assert (difference > 8).mean() < 0.01 and np.abs(near - far).mean() < 0.5


# ------------------------------------------------------------ on the GPU

needs_gpu = pytest.mark.skipif(not gpu_available(), reason="no GPU")


@needs_gpu
@pytest.mark.parametrize(
    "options",
    [
        {"style": "matte"},
        {"style": "fast", "background": None},
        {"style": "flat", "perspective": True, "fog": 0.5},
        {"style": "matte", "toon_levels": 3, "outline_width": 1.2, "ao_samples": 0},
        {"style": "shiny", "samples": 4},
        {"style": "shiny", "samples": 4, "metalness": 1.0, "highlight": (1, -1, 1)},
        {"style": "matte", "light_size": 0.2, "samples": 4},
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
    # auto takes the GPU; a software one need not repeat its edges to the last bit
    auto = scene.render(width=96, height=80, device="auto", **options).astype(int)
    difference = np.abs(auto - gpu).max(axis=2)
    assert (difference > 2).mean() < 0.01 and np.abs(auto - gpu).mean() < 0.5


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


@needs_gpu
def test_the_gpu_draws_scenes_of_one_kind():
    """Spheres alone, and triangles alone: the other kinds are empty."""
    rng = np.random.default_rng(1)
    spheres = Scene()
    spheres.add_spheres(rng.random((40, 3)) * 10, 0.6, (1, 0, 0))
    triangles = Scene()
    corner = rng.random((15, 3)) * 10
    triangles.add_triangles(corner[:, None, :] + rng.normal(size=(15, 3, 3)), (0, 0, 1))
    for scene in (spheres, triangles):
        cpu = scene.render(width=64, height=48).astype(int)
        gpu = scene.render(width=64, height=48, device="gpu").astype(int)
        assert (np.abs(cpu - gpu).max(axis=2) > 2).mean() < 0.01


@needs_gpu
def test_the_gpu_device():
    """The GPU named with its interface; with its ray tracing units where it
    has them, unless MDAPY_GPU_RAY_TRACING=0."""
    name = _native.gpu_device()
    interface = {"darwin": "Metal", "win32": "Direct3D 12"}.get(sys.platform, "Vulkan")
    assert name.endswith((f" ({interface})", f" ({interface}, ray tracing)"))
    if os.environ.get("MDAPY_GPU_RAY_TRACING") == "0":
        assert "ray tracing" not in name


@needs_gpu
def test_the_gpu_without_its_ray_tracing_units():
    """With MDAPY_GPU_RAY_TRACING=0 the GPU traces through the hierarchy of
    the shader, as GPUs without ray tracing units do: the same pictures."""
    if "ray tracing" not in _native.gpu_device():
        pytest.skip("no ray tracing units")
    tests = [
        f"{__file__}::{name}"
        for name in (
            "test_the_gpu_draws_what_the_processor_draws",
            "test_the_gpu_draws_scenes_of_one_kind",
            "test_the_gpu_device",
        )
    ]
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
        env=dict(os.environ, MDAPY_GPU_RAY_TRACING="0"),
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout


def test_no_gpu_has_no_name():
    if not gpu_available():
        assert _native.gpu_device() == ""


@pytest.mark.skipif("MDAPY_SOFTWARE_GPU" in os.environ, reason="asked for one")
def test_a_driver_on_the_processor_is_no_gpu():
    # what Windows offers in a virtual machine without a GPU, 12 to 30 times slower
    # than the threads of mdapy
    assert "Basic Render" not in _native.gpu_device()


SHADER = Path(__file__).parents[2] / "src/cpp/visualization/render_vulkan.comp"


def test_the_vulkan_shaders_are_compiled_from_their_source(tmp_path):
    """render_vulkan.spv and render_vulkan_rt.spv are what the commands written
    in render_vulkan.comp make, with the glslc named there."""
    text = SHADER.read_text()
    version = re.search(r"glslc of shaderc ([\d.]+)", text).group(1)
    commands = [line.split() for line in re.findall(r"^//\s+(glslc .*)$", text, re.M)]
    assert len(commands) == 2
    beside_python = str(Path(sys.executable).parent)
    glslc = shutil.which("glslc") or shutil.which("glslc", path=beside_python)
    if glslc is None:
        pytest.skip("no glslc")
    found = subprocess.run([glslc, "--version"], capture_output=True, text=True).stdout
    if f"shaderc v{version} " not in found:
        pytest.skip(f"glslc is not of shaderc {version}")
    for command in commands:
        output = command[-1]
        compiled = tmp_path / output
        names = {"render_vulkan.comp": str(SHADER), output: str(compiled)}
        subprocess.run([glslc] + [names.get(a, a) for a in command[1:]], check=True)
        assert compiled.read_bytes() == (SHADER.parent / output).read_bytes(), output


def test_the_direct3d_shaders_are_compiled_from_their_source(tmp_path):
    """render_d3d12.dxil and render_d3d12_rt.dxil are what the commands written
    in render_d3d12.hlsl make, with the dxc of the Windows SDK named there."""
    source = SHADER.with_name("render_d3d12.hlsl")
    text = source.read_text()
    version = re.search(r"the Windows SDK ([\d.]+)", text).group(1)
    commands = [line.split() for line in re.findall(r"^//\s+(dxc .*)$", text, re.M)]
    assert len(commands) == 2
    dxc = Path(f"C:/Program Files (x86)/Windows Kits/10/bin/{version}/x64/dxc.exe")
    if not dxc.exists():
        pytest.skip("no dxc")
    found = subprocess.run(
        [str(dxc), "--version"], capture_output=True, text=True
    ).stdout
    wanted = re.search(r"dxcompiler ([\d.]+)", text).group(1)
    if wanted not in found:
        pytest.skip(f"dxc is not of dxcompiler {wanted}")
    for command in commands:
        output = command[command.index("-Fo") + 1]
        compiled = tmp_path / output
        arguments = [str(compiled) if a == output else a for a in command[1:]]
        subprocess.run([str(dxc), *arguments], check=True, cwd=source.parent)
        assert compiled.read_bytes() == (source.parent / output).read_bytes(), output


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


def _fake_ipython(monkeypatch):
    """A stand-in for IPython.display, which keeps what it is asked to show."""
    shown = []

    class Image:
        def __init__(self, data, width=None):
            self.data, self.width = data, width

    module = types.ModuleType("IPython.display")
    module.Image = Image
    module.display = shown.append
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", module)
    return shown


def test_show_image_of_a_picture_and_of_a_file(monkeypatch, tmp_path):
    shown = _fake_ipython(monkeypatch)
    picture = np.zeros((3, 2, 4), dtype=np.uint8)
    picture[..., 0] = 200
    picture[..., 3] = 255
    show_image(picture, width=40)
    assert shown[0].width == 40
    assert shown[0].data.startswith(b"\x89PNG\r\n\x1a\n")
    path = tmp_path / "picture.png"
    write_png(path, picture)
    assert path.read_bytes() == shown[0].data  # the same file as write_png
    show_image(path)
    show_image(str(path))
    assert shown[1].data == shown[2].data == shown[0].data
    assert shown[1].width is None


def test_show_image_errors(monkeypatch, tmp_path):
    _fake_ipython(monkeypatch)
    with pytest.raises(InvalidParameter, match="RGBA"):
        show_image(np.zeros((3, 2)))
    with pytest.raises(FileNotFoundError):
        show_image(tmp_path / "missing.png")
    monkeypatch.setitem(sys.modules, "IPython.display", None)
    with pytest.raises(ImportError, match="ipython"):
        show_image(np.zeros((3, 2, 4), dtype=np.uint8))


@pytest.mark.parametrize(
    "options, message",
    [
        ({"radius": np.where(np.arange(4) % 2, 1.0, np.nan)}, "radii must be finite"),
        ({"radius": np.nan}, "radii must be finite"),
        ({"outline_width": 1000.0}, "outline_width must be from 0 to 100"),
        ({"outline_width": -1.0}, "outline_width must be from 0 to 100"),
        ({"radius": np.inf}, "radii must be finite"),
        ({"radius": -1.0}, "radii must not be negative"),
        ({"radius": np.ones(3)}, "radius must be one number"),
        ({"box_radius": np.nan}, "radii must be finite"),
        ({"bond_radius": np.nan, "bonds": 5.0}, "radii must be finite"),
        ({"zoom": 0}, "zoom"),
        ({"zoom": np.nan}, "zoom"),
        ({"zoom": np.inf}, "zoom"),
        ({"view": (0, 0, 0)}, "view"),
        ({"view": (np.nan, 0, 0)}, "view"),
        ({"up": (0, 0, 0)}, "up"),
        ({"perspective": True, "fov": 0}, "fov"),
        ({"perspective": True, "fov": 180}, "fov"),
        ({"width": 2**31}, "at most"),
        ({"supersample": 0}, "supersample"),
        ({"ao_samples": -1}, "ao_samples"),
        ({"toon_levels": -1}, "toon_levels"),
        ({"fog": np.nan}, "fog"),
        ({"key": (0, 0, 0)}, "key"),
        ({"background": (np.nan, 0, 0)}, "background"),
        ({"atoms": [0.5]}, "mask or indices"),
        ({"atoms": [9]}, "below 4"),
        ({"color": np.array(["a"] * 4)}, "must hold numbers"),
    ],
)
def test_parameters_that_make_no_picture(options, message):
    # a radius of NaN crashed the building of the hierarchy
    system = mdapy.System(
        [[1.0, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 2]], mdapy.Box(5.0), elements="Cu"
    )
    with pytest.raises(InvalidParameter, match=message):
        render(system, **{"width": 16, "height": 16, **options})


def test_a_value_of_nan_still_colors_an_atom():
    # what a column holds for atoms without a value is no mistake
    system = mdapy.System([[1.0, 1, 1], [2, 1, 1]], mdapy.Box(5.0), elements="Cu")
    picture = render(system, color=np.array([np.nan, 1.0]), width=16, height=16)
    assert np.isfinite(picture).all()


def _legend_system():
    system = build_crystal("fcc", 3.6, elements="Cu", repeat=(3, 3, 3))
    return system.with_columns(height=system["positions"][:, 2])


LEGEND = dict(width=240, height=180)


def test_a_legend_has_a_column_of_its_own(tmp_path):
    """The legend never covers the atoms: the scene is framed beside it,
    in the width the column leaves."""
    pytest.importorskip("matplotlib")
    from mdapy.visualization._legend import _legend_width

    system = _legend_system()
    cna = CommonNeighborAnalysis().compute(system)
    path = tmp_path / "legend.png"
    marked = render(system, path, color=cna, legend=True, **LEGEND)
    assert path.stat().st_size > 0
    assert marked.shape == (180, 240, 4) and marked.dtype == np.uint8
    column = _legend_width(180, names=["fcc"])
    assert 30 < column < 120
    beside = render(system, color=cna, width=240 - column, height=180)
    np.testing.assert_array_equal(marked[:, column:], beside)
    # the column: the text at its top, the background below
    assert (marked[:30, :column, :3] < 128).any()
    assert (marked[60:, :column] == 255).all()
    # a longer name, a wider column
    longer = _legend_width(180, names=["fcc", "hexagonal_diamond_first_neighbor"])
    assert longer > column + 100
    titled = _legend_width(180, title="a long title of the values", bar=(0, 1, "jet"))
    assert titled > _legend_width(180, bar=(0, 1, "jet"))


def test_a_legend_wider_than_the_picture():
    pytest.importorskip("matplotlib")
    system = _legend_system()
    picture = render(system, legend={"a long name " * 4: "red"}, width=60, height=60)
    assert picture.shape == (60, 60, 4)
    with pytest.raises(InvalidParameter, match="width and height must be positive"):
        render(system, legend=True, width=0, height=60)


def test_a_legend_on_a_transparent_background():
    pytest.importorskip("matplotlib")
    from mdapy.visualization._legend import _legend_width

    system = _legend_system()
    picture = render(system, legend="Cu", background=None, **LEGEND)
    column = _legend_width(180, names=["Cu"], title="Cu")
    # dark text over nothing; the rest of the column stays transparent
    assert (picture[:30, :column, 3] > 200).any()
    assert (picture[80:, :column, 3] == 0).all()


def test_a_legend_names_what_is_shown(monkeypatch):
    pytest.importorskip("matplotlib")
    system = _legend_system()
    mixed = system.with_columns(element=(["Cu", "Ni"] * len(system))[: len(system)])
    cna = CommonNeighborAnalysis().compute(system)
    # of the atoms shown: no atoms of one kind, no name of it
    drawn = {}
    original = render_module._with_legend

    def spy(image, background, **items):
        drawn.update(items)
        return original(image, background, **items)

    monkeypatch.setattr(render_module, "_with_legend", spy)
    render(mixed, legend=True, **LEGEND)
    assert sorted(drawn["names"]) == ["Cu", "Ni"]
    render(mixed, atoms=mixed["element"] == "Ni", legend=True, **LEGEND)
    assert list(drawn["names"]) == ["Ni"]
    render(system, color=cna, legend="Structure", **LEGEND)
    assert list(drawn["names"]) == ["fcc"] and drawn["title"] == "Structure"
    render(system, color="height", legend=True, **LEGEND)
    low, high, name = drawn["bar"]
    assert drawn["title"] == "height" and name == "viridis"
    assert low == pytest.approx(system["height"].min())
    assert high == pytest.approx(system["height"].max())
    values = system["height"] * 2
    render(
        system,
        color=values,
        value_range=(0, 9),
        colormap="plasma",
        legend="z",
        **LEGEND,
    )
    assert drawn["bar"] == (0.0, 9.0, "plasma") and drawn["title"] == "z"
    render(system, color="red", legend={"mine": "red", "yours": (0, 0, 1)}, **LEGEND)
    assert drawn["names"] == ["mine", "yours"]
    # discrete values: the values shown, in the colors they have among all of them
    kinds = np.arange(len(system)) % 3 + 5
    render(system, color=kinds, colormap="labels", legend="kind", **LEGEND)
    assert drawn["names"] == ["5", "6", "7"] and drawn["title"] == "kind"
    np.testing.assert_array_equal(drawn["colors"], LABEL_COLORS[:3])
    render(
        system, atoms=kinds > 5, color=kinds, colormap="labels", legend=True, **LEGEND
    )
    assert drawn["names"] == ["6", "7"]
    np.testing.assert_array_equal(drawn["colors"], LABEL_COLORS[1:3])
    render(
        system.with_columns(half=(kinds > 5) * 0.5),
        color="half",
        colormap="labels",
        legend=True,
        **LEGEND,
    )
    assert drawn["names"] == ["0.0", "0.5"] and drawn["title"] == "half"
    with pytest.raises(InvalidParameter, match="at most 12 values, not 108"):
        render(system, color=np.arange(108), colormap="labels", legend=True, **LEGEND)


@pytest.mark.parametrize(
    "color",
    ["red", (1.0, 0.0, 0.0), np.zeros((108, 3))],
    ids=["a name", "one color", "colors of the atoms"],
)
def test_a_legend_of_colors_it_cannot_explain(color):
    with pytest.raises(InvalidParameter, match="mapping of names to colors"):
        render(_legend_system(), color=color, legend=True, **LEGEND)


def test_a_legend_on_a_dark_background():
    pytest.importorskip("matplotlib")
    system = _legend_system()
    light = render(system, legend="Cu", **LEGEND)
    dark = render(system, legend="Cu", background="black", **LEGEND)
    # its text in the color that shows: bright pixels in the dark corner
    assert dark[:12, :60, :3].max() > 200 and light[:12, :60, :3].min() < 100


def test_no_legend_is_the_default_and_needs_no_matplotlib(monkeypatch):
    system = _legend_system()
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.figure", None)
    monkeypatch.setitem(sys.modules, "matplotlib.backends.backend_agg", None)
    assert render(system, **LEGEND).shape == (180, 240, 4)
    with pytest.raises(ImportError, match="a legend needs matplotlib"):
        render(system, legend=True, **LEGEND)
