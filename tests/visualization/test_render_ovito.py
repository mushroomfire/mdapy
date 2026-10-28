# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Pictures against the renderers of OVITO 3.17.1 with the same camera: the
fast style against its OpenGL renderer, the matte style against its Tachyon
renderer. Skipped where OVITO is not installed."""

import numpy as np
import pytest

from mdapy.build import build_crystal
from mdapy.visualization import render

pytest.importorskip("ovito")
from ovito.data import DataCollection  # noqa: E402
from ovito.pipeline import Pipeline, StaticSource  # noqa: E402
from ovito.qt_compat import QtGui  # noqa: E402
from ovito.vis import OpenGLRenderer, TachyonRenderer, Viewport  # noqa: E402

WIDTH, HEIGHT = 300, 225


@pytest.fixture(scope="module")
def block():
    """A block of fcc atoms in two colors, the same pictures to compare: the
    picture of OVITO and its camera."""
    system = build_crystal("fcc", 3.615, elements="Cu", repeat=(5, 5, 5))
    positions = system["positions"]
    colors = np.where(
        (positions[:, 0] > 9.0)[:, None], [0.4, 1.0, 0.4], [0.95, 0.95, 0.95]
    )
    data = DataCollection()
    data.create_cell(np.hstack([system.box.matrix.T, [[0.0], [0.0], [0.0]]]))
    data.cell.vis.enabled = False
    particles = data.create_particles(count=len(system))
    particles.create_property("Position", data=positions)
    particles.create_property("Color", data=colors)
    particles.create_property("Radius", data=np.full(len(system), 1.28))
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.add_to_scene()
    viewport = Viewport(type=Viewport.Type.Perspective, camera_dir=(-1, -1, -0.8))
    viewport.zoom_all((WIDTH, HEIGHT))
    yield system, colors, viewport
    pipeline.remove_from_scene()


def compare(block, renderer, style):
    """The mean difference over the atoms, of 255, and the mean brightness
    of OVITO's picture and of mdapy's."""
    system, colors, viewport = block
    try:
        image = viewport.render_image(
            size=(WIDTH, HEIGHT), background=(1, 1, 1), renderer=renderer
        )
    except RuntimeError as error:  # no OpenGL without a display
        pytest.skip(f"OVITO cannot render here: {error}")
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    rows = np.frombuffer(image.constBits(), np.uint8).reshape(HEIGHT, -1)
    theirs = rows[:, : 4 * WIDTH].reshape(HEIGHT, WIDTH, 4)[..., :3].astype(float)
    forward = np.array(viewport.camera_dir) / np.linalg.norm(viewport.camera_dir)
    distance = 100.0
    ours = render(
        system,
        style=style,
        color=colors,
        radius=1.28,
        box=False,
        view=forward,
        up=(0, 0, 1),
        perspective=True,
        fov=np.degrees(viewport.fov),
        focus=np.array(viewport.camera_pos) + forward * distance,
        span=2 * distance * np.tan(viewport.fov / 2),
        width=WIDTH,
        height=HEIGHT,
    )[..., :3].astype(float)
    atoms = (theirs.sum(axis=2) < 3 * 250) | (ours.sum(axis=2) < 3 * 250)
    return (
        np.abs(theirs - ours)[atoms].mean(),
        theirs[atoms].mean(),
        ours[atoms].mean(),
    )


def test_fast_is_the_opengl_renderer(block):
    difference, theirs, ours = compare(
        block, OpenGLRenderer(antialiasing_level=2), "fast"
    )
    # the same lighting: only the edges of the atoms differ a little
    assert difference < 1 and abs(theirs - ours) < 1


def test_matte_is_the_tachyon_renderer(block):
    difference, theirs, ours = compare(block, TachyonRenderer(), "matte")
    # the same lighting, the noise of the occlusion apart
    assert difference < 10 and abs(theirs - ours) < 2
