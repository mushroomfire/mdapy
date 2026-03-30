"""
mdapy/render.py  -  Python wrapper for the Tachyon ray-tracing renderer (Tachyon 0.99.5)
==========================================================================================

Quick start
-----------
::

    import mdapy as mp
    from mdapy.render import TachyonRender

    sys = mp.System("model.xyz")
    ren = TachyonRender(width=800, height=600)
    img = ren.render_system(sys)  # numpy (H, W, 4) uint8 RGBA

    from PIL import Image
    Image.fromarray(img).save("out.png")

Camera parameters (OVITO-compatible)
--------------------------------------
::

    from mdapy.render import CameraParams, preset_camera
    import math

    # Perspective (default)
    cam = CameraParams(
        is_perspective=True,
        field_of_view=math.radians(40),  # vertical FOV in radians
        position=(0, 0, 60),
        direction=(0, 0, -1),
        up=(0, 1, 0),
    )

    # Orthographic
    cam_ortho = CameraParams(
        is_perspective=False,
        field_of_view=25.0,  # viewport half-height in world units
        position=(0, 0, 100),
        direction=(0, 0, -1),
        up=(0, 1, 0),
    )

Preset views matching OVITO GUI defaults
-----------------------------------------
::

    pos = system.get_positions().to_numpy()
    r   = 1.3  # atom radius

    cam = preset_camera("top",         pos, max_radius=r)  # OVITO Top
    cam = preset_camera("left",        pos, max_radius=r)  # OVITO Left
    cam = preset_camera("perspective", pos, max_radius=r)  # OVITO Perspective
    cam = preset_camera("orthographic",pos, max_radius=r)  # OVITO Ortho

Custom box edge appearance
---------------------------
::

    img = ren.render_system(
        sys,
        draw_box=True,
        box_edge_radius=0.08,
        box_color=(0.0, 0.0, 0.0),   # black, (R,G,B) or (R,G,B,A)
    )
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import numpy as np

import mdapy._tachyon as _tachyon_mod
from mdapy._tachyon import (
    TachyonRenderer as _TachyonRenderer,
    RenderParams as _RenderParams,
    CameraParams as _CameraParams,
    Vec3 as _Vec3,
)

# GPU backend: only available when mdapy was built with OptiX.
_HAS_OPTIX: bool = _tachyon_mod.has_optix()
if _HAS_OPTIX:
    print('have optix')
    from mdapy._tachyon import TachyonOptiXRenderer as _TachyonOptiXRenderer


def is_gpu_available() -> bool:
    """Return True if this mdapy build supports GPU rendering (OptiX 7+)."""
    return _HAS_OPTIX


# ─── CameraParams ─────────────────────────────────────────────────────────────
class CameraParams:
    """
    Camera parameters compatible with OVITO's ViewProjectionParameters.

    Perspective mode
    ~~~~~~~~~~~~~~~~
    ``field_of_view``: vertical angle in **radians**.
    Tachyon zoom = 0.5 / tan(fov/2)  — same formula as OVITO.

    Orthographic mode
    ~~~~~~~~~~~~~~~~~
    ``field_of_view``: viewport half-height in world units.
    Tachyon zoom = 0.5 / field_of_view.
    """

    def __init__(
        self,
        is_perspective: bool = True,
        field_of_view: float = math.radians(40),
        position: Tuple[float, float, float] = (0.0, 0.0, 50.0),
        direction: Tuple[float, float, float] = (0.0, 0.0, -1.0),
        up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        znear: float = 0.0,
        dof_enabled: bool = False,
        dof_focal_len: float = 40.0,
        dof_aperture: float = 0.01,
    ):
        self.is_perspective = bool(is_perspective)
        self.field_of_view = float(field_of_view)
        self.position = tuple(float(v) for v in position)
        self.direction = tuple(float(v) for v in direction)
        self.up = tuple(float(v) for v in up)
        self.znear = float(znear)
        self.dof_enabled = bool(dof_enabled)
        self.dof_focal_len = float(dof_focal_len)
        self.dof_aperture = float(dof_aperture)

    def _to_cpp(self) -> _CameraParams:
        cp = _CameraParams()
        cp.is_perspective = self.is_perspective
        cp.field_of_view = self.field_of_view
        cp.position = _Vec3(*self.position)
        cp.direction = _Vec3(*self.direction)
        cp.up = _Vec3(*self.up)
        cp.znear = self.znear
        cp.dof_enabled = self.dof_enabled
        cp.dof_focal_len = self.dof_focal_len
        cp.dof_aperture = self.dof_aperture
        return cp

    def __repr__(self):
        mode = "perspective" if self.is_perspective else "orthographic"
        fov = (
            math.degrees(self.field_of_view)
            if self.is_perspective
            else self.field_of_view
        )
        unit = "deg" if self.is_perspective else "world units"
        return f"CameraParams({mode}, fov={fov:.1f}{unit}, pos={self.position})"


# ---------------------------------------------------------------------------
# TachyonRender
# ---------------------------------------------------------------------------
class TachyonRender:
    """
    mdapy Tachyon ray-tracing renderer (Tachyon 0.99.5).

    Supports two backends selectable via the ``backend`` parameter:

    ``"cpu"``  (default)
        Classic multi-threaded CPU renderer.  Always available.
        Uses Tachyon's POSIX-thread parallelism.

    ``"gpu"``
        NVIDIA OptiX 7 GPU ray-tracing backend (RTX hardware acceleration).
        Only available when mdapy was compiled with CUDA + OptiX support.
        Raises ``RuntimeError`` at construction time if no CUDA GPU is found.
        Check availability with :func:`is_gpu_available`.

    ``"auto"``
        Use GPU if available, fall back to CPU silently.

    Parameters
    ----------
    width, height : int
        Output image resolution in pixels.
    backend : str
        Rendering backend: ``"cpu"``, ``"gpu"``, or ``"auto"``.  Default ``"cpu"``.
    antialiasing : bool
        Enable anti-aliasing.  Default True.
    aa_samples : int
        Anti-aliasing samples per pixel.  Default 12.
    ao : bool
        Enable ambient occlusion.  Default True.
    ao_samples : int
        AO samples.  Default 12.
    ao_brightness : float
        Sky-light brightness for AO.  Default 0.8.
    ao_max_dist : float
        Maximum AO occlusion distance.  Default unlimited.
    shadows : bool
        Enable hard shadows (CPU only; GPU always uses shadows).  Default True.
    direct_light_intensity : float
        Directional light intensity.  Default 0.9.
    background : tuple
        Background colour (R, G, B) or (R, G, B, A) in [0, 1].  Default black.
    num_threads : int
        CPU thread count (CPU backend only).  0 = auto.  Default 0.

    Examples
    --------
    >>> ren_cpu = TachyonRender(width=800, height=600)                  # CPU
    >>> ren_gpu = TachyonRender(width=800, height=600, backend="gpu")   # GPU
    >>> ren_auto= TachyonRender(width=800, height=600, backend="auto")  # auto
    >>> print(ren_cpu.backend)
    cpu
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        backend: str = "cpu",
        antialiasing: bool = True,
        aa_samples: int = 12,
        ao: bool = True,
        ao_samples: int = 12,
        ao_brightness: float = 0.8,
        ao_max_dist: float = 3.402823e+38,  # RT_AO_MAXDIST_UNLIMITED
        shadows: bool = True,
        direct_light_intensity: float = 0.9,
        background: tuple = (0.0, 0.0, 0.0),
        num_threads: int = 0,
    ):
        backend = backend.lower().strip()
        if backend not in ("cpu", "gpu", "auto"):
            raise ValueError(f"backend must be 'cpu', 'gpu', or 'auto', got {backend!r}")

        # Resolve "auto": pick GPU if available, else CPU.
        if backend == "auto":
            backend = "gpu" if _HAS_OPTIX else "cpu"

        if backend == "gpu":
            if not _HAS_OPTIX:
                raise RuntimeError(
                    "GPU backend requested but mdapy was not compiled with "
                    "OptiX support.  Rebuild with CUDA + OptiX, or use "
                    "backend='cpu'."
                )
            self._renderer = _TachyonOptiXRenderer()
        else:
            self._renderer = _TachyonRenderer()

        self._backend = backend   # "cpu" or "gpu" (resolved)

        rp = _RenderParams()
        rp.width = int(width)
        rp.height = int(height)
        rp.antialiasing_enabled = bool(antialiasing)
        rp.antialiasing_samples = int(aa_samples)
        rp.ao_enabled = bool(ao)
        rp.ao_samples = int(ao_samples)
        rp.ao_brightness = float(ao_brightness)
        rp.ao_max_dist = float(ao_max_dist)
        rp.shadows_enabled = bool(shadows)
        rp.direct_light_intensity = float(direct_light_intensity)
        bg = tuple(background)
        rp.bg_r = float(bg[0])
        rp.bg_g = float(bg[1])
        rp.bg_b = float(bg[2])
        rp.bg_a = float(bg[3]) if len(bg) > 3 else 1.0
        rp.num_threads = int(num_threads)
        self._rp = rp

    @property
    def backend(self) -> str:
        """The active rendering backend: ``'cpu'`` or ``'gpu'``."""
        return self._backend

    def __repr__(self) -> str:
        rp = self._rp
        return (f"TachyonRender(backend={self._backend!r}, "
                f"size=({rp.width}x{rp.height}), "
                f"ao={rp.ao_enabled}, aa={rp.antialiasing_enabled})")

    # Main rendering interface
    def render(
        self,
        positions: np.ndarray,
        colors: np.ndarray,
        radii: np.ndarray,
        camera: Optional[CameraParams] = None,
        box_edges: Optional[np.ndarray] = None,
        box_edge_radius: float = 0.05,
        box_color: tuple = (1.0, 1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """
        Render spherical particles and optional simulation-cell box edges.

        Parameters
        ----------
        positions       : (N, 3) float64  Particle positions.
        colors          : (N, 4) float32  Per-particle RGBA colour, values in [0, 1].
        radii           : (N,)  float32   Per-particle radius.
        camera          : CameraParams or None.  If None, a perspective camera
                          is generated automatically from the bounding box.
        box_edges       : (M, 2, 3) float64 or None.  Pairs of endpoints for each
                          box edge cylinder.  Pass None to skip drawing.
        box_edge_radius : float.  Cylinder radius for box edges.  Default 0.05.
        box_color       : tuple (R, G, B) or (R, G, B, A), values in [0, 1].
                          Default opaque white (1, 1, 1, 1).

        Returns
        -------
        numpy.ndarray, shape (H, W, 4), dtype uint8, RGBA image.
        """
        positions = np.ascontiguousarray(positions, dtype=np.float64)
        colors    = np.ascontiguousarray(colors,    dtype=np.float32)
        radii     = np.ascontiguousarray(radii,     dtype=np.float32)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be (N,3), got {positions.shape}")
        if colors.ndim != 2 or colors.shape[1] != 4:
            raise ValueError(f"colors must be (N,4), got {colors.shape}")
        if radii.ndim != 1:
            raise ValueError(f"radii must be (N,), got {radii.shape}")

        if camera is None:
            max_r = float(radii.max()) if len(radii) > 0 else 0.0
            camera = _auto_camera(positions, max_radius=max_r)
        cpp_cam = camera._to_cpp()

        # Validate and prepare box_edges
        if box_edges is not None:
            box_edges = np.ascontiguousarray(box_edges, dtype=np.float64)
            if box_edges.ndim != 3 or box_edges.shape[1:] != (2, 3):
                raise ValueError(f"box_edges must be (M,2,3), got {box_edges.shape}")
            if box_edges.shape[0] == 0:
                box_edges = None

        # Parse box edge colour
        bc = tuple(float(v) for v in box_color)
        box_r = bc[0]
        box_g = bc[1]
        box_b = bc[2]
        box_a = bc[3] if len(bc) > 3 else 1.0

        img = self._renderer.render(
            self._rp,
            cpp_cam,
            positions,
            colors,
            radii,
            box_edges,
            float(box_edge_radius),
            box_r, box_g, box_b, box_a,
        )
        return np.array(img, copy=False)

    # Convenience wrapper for mdapy.System
    def render_system(
        self,
        system,
        colors: Optional[np.ndarray] = None,
        radii: Optional[np.ndarray] = None,
        camera: Optional[CameraParams] = None,
        draw_box: bool = True,
        box_edge_radius: float = 0.05,
        box_color: tuple = (1.0, 1.0, 1.0, 1.0),
        default_radius: float = 1.0,
    ) -> np.ndarray:
        """
        Render a ``mdapy.System`` object in one call.

        Parameters
        ----------
        system          : mp.System  The atomistic system to render.
        colors          : (N, 4) float32 or None.  If None, Jmol colours are
                          assigned by element type.
        radii           : (N,) float32 or None.  If None, ``default_radius`` is
                          used for every particle.
        camera          : CameraParams or None.  Auto-generated if not provided.
        draw_box        : bool.  Whether to draw simulation-cell edges.  Default True.
        box_edge_radius : float.  Cylinder radius for cell edges.  Default 0.05.
        box_color       : tuple (R, G, B) or (R, G, B, A), values in [0, 1].
                          Default opaque white (1, 1, 1, 1).
        default_radius  : float.  Fallback radius when ``radii`` is None.  Default 1.0.
        """
        pos = system.get_positions().to_numpy()  # (N,3)
        N = len(pos)

        if colors is None:
            colors = _default_colors(system, N)
        colors = np.ascontiguousarray(colors, dtype=np.float32)

        if radii is None:
            radii = np.full(N, default_radius, dtype=np.float32)
        radii = np.ascontiguousarray(radii, dtype=np.float32)

        box_edges = _box_edges(system) if draw_box else None

        return self.render(
            pos,
            colors,
            radii,
            camera=camera,
            box_edges=box_edges,
            box_edge_radius=box_edge_radius,
            box_color=box_color,
        )


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def _bbox(positions: np.ndarray, max_radius: float = 0.0):
    """
    Compute the bounding box of particle positions.

    Parameters
    ----------
    positions  : (N,3) float64
    max_radius : largest particle radius; added to each half-extent so that
                 the camera frustum covers the full sphere, not just the centre.

    Returns
    -------
    center : (3,) centroid of the bounding box
    half   : (3,) half-extents (already inflated by max_radius)
    pmin   : (3,) minimum corner of the coordinate range
    pmax   : (3,) maximum corner of the coordinate range
    """
    pmin = positions.min(axis=0)
    pmax = positions.max(axis=0)
    center = (pmin + pmax) * 0.5
    half = (pmax - pmin) * 0.5 + max_radius
    return center, half, pmin, pmax


def _auto_camera(positions: np.ndarray, max_radius: float = 0.0) -> CameraParams:
    """Return a perspective camera looking at the structure from +Z (OVITO default)."""
    return preset_camera("perspective", positions, max_radius=max_radius)


# ---------------------------------------------------------------------------
# Preset camera factory
# ---------------------------------------------------------------------------

#: All supported preset view names.
PRESET_VIEWS = ("perspective", "orthographic", "top", "bottom",
                "front", "back", "left", "right")


def preset_camera(
    view: str,
    positions: np.ndarray,
    fov_deg: float = 40.0,
    margin: float = 1.0,
    max_radius: float = 0.0,
) -> CameraParams:
    """
    Build a camera that matches one of OVITO's four default viewport orientations.

    OVITO coordinate convention
    ---------------------------
    OVITO uses a right-handed coordinate system where **Z is the global up axis**
    (configurable, but Z by default).  The four standard viewports are:

    +-----------------+-------------------------------+------------------+----------+
    | Name            | Meaning                       | camera_dir (OVITO)| up      |
    +=================+===============================+==================+==========+
    | ``"top"``       | Look down the -Z axis         | (0, 0, -1)       | (0,1,0)  |
    |                 | → sees XY plane               |                  |          |
    +-----------------+-------------------------------+------------------+----------+
    | ``"left"``      | Look along -X                 | (-1, 0, 0)       | (0,1,0)  |
    |                 | → sees ZY plane (Z→right)     |                  |          |
    +-----------------+-------------------------------+------------------+----------+
    | ``"perspective"``| Tilted view from +X+Y+Z      | (-1,-1,-1)/√3    | (0,0,1)  |
    | / ``"ortho"``   | → 3D isometric look           |                  |          |
    +-----------------+-------------------------------+------------------+----------+

    All eight available view names
    -------------------------------
    - ``"perspective"``  – Perspective projection, tilted isometric camera
                           (matches OVITO *Perspective* viewport)
    - ``"orthographic"`` – Same direction as perspective but parallel projection
                           (matches OVITO *Ortho* viewport)
    - ``"top"``          – Orthographic, camera_dir=(0,0,-1), up=(0,1,0)
                           looks at the XY plane
                           (matches OVITO *Top* viewport)
    - ``"bottom"``       – Orthographic, camera_dir=(0,0,+1), up=(0,1,0)
    - ``"left"``         – Orthographic, camera_dir=(-1,0,0), up=(0,1,0)
                           looks at the ZY plane, Z points right, Y points up
                           (matches OVITO *Left* viewport)
    - ``"right"``        – Orthographic, camera_dir=(+1,0,0), up=(0,1,0)
    - ``"front"``        – Orthographic, camera_dir=(0,-1,0), up=(0,0,1)
                           looks at the XZ plane from above in the Y direction
    - ``"back"``         – Orthographic, camera_dir=(0,+1,0), up=(0,0,1)

    Parameters
    ----------
    view       : str
        View name.  Must be one of :data:`PRESET_VIEWS`.
    positions  : (N, 3) array_like
        Particle positions in world coordinates.
    fov_deg    : float, optional
        Vertical field of view in *degrees* for the perspective camera.
        Default is 40°.
    margin     : float, optional
        Extra padding around the structure in world units (Å).
        Applied to all views.  Default is 1.0 Å.
    max_radius : float, optional
        Largest particle radius.  Pass ``radii.max()`` so that edge atoms
        are not clipped.  Default 0 (uses only atom centres).

    Returns
    -------
    CameraParams

    Notes
    -----
    **About** ``CameraParams.znear`` **(orthographic cross-section clipping)**

    ``znear`` defaults to 0 and you normally do not need to change it.
    Setting it to a positive value shifts the camera plane forward along the
    view direction by that many world units, effectively clipping away the
    front portion of the structure.  This is useful for cross-section renders::

        cam = preset_camera("top", pos, max_radius=1.3)
        cam.znear = 9.0   # discard atoms in front of the z=9 Å plane
    """
    view = view.lower().strip()
    if view not in PRESET_VIEWS:
        raise ValueError(f"Unknown view '{view}'. Choose from: {PRESET_VIEWS}")

    center, half, pmin, pmax = _bbox(positions, max_radius)

    # ------------------------------------------------------------------
    # Isometric views: perspective and orthographic
    # Both look from the +X+Y+Z octant toward the structure centre,
    # matching OVITO's default Perspective / Ortho viewport orientation.
    # camera_dir = (-1,-1,-1)/sqrt(3),  up = (0,0,1) (OVITO global up = Z)
    # ------------------------------------------------------------------
    if view in ("perspective", "orthographic"):
        # Tilted direction: equal components along -X, -Y, -Z
        d = np.array([-1.0, -1.0, -1.0]) / np.sqrt(3.0)
        up = np.array([0.0, 0.0, 1.0])

        # For the isometric view the "screen half-size" is the projection
        # of the bounding-box half-diagonal onto the plane perpendicular to d.
        # A conservative bound: use the full 3-D half-diagonal.
        screen_half = float(np.linalg.norm(half))
        cam_dist = screen_half * 3.0 + margin * 2.0  # generous pull-back

        if view == "perspective":
            fov = math.radians(fov_deg)
            dist = (screen_half + margin) / math.tan(fov * 0.5)
            dist = max(dist, cam_dist)  # never closer than cam_dist
            return CameraParams(
                is_perspective=True,
                field_of_view=fov,
                position=tuple(center - d * dist),
                direction=tuple(d),
                up=tuple(up),
            )
        else:
            fov_ortho = screen_half + margin
            return CameraParams(
                is_perspective=False,
                field_of_view=fov_ortho,
                position=tuple(center - d * cam_dist),
                direction=tuple(d),
                up=tuple(up),
            )

    # ------------------------------------------------------------------
    # Axis-aligned orthographic views
    #
    # OVITO conventions confirmed from GUI axis tripods and Python API docs:
    #
    #   "top"    camera_dir=(0,0,-1), up=(0,1,0) → XY plane, X→right, Y→up
    #   "bottom" camera_dir=(0,0,+1), up=(0,1,0)
    #   "left"   camera_dir=(-1,0,0), up=(0,1,0) → ZY plane, Z→right, Y→up
    #   "right"  camera_dir=(+1,0,0), up=(0,1,0)
    #   "front"  camera_dir=(0,-1,0), up=(0,0,1) → XZ plane, X→right, Z→up
    #   "back"   camera_dir=(0,+1,0), up=(0,0,1)
    #
    # ax_h : world axis index that maps to screen-right  (horizontal)
    # ax_v : world axis index that maps to screen-up     (vertical)
    # ------------------------------------------------------------------
    VIEW_DEFS = {
        #            direction         up_vec        ax_h  ax_v
        "top":    (( 0,  0, -1), ( 0,  1,  0),       0,    1),
        "bottom": (( 0,  0, +1), ( 0,  1,  0),       0,    1),
        "left":   ((-1,  0,  0), ( 0,  1,  0),       2,    1),
        "right":  ((+1,  0,  0), ( 0,  1,  0),       2,    1),
        "front":  (( 0, -1,  0), ( 0,  0,  1),       0,    2),
        "back":   (( 0, +1,  0), ( 0,  0,  1),       0,    2),
    }

    direction, up_vec, ax_h, ax_v = VIEW_DEFS[view]
    direction = np.array(direction, dtype=float)
    up_vec    = np.array(up_vec,    dtype=float)

    # fov_ortho = half-height of the viewport in world units.
    # Take the max of horizontal and vertical half-extents so the full
    # structure fits regardless of image aspect ratio (assumes square output;
    # for non-square you may need to adjust fov_ortho manually).
    fov_ortho = float(max(half[ax_v], half[ax_h])) + margin

    # Pull the camera back far enough so the entire depth of the structure
    # is in front of the camera plane.
    depth_axis = int(np.argmax(np.abs(direction)))
    depth_span = float(half[depth_axis])
    cam_dist = depth_span + float(np.linalg.norm(half)) + 1.0
    cam_pos = center - direction * cam_dist

    return CameraParams(
        is_perspective=False,
        field_of_view=fov_ortho,
        position=tuple(cam_pos),
        direction=tuple(direction),
        up=tuple(up_vec),
    )


def _default_colors(system, N: int) -> np.ndarray:
    """Jmol colour scheme; returns grey (0.7, 0.7, 0.7) for unknown elements."""
    JMOL = {
        "H":  (1.00, 1.00, 1.00),
        "He": (0.85, 1.00, 1.00),
        "Li": (0.80, 0.50, 1.00),
        "Be": (0.76, 1.00, 0.00),
        "B":  (1.00, 0.71, 0.71),
        "C":  (0.56, 0.56, 0.56),
        "N":  (0.19, 0.31, 0.97),
        "O":  (1.00, 0.05, 0.05),
        "F":  (0.56, 0.88, 0.31),
        "Na": (0.67, 0.36, 0.95),
        "Mg": (0.54, 1.00, 0.00),
        "Al": (0.75, 0.65, 0.65),
        "Si": (0.94, 0.78, 0.63),
        "P":  (1.00, 0.50, 0.00),
        "S":  (1.00, 1.00, 0.19),
        "Cl": (0.12, 0.94, 0.12),
        "K":  (0.56, 0.25, 0.83),
        "Ca": (0.24, 1.00, 0.00),
        "Ti": (0.75, 0.76, 0.78),
        "Cr": (0.54, 0.60, 0.78),
        "Mn": (0.61, 0.48, 0.78),
        "Fe": (0.88, 0.40, 0.20),
        "Co": (0.94, 0.56, 0.63),
        "Ni": (0.31, 0.82, 0.31),
        "Cu": (0.78, 0.50, 0.20),
        "Zn": (0.49, 0.50, 0.69),
        "Au": (1.00, 0.82, 0.14),
        "Ag": (0.75, 0.75, 0.75),
        "Pt": (0.82, 0.82, 0.88),
        "Pd": (0.00, 0.41, 0.52),
    }
    colors = np.ones((N, 4), dtype=np.float32)
    try:
        type_names = system.data["element"]
        for i, tn in enumerate(type_names):
            col = JMOL.get(tn.capitalize(), (0.7, 0.7, 0.7))
            colors[i, 0] = col[0]
            colors[i, 1] = col[1]
            colors[i, 2] = col[2]
            colors[i, 3] = 1.0
    except Exception:
        colors[:] = [0.7, 0.7, 0.7, 1.0]
    return colors


def _box_edges(system) -> Optional[np.ndarray]:
    """
    Generate the 12 edges of the simulation cell as line segments.

    Parameters
    ----------
    system : mdapy.System
        system.box.box    : (3,3) row vectors a, b, c
        system.box.origin : (3,)  cell origin

    Returns
    -------
    (12, 2, 3) float64  or  None if box information is unavailable.
    """
    try:
        box = np.asarray(system.box.box, dtype=np.float64)    # (3,3)
        origin = np.asarray(system.box.origin, dtype=np.float64)  # (3,)
    except AttributeError:
        return None
    a, b, c = box[0], box[1], box[2]
    o = origin
    v = np.array([
        o,           o + a,
        o + b,       o + a + b,
        o + c,       o + a + c,
        o + b + c,   o + a + b + c,
    ])
    idx = [
        (0, 1), (2, 3), (4, 5), (6, 7),  # along a
        (0, 2), (1, 3), (4, 6), (5, 7),  # along b
        (0, 4), (1, 5), (2, 6), (3, 7),  # along c
    ]
    edges = np.empty((12, 2, 3), dtype=np.float64)
    for k, (i, j) in enumerate(idx):
        edges[k, 0] = v[i]
        edges[k, 1] = v[j]
    return edges


if __name__ == "__main__":
    import math
    import numpy as np
    import mdapy as mp
    import matplotlib.pyplot as plt

    sys_al = mp.build_hea(
        ["Cr", "Co", "Ni"], [1/3, 1/3, 1/3],
        "fcc", 3.6, nx=5, ny=5, nz=5, random_seed=1
    )
    print(f"  N atoms: {sys_al.N}")

    ren = TachyonRender(
        width=600,
        height=600,
        backend='cpu',
        background=(1, 1, 1),
        direct_light_intensity=1.2,
        aa_samples=12,
        ao_brightness=1.0,
    )

    pos = sys_al.get_positions().to_numpy()
    r   = 1.3  # atom radius

    # Render four views matching OVITO's default viewport layout
    views  = ["perspective", "top", "front", "left"]
    titles = ["Perspective (-Z dir)", "Top (dir=-Z)", "Front (dir=-Y)", "Left (dir=-X)"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, view_name, title in zip(axes.flat, views, titles):
        # max_radius ensures edge atom spheres are fully visible (not just centres)
        cam = preset_camera(view_name, pos, max_radius=r)
        img = ren.render_system(
            sys_al,
            camera=cam,
            default_radius=r,
            box_color=(0, 0, 0, 1),
            box_edge_radius=0.04,
        )
        ax.imshow(img)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    plt.suptitle("mdapy TachyonRender — 4 views (OVITO style)", fontsize=15)
    plt.tight_layout()
    plt.show()

    # Single-view examples:
    # cam = preset_camera("top",         pos, max_radius=r)   # top-down
    # cam = preset_camera("front",       pos, max_radius=r)   # front face
    # cam = preset_camera("left",        pos, max_radius=r)   # left side
    # cam = preset_camera("perspective", pos, max_radius=r)   # perspective (default)
    # cam = preset_camera("perspective", pos, fov_deg=60, max_radius=r)

    # Cross-section example using znear:
    # cam = preset_camera("top", pos, max_radius=r)
    # cam.znear = pos[:, 2].max() / 2  # clip away the front half along Z