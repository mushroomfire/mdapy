# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
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

from mdapy.system import System
from mdapy.data import ele_radius, ele_rgb, type_rgb
import polars as pl

# GPU backend: only available when mdapy was built with OptiX.
_HAS_OPTIX: bool = _tachyon_mod.has_optix()
if _HAS_OPTIX:
    from mdapy._tachyon import TachyonOptiXRenderer as _TachyonOptiXRenderer


def is_gpu_available() -> bool:
    """Return True if this mdapy build supports GPU rendering (OptiX 7+)."""
    return _HAS_OPTIX


def load_image(path: str) -> np.ndarray:
    """
    Load an image file and return a ``(H, W, 4)`` uint8 RGBA numpy array.

    Supports PNG, JPEG, BMP, TGA, GIF, PSD, HDR, PIC, PNM (via stb_image).

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    numpy.ndarray, shape (H, W, 4), dtype uint8, RGBA.
    """
    return np.array(_tachyon_mod.load_image(path), copy=False)


def save_image(path: str, img: np.ndarray) -> None:
    """
    Save a ``(H, W, 4)`` uint8 RGBA image array to a file.

    Format is determined by the file extension:

    - ``.png``        — RGBA, lossless (alpha preserved)
    - ``.jpg/.jpeg``  — RGB, lossy (alpha discarded, quality 95)
    - ``.bmp``        — RGB
    - ``.tga``        — RGBA
    - other           — treated as ``.png``

    Parameters
    ----------
    path : str
        Destination file path.
    img  : numpy.ndarray, shape (H, W, 4), dtype uint8
        RGBA image to save.
    """
    img = np.ascontiguousarray(img, dtype=np.uint8)
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"img must be (H, W, 4) uint8, got shape {img.shape}")
    _tachyon_mod.save_image(path, img)


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

    Notes
    -----
    Image size (``width`` / ``height``) is specified per-call in
    :meth:`render` / :meth:`render_system`, so the same renderer instance
    can produce images at different resolutions without re-initialisation.

    Examples
    --------
    >>> ren_cpu = TachyonRender()                     # CPU, default settings
    >>> ren_gpu = TachyonRender(backend="gpu")         # GPU
    >>> ren_auto = TachyonRender(backend="auto")       # auto-select
    >>> # render at 1920×1080; save directly to PNG
    >>> ren_cpu.render_system(sys, width=1920, height=1080,
    ...                       output_figure="out.png")
    >>> # render with transparent background
    >>> ren_cpu.render_system(sys, width=800, height=600,
    ...                       output_figure="out.png", transparent=True)
    >>> print(ren_cpu.backend)
    cpu
    """

    def __init__(
        self,
        backend: str = "cpu",
        antialiasing: bool = True,
        aa_samples: int = 12,
        ao: bool = True,
        ao_samples: int = 12,
        ao_brightness: float = 0.8,
        ao_max_dist: float = 3.402823e38,  # RT_AO_MAXDIST_UNLIMITED
        shadows: bool = True,
        direct_light_intensity: float = 0.9,
        background: tuple = (0.0, 0.0, 0.0),
        num_threads: int = 0,
    ):
        backend = backend.lower().strip()
        if backend not in ("cpu", "gpu", "auto"):
            raise ValueError(
                f"backend must be 'cpu', 'gpu', or 'auto', got {backend!r}"
            )

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

        self._backend = backend  # "cpu" or "gpu" (resolved)

        rp = _RenderParams()
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
        return (
            f"TachyonRender(backend={self._backend!r}, "
            f"ao={rp.ao_enabled}, aa={rp.antialiasing_enabled})"
        )

    # Main rendering interface
    def render(
        self,
        positions: np.ndarray,
        colors: np.ndarray,
        radii: np.ndarray,
        camera: Optional[CameraParams] = None,
        bond_edges: Optional[np.ndarray] = None,
        bond_colors: Optional[np.ndarray] = None,
        bond_radius: float = 0.1,
        bond_color: tuple = (0.8, 0.8, 0.8, 1.0),
        box_edges: Optional[np.ndarray] = None,
        box_edge_radius: float = 0.05,
        box_color: tuple = (1.0, 1.0, 1.0, 1.0),
        width: int = 800,
        height: int = 600,
        output_figure: Optional[str] = None,
        transparent: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Render spherical particles and optional bond / box cylinders.

        Parameters
        ----------
        positions       : (N, 3) float64  Particle positions.
        colors          : (N, 4) float32  Per-particle RGBA colour, values in [0, 1].
        radii           : (N,)  float32   Per-particle radius.
        camera          : CameraParams or None.  If None, a perspective camera
                          is generated automatically from the bounding box.
        bond_edges      : (K, 2, 3) float64 or None. Pairs of endpoints for each
                          bond cylinder. Pass None to skip drawing bonds.
        bond_colors     : (K, 4) float32 or None. Per-bond RGBA colours.
                          If None, `bond_color` is used for all bond cylinders.
        bond_radius     : float. Cylinder radius for bonds. Default 0.1.
        bond_color      : tuple (R, G, B) or (R, G, B, A), values in [0, 1].
                          Default opaque grey.
        box_edges       : (M, 2, 3) float64 or None.  Pairs of endpoints for each
                          box edge cylinder.  Pass None to skip drawing.
        box_edge_radius : float.  Cylinder radius for box edges.  Default 0.05.
        box_color       : tuple (R, G, B) or (R, G, B, A), values in [0, 1].
                          Default opaque white (1, 1, 1, 1).
        width, height   : int.  Output image size in pixels.  Default 800 × 600.
        output_figure   : str or None.  If given, save the image to this path
                          (format inferred from extension: png/jpg/bmp/tga) and
                          return None.  If None, return the RGBA numpy array.
        transparent     : bool.  When True, pixels whose RGB matches the background
                          colour are made fully transparent (alpha=0).  Only
                          meaningful for PNG output or when inspecting the array.
                          Default False.

        Returns
        -------
        numpy.ndarray, shape (H, W, 4), dtype uint8, RGBA image — or None if
        ``output_figure`` is provided.
        """
        positions = np.ascontiguousarray(positions, dtype=np.float64)
        colors = np.ascontiguousarray(colors, dtype=np.float32)
        radii = np.ascontiguousarray(radii, dtype=np.float32)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be (N,3), got {positions.shape}")
        if colors.ndim != 2 or colors.shape[1] != 4:
            raise ValueError(f"colors must be (N,4), got {colors.shape}")
        if radii.ndim != 1:
            raise ValueError(f"radii must be (N,), got {radii.shape}")

        # Update resolution per-call so the same renderer can be reused at
        # different sizes without reconstructing the backend object.
        self._rp.width = int(width)
        self._rp.height = int(height)

        if camera is None:
            max_r = float(radii.max()) if len(radii) > 0 else 0.0
            camera = _auto_camera(positions, max_radius=max_r)
        cpp_cam = camera._to_cpp()

        if bond_edges is not None:
            bond_edges = np.ascontiguousarray(bond_edges, dtype=np.float64)
            if bond_edges.ndim != 3 or bond_edges.shape[1:] != (2, 3):
                raise ValueError(f"bond_edges must be (K,2,3), got {bond_edges.shape}")
            if bond_edges.shape[0] == 0:
                bond_edges = None
            elif bond_colors is None:
                bc = tuple(float(v) for v in bond_color)
                bond_colors = np.tile(
                    np.array(
                        [bc[0], bc[1], bc[2], bc[3] if len(bc) > 3 else 1.0],
                        dtype=np.float32,
                    ),
                    (bond_edges.shape[0], 1),
                )
            else:
                bond_colors = np.ascontiguousarray(bond_colors, dtype=np.float32)
                if bond_colors.ndim != 2 or bond_colors.shape[1] != 4:
                    raise ValueError(
                        f"bond_colors must be (K,4), got {bond_colors.shape}"
                    )
                if bond_colors.shape[0] != bond_edges.shape[0]:
                    raise ValueError(
                        "bond_edges and bond_colors must contain the same number of cylinders"
                    )

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

        raw = self._renderer.render(
            self._rp,
            cpp_cam,
            positions,
            colors,
            radii,
            bond_edges,
            bond_colors,
            float(bond_radius),
            box_edges,
            float(box_edge_radius),
            box_r,
            box_g,
            box_b,
            box_a,
        )

        # transparent: detect background pixels by comparing to bg RGB and zero
        # out their alpha channel.  Pixels that differ from the background by
        # more than 1 LSB in any channel are treated as geometry (alpha=255).
        if transparent:
            img = np.array(raw)  # writable copy
            bg = np.array(
                [self._rp.bg_r, self._rp.bg_g, self._rp.bg_b], dtype=np.float32
            ) * 255.0
            diff = np.abs(img[:, :, :3].astype(np.float32) - bg).max(axis=2)
            img[:, :, 3] = np.where(diff < 1.5, 0, 255).astype(np.uint8)
        else:
            img = np.array(raw, copy=False)

        if output_figure is not None:
            _tachyon_mod.save_image(output_figure, img)
            return None

        return img

    # Convenience wrapper for mdapy.System
    def render_system(
        self,
        system: System,
        colors: Optional[np.ndarray] = None,
        radii: Optional[np.ndarray] = None,
        camera: Optional[CameraParams] = None,
        draw_bond: bool = False,
        bond: Optional[np.ndarray] = None,
        bond_radius: float = 0.1,
        bond_color: tuple = (0.8, 0.8, 0.8, 1.0),
        bond_color_mode: str = "uniform",
        draw_box: bool = True,
        box_edge_radius: float = 0.05,
        box_color: tuple = (1.0, 1.0, 1.0, 1.0),
        default_radius: float = 1.0,
        width: int = 800,
        height: int = 600,
        output_figure: Optional[str] = None,
        transparent: bool = False,
    ) -> Optional[np.ndarray]:
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
        draw_bond       : bool. Whether to draw bond cylinders. Default False.
        bond            : (Nbond, 2) int array or None. If None and
                          ``draw_bond=True``, ``system.bond`` is used.
        bond_radius     : float. Cylinder radius for bonds. Default 0.1.
        bond_color      : tuple (R, G, B) or (R, G, B, A), values in [0, 1].
                          Used when ``bond_color_mode='uniform'``.
        bond_color_mode : str. ``'uniform'`` or ``'atom'``. In ``'atom'`` mode
                          each bond is split into two half-cylinders coloured
                          by the connected atoms.
        draw_box        : bool.  Whether to draw simulation-cell edges.  Default True.
        box_edge_radius : float.  Cylinder radius for cell edges.  Default 0.05.
        box_color       : tuple (R, G, B) or (R, G, B, A), values in [0, 1].
                          Default opaque white (1, 1, 1, 1).
        default_radius  : float.  Fallback radius when ``radii`` is None.  Default 1.0.
        width, height   : int.  Output image size in pixels.  Default 800 × 600.
        output_figure   : str or None.  If given, save image to this path and
                          return None.  Format inferred from extension.
        transparent     : bool.  Transparent background (PNG recommended).
        """
        pos = system.get_positions().to_numpy()  # (N,3)

        if colors is None:
            colors = _default_colors(system)
        colors = np.ascontiguousarray(colors, dtype=np.float32)

        if radii is not None:
            radii = np.ascontiguousarray(radii, dtype=np.float32)
        else:
            if "element" in system.data.columns:
                radii = np.array(
                    system.data.with_columns(
                        pl.col("element")
                        .replace_strict(
                            ele_radius,
                            default=default_radius,
                        )
                        .alias("radius")
                    )["radius"]
                    / 2,
                    np.float32,
                )
            else:
                radii = np.full(system.N, default_radius, dtype=np.float32)

        box_edges = _box_edges(system) if draw_box else None
        bond_edges = None
        bond_colors = None
        if draw_bond:
            if bond is None:
                if not hasattr(system, "bond"):
                    raise ValueError(
                        "draw_bond=True requires a bond array or system.cal_build_bond() first."
                    )
                bond = system.bond
            bond_edges, bond_colors = _bond_edges(
                system,
                bond,
                colors,
                radii,
                bond_radius,
                bond_color_mode,
            )

        return self.render(
            pos,
            colors,
            radii,
            camera=camera,
            bond_edges=bond_edges,
            bond_colors=bond_colors if bond_color_mode == "atom" else None,
            bond_radius=bond_radius,
            bond_color=bond_color,
            box_edges=box_edges,
            box_edge_radius=box_edge_radius,
            box_color=box_color,
            width=width,
            height=height,
            output_figure=output_figure,
            transparent=transparent,
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
PRESET_VIEWS = (
    "perspective",
    "orthographic",
    "top",
    "bottom",
    "front",
    "back",
    "left",
    "right",
)


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
    OVITO uses a right-handed coordinate system where **Z is the global up axis**.
    The eight standard viewports are:

    +-----------------+-------------------------------+-------------------+---------+
    | Name            | Meaning                       | camera_dir (OVITO)| up      |
    +=================+===============================+===================+=========+
    | ``"top"``       | Look down the -Z axis         | (0, 0, -1)        | (0,1,0) |
    |                 | → sees XY plane               |                   |         |
    +-----------------+-------------------------------+-------------------+---------+
    | ``"front"``     | Look along +Y axis            | (0, 1, 0)         | (0,0,1) |
    |                 | → sees XZ plane               |                   |         |
    +-----------------+-------------------------------+-------------------+---------+
    | ``"left"``      | Look along +X axis            | (1, 0, 0)         | (0,0,1) |
    |                 | → sees YZ plane               |                   |         |
    +-----------------+-------------------------------+-------------------+---------+
    | ``"perspective"``| Tilted view from +X+Y+Z      | (-1,-1,-1)/√3     | (0,0,1) |
    | / ``"ortho"``   | → 3D isometric look           |                   |         |
    +-----------------+-------------------------------+-------------------+---------+

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
    - ``"front"``        – Orthographic, camera_dir=(0,+1,0), up=(0,0,1)
                           looks at the XZ plane, X points right, Z points up
                           (matches OVITO *Front* viewport)
    - ``"back"``         – Orthographic, camera_dir=(0,-1,0), up=(0,0,1)
    - ``"left"``         – Orthographic, camera_dir=(+1,0,0), up=(0,0,1)
                           looks at the YZ plane, Y points right, Z points up
                           (matches OVITO *Left* viewport)
    - ``"right"``        – Orthographic, camera_dir=(-1,0,0), up=(0,0,1)

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
        cam.znear = 9.0  # discard atoms in front of the z=9 Å plane
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
    # OVITO conventions (verified against OVITO GUI axis tripods):
    #
    #   "top"    camera_dir=(0,0,-1), up=(0,1,0) → XY plane, X→right, Y→up
    #   "bottom" camera_dir=(0,0,+1), up=(0,1,0)
    #   "front"  camera_dir=(0,+1,0), up=(0,0,1) → XZ plane, X→right, Z→up
    #   "back"   camera_dir=(0,-1,0), up=(0,0,1)
    #   "left"   camera_dir=(+1,0,0), up=(0,0,1) → YZ plane, Y→right, Z→up
    #   "right"  camera_dir=(-1,0,0), up=(0,0,1)
    #
    # ax_h : world axis index that maps to screen-right  (horizontal)
    # ax_v : world axis index that maps to screen-up     (vertical)
    # ------------------------------------------------------------------
    VIEW_DEFS = {
        #            direction         up_vec        ax_h  ax_v
        "top": ((0, 0, -1), (0, 1, 0), 0, 1),
        "bottom": ((0, 0, +1), (0, 1, 0), 0, 1),
        "front": ((0, +1, 0), (0, 0, 1), 0, 2),
        "back": ((0, -1, 0), (0, 0, 1), 0, 2),
        "left": ((+1, 0, 0), (0, 0, 1), 1, 2),
        "right": ((-1, 0, 0), (0, 0, 1), 1, 2),
    }

    direction, up_vec, ax_h, ax_v = VIEW_DEFS[view]
    direction = np.array(direction, dtype=float)
    up_vec = np.array(up_vec, dtype=float)

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


def _default_colors(system: System) -> np.ndarray:
    """Jmol colour scheme; returns grey (0.7, 0.7, 0.7) for unknown elements."""

    if "element" in system.data.columns:
        colors = (
            system.data.with_columns(
                pl.col("element")
                .replace_strict(
                    ele_rgb,
                    default=[255 * 0.7] * 3,
                    return_dtype=pl.Array(pl.Float32, 3),
                )
                .alias("color")
            )["color"].to_numpy()
            / 255
        )
        colors = np.c_[colors, np.ones(system.N)].astype(np.float32)
    elif "type" in system.data.columns:
        colors = (
            system.data.with_columns(
                (pl.col("type") % 9)
                .replace_strict(
                    type_rgb,
                    default=[255 * 0.7] * 3,
                    return_dtype=pl.Array(pl.Float32, 3),
                )
                .alias("color")
            )["color"].to_numpy()
            / 255
        )
        colors = np.c_[colors, np.ones(system.N)].astype(np.float32)
    else:
        colors = np.full((system.N, 4), [0.7, 0.7, 0.7, 1.0], np.float32)

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
        box = np.asarray(system.box.box, dtype=np.float64)  # (3,3)
        origin = np.asarray(system.box.origin, dtype=np.float64)  # (3,)
    except AttributeError:
        return None
    a, b, c = box[0], box[1], box[2]
    o = origin
    v = np.array(
        [
            o,
            o + a,
            o + b,
            o + a + b,
            o + c,
            o + a + c,
            o + b + c,
            o + a + b + c,
        ]
    )
    idx = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),  # along a
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),  # along b
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # along c
    ]
    edges = np.empty((12, 2, 3), dtype=np.float64)
    for k, (i, j) in enumerate(idx):
        edges[k, 0] = v[i]
        edges[k, 1] = v[j]
    return edges


def _bond_edges(
    system: System,
    bond: np.ndarray,
    atom_colors: np.ndarray,
    atom_radii: Optional[np.ndarray] = None,
    bond_radius: float = 0.1,
    color_mode: str = "uniform",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Generate bond cylinder segments from a bond index array."""
    color_mode = color_mode.lower().strip()
    if color_mode not in {"uniform", "atom"}:
        raise ValueError(
            f"bond_color_mode must be 'uniform' or 'atom', got {color_mode!r}"
        )

    bond = np.ascontiguousarray(bond, dtype=np.int32)
    if bond.ndim != 2 or bond.shape[1] != 2:
        raise ValueError(f"bond must be (Nbond,2), got {bond.shape}")
    if bond.shape[0] == 0:
        return np.empty((0, 2, 3), dtype=np.float64), None

    pos = system.get_positions().to_numpy()
    origin = np.asarray(system.box.origin, dtype=np.float64)
    box = np.asarray(system.box.box, dtype=np.float64)
    inv_box = np.asarray(system.box.inverse_box, dtype=np.float64)
    boundary = np.asarray(system.box.boundary, dtype=np.int32)
    if atom_radii is None:
        atom_radii = np.zeros(system.N, dtype=np.float64)
    else:
        atom_radii = np.ascontiguousarray(atom_radii, dtype=np.float64)
        if atom_radii.ndim != 1 or atom_radii.shape[0] != system.N:
            raise ValueError(
                f"atom_radii must be (N,), got {atom_radii.shape}"
            )

    def _split_fractional_segment(s0: np.ndarray, ds: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        pieces = []
        current = s0.copy()
        remaining = ds.copy()

        while np.linalg.norm(remaining) > 1e-12:
            target = current + remaining
            t_hit = 1.0
            hit_dims = []
            for dim in range(3):
                if boundary[dim] != 1 or abs(remaining[dim]) < 1e-12:
                    continue
                if target[dim] < 0.0:
                    t = (0.0 - current[dim]) / remaining[dim]
                elif target[dim] >= 1.0:
                    t = (1.0 - current[dim]) / remaining[dim]
                else:
                    continue
                if t < 1e-12 or t > 1.0 + 1e-12:
                    continue
                if t < t_hit - 1e-12:
                    t_hit = t
                    hit_dims = [dim]
                elif abs(t - t_hit) < 1e-12:
                    hit_dims.append(dim)

            if not hit_dims:
                pieces.append((current.copy(), target.copy()))
                break

            hit_point = current + t_hit * remaining
            inside_point = hit_point.copy()
            for dim in hit_dims:
                inside_point[dim] = 0.0 if remaining[dim] < 0.0 else 1.0
            pieces.append((current.copy(), inside_point))

            remaining = (1.0 - t_hit) * remaining
            current = hit_point.copy()
            for dim in hit_dims:
                if remaining[dim] < 0.0:
                    current[dim] += 1.0
                else:
                    current[dim] -= 1.0

        return pieces

    edge_list = []
    color_list = []

    def _segment_crosses_boundary(start: np.ndarray, disp: np.ndarray) -> bool:
        s0 = (start - origin) @ inv_box
        s0 = s0 - np.floor(s0)
        target = s0 + disp @ inv_box
        for dim in range(3):
            if boundary[dim] != 1:
                continue
            if target[dim] < -1e-12 or target[dim] >= 1.0 + 1e-12:
                return True
        return False

    def _append_segment(
        start: np.ndarray,
        disp: np.ndarray,
        color: Optional[np.ndarray] = None,
    ) -> None:
        if np.linalg.norm(disp) < 1e-12:
            return
        s0 = (start - origin) @ inv_box
        ds = disp @ inv_box
        s0 = s0 - np.floor(s0)
        pieces = _split_fractional_segment(s0, ds)
        for s_a, s_b in pieces:
            a = origin + s_a @ box
            b = origin + s_b @ box
            if np.linalg.norm(b - a) < 1e-12:
                continue
            edge_list.append(np.stack((a, b), axis=0))
            if color is not None:
                color_list.append(color)

    for i, j in bond:
        p0 = pos[i].astype(np.float64)
        rij = system.box.pbc(pos[j] - pos[i]).astype(np.float64)
        total_len = float(np.linalg.norm(rij))
        if total_len < 1e-12:
            continue
        unit = rij / total_len
        ri = max(0.0, float(atom_radii[i]))
        rj = max(0.0, float(atom_radii[j]))

        # For visualization we let the cylinder embed slightly into each atom
        # sphere so the seam is hidden by the atom, which looks more natural
        # than trimming exactly at the sphere boundary.
        trim_i = max(0.0, ri - 1.15 * bond_radius)
        trim_j = max(0.0, rj - 1.15 * bond_radius)

        # Trim the cylinder so it starts at the atom surface instead of
        # penetrating the sphere interior.
        visible_len = total_len - trim_i - trim_j
        if visible_len <= 1e-12:
            continue

        crosses_boundary = _segment_crosses_boundary(p0, rij)
        if crosses_boundary:
            half_len = total_len * 0.5
            seg0_len = half_len - trim_i
            seg1_len = half_len - trim_j
            if seg0_len > 1e-12:
                _append_segment(
                    p0 + unit * trim_i,
                    unit * seg0_len,
                    atom_colors[i] if color_mode == "atom" else None,
                )
            if seg1_len > 1e-12:
                _append_segment(
                    pos[j].astype(np.float64) - unit * trim_j,
                    -unit * seg1_len,
                    atom_colors[j] if color_mode == "atom" else None,
                )
        elif color_mode == "atom":
            half_visible = visible_len * 0.5
            _append_segment(
                p0 + unit * trim_i,
                unit * half_visible,
                atom_colors[i],
            )
            _append_segment(
                pos[j].astype(np.float64) - unit * trim_j,
                -unit * half_visible,
                atom_colors[j],
            )
        else:
            _append_segment(p0 + unit * trim_i, unit * visible_len, None)

    if len(edge_list) == 0:
        return np.empty((0, 2, 3), dtype=np.float64), None

    edges = np.asarray(edge_list, dtype=np.float64)
    if color_mode == "uniform":
        return edges, None
    colors = np.asarray(color_list, dtype=np.float32)
    return edges, colors


if __name__ == "__main__":
    from pathlib import Path
    import mdapy as mp
    import matplotlib.pyplot as plt
    import polars as pl

    demo = mp.System(
        data=pl.DataFrame(
            {
                "x": [0.000, 0.9572, -0.2390, 3.000, 3.9572, 2.7610],
                "y": [0.000, 0.0000, 0.9270, 0.000, 0.0000, 0.9270],
                "z": [0.000, 0.0000, 0.0000, 0.000, 0.0000, 0.0000],
                "element": ["O", "H", "H", "O", "H", "H"],
            }
        ),
        box=[8.0, 6.0, 6.0],
    )
    demo.cal_build_bond({("O", "H"): 1.2, ("H", "H"): 0.8, ("O", "O"): 1.5})

    ren = TachyonRender(
        backend="cpu",
        background=(1, 1, 1),
        antialiasing=True,
        aa_samples=16,
        ao=True,
        ao_samples=16,
        ao_brightness=0.9,
        direct_light_intensity=1.1,
    )

    pos = demo.get_positions().to_numpy()
    radii = np.array([0.55 if e == "O" else 0.32 for e in demo.data["element"]], np.float32)
    cam = preset_camera("perspective", pos, max_radius=float(radii.max()), margin=1.5)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    for ax, mode, title in zip(
        axes,
        ["uniform", "atom"],
        ["Uniform Bond Color", "Bond Color From Atoms"],
    ):
        img = ren.render_system(
            demo,
            camera=cam,
            radii=radii,
            draw_bond=True,
            bond_radius=0.12,
            bond_color=(0.20, 0.20, 0.20, 1.0),
            bond_color_mode=mode,
            draw_box=False,
            width=900,
            height=700,
        )
        ax.imshow(img)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    out = Path(__file__).resolve().parents[2] / "build" / "render_bond_demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"Bond render demo saved to: {out}")
    if "agg" not in plt.get_backend().lower():
        plt.show()

    # Previous atom-only stress test example has been intentionally commented
    # out to keep `python -m mdapy.render` focused on the new bond rendering.
