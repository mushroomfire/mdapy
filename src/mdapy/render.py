"""
mdapy/render.py  —  Tachyon 渲染器 Python 封装
================================================

快速开始
--------
::

    import mdapy as mp
    from mdapy.render import TachyonRender

    sys = mp.System("model.xyz")
    ren = TachyonRender(width=800, height=600)
    img = ren.render_system(sys)          # numpy (H,W,4) uint8 RGBA

    from PIL import Image
    Image.fromarray(img).save("out.png")

相机参数（与 OVITO 完全对应）
------------------------------
::

    from mdapy.render import CameraParams
    import math

    # 透视（默认）
    cam = CameraParams(
        is_perspective = True,
        field_of_view  = math.radians(40),   # 垂直视角，单位弧度
        position       = (0, 0, 60),
        direction      = (0, 0, -1),
        up             = (0, 1,  0),
    )

    # 正交
    cam_ortho = CameraParams(
        is_perspective = False,
        field_of_view  = 25.0,               # 视口半高，世界坐标
        position       = (0, 0, 100),
        direction      = (0, 0, -1),
        up             = (0, 1,  0),
    )
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import numpy as np

from mdapy._tachyon import (
    TachyonRenderer as _TachyonRenderer,
    RenderParams    as _RenderParams,
    CameraParams    as _CameraParams,
    Vec3            as _Vec3,
)


# ─── CameraParams ─────────────────────────────────────────────────────────────
class CameraParams:
    """
    相机参数，与 OVITO ``ViewProjectionParameters`` 完全对应。

    透视模式
    ~~~~~~~~
    ``field_of_view`` : 垂直视角（**弧度**）
    zoom = 0.5 / tan(fov/2)   ← 与 OVITO 公式相同

    正交模式
    ~~~~~~~~
    ``field_of_view`` : 视口半高（世界坐标）
    zoom = 0.5 / field_of_view
    """

    def __init__(self,
                 is_perspective: bool  = True,
                 field_of_view : float = math.radians(40),
                 position      : Tuple[float,float,float] = (0., 0., 50.),
                 direction     : Tuple[float,float,float] = (0., 0., -1.),
                 up            : Tuple[float,float,float] = (0., 1.,  0.),
                 znear         : float = 0.0,
                 dof_enabled   : bool  = False,
                 dof_focal_len : float = 40.0,
                 dof_aperture  : float = 0.01):
        self.is_perspective = bool(is_perspective)
        self.field_of_view  = float(field_of_view)
        self.position  = tuple(float(v) for v in position)
        self.direction = tuple(float(v) for v in direction)
        self.up        = tuple(float(v) for v in up)
        self.znear     = float(znear)
        self.dof_enabled   = bool(dof_enabled)
        self.dof_focal_len = float(dof_focal_len)
        self.dof_aperture  = float(dof_aperture)

    def _to_cpp(self) -> _CameraParams:
        cp = _CameraParams()
        cp.is_perspective = self.is_perspective
        cp.field_of_view  = self.field_of_view
        cp.position  = _Vec3(*self.position)
        cp.direction = _Vec3(*self.direction)
        cp.up        = _Vec3(*self.up)
        cp.znear     = self.znear
        cp.dof_enabled   = self.dof_enabled
        cp.dof_focal_len = self.dof_focal_len
        cp.dof_aperture  = self.dof_aperture
        return cp

    def __repr__(self):
        mode = "perspective" if self.is_perspective else "orthographic"
        fov  = math.degrees(self.field_of_view) if self.is_perspective \
               else self.field_of_view
        unit = "deg" if self.is_perspective else "world"
        return (f"CameraParams({mode}, fov={fov:.1f}{unit}, "
                f"pos={self.position})")


# ─── TachyonRender ────────────────────────────────────────────────────────────
class TachyonRender:
    """
    mdapy Tachyon 渲染器。

    Parameters
    ----------
    width, height : int
        输出分辨率（像素）。
    antialiasing : bool
        抗锯齿，默认 True。
    aa_samples : int
        抗锯齿采样数，默认 12。
    ao : bool
        Ambient Occlusion，默认 True。
    ao_samples : int
        AO 采样数，默认 12。
    ao_brightness : float
        AO 天光亮度，默认 0.8。
    shadows : bool
        阴影，默认 True。
    direct_light_intensity : float
        直接光强度，默认 0.9。
    background : tuple
        背景色 (R,G,B) 或 (R,G,B,A)，值域 [0,1]，默认黑色。
    num_threads : int
        渲染线程数，0 = Tachyon 自动，默认 0。
    """

    def __init__(self,
                 width                 : int   = 800,
                 height                : int   = 600,
                 antialiasing          : bool  = True,
                 aa_samples            : int   = 12,
                 ao                   : bool  = True,
                 ao_samples            : int   = 12,
                 ao_brightness         : float = 0.8,
                 shadows               : bool  = True,
                 direct_light_intensity: float = 0.9,
                 background            : tuple = (0., 0., 0.),
                 num_threads           : int   = 0):

        self._renderer = _TachyonRenderer()

        rp = _RenderParams()
        rp.width                  = int(width)
        rp.height                 = int(height)
        rp.antialiasing_enabled   = bool(antialiasing)
        rp.antialiasing_samples   = int(aa_samples)
        rp.ao_enabled             = bool(ao)
        rp.ao_samples             = int(ao_samples)
        rp.ao_brightness          = float(ao_brightness)
        rp.shadows_enabled        = bool(shadows)
        rp.direct_light_intensity = float(direct_light_intensity)
        bg = tuple(background)
        rp.bg_r = float(bg[0])
        rp.bg_g = float(bg[1])
        rp.bg_b = float(bg[2])
        rp.bg_a = float(bg[3]) if len(bg) > 3 else 1.0
        rp.num_threads = int(num_threads)
        self._rp = rp

    # ── 主渲染接口 ─────────────────────────────────────────────────────────
    def render(self,
               positions       : np.ndarray,
               colors          : np.ndarray,
               radii           : np.ndarray,
               camera          : Optional[CameraParams] = None,
               box_edges       : Optional[np.ndarray]   = None,
               box_edge_radius : float = 0.05,
               box_color       : tuple = (1., 1., 1., 1.),
               ) -> np.ndarray:
        """
        渲染球形粒子（及可选的晶胞框）。

        Parameters
        ----------
        positions  : (N,3) float64  粒子坐标
        colors     : (N,4) float32  RGBA ∈ [0,1]
        radii      : (N,)  float32  粒子半径
        camera     : CameraParams；None 时根据包围盒自动估算透视相机
        box_edges  : (M,2,3) float64  晶胞棱线端点；None 则不画
        box_edge_radius : 棱线圆柱半径
        box_color  : 棱线颜色 (R,G,B) 或 (R,G,B,A)

        Returns
        -------
        numpy.ndarray  shape (H,W,4)  dtype uint8  RGBA
        """
        # 确保正确的内存布局和类型
        positions = np.ascontiguousarray(positions, dtype=np.float64)
        colors    = np.ascontiguousarray(colors,    dtype=np.float32)
        radii     = np.ascontiguousarray(radii,     dtype=np.float32)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions 必须是 (N,3)，实际为 {positions.shape}")
        if colors.ndim != 2 or colors.shape[1] != 4:
            raise ValueError(f"colors 必须是 (N,4)，实际为 {colors.shape}")
        if radii.ndim != 1:
            raise ValueError(f"radii 必须是 (N,)，实际为 {radii.shape}")

        if camera is None:
            camera = _auto_camera(positions)
        cpp_cam = camera._to_cpp()

        # 处理 box_edges
        if box_edges is not None:
            box_edges = np.ascontiguousarray(box_edges, dtype=np.float64)
            if box_edges.ndim != 3 or box_edges.shape[1:] != (2, 3):
                raise ValueError(f"box_edges 必须是 (M,2,3)，实际为 {box_edges.shape}")
            if box_edges.shape[0] == 0:
                box_edges = None

        # 把 box_color 和 radius 写入 RenderParams 之外单独传递
        # 用临时 RenderParams 覆盖不合适，改为在 C++ 侧 BoxEdgeData 里设置
        # 这里的做法：把颜色信息编码进 box_edges 之前先调整
        # 实际上 C++ 侧 BoxEdgeData 的颜色固定为白色(1,1,1,1), radius=0.05
        # 如果需要自定义，在这里预处理或扩展 C++ 接口
        # 当前版本：通过更新 _rp 不影响其他参数，直接传 box_edges 即可
        # 如需自定义 box 颜色/半径，可在此处先对数据做处理

        # 调用 C++ 渲染
        img = self._renderer.render(
            self._rp, cpp_cam,
            positions, colors, radii,
            box_edges,
        )
        return np.array(img, copy=False)

    # ── 从 mdapy.System 渲染 ───────────────────────────────────────────────
    def render_system(self,
                      system,
                      colors         : Optional[np.ndarray] = None,
                      radii          : Optional[np.ndarray] = None,
                      camera         : Optional[CameraParams] = None,
                      draw_box       : bool  = True,
                      box_edge_radius: float = 0.05,
                      box_color      : tuple = (1., 1., 1., 1.),
                      default_radius : float = 1.0,
                      ) -> np.ndarray:
        """
        从 ``mdapy.System`` 一键渲染。

        Parameters
        ----------
        system         : mp.System
        colors         : (N,4) float32；None 时按元素 Jmol 配色
        radii          : (N,)  float32；None 时使用 default_radius
        camera         : None 时自动估算
        draw_box       : 是否画晶胞框
        box_edge_radius: 晶胞棱线半径
        box_color      : 晶胞棱线颜色
        default_radius : 默认粒子半径
        """
        pos = system.get_positions().to_numpy()   # (N,3)
        N   = len(pos)

        if colors is None:
            colors = _default_colors(system, N)
        colors = np.ascontiguousarray(colors, dtype=np.float32)

        if radii is None:
            radii = np.full(N, default_radius, dtype=np.float32)
        radii = np.ascontiguousarray(radii, dtype=np.float32)

        box_edges = _box_edges(system) if draw_box else None

        return self.render(pos, colors, radii,
                           camera=camera,
                           box_edges=box_edges,
                           box_edge_radius=box_edge_radius,
                           box_color=box_color)


# ─── 辅助函数（模块级）────────────────────────────────────────────────────────

def _auto_camera(positions: np.ndarray) -> CameraParams:
    """根据粒子包围盒自动估算透视相机。"""
    pmin   = positions.min(axis=0)
    pmax   = positions.max(axis=0)
    center = (pmin + pmax) * 0.5
    extent = float(np.linalg.norm(pmax - pmin))
    fov    = math.radians(40.0)
    dist   = (extent * 0.5) / math.tan(fov * 0.5) * 1.3 + 1.0
    return CameraParams(
        is_perspective = True,
        field_of_view  = fov,
        position       = tuple(center + np.array([0., 0., dist])),
        direction      = (0., 0., -1.),
        up             = (0., 1.,  0.),
    )


def _default_colors(system, N: int) -> np.ndarray:
    """Jmol 配色方案，未知元素用灰色。"""
    JMOL = {
        'H' :(1.00,1.00,1.00), 'He':(0.85,1.00,1.00),
        'Li':(0.80,0.50,1.00), 'Be':(0.76,1.00,0.00),
        'B' :(1.00,0.71,0.71), 'C' :(0.56,0.56,0.56),
        'N' :(0.19,0.31,0.97), 'O' :(1.00,0.05,0.05),
        'F' :(0.56,0.88,0.31), 'Na':(0.67,0.36,0.95),
        'Mg':(0.54,1.00,0.00), 'Al':(0.75,0.65,0.65),
        'Si':(0.94,0.78,0.63), 'P' :(1.00,0.50,0.00),
        'S' :(1.00,1.00,0.19), 'Cl':(0.12,0.94,0.12),
        'K' :(0.56,0.25,0.83), 'Ca':(0.24,1.00,0.00),
        'Ti':(0.75,0.76,0.78), 'Cr':(0.54,0.60,0.78),
        'Mn':(0.61,0.48,0.78), 'Fe':(0.88,0.40,0.20),
        'Co':(0.94,0.56,0.63), 'Ni':(0.31,0.82,0.31),
        'Cu':(0.78,0.50,0.20), 'Zn':(0.49,0.50,0.69),
        'Au':(1.00,0.82,0.14), 'Ag':(0.75,0.75,0.75),
        'Pt':(0.82,0.82,0.88), 'Pd':(0.00,0.41,0.52),
    }
    colors = np.ones((N, 4), dtype=np.float32)
    try:
        
        type_names = system.data['element']
        for i, tn in enumerate(type_names):
            col  = JMOL.get(tn.capitalize(), (0.7, 0.7, 0.7))
            colors[i, 0] = col[0]
            colors[i, 1] = col[1]
            colors[i, 2] = col[2]
            colors[i, 3] = 1.0
    except Exception:
        colors[:] = [0.7, 0.7, 0.7, 1.0]
    return colors


def _box_edges(system) -> Optional[np.ndarray]:
    """
    从 system.box 生成 12 条晶胞棱线，返回 (12,2,3) float64。

    system.box.box    : (3,3)  每行是 a/b/c 向量
    system.box.origin : (3,)   原点坐标
    """
    try:
        box    = np.asarray(system.box.box,    dtype=np.float64)  # (3,3)
        origin = np.asarray(system.box.origin, dtype=np.float64)  # (3,)
    except AttributeError:
        return None
    a, b, c = box[0], box[1], box[2]
    o = origin
    v = np.array([o, o+a, o+b, o+a+b, o+c, o+a+c, o+b+c, o+a+b+c])
    idx = [(0,1),(2,3),(4,5),(6,7),  # 沿 a
           (0,2),(1,3),(4,6),(5,7),  # 沿 b
           (0,4),(1,5),(2,6),(3,7)]  # 沿 c
    edges = np.empty((12, 2, 3), dtype=np.float64)
    for k, (i, j) in enumerate(idx):
        edges[k, 0] = v[i]
        edges[k, 1] = v[j]
    return edges


# ─── 独立示例（python render.py 直接运行）────────────────────────────────────
if __name__ == '__main__':
    import math
    import numpy as np
    try:
        import mdapy as mp
        _HAS_MDAPY = True
    except ImportError:
        _HAS_MDAPY = False

    # # ── 示例 1：纯 numpy 数组，无需 mdapy ──────────────────────────────────
    # print("=" * 50)
    # print("示例 1：纯 numpy 粒子 + 手动晶胞框")
    # print("=" * 50)

    # np.random.seed(0)
    # N   = 100
    # pos = np.random.uniform(-8, 8, (N, 3)).astype(np.float64)
    # col = np.ones((N, 4), dtype=np.float32)
    # col[:, 0] = 0.2; col[:, 1] = 0.5; col[:, 2] = 0.9
    # rad = np.full(N, 0.8, dtype=np.float32)

    # # 手动构建 12 条晶胞棱线
    # o  = np.array([-9., -9., -9.])
    # av = np.array([18.,  0.,  0.])
    # bv = np.array([ 0., 18.,  0.])
    # cv = np.array([ 0.,  0., 18.])
    # verts = [o, o+av, o+bv, o+av+bv, o+cv, o+av+cv, o+bv+cv, o+av+bv+cv]
    # eidx  = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
    # box_edges = np.zeros((12, 2, 3), dtype=np.float64)
    # for k, (i, j) in enumerate(eidx):
    #     box_edges[k, 0] = verts[i]
    #     box_edges[k, 1] = verts[j]

    # # 透视相机
    # cam = CameraParams(
    #     is_perspective = True,
    #     field_of_view  = math.radians(40),
    #     position       = (12., 8., 45.),
    #     direction      = (-0.25, -0.15, -1.),
    #     up             = (0., 1., 0.),
    # )
    # # 归一化 direction
    # d = np.array(cam.direction)
    # cam.direction = tuple(d / np.linalg.norm(d))

    # ren = TachyonRender(width=800, height=600, background=(0.05, 0.05, 0.12))
    # img = ren.render(pos, col, rad, camera=cam, box_edges=box_edges)
    # print(f"  图像 shape={img.shape}  非零像素={( img[:,:,:3]>0).any(2).sum()}")

    # try:
    #     from PIL import Image
    #     Image.fromarray(img).save("example1_numpy.png")
    #     print("  已保存 example1_numpy.png")
    # except ImportError:
    #     import struct, zlib
    #     # 手写最小 PNG（RGB）
    #     def save_png_rgb(path, arr):
    #         H, W = arr.shape[:2]
    #         raw = b"".join(b'\x00' + arr[y, :, :3].tobytes() for y in range(H))
    #         def chunk(tag, data):
    #             c = zlib.crc32(tag+data) & 0xffffffff
    #             return struct.pack('>I',len(data))+tag+data+struct.pack('>I',c)
    #         sig = b'\x89PNG\r\n\x1a\n'
    #         ihdr= chunk(b'IHDR', struct.pack('>IIBBBBB',W,H,8,2,0,0,0))
    #         idat= chunk(b'IDAT', zlib.compress(raw,9))
    #         iend= chunk(b'IEND', b'')
    #         with open(path,'wb') as f: f.write(sig+ihdr+idat+iend)
    #     save_png_rgb("example1_numpy.png", img)
    #     print("  已保存 example1_numpy.png（内置 PNG 写入器）")

    # # ── 示例 2：正交相机 ────────────────────────────────────────────────────
    # print()
    # print("=" * 50)
    # print("示例 2：正交相机（俯视）")
    # print("=" * 50)

    # cam_ortho = CameraParams(
    #     is_perspective = False,
    #     field_of_view  = 12.0,    # 视口半高 = 12 世界单位
    #     position       = (0., 0., 100.),
    #     direction      = (0., 0., -1.),
    #     up             = (0., 1.,  0.),
    # )
    # img2 = ren.render(pos, col, rad, camera=cam_ortho, box_edges=box_edges)
    # print(f"  图像 shape={img2.shape}  非零像素={(img2[:,:,:3]>0).any(2).sum()}")
    # try:
    #     from PIL import Image
    #     Image.fromarray(img2).save("example2_ortho.png")
    #     print("  已保存 example2_ortho.png")
    # except ImportError:
    #     pass

    # # ── 示例 3：稳定性测试（10 次重复渲染）────────────────────────────────
    # print()
    # print("=" * 50)
    # print("示例 3：稳定性测试（10 次重复渲染）")
    # print("=" * 50)
    # ren_fast = TachyonRender(width=200, height=150,
    #                          antialiasing=False, ao=False, shadows=False)
    # for i in range(10):
    #     img_t = ren_fast.render(pos, col, rad, box_edges=box_edges)
    #     assert img_t.shape == (150, 200, 4), f"第 {i} 次形状错误"
    # print("  10 次全部通过 ✓")

    # ── 示例 4：从 mdapy.System（如果可用）───────────────────────────────
    import matplotlib.pyplot as plt
    if _HAS_MDAPY:
        print()
        print("=" * 50)
        print("示例 4：从 mdapy.System 渲染 FCC Al")
        print("=" * 50)
        sys_al = mp.build_crystal('Ni', 'fcc', 3.6, nx=5, ny=5, nz=5)
        # sys_al.write_xyz('test.xyz')
        print(f"  原子数：{sys_al.N}")
        cam = CameraParams(False, field_of_view=13.42, position=(6.59559, 5.99291, 9.8026), direction=(0.58, 0.76, -0.31),
                            up=(0.19, 0.25, 0.95), znear=-15 ,dof_enabled=True)
        ren2  = TachyonRender(width=900, height=900,
                              background=(1, 1, 1), direct_light_intensity=1.2, 
                              aa_samples=20, ao_brightness=1.)
        img_al = ren2.render_system(sys_al, camera=cam, 
                                    default_radius=1.3, box_color=(0, 0, 0, 1))
        plt.imshow(img_al); plt.axis("off"); plt.tight_layout(); plt.show()
    else:
        print("\n（mdapy 未安装，跳过示例 4）")