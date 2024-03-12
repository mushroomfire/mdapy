import k3d
import numpy as np
import polars as pl
import taichi as ti
import matplotlib as mpl


@ti.kernel
def value2color(
    colors_rgb: ti.types.ndarray(element_dim=1),
    value: ti.types.ndarray(),
    vmin: float,
    vmax: float,
    colors: ti.types.ndarray(),
):
    delta = vmax - vmin
    N = colors_rgb.shape[0]
    fac = (N - 1) / delta

    for i in range(value.shape[0]):
        val = ti.float64(value[i])
        if val > vmax:
            val = vmax
        elif val < vmin:
            val = vmin
        r, g, b = colors_rgb[ti.floor((val - vmin) * fac, int)]
        colors[i] = (r << 16) + (g << 8) + b


class Visualize:
    rgb_type = np.array(
        [
            [85, 170, 255],
            [102, 102, 255],
            [255, 255, 178],
            [255, 102, 255],
            [255, 255, 0],
            [204, 255, 179],
            [179, 0, 255],
        ],
        np.uint32,
    )
    rgb_structure_type = np.array(
        [
            [243, 243, 243],
            [102, 255, 102],
            [255, 102, 102],
            [102, 102, 255],
            [243, 204, 51],
            [160, 20, 254],
            [19, 160, 254],
            [254, 137, 0],
            [160, 120, 254],
        ],
        np.uint32,
    )

    def __init__(self, data, box) -> None:
        assert isinstance(data, pl.DataFrame)
        self.data = data
        self.label = None
        self.init_plot(*self.box2lines(box))

    def box2lines(self, box):
        assert isinstance(box, np.ndarray)
        assert box.shape == (3, 2) or box.shape == (4, 3)
        if box.shape == (3, 2):
            new_box = np.zeros((4, 3), dtype=box.dtype)
            new_box[0, 0], new_box[1, 1], new_box[2, 2] = box[:, 1] - box[:, 0]
            new_box[-1] = box[:, 0]
        else:
            assert box[0, 1] == 0
            assert box[0, 2] == 0
            assert box[1, 2] == 0
            new_box = box
        vertices = np.zeros((8, 3), dtype=np.float32)
        origin = new_box[-1]
        AB = new_box[0]
        AD = new_box[1]
        AA1 = new_box[2]
        vertices[0] = origin
        vertices[1] = origin + AB
        vertices[2] = origin + AB + AD
        vertices[3] = origin + AD
        vertices[4] = vertices[0] + AA1
        vertices[5] = vertices[1] + AA1
        vertices[6] = vertices[2] + AA1
        vertices[7] = vertices[3] + AA1
        indices = np.zeros((12, 2), dtype=np.float32)
        indices[0] = [0, 1]
        indices[1] = [1, 2]
        indices[2] = [2, 3]
        indices[3] = [3, 0]
        indices[4] = [0, 4]
        indices[5] = [1, 5]
        indices[6] = [2, 6]
        indices[7] = [3, 7]
        indices[8] = [4, 5]
        indices[9] = [5, 6]
        indices[10] = [6, 7]
        indices[11] = [7, 4]
        return vertices, indices

    def init_color(self):
        if "color" not in self.data.columns:
            if "structure_types" not in self.data.columns:
                self.atom_colored_by_atom_type()
            else:
                self.atom_colored_by_structure_type()

    def init_plot(self, vertices, indices):
        self.init_color()
        self.plot = k3d.plot()
        self.box = k3d.lines(
            vertices,
            indices,
            color=0,
            indices_type="segment",
            width=1.5,
            shader="simple",
        )
        self.atoms = k3d.points(
            self.data.select("x", "y", "z").to_numpy().astype(np.float32),
            colors=np.array(self.data["colors"].to_numpy(), np.uint32),
            shader="3d",
            point_size=2.5,
        )
        self.plot += self.box
        self.plot += self.atoms
        self.plot.grid_visible = False

    def display(self):
        self.plot.display()

    def close(self):
        self.plot.close()

    def atom_colored_by_atom_type(self):
        decimal = (
            (self.rgb_type[:, 0] << 16)
            + (self.rgb_type[:, 1] << 8)
            + self.rgb_type[:, 2]
        )
        type2color = {i: j for i, j in enumerate(decimal)}
        self.data = self.data.with_columns(
            ((pl.col("type") - 1) % len(decimal)).replace(type2color).alias("colors")
        )

    def atom_colored_by_structure_type(self):
        decimal = (
            (self.rgb_structure_type[:, 0] << 16)
            + (self.rgb_structure_type[:, 1] << 8)
            + self.rgb_structure_type[:, 2]
        )
        type2color = {i: j for i, j in enumerate(decimal)}
        self.data = self.data.with_columns(
            (pl.col("structure_types") % len(decimal))
            .replace(type2color)
            .alias("colors")
        )

    def atoms_colored_by(self, values, vmin=None, vmax=None, cmap="rainbow"):
        value_name = values
        if isinstance(values, str):
            assert values in self.data.columns
            if values == "type":
                self.atom_colored_by_atom_type()
                self.atoms.colors = np.array(self.data["colors"].to_numpy(), np.uint32)
                self.atoms.color_map = []
                self.atoms.color_range = []
                if self.label is not None:
                    self.plot -= self.label
                    self.label = None
                return
            elif values == "structure_types":
                self.atom_colored_by_structure_type()
                self.atoms.colors = np.array(self.data["colors"].to_numpy(), np.uint32)
                self.atoms.color_map = []
                self.atoms.color_range = []
                if self.label is not None:
                    self.plot -= self.label
                    self.label = None
                return
            else:
                assert self.data[values].dtype in pl.NUMERIC_DTYPES
                values = self.data[values].to_numpy()

        else:
            assert values.shape[0] == self.data.shape[0]

        if vmin is not None and vmax is not None:
            assert vmin < vmax
        else:
            vmin, vmax = float(values.min()), float(values.max())

        cmap = mpl.colormaps[cmap]

        colors_rgb = np.array(cmap(range(256))[:, :-1] * 255, dtype=int)

        colors = np.zeros(values.shape[0], dtype=int)
        if vmax - vmin > 1e-4:
            value2color(colors_rgb, values, vmin, vmax, colors)
        else:
            r, g, b = colors_rgb[int(len(colors_rgb) / 2)]
            colors += (r << 16) + (g << 8) + b

        colors = colors.astype(np.uint32)
        self.atoms.colors = colors
        c_cmap = (
            np.c_[np.linspace(0, 1, 256), cmap(np.linspace(0, 1, 256))[:, :-1]]
            .flatten()
            .astype(np.float32)
        )
        self.atoms.color_map = c_cmap
        if vmax - vmin > 1e-4:
            self.atoms.color_range = [vmin, vmax]
        else:
            self.atoms.color_range = [vmin - 5, vmin + 5]

        if isinstance(value_name, str):
            if self.label is None:
                self.label = k3d.text2d(
                    value_name,
                    position=(0.015, 0.43),
                    size=2,
                    is_html=True,
                    label_box=False,
                    color=0,
                )
                self.plot += self.label
            else:
                self.label.text = value_name
        self.data = self.data.with_columns(pl.lit(colors).alias("colors"))
