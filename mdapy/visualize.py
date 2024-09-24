# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import k3d
import numpy as np
import polars as pl
import taichi as ti
import matplotlib as mpl

try:
    from .box import init_box
    from .tool_function import ele_radius, ele_dict, struc_dict, type_dict
except Exception:
    from box import init_box
    from tool_function import ele_radius, ele_dict, struc_dict, type_dict


for i in ele_radius.keys():
    ele_radius[i] = str(ele_radius[i])
ele_dict_str = {}
for i in ele_dict.keys():
    ele_dict_str[i] = str(ele_dict[i])


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
    def __init__(self, data, box) -> None:
        assert isinstance(data, pl.DataFrame)
        self.data = data
        self.label = None
        self.init_plot(*self.box2lines(box))

    def box2lines(self, box):
        new_box, _, _ = init_box(box)

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
            if "type_name" in self.data.columns:
                self.atom_colored_by_atom_type_name()
            else:
                self.atom_colored_by_atom_type()

    def init_radius(self):
        if "radius" not in self.data.columns:
            if "type_name" in self.data.columns:
                self.data = self.data.with_columns(
                    pl.col("type_name")
                    .replace_strict(ele_radius, default="2.0")
                    .cast(pl.Float32)
                    .alias("radius")
                )
            else:
                self.data = self.data.with_columns(
                    pl.lit(2.0, pl.Float32).alias("radius")
                )

    def init_plot(self, vertices, indices):
        self.plot = k3d.plot(height=600)
        self.init_color()
        self.init_radius()

        self.box = k3d.lines(
            vertices,
            indices,
            color=0,
            indices_type="segment",
            width=1.5,
            shader="simple",
            group="box",
        )
        self.atoms = k3d.points(
            self.data.select("x", "y", "z").to_numpy().astype(np.float32),
            colors=np.array(self.data["color"].to_numpy(), np.uint32),
            shader="3d",
            point_sizes=self.data["radius"].to_numpy().astype(np.float32),
            group="atoms",
        )
        self.plot += self.box
        self.plot += self.atoms
        self.plot.grid_visible = False

        if "type_name" in self.data.columns:
            res = self.data.unique("type_name").sort("type").select("type_name", "type")
            res = {res[i, 0]: res[i, 1] for i in range(res.shape[0])}
            pos = [0.0, 0.0]
            for i, j in enumerate(self.data["type_name"].unique()):
                pos[0] = i * 0.03
                self.plot += k3d.text2d(
                    j,
                    position=pos,
                    size=1.5,
                    is_html=True,
                    label_box=True,
                    color=ele_dict[j],
                    group="type_name",
                    name=f"{j} (Type {res[j]})",
                )
        else:
            pos = [0.0, 0.0]
            for i, j in enumerate(self.data["type"].unique()):
                pos[0] = i * 0.05
                if pos[0] > 0.45:
                    pos[0] = (i - 10) * 0.05
                    pos[1] = 0.07
                self.plot += k3d.text2d(
                    f"Type {j:2}",
                    position=pos,
                    size=1.5,
                    is_html=True,
                    label_box=True,
                    color=type_dict[(j - 1) % 9 + 1],
                    group="type",
                    name=f"Type {j}",
                )

    def hide_object_by_group_name(self, name, remove=False):
        for i in self.plot.objects:
            if i.group == name:
                if remove:
                    self.plot -= i
                else:
                    i.visible = False

    def show_object_by_group_name(self, name):
        found_name = False
        for i in self.plot.objects:
            if i.group == name:
                i.visible = True
                found_name = True
        if not found_name:
            print(f"Did not find {name}.")

    def display(self):
        self.plot.display()
        self.hide_object_by_group_name("type_name")
        self.hide_object_by_group_name("type")

    def close(self):
        self.plot.close()

    def delete_color_bar(self):
        self.atoms.color_map = []
        self.atoms.color_range = []
        if self.label is not None:
            self.label.visible = False
        # if self.label is not None:
        #     self.plot -= self.label
        #     self.label = None

    def atom_colored_by_atom_type(self):
        self.data = self.data.with_columns(
            ((pl.col("type") - 1) % 9 + 1)
            .replace_strict(type_dict, return_dtype=pl.UInt32)
            .alias("color")
        )
        if hasattr(self, "atoms"):
            self.atoms.colors = np.array(self.data["color"].to_numpy(), np.uint32)
            self.delete_color_bar()

    def atom_colored_by_atom_type_name(self):
        n = 1
        for i in self.data["type_name"].unique():
            if i not in ele_dict.keys():
                ele_dict_str[i] = str(type_dict[n % 9])
                n += 1
        self.data = self.data.with_columns(
            pl.col("type_name")
            .replace_strict(ele_dict_str)
            .cast(pl.UInt32)
            .alias("color")
        )
        if hasattr(self, "atoms"):
            self.atoms.colors = np.array(self.data["color"].to_numpy(), np.uint32)
            for i in ["ptm", "cna", "aja", "ids"]:
                self.hide_object_by_group_name(i)
            self.delete_color_bar()

    def atom_colored_by_structure_type(self, method, show_label=False):
        avia_method = ["ptm", "cna", "aja", "ids"]
        assert method in avia_method
        assert method in self.data.columns
        N = self.data.shape[0]
        for i in avia_method:
            if method != i:
                self.hide_object_by_group_name(i)
        if method == "ptm":
            struc = {
                0: "Other",
                1: "FCC",
                2: "HCP",
                3: "BCC",
                4: "ICO",
                5: "Simple cubic",
                6: "Cubic diamond",
                7: "Hexagonal diamond",
                8: "Graphene",
            }
        elif method == "cna" or method == "aja":
            struc = {
                0: "Other",
                1: "FCC",
                2: "HCP",
                3: "BCC",
                4: "ICO",
            }
        elif method == "ids":
            struc = {
                0: "Other",
                1: "Cubic diamond",
                2: "Cubic diamond (1st neighbor)",
                3: "Cubic diamond (2nd neighbor)",
                4: "Hexagonal diamond",
                5: "Hexagonal diamond (1st neighbor)",
                6: "Hexagonal diamond (2nd neighbor)",
            }

        color_struc = {i: struc_dict[struc[i]] for i in struc.keys()}
        self.data = self.data.with_columns(
            pl.col(method)
            .replace_strict(color_struc, return_dtype=pl.UInt32)
            .alias("color")
        )
        number = self.data.group_by(method).len()
        number_dict = {number[i, 0]: number[i, 1] for i in range(number.shape[0])}
        pos = [0.0, 0.0]
        for i in struc.keys():
            pos[1] = i * 0.07
            if i in number_dict.keys():
                n = number_dict[i]
            else:
                n = 0
            self.plot += k3d.text2d(
                f"{struc[i]} {n} {(n/N)*100:.1f}%",
                position=pos,
                size=1.5,
                is_html=True,
                label_box=True,
                color=color_struc[i],
                group=method,
                name=struc[i],
            )

        self.atoms.colors = np.array(self.data["color"].to_numpy(), np.uint32)
        if not show_label:
            self.hide_object_by_group_name(method)

        self.delete_color_bar()

    def atom_colored_by(self, values, vmin=None, vmax=None, cmap="rainbow"):
        value_name = values
        if isinstance(values, str):
            assert values in self.data.columns
            if values != "type_name":
                assert self.data[values].dtype in pl.NUMERIC_DTYPES
            if values == "type":
                self.atom_colored_by_atom_type()
                return
            elif values == "type_name":
                self.atom_colored_by_atom_type_name()
                return
            elif values in ["ptm", "cna", "aja", "ids"]:
                self.atom_colored_by_structure_type(values)
                return
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
                    position=(0.01, 0.5),
                    size=2,
                    is_html=True,
                    label_box=False,
                    color=0,
                    group="colorbar",
                )
                self.plot += self.label
            else:
                self.label.text = value_name
                self.label.visible = True
        self.data = self.data.with_columns(pl.lit(colors).alias("colors"))
