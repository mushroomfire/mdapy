# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

try:
    import k3d
except ImportError:
    raise ImportError("One need install k3d: https://k3d-jupyter.org/user/index.html")

import numpy as np
import polars as pl
from k3d import Plot
from k3d.objects import Text2d
from mdapy.system import System
from mdapy.data import ele_radius, ele_dict, type_dict, struc_dict
from typing import Optional, Tuple


class View:
    """
    Visualize atomic systems using k3d.

    Parameters
    ----------
    system : System
        MDAPY System object containing atomic positions, box information
        and optional per-atom properties such as element, type, radius, etc.

    Attributes
    ----------
    system : System
        The input atomic system.
    plot : Plot
        The k3d canvas used for visualization.
    atoms : k3d.points
        Object storing atomic coordinates, colors and radii.
    box : k3d.lines
        Object representing the simulation box.
    label : Text2d or None
        The text label for colorbar. Only created when `colored_by()` is used.
    """

    def __init__(self, system: System):
        self.system = system
        self.label: Optional[Text2d] = None
        self.init_plot()

    def _box2lines(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert simulation box into line vertices and indices.

        Returns
        -------
        vertices : (8, 3) float32 ndarray
            Coordinates of the eight box corners.
        indices : (12, 2) float32 ndarray
            Pairs of indices defining the 12 box edges.
        """
        vertices = np.zeros((8, 3), dtype=np.float32)
        origin = self.system.box.origin
        AB = self.system.box.box[0]
        AD = self.system.box.box[1]
        AA1 = self.system.box.box[2]

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

    def colored_by_type(self) -> None:
        """
        Color atoms using their type values.

        Notes
        -----
        The type is mapped cyclically into nine predefined colors.
        Colors are updated in `system.data["color"]` and applied
        to the k3d point object if already initialized.
        """
        self.system.update_data(
            self.system.data.with_columns(
                ((pl.col("type") - 1) % 9 + 1)
                .replace_strict(type_dict, return_dtype=pl.UInt32)
                .alias("color")
            )
        )
        if hasattr(self, "atoms"):
            self.atoms.colors = self.system.data["color"].to_numpy()

    def colored_by_element(self) -> None:
        """
        Color atoms using their element name.

        Notes
        -----
        Element symbols are mapped to integer colors according to `ele_dict`.
        """
        self.system.update_data(
            self.system.data.with_columns(
                pl.col("element")
                .replace_strict(ele_dict, return_dtype=pl.UInt32)
                .alias("color")
            )
        )
        if hasattr(self, "atoms"):
            self.atoms.colors = self.system.data["color"].to_numpy()

    def _init_color(self) -> None:
        """
        Initialize color column.

        Notes
        -----
        If `color` exists, cast to uint32.
        If not, determine color based on element or type.
        """
        if "color" not in self.system.data.columns:
            if "element" in self.system.data.columns:
                self.colored_by_element()
            else:
                assert "type" in self.system.data.columns
                self.colored_by_type()
        else:
            self.system.update_data(
                self.system.data.with_columns(pl.col("color").cast(pl.UInt32))
            )

    def _init_radius(self) -> None:
        """
        Initialize radius column.

        Notes
        -----
        If element exists, use element-specific radius.
        Otherwise assign a default radius of 2.0.
        """
        if "radius" not in self.system.data.columns:
            if "element" in self.system.data.columns:
                self.system.update_data(
                    self.system.data.with_columns(
                        pl.col("element")
                        .replace_strict(ele_radius, return_dtype=pl.Float32)
                        .alias("radius")
                    )
                )
            else:
                self.system.update_data(
                    self.system.data.with_columns(
                        pl.lit(2.0, pl.Float32).alias("radius")
                    )
                )
        else:
            self.system.update_data(
                self.system.data.with_columns(pl.col("radius").cast(pl.Float32))
            )

    def init_plot(self) -> None:
        """
        Initialize the visualization canvas.

        Notes
        -----
        - Creates box lines, atomic points.
        - Initializes default color and radius.
        - Creates element/type legends using `text2d`.
        """
        vertices, indices = self._box2lines()
        self.plot: Plot = k3d.plot(height=600)

        self._init_color()
        self._init_radius()

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
            self.system.data.select("x", "y", "z").cast(pl.Float32).to_numpy(),
            colors=self.system.data["color"].to_numpy(),
            shader="3d",
            point_sizes=self.system.data["radius"].to_numpy(),
            group="atoms",
        )

        self.plot += self.box
        self.plot += self.atoms
        self.plot.grid_visible = False

        # Legend drawing
        if "element" in self.system.data.columns:
            res = self.system.data["element"].unique().sort()
            pos = [0.0, 0.0]
            for i, j in enumerate(res, start=1):
                pos[0] = i * 0.03
                self.plot += k3d.text2d(
                    j,
                    position=pos,
                    size=1.5,
                    is_html=True,
                    label_box=True,
                    color=ele_dict[j],
                    group="element",
                    name=f"{j}",
                )
        else:
            pos = [0.0, 0.0]
            for i, j in enumerate(self.system.data["type"].unique()):
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

    def display(self) -> None:
        """
        Display the k3d plot in supported environments (e.g., Jupyter).
        """
        self.plot.display()

    def close(self) -> None:
        """
        Close the k3d plot and release the rendering canvas.
        """
        self.plot.close()

    def hide_object_by_group_name(self, name: str, remove: bool = False) -> None:
        """
        Hide or remove k3d objects by their group name.

        Parameters
        ----------
        name : str
            The object group to hide/remove.
        remove : bool, default False
            If True, remove the object entirely.
            If False, only hide it visually.
        """
        for i in self.plot.objects:
            if i.group == name:
                if remove:
                    self.plot -= i
                else:
                    i.visible = False

    def delete_color_bar(self) -> None:
        """
        Remove existing colorbar from the plot.
        """
        self.atoms.color_map = []
        self.atoms.color_range = []
        if self.label is not None:
            self.label.visible = False

    def _colored_by_structure_type(self, method: str, show_label: bool = False) -> None:
        """
        Color atoms based on structural classification.

        Parameters
        ----------
        method : {'ptm', 'cna', 'aja', 'ids'}
            Column name storing structural type.
        show_label : bool, default False
            Whether to show per-structure text labels.

        Notes
        -----
        Updates atom colors using predefined structure → color mapping.
        Clears colorbar since structure type does not require continuous scale.
        """
        avia_method = ["ptm", "cna", "aja", "ids"]
        assert method in avia_method
        assert method in self.system.data.columns
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

        self.system.update_data(
            self.system.data.with_columns(
                pl.col(method)
                .replace_strict(color_struc, return_dtype=pl.UInt32)
                .alias("color")
            )
        )

        number = self.system.data.group_by(method).len()
        number_dict = {number[i, 0]: number[i, 1] for i in range(number.shape[0])}

        N = self.system.N
        pos = [0.0, 0.0]

        for i in struc.keys():
            pos[1] = i * 0.07
            n = number_dict.get(i, 0)
            self.plot += k3d.text2d(
                f"{struc[i]} {n} {(n / N) * 100:.1f}%",
                position=pos,
                size=1.5,
                is_html=True,
                label_box=True,
                color=color_struc[i],
                group=method,
                name=struc[i],
            )

        self.atoms.colors = self.system.data["color"].to_numpy()

        if not show_label:
            self.hide_object_by_group_name(method)

        self.delete_color_bar()

    def colored_by(
        self,
        name: str,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "rainbow",
    ) -> None:
        """
        Color atoms based on a given scalar per-atom quantity.

        Parameters
        ----------
        name : str
            Column name in `system.data` used for coloring.
        vmin : float, optional
            Minimum value of the colormap. If None, automatically determined.
        vmax : float, optional
            Maximum value of the colormap. If None, automatically determined.
        cmap : str, default "rainbow"
            Matplotlib colormap name.

        Notes
        -----
        - If `name` is "element" or "type", discrete coloring is applied.
        - If `name` is a structure classifier ("ptm", "cna", "aja", "ids"),
          structure coloring is used.
        - Otherwise, continuous colormap coloring is used.
        - Colorbar is updated accordingly.
        """
        assert name in self.system.data.columns

        # Element/type special handling
        if name == "element":
            self.colored_by_element()
            return
        elif name == "type":
            self.colored_by_type()
            return
        elif name in ["ptm", "cna", "aja", "ids"]:
            self._colored_by_structure_type(name, True)
            return

        import matplotlib as mpl

        # Determine range
        if vmin is not None and vmax is not None:
            assert vmin < vmax
        else:
            vmin = float(self.system.data[name].min())
            vmax = float(self.system.data[name].max())

        cmap_obj = mpl.colormaps[cmap]
        colors_rgb = np.array(cmap_obj(range(256))[:, :-1] * 255, dtype=np.uint32)
        delta = vmax - vmin

        if delta < 1e-4:
            # Assign middle color when no range exists
            r, g, b = colors_rgb[len(colors_rgb) // 2]
            colors = np.full(self.system.N, (r << 16) + (g << 8) + b, np.uint32)
        else:
            N = colors_rgb.shape[0]
            factor = (N - 1) / delta

            index = (
                self.system.data.select(
                    pl.when(pl.col(name) > vmax)
                    .then(pl.lit(vmax))
                    .when(pl.col(name) < vmin)
                    .then(pl.lit(vmin))
                    .otherwise(pl.col(name))
                    .alias(name)
                )
                .select(((pl.col(name) - vmin) * factor).cast(pl.UInt32))[name]
                .to_numpy()
            )

            r, g, b = colors_rgb[index].T
            colors = (r << 16) + (g << 8) + b

        self.atoms.colors = colors

        # Color map
        c_cmap = (
            np.c_[np.linspace(0, 1, 256), cmap_obj(np.linspace(0, 1, 256))[:, :-1]]
            .flatten()
            .astype(np.float32)
        )
        self.atoms.color_map = c_cmap

        # Color range
        if delta > 1e-4:
            self.atoms.color_range = [vmin, vmax]
        else:
            self.atoms.color_range = [vmin - 3, vmin + 3]

        # Colorbar label
        if self.label is None:
            self.label = k3d.text2d(
                name,
                position=(0.01, 0.5),
                size=2,
                is_html=True,
                label_box=False,
                color=0,
                group="colorbar",
            )
            self.plot += self.label
        else:
            self.label.text = name
            self.label.visible = True

        # Hide structure/type/element legends
        for i in self.plot.objects:
            if i.group in ["ptm", "cna", "aja", "ids", "element", "type"]:
                self.hide_object_by_group_name(i.group)

        self.system.update_data(self.system.data.with_columns(color=colors))
