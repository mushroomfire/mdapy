# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.


import numpy as np
import polars as pl
import polyscope as ps
import polyscope.imgui as psim
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from . import __version__
import taichi as ti

try:
    from .system import System
    from .lattice_maker import LatticeMaker
    from .create_polycrystalline import CreatePolycrystalline
except Exception:
    from system import System
    from lattice_maker import LatticeMaker
    from create_polycrystalline import CreatePolycrystalline
import argparse


def init_global_parameters():
    globals()["system"] = None
    globals()["atoms"] = None
    globals()["filename"] = ""
    globals()["init_file"] = False
    rgb_structure_type = (
        np.array(
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
        )
        / 255
    )
    globals()["Other"] = rgb_structure_type[0]
    globals()["FCC"] = rgb_structure_type[1]
    globals()["HCP"] = rgb_structure_type[2]
    globals()["BCC"] = rgb_structure_type[3]
    globals()["ICO"] = rgb_structure_type[4]
    globals()["Simple_Cubic"] = rgb_structure_type[5]
    globals()["Cubic_Diamond"] = rgb_structure_type[6]
    globals()["Hexagonal_Diamond"] = rgb_structure_type[7]
    globals()["Graphene"] = rgb_structure_type[8]
    globals()["cna_rc"] = 3.0
    globals()["csp_n_neigh"] = 12
    globals()["rx"] = 1
    globals()["ry"] = 1
    globals()["rz"] = 1
    globals()["dump_compress"] = False

    globals()["data_fmt"] = ["atomic", "charge"]
    globals()["data_fmt_default"] = "atomic"

    globals()["ptm_other"] = True
    globals()["ptm_fcc"] = True
    globals()["ptm_hcp"] = True
    globals()["ptm_bcc"] = True
    globals()["ptm_ico"] = False
    globals()["ptm_scubic"] = False
    globals()["ptm_dcubic"] = False
    globals()["ptm_dhex"] = False
    globals()["ptm_gra"] = False

    globals()["ptm_rmsd_threshold"] = 0.1
    globals()["ptm_return_rmsd"] = False
    globals()["ptm_return_interatomic_distance"] = False
    globals()["ptm_return_wxyz"] = False

    globals()["spatial_directions"] = ["x", "y", "z", "xy", "xz", "yz", "xyz"]
    globals()["spatial_direction_default"] = "x"
    globals()["input_values"] = ["id", "type", "x", "y", "z"]
    globals()["input_value_default"] = "id"
    globals()["operations"] = ["mean", "sum", "min", "max"]
    globals()["operation_default"] = "mean"
    globals()["wbin"] = 5.0
    globals()["binning_plot"] = True

    globals()["temp_element"] = "H, C, O"
    globals()["temp_rc"] = 5.0
    globals()["temp_unit"] = ["metal", "charge"]
    globals()["temp_unit_default"] = "metal"

    globals()["sbo_mode"] = ["nearest", "cutoff"]
    globals()["sbo_mode_default"] = "nearest"
    globals()["sbo_neigh_number"] = 12
    globals()["sbo_cutoff"] = 4.0
    globals()["sbo_qlist"] = "4, 6, 8"
    globals()["sbo_wlflag"] = False
    globals()["sbo_wlhatflag"] = False
    globals()["sbo_solidliquid"] = False
    globals()["sbo_solidliquid_threshold"] = 0.7
    globals()["sbo_solidliquid_nbond"] = 7
    globals()["solid"] = np.array([59, 198, 170]) / 255
    globals()["liquid"] = np.array([52, 68, 223]) / 255

    globals()["entropy_rc"] = 5.0
    globals()["entropy_sigma"] = 0.2
    globals()["entropy_use_local_density"] = False
    globals()["entropy_compute_average"] = True
    globals()["entropy_average_rc"] = 4.0

    globals()["rdf_rc"] = 5.0
    globals()["rdf_nbin"] = 200
    globals()["rdf_plot"] = True
    globals()["rdf_plot_partial"] = False

    globals()["cnp_rc"] = 3.0
    globals()["wcp_rc"] = 3.0
    globals()["wcp_element_list"] = "Al, Fe, Cu"
    globals()["wcp_res"] = ""

    globals()["cluster_rc"] = 5.0
    globals()["cluster_num"] = 0

    globals()["SF_rmsd_threshold"] = 0.1
    globals()["ISF"] = np.array([0.9, 0.7, 0.2])
    globals()["ESF"] = np.array([132 / 255, 25 / 255, 255 / 255])
    globals()["TB"] = np.array([14 / 255, 133 / 255, 160 / 255])

    globals()["lattice_type_default"] = "FCC"
    globals()["lattice_type_list"] = ["FCC", "BCC", "HCP", "GRA"]
    globals()["lattice_constant"] = 4.05
    globals()["lrx"] = 1
    globals()["lry"] = 1
    globals()["lrz"] = 1
    globals()["orientation_x"] = "1, 0, 0"
    globals()["orientation_y"] = "0, 1, 0"
    globals()["orientation_z"] = "0, 0, 1"

    globals()["box_length"] = "100, 100, 100"
    globals()["grain_number"] = 10
    globals()["metal_lattice_type_default"] = "FCC"
    globals()["metal_lattice_type_list"] = ["FCC", "BCC", "HCP"]
    globals()["metal_lattice_constant"] = 3.615
    globals()["random_seed"] = 1
    globals()["metal_overlap_dis"] = 2.5
    globals()["add_graphene"] = False
    globals()["gra_lattice_constant"] = 1.42
    globals()["metal_gra_overlap_dis"] = 3.0825
    globals()["gra_overlap_dis"] = 1.2

    globals()["xyz_classical"] = False
    globals()["xyz_give_type"] = False
    globals()["xyz_type_list"] = "Al, Fe"

    globals()["cif_give_type"] = False
    globals()["cif_type_list"] = "Al, Fe"

    globals()["poscar_give_type"] = False
    globals()["poscar_type_list"] = "Al, Fe"
    globals()["reduced_pos"] = False
    globals()["save_velocity"] = False

    globals()["species_element_list"] = "C, H, O"
    globals()["identify_mode_default"] = "check most"
    globals()["identify_mode"] = ["check most", "search species"]
    globals()["search_species"] = "H2O, CO2, H2"
    globals()["check_most"] = 10
    globals()["species"] = None

    globals()["cell_length"] = 4.0
    globals()["out_void"] = False
    globals()["void_number"] = None
    globals()["void_volume"] = None


def box2lines(box):
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


def box2axis(box):
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

    AB = new_box[0]
    AD = new_box[1]
    AA1 = new_box[2]
    fac = 0.1
    origin = new_box[-1] - (AB + AD + AA1) * fac * 2
    vertices = np.zeros((22, 3), dtype=np.float32)
    min_length = max(
        4, min([np.linalg.norm(AB), np.linalg.norm(AD), np.linalg.norm(AA1)]) * fac
    )

    vertices[0] = origin
    vertices[1] = origin + np.array([min_length * 2, 0.0, 0.0])
    vertices[2] = origin + np.array([0.0, min_length * 2, 0.0])
    vertices[3] = origin + np.array(
        [
            0.0,
            0.0,
            min_length * 2,
        ]
    )

    # plot x
    vertices[4] = vertices[1] + np.array([min_length * 0.5, 0, min_length * 0.5])
    vertices[5] = vertices[1] + np.array([min_length * 0.5 * 3, 0, -min_length * 0.5])
    vertices[6] = vertices[1] + np.array([min_length * 0.5, 0, -min_length * 0.5])
    vertices[7] = vertices[1] + np.array([min_length * 0.5 * 3, 0, min_length * 0.5])

    # plot y
    vertices[8] = vertices[2] + np.array([0, min_length * 0.5, min_length * 0.5])
    vertices[9] = vertices[2] + np.array([0, min_length * 0.5 * 2, 0])
    vertices[10] = vertices[2] + np.array([0, min_length * 0.5 * 3, min_length * 0.5])
    vertices[11] = vertices[2] + np.array([0, min_length * 0.5 * 2, -min_length * 0.5])

    # plot z
    vertices[12] = vertices[3] + np.array([min_length * 0.5, 0, min_length * 0.5])
    vertices[13] = vertices[3] + np.array([-min_length * 0.5, 0, min_length * 0.5])
    vertices[14] = vertices[3] + np.array([-min_length * 0.5, 0, min_length * 0.5 * 3])
    vertices[15] = vertices[3] + np.array([min_length * 0.5, 0, min_length * 0.5 * 3])

    # plot x arrow
    vertices[16] = vertices[1] + np.array([-min_length * 0.5, 0, min_length * 0.5])
    vertices[17] = vertices[1] + np.array([-min_length * 0.5, 0, -min_length * 0.5])

    # plot y arrow
    vertices[18] = vertices[2] + np.array([0, -min_length * 0.5, min_length * 0.5])
    vertices[19] = vertices[2] + np.array([0, -min_length * 0.5, -min_length * 0.5])

    # plot x arrow
    vertices[20] = vertices[3] + np.array([-min_length * 0.5, 0, -min_length * 0.5])
    vertices[21] = vertices[3] + np.array([min_length * 0.5, 0, -min_length * 0.5])

    indices = np.zeros((17, 2), dtype=np.float32)
    indices[0] = [0, 1]
    indices[1] = [0, 2]
    indices[2] = [0, 3]

    # plot x
    indices[3] = [4, 5]
    indices[4] = [6, 7]
    # plot y
    indices[5] = [8, 9]
    indices[6] = [9, 10]
    indices[7] = [9, 11]
    # plot z
    indices[8] = [12, 13]
    indices[9] = [13, 15]
    indices[10] = [14, 15]
    # plot x arrow
    indices[11] = [1, 16]
    indices[12] = [1, 17]
    # plot y arrow
    indices[13] = [2, 18]
    indices[14] = [2, 19]
    # plot z arrow
    indices[15] = [3, 20]
    indices[16] = [3, 21]

    return vertices, indices


def loadfile(filename):
    try:
        ps.info(f"Loading {filename} ...")
        system = System(rf"{filename}")
        atoms = ps.register_point_cloud("Atoms", system.pos)
        atoms.set_radius(1.2, relative=False)
        for name in system.data.columns:
            if name == "type":
                atoms.add_scalar_quantity(
                    name, system.data[name].view(), enabled=True, cmap="jet"
                )
            else:
                if system.data[name].dtype in pl.NUMERIC_DTYPES:
                    atoms.add_scalar_quantity(
                        name, system.data[name].view(), cmap="jet"
                    )
        vertices, indices = box2lines(system.box)
        ps.register_curve_network(
            "Box", vertices, indices, color=(0, 0, 0), radius=0.0015
        )
        vertices, indices = box2axis(system.box)
        ps.register_curve_network(
            "Axis",
            vertices,
            indices,
            color=(0.5, 0.5, 0.5),
            radius=0.0015,
            enabled=False,
        )

        ps.reset_camera_to_home_view()
        ps.info(f"Load {filename} successfully.")
        globals()["system"] = system
        globals()["atoms"] = atoms
    except Exception as e:
        ps.error(str(e))


def whether_init_file(filename):
    global init_file
    if init_file:
        loadfile(filename)
        init_file = False


def load_file_gui():
    if psim.TreeNode("Load file"):
        if psim.Button("Load"):
            Tk().withdraw()
            filename = askopenfilename(
                title="mdapy file loader",
                filetypes=[
                    (
                        "",
                        (
                            "*.dump",
                            "*.dump.gz",
                            "*.data",
                            "*.lmp",
                            "*.xyz",
                            "*.POSCAR",
                            "*.cif",
                        ),
                    ),
                ],
            )
            if len(filename) > 0:
                loadfile(filename)
        psim.TreePop()


def save_file_POSCAR():
    global poscar_give_type, poscar_type_list, reduced_pos, save_velocity

    if psim.TreeNode("Save POSCAR"):
        _, poscar_give_type = psim.Checkbox("assign elemetal name", poscar_give_type)
        if poscar_give_type:
            _, poscar_type_list = psim.InputText("elemental list", poscar_type_list)

        _, reduced_pos = psim.Checkbox("reduced pos", reduced_pos)
        _, save_velocity = psim.Checkbox("save velocity", save_velocity)
        type_name = None
        if poscar_give_type:
            if len(poscar_type_list) > 0:
                type_name = [i.strip() for i in poscar_type_list.split(",")]

        if psim.Button("Save"):
            try:
                if isinstance(system, System):
                    Tk().withdraw()
                    outputname = asksaveasfilename()
                    if len(outputname) > 0:
                        ps.info(f"Saving {outputname} ...")
                        system.write_POSCAR(
                            rf"{outputname}",
                            type_name=type_name,
                            reduced_pos=reduced_pos,
                            save_velocity=save_velocity,
                        )
                        ps.info(f"Save {outputname} successfully.")
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))
        psim.TreePop()


def save_file_xyz():
    global system, xyz_classical, xyz_give_type, xyz_type_list
    if psim.TreeNode("Save XYZ"):
        _, xyz_give_type = psim.Checkbox("assign elemental name", xyz_give_type)
        if xyz_give_type:
            _, xyz_type_list = psim.InputText("elemental list", xyz_type_list)
        _, xyz_classical = psim.Checkbox("classical", xyz_classical)
        type_name = None
        if xyz_give_type:
            if len(xyz_type_list) > 0:
                type_name = [i.strip() for i in xyz_type_list.split(",")]

        if psim.Button("Save"):
            try:
                if isinstance(system, System):
                    Tk().withdraw()
                    outputname = asksaveasfilename()
                    if len(outputname) > 0:
                        ps.info(f"Saving {outputname} ...")
                        system.write_xyz(
                            rf"{outputname}",
                            type_name=type_name,
                            classical=xyz_classical,
                        )
                        ps.info(f"Save {outputname} successfully.")
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))
        psim.TreePop()


def save_file_cif():
    global system, cif_give_type, cif_type_list
    if psim.TreeNode("Save cif"):
        _, cif_give_type = psim.Checkbox("assign elemental name", cif_give_type)
        if cif_give_type:
            _, cif_type_list = psim.InputText("elemental list", cif_type_list)

        type_name = None
        if cif_give_type:
            if len(cif_type_list) > 0:
                type_name = [i.strip() for i in cif_type_list.split(",")]

        if psim.Button("Save"):
            try:
                if isinstance(system, System):
                    Tk().withdraw()
                    outputname = asksaveasfilename()
                    if len(outputname) > 0:
                        ps.info(f"Saving {outputname} ...")
                        system.write_cif(
                            rf"{outputname}",
                            type_name=type_name,
                        )
                        ps.info(f"Save {outputname} successfully.")
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))
        psim.TreePop()


def save_file_dump():
    global dump_compress
    if psim.TreeNode("Save Dump"):
        _, dump_compress = psim.Checkbox("compressed", dump_compress)

        if psim.Button("Save"):
            try:
                if isinstance(system, System):
                    Tk().withdraw()
                    outputname = asksaveasfilename()
                    if len(outputname) > 0:
                        ps.info(f"Saving {outputname} ...")
                        system.write_dump(rf"{outputname}", compress=dump_compress)
                        ps.info(f"Save {outputname} successfully.")
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))
        psim.TreePop()


def save_file_data():
    global data_fmt, data_fmt_default, system
    if psim.TreeNode("Save Data"):
        psim.PushItemWidth(100)
        changed = psim.BeginCombo("data format", data_fmt_default)
        if changed:
            for val in data_fmt:
                _, selected = psim.Selectable(val, data_fmt_default == val)
                if selected:
                    data_fmt_default = val
            psim.EndCombo()
        psim.PopItemWidth()

        if psim.Button("Save"):
            try:
                if isinstance(system, System):
                    Tk().withdraw()
                    outputname = asksaveasfilename()
                    if len(outputname) > 0:
                        ps.info(f"Saving {outputname} ...")
                        system.write_data(
                            rf"{outputname}", data_format=data_fmt_default
                        )
                        ps.info(f"Save {outputname} successfully.")
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))
        psim.TreePop()


def save_file_gui():
    if psim.TreeNode("Save file"):
        save_file_dump()
        save_file_data()
        save_file_xyz()
        save_file_POSCAR()
        save_file_cif()
        psim.TreePop()


def centro_symmetry_parameter():
    global csp_n_neigh, system, atoms
    if psim.TreeNode("CentroSymmetry Parameter"):
        psim.TextUnformatted(
            "For FCC is 12, for BCC is 8. It must be a positive even number."
        )

        _, csp_n_neigh = psim.InputInt("neighbor number", csp_n_neigh)

        if psim.Button("Compute"):
            try:
                if isinstance(system, System):
                    ps.info("Calculating CentroSymmetry Parameter ...")
                    system.cal_centro_symmetry_parameter(N=csp_n_neigh)
                    atoms.add_scalar_quantity(
                        "csp", system.data["csp"].view(), enabled=True, cmap="jet"
                    )
                    ps.info("Calculate CentroSymmetry Parameter successfully.")
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))
        psim.TreePop()


def common_neighbor_parameter():
    global system, atoms, cnp_rc
    if psim.TreeNode("Common Neighbor Parameter"):
        psim.TextUnformatted("Give a cutoff distance")

        _, cnp_rc = psim.InputFloat("cutoff distance", cnp_rc)

        if psim.Button("Compute"):
            try:
                if isinstance(system, System):
                    ps.info("Calculating Commen Neighbor Parameter...")
                    system.cal_common_neighbor_parameter(rc=cnp_rc, max_neigh=None)
                    atoms.add_scalar_quantity(
                        "cnp", system.data["cnp"].view(), cmap="jet", enabled=True
                    )
                    ps.info("Calculate Commen Neighbor Parameter successfully.")
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))

        psim.TreePop()


def cluster_analysis():
    global system, atoms, cluster_rc, cluster_num
    if psim.TreeNode("Cluster Analysis"):
        _, cluster_rc = psim.InputFloat("cutoff distance", cluster_rc)

        if psim.Button("Compute"):
            try:
                if isinstance(system, System):
                    ps.info("Calculating Cluster Analysis...")
                    cluster_num = system.cal_cluster_analysis(
                        rc=cluster_rc, max_neigh=None
                    )
                    atoms.add_scalar_quantity(
                        "cluster_id",
                        system.data["cluster_id"].view(),
                        cmap="jet",
                        enabled=True,
                    )
                    ps.info("Calculating Cluster Analysis successfully.")
                else:
                    ps.warning("System not found")
            except Exception as e:
                ps.error(str(e))

        if cluster_num > 0:
            psim.TextUnformatted("The number of cluster: {}".format(cluster_num))


def common_neighbor_analysis():
    global cna_rc, system, atoms, Other, FCC, HCP, BCC, ICO
    if psim.TreeNode("Commen Neighbor Analysis"):
        _, Other = psim.ColorEdit3("Other", Other)
        psim.SameLine()
        _, FCC = psim.ColorEdit3("FCC", FCC)
        psim.Separator()
        _, HCP = psim.ColorEdit3("HCP", HCP)
        psim.SameLine()
        _, BCC = psim.ColorEdit3("BCC", BCC)
        psim.Separator()
        _, ICO = psim.ColorEdit3("ICO", ICO)

        psim.TextUnformatted("Give a cutoff distance")

        _, cna_rc = psim.InputFloat("cutoff_cna", cna_rc)

        if psim.Button("Compute"):
            try:
                if isinstance(system, System):
                    ps.info("Calculating Commen Neighbor Analysis...")
                    color = np.array([Other, FCC, HCP, BCC, ICO])

                    system.cal_common_neighbor_analysis(cna_rc, max_neigh=None)
                    rgb = (
                        system.data.select(
                            rgb=pl.col("cna").replace(
                                {i: j for i, j in enumerate(color)}, default=None
                            )
                        )["rgb"]
                        .list.to_array(3)
                        .to_numpy()
                    )
                    atoms.add_color_quantity("cna_struc", rgb, enabled=True)
                    atoms.add_scalar_quantity(
                        "cna", system.data["cna"].view(), cmap="jet"
                    )
                    ps.info("Calculate Commen Neighbor Analysis successfully.")
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))

        psim.TreePop()


def warren_cowley_parameter():
    global system, wcp_rc, wcp_element_list, wcp_res

    if psim.TreeNode("Warren Cowley Parameter"):
        _, wcp_rc = psim.InputFloat("cutoff distance", wcp_rc)
        if psim.Button("Compute"):
            try:
                if isinstance(system, System):
                    ps.info("Calculating Warren Cowley Parameter...")
                    system.cal_warren_cowley_parameter(rc=wcp_rc, max_neigh=None)
                    wcp_res = str(np.round(system.WarrenCowleyParameter.WCP, 2))
                    ps.info("Calculate Warren Cowley Parameter successfully.")
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))
        psim.TextUnformatted("The WCP is:")
        if len(wcp_res) > 0:
            psim.InputTextMultiline("", wcp_res, (240, 100))
        _, wcp_element_list = psim.InputText("elemental list", wcp_element_list)
        psim.SameLine()
        if psim.Button("Plot"):
            if isinstance(system, System):
                if hasattr(system, "WarrenCowleyParameter"):
                    element_list = [i.strip() for i in wcp_element_list.split(",")]
                    if len(element_list) == system.WarrenCowleyParameter.WCP.shape[0]:
                        system.WarrenCowleyParameter.plot(element_list)
                    else:
                        system.WarrenCowleyParameter.plot()
                else:
                    ps.warning("You should press Compute Button first.")
            else:
                ps.warning("System not found.")
        psim.TreePop()


def radiul_distribution_function():
    global system, rdf_rc, rdf_nbin, rdf_plot, rdf_plot_partial
    if psim.TreeNode("Radiul Distribution Function"):
        _, rdf_rc = psim.InputFloat("cutoff distance", rdf_rc)
        _, rdf_nbin = psim.InputInt("number of binds", rdf_nbin)
        _, rdf_plot = psim.Checkbox("plot the total rdf", rdf_plot)
        _, rdf_plot_partial = psim.Checkbox("plot the partial rdf", rdf_plot_partial)
        if psim.Button("Compute"):
            try:
                if isinstance(system, System):
                    ps.info("Calculating Radiul Distribution Function...")
                    system.cal_pair_distribution(
                        rc=rdf_rc, nbin=rdf_nbin, max_neigh=None
                    )
                    ps.info("Calculate Radiul Distribution Function successfully.")
                    if rdf_plot:
                        system.PairDistribution.plot()
                    if rdf_plot_partial:
                        system.PairDistribution.plot_partial()
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))

        if psim.Button("Save"):
            if isinstance(system, System):
                if hasattr(system, "PairDistribution"):
                    Tk().withdraw()
                    outputname = asksaveasfilename()
                    try:
                        if len(outputname) > 0:
                            ps.info(f"Saving rdf results to {outputname} ...")
                            np.savetxt(
                                outputname,
                                np.c_[
                                    system.PairDistribution.r,
                                    system.PairDistribution.g_total,
                                ],
                                header="r g(r)",
                                delimiter=" ",
                            )
                            ps.info(f"Save rdf results to {outputname} successfully.")
                    except Exception:
                        ps.error(f"Save {outputname} fail. Change a right outputname.")
                else:
                    ps.warning("One should press compute Button first.")
            else:
                ps.warning("System not found.")
        psim.SameLine()
        if psim.Button("Save Partial"):
            if isinstance(system, System):
                if hasattr(system, "PairDistribution"):
                    Tk().withdraw()
                    outputname = asksaveasfilename()
                    try:
                        if len(outputname) > 0:
                            ps.info(f"Saving rdf_partial results to {outputname} ...")
                            with open(outputname, "w") as op:
                                op.write("r: ")
                                for i in system.PairDistribution.r:
                                    op.write(f"{i} ")
                                op.write("\n")
                                g = system.PairDistribution.g
                                for i in range(g.shape[0]):
                                    for j in range(i, g.shape[1]):
                                        op.write(f"{i+1}-{j+1}: ")
                                        for k in range(g.shape[2]):
                                            op.write(f"{g[i, j, k]} ")
                                        op.write("\n")
                            ps.info(
                                f"Save rdf_partial results to {outputname} successfully."
                            )
                    except Exception:
                        ps.error(f"Save {outputname} fail. Change a right outputname.")
                else:
                    ps.warning("One should press compute Button first.")
            else:
                ps.warning("System not found.")

        psim.TreePop()


def atomic_entropy():
    global system, atoms, entropy_rc, entropy_sigma, entropy_use_local_density, entropy_compute_average, entropy_average_rc
    if psim.TreeNode("Atomic Entropy"):
        _, entropy_rc = psim.InputFloat("cutoff distance", entropy_rc)
        _, entropy_sigma = psim.InputFloat("sigma", entropy_sigma)
        _, entropy_use_local_density = psim.Checkbox(
            "use local density", entropy_use_local_density
        )
        _, entropy_compute_average = psim.Checkbox(
            "compute_average", entropy_compute_average
        )

        if entropy_compute_average:
            psim.TextUnformatted(
                "Average distance should be smaller than cutoff distance."
            )
            _, entropy_average_rc = psim.InputFloat(
                "average distance", entropy_average_rc
            )

        if psim.Button("Compute"):
            try:
                if isinstance(system, System):
                    ps.info("Calculating Atomic Entropy...")
                    system.cal_atomic_entropy(
                        rc=entropy_rc,
                        sigma=entropy_sigma,
                        use_local_density=entropy_use_local_density,
                        compute_average=entropy_compute_average,
                        average_rc=entropy_average_rc,
                        max_neigh=None,
                    )
                    atoms.add_scalar_quantity(
                        "atomic_entropy",
                        system.data["atomic_entropy"].view(),
                        cmap="jet",
                        enabled=True,
                    )
                    if entropy_compute_average:
                        atoms.add_scalar_quantity(
                            "ave_atomic_entropy",
                            system.data["ave_atomic_entropy"].view(),
                            cmap="jet",
                            enabled=True,
                        )
                        ps.info("Calculate Atomic Entropy successfully.")
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))
        psim.TreePop()


def ackland_jones_analysis():
    global system, atoms, Other, FCC, HCP, BCC, ICO
    if psim.TreeNode("Ackland Jones Analysis"):
        _, Other = psim.ColorEdit3("Other", Other)
        psim.SameLine()
        _, FCC = psim.ColorEdit3("FCC", FCC)
        psim.Separator()
        _, HCP = psim.ColorEdit3("HCP", HCP)
        psim.SameLine()
        _, BCC = psim.ColorEdit3("BCC", BCC)
        psim.Separator()
        _, ICO = psim.ColorEdit3("ICO", ICO)

        if psim.Button("Compute"):
            try:
                if isinstance(system, System):
                    ps.info("Calculating Ackland Jones Analysis...")
                    color = np.array([Other, FCC, HCP, BCC, ICO])
                    system.cal_ackland_jones_analysis()
                    rgb = (
                        system.data.select(
                            rgb=pl.col("aja").replace(
                                {i: j for i, j in enumerate(color)}, default=None
                            )
                        )["rgb"]
                        .list.to_array(3)
                        .to_numpy()
                    )
                    atoms.add_color_quantity("aja_struc", rgb, enabled=True)
                    atoms.add_scalar_quantity(
                        "aja", system.data["aja"].view(), cmap="jet"
                    )
                    ps.info("Calculate Ackland Jones Analysis successfully.")
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))
        psim.TreePop()


def steinhardt_bond_orientation():
    global system, atoms, sbo_neigh_number, sbo_cutoff, sbo_mode
    global sbo_mode_default, sbo_qlist, sbo_wlflag, sbo_wlhatflag, sbo_solidliquid
    global solid, liquid, sbo_solidliquid_threshold, sbo_solidliquid_nbond
    if psim.TreeNode("Steinhardt Bond Orientation"):
        psim.PushItemWidth(100)
        changed = psim.BeginCombo("neighbor mode", sbo_mode_default)
        if changed:
            for val in sbo_mode:
                _, selected = psim.Selectable(val, sbo_mode_default == val)
                if selected:
                    sbo_mode_default = val
            psim.EndCombo()
        if sbo_mode_default == "nearest":
            _, sbo_neigh_number = psim.InputInt("neighbor number", sbo_neigh_number)
        else:
            _, sbo_cutoff = psim.InputFloat("cutoff distance", sbo_cutoff)
        _, sbo_qlist = psim.InputText("qlist", sbo_qlist)
        _, sbo_wlflag = psim.Checkbox("wlflag", sbo_wlflag)
        psim.SameLine()
        _, sbo_wlhatflag = psim.Checkbox("wlhatflag", sbo_wlhatflag)
        psim.Separator()
        _, sbo_solidliquid = psim.Checkbox("solidliquid", sbo_solidliquid)
        if sbo_solidliquid:
            _, sbo_solidliquid_threshold = psim.InputFloat(
                "solid threshold", sbo_solidliquid_threshold
            )
            _, sbo_solidliquid_nbond = psim.InputInt(
                "bond number threshold", sbo_solidliquid_nbond
            )
            psim.PushItemWidth(150)
            _, solid = psim.ColorEdit3("Solid", solid)
            _, liquid = psim.ColorEdit3("Liquid", liquid)
            psim.PopItemWidth()

        psim.PopItemWidth()
        if psim.Button("Compute"):
            try:
                if isinstance(system, System):
                    ps.info("Calculating Steinhardt Bond Orientation...")
                    qlist = [int(i.strip()) for i in sbo_qlist.split(",")]
                    if sbo_mode_default == "nearest":
                        nnn = sbo_neigh_number
                        rc = 0.0
                    else:
                        nnn = 0
                        rc = sbo_cutoff
                    if sbo_solidliquid:
                        threshold = 0.7
                        n_bond = 7
                    else:
                        threshold = sbo_solidliquid_threshold
                        n_bond = sbo_solidliquid_nbond
                    system.cal_steinhardt_bond_orientation(
                        nnn=nnn,
                        qlist=qlist,
                        rc=rc,
                        wlflag=sbo_wlflag,
                        wlhatflag=sbo_wlhatflag,
                        solidliquid=sbo_solidliquid,
                        max_neigh=None,
                        threshold=threshold,
                        n_bond=n_bond,
                    )

                    for i in qlist:
                        if i == qlist[0]:
                            atoms.add_scalar_quantity(
                                f"ql{i}",
                                system.data[f"ql{i}"].view(),
                                cmap="jet",
                                enabled=True,
                            )
                        else:
                            atoms.add_scalar_quantity(
                                f"ql{i}", system.data[f"ql{i}"].view(), cmap="jet"
                            )
                    if sbo_wlflag:
                        for i in qlist:
                            atoms.add_scalar_quantity(
                                f"wl{i}", system.data[f"wl{i}"].view(), cmap="jet"
                            )
                    if sbo_wlhatflag:
                        for i in qlist:
                            atoms.add_scalar_quantity(
                                f"whl{i}", system.data[f"whl{i}"].view(), cmap="jet"
                            )
                    if sbo_solidliquid:
                        rgb = (
                            system.data.select(
                                rgb=pl.col("solidliquid").replace(
                                    {0: liquid, 1: solid}, default=None
                                )
                            )["rgb"]
                            .list.to_array(3)
                            .to_numpy()
                        )
                        atoms.add_color_quantity("solidliquid_color", rgb, enabled=True)
                        atoms.add_scalar_quantity(
                            f"solidliquid",
                            system.data["solidliquid"].view(),
                            cmap="jet",
                        )
                    ps.info("Calculate Steinhardt Bond Orientation successfully.")
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))


def polyhedral_template_matching():
    global system, atoms, Other, FCC, HCP, BCC, ICO, Simple_Cubic, Cubic_Diamond, Hexagonal_Diamond, Graphene
    global ptm_other, ptm_fcc, ptm_hcp, ptm_bcc, ptm_ico, ptm_scubic, ptm_dcubic, ptm_dhex, ptm_gra
    global ptm_return_rmsd, ptm_return_interatomic_distance, ptm_return_wxyz, ptm_rmsd_threshold

    if psim.TreeNode("Polyhedral Template Matching"):
        psim.TextUnformatted("Select which structure you want to identify:")

        _, Other = psim.ColorEdit3("Other", Other)
        psim.SameLine()
        _, ptm_other = psim.Checkbox(" " * 1, ptm_other)

        _, FCC = psim.ColorEdit3("FCC", FCC)
        psim.SameLine()
        _, ptm_fcc = psim.Checkbox(" " * 2, ptm_fcc)

        _, HCP = psim.ColorEdit3("HCP", HCP)
        psim.SameLine()
        _, ptm_hcp = psim.Checkbox(" " * 3, ptm_hcp)

        _, BCC = psim.ColorEdit3("BCC", BCC)
        psim.SameLine()
        _, ptm_bcc = psim.Checkbox(" " * 4, ptm_bcc)

        _, ICO = psim.ColorEdit3("ICO", ICO)
        psim.SameLine()
        _, ptm_ico = psim.Checkbox(" " * 5, ptm_ico)

        _, Simple_Cubic = psim.ColorEdit3("Simple_Cubic", Simple_Cubic)
        psim.SameLine()
        _, ptm_scubic = psim.Checkbox(" " * 6, ptm_scubic)

        _, Cubic_Diamond = psim.ColorEdit3("Cubic_Diamond", Cubic_Diamond)
        psim.SameLine()
        _, ptm_dcubic = psim.Checkbox(" " * 7, ptm_dcubic)

        _, Hexagonal_Diamond = psim.ColorEdit3("Hexagonal_Diamond", Hexagonal_Diamond)
        psim.SameLine()
        _, ptm_dhex = psim.Checkbox(" " * 8, ptm_dhex)

        _, Graphene = psim.ColorEdit3("Graphene", Graphene)
        psim.SameLine()
        _, ptm_gra = psim.Checkbox(" " * 9, ptm_gra)

        _, ptm_rmsd_threshold = psim.InputFloat("rmsd threshold", ptm_rmsd_threshold)
        _, ptm_return_rmsd = psim.Checkbox("return rmsd", ptm_return_rmsd)
        _, ptm_return_interatomic_distance = psim.Checkbox(
            "return interatomic distance", ptm_return_interatomic_distance
        )
        _, ptm_return_wxyz = psim.Checkbox("return wxyz", ptm_return_wxyz)

        if psim.Button("Compute"):
            try:
                if isinstance(system, System):
                    structure = []
                    for check, stype in zip(
                        [
                            ptm_fcc,
                            ptm_hcp,
                            ptm_bcc,
                            ptm_ico,
                            ptm_scubic,
                            ptm_dcubic,
                            ptm_dhex,
                            ptm_gra,
                        ],
                        ["fcc", "hcp", "bcc", "ico", "sc", "dcub", "dhex", "graphene"],
                    ):
                        if check:
                            structure.append(stype)
                    structure = "-".join(structure)
                    color = np.array(
                        [
                            Other,
                            FCC,
                            HCP,
                            BCC,
                            ICO,
                            Simple_Cubic,
                            Cubic_Diamond,
                            Hexagonal_Diamond,
                            Graphene,
                        ]
                    )
                    if len(structure) > 0:
                        ps.info("Calculating Polyhedral Template Matching ...")
                        system.cal_polyhedral_template_matching(
                            structure=structure,
                            rmsd_threshold=ptm_rmsd_threshold,
                            return_rmsd=ptm_return_rmsd,
                            return_atomic_distance=ptm_return_interatomic_distance,
                            return_wxyz=ptm_return_wxyz,
                        )

                        rgb = (
                            system.data.select(
                                rgb=pl.col("structure_types").replace(
                                    {i: j for i, j in enumerate(color)}, default=None
                                )
                            )["rgb"]
                            .list.to_array(3)
                            .to_numpy()
                        )
                        atoms.add_color_quantity("ptm_struc", rgb, enabled=True)
                        atoms.add_scalar_quantity(
                            "ptm", system.data["structure_types"].view(), cmap="jet"
                        )
                        if ptm_return_rmsd:
                            atoms.add_scalar_quantity(
                                "rmsd", system.data["rmsd"].view(), cmap="jet"
                            )
                        if ptm_return_interatomic_distance:
                            atoms.add_scalar_quantity(
                                "interatomic_distance",
                                system.data["interatomic_distance"].view(),
                                cmap="jet",
                            )
                        if ptm_return_wxyz:
                            for name in ["qw", "qx", "qy", "qz"]:
                                atoms.add_scalar_quantity(
                                    name, system.data[name].view(), cmap="jet"
                                )
                        ps.info("Calculate Polyhedral Template Matching successfully.")
                    else:
                        ps.error("At least select one structure.")
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))

        psim.TreePop()


def voronoi_analysis():
    global system, atoms
    if psim.TreeNode("Voronoi Analysis"):
        psim.TextUnformatted("Compute the voronoi volume, number and cavity radius.")

        if psim.Button("Compute"):
            try:
                if isinstance(system, System):
                    ps.info("Calculating Voronoi Analysis...")
                    system.cal_voronoi_volume()

                    atoms.add_scalar_quantity(
                        "voronoi_volume",
                        system.data["voronoi_volume"].view(),
                        enabled=True,
                        cmap="jet",
                    )
                    atoms.add_scalar_quantity(
                        "voronoi_number",
                        system.data["voronoi_number"].view(),
                        cmap="jet",
                    )
                    atoms.add_scalar_quantity(
                        "cavity_radius", system.data["cavity_radius"].view(), cmap="jet"
                    )
                    ps.info("Calculate Voronoi Analysis successfully.")
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))
        psim.TreePop()


def identify_species():
    global system, atoms, species_element_list, identify_mode_default, identify_mode, search_species, check_most, species

    if psim.TreeNode("Identify Species"):
        _, species_element_list = psim.InputText("elemental name", species_element_list)

        psim.PushItemWidth(100)
        changed = psim.BeginCombo("identify mode", identify_mode_default)
        if changed:
            for val in identify_mode:
                _, selected = psim.Selectable(val, identify_mode_default == val)
                if selected:
                    identify_mode_default = val
            psim.EndCombo()
        psim.PopItemWidth()

        if identify_mode_default == "search species":
            _, search_species = psim.InputText("target species", search_species)
        else:
            _, check_most = psim.InputInt("check most", check_most)

        if psim.Button("Compute"):
            try:
                if isinstance(system, System):
                    ps.info("Identifing species...")
                    element_list = [i.strip() for i in species_element_list.split(",")]
                    if identify_mode_default == "search species":
                        search_species_list = [
                            i.strip() for i in search_species.split(",")
                        ]
                        assert (
                            len(search_species_list) > 0
                        ), "At least give one species."
                        species = system.cal_species_number(
                            element_list, search_species=search_species_list
                        )
                    else:
                        species = system.cal_species_number(
                            element_list, check_most=check_most
                        )
                    atoms.add_scalar_quantity(
                        "cluster_id",
                        system.data["cluster_id"].view(),
                        cmap="jet",
                        enabled=True,
                    )
                    ps.info("Identify species successfully.")
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))

        if species is not None:
            res = ""
            num = 0
            for key in species.keys():
                res += f"{key}:{species[key]}\n"
                num += 1

            psim.InputTextMultiline("", res, (150, num * 20))

        psim.TreePop()


def replicate():
    global system, atoms, rx, ry, rz
    if psim.TreeNode("Replicate"):
        psim.TextUnformatted("Replicate system along x, y, z directions.")
        _, rx = psim.InputInt("x", rx)
        _, ry = psim.InputInt("y", ry)
        _, rz = psim.InputInt("z", rz)
        if psim.Button("Compute"):
            try:
                if isinstance(system, System):
                    ps.info("Replicating system...")
                    system.replicate(rx, ry, rz)
                    ps.remove_point_cloud("Atoms")
                    ps.remove_curve_network("Box")
                    # ps.remove_all_structures()
                    atoms = ps.register_point_cloud("Atoms", system.pos)
                    atoms.set_radius(1.2, relative=False)
                    for name in system.data.columns:
                        if system.data[name].dtype in pl.NUMERIC_DTYPES:
                            atoms.add_scalar_quantity(
                                name, system.data[name].view(), cmap="jet"
                            )
                    vertices, indices = box2lines(system.box)
                    ps.register_curve_network(
                        "Box", vertices, indices, color=(0, 0, 0), radius=0.0015
                    )
                    # vertices, indices = box2axis(system.box)
                    # ps.register_curve_network(
                    #     "Axis",
                    #     vertices,
                    #     indices,
                    #     color=(0.5, 0.5, 0.5),
                    #     radius=0.0015,
                    #     enabled=False,
                    # )
                    ps.info("Replicate system successfully.")
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))
        psim.TreePop()


def void_analysis():
    global system, cell_length, out_void, void_number, void_volume

    if psim.TreeNode("Void Analysis"):
        _, cell_length = psim.InputFloat("cell length", cell_length)
        _, out_void = psim.Checkbox("save void distribution", out_void)

        if psim.Button("Compute"):
            try:
                if isinstance(system, System):
                    ps.info("Calculating Void Analysis...")
                    if out_void:
                        Tk().withdraw()
                        outputname = asksaveasfilename()
                        if len(outputname) > 0:
                            void_number, void_volume = system.cal_void_distribution(
                                cell_length, out_void=out_void, out_name=outputname
                            )
                    else:
                        void_number, void_volume = system.cal_void_distribution(
                            cell_length
                        )
                    ps.info("Calculate Void Analysis successfully.")
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))

        if void_number is not None and void_volume is not None:
            psim.TextUnformatted(
                f"The void number is: {void_number}.\nThe void volume is: {void_volume} A^3."
            )

        psim.TreePop()


def identify_SFs_TBs():
    global system, atoms, SF_rmsd_threshold, Other, FCC, HCP, BCC, ISF, ESF, TB

    if psim.TreeNode("Identify SFs and TBs"):
        _, Other = psim.ColorEdit3("Other", Other)
        psim.SameLine()
        _, FCC = psim.ColorEdit3("FCC", FCC)
        _, HCP = psim.ColorEdit3("HCP", HCP)
        psim.SameLine()
        _, BCC = psim.ColorEdit3("BCC", BCC)
        _, ISF = psim.ColorEdit3("ISF", ISF)
        psim.SameLine()
        _, ESF = psim.ColorEdit3("ESF", ESF)
        _, TB = psim.ColorEdit3("TB", TB)

        _, SF_rmsd_threshold = psim.InputFloat("rmsd threshold", SF_rmsd_threshold)
        if psim.Button("Compute"):
            if isinstance(system, System):
                try:
                    ps.info("Calculating Identify SFs and TBs...")
                    system.cal_identify_SFs_TBs(rmsd_threshold=SF_rmsd_threshold)

                    rgb = (
                        system.data.with_columns(
                            pl.when(pl.col("structure_types") == 1)
                            .then(FCC)
                            .when(pl.col("structure_types") == 3)
                            .then(BCC)
                            .when(pl.col("fault_types") == 4)
                            .then(HCP)
                            .when(pl.col("fault_types") == 2)
                            .then(ISF)
                            .when(pl.col("fault_types") == 5)
                            .then(ESF)
                            .when(pl.col("fault_types") == 3)
                            .then(TB)
                            .otherwise(Other)
                            .alias("rgb")
                        )["rgb"]
                        .list.to_array(3)
                        .to_numpy()
                    )

                    atoms.add_color_quantity("SFTB_struc", rgb, enabled=True)
                    atoms.add_scalar_quantity(
                        "ptm_struc", system.data["structure_types"].view(), cmap="jet"
                    )
                    atoms.add_scalar_quantity(
                        "fault_types", system.data["fault_types"].view(), cmap="jet"
                    )
                    ps.info("Calculate Identify SFs and TBs successfully.")

                except Exception as e:
                    ps.error(str(e))

            else:
                ps.warning("System not found.")

        psim.TreePop()


def atomic_temperature():
    global system, atoms, temp_element, temp_rc, temp_unit_default, temp_unit
    if psim.TreeNode("Atomic Temperature"):
        psim.TextUnformatted(
            "Calculate atomic temperature subtracting \naverage velocity of center of mass."
        )
        _, temp_element = psim.InputText("elemental list", temp_element)
        _, temp_rc = psim.InputFloat("cutoff distance", temp_rc)
        psim.PushItemWidth(100)
        changed = psim.BeginCombo("units", temp_unit_default)
        if changed:
            for val in temp_unit:
                _, selected = psim.Selectable(val, temp_unit_default == val)
                if selected:
                    temp_unit_default = val
            psim.EndCombo()
        psim.PopItemWidth()
        if psim.Button("compute"):
            try:
                if isinstance(system, System):
                    ps.info("Calculating Atomic Temperature...")
                    elemental_list = [i.strip() for i in temp_element.split(",")]
                    system.cal_atomic_temperature(
                        elemental_list=elemental_list,
                        rc=temp_rc,
                        units=temp_unit_default,
                    )
                    atoms.add_scalar_quantity(
                        "atomic_temp",
                        system.data["atomic_temp"].view(),
                        cmap="jet",
                        enabled=True,
                    )
                    ps.info("Calculate Atomic Temperature successfully.")
                else:
                    ps.warning("System not found.")
            except Exception as e:
                ps.error(str(e))
        psim.TreePop()


def spatial_binning():
    global spatial_directions, spatial_direction_default, input_values, input_value_default, wbin, operations, operation_default, binning_plot

    if psim.TreeNode("Spatial Binning"):
        try:
            psim.TextUnformatted("Spatial binning feature.")
            if isinstance(system, System):
                input_values = system.data.columns
            psim.PushItemWidth(80)
            changed1 = psim.BeginCombo("direction", spatial_direction_default)
            if changed1:
                for val in spatial_directions:
                    _, selected = psim.Selectable(val, spatial_direction_default == val)
                    if selected:
                        spatial_direction_default = val
                psim.EndCombo()
            psim.SameLine()
            changed2 = psim.BeginCombo("input value", input_value_default)
            if changed2:
                for val in input_values:
                    _, selected = psim.Selectable(val, input_value_default == val)
                    if selected:
                        input_value_default = val
                psim.EndCombo()
            psim.Separator()
            changed3 = psim.BeginCombo("operation", operation_default)
            if changed3:
                for val in operations:
                    _, selected = psim.Selectable(val, operation_default == val)
                    if selected:
                        operation_default = val
                psim.EndCombo()
            psim.PopItemWidth()

            _, wbin = psim.InputFloat("width of bin", wbin)
            _, binning_plot = psim.Checkbox("plot results", binning_plot)

            if psim.Button("compute"):
                if isinstance(system, System):
                    ps.info(f"Binning system along {spatial_direction_default}...")
                    system.spatial_binning(
                        spatial_direction_default,
                        input_value_default,
                        wbin,
                        operation_default,
                    )
                    if binning_plot:
                        system.Binning.plot(input_value_default)
                    ps.info(
                        f"Binning system along {spatial_direction_default} successfully."
                    )
                else:
                    ps.warning("System not found.")

            psim.SameLine()
            if psim.Button("save results"):
                if isinstance(system, System):
                    if hasattr(system, "Binning"):
                        Tk().withdraw()
                        outputname = asksaveasfilename()
                        try:
                            if len(outputname) > 0:
                                ps.info(f"Saving binning results to {outputname} ...")
                                if spatial_direction_default in ["x", "y", "z"]:
                                    np.savetxt(
                                        rf"{outputname}",
                                        np.c_[
                                            system.Binning.coor[
                                                spatial_direction_default
                                            ],
                                            system.Binning.res[:, 1],
                                        ],
                                        header=f"{spatial_direction_default} {input_value_default}-{operation_default}",
                                        delimiter=" ",
                                    )
                                elif spatial_direction_default in ["xy", "xz", "yz"]:
                                    with open(rf"{outputname}", "w") as op:
                                        for i in spatial_direction_default:
                                            op.write(i)
                                            op.write(": ")
                                            for j in system.Binning.coor[
                                                spatial_direction_default[0]
                                            ]:
                                                op.write(f"{j} ")
                                            op.write("\n")
                                        op.write("data:\n")
                                        data = system.Binning.res[:, :, 1]
                                        for i in range(data.shape[0]):
                                            for j in range(data.shape[1]):
                                                op.write(f"{data[i,j]} ")
                                            op.write("\n")
                                else:
                                    with open(rf"{outputname}", "w") as op:
                                        for i in spatial_direction_default:
                                            op.write(i)
                                            op.write(": ")
                                            for j in system.Binning.coor[
                                                spatial_direction_default[0]
                                            ]:
                                                op.write(f"{j} ")
                                            op.write("\n")
                                        op.write("data:\n")
                                        data = system.Binning.res[:, :, :, 1]
                                        for i in range(data.shape[0]):
                                            op.write(f"x rows {i}\n")
                                            for j in range(data.shape[1]):
                                                for k in range(data.shape[2]):
                                                    op.write(f"{data[i, j, k]} ")
                                                op.write("\n")

                                ps.info(f"Save {outputname} successfully.")
                        except Exception:
                            ps.error(
                                f"Save {outputname} fail. Change a right outputname."
                            )
                    else:
                        ps.warning("One should press compute Button first!")
                else:
                    ps.warning("System not found.")

        except Exception as e:
            ps.error(str(e))

        psim.TreePop()


def build_polycrystal():
    global box_length, grain_number, metal_lattice_type_default, metal_lattice_type_list, metal_lattice_constant, random_seed
    global metal_overlap_dis, add_graphene, gra_lattice_constant, metal_gra_overlap_dis, gra_overlap_dis

    if psim.TreeNode("Build Polycrystal"):
        psim.TextUnformatted("Given the box length along x, y, z directions.")
        _, box_length = psim.InputText("Box Length", box_length)
        _, grain_number = psim.InputInt("Grain Number", grain_number)
        psim.PushItemWidth(100)
        changed = psim.BeginCombo("Metallic Lattice Type", metal_lattice_type_default)
        if changed:
            for val in metal_lattice_type_list:
                _, selected = psim.Selectable(val, metal_lattice_type_default == val)
                if selected:
                    metal_lattice_type_default = val
            psim.EndCombo()
        psim.PopItemWidth()
        _, metal_lattice_constant = psim.InputFloat(
            "Metallic Lattice Constant", metal_lattice_constant
        )
        _, random_seed = psim.InputInt("Random Seed", random_seed)
        _, metal_overlap_dis = psim.InputFloat(
            "Metallic Overlap Distance", metal_overlap_dis
        )

        _, add_graphene = psim.Checkbox("Add Graphene", add_graphene)

        if add_graphene:
            _, gra_lattice_constant = psim.InputFloat(
                "Graphene Lattice Constant", gra_lattice_constant
            )
            _, metal_gra_overlap_dis = psim.InputFloat(
                "Gra/Metal Overlap Distance", metal_gra_overlap_dis
            )
            _, gra_overlap_dis = psim.InputFloat(
                "Graphene Overlap Distance", gra_overlap_dis
            )

        if psim.Button("Build"):
            try:
                ps.info("Building polycrystal...")
                box = np.array([[0.0, float(i.strip())] for i in box_length.split(",")])
                poly = CreatePolycrystalline(
                    box=box,
                    seednumber=grain_number,
                    metal_latttice_constant=metal_lattice_constant,
                    metal_lattice_type=metal_lattice_type_default,
                    randomseed=random_seed,
                    metal_overlap_dis=metal_overlap_dis,
                    add_graphene=add_graphene,
                    gra_lattice_constant=gra_lattice_constant,
                    metal_gra_overlap_dis=metal_gra_overlap_dis,
                    gra_overlap_dis=gra_overlap_dis,
                )
                poly.compute(save_dump=False)
                system = System(data=poly.data, box=poly.box)
                ps.remove_all_structures()
                atoms = ps.register_point_cloud("Atoms", system.pos)
                atoms.set_radius(1.2, relative=False)
                for name in system.data.columns:
                    if system.data[name].dtype in pl.NUMERIC_DTYPES:
                        if name == "grainid":
                            atoms.add_scalar_quantity(
                                name, system.data[name].view(), cmap="jet", enabled=True
                            )
                        else:
                            atoms.add_scalar_quantity(
                                name, system.data[name].view(), cmap="jet"
                            )
                vertices, indices = box2lines(system.box)
                ps.register_curve_network(
                    "Box", vertices, indices, color=(0, 0, 0), radius=0.0015
                )
                vertices, indices = box2axis(system.box)
                ps.register_curve_network(
                    "Axis",
                    vertices,
                    indices,
                    color=(0.5, 0.5, 0.5),
                    radius=0.0015,
                    enabled=False,
                )
                ps.reset_camera_to_home_view()
                ps.info("Builde polycrystal successfully.")
                globals()["system"] = system
                globals()["atoms"] = atoms
            except Exception as e:
                ps.error(str(e))

        psim.TreePop()


def build_lattice():
    global lattice_type_default, lattice_type_list, lattice_constant, lrx, lry, lrz, orientation_x, orientation_y, orientation_z
    if psim.TreeNode("Build Lattice"):
        psim.PushItemWidth(100)
        changed = psim.BeginCombo("Lattice Type", lattice_type_default)
        if changed:
            for val in lattice_type_list:
                _, selected = psim.Selectable(val, lattice_type_default == val)
                if selected:
                    lattice_type_default = val
            psim.EndCombo()
        psim.PopItemWidth()

        if lattice_type_default in ["FCC", "BCC"]:
            psim.TextUnformatted(
                "You can assign the crystalline orientation,\n which should be mutually orthogonal."
            )
            _, orientation_x = psim.InputText("orientation x", orientation_x)
            _, orientation_y = psim.InputText("orientation y", orientation_y)
            _, orientation_z = psim.InputText("orientation z", orientation_z)

        _, lattice_constant = psim.InputFloat("Lattice Constant", lattice_constant)
        _, lrx = psim.InputInt("x", lrx)
        _, lry = psim.InputInt("y", lry)
        _, lrz = psim.InputInt("z", lrz)

        if psim.Button("Build"):
            try:
                ps.info("Building lattice...")
                orientation = None
                if lattice_type_default in ["FCC", "BCC"]:
                    ox = [int(i.strip()) for i in orientation_x.split(",")]
                    oy = [int(i.strip()) for i in orientation_y.split(",")]
                    oz = [int(i.strip()) for i in orientation_z.split(",")]
                    orientation = np.array([ox, oy, oz])
                lat = LatticeMaker(
                    lattice_constant,
                    lattice_type_default,
                    lrx,
                    lry,
                    lrz,
                    crystalline_orientation=orientation,
                )
                lat.compute()
                system = System(pos=lat.pos, box=lat.box)
                ps.remove_all_structures()
                atoms = ps.register_point_cloud("Atoms", system.pos)
                atoms.set_radius(1.2, relative=False)
                for name in system.data.columns:
                    if system.data[name].dtype in pl.NUMERIC_DTYPES:
                        atoms.add_scalar_quantity(
                            name, system.data[name].view(), cmap="jet"
                        )
                vertices, indices = box2lines(system.box)
                ps.register_curve_network(
                    "Box", vertices, indices, color=(0, 0, 0), radius=0.0015
                )
                vertices, indices = box2axis(system.box)
                ps.register_curve_network(
                    "Axis",
                    vertices,
                    indices,
                    color=(0.5, 0.5, 0.5),
                    radius=0.0015,
                    enabled=False,
                )
                ps.reset_camera_to_home_view()
                ps.info("Build lattice successfully.")
                globals()["system"] = system
                globals()["atoms"] = atoms
            except Exception as e:
                ps.error(str(e))
        psim.TreePop()


def show_system_info():
    global system
    if isinstance(system, System):
        psim.InputTextMultiline(
            "",
            f"Filename: {system.filename}\nAtom Number: {system.N}\nSimulation Box:\n{system.box}\nBoundary: {system.boundary}",
            (500, 150),
        )
    else:
        psim.Text("System not found.")


def callback():
    global filename
    whether_init_file(filename)
    psim.PushItemWidth(150)
    psim.TextUnformatted("File")
    load_file_gui()
    save_file_gui()

    psim.TextUnformatted("Analysis Modifier")
    psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
    if psim.TreeNode("Structure Analysis"):
        ackland_jones_analysis()
        atomic_entropy()
        centro_symmetry_parameter()
        common_neighbor_analysis()
        common_neighbor_parameter()
        identify_SFs_TBs()
        polyhedral_template_matching()
        radiul_distribution_function()
        steinhardt_bond_orientation()
        warren_cowley_parameter()
        psim.TreePop()
    psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
    if psim.TreeNode("Other Analysis"):
        atomic_temperature()
        cluster_analysis()
        identify_species()
        replicate()
        spatial_binning()
        void_analysis()
        voronoi_analysis()
        psim.TreePop()
    psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
    if psim.TreeNode("Build model"):
        build_lattice()
        build_polycrystal()
        psim.TreePop()
    psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
    if psim.TreeNode("System Infomation"):
        show_system_info()
        psim.TreePop()
    psim.PopItemWidth()


def main():
    parser = argparse.ArgumentParser(
        prog="mdapy",
        description="mdapy: A flexible and powerful toolkit to handle the data generated from molecular dynamics simulations.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        required=False,
        help="Show the version of mdapy",
    )
    parser.add_argument(
        "-f", "--filename", default="", type=str, help="filename to load."
    )

    try:
        args = parser.parse_args()
        if args.version:
            print("The version of mdapy is:", __version__)
        else:
            init_global_parameters()
            global filename, init_file
            filename = rf"{args.filename}"
            if len(filename) > 0:
                init_file = True
            ti.init(ti.cpu, default_fp=ti.f64)

            ps.set_program_name("mdapy")
            ps.set_up_dir("z_up")
            ps.set_verbosity(2)
            ps.set_give_focus_on_show(True)
            ps.set_ground_plane_mode("none")
            ps.set_invoke_user_callback_for_nested_show(True)
            ps.set_use_prefs_file(True)
            ps.set_print_prefix("[mdapy] ")
            ps.init()
            ps.set_user_callback(callback)
            ps.show()
            ps.info(f"Exit mdapy GUI mode successfully.")

    except Exception as e:
        print(e)
        parser.print_help()
