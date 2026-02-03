# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from __future__ import annotations
from typing import List, Optional, Tuple, TYPE_CHECKING, Dict, Union
from pathlib import Path
import numpy as np
import polars as pl
from mdapy.calculator import CalculatorMP
from mdapy.build_lattice import build_hea, build_crystal
from mdapy.system import System
import os
import re

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


def run_gpumd(
    system: System,
    dirname: str,
    runin: str,
    nep_file: str,
    gpumd_path: str = "gpumd",
) -> None:
    """Warpper to run GPUMD.

    Parameters
    ----------
    system : System
        Use this to generate model.xyz.
    dirname : str
        Run MD in this dirname. If it has been existing, raise error.
    runin : str
        Use this to generate run.in file. Do not specify potential here, using `nep_file` parameter.
    nep_file : str
        The path of nep.txt. Defaults to nep.txt.
    gpumd_path : str
        The path of gpumd. Defaults to gpumd.
    """

    if os.path.exists(dirname):
        raise FileExistsError(f"{dirname} is existing.")
    if not os.path.exists(nep_file):
        raise FileNotFoundError(f"{nep_file} is not found.")

    os.makedirs(dirname)
    system.write_xyz(f"{dirname}/model.xyz")
    with open(f"{dirname}/run.in", "w") as op:
        op.write(f"potential {os.path.abspath(nep_file)}\n" + runin)

    cwd = os.getcwd()
    os.chdir(dirname)
    try:
        os.system(gpumd_path)
    finally:
        os.chdir(cwd)


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Root-Mean-Square Error (RMSE).

    Parameters
    ----------
    predictions : np.ndarray
        Predicted values.
    targets : np.ndarray
        Target reference values.

    Returns
    -------
    float
        RMSE value.
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def read_thermo(path: str) -> pl.DataFrame:
    """Load GPUMD thermo.out file into a Polars DataFrame.

    The file contains 18 columns:
    T, K, U, Pxx, Pyy, Pzz, Pyz, Pxz, Pxy, ax, ay, az, bx, by, bz, cx, cy, cz.

    Parameters
    ----------
    path : str
        Directory path containing thermo.out.

    Returns
    -------
    pl.DataFrame
        Thermo data with 18 columns.
    """
    return pl.from_numpy(
        np.loadtxt(Path(path, "thermo.out")),
        schema="T K U Pxx Pyy Pzz Pyz Pxz Pxy ax ay az bx by bz cx cy cz".split(),
    )


def plot_nep_train(
    path: str,
    outname: Optional[str] = None,
    figdpi: Optional[int] = 300,
    **kargs,
) -> Tuple[Figure, List[List[Axes]]]:
    """Plot NEP training results, including energy/force/stress scatter plots and loss curves.

    Parameters
    ----------
    path : str
        Path containing NEP output files: loss.out, energy_train.out, force_train.out, stress_train.out.
    outname : Optional[str], optional
        Filename to save figure, by default None.
    figdpi : Optional[int], optional
        DPI of generated figure, by default 300.
    **kargs :
        Extra arguments passed to set_figure().

    Returns
    -------
    Tuple[Figure, List[List[Axes]]]
        The figure and 2×2 axes list.
    """
    from mdapy.plotset import set_figure, save_figure

    fig, axes = set_figure(figsize=(16, 14), figdpi=figdpi, nrow=2, ncol=2, **kargs)
    loss = np.loadtxt(Path(path, "loss.out"))
    e_train = np.loadtxt(Path(path, "energy_train.out"))
    f_train = np.loadtxt(Path(path, "force_train.out"))
    s_train = np.loadtxt(Path(path, "stress_train.out"))

    # --- Energy ---
    x, y = e_train[:, 1], e_train[:, 0]
    axes[0][0].plot(x, y, "o", label=f"RMSE={rmse(x, y) * 1000:.1f} meV")
    axes[0][0].legend()
    axes[0][0].set_xlabel("DFT energy (eV/atom)")
    axes[0][0].set_ylabel("NEP energy (eV/atom)")

    # --- Force ---
    x, y = f_train[:, 3:].flatten(), f_train[:, :3].flatten()
    axes[0][1].plot(
        x, y, "o", label=f"RMSE={rmse(x, y) * 1000:.1f} meV/" + r"$\mathregular{\AA}$"
    )
    axes[0][1].legend(handletextpad=0.01, loc="upper left")
    axes[0][1].set_xlabel(r"DFT force (eV/$\mathregular{\AA}$)")
    axes[0][1].set_ylabel(r"NEP force (eV/$\mathregular{\AA}$)")

    # --- Stress ---
    x, y = s_train[:, 6:].flatten(), s_train[:, :6].flatten()
    axes[1][0].plot(x, y, "o", label=f"RMSE={rmse(x, y):.2f} GPa")
    axes[1][0].legend(handletextpad=0.01, loc="upper left")
    axes[1][0].set_xlabel("DFT stress (GPa)")
    axes[1][0].set_ylabel("NEP stress (GPa)")

    # --- Loss ---
    for i, j in zip([1, 4, 5, 6], "Total E-train F-train V-train".split()):
        axes[1][1].plot(loss[:, 0], loss[:, i], label=j)

    axes[1][1].legend()
    axes[1][1].set_xlabel("Generation")
    axes[1][1].set_ylabel("Loss")
    axes[1][1].set_yscale("log")
    axes[1][1].set_xscale("log")

    # --- diagonal reference lines ---
    for i in [0, 1]:
        for j in [0, 1]:
            if i == 1 and j == 1:
                continue
            xlim = axes[i][j].get_xlim()
            ylim = axes[i][j].get_ylim()
            lo = min(xlim[0], ylim[0])
            hi = max(xlim[1], ylim[1])
            delta = 0.05 * abs(hi - lo)
            lo -= delta
            hi += delta
            lim = [lo, hi]
            axes[i][j].plot(lim, lim, "grey")
            axes[i][j].set_xlim(lim)
            axes[i][j].set_ylim(lim)

    if outname is not None:
        save_figure(fig, outname)

    return fig, axes


def get_sfe_fcc(name: str, a: float, calc: CalculatorMP) -> float:
    """Compute stacking fault energy (SFE) for an FCC crystal.

    Parameters
    ----------
    name : str
        Element name.
    a : float
        Lattice constant.
    calc : CalculatorMP
        MDAPY calculator.

    Returns
    -------
    float
        Stacking fault energy in mJ/m².
    """
    distance = a / 6**0.5
    system = build_crystal(
        name,
        "fcc",
        a,
        nx=3,
        ny=3,
        nz=4,
        miller1=[1, 1, 2],
        miller2=[1, -1, 0],
        miller3=[1, 1, -1],
    )
    calc.results = {}
    system.calc = calc
    system.box.boundary[2] = 0
    e1 = system.get_energy()

    LZ = system.data["z"].max() - system.data["z"].min()
    system.update_data(
        system.data.with_columns(
            pl.when(pl.col("z") > LZ / 2)
            .then(pl.col("x") + distance)
            .otherwise(pl.col("x"))
            .alias("x")
        ),
        reset_calcolator=True,
    )
    system.wrap_pos()
    e2 = system.get_energy()

    factor = system.box.box[0, 0] * system.box.box[1, 1] / 16021.766200000002
    return (e2 - e1) / factor


def get_average_sfe_fcc_hea(
    N: int,
    element_list: List[str],
    element_ratio: List[float],
    a: float,
    calc: CalculatorMP,
) -> np.ndarray:
    """Compute averaged SFE for random FCC HEA configurations.

    Parameters
    ----------
    N : int
        Number of random samples.
    element_list : List[str]
        Element species.
    element_ratio : List[float]
        Element ratios.
    a : float
        Lattice constant.
    calc : CalculatorMP
        MD calculator.

    Returns
    -------
    np.ndarray
        Array of: [i, mean_sfe up to sample i]
    """
    sfe = []
    distance = a / 6**0.5

    for seed in range(1, N + 1):
        system = build_hea(
            element_list,
            element_ratio,
            "fcc",
            a,
            nx=3,
            ny=3,
            nz=4,
            miller1=[1, 1, 2],
            miller2=[1, -1, 0],
            miller3=[1, 1, -1],
            random_seed=seed,
        )
        calc.results = {}
        system.calc = calc
        system.box.boundary[2] = 0
        e1 = system.get_energy()

        LZ = system.data["z"].max() - system.data["z"].min()
        system.update_data(
            system.data.with_columns(
                pl.when(pl.col("z") > LZ / 2)
                .then(pl.col("x") + distance)
                .otherwise(pl.col("x"))
                .alias("x")
            ),
            reset_calcolator=True,
        )
        system.wrap_pos()
        e2 = system.get_energy()

        factor = system.box.box[0, 0] * system.box.box[1, 1] / 16021.766200000002
        sfe.append((e2 - e1) / factor)

    ave_sfe = []
    for i in range(1, len(sfe)):
        ave_sfe.append([i, np.mean(sfe[:i])])
    return np.array(ave_sfe)


def get_eos(
    system: System, scale_start: float, scale_end: float, num: int
) -> np.ndarray:
    """Compute equation of state (EOS) by uniformly scaling volume.

    Parameters
    ----------
    system : System
        Input structure.
    scale_start : float
        Initial scale factor (>0).
    scale_end : float
        Final scale factor (> scale_start).
    num : int
        Number of sampling points.

    Returns
    -------
    np.ndarray
        (num, 2) array of [volume_per_atom, energy_per_atom]
    """
    assert scale_start < scale_end
    assert scale_start > 0

    scale_list = np.linspace(scale_start, scale_end, num)
    eos = []

    for i in scale_list:
        cur = System(
            box=system.box.box * i,
            data=system.data.with_columns(
                pl.col("x") * i, pl.col("y") * i, pl.col("z") * i
            ),
        )
        cur.calc = system.calc
        cur.calc.results = {}
        e = cur.get_energy() / cur.N
        vol = cur.box.volume / cur.N
        eos.append([vol, e])

    return np.array(eos)


class PCA:
    """Simple PCA implementation (similar to sklearn PCA)."""

    def __init__(self, n_components: int):
        """
        Parameters
        ----------
        n_components : int
            Number of principal components to keep.
        """
        self.n_components = n_components
        self.explained_variance: Optional[np.ndarray] = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA and return transformed coordinates.

        Parameters
        ----------
        X : np.ndarray
            Input array shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Projected data with shape (n_samples, n_components).
        """
        X_centered = X - np.mean(X, axis=0)
        cov_matrix = np.cov(X_centered.T)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        components = eigenvectors[:, : self.n_components]
        self.explained_variance = eigenvalues[: self.n_components]

        # same as sklearn: make signs deterministic
        max_abs_idx = np.argmax(np.abs(components), axis=0)
        signs = np.sign(components[max_abs_idx, np.arange(self.n_components)])
        components *= signs

        return np.dot(X_centered, components)


def fps_sample(
    n_sample: int,
    descriptors: np.ndarray,
    start_idx: int = 0,
) -> np.ndarray:
    """This function is used to sample the configurations using farthest point sampling method, based
    on the descriptors. It is helpful to select the structures during active learning process.

    Parameters
    ----------
    n_sample : int
        Number of structures one wants to select.
    descriptors : np.ndarray
        Two dimensional ndarray, it can be any descriptors.
    start_idx : int
        For deterministic results, fix the first sampled point index.
        Defaults to 0.

    Returns
    -------
    sampled_indices : ndarray, shape (n_sample,)
    """

    assert descriptors.ndim == 2, "Only support 2-D ndarray."
    n_points = descriptors.shape[0]
    assert n_sample <= n_points, f"n_sample must <= {n_points}."
    assert n_sample > 0, "n_sample must be a positive number."
    assert start_idx >= 0 and start_idx < n_points, (
        f"start_idx must belong [0, {n_points - 1}]."
    )
    sampled_indices = [start_idx]
    min_distances = np.full(n_points, np.inf)
    farthest_point_idx = start_idx

    for _ in range(n_sample - 1):
        current_point = descriptors[farthest_point_idx]
        dist_to_current_point = np.linalg.norm(descriptors - current_point, axis=1)
        min_distances = np.minimum(min_distances, dist_to_current_point)
        farthest_point_idx = np.argmax(min_distances)
        sampled_indices.append(farthest_point_idx)

    return np.array(sampled_indices, np.int32)


def cfg2xyz(
    file_list: Union[List[str], str],
    type_dict: Dict[str, int],
    output_name: str = "train.xyz",
    f_max: float = 25.0,
) -> None:
    """Convert cfg file for MTP to xyz file for GPUMD, including energy, force and virial.

    Parameters
    ----------
    file_list : List[str] or str
        Single or multi cfg file.
    type_dict : Dict[str, int]
        Map type from number to element, such as {0:'Al', 1:'C'}.
    output_name : str
        Output filename with append mode. Defaults to train.xyz.
    f_max : float
        Force absolute maximum larger than this value will be filtered. Defaults to 25.0 eV/A.
    """
    if isinstance(file_list, str):
        file_list = [file_list]
    with open(output_name, "a") as op:
        for cfg in file_list:
            with open(cfg) as op:
                file = op.read()
            res = file.split("BEGIN_CFG")[1:]
            for f in range(len(res)):
                frame_content = res[f].split("\n")
                N = int(frame_content[2].strip())
                box = []
                for i in frame_content[4:7]:
                    box.extend(i.split())
                type_pos_force = [i.split()[1:] for i in frame_content[8 : 8 + N]]
                _f_max = np.abs(np.array(type_pos_force)[:, -3:].astype(float)).max()
                if _f_max > f_max:
                    continue
                energy = frame_content[8 + N + 1].strip()
                vxx, vyy, vzz, vyz, vxz, vxy = (
                    frame_content[8 + N + 1 + 2].strip().split()
                )
                vyx = vxy
                vzx = vxz
                vzy = vyz

                op.write(f"{N}\n")
                box_str = (
                    "Lattice=" + '"' + "{} {} {} {} {} {} {} {} {}".format(*box) + '"'
                )
                op.write(
                    f'{box_str} energy={energy} virial="{vxx} {vxy} {vxz} {vyx} {vyy} {vyz} {vzx} {vzy} {vzz}" properties=species:S:1:pos:R:3:force:R:3\n'
                )
                for i in range(N):
                    op.write(
                        "{} {} {} {} {} {} {}\n".format(
                            type_dict[int(type_pos_force[i][0])], *type_pos_force[i][1:]
                        )
                    )


def read_OUTCAR(filename: str) -> Union[Dict, bool]:
    """This is only working for single-point calculation OUTCAR.

    Args:
        filename (str): outcar filename.

    Returns:
        Dict[str:Any]: system information, such as 'Natom', 'lattice', 'energy', 'pos_force', 'symbols', 'virial'
    """
    data = {
        "Natom": None,
        "lattice": None,
        "energy": None,
        "pos_force": None,
        "symbols": None,
        "virial": None,
    }
    with open(filename) as f:
        content = f.read()
        if "aborting loop because EDIFF is reached" not in content:
            return False
    lines_content = content.split("\n")

    pattern = (
        r"VOLUME and BASIS-vectors are now.*?\n(.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)"
    )
    match = re.search(pattern, content, re.DOTALL)
    lines = match.group(1).strip().split("\n")
    lattice_lines = lines[-3:]
    lattice = []
    for line in lattice_lines:
        line = re.sub(r"(?<=\d)-", " -", line)
        values = line.split()[:3]
        lattice.extend(values)
    data["lattice"] = " ".join(lattice)

    has_virial = False
    ion_symbols = []
    for i, line in enumerate(lines_content):
        if "number of ions" in line:
            data["Natom"] = int(line.split()[-1])
        if "free  energy   TOTEN" in line:
            data["energy"] = float(line.split()[4])
        if "ISIF" in line:
            has_virial = int(line.split()[2]) != 0
        if "ions per type" in line:
            ion_numbers = [int(j) for j in line.split("=")[1].split()]
        if "POTCAR:" in line:
            symbol = line.split()[2].split("_")[0]
            if symbol not in ion_symbols:
                ion_symbols.append(symbol)
        if "TOTAL-FORCE (eV/Angst)" in line:
            start_idx = i + 2
            end_idx = start_idx + data["Natom"]
            data["pos_force"] = [
                " ".join(lines_content[j].split()) for j in range(start_idx, end_idx)
            ]

    symbols = []
    for i, j in zip(ion_symbols, ion_numbers):
        symbols.extend([i] * j)
    data["symbols"] = symbols
    if has_virial:
        pattern = r"FORCE on cell =-STRESS.*?Total\s+([\d\.\-\s]+)"
        values = list(re.finditer(pattern, content, re.DOTALL))[-1].group(1).split()
        xx, yy, zz, xy, yz, zx = values
        yx, xz, zy = xy, zx, yz
        data["virial"] = f"{xx} {xy} {xz} {yx} {yy} {yz} {zx} {zy} {zz}"
    return data


def outcar2xyz(
    outcar_list: Union[List[str], str], output_path: str = "train.xyz", mode: str = "w", print_no_converge:bool=True
):
    """Convert OUTCAR file for VASP to xyz file for GPUMD, including energy, force and virial.
    It only works for single-point energy calculation generated OUTCAR.

    Parameters
    ----------
    outcar_list : List[str] or str
        Single or multi OUTCAR file.
    output_path : str
        Output filename. Defaults to train.xyz.
    mode : str
        Writting mode, such as 'w' and 'a'. Defaults to 'w'.
    print_no_converge : bool
        Whether print non-converged outcars filename. Defaults to True
    """
    if isinstance(outcar_list, str):
        outcar_list = [outcar_list]

    assert mode in ["w", "a"], "Only support w or a mode."
    not_converged = []
    with open(output_path, mode) as out_f:
        for i, outcar in enumerate(outcar_list, 1):
            data = read_OUTCAR(outcar)
            if not data:
                not_converged.append(outcar)
                continue
            out_f.write(f"{data['Natom']}\n")
            line_properties = "Properties=species:S:1:pos:R:3:forces:R:3"
            energy, lattice = data["energy"], data["lattice"]
            if data["virial"] is not None:
                virial = data["virial"]
                out_f.write(
                    f'energy={energy:.6f} Lattice="{lattice}" virial="{virial}" {line_properties} pbc="T T T"\n'
                )
            else:
                out_f.write(
                    f'energy={energy:.6f} Lattice="{lattice}" {line_properties} pbc="T T T"\n'
                )
            for symbol, pf_line in zip(data["symbols"], data["pos_force"]):
                out_f.write(f"{symbol} {pf_line}\n")
            progress = int(i * 100 / len(outcar_list))
            bar = "#" * (progress // 2) + "." * (50 - progress // 2)
            print(f"\rProgress: [{bar}] {progress}% ({i}/{len(outcar_list)})", end="")
    if len(not_converged):
        print(
            f"\nFound {len(not_converged)} non-converged OUTCARS, we have skiped them:"
        )
        if print_no_converge:
            for i in not_converged:
                print(f"{i} is not converged!")


def outcars2xyz(
    outcar_list: Union[List[str], str], output_path: str = "train.xyz", mode: str = "w", print_no_converge:bool=True
):
    """Convert OUTCAR file for VASP to xyz file for GPUMD, including energy, force and virial.

    Parameters
    ----------
    outcar_list : List[str] or str
        Single or multi OUTCAR file.
    output_path : str
        Output filename. Defaults to train.xyz.
    mode : str
        Writting mode, such as 'w' and 'a'. Defaults to 'w'.
    print_no_converge : bool
        Whether print non-converged outcars filename. Defaults to True
    """
    if isinstance(outcar_list, str):
        outcar_list = [outcar_list]
    not_converged = []
    with open(output_path, mode) as wf:
        for num, file in enumerate(outcar_list):
            text = open(file).read()
            start_lines = [
                m.start()
                for m in re.finditer("aborting loop because EDIFF is reached", text)
            ]
            if not start_lines:
                not_converged.append(file)
                continue

            end_lines = [
                m.start() for m in re.finditer(r"[^ML] energy  without entropy", text)
            ]
            ion_nums = re.findall(r"ions per type\s*=\s*(.*)", text)
            ion_numb_arra = list(map(int, ion_nums[-1].split())) if ion_nums else []
            ion_syms = []
            for m in re.finditer(r"^.*POTCAR:\s+\S+\s+(\S+)", text, flags=re.M):
                s = m.group(1).split("_")[0]
                if s not in ion_syms:
                    ion_syms.append(s)
            m = re.search(r"number of ions\s+NIONS =\s+(\d+)", text)
            syst_numb_atom = int(m.group(1)) if m else 0
            k = 0
            for i in range(len(start_lines)):
                while k < len(end_lines) and start_lines[i] > end_lines[k]:
                    k += 1
                if k >= len(end_lines):
                    break

                block = text[start_lines[i] : end_lines[k]]

                wf.write(f"{syst_numb_atom}\n")

                pattern = r"VOLUME and BASIS-vectors are now.*?\n(.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)"
                match = re.search(pattern, block, re.DOTALL)
                lines = match.group(1).strip().split("\n")
                lattice_lines = lines[-3:]
                lattice = []
                for line in lattice_lines:
                    line = re.sub(r"(?<=\d)-", " -", line)
                    values = line.split()[:3]
                    lattice.extend(values)
                latt = " ".join(lattice)

                m = re.search(r"free  energy   TOTEN\s*=\s*([-.\dEe+]+)", block)
                ener = m.group(1)

                virial = (
                    block[block.index("Total") : block.index("external")]
                    .split("\n")[0]
                    .split()[1:]
                )

                xx, yy, zz, xy, yz, zx = virial
                yx, xz, zy = xy, zx, yz
                virial_str = f"{xx} {xy} {xz} {yx} {yy} {yz} {zx} {zy} {zz}"
                wf.write(
                    f'Lattice="{latt}" Energy={ener} Virial="{virial_str}" pbc="T T T" Properties=species:S:1:pos:R:3:forces:R:3\n'
                )

                syms = []
                for s, n in zip(ion_syms, ion_numb_arra):
                    syms += [s] * n

                forces_block = re.search(r"TOTAL-FORCE \(eV/Angst\)(.*)", block, re.S)
                if forces_block:
                    lines = (
                        forces_block.group(1)
                        .strip()
                        .splitlines()[1 : syst_numb_atom + 1]
                    )
                    for s, l in zip(syms, lines):
                        wf.write(f"{s} {' '.join(l.split())}\n")

            progress = int((num + 1) * 100 / len(outcar_list))
            bar = "#" * (progress // 2) + "." * (50 - progress // 2)
            print(f"\rProgress: [{bar}] {progress}% ({num + 1}/{len(outcar_list)})", end="")
    if len(not_converged):
        print(
            f"\nFound {len(not_converged)} non-converged OUTCARS, we have skiped them:"
        )
        if print_no_converge:
            for i in not_converged:
                print(f"{i} is not converged!")


if __name__ == "__main__":
    pass
