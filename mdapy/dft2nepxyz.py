# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.
import numpy as np
import re
from time import time
from collections import Counter
import os
from tqdm import tqdm


class LabeledSystem:
    """This class is used to read first principle calculation data, obataining the energy, force, box and virial information.
    Those information can be saved as initial database for deep learning training, aiming to develop high accurancy potential
    function.

    The units are listed as below:

    - energy : eV (per-cell)
    - force : eV/Å (per-atom)
    - virial : eV (per-cell)
    - stress : GPa (per-cell)
    - pos : Å (per-atom)
    - box : Å
    Now we only support SCF calculation in CP2K. In the future the AIMD and VASP may also be implemented.

    Args:
        filename (str): filename of CP2K SCF output file.
        fmt (str, optional): DFT calculation code. Defaults to "CP2K-SCF".

    Outputs:
        - **content** (str) - the whole content of input file.
        - **structure** (dict) - a dict contains energy, pos, box, force, type_list and virial (if computed).

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> LS = mp.LabeledSystem('output.log')

        >>> LS.data['energy'] # check energy per cell.
    """

    def __init__(self, filename, fmt="CP2K-SCF"):
        self.filename = filename
        assert fmt in ["CP2K-SCF"], "Only support CP2K-SCF now."
        self.fmt = fmt
        self.get_data()

    def _get_atom_number(self):
        pattern = r"Number of atoms:.*?(?:\n|$)"
        matches = re.findall(pattern, self.content, re.DOTALL)
        if matches:
            return sum(
                [int(i.split()[-1]) for i in matches]
            )  # int(matches[-1].split()[-1]) # sum([int(i.split()[-1]) for i in matches]) # int(matches[-1].split()[-1]) #
        else:
            raise "No atom number found."

    def _get_energy(self):
        pattern = r"ENERGY\|.*(?:\n|$)"
        match = re.search(pattern, self.content)
        if match:
            return (
                float(match.group().split()[-1]) * 27.2113838565563
            )  # converge to eV from CP2K
        else:
            raise "No energy information found."

    def _get_box(self):
        pattern = r"(CELL\| Vector [abc].*?(?:\n|\Z))"
        matches = re.findall(pattern, self.content, re.DOTALL)
        if matches:
            return np.array([line.split()[4:7] for line in matches], float)
        else:
            raise "No box information found."

    def _get_position(self, N):
        pos_line = self.content.index("ATOMIC COORDINATES IN angstrom")
        if pos_line:
            res = self.content[pos_line:].split("\n")
            header = res[2].split()
            start = header.index("X") - len(header)
            pos = []
            for coor_start in range(3, N):
                if len(res[coor_start].split()) != 0:
                    break

            for line in res[coor_start : coor_start + N]:
                line_content = line.split()
                pos.append(line_content[start : start + 3])
            return np.array(pos, float)
        else:
            raise "No position/species information found."

    def _get_force_typelist(self, N):
        try:
            force_line = self.content.index("ATOMIC FORCES in [a.u.]")
            res = self.content[force_line:].split("\n")
            header = res[2].split()[1:]
            element_index = header.index("Element")
            force_index = header.index("X")
            force, type_list = [], []
            for line in res[3 : 3 + N]:
                line_content = line.split()
                force.append(line_content[force_index : force_index + 3])
                type_list.append(line_content[element_index])
            force = np.array(force, float) * 51.42208619083232  # converge to eV/A
            return force, type_list
        except Exception:
            raise "No force information found."

    def _get_virial_stress(self, box):
        try:
            assert (
                "STRESS TENSOR [GPa]" in self.content
                or "stress tensor [GPa]" in self.content
            )
            if "STRESS TENSOR [GPa]" in self.content:  # DIAG
                stress_line = self.content.index("STRESS TENSOR [GPa]")
                stress = np.array(
                    [
                        i.split()[-3:]
                        for i in self.content[stress_line:].split("\n")[3:6]
                    ],
                    float,
                )  # GPa
            elif "stress tensor [GPa]" in self.content:  # OT
                stress_line = self.content.index("stress tensor [GPa]")
                stress = np.array(
                    [
                        i.split()[-3:]
                        for i in self.content[stress_line:].split("\n")[2:5]
                    ],
                    float,
                )  # GPa

            virial = (
                stress
                * 0.006241509125883258
                * np.inner(box[0], np.cross(box[1], box[2]))
            )  # converge to eV
            return virial, stress
        except Exception:
            raise "No virial information found."

    def get_data(self):
        with open(self.filename) as op:
            self.content = op.read()
        self.data = {}
        self.data["N"] = self._get_atom_number()
        self.data["energy"] = self._get_energy()
        self.data["box"] = self._get_box()
        self.data["pos"] = self._get_position(self.data["N"])
        self.data["force"], self.data["type_list"] = self._get_force_typelist(
            self.data["N"]
        )
        try:
            self.data["virial"], self.data["stress"] = self._get_virial_stress(
                self.data["box"]
            )
        except Exception:
            pass


# class MultiLabeledSystem(LabeledSystem):

#     def __init__(self, filename, fmt="CP2K-AIMD"):

#         self.filename = filename
#         assert fmt in ["CP2K-AIMD"], "Only support CP2K-AIMD now."
#         self.fmt = fmt

#     def get_data(self):
#         with open(self.filename) as op:
#             content = op.read()

#         res = content.split('STEP NUMBER')
#         Nframe = len(res)
#         self.data_list = []
#         for frame in range(Nframe):
#             self.content = res[frame]
#             data = {}
#             data["N"] = self._get_atom_number()
#             self.data["energy"] = self._get_energy()
#             self.data["box"] = self._get_box()
#             self.data["pos"] = self._get_position(self.data["N"])
#             self.data["force"], self.data["type_list"] = self._get_force_typelist(
#                 self.data["N"]
#             )
#             try:
#                 self.data["virial"], self.data["stress"] = self._get_virial_stress(
#                     self.data["box"]
#                 )
#             except Exception:
#                 pass


class DFT2NEPXYZ:
    """This class is used to generate `NEP <https://gpumd.org/potentials/nep.html>`_ training XYZ file from many DFT data.
    Now we only support SCF calculation in CP2K. In the future the AIMD and VASP may also be implemented.

    Args:
        filename_list (list): all DFT output file you want to save, such as ['a/output.log', 'b/output.log']
        fmt (str, optional): DFT calculation code. Defaults to "CP2K-SCF".
        interval (int, optional): if provided, we will save it to test.xyz per interval. Defaults to 10.
        energy_shift (dict, optional): unit is eV. Ff provided, the energy will substract the base energy, such as {'Fe':89.0, 'O':50.0}. Defaults to None.
        save_virial (bool, optional): if set False, the virial information will not be saved. Defaults to True.
        force_max (float, optional): if system's abusolute maximum force (in eV/A) is larger than this value, it will not be saved to train.xyz.
        stress_max (float, optional): if system's abusolute maximum stress (in GPa) is larger than this value, it will not be saved to train.xyz.
        mode (str, optional): if mode is 'w', it will generate new train.xyz/test.xyz. If mode is 'a', it will append in current train.xyz/test.xyz. Defaults to 'w'.
    Outputs:
        - Generate train.xyz and test.xyz in current folder (if *interval* provides).
    """

    def __init__(
        self,
        filename_list,
        fmt="CP2K-SCF",
        interval=10,
        energy_shift=None,
        save_virial=True,
        force_max=None,
        stress_max=None,
        mode="w",
    ):
        self.filename_list = filename_list
        assert fmt in ["CP2K-SCF"], "Only support CP2K-SCF now."
        self.fmt = fmt
        if interval is not None:
            interval = int(interval)
        self.interval = interval
        self.energy_shift = energy_shift
        self.save_virial = save_virial
        if force_max is not None:
            force_max = abs(force_max)
        self.force_max = force_max
        if stress_max is not None:
            stress_max = abs(stress_max)
        self.stress_max = stress_max
        assert mode in ["w", "a"], "mode must in ['w', 'a']."
        self.mode = mode
        self._write_xyz()

    def __repr__(self):
        return ""

    def _write_nep_xyz(self, output_name, data):
        with open(output_name, "a") as op:
            op.write(f"{data['N']}\n")
            box_str = (
                "lattice="
                + '"'
                + "{} {} {} {} {} {} {} {} {}".format(*data["box"].flatten().tolist())
                + '"'
            )
            if "virial" in data.keys() and self.save_virial:
                virial_str = (
                    "virial="
                    + '"'
                    + "{} {} {} {} {} {} {} {} {}".format(
                        *data["virial"].flatten().tolist()
                    )
                    + '"'
                )
            else:
                virial_str = ""
            if self.energy_shift is not None:
                num = dict(Counter(data["type_list"]))
                shift = 0.0
                for i in num.keys():
                    shift += num[i] * self.energy_shift[i]
                data["energy"] -= shift
            op.write(
                f"{box_str} energy={data['energy']} {virial_str} properties=species:S:1:pos:R:3:force:R:3\n"
            )
            for i in range(data["N"]):
                op.write(
                    "{} {} {} {} {} {} {}\n".format(
                        data["type_list"][i], *data["pos"][i], *data["force"][i]
                    )
                )

    def _write_xyz(self):
        start = time()
        if self.mode == "w":
            if os.path.exists("train.xyz"):
                os.remove("train.xyz")
            if os.path.exists("test.xyz"):
                os.remove("test.xyz")
        elif self.mode == "a":
            print("Adding frames to train.xyz/test.xyz.")
        frame = 0
        bar = tqdm(range(len(self.filename_list)))
        if self.interval is not None:
            for i in bar:
                filename = self.filename_list[i]
                bar.set_description(f"Saving {frame+1} frames")
                try:
                    LS = LabeledSystem(filename)
                    # LS.get_data()
                    if self.force_max is not None and self.stress_max is not None:
                        if (
                            abs(LS.data["force"]).max() < self.force_max
                            and abs(LS.data["stress"].max()) < self.stress_max
                        ):
                            if i % self.interval == 0:
                                self._write_nep_xyz("test.xyz", LS.data)
                            else:
                                self._write_nep_xyz("train.xyz", LS.data)
                            frame += 1
                    elif self.force_max is not None:
                        if abs(LS.data["force"]).max() < self.force_max:
                            if i % self.interval == 0:
                                self._write_nep_xyz("test.xyz", LS.data)
                            else:
                                self._write_nep_xyz("train.xyz", LS.data)
                            frame += 1
                    elif self.stress_max is not None:
                        if abs(LS.data["stress"].max()) < self.stress_max:
                            if i % self.interval == 0:
                                self._write_nep_xyz("test.xyz", LS.data)
                            else:
                                self._write_nep_xyz("train.xyz", LS.data)
                            frame += 1
                    else:
                        if i % self.interval == 0:
                            self._write_nep_xyz("test.xyz", LS.data)
                        else:
                            self._write_nep_xyz("train.xyz", LS.data)
                        frame += 1
                except Exception:
                    print(f"Warning: Something is wrong for {filename}!")
        else:
            for i in bar:
                filename = self.filename_list[i]
                bar.set_description(f"Saving {frame+1} frames")
                try:
                    LS = LabeledSystem(filename)
                    # LS.get_data()
                    if self.force_max is not None and self.stress_max is not None:
                        if (
                            abs(LS.data["force"]).max() < self.force_max
                            and abs(LS.data["stress"].max()) < self.stress_max
                        ):
                            self._write_nep_xyz("train.xyz", LS.data)
                            frame += 1
                    elif self.force_max is not None:
                        if abs(LS.data["force"]).max() < self.force_max:
                            self._write_nep_xyz("train.xyz", LS.data)
                            frame += 1
                    elif self.stress_max is not None:
                        if abs(LS.data["stress"].max()) < self.stress_max:
                            self._write_nep_xyz("train.xyz", LS.data)
                            frame += 1
                    else:
                        self._write_nep_xyz("train.xyz", LS.data)
                        frame += 1
                except Exception:
                    print(f"Warning: Something is wrong for {filename}!")
        print(f"Saved {frame} frames. Time costs {time()-start} s.")


if __name__ == "__main__":
    # LS = LabeledSystem(r'C:\Users\herrwu\Desktop\output.log')
    # LS = LabeledSystem(r"C:\Users\herrwu\Desktop\cxy_pho\output.log")
    # print(len(LS.data['type_list']))
    from glob import glob

    energy_shift = {
        "Au": -33.138799856187710,
        "Mo": -67.841399471470552,
        "S": -10.060550708954318,
        "Y": -38.141847251215943,
    }
    for i in energy_shift.keys():
        energy_shift[i] *= 27.2113838565563
    print(energy_shift)

    DFT2NEPXYZ(
        glob(r"C:\Users\herrwu\Desktop\cxy_pho\output.log"),
        force_max=None,
        interval=None,
        stress_max=100,
        energy_shift=energy_shift,
    )
    # MS = MultiLabeledSystem(r'D:\Study\Gra-Al\Read_AIMD\output.log')
    # print(MS.filename)
