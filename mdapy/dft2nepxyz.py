# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.
import numpy as np
import re
from time import time
from collections import Counter
import os


class LabeledSystem:

    def __init__(self, filename, fmt="CP2K-SCF"):
        self.filename = filename
        assert fmt in ["CP2K-SCF"], "Only support CP2K-SCF now."
        self.fmt = fmt
        self._get_data()

    def _get_atom_number(self):
        pattern = r"Number of atoms:.*?(?:\n|$)"
        matches = re.findall(pattern, self.content, re.DOTALL)
        if matches:
            return sum([int(i.split()[-1]) for i in matches])
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
            raise "No box inforation found."

    def _get_position(self, N):
        pos_line = self.content.index("ATOMIC COORDINATES IN angstrom")
        if pos_line:
            res = self.content[pos_line:].split("\n")
            header = res[2].split()
            start = header.index("X") - len(header)
            pos = []
            for line in res[4 : 4 + N]:
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

    def _get_virial(self, box):
        try:
            stress_line = self.content.index("STRESS TENSOR [GPa]")
            return (
                np.array(
                    [
                        i.split()[1:]
                        for i in self.content[stress_line:].split("\n")[3:6]
                    ],
                    float,
                )
                * 0.006241509125883258
                * np.inner(box[0], np.cross(box[1], box[2]))
            )  # converge to eV
        except Exception:
            raise "No virial information found."

    def _get_data(self):
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
            self.data["virial"] = self._get_virial(self.data["box"])
        except Exception:
            pass


class DFT2NEPXYZ:

    def __init__(
        self,
        filename_list,
        fmt="CP2K-SCF",
        interval=10,
        energy_shift=None,
        save_virial=True,
    ):
        self.filename_list = filename_list
        assert fmt in ["CP2K-SCF"], "Only support CP2K-SCF now."
        self.fmt = fmt
        if interval is not None:
            interval = int(interval)
        self.interval = interval
        self.energy_shift = energy_shift
        self.save_virial = save_virial
        self.write_xyz()

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

    def write_xyz(self):
        start = time()
        if os.path.exists("train.xyz"):
            os.remove("train.xyz")
        if os.path.exists("test.xyz"):
            os.remove("test.xyz")
        frame = 0
        if self.interval is not None:
            for i, filename in enumerate(self.filename_list, start=1):
                try:
                    LS = LabeledSystem(filename)
                    if i % self.interval == 0:
                        self._write_nep_xyz("test.xyz", LS.data)
                    else:
                        self._write_nep_xyz("train.xyz", LS.data)
                    frame += 1
                except Exception as e:
                    print(f"Warning: Something is wrong for {filename}!")
        else:
            for i, filename in enumerate(self.filename_list, start=1):
                try:
                    LS = LabeledSystem(filename)
                    self._write_nep_xyz("train.xyz", LS.data)
                    frame += 1
                except Exception:
                    print(f"Warning: Something is wrong for {filename}!")
        print(f"Saving {frame} frames. Time costs {time()-start} s.")
