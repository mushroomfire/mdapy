# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

from .potential import EAM
import numpy as np


class EAMAverage(EAM):

    """
    用于生成平均是EAM势函数
    input:
    filename : filename of eam.alloy
    concentration : [0.25, 0.25]
    """

    def __init__(self, filename, concentration, output_name=None):
        super().__init__(filename)
        self.concentration = concentration

        assert (
            len(self.concentration) == self.Nelements
        ), f"Number of concentration list should be equal to {self.Nelements}."
        assert (
            np.sum(concentration) == 1
        ), "Concentration summation should be equal to 1."

        (
            self.embedded_data,
            self.elec_density_data,
            self.rphi_data,
            self.phi_data,
        ) = self.average()

        if output_name is None:
            self.output_name = ""
            for i in self.elements_list[:-1]:
                self.output_name += i
            self.output_name += ".average.eam.alloy"
        else:
            self.output_name = output_name

        self.write_eam_alloy(self.output_name)

    def average(self):

        self.Nelements += 1
        self.elements_list.append("A")
        self.aindex = np.r_[self.aindex, np.zeros(1, dtype=self.aindex.dtype)]
        self.amass = np.r_[
            self.amass, np.array(np.average(self.amass, weights=self.concentration))
        ]
        self.lattice_constant = np.r_[
            self.lattice_constant, np.zeros(1, dtype=self.lattice_constant.dtype)
        ]
        self.lattice_type.append("average")

        # 平均 embedded_data 和 elec_density_data
        new_embedded_data = np.r_[
            self.embedded_data,
            np.zeros((1, self.embedded_data.shape[1]), dtype=self.embedded_data.dtype),
        ]
        new_elec_density_data = np.r_[
            self.elec_density_data,
            np.zeros(
                (1, self.elec_density_data.shape[1]), dtype=self.elec_density_data.dtype
            ),
        ]
        new_embedded_data[-1, :] = np.average(
            self.embedded_data, axis=0, weights=self.concentration
        )
        new_elec_density_data[-1, :] = np.average(
            self.elec_density_data, axis=0, weights=self.concentration
        )

        # 平均 rphi_data
        new_rphi_data = np.concatenate(
            (
                self.rphi_data,
                np.zeros(
                    (self.rphi_data.shape[0], 1, self.rphi_data.shape[2]),
                    dtype=self.rphi_data.dtype,
                ),
            ),
            axis=1,
        )
        new_rphi_data = np.concatenate(
            (
                new_rphi_data,
                np.zeros(
                    (1, new_rphi_data.shape[1], new_rphi_data.shape[2]),
                    dtype=new_rphi_data.dtype,
                ),
            ),
            axis=0,
        )

        new_rphi_data[-1, :-1, :] = np.average(
            self.rphi_data, axis=0, weights=self.concentration
        )
        new_rphi_data[:-1, -1, :] = new_rphi_data[-1, :-1, :]
        column = new_rphi_data[:-1, -1, :]
        new_rphi_data[-1, -1, :] = np.average(
            column, axis=0, weights=self.concentration
        )

        new_phi_data = np.zeros_like(new_rphi_data)

        new_phi_data[:, :, 1:] = new_rphi_data[:, :, 1:] / self.r[1:]
        new_phi_data[:, :, 0] = new_phi_data[:, :, 1]
        return new_embedded_data, new_elec_density_data, new_rphi_data, new_phi_data


if __name__ == "__main__":
    import os

    potential = EAMAverage(
        "./example/CoNiFeAlCu.eam.alloy", [0.25, 0.25, 0.25, 0.075, 0.175]
    )
    potential = EAM("CoNiFeAlCu.average.eam.alloy")
    os.remove("CoNiFeAlCu.average.eam.alloy")
    potential.plot()
