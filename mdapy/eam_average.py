# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np

if __name__ == "__main__":
    from potential import EAM
else:
    from .potential import EAM


class EAMAverage(EAM):

    """This class is used to generate the average EAM (A-atom) potential, which is useful in alloy investigation.
    The A-atom potential has the similar formula with the original EAM potential:

    .. math:: E_{i}=\\sum_{i} F^{A}\\left(\\bar{\\rho}_{i}\\right)+\\frac{1}{2} \\sum_{i, j \\neq i} \\phi_{i j}^{A A},

    .. math:: F^{A}\\left(\\bar{\\rho}_{i}\\right)=\\sum_{\\alpha} c_{\\alpha} F^{\\alpha}\\left(\\bar{\\rho}_{i}\\right),

    .. math:: \\phi_{i j}^{A A}=\\sum_{\\alpha, \\beta } c_{\\alpha} c_{\\beta} \\phi_{i j}^{\\alpha \\beta},

    .. math:: \\quad \\bar{\\rho}_{i}=\\sum_{j \\neq i} \\sum_{\\alpha} c_{\\alpha} \\rho_{i j}^{\\alpha},

    where :math:`A` denotes an average-atom.

    .. note:: If you use this module in publication, you should also cite the original paper.
      `Average-atom interatomic potential for random alloys <https://doi.org/10.1103/PhysRevB.93.104201>`_

    Args:
        filename (str): filename of eam.alloy file.
        concentration (list): atomic ratio list, such as [0.5, 0.5] and the summation should be equal to 1.
        output_name (str, optional): filename of generated average EAM potential.

    Outputs:
        - **generate an averaged eam.alloy potential file with A element.**

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> potential = mp.EAMAverage("./example/CoNiFeAlCu.eam.alloy",
                        [0.2, 0.2, 0.2, 0.2, 0.2]) # Generate the EAMAverage class.

        >>> potential.plot() # plot the results.

    """

    def __init__(self, filename, concentration, output_name=None):
        super().__init__(filename)
        self.concentration = concentration

        assert (
            len(self.concentration) == self.Nelements
        ), f"Number of concentration list should be equal to {self.Nelements}."
        assert np.isclose(
            np.sum(concentration), 1
        ), "Concentration summation should be equal to 1."

        (
            self.embedded_data,
            self.elec_density_data,
            self.rphi_data,
            self.phi_data,
        ) = self._average()

        if output_name is None:
            self.output_name = ""
            for i in self.elements_list[:-1]:
                self.output_name += i
            self.output_name += ".average.eam.alloy"
        else:
            self.output_name = output_name

        self.write_eam_alloy(self.output_name)

    def _average(self):
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

        # average embedded_data and elec_density_data
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

        # average rphi_data
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
