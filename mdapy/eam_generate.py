# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
import polars as pl
import datetime


class EAMGenerate:

    """This class is used to create EAM potential including one or more elements.
    This is the python version translated from the `fortran version by Zhou et. al <https://www.ctcms.nist.gov/potentials/entry/2004--Zhou-X-W-Johnson-R-A-Wadley-H-N-G--W/>`_.

    .. note:: If you use this module in publication, you should also cite the original paper.
      `Misfit-energy-increasing dislocations in vapor-deposited CoFe/NiFe multilayers <https://doi.org/10.1103/PhysRevB.69.144113>`_

    Args:
        elements_list (list): elements list, such as ['Co', 'Ni'], wchich should choose from ["Cu","Ag","Au","Ni","Pd","Pt","Al","Pb","Fe","Mo","Ta","W","Mg","Co","Ti","Zr"].
        output_name (str, optional): filename of generated EAM file.

    Outputs:
        - **generate an eam.alloy potential file.**

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> potential = mp.EAMGenerate(["Co", "Ni", "Fe"]) # Generate the EAMGenerate class.

        >>> potential.plot() # plot the results.
    """

    def __init__(self, elements_list, output_name=None):
        self.elements_list = elements_list
        self.aviliable_elements = [
            "Cu",
            "Ag",
            "Au",
            "Ni",
            "Pd",
            "Pt",
            "Al",
            "Pb",
            "Fe",
            "Mo",
            "Ta",
            "W",
            "Mg",
            "Co",
            "Ti",
            "Zr",
        ]
        for element in self.elements_list:
            assert (
                element in self.aviliable_elements
            ), f"{element} is not in {self.aviliable_elements}!!!"

        self.eam_parameters = [
            "Cu",
            "2.556162",
            "1.554485",
            "21.175871",
            "21.175395",
            "8.127620",
            "4.334731",
            "0.396620",
            "0.548085",
            "0.308782",
            "0.756515",
            "-2.170269",
            "-0.263788",
            "1.088878",
            "-0.817603",
            "-2.19",
            "0.00",
            "0.561830",
            "-2.100595",
            "0.310490",
            "-2.186568",
            "29",
            "63.546",
            "-2.100595",
            "4.334731",
            "0.756515",
            "0.85",
            "1.15",
            "Ag",
            "2.891814",
            "1.106232",
            "14.604100",
            "14.604144",
            "9.132010",
            "4.870405",
            "0.277758",
            "0.419611",
            "0.339710",
            "0.750758",
            "-1.729364",
            "-0.255882",
            "0.912050",
            "-0.561432",
            "-1.75",
            "0.00",
            "0.744561",
            "-1.150650",
            "0.783924",
            "-1.748423",
            "47",
            "107.8682",
            "-1.150650",
            "4.870405",
            "0.750758",
            "0.85",
            "1.15",
            "Au",
            "2.885034",
            "1.529021",
            "19.991632",
            "19.991509",
            "9.516052",
            "5.075228",
            "0.229762",
            "0.356666",
            "0.356570",
            "0.748798",
            "-2.937772",
            "-0.500288",
            "1.601954",
            "-0.835530",
            "-2.98",
            "0.00",
            "1.706587",
            "-1.134778",
            "1.021095",
            "-2.978815",
            "79",
            "196.96654",
            "-1.134778",
            "5.075228",
            "0.748798",
            "0.85",
            "1.15",
            "Ni",
            "2.488746",
            "2.007018",
            "27.562015",
            "27.562031",
            "8.383453",
            "4.471175",
            "0.429046",
            "0.633531",
            "0.443599",
            "0.820658",
            "-2.693513",
            "-0.076445",
            "0.241442",
            "-2.375626",
            "-2.70",
            "0.00",
            "0.265390",
            "-0.152856",
            "0.445470",
            "-2.7",
            "28",
            "58.6934",
            "-0.152856",
            "4.471175",
            "0.820658",
            "0.85",
            "1.15",
            "Pd",
            "2.750897",
            "1.595417",
            "21.335246",
            "21.940073",
            "8.697397",
            "4.638612",
            "0.406763",
            "0.598880",
            "0.397263",
            "0.754799",
            "-2.321006",
            "-0.473983",
            "1.615343",
            "-0.231681",
            "-2.36",
            "0.00",
            "1.481742",
            "-1.675615",
            "1.130000",
            "-2.352753",
            "46",
            "106.42",
            "-1.675615",
            "4.638612",
            "0.754799",
            "0.85",
            "1.15",
            "Pt",
            "2.771916",
            "2.336509",
            "33.367564",
            "35.205357",
            "7.105782",
            "3.789750",
            "0.556398",
            "0.696037",
            "0.385255",
            "0.770510",
            "-1.455568",
            "-2.149952",
            "0.528491",
            "1.222875",
            "-4.17",
            "0.00",
            "3.010561",
            "-2.420128",
            "1.450000",
            "-4.145597",
            "78",
            "195.08",
            "-2.420128",
            "3.789750",
            "0.770510",
            "0.25",
            "1.15",
            "Al",
            "2.863924",
            "1.403115",
            "20.418205",
            "23.195740",
            "6.613165",
            "3.527021",
            "0.314873",
            "0.365551",
            "0.379846",
            "0.759692",
            "-2.807602",
            "-0.301435",
            "1.258562",
            "-1.247604",
            "-2.83",
            "0.00",
            "0.622245",
            "-2.488244",
            "0.785902",
            "-2.824528",
            "13",
            "26.981539",
            "-2.488244",
            "3.527021",
            "0.759692",
            "0.85",
            "1.15",
            "Pb",
            "3.499723",
            "0.647872",
            "8.450154",
            "8.450063",
            "9.121799",
            "5.212457",
            "0.161219",
            "0.236884",
            "0.250805",
            "0.764955",
            "-1.422370",
            "-0.210107",
            "0.682886",
            "-0.529378",
            "-1.44",
            "0.00",
            "0.702726",
            "-0.538766",
            "0.935380",
            "-1.439436",
            "82",
            "207.2",
            "-0.538766",
            "5.212457",
            "0.764955",
            "0.85",
            "1.15",
            "Fe",
            "2.481987",
            "1.885957",
            "20.041463",
            "20.041463",
            "9.818270",
            "5.236411",
            "0.392811",
            "0.646243",
            "0.170306",
            "0.340613",
            "-2.534992",
            "-0.059605",
            "0.193065",
            "-2.282322",
            "-2.54",
            "0.00",
            "0.200269",
            "-0.148770",
            "0.391750",
            "-2.539945",
            "26",
            "55.847",
            "-0.148770",
            "5.236411",
            "0.340613",
            "0.85",
            "1.15",
            "Mo",
            "2.728100",
            "2.723710",
            "29.354065",
            "29.354065",
            "8.393531",
            "4.476550",
            "0.708787",
            "1.120373",
            "0.137640",
            "0.275280",
            "-3.692913",
            "-0.178812",
            "0.380450",
            "-3.133650",
            "-3.71",
            "0.00",
            "0.875874",
            "0.776222",
            "0.790879",
            "-3.712093",
            "42",
            "95.94",
            "0.776222",
            "4.476550",
            "0.275280",
            "0.85",
            "1.15",
            "Ta",
            "2.860082",
            "3.086341",
            "33.787168",
            "33.787168",
            "8.489528",
            "4.527748",
            "0.611679",
            "1.032101",
            "0.176977",
            "0.353954",
            "-5.103845",
            "-0.405524",
            "1.112997",
            "-3.585325",
            "-5.14",
            "0.00",
            "1.640098",
            "0.221375",
            "0.848843",
            "-5.141526",
            "73",
            "180.9479",
            "0.221375",
            "4.527748",
            "0.353954",
            "0.85",
            "1.15",
            "W",
            "2.740840",
            "3.487340",
            "37.234847",
            "37.234847",
            "8.900114",
            "4.746728",
            "0.882435",
            "1.394592",
            "0.139209",
            "0.278417",
            "-4.946281",
            "-0.148818",
            "0.365057",
            "-4.432406",
            "-4.96",
            "0.00",
            "0.661935",
            "0.348147",
            "0.582714",
            "-4.961306",
            "74",
            "183.84",
            "0.348147",
            "4.746728",
            "0.278417",
            "0.85",
            "1.15",
            "Mg",
            "3.196291",
            "0.544323",
            "7.132600",
            "7.132600",
            "10.228708",
            "5.455311",
            "0.137518",
            "0.225930",
            "0.5",
            "1.0",
            "-0.896473",
            "-0.044291",
            "0.162232",
            "-0.689950",
            "-0.90",
            "0.00",
            "0.122838",
            "-0.226010",
            "0.431425",
            "-0.899702",
            "12",
            "24.305",
            "-0.226010",
            "5.455311",
            "1.0",
            "0.85",
            "1.15",
            "Co",
            "2.505979",
            "1.975299",
            "27.206789",
            "27.206789",
            "8.679625",
            "4.629134",
            "0.421378",
            "0.640107",
            "0.5",
            "1.0",
            "-2.541799",
            "-0.219415",
            "0.733381",
            "-1.589003",
            "-2.56",
            "0.00",
            "0.705845",
            "-0.687140",
            "0.694608",
            "-2.559307",
            "27",
            "58.9332",
            "-0.687140",
            "4.629134",
            "1.0",
            "0.85",
            "1.15",
            "Ti",
            "2.933872",
            "1.863200",
            "25.565138",
            "25.565138",
            "8.775431",
            "4.680230",
            "0.373601",
            "0.570968",
            "0.5",
            "1.0",
            "-3.203773",
            "-0.198262",
            "0.683779",
            "-2.321732",
            "-3.22",
            "0.00",
            "0.608587",
            "-0.750710",
            "0.558572",
            "-3.219176",
            "22",
            "47.88",
            "-0.750710",
            "4.680230",
            "1.0",
            "0.85",
            "1.15",
            "Zr",
            "3.199978",
            "2.230909",
            "30.879991",
            "30.879991",
            "8.559190",
            "4.564902",
            "0.424667",
            "0.640054",
            "0.5",
            "1.0",
            "-4.485793",
            "-0.293129",
            "0.990148",
            "-3.202516",
            "-4.51",
            "0.00",
            "0.928602",
            "-0.981870",
            "0.597133",
            "-4.509025",
            "40",
            "91.224",
            "-0.981870",
            "4.564902",
            "1.0",
            "0.85",
            "1.15",
        ]
        self._get_eam_parameters()
        self.output_name = output_name
        self._write_eam_alloy()

    def _diedai(self):
        for i1 in range(self.ntypes):
            for i2 in range(i1 + 1):
                if i1 == i2:
                    for i in range(self.nr):
                        r = i * self.dr
                        if r < self.rst:
                            r = self.rst
                        fvalue = self._prof(i1, r)
                        if self.fmax < fvalue:
                            self.fmax = fvalue
                        self.rho[i, i1] = fvalue
                        psi = self._pair(i1, i2, r)
                        self.rphi[i, i1, i2] = r * psi
                else:
                    for i in range(self.nr):
                        r = i * self.dr
                        if r < self.rst:
                            r = self.rst
                        psi = self._pair(i1, i2, r)
                        self.rphi[i, i1, i2] = r * psi
                        self.rphi[i, i2, i1] = self.rphi[i, i1, i2]
        rhom = self.fmax
        if rhom < 2.0 * self.rhoemax:
            rhom = 2.0 * self.rhoemax
        if rhom < 100.0:
            rhom = 100.0
        self.drho = rhom / (self.nrho - 1.0)
        for it in range(self.ntypes):
            for i in range(self.nrho):
                rhoF = i * self.drho
                self.embdded[i, it] = self._embed(it, rhoF)

    def _prof(self, it, r):
        f = self.fe[it] * np.exp(-self.beta1[it] * (r / self.re[it] - 1.0))
        f = f / (1.0 + (r / self.re[it] - self.ramda1[it]) ** 20)
        return f

    def _pair(self, it1, it2, r):
        if it1 == it2:
            psi1 = self.A[it1] * np.exp(-self.alpha[it1] * (r / self.re[it1] - 1.0))
            psi1 = psi1 / (1.0 + (r / self.re[it1] - self.cai[it1]) ** 20)
            psi2 = self.B[it1] * np.exp(-self.beta[it1] * (r / self.re[it1] - 1.0))
            psi2 = psi2 / (1.0 + (r / self.re[it1] - self.ramda[it1]) ** 20)
            psi = psi1 - psi2
        else:
            psiab, fab = [], []
            for it in [it1, it2]:
                psi1 = self.A[it] * np.exp(-self.alpha[it] * (r / self.re[it] - 1.0))
                psi1 = psi1 / (1.0 + (r / self.re[it] - self.cai[it]) ** 20)
                psi2 = self.B[it] * np.exp(-self.beta[it] * (r / self.re[it] - 1.0))
                psi2 = psi2 / (1.0 + (r / self.re[it] - self.ramda[it]) ** 20)
                psiab.append(psi1 - psi2)
                fab.append(self._prof(it, r))
            psi = 0.5 * (fab[1] / fab[0] * psiab[0] + fab[0] / fab[1] * psiab[1])
        return psi

    def _embed(self, it, rho):
        if rho < self.rhoe[it]:
            Fm33 = self.Fm3[it]
        else:
            Fm33 = self.Fm4[it]
        if rho < self.rhoin[it]:
            emb = (
                self.Fi0[it]
                + self.Fi1[it] * (rho / self.rhoin[it] - 1.0)
                + self.Fi2[it] * (rho / self.rhoin[it] - 1.0) ** 2
                + self.Fi3[it] * (rho / self.rhoin[it] - 1.0) ** 3
            )
        elif rho < self.rhoout[it]:
            emb = (
                self.Fm0[it]
                + self.Fm1[it] * (rho / self.rhoe[it] - 1.0)
                + self.Fm2[it] * (rho / self.rhoe[it] - 1.0) ** 2
                + Fm33 * (rho / self.rhoe[it] - 1.0) ** 3
            )
        else:
            emb = (
                self.Fn[it]
                * (1.0 - self.fnn[it] * np.log(rho / self.rhos[it]))
                * (rho / self.rhos[it]) ** self.fnn[it]
            )
        return emb

    def _get_eam_parameters(self):
        name, data = [], []
        a = 0
        for _ in range(16):
            name.append(self.eam_parameters[a].strip())
            data.append(self.eam_parameters[a + 1 : a + 28])
            a += 28

        self.data = pl.from_numpy(np.array(data, dtype=np.float64).T, schema=name)

        (
            self.re,
            self.fe,
            self.rhoe,
            self.rhos,
            self.alpha,
            self.beta,
            self.A,
            self.B,
            self.cai,
            self.ramda,
            self.Fi0,
            self.Fi1,
            self.Fi2,
            self.Fi3,
            self.Fm0,
            self.Fm1,
            self.Fm2,
            self.Fm3,
            self.fnn,
            self.Fn,
            self.ielement,
            self.amass,
            self.Fm4,
            self.beta1,
            self.ramda1,
            self.rhol,
            self.rhoh,
        ) = self.data.select(self.elements_list).to_numpy()

        self.blat = np.sqrt(2.0) * self.re
        self.rhoin = self.rhol * self.rhoe
        self.rhoout = self.rhoh * self.rhoe
        self.nr = 2000
        self.nrho = 2000
        self.alatmax = self.blat.max()
        self.rhoemax = self.rhoe.max()
        self.ntypes = len(self.elements_list)
        self.rc = np.sqrt(10.0) / 2.0 * self.alatmax
        self.rst = 0.5
        self.dr = self.rc / (self.nr - 1.0)
        self.fmax = -1.0
        self.rho = np.zeros((self.nrho, len(self.elements_list)))
        self.rphi = np.zeros(
            (self.nr, len(self.elements_list), len(self.elements_list))
        )
        self.embdded = np.zeros((self.nr, len(self.elements_list)))
        self._diedai()

    def _write_eam_alloy(self):
        struc = "fcc"
        if self.output_name is None:
            self.output_name = ""
            for i in self.elements_list:
                self.output_name += i
            self.output_name += ".eam.alloy"
        with open(self.output_name, "w") as op:
            op.write(
                f" Python version mixed by mdapy! Based on the previous fortran version by Zhou et. al.\n"
            )
            op.write(f" Building at {datetime.datetime.now()}.\n")
            op.write(
                f" CITATION: X. W. Zhou, R. A. Johnson, H. N. G. Wadley, Phys. Rev. B, 69, 144113(2004)\n"
            )
            op.write(f"    {self.ntypes} ")
            for i in self.elements_list:
                op.write(f"{i} ")
            op.write("\n")
            op.write(f" {self.nrho} {self.drho} {self.nr} {self.dr} {self.rc}\n")
            num = 1
            colnum = 5
            for i in range(self.ntypes):
                op.write(
                    f" {int(self.ielement[i])} {self.amass[i]} {self.blat[i]:.4f} {struc}\n "
                )
                for j in range(self.nrho):
                    op.write(f"{self.embdded[j, i]:.16E} ")
                    if num > colnum - 1:
                        op.write("\n ")
                        num = 0
                    num += 1
                for j in range(self.nr):
                    op.write(f"{self.rho[j, i]:.16E} ")
                    if num > colnum - 1:
                        op.write("\n ")
                        num = 0
                    num += 1
            for i1 in range(self.ntypes):
                for i2 in range(i1 + 1):
                    for j in range(self.nr):
                        op.write(f"{self.rphi[j, i1, i2]:.16E} ")
                        if num > colnum - 1:
                            op.write("\n ")
                            num = 0
                        num += 1


if __name__ == "__main__":
    from potential import EAM
    import os

    potential = EAMGenerate(["Co", "Ni", "Fe", "Al", "Cu"])

    potential = EAM("CoNiFeAlCu.eam.alloy")
    os.remove("CoNiFeAlCu.eam.alloy")
    potential.plot()
