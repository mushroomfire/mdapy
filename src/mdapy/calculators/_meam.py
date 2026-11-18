# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The modified embedded atom method (MEAM) of LAMMPS files: the library and
parameter files read, the parameters of the alloys and the pair functions
set up as LAMMPS pair_style meam does, the rest computed natively."""

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from mdapy import _core
from mdapy.calculators._base import Calculator, CalculatorResult
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.elements import SYMBOLS
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built

_native = _core.calculators

# the lattices of LAMMPS: of one element, and of pairs
_SINGLE = ("fcc", "bcc", "hcp", "dim", "dia", "dia3", "lin", "zig", "tri", "sc")
_PAIR = _SINGLE + ("b1", "c11", "l12", "b2", "ch4")
_SUPPORTED = ("fcc", "bcc", "hcp", "dim", "dia", "sc", "b1", "b2", "l12")
_Z = {
    "fcc": 12, "bcc": 8, "hcp": 12, "dia": 4, "dia3": 4, "dim": 1, "b1": 6, "sc": 6,
    "c11": 10, "l12": 12, "b2": 8, "ch4": 4, "lin": 2, "zig": 2, "tri": 2,
}  # fmt: skip
_KEYWORDS = (
    "Ec", "alpha", "rho0", "delta", "lattce", "attrac", "repuls", "nn2", "Cmin", "Cmax",
    "rc", "delr", "augt1", "gsmooth_factor", "re", "ialloy", "mixture_ref_t",
    "erose_form", "zbl", "emb_lin_neg", "bkgd_dyn", "theta",
)  # fmt: skip
# the number of indices of each keyword with indices
_INDICES = {
    "Ec": 2, "alpha": 2, "rho0": 1, "delta": 2, "lattce": 2, "attrac": 2, "repuls": 2,
    "nn2": 2, "Cmin": 3, "Cmax": 3, "re": 2, "zbl": 2, "theta": 2,
}  # fmt: skip
_SCALARS = {
    "rc": "rc", "delr": "delr", "augt1": "augt1", "gsmooth_factor": "gsmooth_factor",
    "ialloy": "ialloy", "mixture_ref_t": "mix_ref_t", "erose_form": "erose_form",
    "emb_lin_neg": "emb_lin_neg", "bkgd_dyn": "bkgd_dyn",
}  # fmt: skip


def _is_zero(x: Any) -> Any:
    return np.abs(x) < 1e-20


def _fcut(x: Any) -> Any:
    """(1 - (1 - x)^4)^2 between 0 and 1, 0 below, 1 above."""
    x = np.asarray(x, dtype=float)
    a = (1.0 - np.clip(x, 0.0, 1.0)) ** 2
    a = 1.0 - a * a
    return np.where(x >= 1.0, 1.0, np.where(x <= 0.0, 0.0, a * a))


def _shape_factors(lattice: str) -> tuple[float, float, float]:
    if lattice == "hcp":
        return (0.0, 0.0, 1.0 / 3.0)
    if lattice in ("dia", "dia3", "ch4"):
        return (0.0, 0.0, 32.0 / 9.0)
    if lattice == "dim":
        return (1.0, 2.0 / 3.0, 0.40)
    return (0.0, 0.0, 0.0)


def _read_library(path: str, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    """The entries of the names in a MEAM library file, 19 values each over
    one or more lines, the first entry of a name repeated."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError as error:
        raise InvalidParameter(f"cannot read {path!r}: {error}") from None
    words = [
        w for line in lines for w in re.split(r"[' \t\n\r\f]+", line.split("#")[0]) if w
    ]
    entries: dict[str, dict[str, Any]] = {}
    for start in range(0, len(words) - len(words) % 19, 19):
        entry = words[start : start + 19]
        name = entry[0]
        if name not in names or name in entries:
            continue
        lattice = entry[1]
        if lattice not in _SINGLE:
            raise InvalidParameter(f"{path!r}: unknown lattice {lattice!r} of {name!r}")
        try:
            numbers = [float(v) for v in entry[2:]]
        except ValueError:
            raise InvalidParameter(
                f"{path!r}: the entry of {name!r} has a word for a number"
            ) from None
        z, ielement, atwt, alpha, b0, b1, b2, b3, alat, esub, asub, t0, t1, t2, t3, rozero, ibar = (
            numbers
        )  # fmt: skip
        if abs(t0 - 1.0) >= 1e-20:
            raise InvalidParameter(f"{path!r}: t0 of {name!r} must be 1")
        if rozero <= 0.0:
            raise InvalidParameter(f"{path!r}: rho0 of {name!r} must be positive")
        if z != _Z[lattice]:
            raise InvalidParameter(
                f"{path!r}: z of {name!r} is {z:g}, not the coordination {_Z[lattice]} of {lattice}"
            )
        entries[name] = {
            "lattice": lattice, "ielement": int(ielement), "mass": atwt, "alpha": alpha,
            "beta": (b0, b1, b2, b3), "alat": alat, "esub": esub, "asub": asub,
            "t": (t0, t1, t2, t3), "rozero": rozero, "ibar": int(ibar),
        }  # fmt: skip
    missing = [n for n in names if n not in entries]
    if missing:
        raise InvalidParameter(f"{path!r} has no entries of {missing}")
    return entries


class _Setup:
    """The parameters of a MEAM potential of e elements, as LAMMPS sets them
    up: the library, the parameter file, the defaults of the alloys, the
    reference densities and the tables of the pair functions."""

    def __init__(self, library: dict[str, dict[str, Any]], names: Sequence[str]):
        e = self.e = len(names)
        self.lattce = [["fcc"] * e for _ in range(e)]
        self.Ec, self.re, self.alpha = (
            np.zeros((e, e)),
            np.zeros((e, e)),
            np.zeros((e, e)),
        )
        self.delta, self.attrac, self.repuls = (
            np.zeros((e, e)),
            np.zeros((e, e)),
            np.zeros((e, e)),
        )
        self.nn2, self.zbl = np.zeros((e, e), dtype=int), np.ones((e, e), dtype=int)
        self.Cmin, self.Cmax = np.full((e, e, e), 2.0), np.full((e, e, e), 2.8)
        self.rho0 = np.zeros(e)
        self.beta = np.zeros((4, e))
        self.t = np.zeros((4, e))
        self.A, self.ibar, self.ielement, self.masses = (
            np.zeros(e),
            [0] * e,
            [0] * e,
            np.zeros(e),
        )
        for i, name in enumerate(names):
            entry = library[name]
            lattice = self.lattce[i][i] = entry["lattice"]
            self.ielement[i] = entry["ielement"]
            self.masses[i] = entry["mass"]
            self.alpha[i, i] = entry["alpha"]
            self.beta[:, i] = entry["beta"]
            self.Ec[i, i] = entry["esub"]
            self.A[i] = entry["asub"]
            self.t[:, i] = entry["t"]
            self.rho0[i] = entry["rozero"]
            self.ibar[i] = entry["ibar"]
            alat = entry["alat"]
            self.re[i, i] = {
                "fcc": alat / np.sqrt(2.0),
                "bcc": alat * np.sqrt(3.0) / 2.0,
                "dia": alat * np.sqrt(3.0) / 4.0,
                "dia3": alat * np.sqrt(3.0) / 4.0,
            }.get(lattice, alat)
        self.rc, self.delr = 4.0, 0.1
        self.gsmooth_factor, self.augt1 = 99.0, 1
        self.ialloy = self.mix_ref_t = self.emb_lin_neg = self.bkgd_dyn = (
            self.erose_form
        ) = 0

    def set(self, keyword: str, indices: list[int], value: Any, where: str) -> None:
        """A line of a parameter file, as meam_setup_param."""
        if keyword == "theta":
            raise InvalidParameter(
                f"{where}: theta, for the zig and tri lattices, is not supported"
            )
        if keyword in _SCALARS:
            name = _SCALARS[keyword]
            setattr(
                self,
                name,
                float(value)
                if name in ("rc", "delr", "gsmooth_factor")
                else int(float(value)),
            )
            return
        count = _INDICES[keyword]
        if len(indices) < count:
            raise InvalidParameter(f"{where}: {keyword} needs {count} indices")
        if any(k < 0 or k >= self.e for k in indices[:count]):
            raise InvalidParameter(
                f"{where}: an index of {keyword} is outside the {self.e} elements"
            )
        i, j = indices[0], indices[1] if count > 1 else 0
        if keyword == "lattce":
            if value not in _PAIR:
                raise InvalidParameter(f"{where}: unknown lattice {value!r}")
            self.lattce[i][j] = value
        elif keyword == "rho0":
            if float(value) <= 0:
                raise InvalidParameter(f"{where}: rho0 must be positive")
            self.rho0[i] = float(value)
        elif keyword in ("nn2", "zbl"):
            getattr(self, keyword)[min(i, j), max(i, j)] = int(float(value))
        elif keyword in ("Cmin", "Cmax"):
            getattr(self, keyword)[i, j, indices[2]] = float(value)
        else:
            getattr(self, keyword)[i, j] = float(value)

    def done(self) -> None:
        """meam_setup_done: t1 augmented, the alloys' defaults, the reference
        densities and the pair functions."""
        e = self.e
        for i in range(e):
            for j in range(e):
                if self.lattce[i][j] not in _SUPPORTED and (
                    i == j or self.lattce[i][j] != "fcc"
                ):
                    raise InvalidParameter(
                        f"the {self.lattce[i][j]} lattice is not supported; MDAPY takes {list(_SUPPORTED)}"
                    )
            if self.ibar[i] not in (0, 1, 3, 4, -5):
                raise InvalidParameter(
                    f"ibar must be one of 0, 1, 3, 4, -5, not {self.ibar[i]}"
                )
        for name, values in (
            ("ialloy", (0, 1, 2)),
            ("erose_form", (0, 1, 2)),
            ("mix_ref_t", (0, 1)),
        ):
            if getattr(self, name) not in values:
                raise InvalidParameter(
                    f"{name} must be one of {list(values)}, not {getattr(self, name)}"
                )
        self.t[1] = self.t[1] + self.augt1 * 3.0 / 5.0 * self.t[3]
        self.__alloys()
        self.rho_ref = np.array([self.__reference_density(a) for a in range(e)])
        self.nr = 1000
        self.dr = 1.1 * self.rc / self.nr
        self.tables = np.array(
            [self.__table(a, b) for a in range(e) for b in range(a, e)]
        )  # (pairs, 7, nr)

    def __alloys(self) -> None:
        e = self.e
        for i in range(e):
            for j in range(e):
                if i > j:
                    for name in ("re", "Ec", "alpha", "nn2"):
                        getattr(self, name)[i, j] = getattr(self, name)[j, i]
                    self.lattce[i][j] = self.lattce[j][i]
                elif j > i:
                    if _is_zero(self.Ec[i, j]):
                        if self.lattce[i][j] == "l12":
                            self.Ec[i, j] = (
                                3 * self.Ec[i, i] + self.Ec[j, j]
                            ) / 4.0 - self.delta[i, j]
                        else:
                            self.Ec[i, j] = (
                                self.Ec[i, i] + self.Ec[j, j]
                            ) / 2.0 - self.delta[i, j]
                    if _is_zero(self.alpha[i, j]):
                        self.alpha[i, j] = (self.alpha[i, i] + self.alpha[j, j]) / 2.0
                    if _is_zero(self.re[i, j]):
                        self.re[i, j] = (self.re[i, i] + self.re[j, j]) / 2.0
        for i in range(1, e):
            for j in range(i):
                self.Cmin[i, j] = self.Cmin[j, i]
                self.Cmax[i, j] = self.Cmax[j, i]
        with np.errstate(divide="ignore"):
            eb = self.Cmax**2 / (4.0 * (self.Cmax - 1.0))
        self.ebound = np.maximum(np.full((e, e), 2.8**2 / (4.0 * 1.8)), eb.max(axis=2))

    # -- functions of the reference structures ------------------------------

    def _zij2(self, lattice: str, cmin: float, cmax: float) -> tuple[int, float, float]:
        """The second neighbours of a lattice: their number, their distance
        over that of the first, and their screening."""
        if lattice == "dim":
            return 0, 1.0, 0.0
        z2, a, count = {
            "fcc": (6, np.sqrt(2.0), 4), "bcc": (6, 2.0 / np.sqrt(3.0), 4),
            "hcp": (6, np.sqrt(2.0), 4), "b1": (12, np.sqrt(2.0), 2),
            "sc": (12, np.sqrt(2.0), 2), "dia": (12, np.sqrt(8.0 / 3.0), 1),
            "l12": (6, np.sqrt(2.0), 4), "b2": (6, 2.0 / np.sqrt(3.0), 4),
        }[lattice]  # fmt: skip
        c = 4.0 / (a * a) - 1.0
        s = float(_fcut((c - cmin) / (cmax - cmin)))
        return z2, a, s**count

    def _g(self, gamma: Any, ibar: int) -> Any:
        gamma = np.asarray(gamma, dtype=float)
        if ibar in (0, 4):
            switch = -self.gsmooth_factor / (self.gsmooth_factor + 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                low = np.sqrt(
                    1
                    / (self.gsmooth_factor + 1)
                    * (switch / gamma) ** self.gsmooth_factor
                )
            return np.where(
                gamma < switch,
                np.where(gamma == 0.0, 0.0, low),
                np.sqrt(np.maximum(1.0 + gamma, 0.0)),
            )
        if ibar == 1:
            return np.exp(gamma / 2.0)
        if ibar == 3:
            return 2.0 / (1.0 + np.exp(-gamma))
        return np.where(
            1.0 + gamma >= 0,
            np.sqrt(np.abs(1.0 + gamma)),
            -np.sqrt(np.abs(-1.0 - gamma)),
        )

    def _embedding(self, a: int, rhobar: Any) -> Any:
        aec = self.A[a] * self.Ec[a, a]
        rhobar = np.asarray(rhobar, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            positive = aec * rhobar * np.log(rhobar)
        negative = -aec * rhobar if self.emb_lin_neg else np.zeros_like(rhobar)
        return np.where(rhobar > 0.0, positive, negative)

    def _erose(self, r: Any, a: int, b: int) -> Any:
        re_, alpha, ec = self.re[a, b], self.alpha[a, b], self.Ec[a, b]
        repuls, attrac = self.repuls[a, b], self.attrac[a, b]
        astar = alpha * (r / re_ - 1.0)
        a3 = np.where(astar >= 0, attrac, repuls)
        if self.erose_form == 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                value = (
                    -ec
                    * (1 + astar + (-attrac + repuls / r) * astar**3)
                    * np.exp(-astar)
                )
        elif self.erose_form == 2:
            value = -ec * (1 + astar + a3 * astar**3) * np.exp(-astar)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                value = -ec * (1 + astar + a3 * astar**3 / (r / re_)) * np.exp(-astar)
        return np.where(r > 0.0, value, 0.0)

    def _zbl(self, r: Any, z1: int, z2: int) -> Any:
        c = (0.028171, 0.28022, 0.50986, 0.18175)
        d = (0.20162, 0.40290, 0.94229, 3.1998)
        a = 0.4685 / (z1**0.23 + z2**0.23)
        x = r / a
        result = sum(ci * np.exp(-di * x) for ci, di in zip(c, d))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r > 0.0, result * z1 * z2 / r * 14.3997, result)

    def __reference_density(self, a: int) -> float:
        lattice = self.lattce[a][a]
        z = _Z[lattice]
        if self.ibar[a] <= 0:
            gbar = 1.0
        else:
            s = _shape_factors(lattice)
            gam = (self.t[1, a] * s[0] + self.t[2, a] * s[1] + self.t[3, a] * s[2]) / (
                z * z
            )
            gbar = float(self._g(gam, self.ibar[a]))
        rho0 = self.rho0[a] * z
        if self.nn2[a, a] == 1:
            z2, arat, scrn = self._zij2(lattice, self.Cmin[a, a, a], self.Cmax[a, a, a])
            rho0 = (
                rho0 + z2 * self.rho0[a] * np.exp(-self.beta[0, a] * (arat - 1)) * scrn
            )
        return float(rho0 * gbar)

    def _densities(self, r: Any, a: int, b: int) -> tuple[Any, ...]:
        """get_densref: the partial densities at an atom of a and one of b in
        the reference structure of the pair at r."""
        a1, a2 = r / self.re[a, a] - 1.0, r / self.re[b, b] - 1.0
        rhoa01 = self.rho0[a] * np.exp(-self.beta[0, a] * a1)
        rhoa1 = [self.rho0[a] * np.exp(-self.beta[k, a] * a1) for k in (1, 2, 3)]
        rhoa02 = self.rho0[b] * np.exp(-self.beta[0, b] * a2)
        rhoa2 = [self.rho0[b] * np.exp(-self.beta[k, b] * a2) for k in (1, 2, 3)]
        zero = np.zeros_like(rhoa01)
        r11 = r21 = r31 = r12 = r22 = r32 = zero
        lattice = self.lattce[a][b]
        if lattice in ("fcc", "bcc", "b1", "sc", "b2"):
            z = {"fcc": 12.0, "bcc": 8.0, "b1": 6.0, "sc": 6.0, "b2": 8.0}[lattice]
            r01, r02 = z * rhoa02, z * rhoa01
        elif lattice == "dia":
            r01, r02 = 4.0 * rhoa02, 4.0 * rhoa01
            r31, r32 = (
                32.0 / 9.0 * rhoa2[2] * rhoa2[2],
                32.0 / 9.0 * rhoa1[2] * rhoa1[2],
            )
        elif lattice == "hcp":
            r01, r02 = 12 * rhoa02, 12 * rhoa01
            r31, r32 = 1.0 / 3.0 * rhoa2[2] * rhoa2[2], 1.0 / 3.0 * rhoa1[2] * rhoa1[2]
        elif lattice == "dim":
            s = _shape_factors("dim")
            r01, r02 = rhoa02, rhoa01
            r11, r12 = s[0] * rhoa2[0] * rhoa2[0], s[0] * rhoa1[0] * rhoa1[0]
            r21, r22 = s[1] * rhoa2[1] * rhoa2[1], s[1] * rhoa1[1] * rhoa1[1]
            r31, r32 = s[2] * rhoa2[2] * rhoa2[2], s[2] * rhoa1[2] * rhoa1[2]
        else:  # l12
            r01, r02 = 8 * rhoa01 + 4 * rhoa02, 12 * rhoa01
            if self.ialloy == 1:
                r21 = (
                    8.0 / 3.0 * (rhoa1[1] * self.t[2, a] - rhoa2[1] * self.t[2, b]) ** 2
                )
                denom = 8 * rhoa01 * self.t[2, a] ** 2 + 4 * rhoa02 * self.t[2, b] ** 2
                with np.errstate(divide="ignore", invalid="ignore"):
                    r21 = np.where(denom > 0.0, r21 / denom * r01, r21)
            else:
                r21 = 8.0 / 3.0 * (rhoa1[1] - rhoa2[1]) * (rhoa1[1] - rhoa2[1])
        if self.nn2[a, b] == 1:
            z2, arat, scrn = self._zij2(lattice, self.Cmin[a, a, b], self.Cmax[a, a, b])
            rhoa01nn = self.rho0[a] * np.exp(
                -self.beta[0, a] * (arat * r / self.re[a, a] - 1.0)
            )
            rhoa02nn = self.rho0[b] * np.exp(
                -self.beta[0, b] * (arat * r / self.re[b, b] - 1.0)
            )
            if lattice == "l12":
                s111 = float(
                    _fcut(
                        (1.0 - self.Cmin[a, a, a])
                        / (self.Cmax[a, a, a] - self.Cmin[a, a, a])
                    )
                )
                s112 = float(
                    _fcut(
                        (1.0 - self.Cmin[a, a, b])
                        / (self.Cmax[a, a, b] - self.Cmin[a, a, b])
                    )
                )
                s221 = float(
                    _fcut(
                        (1.0 - self.Cmin[b, b, a])
                        / (self.Cmax[b, b, a] - self.Cmin[b, b, a])
                    )
                )
                s11, s22 = s111 * s111 * s112 * s112, s221**4
                r01 = r01 + 6 * s11 * rhoa01nn
                r02 = r02 + 6 * s22 * rhoa02nn
            else:
                r01 = r01 + z2 * scrn * rhoa01nn
                z2, arat, scrn = self._zij2(
                    lattice, self.Cmin[b, b, a], self.Cmax[b, b, a]
                )
                r02 = r02 + z2 * scrn * rhoa02nn
        return r01, r11, r21, r31, r02, r12, r22, r32

    def _t_averages(self, r: Any, a: int, b: int) -> tuple[Any, ...]:
        """get_tavref: the averages of t at an atom of a and one of b."""
        t1, t2 = self.t[1:, a], self.t[1:, b]
        lattice = self.lattce[a][b]
        if self.ialloy == 2:
            return (*t1, *t2)
        if lattice != "l12":
            return (*t2, *t1)
        rhoa01 = self.rho0[a] * np.exp(-self.beta[0, a] * (r / self.re[a, a] - 1.0))
        rhoa02 = self.rho0[b] * np.exp(-self.beta[0, b] * (r / self.re[b, b] - 1.0))
        rho01 = 8 * rhoa01 + 4 * rhoa02
        with np.errstate(divide="ignore", invalid="ignore"):
            first = [
                np.where(
                    rho01 > 0.0,
                    (8 * t1[k] * rhoa01 + 4 * t2[k] * rhoa02) / rho01,
                    (2.0 * t1[k] + t2[k]) / 3.0,
                )
                for k in range(3)
            ]
        return (*first, *t1)

    def phi(self, r: Any, a: int, b: int) -> Any:
        """phi_meam: the pair function of a and b at r."""
        r = np.asarray(r, dtype=float)
        z1, z2, z12 = (
            _Z[self.lattce[a][a]],
            _Z[self.lattce[b][b]],
            _Z[self.lattce[a][b]],
        )
        r01, r11, r21, r31, r02, r12, r22, r32 = self._densities(r, a, b)
        t11, t21, t31, t12, t22, t32 = self._t_averages(r, a, b)
        if self.mix_ref_t == 1:
            if self.ibar[a] <= 0:
                g1 = 1.0
            else:
                s = _shape_factors(self.lattce[a][a])
                g1 = self._g(
                    (s[0] * t11 + s[1] * t21 + s[2] * t31) / (z1 * z1), self.ibar[a]
                )
            if self.ibar[b] <= 0:
                g2 = 1.0
            else:
                s = _shape_factors(self.lattce[b][b])
                g2 = self._g(
                    (s[0] * t12 + s[1] * t22 + s[2] * t32) / (z2 * z2), self.ibar[b]
                )
            bkgd1, bkgd2 = self.rho0[a] * z1 * g1, self.rho0[b] * z2 * g2
        elif self.bkgd_dyn == 1:
            bkgd1, bkgd2 = self.rho0[a] * z1, self.rho0[b] * z2
        else:
            bkgd1, bkgd2 = self.rho_ref[a], self.rho_ref[b]
        with np.errstate(divide="ignore", invalid="ignore"):
            gam1 = np.where(
                r01 < 1.0e-14, 0.0, (t11 * r11 + t21 * r21 + t31 * r31) / (r01 * r01)
            )
            gam2 = np.where(
                r02 < 1.0e-14, 0.0, (t12 * r12 + t22 * r22 + t32 * r32) / (r02 * r02)
            )
        rhobar1 = r01 / bkgd1 * self._g(gam1, self.ibar[a])
        rhobar2 = r02 / bkgd2 * self._g(gam2, self.ibar[b])
        f1, f2 = self._embedding(a, rhobar1), self._embedding(b, rhobar2)
        eu = self._erose(r, a, b)
        if self.lattce[a][b] == "l12":
            z1nn = _Z[self.lattce[a][a]]
            z2nn, arat, scrn = self._zij2(
                self.lattce[a][a], self.Cmin[a, a, a], self.Cmax[a, a, a]
            )
            phiaa = self.phi(r, a, a) + self._series(scrn, z1nn, z2nn, a, a, r, arat)
            value = eu / 3.0 - f1 / 4.0 - f2 / 12.0 - phiaa
        else:
            value = (2 * eu - f1 - f2) / z12
        value = np.where((r01 <= 1e-14) & (r02 <= 1e-14), 0.0, value)
        return np.where(_is_zero(r), 0.0, value)

    def _series(
        self, scrn: float, z1: int, z2: int, a: int, b: int, r: Any, arat: float
    ) -> Any:
        """phi_meam_series: the second neighbours screened, a geometric series
        of phi stopped at its first term of zero, at most 10."""
        r = np.asarray(r, dtype=float)
        total = np.zeros_like(r)
        if not scrn > 0.0:
            return total
        b2nn = -z2 * scrn / z1
        going = np.ones(r.shape, dtype=bool)
        for n in range(1, 11):
            value = b2nn**n * self.phi(r * arat**n, a, b)
            going &= ~_is_zero(value)
            total = total + np.where(going, value, 0.0)
        return total

    def __table(self, a: int, b: int) -> np.ndarray:
        """compute_pair_meam and interpolate_meam: phi of the pair on nr
        points and the coefficients of its spline."""
        r = np.arange(self.nr) * self.dr
        phir = self.phi(r, a, b)
        lattice = self.lattce[a][b]
        if self.nn2[a, b] == 1:
            z1 = _Z[lattice]
            z2, arat, scrn = self._zij2(lattice, self.Cmin[a, a, b], self.Cmax[a, a, b])
            if lattice in ("b1", "b2", "l12", "dia"):
                rarat = r * arat
                za1 = _Z[self.lattce[a][a]]
                za2, aarat, ascrn = self._zij2(
                    self.lattce[a][a], self.Cmin[a, a, a], self.Cmax[a, a, a]
                )
                phiaa = self.phi(rarat, a, a) + self._series(
                    ascrn, za1, za2, a, a, rarat, aarat
                )
                zb1 = _Z[self.lattce[b][b]]
                zb2, barat, bscrn = self._zij2(
                    self.lattce[b][b], self.Cmin[b, b, b], self.Cmax[b, b, b]
                )
                phibb = self.phi(rarat, b, b) + self._series(
                    bscrn, zb1, zb2, b, b, rarat, barat
                )
                if lattice in ("b1", "b2", "dia"):
                    z2, _, scrn = self._zij2(
                        lattice, self.Cmin[a, a, b], self.Cmax[a, a, b]
                    )
                    phir = phir - z2 * scrn / (2 * z1) * phiaa
                    z2, _, scrn2 = self._zij2(
                        lattice, self.Cmin[b, b, a], self.Cmax[b, b, a]
                    )
                    phir = phir - z2 * scrn2 / (2 * z1) * phibb
                else:
                    s111 = float(
                        _fcut(
                            (1.0 - self.Cmin[a, a, a])
                            / (self.Cmax[a, a, a] - self.Cmin[a, a, a])
                        )
                    )
                    s112 = float(
                        _fcut(
                            (1.0 - self.Cmin[a, a, b])
                            / (self.Cmax[a, a, b] - self.Cmin[a, a, b])
                        )
                    )
                    s221 = float(
                        _fcut(
                            (1.0 - self.Cmin[b, b, a])
                            / (self.Cmax[b, b, a] - self.Cmin[b, b, a])
                        )
                    )
                    s11, s22 = s111 * s111 * s112 * s112, s221**4
                    phir = phir - 0.75 * s11 * phiaa - 0.25 * s22 * phibb
            else:
                phir = phir + self._series(scrn, z1, z2, a, b, r, arat)
        if self.zbl[min(a, b), max(a, b)] == 1:
            astar = self.alpha[a, b] * (r / self.re[a, b] - 1.0)
            zbl = self._zbl(r, self.ielement[a], self.ielement[b])
            frac = _fcut(1 - (astar + 1.0) / (-3.0 + 1.0))
            phir = np.where(
                astar <= -3.0,
                zbl,
                np.where(astar < -1.0, frac * phir + (1 - frac) * zbl, phir),
            )
        return _spline(phir, self.dr)


def _spline(values: np.ndarray, delta: float) -> np.ndarray:
    """interpolate_meam: the values, the slopes by fourth-order differences
    (zero at the last point), and the coefficients of the cubic of each
    interval, as (7, n)."""
    n = len(values)
    s1 = np.zeros(n)
    s1[0] = values[1] - values[0]
    s1[1] = 0.5 * (values[2] - values[0])
    s1[n - 2] = 0.5 * (values[n - 1] - values[n - 3])
    s1[n - 1] = 0.0
    s1[2 : n - 2] = (
        (values[:-4] - values[4:]) + 8.0 * (values[3:-1] - values[1:-3])
    ) / 12.0
    s2, s3 = np.zeros(n), np.zeros(n)
    s2[:-1] = 3.0 * (values[1:] - values[:-1]) - 2.0 * s1[:-1] - s1[1:]
    s3[:-1] = s1[:-1] + s1[1:] - 2.0 * (values[1:] - values[:-1])
    return np.array(
        [values, s1, s2, s3, s1 / delta, 2.0 * s2 / delta, 3.0 * s3 / delta]
    )


def _read_parameters(path: str, setup: _Setup) -> None:
    """The lines of a MEAM parameter file, keyword(indices) = value."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError as error:
        raise InvalidParameter(f"cannot read {path!r}: {error}") from None
    for number, line in enumerate(lines, start=1):
        words = [w for w in re.split(r"[=(), '\t\n\r\f]+", line.split("#")[0]) if w]
        if not words:
            continue
        keyword, where = words[0], f"{path!r}, line {number}"
        if keyword not in _KEYWORDS:
            raise InvalidParameter(f"{where}: unknown keyword {keyword!r}")
        try:
            indices = [int(w) - 1 for w in words[1:-1]]
        except ValueError:
            raise InvalidParameter(
                f"{where}: the indices of {keyword} must be whole numbers"
            ) from None
        value: Any = words[-1] if len(words) > 1 else ""
        if keyword != "lattce":
            try:
                value = float(value)
            except ValueError:
                raise InvalidParameter(
                    f"{where}: the value of {keyword} must be a number"
                ) from None
        setup.set(keyword, indices, value, where)


class MEAM(Calculator):
    """Modified embedded atom method of LAMMPS files, computed as LAMMPS
    ``pair_style meam`` computes it.

    Parameters
    ----------
    library : str or os.PathLike
        The library file: the parameters of each element.
    elements : sequence of str or mapping
        The entries of the library to use, in the order the indices of the
        parameter file number them, as the names after the files in
        ``pair_coeff`` of LAMMPS; atoms are matched to an entry by its
        name, or the atomic number the library gives an entry named otherwise.
        A mapping takes the element of the atoms to an entry, as {"Al": "AlS"}.
    parameters : str or os.PathLike, optional
        The parameter file of the alloy and the options; None for none, as
        NULL in LAMMPS.

    Attributes
    ----------
    library : str
        The library file.
    parameters : str or None
        The parameter file.
    elements : tuple of str
        The elements of the atoms, in the order of the entries.
    masses : numpy.ndarray
        (E,) float64, the masses of the entries in amu.
    rc : float64
        The cutoff of the pairs in Å.
    cutoff : float64
        The radius in Å of the neighbors needed: rc, or more where atoms
        farther away screen a pair.
    """

    def __init__(
        self,
        library: str | os.PathLike,
        elements: Sequence[str] | Mapping[str, str],
        parameters: str | os.PathLike | None = None,
    ):
        if isinstance(elements, Mapping):
            symbols, names = tuple(elements), tuple(elements.values())
        else:
            names = (elements,) if isinstance(elements, str) else tuple(elements)
            symbols = None
        if not names or len(set(names)) != len(names):
            raise InvalidParameter("elements must name each entry of the library once")
        self.library = os.fspath(library)
        entries = _read_library(self.library, names)
        if symbols is None:
            # an entry named by an element, as 'Mo', is that element; another,
            # as 'AlS', the element of its atomic number
            found = []
            for name in names:
                z = entries[name]["ielement"]
                if name in SYMBOLS and name != "X":
                    found.append(name)
                elif 0 < z < len(SYMBOLS):
                    found.append(SYMBOLS[z])
                else:
                    raise InvalidParameter(
                        f"the entry {name!r} is not named by an element and its atomic number "
                        f"{z} is none: map the element of the atoms to it, as {{'Al': {name!r}}}"
                    )
            if len(set(found)) != len(found):
                raise InvalidParameter(
                    f"two entries are of the same element ({found}): map the elements of the "
                    "atoms to the entries, as {'Al': 'AlS'}"
                )
            symbols = tuple(found)
        for symbol in symbols:
            if symbol not in SYMBOLS:
                raise InvalidParameter(f"{symbol!r} is not an element")
        setup = _Setup(entries, names)
        self.parameters = None if parameters is None else os.fspath(parameters)
        if self.parameters is not None:
            _read_parameters(self.parameters, setup)
        setup.done()
        self.elements = symbols
        self.masses = setup.masses
        self.rc = float(setup.rc)
        cmax = setup.Cmax[setup.Cmax > 2.0]
        factor = (
            max(1.0, float((cmax / (2 * np.sqrt(cmax - 1))).max()))
            if cmax.size
            else 1.0
        )
        self.cutoff = self.rc * factor
        e = setup.e
        per_element = np.array(
            [
                setup.rho0, *setup.beta, *setup.t[1:], setup.A, np.diag(setup.Ec),
                np.diag(setup.re), setup.rho_ref, [_Z[setup.lattce[a][a]] for a in range(e)],
                [float(i) for i in setup.ibar],
                *np.array([_shape_factors(setup.lattce[a][a]) for a in range(e)]).T,
            ]
        )  # fmt: skip
        self.__species = np.full(len(SYMBOLS), -1, dtype=np.int32)
        for k, symbol in reversed(list(enumerate(symbols))):
            self.__species[SYMBOLS.index(symbol)] = k
        self.__potential = _native.make_meam(
            np.ascontiguousarray(per_element),
            np.ascontiguousarray(setup.ebound),
            np.ascontiguousarray(setup.Cmin),
            np.ascontiguousarray(setup.Cmax),
            np.ascontiguousarray(setup.tables),
            [setup.ialloy, setup.mix_ref_t, setup.emb_lin_neg, setup.bkgd_dyn],
            [setup.rc, setup.delr, setup.gsmooth_factor, setup.dr, self.cutoff],
        )

    def compute(
        self,
        system: System,
        properties: Iterable[str] = ("energy", "forces"),
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> CalculatorResult:
        self._check_properties(properties)
        if "element" not in system.table.columns:
            raise MissingProperty("MEAM needs the element column")
        numbers = system.table["element"].to_physical().to_numpy()
        species = self.__species[numbers]
        if (species < 0).any():
            missing = sorted({SYMBOLS[z] for z in numbers[species < 0]})
            raise InvalidParameter(
                f"the potential has no parameters for {missing}; it has {list(self.elements)}"
            )
        neighbors = _given_or_built(system, self.cutoff, neighbors)
        with neighbors._reading():
            energies, forces, virials = _native.compute_meam(
                self.__potential,
                species,
                np.ascontiguousarray(system["positions"]),
                np.ascontiguousarray(system.box.matrix),
                neighbors.offsets,
                neighbors.j,
                neighbors.shifts,
                progress,
                _native_token(cancel),
            )
        return CalculatorResult(
            energy=float(energies.sum()),
            atomic_energies=energies,
            forces=forces,
            virial=virials.sum(axis=0),
            atomic_virials=virials,
            volume=system.box.volume,
        )

    def __repr__(self) -> str:
        return (
            f"MEAM({self.library!r}, elements={self.elements}, parameters={self.parameters!r}, "
            f"rc={self.rc})"
        )
