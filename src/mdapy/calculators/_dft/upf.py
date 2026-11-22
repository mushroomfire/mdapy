# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Norm-conserving pseudopotentials of the UPF format, version 2, in hartree
atomic units: the local potential, the Kleinman-Bylander projectors with their
D_ij, the core charge of the nonlinear core correction, the charge of the free
atom and its pseudo-orbitals. UPF keeps energies in rydberg; they are halved
here."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mdapy.errors import InvalidParameter

_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")
_ATTRIBUTE = re.compile(r'([\w.]+)\s*=\s*"([^"]*)"')


@dataclass(frozen=True)
class Projector:
    """A projector: its angular momentum and r beta(r) on the radial mesh,
    zero beyond its cutoff."""

    angular: int
    r_beta: np.ndarray


@dataclass(frozen=True)
class Orbital:
    """A pseudo-orbital of the free atom: its label, angular momentum,
    occupation and r chi(r)."""

    label: str
    angular: int
    occupation: float
    r_chi: np.ndarray


@dataclass(frozen=True)
class Pseudopotential:
    """A norm-conserving pseudopotential.

    Attributes
    ----------
    element : str
    valence : float64
        The charge of the ion, Z_v.
    functional : str
        The functional of the UPF header, such as ``"PBE"``.
    r, rab : numpy.ndarray
        (M,) float64 the radial mesh in bohr and its weights dr/di.
    local : numpy.ndarray
        (M,) float64 the local potential in hartree.
    projectors : tuple of Projector
    dij : numpy.ndarray
        (P, P) float64 the strengths of the projectors in hartree.
    core : numpy.ndarray or None
        (M,) float64 the core charge density of the nonlinear core correction,
        rho_core(r), or None without it.
    atom : numpy.ndarray
        (M,) float64 the charge of the free atom, 4 pi r^2 rho(r), which
        integrates to the valence.
    orbitals : tuple of Orbital
    digest : str
        The SHA-256 of the file, to tell the pseudopotentials of labels apart.
    """

    element: str
    valence: float
    functional: str
    r: np.ndarray
    rab: np.ndarray
    local: np.ndarray
    projectors: tuple
    dij: np.ndarray
    core: np.ndarray | None
    atom: np.ndarray
    orbitals: tuple
    digest: str

    @property
    def lmax(self) -> int:
        return max((p.angular for p in self.projectors), default=-1)


def _numbers(text: str) -> np.ndarray:
    return np.array([float(x.replace("d", "e").replace("D", "e"))
                     for x in _NUMBER.findall(text)])  # fmt: skip


def _block(text: str, tag: str) -> tuple[dict, str] | None:
    """The attributes and the content of the first element named tag."""
    found = re.search(
        rf"<{re.escape(tag)}\b([^>]*)>(.*?)</{re.escape(tag)}>", text, re.S
    )
    if found is None:
        return None
    return dict(_ATTRIBUTE.findall(found.group(1))), found.group(2)


def _indexed(text: str, tag: str) -> list[tuple[dict, str]]:
    """The elements tag.1, tag.2, ... in the order of their index."""
    pattern = rf"<{re.escape(tag)}\.(\d+)\b([^>]*)>(.*?)</{re.escape(tag)}\.\1>"
    found = [(int(m.group(1)), dict(_ATTRIBUTE.findall(m.group(2))), m.group(3))
             for m in re.finditer(pattern, text, re.S)]  # fmt: skip
    return [
        (attributes, body) for _, attributes, body in sorted(found, key=lambda x: x[0])
    ]


def _flag(value: str) -> bool:
    return value.strip().upper() in ("T", "TRUE", ".TRUE.")


def read_upf(path: str | Path) -> Pseudopotential:
    """The norm-conserving pseudopotential of a UPF file of version 2.

    Raises
    ------
    InvalidParameter
        For a file of another version or kind (ultrasoft, PAW, spin-orbit),
        or one that misses a part a norm-conserving pseudopotential needs.
    """
    path = Path(path)
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    if not re.search(r'<UPF\s+version\s*=\s*"2', text):
        raise InvalidParameter(f"{path.name}: only UPF files of version 2 are read")
    header = re.search(r"<PP_HEADER\b(.*?)/?>", text, re.S)
    if header is None:
        raise InvalidParameter(f"{path.name}: no PP_HEADER")
    head = dict(_ATTRIBUTE.findall(header.group(1)))
    kind = head.get("pseudo_type", "").strip().upper()
    if (
        kind not in ("NC", "SL")
        or _flag(head.get("is_ultrasoft", "F"))
        or _flag(head.get("is_paw", "F"))
    ):
        raise InvalidParameter(
            f"{path.name}: only norm-conserving pseudopotentials are supported, not "
            f"{kind or 'this kind'}"
        )
    if _flag(head.get("has_so", "F")):
        raise InvalidParameter(
            f"{path.name}: spin-orbit pseudopotentials are not supported"
        )
    mesh = int(head["mesh_size"])

    def radial(tag: str) -> np.ndarray:
        found = _block(text, tag)
        if found is None:
            raise InvalidParameter(f"{path.name}: no {tag}")
        values = _numbers(found[1])
        if len(values) < mesh:
            raise InvalidParameter(
                f"{path.name}: {tag} holds {len(values)} of {mesh} values"
            )
        return values[:mesh]

    r, rab = radial("PP_R"), radial("PP_RAB")
    local = 0.5 * radial("PP_LOCAL")  # Ry to Ha
    projectors = tuple(
        Projector(int(attributes["angular_momentum"]), _numbers(body)[:mesh])
        for attributes, body in _indexed(text, "PP_BETA")
    )
    count = int(head.get("number_of_proj", len(projectors)))
    if len(projectors) != count:
        raise InvalidParameter(f"{path.name}: {len(projectors)} of {count} projectors")
    dij = np.zeros((count, count))
    if count:
        found = _block(text, "PP_DIJ")
        if found is None:
            raise InvalidParameter(f"{path.name}: no PP_DIJ")
        dij = 0.5 * _numbers(found[1])[: count * count].reshape(
            count, count
        )  # Ry to Ha
    core = radial("PP_NLCC") if _flag(head.get("core_correction", "F")) else None
    orbitals = tuple(
        Orbital(attributes.get("label", ""), int(attributes["l"]),
                float(attributes.get("occupation", "0")), _numbers(body)[:mesh])
        for attributes, body in _indexed(text, "PP_CHI")
    )  # fmt: skip
    return Pseudopotential(
        element=head["element"].strip(),
        valence=float(head["z_valence"]),
        functional=head.get("functional", "").strip(),
        r=r,
        rab=rab,
        local=local,
        projectors=projectors,
        dij=dij,
        core=core,
        atom=radial("PP_RHOATOM"),
        orbitals=orbitals,
        digest=hashlib.sha256(raw).hexdigest(),
    )
