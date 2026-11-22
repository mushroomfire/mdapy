# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Pseudopotentials of the UPF format, version 2, in hartree atomic units:
norm-conserving, ultrasoft and PAW. The local potential, the projectors with
their D_ij, the core charge of the nonlinear core correction, the charge of the
free atom and its pseudo-orbitals; the augmentation charges of the ultrasoft
and PAW ones; the partial waves and core charges of PAW. UPF keeps energies in
rydberg; they are halved here."""

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
    """A projector: its angular momentum, r beta(r) on the radial mesh, zero
    beyond its cutoff, and the points of the mesh up to that cutoff (the
    cutoff_radius_index of the file)."""

    angular: int
    r_beta: np.ndarray
    cutoff_index: int


@dataclass(frozen=True)
class Orbital:
    """A pseudo-orbital of the free atom: its label, angular momentum,
    occupation and r chi(r)."""

    label: str
    angular: int
    occupation: float
    r_chi: np.ndarray


@dataclass(frozen=True)
class Paw:
    """The one-center parts of a PAW dataset.

    Attributes
    ----------
    lmax_rho : int
        The largest angular momentum of the one-center densities.
    dx, xmin, zmesh : float64
        The logarithmic mesh r_i = exp(xmin + i dx) / zmesh, i from 0.
    augmentation_points : int
        The points of the mesh that the partial waves and the augmentation
        charges of the one-center densities reach (cutoff_r_index).
    multipoles : numpy.ndarray
        (P, P, 2 lmax + 1) float64 the multipoles of the augmentation charges.
    ae_waves, ps_waves : numpy.ndarray
        (P, M) float64 r phi_i(r) and r phi~_i(r), the all-electron and pseudo
        partial waves, one for each projector.
    occupations : numpy.ndarray
        (P,) float64 the occupations of the partial waves in the free atom.
    ae_core : numpy.ndarray
        (M,) float64 the all-electron core charge density.
    ae_local : numpy.ndarray
        (M,) float64 the all-electron local potential in hartree.
    core_energy : float64
        The energy of the frozen core in hartree.
    """

    lmax_rho: int
    dx: float
    xmin: float
    zmesh: float
    augmentation_points: int
    multipoles: np.ndarray
    ae_waves: np.ndarray
    ps_waves: np.ndarray
    occupations: np.ndarray
    ae_core: np.ndarray
    ae_local: np.ndarray
    core_energy: float


@dataclass(frozen=True)
class Pseudopotential:
    """A norm-conserving, ultrasoft or PAW pseudopotential.

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
    q : numpy.ndarray or None
        (P, P) float64 the integrals q_ij of the augmentation charges of an
        ultrasoft pseudopotential, None for a norm-conserving one.
    augmentation : dict or None
        The augmentation charges r^2 Q_ij^L(r), (M,) float64, by (i, j, L)
        with i <= j counted from 0; for PAW zero beyond its augmentation
        points, without the terms whose multipoles vanish.
    paw : Paw or None
        The one-center parts of a PAW dataset.
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
    q: np.ndarray | None = None
    augmentation: dict | None = None
    paw: Paw | None = None

    @property
    def ultrasoft(self) -> bool:
        """Augmented, as ultrasoft and PAW are."""
        return self.q is not None

    @property
    def kkbeta(self) -> int:
        """The points of the mesh that the projectors and the augmentation
        charges are integrated over: the largest cutoff of the projectors, for
        PAW at least the augmentation points."""
        n = max((p.cutoff_index for p in self.projectors), default=1)
        return max(n, self.paw.augmentation_points) if self.paw is not None else n

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


def _cutoff_index(attributes: dict, beta: np.ndarray) -> int:
    """The points of a projector up to its cutoff: the file's
    cutoff_radius_index, or up to its last value that is not zero."""
    if "cutoff_radius_index" in attributes:
        return int(attributes["cutoff_radius_index"])
    nonzero = np.nonzero(beta)[0]
    return int(nonzero[-1]) + 1 if len(nonzero) else 1


def _flag(value: str) -> bool:
    return value.strip().upper() in ("T", "TRUE", ".TRUE.")


def read_upf(path: str | Path) -> Pseudopotential:
    """The norm-conserving, ultrasoft or PAW pseudopotential of a UPF file of
    version 2; the augmentation charges by angular momentum (q_with_l).

    Raises
    ------
    InvalidParameter
        For a file of another version or kind (spin-orbit), augmentation
        charges without their angular momenta or pseudized inside rinner, or
        one that misses a part the pseudopotential needs.
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
    is_paw = kind == "PAW" or _flag(head.get("is_paw", "F"))
    ultrasoft = is_paw or kind in ("US", "USPP") or _flag(head.get("is_ultrasoft", "F"))
    if kind not in ("NC", "SL", "US", "USPP", "PAW"):
        raise InvalidParameter(
            f"{path.name}: only norm-conserving, ultrasoft and PAW pseudopotentials "
            f"are supported, not {kind or 'this kind'}"
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
        Projector(
            int(attributes["angular_momentum"]), beta, _cutoff_index(attributes, beta)
        )
        for attributes, body in _indexed(text, "PP_BETA")
        for beta in [_numbers(body)[:mesh]]
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
    q = augmentation = None
    if ultrasoft:
        found = _block(text, "PP_AUGMENTATION")
        if found is None:
            raise InvalidParameter(f"{path.name}: no PP_AUGMENTATION")
        attributes, body = found
        if not _flag(attributes.get("q_with_l", "F")):
            raise InvalidParameter(
                f"{path.name}: the augmentation charges lack their angular momenta "
                "(q_with_l); files made by a recent ld1.x have them"
            )
        if int(attributes.get("nqf", "0")) > 0:
            raise InvalidParameter(
                f"{path.name}: augmentation charges pseudized inside rinner"
            )
        qblock = _block(body, "PP_Q")
        if qblock is None:
            raise InvalidParameter(f"{path.name}: no PP_Q")
        q = _numbers(qblock[1])[: count * count].reshape(count, count)
        multipoles = points = None
        if is_paw:
            # as QE: the terms of vanishing multipoles left out, the charges
            # zero beyond the augmentation points
            mblock = _block(body, "PP_MULTIPOLES")
            if mblock is None:
                raise InvalidParameter(f"{path.name}: no PP_MULTIPOLES")
            lmax = max((p.angular for p in projectors), default=0)
            multipoles = (
                _numbers(mblock[1])[: count * count * (2 * lmax + 1)]
                .reshape(2 * lmax + 1, count, count)
                .transpose(2, 1, 0)
            )
            points = int(attributes["cutoff_r_index"])
            epsilon = float(attributes.get("augmentation_epsilon", "1e-12"))
        augmentation = {}
        for m in re.finditer(
            r"<PP_QIJL\.(\d+)\.(\d+)\.(\d+)\b[^>]*>(.*?)</PP_QIJL", body, re.S
        ):
            i, j, ell = int(m.group(1)) - 1, int(m.group(2)) - 1, int(m.group(3))
            values = _numbers(m.group(4))[:mesh]
            if is_paw:
                if abs(multipoles[i, j, ell]) < epsilon:
                    continue
                values = values.copy()
                values[points:] = 0.0
            augmentation[(min(i, j), max(i, j), ell)] = values
        if not augmentation:
            raise InvalidParameter(f"{path.name}: no PP_QIJL")
    paw = None
    if is_paw:
        paw = _read_paw(path.name, text, head, mesh, count, multipoles, points, radial)
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
        q=q,
        augmentation=augmentation,
        paw=paw,
    )


def _read_paw(name, text, head, mesh, count, multipoles, points, radial) -> Paw:
    """The one-center parts of a PAW file."""
    found = re.search(r"<PP_MESH\b([^>]*)>", text)
    grid = dict(_ATTRIBUTE.findall(found.group(1))) if found else {}
    if not {"dx", "xmin", "zmesh"} <= grid.keys():
        raise InvalidParameter(f"{name}: PP_MESH lacks the logarithmic mesh of PAW")
    full = _block(text, "PP_FULL_WFC")
    paw = _block(text, "PP_PAW")
    if full is None or paw is None:
        raise InvalidParameter(f"{name}: no PP_FULL_WFC or PP_PAW")

    def waves(tag: str) -> np.ndarray:
        found = _indexed(full[1], tag)
        if len(found) != count:
            raise InvalidParameter(f"{name}: {len(found)} {tag} of {count}")
        return np.array([_numbers(body)[:mesh] for _, body in found])

    def inner(tag: str, size: int) -> np.ndarray:
        found = _block(paw[1], tag)
        if found is None:
            raise InvalidParameter(f"{name}: no {tag}")
        values = _numbers(found[1])
        if len(values) < size:
            raise InvalidParameter(f"{name}: {tag} holds {len(values)} of {size}")
        return values[:size]

    ae, ps = waves("PP_AEWFC"), waves("PP_PSWFC")
    # the partial waves of the one-center densities reach the augmentation points
    ae[:, points:] = 0.0
    ps[:, points:] = 0.0
    return Paw(
        lmax_rho=int(head["l_max_rho"]),
        dx=float(grid["dx"]),
        xmin=float(grid["xmin"]),
        zmesh=float(grid["zmesh"]),
        augmentation_points=points,
        multipoles=multipoles,
        ae_waves=ae,
        ps_waves=ps,
        occupations=inner("PP_OCCUPATIONS", count),
        ae_core=inner("PP_AE_NLCC", mesh),
        ae_local=0.5 * inner("PP_AE_VLOC", mesh),
        core_energy=0.5 * float(paw[0].get("core_energy", "0")),
    )
