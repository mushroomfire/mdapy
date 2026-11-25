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
    lmax_rho : int32
        The largest angular momentum of the one-center densities.
    dx, xmin, zmesh : float64
        The logarithmic mesh r_i = exp(xmin + i dx) / zmesh, i from 0.
    augmentation_points : int32
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


def _float(text: str) -> float:
    """A number, also in Fortran's notation, 1.d-12."""
    return float(text.strip().replace("d", "e").replace("D", "e"))


def _numbers(text: str) -> np.ndarray:
    return np.array([_float(x) for x in _NUMBER.findall(text)])


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
    """The norm-conserving, ultrasoft or PAW pseudopotential of a UPF file:
    of version 2, its augmentation charges by angular momentum (q_with_l);
    or a norm-conserving or ultrasoft one of version 1, such as those of
    GBRV, its augmentation charges pseudized inside rinner as Quantum
    ESPRESSO pseudizes them.

    Raises
    ------
    InvalidParameter
        For a file of another kind (spin-orbit), augmentation charges of
        version 2 without their angular momenta or pseudized inside rinner,
        or one that misses a part the pseudopotential needs.
    """
    path = Path(path)
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    if not re.search(r'<UPF\s+version\s*=\s*"2', text):
        if "<PP_HEADER>" in text:
            return _read_v1(path, raw, text)
        raise InvalidParameter(f"{path.name}: not a UPF file of version 1 or 2")
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
                _float(attributes.get("occupation", "0")), _numbers(body)[:mesh])
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
            epsilon = _float(attributes.get("augmentation_epsilon", "1e-12"))
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
        valence=_float(head["z_valence"]),
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
        dx=_float(grid["dx"]),
        xmin=_float(grid["xmin"]),
        zmesh=_float(grid["zmesh"]),
        augmentation_points=points,
        multipoles=multipoles,
        ae_waves=ae,
        ps_waves=ps,
        occupations=inner("PP_OCCUPATIONS", count),
        ae_core=inner("PP_AE_NLCC", mesh),
        ae_local=0.5 * inner("PP_AE_VLOC", mesh),
        core_energy=0.5 * _float(paw[0].get("core_energy", "0")),
    )


def _read_v1(path: Path, raw: bytes, text: str) -> Pseudopotential:
    """A norm-conserving or ultrasoft pseudopotential of a UPF file of version 1."""

    def block(tag: str, required: bool = True) -> str | None:
        found = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
        if found is None and required:
            raise InvalidParameter(f"{path.name}: no {tag}")
        return None if found is None else found.group(1)

    lines = [line for line in block("PP_HEADER").splitlines() if line.strip()]
    kind = lines[2].split()[0].upper()
    if kind not in ("NC", "SL", "US"):
        raise InvalidParameter(
            f"{path.name}: only norm-conserving and ultrasoft pseudopotentials are "
            f"read from files of version 1, not {kind}"
        )
    element = lines[1].split()[0]
    nlcc = _flag(lines[3].split()[0])
    functional = " ".join(lines[4][:20].split())  # as QE reads it, 20 characters
    valence = _float(lines[5].split()[0])
    mesh = int(lines[9].split()[0])
    count = int(lines[10].split()[1])

    def radial(tag: str) -> np.ndarray:
        values = _numbers(block(tag))
        if len(values) < mesh:
            raise InvalidParameter(
                f"{path.name}: {tag} holds {len(values)} of {mesh} values"
            )
        return values[:mesh]

    r, rab = radial("PP_R"), radial("PP_RAB")
    local = 0.5 * radial("PP_LOCAL")  # Ry to Ha
    projectors = []
    for body in re.findall(r"<PP_BETA>(.*?)</PP_BETA>", text, re.S):
        # index, l, the points up to the cutoff, then r beta(r) there
        numbers = _numbers(body)
        angular, points = int(numbers[1]), int(numbers[2])
        beta = np.zeros(mesh)
        beta[:points] = numbers[3 : 3 + points]
        projectors.append(Projector(angular, beta, points))
    if len(projectors) != count:
        raise InvalidParameter(f"{path.name}: {len(projectors)} of {count} projectors")
    dij = np.zeros((count, count))
    if count:
        numbers = _numbers(block("PP_DIJ"))
        for k in range(int(numbers[0])):
            i, j, d = numbers[1 + 3 * k : 4 + 3 * k]
            dij[int(i) - 1, int(j) - 1] = dij[int(j) - 1, int(i) - 1] = (
                0.5 * d
            )  # Ry to Ha
    core = radial("PP_NLCC") if nlcc else None
    orbitals = []
    wavefunctions = block("PP_PSWFC", required=False) or ""
    for found in re.finditer(
        r"^\s*(\S+)\s+(\d+)\s+([-+.\dEeDd]+)\s+Wavefunction[^\n]*\n(.*?)(?=^\s*\S+\s+\d+\s+\S+\s+Wavefunction|\Z)",
        wavefunctions, re.M | re.S,
    ):  # fmt: skip
        label, angular, occupation, body = found.groups()
        orbitals.append(
            Orbital(label, int(angular), _float(occupation), _numbers(body)[:mesh])
        )
    q = augmentation = None
    if kind == "US":
        q, augmentation = _augmentation_v1(
            path.name, block("PP_QIJ"), projectors, r, mesh
        )
    return Pseudopotential(
        element=element, valence=valence, functional=functional, r=r, rab=rab,
        local=local, projectors=tuple(projectors), dij=dij, core=core,
        atom=radial("PP_RHOATOM"), orbitals=tuple(orbitals),
        digest=hashlib.sha256(raw).hexdigest(), q=q, augmentation=augmentation,
    )  # fmt: skip


def _augmentation_v1(name: str, body: str, projectors: list, r: np.ndarray, mesh: int):
    """q_ij and the augmentation charges r^2 Q_ij^L(r) by (i, j, L) of a file
    of version 1: one function r^2 Q_ij(r) of each pair for every L, inside
    rinner_L replaced, when nqf > 0, by r^(L+2) sum_k c_k r^(2k) with the
    coefficients of that pair and L."""
    count = len(projectors)
    rinner_block = re.search(r"<PP_RINNER>(.*?)</PP_RINNER>", body, re.S)
    head = body if rinner_block is None else body[: rinner_block.start()]
    nqf = int(_numbers(head)[0])
    rinner = np.zeros(0)
    if nqf > 0:
        if rinner_block is None:
            raise InvalidParameter(f"{name}: nqf > 0 without PP_RINNER")
        # pairs of the index of L + 1 and rinner_L
        rinner = _numbers(rinner_block.group(1)).reshape(-1, 2)[:, 1]
        body = body[rinner_block.end() :]
    q = np.zeros((count, count))
    augmentation = {}
    pairs = re.split(
        r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+i\s+j\s+\(l\(j\)\)[^\n]*$", body, flags=re.M
    )
    for k in range(1, len(pairs) - 1, 4):
        i, j = int(pairs[k]) - 1, int(pairs[k + 1]) - 1
        chunk = pairs[k + 3]
        coefficients = re.search(r"<PP_QFCOEF>(.*?)</PP_QFCOEF>", chunk, re.S)
        values = _numbers(
            chunk if coefficients is None else chunk[: coefficients.start()]
        )
        q[i, j] = q[j, i] = values[0]  # Q_int
        function = values[1 : 1 + mesh]
        if len(function) < mesh:
            raise InvalidParameter(f"{name}: the Q of {i + 1} {j + 1} is short")
        c = None
        if nqf > 0:
            if coefficients is None:
                raise InvalidParameter(f"{name}: no PP_QFCOEF of {i + 1} {j + 1}")
            c = _numbers(coefficients.group(1))[: nqf * len(rinner)].reshape(
                len(rinner), nqf
            )
        li, lj = projectors[i].angular, projectors[j].angular
        for ell in range(abs(li - lj), li + lj + 1, 2):
            charge = function.copy()
            if c is not None:
                inside = r < rinner[ell]
                x = r[inside]
                charge[inside] = x ** (ell + 2) * np.polyval(c[ell][::-1], x * x)
            augmentation[(min(i, j), max(i, j), ell)] = charge
    if not augmentation:
        raise InvalidParameter(f"{name}: no augmentation charges in PP_QIJ")
    return q, augmentation
