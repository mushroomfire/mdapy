# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
Refresh reference fixtures for the per-atom structure-analysis algorithms.

A single .npz file per configuration holds the wrapped positions, the
box matrix, the boundary array, AND every algorithm's reference output
(plus its parameters). Algorithms that don't apply to a given config
just leave their keys out — tests skip when a key is absent.

Algorithms covered:

    csp   — centro-symmetry parameter
    cna   — common neighbor analysis (cutoff-based)
    aja   — Ackland-Jones analysis
    ptm   — polyhedral template matching
    ids   — identify diamond structure
    cnp   — common neighbor parameter
    qlm   — Steinhardt bond-orientation parameters Q_4 / Q_6 (plus avg)

Run manually whenever the reference set or its parameters change:

    python tests/_generate_fixtures/generate_structure_analysis.py

OVITO / freud are imported here but never at test time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from configs import CONFIGS, LATTICE_A    # noqa: E402

from ovito.modifiers import (              # noqa: E402
    CentroSymmetryModifier,
    CommonNeighborAnalysisModifier,
    AcklandJonesModifier,
    PolyhedralTemplateMatchingModifier,
    IdentifyDiamondModifier,
    VoronoiAnalysisModifier,
)
import freud                                # noqa: E402

import mdapy as mp                          # noqa: E402


FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "structure_analysis"


# --- per-algorithm config knobs --------------------------------------------

CSP_K = {"perfect_bcc": 8, "rattled_bcc": 8, "perfect_diamond": 4}
CSP_DEFAULT_K = 12
CSP_SKIP = {"slab_fcc", "wire_fcc", "random_atoms"}

def _cna_cutoff(name: str) -> float | None:
    a = LATTICE_A.get(name)
    if a is None: return None
    if "fcc" in name or "diamond" in name: return 0.854 * a
    if "bcc" in name: return 1.21 * a
    if "hcp" in name: return 1.207 * a
    return 0.854 * a

CNA_SKIP = {"random_atoms"}
CNP_SKIP = {"random_atoms", "wire_fcc", "slab_fcc"}

def _ql_cutoff(name: str) -> float | None:
    a = LATTICE_A.get(name)
    if a is None: return None
    if "fcc" in name or "diamond" in name: return 0.85 * a
    if "bcc" in name: return 0.95 * a
    if "hcp" in name: return 1.05 * a
    return 0.85 * a

QLM_SKIP = {
    "slab_fcc", "wire_fcc",          # open boundaries
    "random_atoms",                  # degenerate neighbor counts
    "highly_tilted_fcc",             # extreme shear, MIC ties
    "sheared_fcc",                   # mild shear: freud per-atom floor diffs
    "perfect_hcp",                   # 120° primitive: freud edge-atom Qℓ
                                     #   diverges from the analytic ideal
                                     #   (7/72 for Q_4) that mdapy returns
}


# --- per-algorithm reference computation -----------------------------------

def _csp(name, system):
    if name in CSP_SKIP: return None
    k = CSP_K.get(name, CSP_DEFAULT_K)
    ovi = system.to_ovito()
    ovi.apply(CentroSymmetryModifier(num_neighbors=k))
    return {
        "csp": np.asarray(ovi.particles["Centrosymmetry"][...], dtype=np.float64),
        "csp_num_neighbors": np.int32(k),
    }

def _cna(name, system):
    if name in CNA_SKIP: return None
    rc = _cna_cutoff(name)
    ovi = system.to_ovito()
    ovi.apply(CommonNeighborAnalysisModifier(
        mode=CommonNeighborAnalysisModifier.Mode.FixedCutoff, cutoff=rc))
    return {
        "cna": np.asarray(ovi.particles["Structure Type"][...], dtype=np.int32),
        "cna_cutoff": np.float64(rc),
    }

def _aja(name, system):
    ovi = system.to_ovito()
    ovi.apply(AcklandJonesModifier())
    return {"aja": np.asarray(ovi.particles["Structure Type"][...], dtype=np.int32)}

def _ptm(name, system):
    ovi = system.to_ovito()
    ovi.apply(PolyhedralTemplateMatchingModifier())
    return {"ptm": np.asarray(ovi.particles["Structure Type"][...], dtype=np.int32)}

def _ids(name, system):
    ovi = system.to_ovito()
    ovi.apply(IdentifyDiamondModifier())
    return {"ids": np.asarray(ovi.particles["Structure Type"][...], dtype=np.int32)}

def _cnp(name, system):
    """No OVITO equivalent; we anchor to mdapy's own current output as a
    regression guard. The perfect-crystal invariant tests in the test
    file provide the absolute correctness baseline."""
    if name in CNP_SKIP: return None
    rc = _cna_cutoff(name)
    s2 = system
    s2.cal_common_neighbor_parameter(rc)
    return {
        "cnp": s2.data["cnp"].to_numpy(allow_copy=False).astype(np.float64).copy(),
        "cnp_cutoff": np.float64(rc),
    }

def _voronoi(name, system):
    """Voronoi cell volume + cavity radius + coordination per atom."""
    ovi = system.to_ovito()
    ovi.apply(VoronoiAnalysisModifier())
    return {
        "voronoi_volume": np.asarray(ovi.particles["Atomic Volume"][...], dtype=np.float64),
        # OVITO's Cavity Radius is half of mdapy's "cavity_radius" convention;
        # store OVITO's value directly and the test multiplies by 2 on load.
        "voronoi_cavity_radius": np.asarray(ovi.particles["Cavity Radius"][...], dtype=np.float64),
        "voronoi_coord": np.asarray(ovi.particles["Coordination"][...], dtype=np.int32),
    }


def _ql(name, system):
    if name in QLM_SKIP: return None
    rc = _ql_cutoff(name)
    if rc is None: return None
    pos = system.data.select(["x", "y", "z"]).to_numpy()
    box = freud.box.Box.from_matrix(system.box.box)
    fsys = (box, pos)
    out = {"ql_cutoff": np.float64(rc)}
    for avg in (False, True):
        ql = freud.order.Steinhardt(l=[4, 6], average=avg)
        ql.compute(fsys, {"r_max": rc})
        for j, l in enumerate([4, 6]):
            tag = f"q{l}_avg" if avg else f"q{l}"
            out[tag] = np.asarray(ql.particle_order[:, j], dtype=np.float64)
    return out


ALGORITHMS = [
    ("csp", _csp), ("cna", _cna), ("aja", _aja), ("ptm", _ptm),
    ("ids", _ids), ("cnp", _cnp), ("qlm", _ql), ("vor", _voronoi),
]


def main():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    # Wipe stale fixtures.
    for f in FIXTURE_DIR.glob("*.npz"):
        f.unlink()

    print(f"Writing {len(CONFIGS)} fixtures to "
          f"{FIXTURE_DIR.relative_to(HERE.parent.parent)}/\n")

    header = f"{'config':<20s}  {'N':>5s}  " + "  ".join(
        f"{algo:>4s}" for algo, _ in ALGORITHMS)
    print(header)
    print("-" * len(header))

    for name, factory in CONFIGS.items():
        system = factory()
        N = system.N
        bundle = {
            "pos": system.data.select(["x", "y", "z"]).to_numpy().astype(np.float64),
            "box": system.box.box.copy().astype(np.float64),
            "boundary": np.asarray(system.box.boundary, dtype=np.int32),
        }
        line = f"{name:<20s}  {N:5d}  "
        for algo, gen in ALGORITHMS:
            try:
                res = gen(name, system)
            except Exception as e:
                print(f"\n  ! {name} / {algo}: {e}\n  ", end="")
                line += "FAIL  "
                continue
            if res is None:
                line += "skip  "
            else:
                bundle.update(res)
                line += " ok   "
        np.savez_compressed(FIXTURE_DIR / f"{name}.npz", **bundle)
        print(line)

    total_kb = sum(p.stat().st_size for p in FIXTURE_DIR.glob("*.npz")) / 1024
    print(f"\nTotal: {total_kb:.1f} KB across {len(CONFIGS)} files")


if __name__ == "__main__":
    main()
