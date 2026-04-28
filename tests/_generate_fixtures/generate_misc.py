# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
Reference fixtures for the remaining structure-analysis tests that
don't fit the per-config scheme used by the structure_analysis bundle.

Each fixture is a single .npz file in tests/fixtures/misc/ that holds
the input data (or pointer to a sample input file) plus the reference
output we want to compare against.

Algorithms covered:

    atomic_strain          ref + cur → per-atom shear / volumetric strain
    wigner_seitz_defect    ref + cur → site occupancy, vacancy / interstitial counts
    rdf                    AlCrNi.xyz → partial RDFs from OVITO
    adf                    water.xyz → partial bond-angle distributions from OVITO
    bond_analysis          water.xyz → bond-length + bond-angle histograms from OVITO
    structure_factor       reproducible random box → S(k) (direct + debye) from freud
    msd                    reproducible random walk → MSD (window + direct) from freud
    structure_entropy      4 box geometries × 3 modes → per-atom entropy from OVITO
    average_neighbor       2 box geometries → per-atom averaged property from OVITO
    fcc_planar_faults      ISF.dump → per-atom planar-fault label from OVITO

Run manually whenever any algorithm or its parameters change:

    python tests/_generate_fixtures/generate_misc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from configs import INPUT_DIR        # noqa: E402

import mdapy as mp                    # noqa: E402
import freud                          # noqa: E402

from ovito.io import import_file       # noqa: E402
from ovito.modifiers import (          # noqa: E402
    AtomicStrainModifier,
    BondAnalysisModifier,
    CoordinationAnalysisModifier,
    CreateBondsModifier,
    ComputePropertyModifier,
    IdentifyFCCPlanarFaultsModifier,
    PolyhedralTemplateMatchingModifier,
    WignerSeitzAnalysisModifier,
)
from ovito.pipeline import (             # noqa: E402
    ReferenceConfigurationModifier,
    ModifierInterface,
)
from ovito.data import (                 # noqa: E402
    CutoffNeighborFinder,
    DataCollection,
)


OUT_DIR = HERE.parent / "fixtures" / "misc"


# ===========================================================================
# atomic_strain
# ===========================================================================

def gen_atomic_strain():
    ref_path = INPUT_DIR / "strain.0.xyz"
    cur_path = INPUT_DIR / "strain.1.xyz"

    pipeline = import_file(str(INPUT_DIR / "strain.*.xyz"))
    pipeline.modifiers.append(AtomicStrainModifier(cutoff=3.0))
    data_plain = pipeline.compute(1)

    pipeline = import_file(str(INPUT_DIR / "strain.*.xyz"))
    pipeline.modifiers.append(AtomicStrainModifier(
        cutoff=3.0,
        affine_mapping=ReferenceConfigurationModifier.AffineMapping.ToReference,
        reference_frame=0,
    ))
    data_affine = pipeline.compute(1)

    np.savez_compressed(
        OUT_DIR / "atomic_strain.npz",
        cutoff=np.float64(3.0),
        ref_filename=str(ref_path.relative_to(INPUT_DIR.parent)),
        cur_filename=str(cur_path.relative_to(INPUT_DIR.parent)),
        shear_strain=np.asarray(data_plain.particles["Shear Strain"][...], dtype=np.float64),
        volumetric_strain=np.asarray(data_plain.particles["Volumetric Strain"][...], dtype=np.float64),
        shear_strain_affine=np.asarray(data_affine.particles["Shear Strain"][...], dtype=np.float64),
        volumetric_strain_affine=np.asarray(data_affine.particles["Volumetric Strain"][...], dtype=np.float64),
    )


# ===========================================================================
# wigner_seitz_defect
# ===========================================================================

def gen_wigner_seitz():
    pipeline = import_file(str(INPUT_DIR / "hea.*.xyz"))
    pipeline.modifiers.append(WignerSeitzAnalysisModifier(
        affine_mapping=ReferenceConfigurationModifier.AffineMapping.ToReference,
    ))
    data = pipeline.compute(1)

    pipeline2 = import_file(str(INPUT_DIR / "hea.*.xyz"))
    pipeline2.modifiers.append(WignerSeitzAnalysisModifier(
        affine_mapping=ReferenceConfigurationModifier.AffineMapping.ToReference,
        output_displaced=True,
    ))
    data2 = pipeline2.compute(1)

    type2element = {t.id: t.name for t in data2.particles.particle_type.types}
    site_type = np.asarray(data2.particles["Site Type"][...], dtype=np.int32)
    site_type_str = np.array([type2element[i] for i in site_type], dtype="<U8")

    np.savez_compressed(
        OUT_DIR / "wigner_seitz.npz",
        ref_filename="input_files/hea.0.xyz",
        cur_filename="input_files/hea.1.xyz",
        vacancy_count=np.int32(int(data.attributes["WignerSeitz.vacancy_count"])),
        interstitial_count=np.int32(int(data.attributes["WignerSeitz.interstitial_count"])),
        site_occupancy=np.asarray(data.particles["Occupancy"][...], dtype=np.int32),
        atom_occupancy=np.asarray(data2.particles["Occupancy"][...], dtype=np.int32),
        atom_site_index=np.asarray(data2.particles["Site Index"][...], dtype=np.int32),
        atom_site_type=site_type_str,
    )


# ===========================================================================
# Radial distribution function (partial RDFs on AlCrNi)
# ===========================================================================

def gen_rdf():
    system = mp.System(str(INPUT_DIR / "AlCrNi.xyz"))
    elements = list(system.data["element"].unique().sort())

    ovi = system.to_ovito()
    ovi.apply(CoordinationAnalysisModifier(cutoff=5.0, number_of_bins=50, partial=True))
    rdf_table = ovi.tables["coordination-rdf"]
    names = list(rdf_table.y.component_names)

    # Map every (i, j) pair to the matching column in OVITO's table.
    K = len(elements)
    g = np.zeros((K, K, 50), dtype=np.float64)
    for i in range(K):
        for j in range(i, K):
            name = f"{elements[i]}-{elements[j]}"
            if name not in names:
                name = f"{elements[j]}-{elements[i]}"
            g[i, j] = rdf_table.y[:, names.index(name)]

    np.savez_compressed(
        OUT_DIR / "rdf.npz",
        input_filename="input_files/AlCrNi.xyz",
        cutoff=np.float64(5.0),
        nbins=np.int32(50),
        elements=np.array(elements, dtype="<U8"),
        g=g,
    )


# ===========================================================================
# Bond analysis (water.xyz: bond-length + bond-angle histograms)
# ===========================================================================

def gen_bond_analysis():
    system = mp.System(str(INPUT_DIR / "water.xyz"))
    ovi = system.to_ovito()
    ovi.apply(CreateBondsModifier(cutoff=2.0))
    ovi.apply(BondAnalysisModifier(bins=40, length_cutoff=2.0))
    length = ovi.tables["bond-length-distr"].xy()
    angle = ovi.tables["bond-angle-distr"].xy()

    np.savez_compressed(
        OUT_DIR / "bond_analysis.npz",
        input_filename="input_files/water.xyz",
        cutoff=np.float64(2.0),
        bins=np.int32(40),
        max_neigh=np.int32(10),
        r_length=length[:, 0].astype(np.float64),
        bond_length_distribution=length[:, 1].astype(np.float64),
        r_angle=angle[:, 0].astype(np.float64),
        bond_angle_distribution=angle[:, 1].astype(np.float64),
    )


# ===========================================================================
# Angular distribution function (water.xyz: per-element-triplet histograms)
# ===========================================================================

def gen_adf():
    system = mp.System(str(INPUT_DIR / "water.xyz"))
    ovi = system.to_ovito()
    ovi.apply(CreateBondsModifier(cutoff=2.0))
    ovi.apply(BondAnalysisModifier(
        bins=40,
        length_cutoff=2.0,
        partition=BondAnalysisModifier.Partition.ByParticleType,
    ))

    histogram = ovi.tables["bond-angle-distr"].y
    components = list(histogram.component_names)
    # We store the histograms keyed by their component name; the test
    # already knows which mdapy index corresponds to each name.
    arrays = {f"adf_{name.replace('-', '_')}": histogram[:, components.index(name)]
              for name in components}

    np.savez_compressed(
        OUT_DIR / "adf.npz",
        input_filename="input_files/water.xyz",
        cutoff=np.float64(2.0),
        bins=np.int32(40),
        component_names=np.array(components, dtype="<U16"),
        **arrays,
    )


# ===========================================================================
# Structure factor (reproducible random box, both modes)
# ===========================================================================

def gen_structure_factor():
    N = 1000
    nbins = 50
    k_min, k_max = 0.1, 10.0

    box, points = freud.data.make_random_system(10, N, seed=0)
    A_points = points[: N // 2]
    B_points = points[N // 2 :]

    out = {
        "N": np.int32(N),
        "nbins": np.int32(nbins),
        "k_min": np.float64(k_min),
        "k_max": np.float64(k_max),
        "box": np.asarray(box.to_matrix(), dtype=np.float64),
        "points": np.asarray(points, dtype=np.float64),
    }

    for mode in ("direct", "debye"):
        if mode == "direct":
            sf = freud.diffraction.StaticStructureFactorDirect(
                bins=nbins, k_max=k_max, k_min=k_min)
        else:
            sf = freud.diffraction.StaticStructureFactorDebye(nbins, k_max, k_min)

        sf.compute((box, A_points), A_points, N)
        out[f"{mode}_11"] = sf.S_k.astype(np.float64).copy()
        sf.compute((box, A_points), B_points, N)
        out[f"{mode}_12"] = sf.S_k.astype(np.float64).copy()
        sf.compute((box, B_points), B_points, N)
        out[f"{mode}_22"] = sf.S_k.astype(np.float64).copy()
        sf.compute((box, points), points, N)
        out[f"{mode}_all"] = sf.S_k.astype(np.float64).copy()

    np.savez_compressed(OUT_DIR / "structure_factor.npz", **out)


# ===========================================================================
# Mean-squared displacement (reproducible random walk; both modes)
# ===========================================================================

def gen_msd():
    """Don't bundle the trajectory — the test reconstructs it from the same
    seed (legacy `np.random.seed(1)` + `np.cumsum(randn(...))`). Only the
    reference MSD curves are stored."""
    Nframe, Nparticles = 200, 1000
    np.random.seed(1)
    pos_list = np.cumsum(np.random.randn(Nframe, Nparticles, 3), axis=0)
    out = {
        "Nframe": np.int32(Nframe),
        "Nparticles": np.int32(Nparticles),
        "seed": np.int32(1),
    }
    for mode in ("window", "direct"):
        f = freud.msd.MSD(mode=mode)
        f.compute(pos_list)
        out[f"msd_{mode}"] = np.asarray(f.msd, dtype=np.float64)
    np.savez_compressed(OUT_DIR / "msd.npz", **out)


# ===========================================================================
# Structure entropy (4 box geometries × 3 modes — per-atom entropy)
# ===========================================================================

class _Entropy(ModifierInterface):
    """OVITO ModifierInterface implementation copied from the original test
    (see git history). Computes a per-atom local entropy."""
    cutoff = 5.0
    sigma = 0.2
    use_local_density = False
    compute_average = False
    average_cutoff = 4.0

    def modify(self, data: DataCollection, **kwargs):
        cutoff = self.cutoff
        sigma = self.sigma
        use_local_density = self.use_local_density
        compute_average = self.compute_average
        average_cutoff = self.average_cutoff
        try:
            trapz = np.trapezoid
        except AttributeError:
            trapz = np.trapz
        global_rho = data.particles.count / data.cell.volume
        finder = CutoffNeighborFinder(cutoff, data)
        local_entropy = np.empty(data.particles.count)
        nbins = int(cutoff / sigma) + 1
        r = np.linspace(0.0, cutoff, num=nbins)
        rsq = r ** 2
        prefactor = rsq * (4 * np.pi * global_rho * np.sqrt(2 * np.pi * sigma ** 2))
        prefactor[0] = prefactor[1]
        for particle_index in range(data.particles.count):
            r_ij = finder.neighbor_distances(particle_index)
            r_diff = np.expand_dims(r, 0) - np.expand_dims(r_ij, 1)
            g_m = np.sum(np.exp(-(r_diff ** 2) / (2.0 * sigma ** 2)), axis=0) / prefactor
            if use_local_density:
                local_volume = 4 / 3 * np.pi * cutoff ** 3
                rho = len(r_ij) / local_volume
                g_m *= global_rho / rho
            else:
                rho = global_rho
            integrand = np.where(g_m >= 1e-10, (g_m * np.log(g_m) - g_m + 1.0) * rsq, rsq)
            local_entropy[particle_index] = -2.0 * np.pi * rho * trapz(integrand, r)
        data.particles_.create_property("Entropy", data=local_entropy)
        if compute_average:
            data.apply(ComputePropertyModifier(
                output_property="Entropy",
                operate_on="particles",
                cutoff_radius=average_cutoff,
                expressions=["Entropy / (NumNeighbors + 1)"],
                neighbor_expressions=["Entropy / (NumNeighbors + 1)"],
            ))


_ENTROPY_CONFIGS = ["rec_box_big.xyz", "rec_box_small.xyz",
                    "tri_box_big.xyz", "tri_box_small.xyz"]


def gen_structure_entropy():
    out = {}
    for fname in _ENTROPY_CONFIGS:
        for mode in ("default", "use_local_density", "compute_average"):
            modifier = _Entropy()
            if mode == "use_local_density":
                modifier.use_local_density = True
            if mode == "compute_average":
                modifier.compute_average = True
            pipeline = import_file(str(INPUT_DIR / fname))
            pipeline.modifiers.append(modifier)
            data = pipeline.compute()
            key = f"{fname.replace('.xyz', '')}__{mode}"
            out[key] = np.asarray(data.particles["Entropy"][...], dtype=np.float64)
    np.savez_compressed(OUT_DIR / "structure_entropy.npz", **out)


# ===========================================================================
# Average-neighbor (per-atom averaged property)
# ===========================================================================

_AVG_CONFIGS = [("rec_box_big.xyz", 4.0), ("tri_box_big.xyz", 4.0)]


def gen_average_neighbor():
    out = {}
    for fname, rc in _AVG_CONFIGS:
        pipeline = import_file(str(INPUT_DIR / fname))
        modifier = ComputePropertyModifier(
            output_property="x_ave",
            operate_on="particles",
            cutoff_radius=rc,
            expressions=["Position.X / (NumNeighbors + 1)"],
            neighbor_expressions=["Position.X / (NumNeighbors + 1)"],
        )
        pipeline.modifiers.append(modifier)
        data = pipeline.compute()
        key = fname.replace(".xyz", "")
        out[f"{key}__cutoff"] = np.float64(rc)
        out[f"{key}__x_ave"] = np.asarray(data.particles["x_ave"][...], dtype=np.float64)
    np.savez_compressed(OUT_DIR / "average_neighbor.npz", **out)


# ===========================================================================
# FCC planar faults (per-atom planar-fault label on ISF.dump)
# ===========================================================================

def gen_fcc_planar_faults():
    pipeline = import_file(str(INPUT_DIR / "ISF.dump"))
    pipeline.modifiers.append(PolyhedralTemplateMatchingModifier(
        output_orientation=True, output_interatomic_distance=True))
    pipeline.modifiers.append(IdentifyFCCPlanarFaultsModifier())
    data = pipeline.compute()
    np.savez_compressed(
        OUT_DIR / "fcc_planar_faults.npz",
        input_filename="input_files/ISF.dump",
        pft=np.asarray(data.particles["Planar Fault Type"][...], dtype=np.int32),
    )


# ---------------------------------------------------------------------------

GENERATORS = [
    ("atomic_strain",       gen_atomic_strain),
    ("wigner_seitz",        gen_wigner_seitz),
    ("rdf",                 gen_rdf),
    ("adf",                 gen_adf),
    ("bond_analysis",       gen_bond_analysis),
    ("structure_factor",    gen_structure_factor),
    ("msd",                 gen_msd),
    ("structure_entropy",   gen_structure_entropy),
    ("average_neighbor",    gen_average_neighbor),
    ("fcc_planar_faults",   gen_fcc_planar_faults),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in OUT_DIR.glob("*.npz"):
        f.unlink()
    print(f"Writing fixtures to {OUT_DIR.relative_to(HERE.parent.parent)}/\n")
    for name, fn in GENERATORS:
        try:
            fn()
            size_kb = (OUT_DIR / f"{name}.npz").stat().st_size / 1024
            print(f"  {name:<22s}  {size_kb:7.1f} KB")
        except Exception as e:
            print(f"  ! {name}: {type(e).__name__}: {e}")
            raise
    total_kb = sum(p.stat().st_size for p in OUT_DIR.glob("*.npz")) / 1024
    print(f"\n  {'TOTAL':<22s}  {total_kb:7.1f} KB ({len(GENERATORS)} files)")


if __name__ == "__main__":
    main()
