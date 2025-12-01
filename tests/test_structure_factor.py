import freud
import mdapy as mp
import numpy as np
import polars as pl


def test_sfc():
    for mode in ["direct", "debye"]:
        N_particles = 1000
        nbins = 50
        k_max = 10
        k_min = 0.1
        atol = 1e-4
        box, points = freud.data.make_random_system(10, N_particles, seed=0)
        system = mp.System(box=box.to_matrix(), pos=points)
        system.update_data(
            system.data.with_columns(
                pl.lit(
                    np.array([1] * (N_particles // 2) + [2] * (N_particles // 2))
                ).alias("type")
            )
        )
        sf1 = system.cal_structure_factor(
            k_min, k_max, nbins, cal_partial=True, mode=mode
        )

        A_points = points[: N_particles // 2]
        B_points = points[N_particles // 2 :]
        if mode == "direct":
            sf = freud.diffraction.StaticStructureFactorDirect(
                bins=nbins, k_max=k_max, k_min=k_min
            )
        else:
            sf = freud.diffraction.StaticStructureFactorDebye(nbins, k_max, k_min)
        sf.compute((box, A_points), A_points, N_particles)
        assert np.allclose(sf1.Sk_partial["1-1"], sf.S_k, atol=atol, equal_nan=True), (
            f"1-1 is different in {mode} mode"
        )
        sf.compute((box, A_points), B_points, N_particles)
        assert np.allclose(sf1.Sk_partial["1-2"], sf.S_k, atol=atol, equal_nan=True), (
            f"1-2 is different in {mode} mode"
        )
        sf.compute((box, B_points), B_points, N_particles)
        assert np.allclose(sf1.Sk_partial["2-2"], sf.S_k, atol=atol, equal_nan=True), (
            f"2-2 is different in {mode} mode"
        )
        sf.compute((box, points), points, N_particles)
        assert np.allclose(sf1.Sk, sf.S_k, atol=atol, equal_nan=True), (
            f"all is different in {mode} mode"
        )
        sf2 = system.cal_structure_factor(
            k_min, k_max, nbins, cal_partial=False, mode=mode
        )
        assert np.allclose(sf2.Sk, sf.S_k, atol=atol, equal_nan=True), (
            f"all is different in {mode} mode"
        )
