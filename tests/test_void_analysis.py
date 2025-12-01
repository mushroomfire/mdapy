from mdapy import build_crystal, VoidAnalysis
import polars as pl


def test_void():
    fcc = build_crystal("Al", "fcc", 4.05, nx=50, ny=50, nz=50)
    void1 = (pl.col("x") - 50) ** 2 + (pl.col("y") - 50) ** 2 + (
        pl.col("z") - 50
    ) ** 2 > 100
    void2 = (pl.col("x") - 100) ** 2 + (pl.col("y") - 100) ** 2 + (
        pl.col("z") - 100
    ) ** 2 > 100
    void3 = (pl.col("x") - 150) ** 2 + (pl.col("y") - 150) ** 2 + (
        pl.col("z") - 150
    ) ** 2 > 400

    fcc.update_data(fcc.data.filter(void1 & void2 & void3))

    void = VoidAnalysis(fcc, 4.1)
    void.compute()
    assert void.void_number == 3, "void number is wrong."
