from ase.build import bulk
from ase.lattice.cubic import FaceCenteredCubic, BodyCenteredCubic, Diamond
from mdapy import build_crystal
import numpy as np


def test_bulk():
    Cu_ase = bulk("Cu", "fcc", a=3.615, cubic=True)
    box = np.array(Cu_ase.get_cell())
    Cu = build_crystal("Cu", "fcc", a=3.615)
    assert np.allclose(box, Cu.box.box), "FCC bulk box is wrong."

    assert np.allclose(
        Cu_ase.get_positions(), Cu.data.select("x", "y", "z").to_numpy()
    ), "FCC bulk pos is wrong."

    Fe_ase = bulk("Fe", "bcc", a=2.83, cubic=True)
    box = np.array(Fe_ase.get_cell())
    Fe = build_crystal("Fe", "bcc", a=2.83)
    assert np.allclose(box, Fe.box.box), "BCC bulk box is wrong."

    assert np.allclose(
        Fe_ase.get_positions(), Fe.data.select("x", "y", "z").to_numpy()
    ), "BCC bulk pos is wrong."

    C_ase = bulk("C", "diamond", a=3.6, cubic=True)
    box = np.array(C_ase.get_cell())
    C = build_crystal("C", "diamond", a=3.6)
    assert np.allclose(box, C.box.box), "Diamond bulk box is wrong."

    assert np.allclose(
        C_ase.get_positions(), C.data.select("x", "y", "z").to_numpy()
    ), "Diamond bulk pos is wrong."


def test_miller():
    Cu_ase = FaceCenteredCubic(
        miller=[[1, -1, 0], [1, 1, -2], [1, 1, 1]], symbol="Cu", latticeconstant=3.615
    )
    box = np.array(Cu_ase.get_cell())
    Cu = build_crystal(
        "Cu", "fcc", a=3.615, miller1=[1, -1, 0], miller2=[1, 1, -2], miller3=[1, 1, 1]
    )

    assert np.allclose(box, Cu.box.box), "FCC bulk miller box is wrong."
    assert len(Cu_ase) == Cu.N, "FCC bulk miller pos is wrong."

    Fe_ase = BodyCenteredCubic(
        directions=[[1, 2, 1], [-1, 0, 1], [1, -1, 1]],
        symbol="Fe",
        latticeconstant=2.83,
    )
    box = np.array(Fe_ase.get_cell())
    Fe = build_crystal(
        "Fe", "bcc", a=2.83, miller1=[1, 2, 1], miller2=[-1, 0, 1], miller3=[1, -1, 1]
    )

    assert np.allclose(box, Fe.box.box), "BCC bulk miller box is wrong."
    assert len(Fe_ase) == Fe.N, "BCC bulk miller pos is wrong."

    C_ase = Diamond(
        directions=[[1, 2, 1], [-1, 0, 1], [1, -1, 1]], symbol="C", latticeconstant=3.6
    )
    box = np.array(C_ase.get_cell())
    C = build_crystal(
        "C", "diamond", a=3.6, miller1=[1, 2, 1], miller2=[-1, 0, 1], miller3=[1, -1, 1]
    )

    assert np.allclose(box, C.box.box), "Diamond bulk miller box is wrong."
    assert len(C_ase) == C.N, "Diamond bulk miller pos is wrong."
