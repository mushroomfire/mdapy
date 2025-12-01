from mdapy import System
from mdapy.minimizer import FIRE
from mdapy.nep import NEP
from mdapy.nep4ase import NEP4ASE
from ase.io import read
from ase.optimize import FIRE2
from ase.filters import UnitCellFilter
import numpy as np


def _test_minimize_full(
    use_abc: bool = False,
    optimize_cell: bool = False,
    mask=None,
    hydrostatic_strain: bool = False,
    constant_volume: bool = False,
    scalar_pressure: float = 0,
):
    atoms = read("input_files/AlCrNi.xyz", format="extxyz")
    system = System(ase_atom=atoms)
    system.calc = NEP("input_files/UNEP-v1.txt")
    atoms.calc = NEP4ASE("input_files/UNEP-v1.txt")
    if optimize_cell:
        aa = UnitCellFilter(
            atoms,
            mask=mask,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
            scalar_pressure=scalar_pressure,
        )
    else:
        aa = atoms
    fy = FIRE2(aa, use_abc=use_abc)
    steps = 5
    fy.run(steps=steps)

    fire = FIRE(
        system,
        use_abc=use_abc,
        optimize_cell=optimize_cell,
        mask=mask,
        hydrostatic_strain=hydrostatic_strain,
        constant_volume=constant_volume,
        scalar_pressure=scalar_pressure,
    )
    fire.run(steps=steps)
    assert np.allclose(atoms.get_stress(), system.get_stress()), "stress is wrong."
    assert np.allclose(atoms.get_forces(), system.get_force()), "force is wrong"
    assert np.allclose(atoms.get_potential_energies(), system.get_energies()), (
        "energy is wrong"
    )


def test_energy_mini():
    _test_minimize_full()
    print("passed 1")
    _test_minimize_full(use_abc=True)
    print("passed 2")
    _test_minimize_full(optimize_cell=True)
    print("passed 3")
    _test_minimize_full(optimize_cell=True, use_abc=True)
    print("passed 4")
    _test_minimize_full(optimize_cell=True, mask=[1, 0, 0, 0, 0, 0])
    print("passed 5")
    _test_minimize_full(optimize_cell=True, hydrostatic_strain=True)
    print("passed 6")
    _test_minimize_full(optimize_cell=True, constant_volume=True)
    print("passed 7")
    _test_minimize_full(optimize_cell=True, scalar_pressure=1)
    print("passed 8")
