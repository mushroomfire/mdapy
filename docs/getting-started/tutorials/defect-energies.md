# Defect energies of copper

The energies of vacancies, surfaces and stacking faults are the first
check of a potential for a metal. This page computes them for copper with
the EAM potential of Zhou et al., and checks that each has converged with
the size of the cell. See [Defect energies](../../simulation/defect-energies.md)
for the functions themselves.

## The lattice constant

Every cell below is built with the lattice constant of the potential, not
the measured one: relax the cubic cell of fcc copper at zero pressure.

```python
import numpy as np

from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.simulation import minimize

copper = EAM.zhou(["Cu"])
cell = build_crystal("fcc", 3.6, elements="Cu")
relaxed = minimize(cell, copper, cell="isotropic", max_stress=1e-6).system
a = relaxed.box.matrix[0, 0]
round(a, 4)  # 3.615 Å
```

## Vacancies

A vacancy disturbs the atoms around it, and in a small periodic cell it
also meets its own images. Compute the formation energy in cells of 2 to 5
cubic cells along each side, until it no longer changes:

```python
from mdapy.simulation import vacancy_formation_energy

for n in (2, 3, 4, 5):
    crystal = build_crystal("fcc", a, elements="Cu", repeat=(n, n, n))
    energy = vacancy_formation_energy(crystal, copper).energy
    print(f"{len(crystal):4d} atoms: {energy:.3f} eV")
```

```text
  32 atoms: 1.301 eV
 108 atoms: 1.279 eV
 256 atoms: 1.275 eV
 500 atoms: 1.273 eV
```

256 atoms are converged to some meV; the measured value is about 1.28 eV.

## Surfaces

For each face, orient the crystal with the face normal to z, about 24 Å
thick and 8 Å wide, and open it with vacuum along z:

```python
from mdapy.simulation import surface_energy

faces = {
    "(100)": [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    "(110)": [(0, 0, 1), (1, -1, 0), (1, 1, 0)],
    "(111)": [(1, 1, -2), (-1, 1, 0), (1, 1, 1)],
}
for face, orientation in faces.items():
    unit = build_crystal("fcc", a, elements="Cu", orientation=orientation)
    lengths = np.linalg.norm(unit.box.matrix, axis=1)
    repeat = np.maximum(1, np.round([8.0, 8.0, 24.0] / lengths)).astype(int)
    slab = build_crystal(
        "fcc", a, elements="Cu", orientation=orientation, repeat=tuple(repeat)
    )
    print(f"{face}: {surface_energy(slab, copper).energy:.3f} J/m²")
```

```text
(100): 1.563 J/m²
(110): 1.743 J/m²
(111): 1.501 J/m²
```

The close-packed (111) face costs least and the open (110) face most, as
for every fcc metal.

## Stacking faults

The generalized stacking fault energy is the energy of slipping the upper
half of a crystal over the lower half on a (111) plane. Along [11-2] it has
a local minimum at a third of the period, a/6 [11-2], the intrinsic
stacking fault, behind the unstable stacking fault energy.

The atoms relax only along the normal of the fault, and how much room they
have for that depends on the cell. Compare the intrinsic stacking fault in
crystals periodic along the normal and in slabs between two free surfaces,
whose energy cancels in the difference, for 9 to 36 (111) planes:

```python
import mdapy
from mdapy.simulation import stacking_fault_energy

partial_period = [a * np.sqrt(6) / 2, 0, 0]  # [11-2]
for n in (3, 6, 9, 12):
    periodic = build_crystal(
        "fcc", a, elements="Cu", orientation=faces["(111)"], repeat=(1, 2, n)
    )
    slab = periodic.with_box(mdapy.Box(periodic.box.matrix, pbc=[True, True, False]))
    energies = [
        stacking_fault_energy(
            crystal, copper, partial_period, points=4, max_force=1e-6
        ).energy[1]
        * 1000
        for crystal in (periodic, slab)
    ]
    print(f"{3 * n:2d} planes: {energies[0]:5.2f} and {energies[1]:5.2f} mJ/m²")
```

```text
 9 planes: 27.27 and 22.62 mJ/m²
18 planes: 24.46 and 22.46 mJ/m²
27 planes: 23.32 and 22.46 mJ/m²
36 planes: 22.60 and 22.46 mJ/m²
```

Between free surfaces the energy has converged at 18 planes. In a periodic
crystal the height of the box stays as it is: the planes next to the fault
move apart and squeeze the others, which costs an energy that fades only
slowly with the thickness. Use slabs, and a tight `max_force`: with the
default of 1e-4 eV/Å the energies scatter by some tenths of a mJ/m².

Then slip along both directions, with 24 steps per period:

```python
layers = build_crystal(
    "fcc", a, elements="Cu", orientation=faces["(111)"], repeat=(1, 2, 6)
)
slab = layers.with_box(mdapy.Box(layers.box.matrix, pbc=[True, True, False]))
partial = stacking_fault_energy(slab, copper, partial_period, points=25, max_force=1e-6)
full = stacking_fault_energy(
    slab, copper, [0, a * np.sqrt(2) / 2, 0], points=25, max_force=1e-6
)
intrinsic = partial.energy[8] * 1000  # mJ/m², at 8/24 of the period
unstable = partial.energy[:9].max() * 1000
round(intrinsic, 1), round(unstable, 1)  # (22.5, 110.5)
```

The measured intrinsic stacking fault energy of copper is about 40 to
45 mJ/m²; this potential gives about half of it, which matters for
anything that involves partial dislocations.

## The curves

```python
from mdapy.visualization import plot

with plot.style():
    fig, ax = plot.figure("single")
    ax.plot(partial.fraction, partial.energy * 1000, label="[11-2]")
    ax.plot(full.fraction, full.energy * 1000, label="[-110]")
    ax.set_xlabel("slip / period")
    ax.set_ylabel("energy (mJ/m²)")
    ax.legend(frameon=False)
    plot.save(fig, "gsfe", formats=("png",))
```

![Generalized stacking fault energy of copper on (111)](images/gsfe.png)
