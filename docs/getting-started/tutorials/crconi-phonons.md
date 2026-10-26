# Phonons of Cr, Co and Ni with a machine-learned potential

A machine-learned potential is checked against the first-principles
calculations it learned from. This page computes the phonon dispersions of
bcc Cr, hcp Co and fcc Ni with the NEP model for CrCoNi alloys of
[Wu et al. (2026)](../../reference/references.md#wu2026) and draws them over
the DFT dispersions (VASP and phonopy) of the same paper. See
[Phonons](../../simulation/phonons.md) for the functions themselves.

!!! note "Cite the model and the data"

    The NEP model and the DFT dispersions come from
    [Wu et al. (2026)](../../reference/references.md#wu2026). Cite that
    paper when you use them.

<!-- files: crconi -->

Download the model [`nep.txt`](crconi/nep.txt) (a copy of the one in the
[NEP-CrCoNi repository](https://github.com/mushroomfire/NEP-CrCoNi), which
also holds the training data) and the DFT dispersions
[`Cr.band.dat`](crconi/Cr.band.dat), [`Co.band.dat`](crconi/Co.band.dat) and
[`Ni.band.dat`](crconi/Ni.band.dat) into the working directory.

## The crystals

Relax the conventional cell of each metal at zero pressure with the model,
and reduce it to its primitive cell:

```python
import numpy as np

from mdapy.build import build_crystal
from mdapy.calculators import NEP
from mdapy.lattice_dynamics import phonons, primitive_cell
from mdapy.simulation import minimize

nep = NEP("nep.txt")


def relaxed(element, structure, a):
    """The conventional cell of a metal, relaxed at zero pressure."""
    start = build_crystal(structure, a, elements=element)
    return minimize(start, nep, cell="full", max_force=1e-6, max_stress=1e-5).system


conventional = {
    "Cr": relaxed("Cr", "bcc", 2.84),
    "Co": relaxed("Co", "hcp", 2.49),
    "Ni": relaxed("Ni", "fcc", 3.52),
}
primitive = {element: primitive_cell(cell) for element, cell in conventional.items()}
for element, cell in conventional.items():
    print(element, np.round(np.linalg.norm(cell.box.matrix, axis=1), 4))
```

```text
Cr [2.843 2.843 2.843]
Co [2.494  2.494  4.0174]
Ni [3.5164 3.5164 3.5164]
```

## Phonons

The force constants come from displacements of the atoms in a supercell,
by default the most nearly cubic one at least twice the cutoff of the
model across:

```python
results = {element: phonons(cell, nep) for element, cell in primitive.items()}
for element, result in results.items():
    print(element, result)
```

```text
Cr Phonons(1 atom, supercell of 250 cells)
Co Phonons(2 atoms, supercell of 90 cells)
Ni Phonons(1 atom, supercell of 256 cells)
```

## Along the paths of the DFT calculations

The DFT dispersions follow the paths below. Their special points are given
in units of the reciprocal vectors of the conventional cells; a matrix takes
them to those of the primitive cells. The DFT path of Ni goes from U to K,
two equivalent points of the Brillouin zone, as a line of its own; here the
path breaks at U and starts again at K.

```python
paths = {
    "Cr": [[("N", [0.5, 0.5, 0]), ("Γ", [0, 0, 0]), ("H", [0, 1, 0]),
            ("P", [0.5, 0.5, 0.5]), ("Γ", [0, 0, 0])]],
    "Co": [[("Γ", [0, 0, 0]), ("K", [1 / 3, 1 / 3, 0]), ("M", [0.5, 0, 0]),
            ("Γ", [0, 0, 0]), ("A", [0, 0, 0.5])]],
    "Ni": [[("Γ", [0, 0, 0]), ("X", [0, 1, 0]), ("U", [0.25, 1, 0.25])],
           [("K", [0.75, 0.75, 0]), ("Γ", [0, 0, 0]), ("L", [0.5, 0.5, 0.5])]],
}

sections, bands = {}, {}
for element, path in paths.items():
    to_primitive = primitive[element].box.matrix @ np.linalg.inv(
        conventional[element].box.matrix
    )
    sections[element] = [[(name, to_primitive @ q) for name, q in s] for s in path]
    bands[element] = results[element].band_structure(sections[element], points=101)
```

## Against DFT

`read_band_structure` reads the band.dat files of phonopy. Each line
between two special points has 101 wave vectors in both, so the
frequencies compare point by point; the DFT lines are stretched onto the
lines of the model, whose lattice constants differ a little.

```python
from mdapy.lattice_dynamics import read_band_structure

dft = {element: read_band_structure(f"{element}.band.dat") for element in paths}


def lines(bands):
    """The distances and frequencies of each line between two special points."""
    n = len(bands.distances) // 101
    return list(zip(np.split(bands.distances, n), np.split(bands.frequencies, n)))


on_path = {}
for element in paths:
    theirs = lines(dft[element])
    if element == "Ni":
        del theirs[2]  # the line from U to K
    on_path[element] = [
        (np.interp(d, d[[0, -1]], x[[0, -1]]), frequencies)
        for (x, _), (d, frequencies) in zip(lines(bands[element]), theirs)
    ]
    difference = np.concatenate(
        [f - g for (_, f), (_, g) in zip(lines(bands[element]), on_path[element])]
    )
    print(f"{element}: {np.sqrt(np.mean(difference**2)):.2f} THz root mean square")
```

```text
Cr: 0.36 THz root mean square
Co: 0.34 THz root mean square
Ni: 0.17 THz root mean square
```

```python
from mdapy.visualization import plot

with plot.style():
    fig, axes = plot.figure("double", ncols=3)
    for ax, element, title in zip(axes, paths, ("bcc Cr", "hcp Co", "fcc Ni")):
        plot.band_structure(bands[element], ax=ax, color="#2a78d6")
        for x, frequencies in on_path[element]:
            ax.plot(x, frequencies, color="#eb6834", lw=0.8, ls="--")
        ax.set_title(title)
        ax.set_ylim(0, None)
    axes[0].plot([], [], color="#2a78d6", label="NEP")
    axes[0].plot([], [], color="#eb6834", lw=0.8, ls="--", label="DFT")
    axes[0].legend(frameon=False)
    plot.save(fig, "crconi-phonons", formats=("png",))
```

![Phonon dispersions of bcc Cr, hcp Co and fcc Ni: the NEP model against DFT](images/crconi-phonons.png)

## Converged with the supercell?

Force constants beyond the supercell are left out. Check the dispersion of
Ni in a supercell of 6 × 6 × 6 conventional cells, given as a matrix of
primitive cells:

```python
matrix = 6 * conventional["Ni"].box.matrix @ np.linalg.inv(primitive["Ni"].box.matrix)
larger = phonons(primitive["Ni"], nep, supercell=np.rint(matrix).astype(int))
again = larger.band_structure(sections["Ni"], points=101)
change = np.abs(again.frequencies - bands["Ni"].frequencies).max()
print(larger, f"{change:.3f} THz")
```

```text
Phonons(1 atom, supercell of 864 cells) 0.018 THz
```

The default supercell is converged to some hundredths of a THz, well below
the differences from DFT; the largest of those, the highest optical branch
of Co at Γ, is the model's own.
