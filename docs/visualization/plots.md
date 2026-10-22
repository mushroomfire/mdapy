# Plots

`mdapy.visualization.plot` sets matplotlib up for the figures of journals
and plots the results of mdapy: parity plots of energies and forces, radial
distribution functions, structure factors, bond angles, mean squared
displacements, stacking fault energies and counts of structures. It needs
matplotlib.

```python
import numpy as np

from mdapy.analysis import RadialDistributionFunction
from mdapy.build import build_crystal, random_alloy
from mdapy.visualization import plot

alloy = random_alloy(
    build_crystal("fcc", 3.6, repeat=(5, 5, 5)), {"Cu": 1, "Ni": 1}, seed=1
)
rdf = RadialDistributionFunction(r_max=8.0, bins=160).compute(alloy)

with plot.style():
    fig, (left, right) = plot.figure("double", nrows=1, ncols=2)
    plot.rdf(rdf, ax=left, partial=True)
    reference = np.random.default_rng(0).normal(size=500)
    plot.parity(
        reference, reference + 0.02, ax=right, label="test", unit="meV/atom", scale=1000
    )
    plot.panel_label(left, "a")
    plot.panel_label(right, "b")
    # plot.save(fig, "figure")  # figure.pdf, figure.svg and figure.png
```

## Parameters

| Function | Does |
|---|---|
| `style()` | a context in which matplotlib uses `STYLE` and the colors `COLORS` |
| `figure(width="single", height=None, *, nrows=1, ncols=1, **options)` | a figure one column (88 mm) or two (180 mm) wide, or `width` mm; `height` in mm, by default 3/4 of a panel's width per row; `options` go to `plt.subplots` |
| `panel_label(ax, text, *, x=-0.12, y=1.02)` | a bold label at the top left of a panel |
| `save(fig, path, formats=("pdf", "svg", "png"))` | the figure in each format, the text of SVG kept as text |
| `parity(reference, predicted, *, ax=None, label="", unit="", scale=1.0, density=False, color=None)` | predicted against reference values with the diagonal, the root mean square error, times `scale`, in the legend; `density` draws a hexagonal map |
| `rdf(result, *, ax=None, partial=False)` | g(r), or the partial functions |
| `structure_factor(result, *, ax=None)` | S(q) |
| `bond_angles(result, *, ax=None)` | the density of bond angles |
| `msd(result, *, timestep=1.0, ax=None, fit=None)` | the MSD against time in ps; `fit=(first, last)` fits a line over those lags, its slope over 6 the diffusion coefficient in the legend |
| `stacking_fault(result, *, ax=None)` | a generalized stacking fault energy in mJ/m² |
| `structure_counts(result, *, ax=None)` | the atoms of each structure type, in its color |

Each plot draws on `ax`, by default a new figure, and returns the axes.

## Results

`STYLE` follows the artwork guides of Nature: sans-serif text of 5 to 7
points, lines of 0.5 to 1 point, no top and right spines, fonts embedded as
TrueType. `COLORS` are those of Okabe and Ito, safe for color blindness.
`MM`, `SINGLE` and `DOUBLE` give millimetres and the widths of the columns
in inches.

## Errors

`figure` raises `InvalidParameter` for a width that is neither single,
double nor a positive number, and `parity` for values of different sizes.
Without matplotlib the plots raise `ImportError`.
