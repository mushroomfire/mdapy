# Units and conventions

MDAPY uses one unit system everywhere: in inputs, outputs, files written by
MDAPY, and stored properties. Readers convert from a file's own units when
reading, and writers convert back when writing.

## Units

| Quantity | Unit |
|---|---|
| length | Å |
| energy | eV |
| time | ps |
| mass | amu |
| temperature | K |
| charge | e |
| velocity | Å/ps |
| force | eV/Å |
| stress, pressure | GPa |
| virial | eV |
| elastic moduli of 2D materials | N/m |
| frequency of vibrations | THz |

These match LAMMPS `metal` units except stress and pressure, which LAMMPS
reports in bar. Files from GPUMD, which uses fs and Å/fs, are converted on
reading and writing.

## Types

The tables and the Python API give exact types, the same on every page:

| Written | Means |
|---|---|
| `float64`, `float32` | a real number of 64 or 32 bits |
| `int8`, `int32`, `int64`, `uint8`, `uint64` | an integer of that many bits |
| `(N, 3) float64` | an array of that shape and type |
| `mdapy.index_dtype` | atom indices: `int32`, or `int64` in MDAPY built with [64-bit atom indices](../getting-started/installation.md#64-bit-atom-indices) |
| indices | an array of atom indices, of any integer type |

As an argument, a Python `float` is a `float64`, a Python `int` stands for an `int32` or
`int64` that holds it, and lists and arrays of other numeric types are converted. Pictures
are computed in `float32`, so their options are given as `float32`.

## Sign conventions

- **Stress** is positive under tension:
  \(\sigma_{\alpha\beta} = \frac{1}{V}\frac{\partial E}{\partial \varepsilon_{\alpha\beta}}\).
- **Pressure** is \(p = -\frac{1}{3}\operatorname{tr}\sigma\), positive under
  compression.
- **Virial** is \(W = -V\sigma\), in eV, positive under compression (the GPUMD
  convention).
- **Voigt order** of symmetric tensors is xx, yy, zz, yz, xz, xy.
- **Burgers vectors** are those of OVITO's dislocation analysis: along the
  sense of the line, the sum of the steps of a counterclockwise circuit
  around it in the perfect crystal, the opposite of Hirth and Lothe's
  (FS/RH). The structure builder puts dislocations in by the same.

## Conversion factors

All conversions in MDAPY use the constants in
[`mdapy.units`](api/utilities.md#mdapy.units). A factor named `A_TO_B` converts a value
in unit A to unit B by multiplication:

```python
from mdapy import units

stress_ev_per_a3 = 0.01
stress_gpa = stress_ev_per_a3 * units.EV_PER_A3_TO_GPA  # 1.602 GPa
```

The defining constants are the exact 2019 SI values (elementary charge,
Boltzmann, Planck and Avogadro constants); the atomic mass constant is the
CODATA 2022 value.
