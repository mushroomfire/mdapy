# Elements

`mdapy.elements` holds the chemical symbols and standard atomic weights used
everywhere in mdapy. The atomic weights are those of IUPAC
([Meija et al. (2016)](../reference/references.md#meija2016)); for an element without stable
isotopes, the mass of its longest-lived isotope.

```python
from mdapy import elements

elements.atomic_numbers(["Cu", "Ni"])  # array([29, 28])
elements.atomic_masses(["Cu", "Ni"])  # array([63.546 , 58.6934])
elements.element_from_mass(26.98)  # 'Al'
```

## Contents

| Name | Meaning |
|---|---|
| `SYMBOLS` | The 118 chemical symbols, indexed by atomic number. Index 0 is `"X"`, an unknown element. |
| `ATOMIC_MASSES` | Standard atomic weights in amu, indexed by atomic number. Read-only. |
| `ELEMENT_DTYPE` | The polars type of the `element` column. |
| `atomic_numbers(symbols)` | Atomic numbers of chemical symbols. |
| `atomic_masses(elements)` | Atomic weights of symbols or atomic numbers. |
| `element_from_mass(mass, tolerance=0.01, candidates=None)` | The element whose atomic weight matches a mass. |

## The `element` column

Per-atom chemical symbols are stored with the polars type `ELEMENT_DTYPE`, an
enumeration of `SYMBOLS` in order of atomic number. Each atom takes one byte,
symbols are checked when the column is created, and the underlying integer
of each value is its atomic number:

```python
import polars as pl

s = pl.Series(["Cu", "Ni"], dtype=elements.ELEMENT_DTYPE)
s.to_physical()  # [29, 28]
```

## Identifying elements from masses

Some files, such as LAMMPS data files, give masses but no chemical symbols.
`element_from_mass` returns the element whose atomic weight lies within
`tolerance` (0.01 amu by default) of the mass, and raises
[`InvalidParameter`](api.md#mdapy.errors.InvalidParameter) when no element or
more than one element matches.

By default only elements with at least one stable isotope are considered:
H to Bi, except Tc and Pm. No two of them have atomic weights within 0.02 amu
of each other, so the match is unique. Radioactive elements must be listed in
`candidates`, because their atomic weights can be almost equal to another
element's: ²⁰⁹Po and Bi differ by 0.002 amu, ²⁴⁷Cm and ²⁴⁷Bk by 0.00004 amu.
