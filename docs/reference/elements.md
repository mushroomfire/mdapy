# Elements

`mdapy.elements` holds the chemical symbols, standard atomic weights and
radii used everywhere in mdapy. The atomic weights are those of IUPAC
([Meija et al. (2016)](references.md#meija2016)); for an element without stable
isotopes, the mass of its longest-lived isotope. The covalent radii are
those of [Cordero et al. (2008)](references.md#cordero2008), with sp³
carbon and low-spin manganese, iron and cobalt; the van der Waals radii
those of [Alvarez (2013)](references.md#alvarez2013).

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
| `COVALENT_RADII`, `VDW_RADII` | Covalent and van der Waals radii in Å, indexed by atomic number; NaN where there is none. Read-only. |
| `covalent_radii(elements)`, `vdw_radii(elements)` | The radii of symbols or atomic numbers; `InvalidParameter` for an element without one. |
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
