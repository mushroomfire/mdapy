# LAMMPS

<!-- skip-test -->
```python
from mdapy.calculators import LAMMPS

calculator = LAMMPS(
    "pair_style eam/alloy\npair_coeff * * CuNi.eam.alloy Cu Ni",
    elements=["Cu", "Ni"],
)
calculator.compute(system, ("energy", "forces", "virial"))
calculator.close()
```

`LAMMPS(pair, elements, *, commands="", cmdargs=(), cutoff=None)` makes any
interaction that LAMMPS computes a calculator, through the LAMMPS Python
module, installed as the
[LAMMPS documentation](https://docs.lammps.org/Python_install.html)
describes.

!!! warning "Metal units only"
    LAMMPS runs in metal units, those of mdapy: Å, eV, ps and amu. The
    commands in `pair` must define the interaction for metal units;
    potentials written for other units, such as real or si, give wrong
    results.

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `pair` | str | | | the LAMMPS commands of the interaction, such as `pair_style` and `pair_coeff`, for metal units |
| `elements` | sequence of str | | | the element of each LAMMPS atom type, from type 1 |
| `commands` | str | `""` | | LAMMPS commands run before `pair`, such as `newton on` |
| `cmdargs` | sequence of str | `()` | | command-line arguments of LAMMPS, such as `["-k", "on", "t", "4", "-sf", "kk"]` for Kokkos with four threads |
| `cutoff` | float | `None` | Å | the largest cutoff of the interaction; asked from LAMMPS when not given, which LAMMPS older than 30 Sep 2026 cannot tell |

| Attribute or method | Meaning |
|---|---|
| `elements` | the element of each atom type |
| `cutoff` | the largest cutoff of the interaction in Å |
| `compute(system, properties, *, neighbors=None)` | every property; LAMMPS searches its own neighbors, so `neighbors` is not used |
| `close()` | closes the LAMMPS instance; the next computation starts a new one |

The calculator keeps one LAMMPS instance while the atoms and their elements
stay the same, as during [`minimize`](minimize.md), also when the box changes but not its
periodic directions; it starts a new one for other atoms, elements or
periodic directions. Any box works: LAMMPS receives it rotated to its restricted
triclinic form, and forces and virials are rotated back. Along non-periodic
directions LAMMPS has fixed boundaries around the atoms. With Kokkos, LAMMPS
is shut down when Python exits.

The constructor raises `InvalidParameter` when LAMMPS rejects the commands,
with the message of LAMMPS, or cannot tell the cutoff that is not given, and
`ImportError` without the LAMMPS Python module; `compute` raises
`InvalidParameter` for atoms of an element that no atom type has.

!!! quote "Please cite"
    [Thompson et al. (2022)](../reference/references.md#thompson2022), and the paper of the potential.
