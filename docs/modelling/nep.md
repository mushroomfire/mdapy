# NEP

`NEP(path)` reads a neuroevolution potential of version 4 in the `nep.txt`
format of GPUMD, without (`nep4`) or with (`nep4_zbl`) a ZBL repulsion, and
computes every property.

<!-- skip-test -->
```python
from mdapy.calculators import NEP

potential = NEP("nep.txt")
result = potential.compute(system, ("energy", "forces", "virial"))
descriptors = potential.descriptors(system)  # (N, dim)
```

| Attribute | Type | Meaning |
|---|---|---|
| `path` | str | the potential file |
| `elements` | tuple of str | the elements of the potential, in the order of the file |
| `cutoff` | float | the largest radial or angular cutoff in Å |

| Method | Returns |
|---|---|
| `compute(system, properties, *, neighbors=None, progress=None, cancel=None)` | as every calculator, with the `progress=` and `cancel=` of [Parallelism](../getting-started/parallelism.md#progress-and-cancellation) |
| `descriptors(system, *, neighbors=None, progress=None, cancel=None)` | (N, dim) descriptors of each atom, scaled as the network sees them: the radial ones, then the angular ones by $l$ and $n$, then the four-body and five-body ones by $n$ |

The atoms need an `element` column; elements the potential does not have
raise `InvalidParameter`. Other models of GPUMD (NEP3, NEP5, charge, dipole
and polarizability models) and malformed files raise
[`FileFormatError`](../reference/api.md#mdapy.errors.FileFormatError).

The model is that of GPUMD ([Fan et al. (2022)](../reference/references.md#fan2022)), version 4
([Song et al. (2024)](../reference/references.md#song2024)): radial descriptors from Chebyshev functions of
the distance, angular ones from real spherical harmonics up to
$l_\text{max}$, with optional four-body and five-body invariants, each scaled
by its factor in the file and fed to one hidden layer of $\tanh$ neurons per
element, which gives the energy $E_i$ of atom $i$. The cutoff of a pair is
the mean of the cutoffs of its two elements. With ZBL, each pair within the
angular cutoff adds the screened Coulomb repulsion of
[Ziegler et al. (1985)](../reference/references.md#ziegler1985),
$V(r) = \frac{k_e Z_i Z_j}{r}\,\phi(r/a)\, f_s(r)$, half to each atom; the
switch $f_s$ is 1 below an inner cutoff and falls as a cosine to 0 at an
outer cutoff, as the `zbl` line of the file chooses:

| `zbl` line | screening $\phi$ | inner cutoff | outer cutoff |
|---|---|---|---|
| `zbl r1 r2` | universal ZBL | r1 | r2 |
| `zbl r1 r2 f` | universal ZBL | 0 | the smaller of r2 and f times the sum of the two covalent radii GPUMD uses |
| `zbl 0 0` | from a table at the end of the file, per pair of elements | from the table | from the table |

As in GPUMD, the derivative of $E_i$ by the vector to neighbor $j$ adds to
the atomic virial of atom $j$, so atomic virials are those of GPUMD; EAM
shares each pair's virial equally between its two atoms, as LAMMPS does. Both
sum to the same virial.

!!! quote "Please cite"
    [Fan et al. (2022)](../reference/references.md#fan2022) and [Song et al. (2024)](../reference/references.md#song2024); with ZBL, also [Ziegler et al. (1985)](../reference/references.md#ziegler1985); and the paper of the potential.
