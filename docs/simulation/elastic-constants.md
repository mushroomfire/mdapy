# Elastic constants

`mdapy.lattice_dynamics.elastic_constants(system, calculator, *, strain=1e-3,
relax=True, max_force=1e-4, max_steps=5000)` gives the elastic constants of a
periodic crystal from its stress under small strains.

<!-- skip-test -->
```python
from mdapy.lattice_dynamics import elastic_constants
from mdapy.simulation import minimize

relaxed = minimize(crystal, potential, cell="full", max_stress=1e-4).system
result = elastic_constants(relaxed, potential)
result.constants  # (6, 6) GPa
result.bulk_modulus(), result.shear_modulus("voigt")
```

!!! warning "Relax the box first"
    The constants are those of the box as it is given. Relax it to zero
    stress first, as above, or the constants belong to a crystal under
    stress.

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `system` | `System` | | | the crystal, periodic in every direction, with a `mass` or an `element` column when its positions relax |
| `calculator` | `Calculator` | | | computes the virial, and energies and forces when the positions relax |
| `strain` | float | 1e-3 | | the magnitude $\delta$ of the strains |
| `relax` | bool | `True` | | whether the positions relax in each strained box (relaxed-ion constants) or stay put (clamped-ion constants) |
| `max_force` | float | 1e-4 | eV/Å | convergence of each relaxation, as in `minimize` |
| `max_steps` | int | 5000 | | largest number of steps of each relaxation |

## Definition

The box and the atoms are strained by $\pm\delta$ in each of the six
components $k$ of the strain in Voigt order xx, yy, zz, yz, xz, xy, the shear
components as engineering strains (half on each off-diagonal entry), and the
constants are central differences of the stress $\sigma$ (positive under
tension) in the same order:

$$
C_{jk} = \frac{\sigma_j(+\delta\, \mathbf e_k) - \sigma_j(-\delta\, \mathbf e_k)}{2\delta}.
$$

With $S = C^{-1}$, the moduli of a polycrystal are the Voigt and Reuss bounds
and their mean, the Hill average
([Hill (1952)](../reference/references.md#hill1952)):

$$
\begin{aligned}
K_\mathrm V &= \tfrac19\left[C_{11} + C_{22} + C_{33} + 2(C_{12} + C_{13} + C_{23})\right], &
K_\mathrm R &= \left[S_{11} + S_{22} + S_{33} + 2(S_{12} + S_{13} + S_{23})\right]^{-1}, \\
G_\mathrm V &= \tfrac1{15}\left[C_{11} + C_{22} + C_{33} - (C_{12} + C_{13} + C_{23}) + 3(C_{44} + C_{55} + C_{66})\right], &
G_\mathrm R &= 15\left[4(S_{11} + S_{22} + S_{33}) - 4(S_{12} + S_{13} + S_{23}) + 3(S_{44} + S_{55} + S_{66})\right]^{-1}, \\
K_\mathrm H &= \tfrac12 (K_\mathrm V + K_\mathrm R), &
G_\mathrm H &= \tfrac12 (G_\mathrm V + G_\mathrm R).
\end{aligned}
$$

## Results

It returns an `ElasticResult`:

| Attribute or method | Type | Meaning |
|---|---|---|
| `constants` | `numpy.ndarray` | (6, 6) elastic constants $C$ in GPa, in Voigt order; read-only |
| `bulk_modulus(average="hill")` | float | the bulk modulus in GPa: the `"voigt"`, `"reuss"` or `"hill"` average |
| `shear_modulus(average="hill")` | float | the shear modulus in GPa, with the same averages |

`elastic_constants` raises `InvalidParameter` for a box that is not periodic
in every direction, a non-positive `strain`, a calculator without the
properties it needs, or a relaxation that does not converge within
`max_steps`; `bulk_modulus` and `shear_modulus` raise it for an unknown
`average`.
