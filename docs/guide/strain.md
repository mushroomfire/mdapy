# Atomic strain

`mdapy.analysis.AtomicStrain(reference, cutoff, *, affine=False)` measures
the local deformation of each atom between a reference configuration and the
current one, from the vectors to its neighbors.

<!-- skip-test -->
```python
from mdapy.analysis import AtomicStrain

strain = AtomicStrain(reference, cutoff=3.0)  # the reference neighbors are found once
for frame in trajectory:
    result = strain.compute(frame)
    result.shear  # von Mises shear strain
    frame = frame.with_columns(result)  # shear_strain, volumetric_strain, d2min
```

The neighbors of atom $i$ are those closer than `cutoff` in the reference,
periodic images included. With $\mathbf d^0_j$ the vectors to them in the
reference and $\mathbf d_j$ in the current configuration, the deformation
gradient minimizes $\sum_j |\mathbf d_j - F \mathbf d^0_j|^2$:

$$
F = \Big(\sum_j \mathbf d_j \mathbf d^{0\,T}_j\Big)\Big(\sum_j \mathbf d^0_j \mathbf d^{0\,T}_j\Big)^{-1},
\qquad
\eta = \tfrac{1}{2}(F^T F - I),
$$

and the `StrainResult` holds the Green–Lagrange `strain` $\eta$, the von
Mises shear strain

$$
\eta_\text{s} = \sqrt{\eta_{xy}^2 + \eta_{xz}^2 + \eta_{yz}^2
+ \tfrac{1}{6}\big[(\eta_{xx}-\eta_{yy})^2 + (\eta_{yy}-\eta_{zz})^2 + (\eta_{xx}-\eta_{zz})^2\big]},
$$

the `volumetric` strain $\operatorname{tr}\eta/3$, and `d2min`, the residual
$D^2_\text{min} = \sum_j |\mathbf d_j - F\mathbf d^0_j|^2$ (Å²) of Falk and
Langer, the non-affine part of the motion. Atoms whose reference neighbors do
not span three dimensions get NaN.

The atoms of the current configuration must be those of the reference in the
same order (when both have an `id` column it is checked). Each atom's
displacement is taken by the nearest periodic image in fractional
coordinates, so positions may be wrapped back into the box and the box may
deform as long as no atom moves by half the box. The current vectors use the
current box, which includes its homogeneous deformation in the strain; with
`affine=True` they use the reference box instead, which removes it.

## Validation

`tests/analysis/test_strain.py` checks that a homogeneous deformation of the
box and the atoms, with the atoms wrapped back into the new box, gives every
atom exactly its Green–Lagrange strain, its von Mises and volumetric strain
and $D^2_\text{min} = 0$, and no strain with `affine=True`; that a rotation is
no strain; agreement with per-atom least squares in NumPy to
10<sup>−12</sup> for random displacements; NaN for a flat layer; and the errors.

## References

- F. Shimizu, S. Ogata and J. Li, Theory of shear banding in metallic glasses
  and molecular dynamics calculations, *Mater. Trans.* **48**, 2923 (2007).
  doi:10.2320/matertrans.MJ200769
- M. L. Falk and J. S. Langer, Dynamics of viscoplastic deformation in
  amorphous solids, *Phys. Rev. E* **57**, 7192 (1998).
  doi:10.1103/PhysRevE.57.7192
