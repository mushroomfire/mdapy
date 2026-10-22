# Point defects

`mdapy.analysis.WignerSeitz(reference, *, affine=False)` finds vacancies and
interstitials by the Wigner–Seitz cells of a reference lattice: each atom of
the current configuration belongs to its nearest reference site, periodic
images included. A site without atoms is a vacancy; a site with more than one
holds interstitials. The atoms need not be the same, nor as many, as in the
reference.

<!-- skip-test -->
```python
from mdapy.analysis import WignerSeitz

defects = WignerSeitz(perfect_crystal)
result = defects.compute(after_cascade)
result.vacancy_count, result.interstitial_count
result.vacancies  # indices of the empty sites
after_cascade = after_cascade.with_columns(result)  # columns site, occupancy
```

The `WignerSeitzResult` has the `site` of each atom and the `occupancy` of
each reference site; `vacancies` and `interstitials` list the empty and the
crowded sites, `vacancy_count` counts the empty ones and
`interstitial_count` the atoms beyond one per site, so that the difference in
the number of atoms is the number of interstitials less the number of
vacancies. Its columns are `site` and `occupancy`, the occupancy of each
atom's site. An atom equally near two sites belongs to the one of lower index.

The current configuration must have the periodicity of the reference. With
`affine=True` its atoms are first mapped into the reference box by their
fractional coordinates, which removes a homogeneous deformation of the box:
without it, a box stretched by 10 % puts atoms on the wrong sites near its far
side.

## Validation

`tests/analysis/test_wigner_seitz.py` checks that thermal motion makes no
defects; that a Frenkel pair is one vacancy and one interstitial; that
removing or adding atoms gives as many vacancies or interstitials; that a
stretched box needs `affine=True`; that sites agree with a brute-force search
over periodic images in a triclinic, partly periodic box with atoms outside
it; and the errors.

## References

- K. Nordlund, M. Ghaly, R. S. Averback, M. Caturla, T. Diaz de la Rubia and
  J. Tarus, Defect production in collision cascades in elemental
  semiconductors and fcc metals, *Phys. Rev. B* **57**, 7556 (1998).
  doi:10.1103/PhysRevB.57.7556
