# Lennard-Jones and Morse

```python
from mdapy.calculators import LennardJones, Morse

# argon and krypton, epsilon in eV and sigma in Å, their pair mixed
lj = LennardJones({"Ar": (0.0104, 3.40), "Kr": (0.0140, 3.65)}, 8.5)
# copper and nickel, D0 in eV, alpha in 1/Å and r0 in Å for every pair
morse = Morse(
    {
        "Cu": (0.3429, 1.3588, 2.866),
        "Ni": (0.4205, 1.4199, 2.780),
        ("Cu", "Ni"): (0.38, 1.39, 2.82),
    },
    7.0,
)
lj.elements, lj.cutoff
```

The energy of two atoms against their distance, of both potentials:

```python
import numpy as np
import mdapy
from mdapy.visualization import plot

distance = np.linspace(2.0, 8.0, 241)


def curve(potential, a, b):
    box = mdapy.Box(30.0, pbc=False)
    pairs = [
        mdapy.System([[0, 0, 0], [r, 0, 0]], box, elements=[a, b]) for r in distance
    ]
    return [potential.compute(pair, ("energy",)).energy for pair in pairs]


with plot.style():
    fig, (left, right) = plot.figure("double", ncols=2)
    for pair in (("Ar", "Ar"), ("Ar", "Kr"), ("Kr", "Kr")):
        left.plot(distance, 1000 * np.array(curve(lj, *pair)), label="–".join(pair))
    left.set(xlabel="Distance (Å)", ylabel="Energy (meV)", ylim=(-16, 10), xlim=(3, 8))
    left.legend()
    for pair in (("Cu", "Cu"), ("Cu", "Ni"), ("Ni", "Ni")):
        right.plot(distance, curve(morse, *pair), label="–".join(pair))
    right.set(xlabel="Distance (Å)", ylabel="Energy (eV)", ylim=(-0.5, 0.6))
    right.legend()
    plot.save(fig, "images/pair-potentials", formats=("png",))
```

![The energy of a pair of the Lennard-Jones and the Morse potential](images/pair-potentials.png)

*The Lennard-Jones potential of argon and krypton, their pair mixed by the
geometric means, and the Morse potential of copper and nickel: each pair
lowest at 2^(1/6) σ, or at r0, by ε or D0.*

`LennardJones(parameters, cutoff, *, mixing="geometric", shift=False)` and
`Morse(parameters, cutoff, *, shift=False)` are pair potentials with numbers
for each pair of elements, computed as LAMMPS computes `pair_style lj/cut`
and `pair_style morse`. Their energy is

$$
E_{LJ}(r) = 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} -
\left( \frac{\sigma}{r} \right)^{6} \right], \qquad
E_{M}(r) = D_0 \left[ e^{-2 \alpha (r - r_0)} - 2 e^{-\alpha (r - r_0)} \right]
$$

below the cutoff of the pair, and zero beyond.

## Parameters

| Parameter | Type | Meaning |
|---|---|---|
| `parameters` | mapping, or sequence of float64 for `LennardJones` | keyed by an element, the numbers of its pairs with itself; by a tuple of two elements, as `("Cu", "Ni")`, those of that pair: (ε, σ) or (ε, σ, cutoff) of `LennardJones`, (D0, α, r0) or (D0, α, r0, cutoff) of `Morse` |
| `cutoff` | float64 | cutoff in Å of the pairs given none |
| `mixing` | str | of `LennardJones`, how the pairs of two elements given alone mix, as `pair_modify mix`: `"geometric"` (ε, σ and the cutoff), `"arithmetic"` (ε geometric, σ and the cutoff arithmetic, Lorentz–Berthelot) or `"sixthpower"` |
| `shift` | bool | take off each pair its energy at its cutoff, as `pair_modify shift yes`, so that the energy goes to zero there |

`LennardJones((ε, σ), cutoff)`, numbers alone, is one kind of atom of any
element: it computes systems that have no elements.

Morse parameters do not mix: every pair of the elements needs its own, as in
LAMMPS. A pair given as a tuple is taken as it is, and an element given both
alone and as its own pair must have the same numbers in both.

| Attribute | Type | Meaning |
|---|---|---|
| `elements` | tuple of str or `None` | the elements, in the order of the parameters; `None` for one kind of atom |
| `cutoffs` | (E, E) float64 | the cutoff of each pair of elements in Å |
| `cutoff` | float64 | the largest of them |
| `epsilon`, `sigma` | (E, E) float64 | of `LennardJones`, the numbers of each pair after mixing |

The energy and the virial of each pair are shared equally between its two
atoms. The long-range corrections of LAMMPS (`pair_modify tail`) and the
weights of bonded neighbors (`special_bonds`) are not part of these
calculators.

## Errors

The calculators raise `InvalidParameter` for parameters that are not a
mapping with an entry, a key that is not an element or a pair of elements,
numbers of the wrong count or not finite, a negative ε or D0, a σ, α or r0
that is not positive, a cutoff that is not positive, an unknown mixing rule,
a pair given twice with other numbers, a pair of elements without numbers,
and atoms of an element they have no numbers for.

!!! quote "Please cite"
    [Jones (1924)](../../reference/references.md#jones1924) or
    [Morse (1929)](../../reference/references.md#morse1929), and the paper of the parameters.
