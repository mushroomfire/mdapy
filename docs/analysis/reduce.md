# Reductions

`reduce` gives one number from the values of a column of the atoms: their
mean, sum, extremes, median, spread, the number of those not zero or the
atom of the lowest or highest, as OVITO's reduce property. Over the frames
of a trajectory, [`TimeSeries`](time-series.md) takes one such number from
each frame.

```python
import numpy as np

import mdapy as mp
from mdapy.analysis import reduce
from mdapy.build import build_crystal

copper = build_crystal("fcc", 3.615, elements="Cu", repeat=(6, 6, 6))
rng = np.random.default_rng(2)
copper = copper.with_columns(
    velocities=rng.normal(0, 2.0, (len(copper), 3)),
    charge=rng.normal(0, 0.1, len(copper)),
)

reduce(copper, "charge", "sum")  # the total charge
reduce(copper, "velocities", "mean")  # (3,) the mean of each component
reduce(copper, "velocities.z", "max")  # the fastest along z
fastest = reduce(
    copper, "velocities.x**2 + velocities.y**2 + velocities.z**2", "argmax"
)
reduce(copper, "charge", "std", atoms=copper.where("positions.z > 10"))
```

`fastest` is the index of the fastest atom among all atoms, also when
`atoms` chooses some of them.

## Parameters

`reduce(system, column, operation="mean", *, atoms=None)`

| Name | Type | Default | Meaning |
|---|---|---|---|
| `system` | System | | the atoms |
| `column` | str | | a column of numbers, a component such as `"positions.z"` or an expression over the columns |
| `operation` | str | `"mean"` | one of `OPERATIONS`, below |
| `atoms` | array_like | `None` | a boolean mask or indices of the atoms to take, such as those of [`System.where`](../getting-started/selecting.md#masks); all by default |

| Operation | Gives |
|---|---|
| `"mean"` | the mean |
| `"sum"` | the sum; 0 of no values |
| `"min"`, `"max"` | the lowest and highest value |
| `"median"` | the median, the mean of the two middle values for an even number |
| `"variance"`, `"std"` | the variance and standard deviation of a sample, divided by the number of values minus one |
| `"nonzero"` | the number of values not zero |
| `"argmin"`, `"argmax"` | the index of the atom of the lowest and highest value, the first of equal ones |

NaN values are left out of every operation. The mean, extremes and median of
no values are NaN, and so are the variance and standard deviation of fewer
than two.

## Results

A float, or an int for `"nonzero"`, `"argmin"` and `"argmax"`; for a column
of several components, such as `"velocities"`, an array of one per
component.

## Errors

`InvalidParameter` for an unknown operation, a column of text, a wrong mask,
or `"argmin"` or `"argmax"` of no values.

## See also

- [Histograms](histogram.md).
- OVITO [Reduce property](https://docs.ovito.org/reference/pipelines/modifiers/reduce_property.html).
