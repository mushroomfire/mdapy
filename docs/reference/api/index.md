# Python API

This reference is generated from the docstrings, which describe parameters, return values
and units. What each function computes, the formulas, examples and references are on the
page of each feature, listed on the [home page](../../index.md).

```python
import mdapy as mp
```

## Structures

- [Structures and files](structures.md)
- [Structure builder](build.md)

## Analysis

- [Structure identification](identification.md)
- [Local order](local-order.md)
- [Defects and deformation](defects.md)
- [Dynamics and trajectories](dynamics.md)
- [Binning and statistics](statistics.md)

## Simulation

- [Potentials](calculators.md): `mdapy.calculators`
- [Simulation](simulation.md): `mdapy.simulation`, energy minimization, minimum energy
  paths and molecular dynamics, and `mdapy.simulation.controls`

## Properties

`mdapy.properties`, a module for each family of properties, by any method:

- [Defects](properties/defects.md): `mdapy.properties.defects`
- [Elastic constants](properties/elastic.md): `mdapy.properties.elastic`
- [Electronic structure](properties/electronic.md): `mdapy.properties.electronic`
- [Phonons](properties/phonons.md): `mdapy.properties.phonons`
- [Transport](properties/transport.md): `mdapy.properties.transport`
- [Thermodynamics](properties/thermodynamics.md): `mdapy.properties.thermodynamics`

## Other

- [MLIP](mlip.md)
- [Visualization](visualization.md)
- [Fitting](fitting.md)
- [Interoperability](interop.md)
- [Threads, errors, elements and units](utilities.md)
