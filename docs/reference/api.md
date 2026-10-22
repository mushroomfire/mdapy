# Python API

This reference is generated from the docstrings, which describe parameters,
return values and units. What each function computes, the formulas, examples
and references are on the page of each feature, listed on the
[home page](../index.md).

```python
import mdapy
```

## Structures

::: mdapy.read

::: mdapy.write

::: mdapy.read_trajectory

::: mdapy.Trajectory

::: mdapy.formats

::: mdapy.detect_format

::: mdapy.System

::: mdapy.Topology

::: mdapy.Box

::: mdapy.core.box.LammpsBox

::: mdapy.NeighborList

## Analyses

::: mdapy.core.Result

::: mdapy.identify.CommonNeighborAnalysis

::: mdapy.identify.StructureResult

::: mdapy.identify.DiamondStructure

::: mdapy.identify.ChillPlus

::: mdapy.identify.CentroSymmetryParameter

::: mdapy.identify.CentroSymmetryResult

::: mdapy.analysis.RadialDistributionFunction

::: mdapy.analysis.RadialDistributionResult

::: mdapy.analysis.StructureFactor

::: mdapy.analysis.StructureFactorResult

::: mdapy.analysis.CoordinationNumber

::: mdapy.analysis.CoordinationResult

::: mdapy.analysis.SteinhardtParameters

::: mdapy.analysis.SteinhardtResult

::: mdapy.analysis.EntropyFingerprint

::: mdapy.analysis.EntropyResult

::: mdapy.analysis.AtomicTemperature

::: mdapy.analysis.TemperatureResult

::: mdapy.analysis.AtomicStrain

::: mdapy.analysis.StrainResult

::: mdapy.analysis.WignerSeitz

::: mdapy.analysis.WignerSeitzResult

::: mdapy.analysis.ClusterAnalysis

::: mdapy.analysis.ClusterResult

::: mdapy.analysis.BondAngleDistribution

::: mdapy.analysis.BondAngleResult

::: mdapy.analysis.WarrenCowley

::: mdapy.analysis.WarrenCowleyResult

## Calculators

::: mdapy.calculators.Calculator

::: mdapy.calculators.CalculatorResult

::: mdapy.calculators.EAM

::: mdapy.calculators.NEP

::: mdapy.calculators.LAMMPS

::: mdapy.calculators.testing.check_calculator

## Simulation

::: mdapy.simulation.minimize

::: mdapy.simulation.MinimizeResult

::: mdapy.simulation.elastic_constants

::: mdapy.simulation.ElasticResult

## Threads and indices

::: mdapy.get_num_threads

::: mdapy.set_num_threads

::: mdapy.num_threads

::: mdapy.CancelToken

::: mdapy.core.indices.index_dtype

## Errors

::: mdapy.errors
    options:
      show_root_heading: true
      heading_level: 3

## Elements

::: mdapy.elements
    options:
      show_root_heading: true
      heading_level: 3

## Units

::: mdapy.units
    options:
      show_root_heading: true
      heading_level: 3
