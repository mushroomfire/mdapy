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

::: mdapy.identify.AcklandJonesAnalysis

::: mdapy.identify.DiamondStructure

::: mdapy.identify.ChillPlus

::: mdapy.identify.CentroSymmetryParameter

::: mdapy.identify.CentroSymmetryResult

::: mdapy.identify.CommonNeighborParameter

::: mdapy.identify.CommonNeighborParameterResult

::: mdapy.analysis.RadialDistributionFunction

::: mdapy.analysis.RadialDistributionResult

::: mdapy.analysis.CoordinationNumber

::: mdapy.analysis.CoordinationResult

::: mdapy.analysis.StructureFactor

::: mdapy.analysis.StructureFactorResult

::: mdapy.analysis.BondAngleDistribution

::: mdapy.analysis.BondAngleResult

::: mdapy.analysis.WarrenCowley

::: mdapy.analysis.WarrenCowleyResult

::: mdapy.analysis.SteinhardtParameters

::: mdapy.analysis.SteinhardtResult

::: mdapy.analysis.EntropyFingerprint

::: mdapy.analysis.EntropyResult

::: mdapy.analysis.VoronoiAnalysis

::: mdapy.analysis.VoronoiResult

::: mdapy.analysis.VoronoiPolyhedra

::: mdapy.analysis.ClusterAnalysis

::: mdapy.analysis.ClusterResult

::: mdapy.analysis.WignerSeitz

::: mdapy.analysis.WignerSeitzResult

::: mdapy.analysis.AtomicStrain

::: mdapy.analysis.StrainResult

::: mdapy.analysis.MeanSquaredDisplacement

::: mdapy.analysis.DisplacementResult

::: mdapy.analysis.LindemannIndex

::: mdapy.analysis.LindemannResult

::: mdapy.analysis.AtomicTemperature

::: mdapy.analysis.TemperatureResult

## Structure builder

::: mdapy.build.build_crystal

::: mdapy.build.random_alloy

::: mdapy.build.sqs_alloy

::: mdapy.build.add_vacancies

::: mdapy.build.add_interstitials

::: mdapy.build.substitute

::: mdapy.build.slip

::: mdapy.build.add_vacuum

::: mdapy.build.insert_dislocation

::: mdapy.build.edge_dislocation

::: mdapy.build.build_polycrystal

::: mdapy.build.random_rotations

::: mdapy.build.twin_lamellae

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

::: mdapy.simulation.vacancy_formation_energy

::: mdapy.simulation.VacancyResult

::: mdapy.simulation.surface_energy

::: mdapy.simulation.SurfaceResult

::: mdapy.simulation.stacking_fault_energy

::: mdapy.simulation.StackingFaultResult

## Lattice dynamics

::: mdapy.lattice_dynamics.elastic_constants

::: mdapy.lattice_dynamics.ElasticResult

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
