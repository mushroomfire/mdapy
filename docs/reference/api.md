# Python API

This reference is generated from the docstrings, which describe parameters,
return values and units. What each function computes, the formulas, examples
and references are on the page of each feature, listed on the
[home page](../index.md).

```python
import mdapy as mp
```

## Structures

::: mdapy.read

::: mdapy.write

::: mdapy.read_trajectory

::: mdapy.Trajectory

::: mdapy.io.Mapped

::: mdapy.formats

::: mdapy.detect_format

::: mdapy.System

::: mdapy.Topology

::: mdapy.Box

::: mdapy.TriangleMesh

::: mdapy.core.box.LammpsBox

::: mdapy.NeighborList

## Analyses

::: mdapy.core.Result

::: mdapy.identify.CommonNeighborAnalysis

::: mdapy.identify.StructureResult

::: mdapy.identify.AcklandJonesAnalysis

::: mdapy.identify.PolyhedralTemplateMatching

::: mdapy.identify.PTMResult

::: mdapy.identify.PlanarFaults

::: mdapy.identify.PlanarFaultResult

::: mdapy.identify.GrainSegmentation

::: mdapy.identify.GrainResult

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

::: mdapy.analysis.SpatialCorrelation

::: mdapy.analysis.SpatialCorrelationResult

::: mdapy.analysis.BondAngleDistribution

::: mdapy.analysis.BondAngleResult

::: mdapy.analysis.BondLengthDistribution

::: mdapy.analysis.BondLengthResult

::: mdapy.analysis.Histogram

::: mdapy.analysis.HistogramResult

::: mdapy.analysis.reduce

::: mdapy.analysis.RingStatistics

::: mdapy.analysis.RingResult

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

::: mdapy.analysis.SurfaceMesh

::: mdapy.analysis.SurfaceMeshResult

::: mdapy.analysis.SurfaceRegions

::: mdapy.analysis.DislocationAnalysis

::: mdapy.analysis.DislocationResult

::: mdapy.analysis.DislocationLine

::: mdapy.analysis.CrystalClusters

::: mdapy.analysis.WignerSeitz

::: mdapy.analysis.WignerSeitzResult

::: mdapy.analysis.AtomicDisplacement

::: mdapy.analysis.AtomicDisplacementResult

::: mdapy.analysis.SpatialBinning

::: mdapy.analysis.SpatialBinningResult

::: mdapy.analysis.isosurface

::: mdapy.analysis.AtomicStrain

::: mdapy.analysis.StrainResult

::: mdapy.analysis.ElasticStrain

::: mdapy.analysis.ElasticStrainResult

::: mdapy.analysis.MeanSquaredDisplacement

::: mdapy.analysis.DisplacementResult

::: mdapy.analysis.LindemannIndex

::: mdapy.analysis.LindemannResult

::: mdapy.analysis.TimeAverage

::: mdapy.analysis.TimeAverageResult

::: mdapy.analysis.TimeSeries

::: mdapy.analysis.TimeSeriesResult

::: mdapy.analysis.FrameDifference

::: mdapy.analysis.FrameDifferenceResult

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

::: mdapy.build.Dislocation

::: mdapy.build.dislocation_cylinder

::: mdapy.build.dislocation_quadrupole

::: mdapy.build.build_polycrystal

::: mdapy.build.random_rotations

::: mdapy.build.twin_lamellae

::: mdapy.transform.replicate

::: mdapy.transform.wrap

::: mdapy.transform.unwrap

::: mdapy.transform.remove_drift

::: mdapy.transform.freeze

::: mdapy.transform.smooth_trajectory

::: mdapy.transform.create_bonds

::: mdapy.transform.combine

::: mdapy.transform.slice

::: mdapy.transform.slice_mask

## MLIP

::: mdapy.mlip.SOAP

::: mdapy.mlip.structure_descriptors

::: mdapy.mlip.farthest_point_sampling

::: mdapy.mlip.principal_components

::: mdapy.mlip.PrincipalComponents

::: mdapy.mlip.perturb

::: mdapy.mlip.predict

::: mdapy.mlip.read_gpumd

::: mdapy.mlip.Predictions

::: mdapy.mlip.committee

::: mdapy.mlip.CommitteeResult

::: mdapy.mlip.novelty

::: mdapy.mlip.check_frames

## Visualization

::: mdapy.visualization.render

::: mdapy.visualization.Scene

::: mdapy.visualization.atom_colors

::: mdapy.visualization.atom_radii

::: mdapy.visualization.color_coding

::: mdapy.visualization.color_by_type

::: mdapy.visualization.assign_color

::: mdapy.visualization.voxel_faces

::: mdapy.visualization.colormap

::: mdapy.visualization.style_settings

::: mdapy.visualization.write_png

::: mdapy.visualization.plot

## Calculators

::: mdapy.calculators.Calculator

::: mdapy.calculators.CalculatorResult

::: mdapy.calculators.EAM

::: mdapy.calculators.NEP

::: mdapy.calculators.LAMMPS

::: mdapy.calculators.VASP

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

## Interoperability

::: mdapy.interop.to_ase

::: mdapy.interop.from_ase

::: mdapy.interop.to_ase_calculator

::: mdapy.interop.from_ase_calculator

::: mdapy.interop.to_ovito

::: mdapy.interop.from_ovito

::: mdapy.interop.to_pymatgen

::: mdapy.interop.from_pymatgen

::: mdapy.interop.to_freud

::: mdapy.interop.from_freud

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
