# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Analyses of the properties of systems."""

from mdapy.analysis._binning import REDUCTIONS, SpatialBinning, SpatialBinningResult
from mdapy.analysis._bonds import (
    BondAngleDistribution,
    BondAngleResult,
    ClusterAnalysis,
    ClusterResult,
    WarrenCowley,
    WarrenCowleyResult,
)
from mdapy.analysis._correlation import SpatialCorrelation, SpatialCorrelationResult
from mdapy.analysis._dislocations import (
    CrystalClusters,
    DislocationAnalysis,
    DislocationLine,
    DislocationResult,
)
from mdapy.analysis._displacement import AtomicDisplacement, AtomicDisplacementResult
from mdapy.analysis._distribution import (
    CoordinationNumber,
    CoordinationResult,
    RadialDistributionFunction,
    RadialDistributionResult,
    StructureFactor,
    StructureFactorResult,
)
from mdapy.analysis._dynamics import (
    DisplacementResult,
    LindemannIndex,
    LindemannResult,
    MeanSquaredDisplacement,
    TimeAverage,
    TimeAverageResult,
    trajectory_lines,
)
from mdapy.analysis._elastic_strain import ElasticStrain, ElasticStrainResult
from mdapy.analysis._entropy import EntropyFingerprint, EntropyResult
from mdapy.analysis._isosurface import isosurface
from mdapy.analysis._polyhedra import (
    CoordinationPolyhedra,
    CoordinationPolyhedraResult,
)
from mdapy.analysis._species import ChemicalSpecies, SpeciesResult
from mdapy.analysis._steinhardt import SteinhardtParameters, SteinhardtResult
from mdapy.analysis._strain import AtomicStrain, StrainResult
from mdapy.analysis._surface_mesh import SurfaceMesh, SurfaceMeshResult, SurfaceRegions
from mdapy.analysis._symmetry import SpaceGroup, SpaceGroupResult
from mdapy.analysis._temperature import AtomicTemperature, TemperatureResult
from mdapy.analysis._voronoi import (
    VoronoiAnalysis,
    VoronoiPolyhedra,
    VoronoiResult,
)
from mdapy.analysis._wigner_seitz import WignerSeitz, WignerSeitzResult

__all__ = [
    "REDUCTIONS",
    "AtomicDisplacement",
    "AtomicDisplacementResult",
    "SpatialBinning",
    "SpatialBinningResult",
    "SpatialCorrelation",
    "SpatialCorrelationResult",
    "isosurface",
    "DisplacementResult",
    "LindemannIndex",
    "LindemannResult",
    "MeanSquaredDisplacement",
    "TimeAverage",
    "TimeAverageResult",
    "trajectory_lines",
    "AtomicStrain",
    "AtomicTemperature",
    "BondAngleDistribution",
    "BondAngleResult",
    "ChemicalSpecies",
    "ClusterAnalysis",
    "ClusterResult",
    "CrystalClusters",
    "DislocationAnalysis",
    "DislocationLine",
    "DislocationResult",
    "ElasticStrain",
    "ElasticStrainResult",
    "CoordinationNumber",
    "CoordinationPolyhedra",
    "CoordinationPolyhedraResult",
    "CoordinationResult",
    "EntropyFingerprint",
    "EntropyResult",
    "RadialDistributionFunction",
    "RadialDistributionResult",
    "SteinhardtParameters",
    "SteinhardtResult",
    "SpaceGroup",
    "SpaceGroupResult",
    "StrainResult",
    "StructureFactor",
    "StructureFactorResult",
    "SurfaceMesh",
    "SurfaceMeshResult",
    "SurfaceRegions",
    "TemperatureResult",
    "VoronoiAnalysis",
    "VoronoiPolyhedra",
    "VoronoiResult",
    "WarrenCowley",
    "WarrenCowleyResult",
    "SpeciesResult",
    "WignerSeitz",
    "WignerSeitzResult",
]
