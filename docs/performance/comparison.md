# mdapy, OVITO and freud

The analyses that mdapy shares with [OVITO](https://www.ovito.org) and
[freud](https://freud.readthedocs.io), computed on the same systems with
the same parameters, from 4 000 to 16 million atoms. A program is timed only
where it gives the result of mdapy, atom by atom.

## Results

A node of the Triton cluster of Aalto University, alone for the run: two
Intel Xeon Gold 6248 (40 cores, two NUMA nodes), 191 GB, Linux. Python 3.13,
mdapy 2.0.0.dev0 built with GCC 12.3, OVITO 3.17.0 and freud 3.6.1 from
pip, each with 40 threads.

![Time of OVITO and of freud over the time of mdapy for each case and size](images/comparison.png)

### What they show

- **OVITO**, from 32 000 atoms on: mdapy builds neighbor lists 4 to 6 times
  faster in a quarter of the memory, Voronoi cells 16 times faster and
  Steinhardt parameters over Voronoi faces 25 to 200 times; the other
  analyses take 1.2 to 3.2 times less time, cluster analysis 13 times less at
  16 million atoms. Two are not faster: common neighbor analysis with a
  fixed cutoff is on par from 256 000 atoms on (OVITO 1.3 times faster at
  256 000), and atomic strain takes 1.2 to 2.5 times as long as OVITO's.
- **freud**: neighbor lists 8 to 22 times faster in a sixth of the memory,
  the 12 nearest neighbors 4 to 7 times, Voronoi cells 26 times, Steinhardt
  parameters and the radial distribution function 2.4 to 4.6 times. freud's
  cluster analysis is faster below 256 000 atoms (1.6 times at 32 000) and
  slower above.
- **One neighbor search for several analyses**: the six analyses of copper
  from one list take half the time of the six cases timed apart (7.8 s
  against 15.1 s at 16 million atoms, 0.97 s against 1.85 s at 2 million;
  the radial distribution function apart reaches 6 Å instead of 5 Å).
  OVITO's pipeline of the six takes about the sum of its cases (25.8 s
  against 26.7 s).
- **Memory**: mdapy's Voronoi analysis returns every face of every cell,
  with the atom across it, its area and its edges (11.8 GB at 16 million
  atoms); OVITO's keeps only the volumes and coordination unless asked for
  bonds or polyhedra.
- **4 000 atoms** take a few milliseconds, where starting the threads
  counts: OVITO is faster there for cluster analysis, Wigner–Seitz
  occupancies and atomic strain, freud for cluster analysis.
- **Not timed**: freud's Steinhardt parameters of the 12 nearest at 16
  million atoms, alone and with the radial distribution function, were
  compared under an earlier rule, values within 1e-4 of mdapy's, which ten
  of 82 million values, of six atoms, missed by up to 1.8e-4; the rule
  below came after, and they were not run again. Steinhardt parameters
  over Voronoi faces were not run at 16 million atoms: OVITO took 678 s at
  2 million.

### Times and memory

Every case at every size: the shortest of three times, and the memory each
computation took beyond the system, the peak resident memory of the
process minus its peak before the timings.

#### OVITO

| Case | Atoms | mdapy 2.0.0.dev0 (s) | OVITO 3.17.0 (s) | OVITO / mdapy | mdapy (MB) | OVITO (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Neighbor list, 5 Å | 4,000 | 0.005 | 0.008 | 1.54 | 0 | 10 |
| Neighbor list, 5 Å | 32,000 | 0.008 | 0.034 | 4.13 | 18 | 131 |
| Neighbor list, 5 Å | 256,000 | 0.042 | 0.179 | 4.26 | 185 | 758 |
| Neighbor list, 5 Å | 2,048,000 | 0.250 | 1.368 | 5.48 | 1337 | 5324 |
| Neighbor list, 5 Å | 16,384,000 | 1.857 | 11.200 | 6.03 | 10310 | 41419 |
| 12 nearest neighbors | 4,000 | 0.001 | 0.009 | 6.27 | 0 | 0 |
| 12 nearest neighbors | 32,000 | 0.005 | 0.012 | 2.33 | 2 | 12 |
| 12 nearest neighbors | 256,000 | 0.034 | 0.053 | 1.54 | 41 | 101 |
| 12 nearest neighbors | 2,048,000 | 0.274 | 0.442 | 1.62 | 330 | 752 |
| 12 nearest neighbors | 16,384,000 | 2.203 | 3.889 | 1.77 | 2485 | 5544 |
| Adaptive common neighbor analysis | 4,000 | 0.003 | 0.015 | 5.84 | 0 | 0 |
| Adaptive common neighbor analysis | 32,000 | 0.008 | 0.021 | 2.85 | 0 | 9 |
| Adaptive common neighbor analysis | 256,000 | 0.048 | 0.074 | 1.54 | 1 | 50 |
| Adaptive common neighbor analysis | 2,048,000 | 0.381 | 0.598 | 1.57 | 4 | 118 |
| Adaptive common neighbor analysis | 16,384,000 | 3.179 | 5.142 | 1.62 | 0 | 43 |
| Common neighbor analysis, cutoff 3.09 Å | 4,000 | 0.003 | 0.007 | 2.54 | 0 | 0 |
| Common neighbor analysis, cutoff 3.09 Å | 32,000 | 0.006 | 0.009 | 1.59 | 3 | 5 |
| Common neighbor analysis, cutoff 3.09 Å | 256,000 | 0.031 | 0.023 | 0.76 | 32 | 29 |
| Common neighbor analysis, cutoff 3.09 Å | 2,048,000 | 0.243 | 0.233 | 0.96 | 136 | 2 |
| Common neighbor analysis, cutoff 3.09 Å | 16,384,000 | 1.887 | 1.866 | 0.99 | 904 | 0 |
| Ackland–Jones analysis | 4,000 | 0.002 | 0.005 | 2.05 | 0 | 1 |
| Ackland–Jones analysis | 32,000 | 0.006 | 0.010 | 1.68 | 0 | 8 |
| Ackland–Jones analysis | 256,000 | 0.040 | 0.071 | 1.77 | 0 | 50 |
| Ackland–Jones analysis | 2,048,000 | 0.326 | 0.574 | 1.76 | 4 | 118 |
| Ackland–Jones analysis | 16,384,000 | 2.725 | 4.923 | 1.81 | 0 | 43 |
| Polyhedral template matching | 4,000 | 0.013 | 0.027 | 2.02 | 4 | 1 |
| Polyhedral template matching | 32,000 | 0.033 | 0.040 | 1.21 | 13 | 12 |
| Polyhedral template matching | 256,000 | 0.202 | 0.277 | 1.37 | 73 | 39 |
| Polyhedral template matching | 2,048,000 | 1.578 | 2.220 | 1.41 | 505 | 178 |
| Polyhedral template matching | 16,384,000 | 12.620 | 19.354 | 1.53 | 3909 | 51 |
| Centro-symmetry parameter, 12 neighbors | 4,000 | 0.002 | 0.010 | 6.54 | 0 | 0 |
| Centro-symmetry parameter, 12 neighbors | 32,000 | 0.005 | 0.015 | 2.74 | 0 | 5 |
| Centro-symmetry parameter, 12 neighbors | 256,000 | 0.035 | 0.058 | 1.66 | 0 | 42 |
| Centro-symmetry parameter, 12 neighbors | 2,048,000 | 0.280 | 0.478 | 1.71 | 8 | 125 |
| Centro-symmetry parameter, 12 neighbors | 16,384,000 | 2.280 | 4.218 | 1.85 | 0 | 24 |
| Diamond structure | 4,096 | 0.002 | 0.004 | 1.50 | 0 | 2 |
| Diamond structure | 32,768 | 0.007 | 0.012 | 1.66 | 0 | 21 |
| Diamond structure | 262,144 | 0.042 | 0.069 | 1.64 | 10 | 80 |
| Diamond structure | 2,000,376 | 0.311 | 0.524 | 1.68 | 111 | 271 |
| Diamond structure | 16,387,064 | 2.684 | 4.422 | 1.65 | 634 | 1653 |
| Steinhardt q2–q6, 12 neighbors | 4,000 | 0.002 | 0.014 | 6.59 | 0 | 0 |
| Steinhardt q2–q6, 12 neighbors | 32,000 | 0.007 | 0.017 | 2.55 | 1 | 10 |
| Steinhardt q2–q6, 12 neighbors | 256,000 | 0.048 | 0.063 | 1.32 | 10 | 58 |
| Steinhardt q2–q6, 12 neighbors | 2,048,000 | 0.373 | 0.515 | 1.38 | 60 | 79 |
| Steinhardt q2–q6, 12 neighbors | 16,384,000 | 3.021 | 4.414 | 1.46 | 172 | 169 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 4,000 | 0.010 | 0.304 | 30.00 | 3 | 1 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 32,000 | 0.057 | 1.398 | 24.64 | 53 | 13 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 256,000 | 0.388 | 22.697 | 58.47 | 427 | 42 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 2,048,000 | 3.318 | 677.980 | 204.32 | 3133 | 56 |
| Radial distribution function, 6 Å, 200 bins | 4,000 | 0.005 | 0.005 | 1.00 | 0 | 0 |
| Radial distribution function, 6 Å, 200 bins | 32,000 | 0.007 | 0.009 | 1.24 | 2 | 4 |
| Radial distribution function, 6 Å, 200 bins | 256,000 | 0.030 | 0.039 | 1.31 | 24 | 18 |
| Radial distribution function, 6 Å, 200 bins | 2,048,000 | 0.222 | 0.298 | 1.35 | 72 | 0 |
| Radial distribution function, 6 Å, 200 bins | 16,384,000 | 1.756 | 2.602 | 1.48 | 399 | 0 |
| Bond angle distribution, 3 Å, 180 bins | 4,000 | 0.002 | 0.014 | 5.72 | 1 | 2 |
| Bond angle distribution, 3 Å, 180 bins | 32,000 | 0.007 | 0.021 | 3.21 | 4 | 37 |
| Bond angle distribution, 3 Å, 180 bins | 256,000 | 0.032 | 0.079 | 2.43 | 21 | 256 |
| Bond angle distribution, 3 Å, 180 bins | 2,048,000 | 0.263 | 0.596 | 2.27 | 166 | 1355 |
| Bond angle distribution, 3 Å, 180 bins | 16,384,000 | 2.108 | 5.421 | 2.57 | 1163 | 7005 |
| Voronoi volumes and coordination | 4,000 | 0.005 | 0.047 | 8.87 | 1 | 0 |
| Voronoi volumes and coordination | 32,000 | 0.023 | 0.370 | 16.28 | 22 | 2 |
| Voronoi volumes and coordination | 256,000 | 0.174 | 3.017 | 17.31 | 208 | 8 |
| Voronoi volumes and coordination | 2,048,000 | 1.475 | 23.879 | 16.19 | 1547 | 22 |
| Voronoi volumes and coordination | 16,384,000 | 11.833 | 188.808 | 15.96 | 11836 | 2 |
| Cluster analysis, 2 Å | 3,998 | 0.004 | 0.002 | 0.63 | 10 | 0 |
| Cluster analysis, 2 Å | 31,966 | 0.012 | 0.018 | 1.50 | 1 | 6 |
| Cluster analysis, 2 Å | 255,756 | 0.065 | 0.157 | 2.43 | 72 | 25 |
| Cluster analysis, 2 Å | 2,046,192 | 0.720 | 2.010 | 2.79 | 435 | 0 |
| Cluster analysis, 2 Å | 16,369,777 | 7.073 | 93.526 | 13.22 | 1792 | 0 |
| Wigner–Seitz occupancies | 3,974 | 0.001 | 0.001 | 0.82 | 0 | 1 |
| Wigner–Seitz occupancies | 31,811 | 0.003 | 0.006 | 1.87 | 0 | 8 |
| Wigner–Seitz occupancies | 254,734 | 0.019 | 0.036 | 1.90 | 0 | 36 |
| Wigner–Seitz occupancies | 2,037,841 | 0.137 | 0.315 | 2.30 | 15 | 172 |
| Wigner–Seitz occupancies | 16,302,490 | 1.273 | 2.390 | 1.88 | 0 | 416 |
| Six analyses of copper, one neighbor search | 4,000 | 0.007 | 0.064 | 9.79 | 1 | 8 |
| Six analyses of copper, one neighbor search | 32,000 | 0.025 | 0.099 | 3.95 | 20 | 85 |
| Six analyses of copper, one neighbor search | 256,000 | 0.131 | 0.372 | 2.84 | 187 | 481 |
| Six analyses of copper, one neighbor search | 2,048,000 | 0.971 | 2.933 | 3.02 | 1339 | 1663 |
| Six analyses of copper, one neighbor search | 16,384,000 | 7.772 | 25.770 | 3.32 | 10310 | 8348 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 4,000 | 0.011 | 0.019 | 1.83 | 1 | 1 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 32,000 | 0.014 | 0.024 | 1.78 | 19 | 10 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 256,000 | 0.071 | 0.091 | 1.27 | 187 | 83 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 2,048,000 | 0.485 | 0.725 | 1.49 | 1339 | 134 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 16,384,000 | 3.783 | 6.039 | 1.60 | 10310 | 391 |
| Atomic strain, 3 Å | 4,000 | 0.003 | 0.001 | 0.44 | 0 | 0 |
| Atomic strain, 3 Å | 32,000 | 0.005 | 0.003 | 0.59 | 5 | 6 |
| Atomic strain, 3 Å | 256,000 | 0.023 | 0.013 | 0.57 | 76 | 44 |
| Atomic strain, 3 Å | 2,048,000 | 0.210 | 0.085 | 0.40 | 466 | 231 |
| Atomic strain, 3 Å | 16,384,000 | 1.367 | 1.133 | 0.83 | 3715 | 1520 |

#### freud

| Case | Atoms | mdapy 2.0.0.dev0 (s) | freud 3.6.1 (s) | freud / mdapy | mdapy (MB) | freud (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Neighbor list, 5 Å | 4,000 | 0.005 | 0.008 | 1.40 | 0 | 23 |
| Neighbor list, 5 Å | 32,000 | 0.008 | 0.064 | 7.87 | 18 | 200 |
| Neighbor list, 5 Å | 256,000 | 0.042 | 0.584 | 13.88 | 185 | 1582 |
| Neighbor list, 5 Å | 2,048,000 | 0.250 | 4.842 | 19.40 | 1337 | 9671 |
| Neighbor list, 5 Å | 16,384,000 | 1.857 | 40.854 | 22.00 | 10310 | 61557 |
| 12 nearest neighbors | 4,000 | 0.001 | 0.002 | 1.75 | 0 | 4 |
| 12 nearest neighbors | 32,000 | 0.005 | 0.022 | 4.23 | 2 | 52 |
| 12 nearest neighbors | 256,000 | 0.034 | 0.187 | 5.46 | 41 | 428 |
| 12 nearest neighbors | 2,048,000 | 0.274 | 1.653 | 6.04 | 330 | 3269 |
| 12 nearest neighbors | 16,384,000 | 2.203 | 14.442 | 6.56 | 2485 | 18948 |
| Steinhardt q2–q6, 12 neighbors | 4,000 | 0.002 | 0.003 | 1.38 | 0 | 2 |
| Steinhardt q2–q6, 12 neighbors | 32,000 | 0.007 | 0.022 | 3.21 | 1 | 15 |
| Steinhardt q2–q6, 12 neighbors | 256,000 | 0.048 | 0.187 | 3.91 | 10 | 127 |
| Steinhardt q2–q6, 12 neighbors | 2,048,000 | 0.373 | 1.716 | 4.61 | 60 | 917 |
| Steinhardt q2–q6, 12 neighbors | 16,384,000 | 3.021 | not timed, see below |  | 172 |  |
| Steinhardt q2–q6, Voronoi faces weighted by area | 4,000 | 0.010 | 0.103 | 10.15 | 3 | 7 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 32,000 | 0.057 | 0.845 | 14.89 | 53 | 61 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 256,000 | 0.388 | 6.906 | 17.79 | 427 | 461 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 2,048,000 | 3.318 | 54.456 | 16.41 | 3133 | 3672 |
| Radial distribution function, 6 Å, 200 bins | 4,000 | 0.005 | 0.009 | 1.74 | 0 | 31 |
| Radial distribution function, 6 Å, 200 bins | 32,000 | 0.007 | 0.017 | 2.38 | 2 | 34 |
| Radial distribution function, 6 Å, 200 bins | 256,000 | 0.030 | 0.091 | 3.07 | 24 | 53 |
| Radial distribution function, 6 Å, 200 bins | 2,048,000 | 0.222 | 0.730 | 3.30 | 72 | 208 |
| Radial distribution function, 6 Å, 200 bins | 16,384,000 | 1.756 | 6.594 | 3.76 | 399 | 1171 |
| Voronoi volumes and coordination | 4,000 | 0.005 | 0.074 | 14.10 | 1 | 6 |
| Voronoi volumes and coordination | 32,000 | 0.023 | 0.602 | 26.47 | 22 | 61 |
| Voronoi volumes and coordination | 256,000 | 0.174 | 4.844 | 27.79 | 208 | 461 |
| Voronoi volumes and coordination | 2,048,000 | 1.475 | 38.822 | 26.32 | 1547 | 3651 |
| Voronoi volumes and coordination | 16,384,000 | 11.833 | 311.553 | 26.33 | 11836 | 29378 |
| Cluster analysis, 2 Å | 3,998 | 0.004 | 0.001 | 0.23 | 10 | 0 |
| Cluster analysis, 2 Å | 31,966 | 0.012 | 0.007 | 0.62 | 1 | 0 |
| Cluster analysis, 2 Å | 255,756 | 0.065 | 0.070 | 1.08 | 72 | 26 |
| Cluster analysis, 2 Å | 2,046,192 | 0.720 | 0.861 | 1.20 | 435 | 250 |
| Cluster analysis, 2 Å | 16,369,777 | 7.073 | 9.657 | 1.37 | 1792 | 0 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 4,000 | 0.011 | 0.012 | 1.11 | 1 | 32 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 32,000 | 0.014 | 0.035 | 2.55 | 19 | 18 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 256,000 | 0.071 | 0.222 | 3.11 | 187 | 143 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 2,048,000 | 0.485 | 1.924 | 3.96 | 1339 | 848 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 16,384,000 | 3.783 | not timed, see below |  | 10310 |  |

## Cases

| Case | System | mdapy | OVITO | freud |
|---|---|---|---|---|
| Neighbor list, 5 Å | fcc Cu | `NeighborList.build(system, 5.0)` | `CutoffNeighborFinder(5.0, data).find_all()` | `AABBQuery(box, points).query(points, {"r_max": 5.0, "exclude_ii": True}).toNeighborList()` |
| 12 nearest neighbors | fcc Cu | `NeighborList.build(system, num_neighbors=12)` | `NearestNeighborFinder(12, data).find_all()` | the same query with `{"num_neighbors": 12, "exclude_ii": True}` |
| Adaptive common neighbor analysis | fcc Cu | `CommonNeighborAnalysis()` | `CommonNeighborAnalysisModifier()` | |
| Common neighbor analysis, cutoff 3.09 Å | fcc Cu | `CommonNeighborAnalysis(3.086)` | `CommonNeighborAnalysisModifier(mode=FixedCutoff, cutoff=3.086)` | |
| Ackland–Jones analysis | fcc Cu | `AcklandJonesAnalysis()` | `AcklandJonesModifier()` | |
| Polyhedral template matching | fcc Cu | `PolyhedralTemplateMatching()` | `PolyhedralTemplateMatchingModifier()` | |
| Centro-symmetry parameter | fcc Cu | `CentroSymmetryParameter(12)` | `CentroSymmetryModifier(num_neighbors=12)` | |
| Diamond structure | diamond Si | `DiamondStructure()` | `IdentifyDiamondModifier()` | |
| Steinhardt q2–q6, 12 neighbors | fcc Cu | `SteinhardtParameters((2, 3, 4, 5, 6), num_neighbors=12)` | `SteinhardtParametersModifier(neighbor_mode=NearestNeighbors, num_neighbors=12, max_l=6)` | `Steinhardt([2, 3, 4, 5, 6])` with `{"num_neighbors": 12, "exclude_ii": True}` |
| Steinhardt q2–q6, Voronoi | fcc Cu | `SteinhardtParameters((2, 3, 4, 5, 6), voronoi=True, weighted=True)` | `SteinhardtParametersModifier(neighbor_mode=Voronoi, max_l=6)` | `Voronoi()`, then `Steinhardt([2, 3, 4, 5, 6], weighted=True)` on its neighbors |
| Radial distribution function | fcc Cu | `RadialDistributionFunction(6.0, bins=200)` | `CoordinationAnalysisModifier(cutoff=6.0, number_of_bins=200)` | `RDF(bins=200, r_max=6.0)` |
| Bond angle distribution | fcc Cu | `BondAngleDistribution(cutoff=3.0, bins=180)` | `CreateBondsModifier(cutoff=3.0)`, `BondAngleDistributionModifier(bins=180)` | |
| Voronoi volumes and coordination | fcc Cu | `VoronoiAnalysis()` | `VoronoiAnalysisModifier()` | `Voronoi()` |
| Cluster analysis, 2 Å | random points | `ClusterAnalysis(cutoff=2.0)` | `ClusterAnalysisModifier(cutoff=2.0, sort_by_size=True)` | `Cluster()` with `{"r_max": 2.0}` |
| Wigner–Seitz occupancies | fcc Cu, 0.5 % vacancies | `WignerSeitz(reference)` | `WignerSeitzAnalysisModifier()` | |
| Atomic strain, 3 Å | strained fcc Cu | `AtomicStrain(reference, 3.0)` | `AtomicStrainModifier(cutoff=3.0)` | |
| Six analyses of copper, one neighbor search | fcc Cu | `NeighborList.build(system, 5.0)` given to the analyses of adaptive common neighbors, Ackland–Jones, centro-symmetry, Steinhardt (12 neighbors), the radial distribution function within 5 Å and bond angles within 3 Å | one pipeline of the six modifiers | |
| Steinhardt q2–q6 and the RDF, one neighbor search | fcc Cu | `NeighborList.build(system, 5.0)` given to both | one pipeline of the two modifiers | one `AABBQuery` for both |

The fcc copper has its atoms displaced at random by 0.1 Å along each axis,
as at room temperature; the silicon by 0.05 Å. The random points have the
density of copper and, with a cutoff of 2 Å, are close to percolation, so
that clusters of every size form; points with a neighbor within 1e-4 Å of
2 Å are left out, so that single precision gives the same bonds. OVITO's
coordination analysis also counts the neighbors of each atom along with the
radial distribution function.

## How it is measured

- Each program runs in a process of its own, with the same number of
  threads (`MDAPY_NUM_THREADS`, `OVITO_THREAD_COUNT` and
  `freud.parallel.set_num_threads`).
- Before the timings, a separate process computes the case with mdapy and
  with the other program and compares the results. With OVITO, which
  computes in double precision as mdapy does: the same structure types,
  cluster sizes, occupancies and bins, floating-point values within 1e-10
  (1e-8 for Steinhardt parameters weighted by Voronoi faces, whose areas
  come from two tessellations), for neighbor lists the same number of
  neighbors of each atom and the same sum of their distances, and for
  Voronoi cells the same volumes and numbers of faces, but for faces of no
  area that one tessellation of millions of atoms finds and the other not
  (at most one atom in 100 000). freud computes in single precision, which
  rounds positions in a box of 600 Å by some 3e-5 Å: values agree within
  1e-4; the numbers of neighbors and the bins of the radial distribution
  function within the pairs closer than 1e-4 Å to a cutoff or the edge of a
  bin; Steinhardt parameters of the 12 nearest agree with those mdapy
  computes from the positions freud has, within 3e-4, for the atoms whose
  12th and 13th neighbors are farther apart than 1e-4 Å, nearly all (freud's
  harmonics in single precision differ by 2e-5 for all but a few atoms in a
  million, by up to 2e-4 for those); every Voronoi face of freud above
  0.01 Å² is one of mdapy, and the faces freud leaves out, of distant
  neighbors, are all below 0.02 Å² (0.1 % of the faces at 16 million atoms),
  so its Steinhardt parameters over Voronoi faces agree within 1e-3. A
  program that differs is not timed.
- Each process first computes the case for 500 atoms, so that loading
  libraries and starting threads are not timed, then three times for the
  system; the shortest time counts.
- The time runs from the call to the result, with every neighbor search.
  Making the system and converting it into OVITO's data collection or
  freud's box and points are not timed. OVITO computes a new pipeline each
  time, so nothing is cached.
- A process that grows beyond 170 GB or runs for more than four hours is
  stopped, and a size is skipped where the previous one predicts more
  memory.

## Running it

The comparison is `benchmarks/comparison.py` in the source repository. It
needs OVITO (`pip install ovito`), freud (`pip install freud-analysis`)
and, for the figure, matplotlib.

```bash
python benchmarks/comparison.py run                     # every case
python benchmarks/comparison.py run cna voronoi --sizes 4000 32000 --threads 1
python benchmarks/comparison.py run --sizes 4000 32000 256000 2048000 16384000 \
    --threads 40 --memory 170 --timeout 14400              # as above
python benchmarks/comparison.py table results.json      # Markdown tables
python benchmarks/comparison.py plot results.json ratio.png
```

`run` appends one line of JSON per case and size to `results.json` (the
option `--output`): the machine, the versions, the agreement of each
program, and the times and peak memory of each. A new case is a `Case` in
`_cases()`: how to make its system, the mdapy call, and for each other
program its call and how to compare its result with that of mdapy.
