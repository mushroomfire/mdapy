# MDAPY, OVITO and freud

The analyses that MDAPY shares with [OVITO](https://www.ovito.org) and
[freud](https://freud.readthedocs.io), computed on the same systems with
the same parameters, from 4 000 to 2 million atoms. A program is timed only
where it gives the result of MDAPY, atom by atom.

## Results

A node of the Triton cluster of Aalto University, alone for the run: two
Intel Xeon Gold 6248 (40 cores, two NUMA nodes), 191 GB, Linux. Python
3.13, MDAPY 2.1.0rc1 built with GCC 12.3, OVITO 3.17.1
and freud 3.6.1 from pip, each with 40 threads. [Reading and writing
files](files.md) and [building structures](builders.md) ran on another
node, with Intel Xeon Gold 6148, so their times do not compare with these.

![Time of OVITO and of freud over the time of MDAPY for each case and size](images/comparison.png)

*How many times faster MDAPY is: a row for each case, the largest gain on
top, and a point for each size, larger and darker for more atoms; the number
on the right is the speed-up at the largest size. Left of 1×, the other program
is faster.*

### What they show

- **OVITO**: at 2 million atoms MDAPY is faster in every analysis they
  share. It builds neighbor lists 7.5 times faster in a quarter of the
  memory, Voronoi cells 22 to 24 times faster, computes Steinhardt
  parameters over Voronoi faces 502 times faster (1.3 s against 11
  minutes) and finds clusters 5.7 times faster; the other analyses take 1.2
  to 3.7 times less time. From 32 000 atoms on, only atomic strain is not
  faster: OVITO is faster below some 100 000 atoms (3 times at 4 000 and
  1.3 times at 32 000), and MDAPY is 1.1 times faster at 256 000 and 1.2
  times at 2 million.
- **freud**: from 256 000 atoms on, MDAPY is faster in every analysis:
  neighbor lists 23 to 26 times, in a seventh of the memory; the 12 nearest
  neighbors 6 to 9 times; Voronoi cells 32 to 39 times; Steinhardt
  parameters over Voronoi faces 39 to 42 times; Steinhardt parameters and
  the radial distribution function 2.9 to 4.7 times; cluster analysis 2.2
  to 2.5 times. At 32 000 atoms the two are on par in cluster analysis.
- **One neighbor search for several analyses**: the six analyses of copper
  from one list take 0.80 s at 2 million atoms, against 1.54 s for the six
  cases timed apart (0.11 s against 0.21 s at 256 000; the radial
  distribution function apart reaches 6 Å instead of 5 Å). OVITO's pipeline
  of the six takes about the sum of its cases (2.97 s against 3.11 s).
- **Voronoi faces on demand**: the volumes and coordination of 2 million
  cells take MDAPY 0.99 s and 205 MB; with every face, its neighbor, area
  and edges, 1.17 s and 1.7 GB.
- **Polyhedral template matching** finds the neighbors of each atom as it
  needs them: 174 MB beyond the system at 2 million atoms, against 248 MB
  for OVITO.
- **A trajectory**: the mean squared displacement of 20 frames read from a
  dump, most of the time spent reading, takes 1.6 to 1.7 times less time
  than with OVITO from 32 000 atoms on.
- **4 000 atoms** take a few milliseconds; MDAPY is faster there in every
  analysis but cluster analysis, Wigner–Seitz occupancies and atomic
  strain against OVITO, with the mean squared displacement on par, and
  cluster analysis against freud (0.0056 s against 0.0009 s), with the 12
  nearest neighbors on par.

### Times and memory

Every case at every size: the shortest of three times, and the memory each
computation took beyond the system, the peak resident memory of the
process minus its peak before the timings.

#### OVITO

| Case | Atoms | MDAPY 2.1.0rc1 (s) | OVITO 3.17.1 (s) | OVITO / MDAPY | MDAPY (MB) | OVITO (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Neighbor list, 5 Å | 4,000 | 0.0038 | 0.0084 | 2.2 | 2 | 12 |
| Neighbor list, 5 Å | 32,000 | 0.0088 | 0.035 | 4.0 | 23 | 93 |
| Neighbor list, 5 Å | 256,000 | 0.025 | 0.175 | 6.9 | 180 | 726 |
| Neighbor list, 5 Å | 2,048,000 | 0.185 | 1.381 | 7.5 | 1431 | 5416 |
| 12 nearest neighbors | 4,000 | 0.0025 | 0.0082 | 3.3 | 0 | 3 |
| 12 nearest neighbors | 32,000 | 0.0051 | 0.012 | 2.4 | 1 | 13 |
| 12 nearest neighbors | 256,000 | 0.030 | 0.049 | 1.6 | 55 | 111 |
| 12 nearest neighbors | 2,048,000 | 0.181 | 0.439 | 2.4 | 414 | 828 |
| Adaptive common neighbor analysis | 4,000 | 0.0018 | 0.015 | 8.3 | 0 | 0 |
| Adaptive common neighbor analysis | 32,000 | 0.0050 | 0.020 | 4.1 | 2 | 8 |
| Adaptive common neighbor analysis | 256,000 | 0.039 | 0.075 | 2.0 | 13 | 48 |
| Adaptive common neighbor analysis | 2,048,000 | 0.214 | 0.598 | 2.8 | 128 | 194 |
| Common neighbor analysis, cutoff 3.09 Å | 4,000 | 0.0029 | 0.0066 | 2.3 | 0 | 0 |
| Common neighbor analysis, cutoff 3.09 Å | 32,000 | 0.0045 | 0.0091 | 2.0 | 0 | 6 |
| Common neighbor analysis, cutoff 3.09 Å | 256,000 | 0.011 | 0.024 | 2.2 | 21 | 26 |
| Common neighbor analysis, cutoff 3.09 Å | 2,048,000 | 0.068 | 0.156 | 2.3 | 210 | 90 |
| Ackland–Jones analysis | 4,000 | 0.0012 | 0.0045 | 3.7 | 0 | 1 |
| Ackland–Jones analysis | 32,000 | 0.0069 | 0.010 | 1.5 | 0 | 8 |
| Ackland–Jones analysis | 256,000 | 0.040 | 0.067 | 1.7 | 14 | 45 |
| Ackland–Jones analysis | 2,048,000 | 0.301 | 0.559 | 1.9 | 128 | 194 |
| Polyhedral template matching | 4,000 | 0.015 | 0.026 | 1.7 | 0 | 1 |
| Polyhedral template matching | 32,000 | 0.036 | 0.040 | 1.1 | 1 | 12 |
| Polyhedral template matching | 256,000 | 0.235 | 0.275 | 1.2 | 24 | 40 |
| Polyhedral template matching | 2,048,000 | 1.777 | 2.222 | 1.3 | 174 | 248 |
| Centro-symmetry parameter, 12 neighbors | 4,000 | 0.0018 | 0.010 | 5.7 | 0 | 0 |
| Centro-symmetry parameter, 12 neighbors | 32,000 | 0.0058 | 0.015 | 2.6 | 2 | 8 |
| Centro-symmetry parameter, 12 neighbors | 256,000 | 0.033 | 0.058 | 1.8 | 12 | 40 |
| Centro-symmetry parameter, 12 neighbors | 2,048,000 | 0.252 | 0.483 | 1.9 | 142 | 185 |
| Diamond structure | 4,096 | 0.0026 | 0.0033 | 1.3 | 0 | 1 |
| Diamond structure | 32,768 | 0.0049 | 0.010 | 2.1 | 0 | 21 |
| Diamond structure | 262,144 | 0.034 | 0.068 | 2.0 | 19 | 76 |
| Diamond structure | 2,000,376 | 0.197 | 0.514 | 2.6 | 173 | 354 |
| Steinhardt q2–q6, 12 neighbors | 4,000 | 0.0021 | 0.014 | 6.7 | 0 | 0 |
| Steinhardt q2–q6, 12 neighbors | 32,000 | 0.0066 | 0.017 | 2.5 | 1 | 7 |
| Steinhardt q2–q6, 12 neighbors | 256,000 | 0.047 | 0.062 | 1.3 | 22 | 57 |
| Steinhardt q2–q6, 12 neighbors | 2,048,000 | 0.368 | 0.518 | 1.4 | 141 | 155 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 4,000 | 0.0033 | 0.295 | 89.1 | 0 | 1 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 32,000 | 0.022 | 1.399 | 62.3 | 26 | 12 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 256,000 | 0.172 | 22.574 | 131.4 | 227 | 47 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 2,048,000 | 1.311 | 658.063 | 501.9 | 1736 | 139 |
| Radial distribution function, 6 Å, 200 bins | 4,000 | 0.0037 | 0.0060 | 1.6 | 0 | 0 |
| Radial distribution function, 6 Å, 200 bins | 32,000 | 0.0043 | 0.0095 | 2.2 | 0 | 4 |
| Radial distribution function, 6 Å, 200 bins | 256,000 | 0.029 | 0.045 | 1.6 | 29 | 18 |
| Radial distribution function, 6 Å, 200 bins | 2,048,000 | 0.199 | 0.355 | 1.8 | 142 | 69 |
| Bond angle distribution, 3 Å, 180 bins | 4,000 | 0.0037 | 0.014 | 3.8 | 0 | 2 |
| Bond angle distribution, 3 Å, 180 bins | 32,000 | 0.0083 | 0.020 | 2.4 | 1 | 35 |
| Bond angle distribution, 3 Å, 180 bins | 256,000 | 0.026 | 0.079 | 3.0 | 27 | 254 |
| Bond angle distribution, 3 Å, 180 bins | 2,048,000 | 0.203 | 0.595 | 2.9 | 236 | 1423 |
| Voronoi volumes and coordination | 4,000 | 0.0054 | 0.046 | 8.6 | 0 | 0 |
| Voronoi volumes and coordination | 32,000 | 0.022 | 0.372 | 16.7 | 1 | 2 |
| Voronoi volumes and coordination | 256,000 | 0.124 | 2.957 | 23.8 | 22 | 10 |
| Voronoi volumes and coordination | 2,048,000 | 0.991 | 23.851 | 24.1 | 205 | 91 |
| Voronoi cells with their faces | 4,000 | 0.0030 | 0.049 | 16.2 | 0 | 2 |
| Voronoi cells with their faces | 32,000 | 0.018 | 0.392 | 21.6 | 22 | 18 |
| Voronoi cells with their faces | 256,000 | 0.151 | 3.159 | 21.0 | 223 | 151 |
| Voronoi cells with their faces | 2,048,000 | 1.168 | 25.336 | 21.7 | 1682 | 989 |
| Cluster analysis, 2 Å | 3,998 | 0.0056 | 0.0028 | 0.5 | 0 | 0 |
| Cluster analysis, 2 Å | 31,966 | 0.0084 | 0.019 | 2.3 | 1 | 6 |
| Cluster analysis, 2 Å | 255,756 | 0.033 | 0.174 | 5.3 | 38 | 26 |
| Cluster analysis, 2 Å | 2,046,192 | 0.350 | 2.000 | 5.7 | 220 | 0 |
| Wigner–Seitz occupancies | 3,974 | 0.0014 | 0.0010 | 0.7 | 0 | 1 |
| Wigner–Seitz occupancies | 31,811 | 0.0027 | 0.0055 | 2.0 | 0 | 8 |
| Wigner–Seitz occupancies | 254,734 | 0.022 | 0.036 | 1.7 | 10 | 44 |
| Wigner–Seitz occupancies | 2,037,841 | 0.091 | 0.322 | 3.5 | 126 | 262 |
| Mean squared displacement, 20 frames read from a dump | 4,000 | 0.043 | 0.041 | 1.0 | 22 | 10 |
| Mean squared displacement, 20 frames read from a dump | 32,000 | 0.166 | 0.260 | 1.6 | 181 | 50 |
| Mean squared displacement, 20 frames read from a dump | 256,000 | 1.169 | 1.891 | 1.6 | 585 | 303 |
| Mean squared displacement, 20 frames read from a dump | 2,048,000 | 9.061 | 14.999 | 1.7 | 2448 | 862 |
| Six analyses of copper, one neighbor search | 4,000 | 0.0081 | 0.065 | 8.0 | 2 | 8 |
| Six analyses of copper, one neighbor search | 32,000 | 0.018 | 0.094 | 5.1 | 24 | 79 |
| Six analyses of copper, one neighbor search | 256,000 | 0.110 | 0.367 | 3.3 | 180 | 459 |
| Six analyses of copper, one neighbor search | 2,048,000 | 0.798 | 2.971 | 3.7 | 1432 | 1781 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 4,000 | 0.0056 | 0.020 | 3.6 | 2 | 1 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 32,000 | 0.014 | 0.025 | 1.8 | 19 | 12 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 256,000 | 0.058 | 0.095 | 1.6 | 197 | 84 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 2,048,000 | 0.427 | 0.753 | 1.8 | 1432 | 191 |
| Atomic strain, 3 Å | 4,000 | 0.0033 | 0.0011 | 0.4 | 0 | 0 |
| Atomic strain, 3 Å | 32,000 | 0.0043 | 0.0034 | 0.8 | 0 | 6 |
| Atomic strain, 3 Å | 256,000 | 0.013 | 0.014 | 1.1 | 46 | 45 |
| Atomic strain, 3 Å | 2,048,000 | 0.075 | 0.090 | 1.2 | 242 | 248 |

#### freud

| Case | Atoms | MDAPY 2.1.0rc1 (s) | freud 3.6.1 (s) | freud / MDAPY | MDAPY (MB) | freud (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Neighbor list, 5 Å | 4,000 | 0.0038 | 0.0075 | 2.0 | 2 | 21 |
| Neighbor list, 5 Å | 32,000 | 0.0088 | 0.067 | 7.7 | 23 | 200 |
| Neighbor list, 5 Å | 256,000 | 0.025 | 0.581 | 23.0 | 180 | 1593 |
| Neighbor list, 5 Å | 2,048,000 | 0.185 | 4.871 | 26.3 | 1431 | 9710 |
| 12 nearest neighbors | 4,000 | 0.0025 | 0.0023 | 0.9 | 0 | 2 |
| 12 nearest neighbors | 32,000 | 0.0051 | 0.020 | 3.9 | 1 | 53 |
| 12 nearest neighbors | 256,000 | 0.030 | 0.186 | 6.2 | 55 | 432 |
| 12 nearest neighbors | 2,048,000 | 0.181 | 1.646 | 9.1 | 414 | 3310 |
| Steinhardt q2–q6, 12 neighbors | 4,000 | 0.0021 | 0.0034 | 1.6 | 0 | 1 |
| Steinhardt q2–q6, 12 neighbors | 32,000 | 0.0066 | 0.022 | 3.4 | 1 | 15 |
| Steinhardt q2–q6, 12 neighbors | 256,000 | 0.047 | 0.199 | 4.2 | 22 | 127 |
| Steinhardt q2–q6, 12 neighbors | 2,048,000 | 0.368 | 1.713 | 4.7 | 141 | 951 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 4,000 | 0.0033 | 0.103 | 31.1 | 0 | 6 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 32,000 | 0.022 | 0.841 | 37.5 | 26 | 61 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 256,000 | 0.172 | 6.692 | 39.0 | 227 | 465 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 2,048,000 | 1.311 | 54.447 | 41.5 | 1736 | 3715 |
| Radial distribution function, 6 Å, 200 bins | 4,000 | 0.0037 | 0.0084 | 2.3 | 0 | 0 |
| Radial distribution function, 6 Å, 200 bins | 32,000 | 0.0043 | 0.017 | 3.9 | 0 | 34 |
| Radial distribution function, 6 Å, 200 bins | 256,000 | 0.029 | 0.083 | 2.9 | 29 | 62 |
| Radial distribution function, 6 Å, 200 bins | 2,048,000 | 0.199 | 0.724 | 3.6 | 142 | 231 |
| Voronoi volumes and coordination | 4,000 | 0.0054 | 0.074 | 13.8 | 0 | 6 |
| Voronoi volumes and coordination | 32,000 | 0.022 | 0.603 | 27.1 | 1 | 61 |
| Voronoi volumes and coordination | 256,000 | 0.124 | 4.814 | 38.8 | 22 | 465 |
| Voronoi volumes and coordination | 2,048,000 | 0.991 | 38.785 | 39.1 | 205 | 3692 |
| Voronoi cells with their faces | 4,000 | 0.0030 | 0.072 | 24.0 | 0 | 6 |
| Voronoi cells with their faces | 32,000 | 0.018 | 0.597 | 32.9 | 22 | 61 |
| Voronoi cells with their faces | 256,000 | 0.151 | 4.799 | 31.9 | 223 | 465 |
| Voronoi cells with their faces | 2,048,000 | 1.168 | 37.586 | 32.2 | 1682 | 3693 |
| Cluster analysis, 2 Å | 3,998 | 0.0056 | 0.0009 | 0.2 | 0 | 2 |
| Cluster analysis, 2 Å | 31,966 | 0.0084 | 0.0075 | 0.9 | 1 | 0 |
| Cluster analysis, 2 Å | 255,756 | 0.033 | 0.072 | 2.2 | 38 | 25 |
| Cluster analysis, 2 Å | 2,046,192 | 0.350 | 0.867 | 2.5 | 220 | 250 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 4,000 | 0.0056 | 0.012 | 2.2 | 2 | 32 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 32,000 | 0.014 | 0.040 | 2.8 | 19 | 17 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 256,000 | 0.058 | 0.231 | 4.0 | 197 | 149 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 2,048,000 | 0.427 | 1.797 | 4.2 | 1432 | 872 |

## Cases

| Case | System | MDAPY | OVITO | freud |
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
| Voronoi cells with their faces | fcc Cu | `VoronoiAnalysis(faces=True)` | `VoronoiAnalysisModifier(generate_bonds=True)` | `Voronoi()` |
| Cluster analysis, 2 Å | random points | `ClusterAnalysis(cutoff=2.0)` | `ClusterAnalysisModifier(cutoff=2.0, sort_by_size=True)` | `Cluster()` with `{"r_max": 2.0}` |
| Wigner–Seitz occupancies | fcc Cu, 0.5 % vacancies | `WignerSeitz(reference)` | `WignerSeitzAnalysisModifier()` | |
| Atomic strain, 3 Å | strained fcc Cu | `AtomicStrain(reference, 3.0)` | `AtomicStrainModifier(cutoff=3.0)` | |
| Mean squared displacement, 20 frames | fcc Cu walking at random, read from a dump | `MeanSquaredDisplacement(unwrap=False).compute(read_trajectory(path))` | `DiffusionModifier()` over the frames of the file | |
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
- Before the timings, a separate process computes the case with MDAPY and
  with the other program and compares the results. With OVITO, which
  computes in double precision as MDAPY does: the same structure types,
  cluster sizes, occupancies and bins, floating-point values within 1e-10
  (1e-8 for Steinhardt parameters weighted by Voronoi faces, whose areas
  come from two tessellations), for neighbor lists the same number of
  neighbors of each atom and the same sum of their distances, and for
  Voronoi cells the same volumes and numbers of faces, and with the faces
  as many of OVITO's bonds at each atom, but for faces of no area that one
  tessellation of millions of atoms finds and the other not (at most one
  atom in 100 000). freud computes in single precision, which
  rounds positions in a box of 600 Å by some 3e-5 Å: values agree within
  1e-4; the numbers of neighbors and the bins of the radial distribution
  function within the pairs closer than 1e-4 Å to a cutoff or the edge of a
  bin; Steinhardt parameters of the 12 nearest agree with those MDAPY
  computes from the positions freud has, within 3e-4, for the atoms whose
  12th and 13th neighbors are farther apart than 1e-4 Å, nearly all (freud's
  harmonics in single precision differ by 2e-5 for all but a few atoms in a
  million, by up to 2e-4 for those); every Voronoi face of freud above
  0.01 Å² is one of MDAPY, and the faces freud leaves out, of distant
  neighbors, are all below 0.02 Å², so its Steinhardt parameters over Voronoi faces agree within 1e-3.
  The mean squared displacements agree within 1e-10. A program that
  differs is not timed.
- Each process first computes the case for 500 atoms, so that loading
  libraries and starting threads are not timed, then three times for the
  system; the shortest time counts.
- The time runs from the call to the result, with every neighbor search.
  Making the system and converting it into OVITO's data collection or
  freud's box and points are not timed. OVITO computes a new pipeline each
  time, so nothing is cached.
- A process that grows beyond 170 GB or runs for more than 15 minutes is
  stopped (80 minutes for Steinhardt parameters over Voronoi faces at 2
  million atoms, which take OVITO 11 minutes each time), and a size is
  skipped where the previous one predicts more memory.

## Running it

The comparison is `benchmarks/comparison.py` in the source repository. It
needs OVITO (`pip install ovito`), freud (`pip install freud-analysis`)
and, for the figure, matplotlib.

```bash
python benchmarks/comparison.py run                     # every case
python benchmarks/comparison.py run cna voronoi --sizes 4000 32000 --threads 1
python benchmarks/comparison.py run --sizes 4000 32000 256000 2048000 \
    --threads 40 --memory 170 --timeout 900                # as above
python benchmarks/comparison.py table results.json      # Markdown tables
python benchmarks/comparison.py plot results.json ratio.png
```

`run` appends one line of JSON per case and size to `results.json` (the
option `--output`): the machine, the versions, the agreement of each
program, and the times and peak memory of each. A new case is a `Case` in
`_cases()`: how to make its system, the MDAPY call, and for each other
program its call and how to compare its result with that of MDAPY.
