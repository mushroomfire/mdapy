# mdapy, OVITO and freud

The analyses that mdapy shares with [OVITO](https://www.ovito.org) and
[freud](https://freud.readthedocs.io), computed on the same systems with
the same parameters. A program is timed only where it gives the result of
mdapy, atom by atom.

## Results

Apple M5 (4 performance and 6 efficiency cores, 24 GB), macOS, Python
3.13, mdapy 2.0.0.dev0, OVITO 3.17.0 and freud 3.6.1 from pip, each with
10 threads.

![Time of OVITO and of freud over the time of mdapy for each case and size](images/comparison.png)

At the largest size timed for each case: the shortest of three times, and
the memory each computation took beyond the system: the peak resident
memory of the process minus its peak before the timings.

### OVITO

| Case | Atoms | mdapy 2.0.0.dev0 (s) | OVITO 3.17.0 (s) | OVITO / mdapy | mdapy (MB) | OVITO (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Neighbor list, 5 Å | 256,000 | 0.056 | 0.071 | 1.26 | 199 | 760 |
| 12 nearest neighbors | 2,048,000 | 0.349 | 0.431 | 1.23 | 438 | 853 |
| Adaptive common neighbor analysis | 2,048,000 | 0.505 | 0.675 | 1.34 | 88 | 147 |
| Common neighbor analysis, cutoff 3.09 Å | 2,048,000 | 0.173 | 0.188 | 1.09 | 244 | 106 |
| Ackland–Jones analysis | 2,048,000 | 0.548 | 0.653 | 1.19 | 87 | 144 |
| Polyhedral template matching | 2,048,000 | 3.039 | 3.575 | 1.18 | 622 | 147 |
| Centro-symmetry parameter, 12 neighbors | 2,048,000 | 0.427 | 0.520 | 1.22 | 85 | 113 |
| Diamond structure | 2,000,376 | 0.306 | 0.405 | 1.32 | 326 | 357 |
| Steinhardt q2–q6, 12 neighbors | 2,048,000 | 0.534 | 0.549 | 1.03 | 163 | 188 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 256,000 | 0.333 | 15.348 | 46.09 | 457 | 40 |
| Radial distribution function, 6 Å, 200 bins | 2,048,000 | 0.299 | 0.526 | 1.76 | 181 | 85 |
| Bond angle distribution, 3 Å, 180 bins | 2,048,000 | 0.370 | 0.615 | 1.66 | 249 | 1350 |
| Voronoi volumes and coordination | 2,048,000 | 1.926 | 10.071 | 5.23 | 1896 | 63 |
| Cluster analysis, 2 Å | 2,046,192 | 0.389 | 0.989 | 2.54 | 96 | 28 |
| Wigner–Seitz occupancies | 2,037,841 | 0.117 | 0.188 | 1.61 | 115 | 153 |
| Atomic strain, 3 Å | 2,048,000 | 0.147 | 0.074 | 0.50 | 483 | 263 |

### freud

| Case | Atoms | mdapy 2.0.0.dev0 (s) | freud 3.6.1 (s) | freud / mdapy | mdapy (MB) | freud (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Neighbor list, 5 Å | 256,000 | 0.056 | 0.276 | 4.92 | 199 | 1453 |
| 12 nearest neighbors | 2,048,000 | 0.349 | 0.963 | 2.76 | 438 | 2718 |
| Steinhardt q2–q6, 12 neighbors | 2,048,000 | 0.534 | 1.816 | 3.40 | 163 | 934 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 2,048,000 | 3.856 | 25.440 | 6.60 | 3383 | 2718 |
| Radial distribution function, 6 Å, 200 bins | 2,048,000 | 0.299 | 1.153 | 3.86 | 181 | 235 |
| Voronoi volumes and coordination | 256,000 | 0.221 | 2.006 | 9.09 | 195 | 508 |
| Cluster analysis, 2 Å | 2,046,192 | 0.389 | 0.759 | 1.95 | 96 | 82 |

- From some 30 000 atoms on, the ratios hardly change with the size; below,
  a fixed cost of a few milliseconds per call in OVITO's pipelines favors
  mdapy. Timings vary by some 15 % between runs on this machine, which
  slows down as it heats up.
- Against OVITO, the analyses on the neighbors of each atom (structure
  identification, Steinhardt parameters, distributions) are 1.0 to 1.8
  times faster in mdapy. Those on the nearest neighbors find them atom by
  atom as they go, without a list, and take less memory than in OVITO.
  Voronoi tessellation is five times faster in mdapy, and Steinhardt
  parameters over Voronoi neighbors weighted by area 18 to 46 times; OVITO
  took more than ten minutes for the latter at two million atoms, which
  was stopped.
- Against freud, mdapy is faster in every case from 30 000 atoms on: 2 to
  5 times for neighbor searches, Steinhardt parameters and the radial
  distribution function, 1.2 to 2 times for clusters, and 6 to 10 times for
  Voronoi tessellations and Steinhardt parameters over their faces,
  although freud computes in single precision. freud
  keeps the pairs of its searches with their distances and weights, and
  took more memory for them; its Voronoi analysis of two million atoms was
  predicted to need more than 8 GB and was skipped.
- Polyhedral template matching runs the same library in mdapy and OVITO;
  mdapy first lists the 18 nearest neighbors of every atom, which takes
  four times the memory of OVITO, which finds them as it goes.
- The Voronoi analysis of mdapy keeps every face of every cell, about 1 kB
  per atom, for Voronoi neighbors and indices later.
- Atomic strain is the one case where OVITO is faster, by two times: most
  of mdapy's time goes into finding the neighbors in the reference, with
  the exact distances of a neighbor list. Given a list of the reference
  (`neighbors=`), built once for every frame, a frame takes 0.05 s.
- The neighbor lists of two million atoms were not timed: the comparison,
  which holds the pairs of two programs at once, was predicted to need
  more than 8 GB.

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
  come from two tessellations), and for neighbor lists the same number of
  neighbors of each atom and the same sum of their distances. freud
  computes in single precision, which rounds positions in a box of 300 Å by
  some 1e-5 Å: values agree within 1e-4, the numbers of neighbors and the
  bins of the radial distribution function within the pairs closer than
  1e-4 Å to a cutoff or the edge of a bin, and Voronoi faces above 0.01 Å²
  are the same (freud leaves out some faces of distant neighbors below
  some 0.002 Å², so its Steinhardt parameters over Voronoi faces agree
  within 1e-3). A program that differs is not timed.
- Each process first computes the case for 500 atoms, so that loading
  libraries and starting threads are not timed, then three times for the
  system; the shortest time counts.
- The time runs from the call to the result, with every neighbor search.
  Making the system and converting it into OVITO's data collection or
  freud's box and points are not timed. OVITO computes a new pipeline each
  time, so nothing is cached.
- A process that grows beyond 8 GB is stopped, and a size is skipped where
  the previous one predicts more; a process that runs for more than ten
  minutes is stopped.

## Running it

The comparison is `benchmarks/comparison.py` in the source repository. It
needs OVITO (`pip install ovito`), freud (`pip install freud-analysis`)
and, for the figure, matplotlib.

```bash
python benchmarks/comparison.py run                     # every case
python benchmarks/comparison.py run cna voronoi --sizes 4000 32000 --threads 1
python benchmarks/comparison.py table results.json      # Markdown tables
python benchmarks/comparison.py plot results.json ratio.png
```

`run` appends one line of JSON per case and size to `results.json` (the
option `--output`): the machine, the versions, the agreement of each
program, and the times and peak memory of each. A new case is a `Case` in
`_cases()`: how to make its system, the mdapy call, and for each other
program its call and how to compare its result with that of mdapy.
