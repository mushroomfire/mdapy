# mdapy and OVITO

The analyses that mdapy and [OVITO](https://www.ovito.org) both have,
computed on the same systems with the same parameters. A case is timed only
where both programs give the same result, atom by atom.

## Results

Apple M5 (4 performance and 6 efficiency cores, 24 GB), macOS, Python
3.13, mdapy 2.0.0.dev0 and OVITO 3.17.0 from pip, both with 10 threads.

![Time of OVITO over the time of mdapy for each case and size](images/ovito.png)

At the largest size that ran for each case: the shortest of three times,
and the memory each computation took beyond the system: the peak resident
memory of the process minus its peak before the timings.

| Case | Atoms | mdapy (s) | OVITO (s) | OVITO / mdapy | mdapy (MB) | OVITO (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Neighbor list, 5 Å | 256,000 | 0.059 | 0.073 | 1.24 | 192 | 765 |
| 12 nearest neighbors | 2,048,000 | 0.368 | 0.466 | 1.27 | 438 | 860 |
| Adaptive common neighbor analysis | 2,048,000 | 0.511 | 0.806 | 1.58 | 86 | 165 |
| Common neighbor analysis, cutoff 3.09 Å | 2,048,000 | 0.177 | 0.231 | 1.30 | 243 | 114 |
| Ackland–Jones analysis | 2,048,000 | 0.558 | 0.814 | 1.46 | 86 | 158 |
| Centro-symmetry parameter, 12 neighbors | 2,048,000 | 0.424 | 0.611 | 1.44 | 79 | 127 |
| Diamond structure | 2,000,376 | 0.303 | 0.486 | 1.60 | 329 | 370 |
| Steinhardt q2–q6, 12 neighbors | 2,048,000 | 0.546 | 0.658 | 1.21 | 165 | 205 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 256,000 | 0.325 | 38.333 | 117.80 | 469 | 47 |
| Radial distribution function, 6 Å, 200 bins | 2,048,000 | 0.296 | 0.509 | 1.72 | 183 | 86 |
| Bond angle distribution, 3 Å, 180 bins | 2,048,000 | 0.322 | 0.600 | 1.86 | 255 | 1395 |
| Voronoi volumes and coordination | 2,048,000 | 1.810 | 11.832 | 6.54 | 1952 | 67 |
| Cluster analysis, 2 Å | 2,048,000 | 0.342 | 1.101 | 3.22 | 558 | 217 |
| Wigner–Seitz occupancies | 2,037,841 | 0.118 | 0.192 | 1.63 | 121 | 141 |
| Atomic strain, 3 Å | 2,048,000 | 0.147 | 0.075 | 0.51 | 483 | 268 |

- From some 30 000 atoms on, the ratios hardly change with the size; below,
  a fixed cost of a few milliseconds per call in OVITO's pipelines favors
  mdapy. Timings vary by some 15 % between runs on this machine, which
  slows down as it heats up.
- The analyses on the neighbors of each atom (structure identification,
  Steinhardt parameters, distributions) are 1.2 to 1.9 times faster in
  mdapy. Those on the nearest neighbors find them atom by atom as they go,
  without a list, and take less memory than in OVITO. Voronoi tessellation
  is six times faster in mdapy, and Steinhardt parameters over Voronoi
  neighbors weighted by area 40 to 120 times; OVITO took more than five
  minutes for the latter at two million atoms, which was stopped.
- The Voronoi analysis of mdapy keeps every face of every cell, about 1 kB
  per atom, for Voronoi neighbors and indices later.
- Atomic strain is the one case where OVITO is faster, by two times: most of
  mdapy's time goes into finding the neighbors in the reference, with the
  exact distances of a neighbor list. Given a list of the reference
  (`neighbors=`), built once for every frame, a frame takes 0.05 s.

## Cases

| Case | System | mdapy | OVITO |
|---|---|---|---|
| Neighbor list, 5 Å | fcc Cu | `NeighborList.build(system, 5.0)` | `CutoffNeighborFinder(5.0, data).find_all()` |
| 12 nearest neighbors | fcc Cu | `NeighborList.build(system, num_neighbors=12)` | `NearestNeighborFinder(12, data).find_all()` |
| Adaptive common neighbor analysis | fcc Cu | `CommonNeighborAnalysis()` | `CommonNeighborAnalysisModifier()` |
| Common neighbor analysis, cutoff 3.09 Å | fcc Cu | `CommonNeighborAnalysis(3.086)` | `CommonNeighborAnalysisModifier(mode=FixedCutoff, cutoff=3.086)` |
| Ackland–Jones analysis | fcc Cu | `AcklandJonesAnalysis()` | `AcklandJonesModifier()` |
| Centro-symmetry parameter | fcc Cu | `CentroSymmetryParameter(12)` | `CentroSymmetryModifier(num_neighbors=12)` |
| Diamond structure | diamond Si | `DiamondStructure()` | `IdentifyDiamondModifier()` |
| Steinhardt q2–q6, 12 neighbors | fcc Cu | `SteinhardtParameters((2, 3, 4, 5, 6), num_neighbors=12)` | `SteinhardtParametersModifier(neighbor_mode=NearestNeighbors, num_neighbors=12, max_l=6)` |
| Steinhardt q2–q6, Voronoi | fcc Cu | `SteinhardtParameters((2, 3, 4, 5, 6), voronoi=True, weighted=True)` | `SteinhardtParametersModifier(neighbor_mode=Voronoi, max_l=6)` |
| Radial distribution function | fcc Cu | `RadialDistributionFunction(6.0, bins=200)` | `CoordinationAnalysisModifier(cutoff=6.0, number_of_bins=200)` |
| Bond angle distribution | fcc Cu | `BondAngleDistribution(cutoff=3.0, bins=180)` | `CreateBondsModifier(cutoff=3.0)`, `BondAngleDistributionModifier(bins=180)` |
| Voronoi volumes and coordination | fcc Cu | `VoronoiAnalysis()` | `VoronoiAnalysisModifier()` |
| Cluster analysis, 2 Å | random points | `ClusterAnalysis(cutoff=2.0)` | `ClusterAnalysisModifier(cutoff=2.0, sort_by_size=True)` |
| Wigner–Seitz occupancies | fcc Cu, 0.5 % vacancies | `WignerSeitz(reference)` | `WignerSeitzAnalysisModifier()` |
| Atomic strain, 3 Å | strained fcc Cu | `AtomicStrain(reference, 3.0)` | `AtomicStrainModifier(cutoff=3.0)` |

The fcc copper has its atoms displaced at random by 0.1 Å along each axis,
as at room temperature; the silicon by 0.05 Å. The random points have the
density of copper and, with a cutoff of 2 Å, are close to percolation, so
that clusters of every size form. OVITO's coordination analysis also counts
the neighbors of each atom along with the radial distribution function.

## How it is measured

- Each program runs in a process of its own, with the same number of
  threads (`MDAPY_NUM_THREADS` and `OVITO_THREAD_COUNT`).
- Before the timings, a separate process computes the case with both
  programs and compares the results: the same structure types, cluster
  sizes, occupancies and bins, floating-point values within 1e-10, and for
  neighbor lists the same number of neighbors of each atom and the same sum
  of their distances. A case that differs is not timed.
- Each process first computes the case for 500 atoms, so that loading
  libraries and starting threads are not timed, then three times for the
  system; the shortest time counts.
- The time runs from the call to the result, with every neighbor search.
  Making the system and converting it into OVITO's data collection are not
  timed. OVITO computes a new pipeline each time, so nothing is cached.
- A process that grows beyond 8 GB is stopped, and a size is skipped where
  the previous one predicts more.

## Running it

The comparison is `benchmarks/ovito_comparison.py` in the source
repository. It needs OVITO (`pip install ovito`) and, for the figure,
matplotlib.

```bash
python benchmarks/ovito_comparison.py run                     # every case
python benchmarks/ovito_comparison.py run cna voronoi --sizes 4000 32000 --threads 1
python benchmarks/ovito_comparison.py table results.json      # Markdown table
python benchmarks/ovito_comparison.py plot results.json ratio.png
```

`run` appends one line of JSON per case and size to `results.json` (the
option `--output`): the machine, the agreement, and the times and peak
memory of each program. A new case is a `Case` in `_cases()`: how to make
its system, the mdapy and OVITO calls, and how to compare their results.
