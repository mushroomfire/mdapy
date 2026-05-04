# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Special Quasirandom Structure (SQS) generation, modeled after ATAT mcsqs.

The Python class :class:`SQS` takes an :class:`mdapy.System` containing a
random alloy (e.g. produced by :func:`mdapy.build_hea`), enumerates pair /
triplet / quadruplet clusters from the neighbor list, and uses the C++
Monte-Carlo engine in ``mdapy._sqs`` to drive correlation functions to their
fully-random target — i.e. an SQS. All cell shapes are supported (cubic,
orthorhombic, triclinic) and small boxes that mdapy.Neighbor replicates
internally are handled by folding ghost neighbor indices back to the primary
cell.

Mathematical principle
----------------------

**Trigonometric basis.** For an :math:`m`-component sublattice we use the same
per-site basis ATAT does (van de Walle, *CALPHAD* **42**, 13–18, 2013):

.. math::

   \\phi_{2t-1}(s) = -\\cos(2\\pi t s / m), \\quad t=1,\\dots,\\lfloor m/2 \\rfloor

   \\phi_{2t}(s) = -\\sin(2\\pi t s / m), \\quad t=1,\\dots,\\lceil m/2 \\rceil-1

with :math:`s\\in\\{0,\\dots,m-1\\}` indexing the species on the site. This
gives :math:`m-1` independent functions per site. For binary (:math:`m=2`)
the single function reduces to :math:`(-1)^s`; for an equiatomic 5-element
HEA (:math:`m=5`) all four single-site averages :math:`\\langle\\phi_k\\rangle`
vanish.

**Cluster correlation.** A *cluster instance* is a specific atom-tuple
:math:`(a_1,\\dots,a_n)`. A *channel* groups instances that share both a
distance shell and the same function tuple :math:`(f_1,\\dots,f_n)`. The
channel correlation is the average sigma over its instances,

.. math::

   \\pi_\\alpha = \\frac{1}{N_\\alpha} \\sum_{\\text{instances}}
                \\prod_{p=1}^{n} \\phi_{f_p}(s_{a_p}),

with the fully-random target

.. math::

   \\pi_\\alpha^{\\mathrm{rand}} = \\prod_{p=1}^{n}
                                  \\sum_{s} c_s\\, \\phi_{f_p}(s),

where :math:`c_s` is the global concentration of species :math:`s`. For an
equiatomic alloy this is zero on every multi-body channel.

**ATAT objective with the d1 perfect-match reward.** The default objective
(``objective='atat'``) faithfully reproduces ATAT's
``calc_objective_func`` (atat/src/mcsqs.c++:52-94):

1. Compute :math:`\\Delta_\\alpha = |\\pi_\\alpha-\\pi_\\alpha^{\\mathrm{rand}}|`
   per channel.
2. For each body order :math:`p=n-2`, find ``maxdist[p]`` — the largest
   diameter such that every channel of body :math:`p` with diameter
   :math:`\\le \\mathrm{maxdist}[p]` has :math:`\\Delta_\\alpha \\le
   \\mathrm{tol}`.
3. Make ``maxdist`` monotonically non-increasing in :math:`p`, then set
   :math:`d_1 = \\min_p \\mathrm{maxdist}[p]`.  This is the longest range
   where *all* correlations of *all* body orders are perfectly matched.
4. The objective is

   .. math::

      \\mathcal{O}
      = \\frac{\\sum_{\\alpha:d_\\alpha\\ge d_1} w_\\alpha \\Delta_\\alpha}
             {\\sum_{\\alpha:d_\\alpha\\ge d_1} w_\\alpha}
        - w_{\\mathrm{dist}} \\sum_{p}\\rho^{p}\\,
          \\mathrm{maxdist}[p]/d_{\\min}

   with weights
   :math:`w_\\alpha = \\exp(-\\lambda(d_\\alpha-d_{\\min}))\\,\\rho^{n_\\alpha-2}`.

The first term penalises any residual mismatch outside the perfect-match
range; the second is a *reward* for extending it. As soon as a Metropolis
swap pushes :math:`d_1` outward, the objective drops sharply, biasing the
optimizer toward configurations that match short range exactly before
working on longer range.

The simpler ``objective='abs'`` mode replaces the formula with the plain
weighted sum :math:`\\mathcal{O}=\\sum_\\alpha w_\\alpha \\Delta_\\alpha`.

Computation pipeline
--------------------

1. Extract types and concentrations from the input :class:`mdapy.System`
   (uses the ``element`` column when present, else ``type``).
2. Build the neighbor list at ``rc = max(cutoffs.values())``. Triclinic
   and small / replicated boxes are handled transparently — when
   ``mdapy.Neighbor`` enlarges a small box, ghost indices
   :math:`j_{\\mathrm{enl}}` are folded back to the primary cell via
   :math:`j_\\mathrm{real} = j_\\mathrm{enl} \\bmod N_\\mathrm{orig}` while
   each periodic image still counts as a distinct geometric instance.
3. Enumerate pair / triplet / quad clusters directly from the neighbor
   list, binning by the *sorted tuple* of pairwise distances so different
   shapes with the same diameter map to different shells.
4. Expand each cluster into channels by enumerating function-tuples
   :math:`(f_1,\\dots,f_n) \\in \\{0,\\dots,m-2\\}^n`. Two instances with
   the same shell and function tuple are aggregated into one channel.
5. Build an atom-to-instance index so a swap on atoms :math:`(i,j)` only
   revisits the instances containing them.
6. Run Monte-Carlo with ``n_replicas`` independent chains over OpenMP
   threads. Each Metropolis move picks two atoms of different species
   from the same sublattice and applies the swap; on accept/reject the
   per-channel :math:`\\sum\\sigma` is updated incrementally and the
   objective is recomputed in :math:`O(N_\\mathrm{channels})` time.
7. Return the lowest-objective replica's atom labels into a fresh
   :class:`mdapy.System` with the same cell and positions as the input.

How to verify a result is an SQS
--------------------------------

:meth:`SQS.is_sqs` combines three independent checks:

* **Absolute correlation residual:**
  :math:`\\max_\\alpha |\\pi_\\alpha - \\pi_\\alpha^{\\mathrm{rand}}| <
  \\mathrm{tol}`. A common acceptance threshold is **0.03** for a
  few-hundred-atom cell; tighten as :math:`N` grows. Equivalent to "the
  longest-matched range covers the whole cluster set".
* **Statistical baseline:** generate ``n_random`` random species
  permutations preserving composition; for each compute
  :math:`\\max_\\alpha \\Delta_\\alpha`. The SQS passes if its
  :math:`\\max\\Delta` lies at or below the
  :math:`(100-\\mathrm{percentile})`-th quantile of that distribution.
  This automatically accounts for the discreteness floor at small
  :math:`N` — the SQS only has to be more random than at least
  ``percentile`` % of random samples.
* **Warren–Cowley short-range order:**
  :math:`\\alpha_{ij}^{(n)} = 1 - P_{ij}^{(n)}/c_j` at the smallest pair
  shell. The off-diagonal :math:`|\\alpha_{ij}|` must be below ``tol``.
  This is the most physically intuitive check and the one to report in a
  paper.

For an equiatomic 3- or 5-element fcc HEA at 100+ atoms a converged SQS
typically gives Warren–Cowley off-diagonal values *exactly* zero (every
atom's NN1 shell holds the same fraction of every species).

Quick recipe
------------

>>> import mdapy as mp
>>> sys_init = mp.build_hea(
...     ("Fe", "Ni", "Co", "Mn", "Cr"), (0.2,)*5,
...     "fcc", 3.55, nx=4, ny=4, nz=4, random_seed=0,
... )
>>> sqs = mp.SQS(
...     sys_init, cutoffs={2: 4.0, 3: 3.0, 4: 3.0},
...     n_replicas=4, max_steps=200000, T=0.02,
... ).compute()
>>> verdict, report = sqs.is_sqs(tol=0.05, return_report=True)
>>> sqs.system.write_data("sqs_alloy.data")     # use the optimized alloy

Validated against ATAT mcsqs 3.50 on a 108-atom equiatomic FeNiCr fcc
test (``-2=4.0 -3=3.0``): both reach the optimal Warren–Cowley result
(every atom's 12 NN shell has exactly 4 of each species); mdapy's
deeper objective additionally matches the NN2 pair shell exactly in the
same wall time.

Citation
--------

This class implements the *mcsqs* algorithm of ATAT. If you use it in a
publication, please cite the original method paper in addition to mdapy:

  A. van de Walle, P. Tiwary, M. de Jong, D. L. Olmsted, M. Asta,
  A. Dick, D. Shin, Y. Wang, L.-Q. Chen, Z.-K. Liu.
  *Efficient stochastic generation of special quasirandom structures.*
  CALPHAD **42**, 13–18 (2013).

References
----------

* van de Walle et al., *CALPHAD* **42**, 13–18 (2013).
* Zunger, Wei, Ferreira, Bernard, *Phys. Rev. Lett.* **65**, 353 (1990).
* Cowley, *Phys. Rev.* **77**, 669 (1950).
"""

from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING, Tuple
from itertools import combinations
import numpy as np
import polars as pl

from mdapy import _sqs

if TYPE_CHECKING:
    from mdapy.system import System


# Objective function modes accepted by SQSEngine.objective_mode
_OBJ_MODE = {"abs": 0, "atat": 1}


class SQS:
    """Generate a Special Quasirandom Structure from a random alloy template.

    Parameters
    ----------
    system : mdapy.System
        Input alloy. Must contain an ``element`` (or ``type``) column with the
        same number of distinct species as the desired SQS. Atom positions and
        cell are taken as-is — only the species labels are reshuffled.
    cutoffs : dict[int, float]
        Cluster cutoffs in Å, keyed by cluster size:

        * ``2`` (required): pair-cluster cutoff.
        * ``3`` (optional): triplet cutoff (all three pairwise distances must
          be within this value).
        * ``4`` (optional): quadruplet cutoff (all six pairwise distances).

    n_replicas : int, default=4
        Number of independent Monte-Carlo chains run in parallel via OpenMP.
        The chain with the lowest objective wins.
    max_steps : int, default=50000
        Maximum swap attempts per chain. ``0`` means "evaluate the input
        only" (no shuffling, no MC) — useful for diagnostics.
    T : float, default=0.05
        Metropolis temperature. Smaller values are greedier; ATAT default
        is 1.0 but for equi-atomic HEAs values of 0.01-0.1 converge faster.
    objective : {'atat', 'abs'}, default='atat'
        ``'atat'`` reproduces ATAT mcsqs's `d1`-reward objective. ``'abs'``
        is a simple weighted sum of ``|pi - target|``.
    atat_tol : float, default=1e-3
        Tolerance below which a channel counts as "perfectly matched"
        (only meaningful with ``objective='atat'``).
    shell_tol : float, default=0.05
        Distance tolerance (in Å) when grouping clusters into shells.
    weight_decay : float, default=0.0
        ``lambda`` in the per-shell weight ``w = exp(-lambda * (d - d_min))``.
    weight_npts : float, default=1.0
        ``rho`` exponent: weight per extra body order; channels of size n_pts
        get an extra ``rho^(n_pts-2)`` factor.
    seed : int, default=0
        Base RNG seed; each replica gets ``seed`` plus a per-replica offset.

    Attributes
    ----------
    system : mdapy.System
        Output SQS structure (set by :meth:`compute`). Same atoms / cell as
        the input but with the ``element`` / ``type`` columns reshuffled.
    objective : float
        Final value of the Monte-Carlo objective.
    correlations : numpy.ndarray
        Per-channel correlations of the SQS, shape ``(n_channels,)``.
    channel_info : list[dict]
        One entry per channel: ``n_pts``, ``shell``, ``diameter``,
        ``funcs``, ``n_instances``, ``target``, ``corr``.
    """

    def __init__(
        self,
        system: "System",
        cutoffs: Dict[int, float],
        n_replicas: int = 4,
        max_steps: int = 50000,
        T: float = 0.05,
        objective: str = "atat",
        atat_tol: float = 1e-3,
        shell_tol: float = 0.05,
        weight_decay: float = 0.0,
        weight_npts: float = 1.0,
        seed: int = 0,
    ):
        if 2 not in cutoffs:
            raise ValueError("cutoffs must include key 2 (pair cutoff in Å)")
        for k in cutoffs:
            if k not in (2, 3, 4):
                raise ValueError(
                    f"only 2-, 3- and 4-body cutoffs are supported (got {k})"
                )
        if objective not in _OBJ_MODE:
            raise ValueError(f"objective must be one of {list(_OBJ_MODE)}")

        self._sys_in       = system
        self.cutoffs       = dict(cutoffs)
        self.n_replicas    = int(n_replicas)
        self.max_steps     = int(max_steps)
        self.T             = float(T)
        self.objective_mode_str = str(objective)
        self.atat_tol      = float(atat_tol)
        self.shell_tol     = float(shell_tol)
        self.weight_decay  = float(weight_decay)
        self.weight_npts   = float(weight_npts)
        self.seed          = int(seed)

        # filled in by compute()
        self.system: Optional["System"] = None
        self.objective: Optional[float] = None
        self.correlations: Optional[np.ndarray] = None
        self.channel_info: Optional[list] = None
        self._engine = None
        self._best_types: Optional[np.ndarray] = None

    # ----------------------------------------------------------------- helpers
    def _extract_types(self):
        """Return ``(type_array_0idx, n_species, species_labels, label_kind)``."""
        df = self._sys_in.data
        if "element" in df.columns:
            unique = df["element"].unique().sort().to_list()
            ele2idx = {e: i for i, e in enumerate(unique)}
            type_arr = np.array(
                [ele2idx[e] for e in df["element"].to_list()], dtype=np.int64
            )
            return type_arr, len(unique), unique, "element"
        if "type" in df.columns:
            t = df["type"].to_numpy() - 1
            n = int(t.max()) + 1
            return t.astype(np.int64), n, list(range(n)), "type"
        raise ValueError("System must have an 'element' or 'type' column")

    def _build_neighbor(self, rc: float):
        """Build neighbor list at given ``rc`` and return (verlet, dist, nnum, N_orig).

        ``verlet[i, k]`` may index a ghost atom when mdapy replicated a small
        box; map back via ``j_real = verlet[i, k] % N_orig``. The verlet table
        itself has shape ``(N_enlarged, max_neigh)`` — we use the full table so
        that ghost-ghost distances (needed for triplet / quad enumeration) are
        available.
        """
        sys_ = self._sys_in
        sys_.build_neighbor(rc=rc)
        return (
            sys_.verlet_list,
            sys_.distance_list,
            sys_.neighbor_number,
            int(sys_.N),
        )

    def _enumerate_clusters(self):
        """Enumerate pair / triplet / quad clusters; return (per_body, all_diams).

        Returns
        -------
        per_body : list[(n_pts, clusters_per_shell_dict, shell_diameters)]
            One entry per body order. ``clusters_per_shell_dict`` maps a
            *local* shell index (per-body) to a list of cluster atom-index
            tuples (using the *original* / primary cell indexing).
        all_diams : list[float]
            Globally-sorted distinct shell diameters (so multiple body orders
            sharing a diameter map to the same shell weight).
        """
        rc_max = max(self.cutoffs.values())
        verlet, dist, nnum, N_orig = self._build_neighbor(rc_max)

        # Build per-(enlarged)atom dict: atom -> {neighbor_enl: distance}.
        # We need this for triplet / quad enumeration where we look up
        # distances between pairs of neighbors of a central atom.
        N_enl = verlet.shape[0]
        nbr_dist: list[dict] = [dict() for _ in range(N_enl)]
        for i in range(N_enl):
            n = int(nnum[i])
            for k in range(n):
                j = int(verlet[i, k])
                if j < 0:
                    continue
                nbr_dist[i][j] = float(dist[i, k])

        per_body = []

        # ---------------- pairs -----------------
        rc2 = float(self.cutoffs[2])
        pair_shell_diams: list[float] = []
        pair_per_shell: dict[int, list] = {}
        seen_keys = set()      # (i, j_real, distance_key) — dedup self-image rev

        for i in range(N_orig):
            n = int(nnum[i])
            for k in range(n):
                j_enl = int(verlet[i, k])
                if j_enl < 0:
                    continue
                d = float(dist[i, k])
                if d > rc2 + 1e-9:
                    continue
                j_real = j_enl % N_orig
                # Avoid the trivial (i,i,distance=0) self entry that some
                # neighbor builders include for replicated cells.
                if j_real == i and j_enl == i:
                    continue
                # When the central cell holds the only image, skip the symmetric
                # (j,i) duplicate. For replicated cells, both directions encode
                # different geometric instances — keep them.
                if j_enl < N_orig and j_enl <= i:
                    continue
                # bin by diameter
                sh = self._bin_diameter(d, pair_shell_diams)
                pair_per_shell.setdefault(sh, []).append([i, j_real])
        per_body.append((2, pair_per_shell, pair_shell_diams))

        # ---------------- triplets -----------------
        if 3 in self.cutoffs:
            rc3 = float(self.cutoffs[3])
            trip_shell_keys: list[Tuple[float, float, float]] = []
            trip_per_shell: dict[int, list] = {}

            for i in range(N_orig):
                # enlarged neighbors of i
                nbi = nbr_dist[i]
                for j_enl, k_enl in combinations(sorted(nbi.keys()), 2):
                    d_ij = nbi[j_enl]
                    d_ik = nbi[k_enl]
                    if d_ij > rc3 + 1e-9 or d_ik > rc3 + 1e-9:
                        continue
                    d_jk = nbr_dist[j_enl].get(k_enl)
                    if d_jk is None or d_jk > rc3 + 1e-9:
                        continue
                    # canonical shape signature: sorted pairwise distances
                    sig = tuple(sorted((d_ij, d_ik, d_jk)))
                    sh = self._bin_signature(sig, trip_shell_keys)
                    trip_per_shell.setdefault(sh, []).append(
                        [i, j_enl % N_orig, k_enl % N_orig]
                    )
            per_body.append((3, trip_per_shell, [k[-1] for k in trip_shell_keys]))

        # ---------------- quadruplets -----------------
        if 4 in self.cutoffs:
            rc4 = float(self.cutoffs[4])
            quad_shell_keys: list[Tuple[float, ...]] = []
            quad_per_shell: dict[int, list] = {}

            for i in range(N_orig):
                nbi = nbr_dist[i]
                for j_enl, k_enl, l_enl in combinations(sorted(nbi.keys()), 3):
                    d_ij = nbi[j_enl]; d_ik = nbi[k_enl]; d_il = nbi[l_enl]
                    if max(d_ij, d_ik, d_il) > rc4 + 1e-9:
                        continue
                    d_jk = nbr_dist[j_enl].get(k_enl)
                    d_jl = nbr_dist[j_enl].get(l_enl)
                    d_kl = nbr_dist[k_enl].get(l_enl)
                    if d_jk is None or d_jl is None or d_kl is None:
                        continue
                    if max(d_jk, d_jl, d_kl) > rc4 + 1e-9:
                        continue
                    sig = tuple(sorted(
                        (d_ij, d_ik, d_il, d_jk, d_jl, d_kl)
                    ))
                    sh = self._bin_signature(sig, quad_shell_keys)
                    quad_per_shell.setdefault(sh, []).append([
                        i, j_enl % N_orig, k_enl % N_orig, l_enl % N_orig
                    ])
            per_body.append((4, quad_per_shell, [k[-1] for k in quad_shell_keys]))

        # Globally collect all distinct shell diameters for the C++ shell array.
        all_diams: list[float] = []
        global_map: list[list[int]] = []   # per-body local_shell -> global shell
        for n_pts, per_shell, diams in per_body:
            mapping = []
            for d in diams:
                found = -1
                for gi, gd in enumerate(all_diams):
                    if abs(gd - d) < self.shell_tol:
                        found = gi; break
                if found < 0:
                    all_diams.append(d)
                    found = len(all_diams) - 1
                mapping.append(found)
            global_map.append(mapping)

        return per_body, all_diams, global_map

    def _bin_diameter(self, d: float, bins: list) -> int:
        """Round-and-bin a single distance; create a new bin if no match."""
        for k, ref in enumerate(bins):
            if abs(d - ref) < self.shell_tol:
                return k
        bins.append(d)
        return len(bins) - 1

    def _bin_signature(self, sig: Tuple[float, ...], bins: list) -> int:
        """Bin a multi-distance shape signature; bins are sorted-distance tuples."""
        for k, ref in enumerate(bins):
            if len(ref) != len(sig):
                continue
            if all(abs(a - b) < self.shell_tol for a, b in zip(sig, ref)):
                return k
        bins.append(sig)
        return len(bins) - 1

    # -------------------------------------------------------------------- run
    def compute(self) -> "SQS":
        """Run the SQS optimization. Returns ``self`` for chaining."""
        type_arr, n_species, labels, label_kind = self._extract_types()
        n_atoms = int(self._sys_in.N)

        conc = np.zeros(n_species)
        for t in type_arr:
            conc[int(t)] += 1.0
        conc /= n_atoms

        engine = _sqs.SQSEngine()
        engine.n_atoms = n_atoms
        engine.set_species(n_species, conc.tolist())
        engine.objective_mode = _OBJ_MODE[self.objective_mode_str]
        engine.atat_tol       = self.atat_tol
        engine.atat_w_dist    = 1.0    # held fixed; user controls via T / decay
        engine.atat_w_npts    = self.weight_npts

        per_body, all_diams, global_map = self._enumerate_clusters()
        d_min = min(all_diams)

        for (n_pts, per_shell, _diams), shell_map in zip(per_body, global_map):
            for local_s, clusters in per_shell.items():
                gs = shell_map[local_s]
                arr = np.array(clusters, dtype=np.int32)
                flat = arr.flatten().tolist()
                engine.add_cluster_body(n_pts, flat, [int(gs)] * len(clusters))

        shell_weights = [
            float(np.exp(-self.weight_decay * (d - d_min))) for d in all_diams
        ]
        engine.shell_weight   = shell_weights
        engine.shell_diameter = list(all_diams)
        engine.finalize()

        if self.max_steps <= 0:
            best_types = type_arr.tolist()
            best_corr  = engine.correlations(best_types)
            best_obj   = engine.objective(best_types)
        else:
            best_types, best_obj, best_corr = engine.run_mc(
                type_arr.tolist(),
                int(self.max_steps),
                float(self.T),
                int(self.n_replicas),
                int(self.seed),
            )

        from mdapy.system import System
        new_data = self._sys_in.data.clone()
        new_type = np.array(best_types, dtype=np.int32) + 1
        if label_kind == "element":
            new_elem = [labels[t] for t in best_types]
            new_data = new_data.with_columns(pl.Series("element", new_elem))
        # Always keep a `type` column on the output so System.write_data() works
        # without further wrangling. If the input had only `element`, this adds
        # a `type` column consistent with the alphabetic ordering of species.
        new_data = new_data.with_columns(pl.Series("type", new_type))

        self.system    = System(data=new_data, box=self._sys_in.box)
        self.objective = float(best_obj)
        self.correlations = np.asarray(best_corr, dtype=np.float64)
        self._engine     = engine
        self._best_types = np.asarray(best_types, dtype=np.int64)

        infos = []
        for i in range(engine.num_channels()):
            n_pts, shell, funcs, n_inst, target = engine.channel_info(i)
            infos.append({
                "n_pts":        int(n_pts),
                "shell":        int(shell),
                "diameter":     float(all_diams[shell]),
                "funcs":        list(funcs),
                "n_instances":  int(n_inst),
                "target":       float(target),
                "corr":         float(self.correlations[i]),
            })
        self.channel_info = infos
        return self

    # -------------------------------------------------------------- judgment
    def is_sqs(
        self,
        tol: float = 0.03,
        n_random: int = 50,
        percentile: float = 95.0,
        return_report: bool = False,
        seed: int = 12345,
    ):
        """Decide whether ``self.system`` qualifies as an SQS.

        Three independent checks are evaluated and combined into a single
        boolean verdict:

        1. **Absolute**: max ``|pi - target|`` across all channels is below
           ``tol``. Default ``0.03`` matches the empirical 5-element fcc HEA
           noise floor at ~100 atoms; tighten for larger cells.
        2. **Statistical baseline**: draw ``n_random`` random species
           assignments preserving composition; for each compute the per-channel
           absolute residual and record its maximum. The SQS passes if its own
           ``max |pi - target|`` lies at or below the ``(100 - percentile)``-th
           quantile of that distribution — i.e. it is more random than at
           least ``percentile`` % of random samples.
        3. **Warren-Cowley**: max ``|alpha_ij|`` (off-diagonal) at the
           smallest pair cutoff is below ``tol``.

        Parameters
        ----------
        tol : float
            Threshold for the absolute and Warren-Cowley checks.
        n_random : int
            Number of random samples used in the statistical baseline check.
        percentile : float
            Percentile threshold for the statistical baseline (95 = "more
            random than 95 % of random assignments").
        return_report : bool
            If ``True``, return ``(verdict, report_dict)`` instead of just
            the boolean.
        seed : int
            RNG seed for the random-sample draws.

        Returns
        -------
        verdict : bool
            ``True`` if all three checks pass.
        report : dict, optional
            Diagnostic statistics — included only when ``return_report``.
        """
        if self._engine is None:
            raise RuntimeError("call compute() before is_sqs()")

        engine = self._engine
        types  = self._best_types.tolist()

        # 1) absolute check
        delta_all = np.array(engine.per_channel_delta(types))
        max_delta = float(np.max(delta_all)) if len(delta_all) else 0.0
        absolute_pass = max_delta < tol

        # 2) statistical baseline.
        # Compute, for each random sample, its max |Delta| across channels.
        # The SQS passes if its max |Delta| sits at or below the
        # ``(100 - percentile)``-th percentile of that distribution — i.e. it
        # is more random than ``percentile`` % of random assignments.
        # (Comparing per-channel against per-channel percentile would over-
        # reject because at the 95-th percentile we'd expect 5% of channels
        # to fail by chance even on a true SQS.)
        rng = np.random.default_rng(seed)
        rand_max = np.zeros(n_random)
        for r in range(n_random):
            shuffled = rng.permutation(self._best_types).tolist()
            rand_max[r] = float(np.max(np.array(engine.per_channel_delta(shuffled))))
        rand_threshold = float(np.percentile(rand_max, 100.0 - percentile))
        statistical_pass = max_delta <= rand_threshold + 1e-12

        # 3) Warren-Cowley check
        sys_out = self.system
        rc_wcp = float(self.cutoffs[2])
        # ensure a fresh neighbor list at the SHORTEST relevant cutoff;
        # use the smallest pair shell diameter + a small margin
        shortest_pair_d = min(
            ci["diameter"] for ci in self.channel_info if ci["n_pts"] == 2
        )
        rc_wcp = shortest_pair_d + self.shell_tol
        sys_out.build_neighbor(rc=rc_wcp)
        wcp = sys_out.cal_warren_cowley_parameter(rc=rc_wcp)
        wcp_off = wcp.WCP - np.diag(np.diag(wcp.WCP))
        max_wcp_off = float(np.max(np.abs(wcp_off)))
        wcp_pass = max_wcp_off < tol

        verdict = absolute_pass and statistical_pass and wcp_pass
        if not return_report:
            return verdict
        report = {
            "verdict":          verdict,
            "absolute": {
                "pass":             absolute_pass,
                "max_delta":        max_delta,
                "tol":              tol,
            },
            "statistical": {
                "pass":             statistical_pass,
                "n_random":         n_random,
                "percentile":       percentile,
                "sqs_max_delta":    max_delta,
                "rand_threshold":   rand_threshold,
                "rand_max_mean":    float(np.mean(rand_max)),
            },
            "warren_cowley": {
                "pass":             wcp_pass,
                "rc":               rc_wcp,
                "max_off_diag":     max_wcp_off,
                "tol":              tol,
                "matrix":           wcp.WCP,
            },
        }
        return verdict, report


if __name__ == "__main__":
    pass
