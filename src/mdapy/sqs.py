# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Special Quasirandom Structure (SQS) generation, modeled after ATAT mcsqs.

Take an :class:`mdapy.System` containing a random alloy (e.g. produced by
:func:`mdapy.build_hea`), enumerate pair / triplet / quadruplet clusters from
the neighbor list, and run a Monte-Carlo swap engine to drive the cluster
correlations to their fully-random target — i.e. an SQS. All cell shapes are
supported (cubic, orthorhombic, triclinic) and small boxes that
``mdapy.Neighbor`` replicates internally are handled by folding ghost
neighbor indices back to the primary cell.

Quick recipe
------------

>>> import mdapy as mp
>>> sys_init = mp.build_hea(
...     ("Cr", "Co", "Ni"), (1/3,) * 3,
...     "fcc", 3.53, nx=3, ny=3, nz=3, random_seed=1,
... )
>>> sqs = mp.SQS(sys_init, cutoffs={2: 4.0, 3: 3.0}).compute()
>>> sqs.is_sqs()                      # prints a one-screen summary
>>> sqs.system.write_data("sqs.data") # write the resulting structure

How to choose ``cutoffs``
-------------------------

``cutoffs`` is a dict keyed by cluster body order (``2`` = pair, ``3`` =
triplet, ``4`` = quadruplet). Only ``2`` is required; ``3`` and ``4`` are
optional and add longer-range matching at the cost of more channels.

The right value is "just past the shell you want to constrain". The first
two coordination shells (NN1, NN2) for the common metallic lattices, with
``a`` the conventional lattice parameter:

* **fcc**: NN1 = ``a / sqrt(2)`` ≈ ``0.707 a``,  NN2 = ``a``
* **bcc**: NN1 = ``sqrt(3)/2 * a`` ≈ ``0.866 a``,  NN2 = ``a``
* **hcp** (ideal c/a): NN1 = ``a``,  NN2 = ``sqrt(2) a``

Practical choice — covering NN1+NN2 with pairs and NN1 alone with
triplets is a good default:

================  ====================  =====================
Lattice           ``cutoffs[2]``        ``cutoffs[3]`` (opt.)
================  ====================  =====================
fcc, ``a=3.53``   ``4.0`` (NN1+NN2)     ``3.0`` (NN1 only)
bcc, ``a=2.87``   ``3.5`` (NN1+NN2)     ``2.7`` (NN1 only)
hcp, ``a=2.95``   ``4.5`` (NN1+NN2)     ``3.1`` (NN1 only)
================  ====================  =====================

If you don't know ``a``: build a quick neighbor list, plot the RDF, and pick
``cutoffs[2]`` halfway between the second and third RDF peaks.

What the parameters mean
------------------------

* ``cutoffs`` *(required)* — see above.
* ``n_replicas`` — number of independent MC chains run in parallel; the
  one with the lowest objective wins. More replicas help when the landscape
  has multiple minima. Default 4.
* ``max_steps`` — swap attempts per chain. Each chain remembers its
  *best-ever* state, so increasing ``max_steps`` is monotonically helpful
  (it cannot make the result worse). Set ``0`` to evaluate the input as-is
  without running MC — useful for diagnostics. Default 100000.
* ``T`` — Metropolis temperature. Smaller is greedier; 0.01–0.1 work well
  for equiatomic HEAs. Default 0.05.
* ``seed`` — RNG seed for reproducibility.

Mathematical background
-----------------------

We use the trigonometric per-site basis of van de Walle (CALPHAD 42, 13–18,
2013): for an ``m``-component sublattice each species ``s`` maps to ``m-1``
basis values ``phi_k(s)``. A *channel* groups cluster instances that share
both a distance shell and the same function tuple ``(f_1, ..., f_n)``. The
channel correlation is

    pi = (1/N_inst) sum_inst prod_p phi_{f_p}(s_{a_p})

with target (under fully-random species placement) prod_p < phi_{f_p} >.
For an equiatomic alloy every multi-body target is exactly zero.

The objective implemented is ATAT mcsqs's ``d1``-perfect-match formula: it
combines a weighted residual sum with a reward for extending the longest
range over which *all* correlations are matched within a small tolerance.

Citation
--------

If you use this in a publication, please cite:

  A. van de Walle, P. Tiwary, M. de Jong, D. L. Olmsted, M. Asta,
  A. Dick, D. Shin, Y. Wang, L.-Q. Chen, Z.-K. Liu.
  *Efficient stochastic generation of special quasirandom structures.*
  CALPHAD **42**, 13–18 (2013).
"""

from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING, Tuple
import numpy as np
import polars as pl

from mdapy import _sqs
from mdapy.parallel import get_num_threads

if TYPE_CHECKING:
    from mdapy.system import System


# Internal constants — exposed only via the engine to keep the user-facing API small.
_OBJECTIVE_MODE_ATAT = 1  # 0 = "abs" (plain sum), 1 = "atat" (d1 reward)
_ATAT_TOL = 1e-3
_SHELL_TOL = 0.05
_WEIGHT_DECAY = 0.0
_WEIGHT_NPTS = 1.0


class SQS:
    """Generate a Special Quasirandom Structure from a random alloy template.

    Parameters
    ----------
    system : mdapy.System
        Input alloy. Must contain an ``element`` (or ``type``) column with the
        same number of distinct species as the desired SQS. Atom positions and
        cell are taken as-is — only species labels are reshuffled.
    cutoffs : dict[int, float]
        Cluster cutoffs in Å, keyed by cluster size: ``2`` (required) is the
        pair cutoff; ``3`` and ``4`` (optional) add triplet / quadruplet
        constraints. See the module docstring for how to choose them per
        crystal structure.
    n_replicas : int, default=4
        Number of independent Monte-Carlo chains run in parallel via OpenMP.
    max_steps : int, default=100000
        Swap attempts per chain. ``0`` evaluates the input only (no MC).
        The engine tracks the best-ever state per chain, so larger values
        are monotonically helpful.
    T : float, default=0.05
        Metropolis temperature.
    seed : int, default=0
        RNG seed; each replica gets ``seed`` plus a per-replica offset.

    Attributes
    ----------
    system : mdapy.System
        Output SQS structure (set by :meth:`compute`). Same atoms / cell as
        the input but with the ``element`` / ``type`` columns reshuffled.
    objective : float
        Final value of the Monte-Carlo objective (lower is better).
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
        max_steps: int = 100000,
        T: float = 0.05,
        seed: int = 0,
    ):
        if 2 not in cutoffs:
            raise ValueError("cutoffs must include key 2 (pair cutoff in Å)")
        for k in cutoffs:
            if k not in (2, 3, 4):
                raise ValueError(
                    f"only 2-, 3- and 4-body cutoffs are supported (got {k})"
                )

        self._sys_in = system
        self.cutoffs = dict(cutoffs)
        self.n_replicas = int(n_replicas)
        self.max_steps = int(max_steps)
        self.T = float(T)
        self.seed = int(seed)

        # filled in by compute()
        self.system: Optional["System"] = None
        self.objective: Optional[float] = None
        self.correlations: Optional[np.ndarray] = None
        self.channel_info: Optional[list] = None
        self._engine = None
        self._best_types: Optional[np.ndarray] = None
        self._species_labels: Optional[list] = None

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

    def _build_image_neighbors(self, rc_max: float):
        """Per-atom list of ``(j_primary, image_offset_cartesian, distance)``
        entries with ``distance ≤ rc_max``, enumerating *every* periodic
        image direction (not collapsed by min-image).

        This mirrors what ATAT mcsqs does internally — its
        ``LatticePointInCellIterator × equivalent_clusters`` loop visits
        every (origin, displacement-vector) pair once, regardless of
        whether the displacement wraps around the box. mdapy.Neighbor's
        verlet list, by contrast, replicates the cell when ``rc`` exceeds
        a half-box edge and folds ghost neighbours back to primary indices,
        which over-counts some pairs and under-counts others; the
        Metropolis chain then optimises a slightly biased correlation set.

        Doing the enumeration in pure Python here means we don't depend on
        ``mdapy.Neighbor`` at all for SQS. The cost is O(N² · n_images)
        which is trivial at the cell sizes SQS users care about
        (≤ a few hundred atoms).
        """
        sys_ = self._sys_in
        pos = np.column_stack(
            [
                sys_.data["x"].to_numpy(),
                sys_.data["y"].to_numpy(),
                sys_.data["z"].to_numpy(),
            ]
        ).astype(float)
        box = np.array(sys_.box.box, dtype=float)
        N = int(sys_.N)

        # Search radius along each lattice vector. Using box-vector lengths
        # is conservative for triclinic cells (the *projected* radius would
        # be smaller, so we just look a bit further than strictly necessary).
        box_lens = np.linalg.norm(box, axis=1)
        nmax = [max(1, int(np.ceil(rc_max / l)) + 1) for l in box_lens]

        # Enumerate cartesian image offsets once.
        offsets = []
        for nx in range(-nmax[0], nmax[0] + 1):
            for ny in range(-nmax[1], nmax[1] + 1):
                for nz in range(-nmax[2], nmax[2] + 1):
                    img = nx * box[0] + ny * box[1] + nz * box[2]
                    offsets.append((nx, ny, nz, img))

        # For each image, vectorize the all-pair distance check across N atoms.
        nbrs: list[list] = [[] for _ in range(N)]
        for nx, ny, nz, img in offsets:
            # delta[i, j] = pos[j] + img - pos[i]
            delta = pos[None, :, :] + img[None, None, :] - pos[:, None, :]
            dist = np.linalg.norm(delta, axis=2)  # (N, N)
            mask = dist <= rc_max + 1e-9
            if nx == 0 and ny == 0 and nz == 0:
                np.fill_diagonal(mask, False)  # skip self at zero image
            ii, jj = np.nonzero(mask)
            for i, j in zip(ii.tolist(), jj.tolist()):
                nbrs[i].append((j, img, float(dist[i, j])))
        return nbrs, pos, box

    def _enumerate_clusters(self):
        """Enumerate pair / triplet / quad clusters from the image-aware
        neighbour list. Each ``(origin atom, image-direction)`` is one
        cluster instance — same convention as ATAT mcsqs."""
        rc_max = max(self.cutoffs.values())
        nbrs, pos, _ = self._build_image_neighbors(rc_max)
        N = len(nbrs)

        per_body = []

        # ---------------- pairs -----------------
        rc2 = float(self.cutoffs[2])
        pair_shell_diams: list[float] = []
        pair_per_shell: dict[int, list] = {}
        for i in range(N):
            for j, _img, d_ij in nbrs[i]:
                if d_ij > rc2 + 1e-9:
                    continue
                sh = self._bin_diameter(d_ij, pair_shell_diams)
                pair_per_shell.setdefault(sh, []).append([i, j])
        per_body.append((2, pair_per_shell, pair_shell_diams))

        # ---------------- triplets -----------------
        if 3 in self.cutoffs:
            rc3 = float(self.cutoffs[3])
            trip_shell_keys: list[Tuple[float, ...]] = []
            trip_per_shell: dict[int, list] = {}
            for i in range(N):
                valid_i = [(j, img, d) for (j, img, d) in nbrs[i] if d <= rc3 + 1e-9]
                # Enumerate ordered pairs (a, b) with a < b among i's image-neighbours
                # to avoid double-counting the same (i, j_a, j_b) triplet.
                for a in range(len(valid_i)):
                    ja, img_a, d_ia = valid_i[a]
                    pa = pos[ja] + img_a
                    for b in range(a + 1, len(valid_i)):
                        jb, img_b, d_ib = valid_i[b]
                        pb = pos[jb] + img_b
                        d_ab = float(np.linalg.norm(pa - pb))
                        if d_ab > rc3 + 1e-9:
                            continue
                        sig = tuple(sorted((d_ia, d_ib, d_ab)))
                        sh = self._bin_signature(sig, trip_shell_keys)
                        trip_per_shell.setdefault(sh, []).append([i, ja, jb])
            per_body.append((3, trip_per_shell, [k[-1] for k in trip_shell_keys]))

        # ---------------- quadruplets -----------------
        if 4 in self.cutoffs:
            rc4 = float(self.cutoffs[4])
            quad_shell_keys: list[Tuple[float, ...]] = []
            quad_per_shell: dict[int, list] = {}
            for i in range(N):
                valid_i = [(j, img, d) for (j, img, d) in nbrs[i] if d <= rc4 + 1e-9]
                for a in range(len(valid_i)):
                    ja, img_a, d_ia = valid_i[a]
                    pa = pos[ja] + img_a
                    for b in range(a + 1, len(valid_i)):
                        jb, img_b, d_ib = valid_i[b]
                        pb = pos[jb] + img_b
                        d_ab = float(np.linalg.norm(pa - pb))
                        if d_ab > rc4 + 1e-9:
                            continue
                        for c in range(b + 1, len(valid_i)):
                            jc, img_c, d_ic = valid_i[c]
                            pc = pos[jc] + img_c
                            d_ac = float(np.linalg.norm(pa - pc))
                            if d_ac > rc4 + 1e-9:
                                continue
                            d_bc = float(np.linalg.norm(pb - pc))
                            if d_bc > rc4 + 1e-9:
                                continue
                            sig = tuple(sorted((d_ia, d_ib, d_ic, d_ab, d_ac, d_bc)))
                            sh = self._bin_signature(sig, quad_shell_keys)
                            quad_per_shell.setdefault(sh, []).append([i, ja, jb, jc])
            per_body.append((4, quad_per_shell, [k[-1] for k in quad_shell_keys]))

        # Globally collect all distinct shell diameters for the C++ shell array.
        all_diams: list[float] = []
        global_map: list[list[int]] = []
        for n_pts, per_shell, diams in per_body:
            mapping = []
            for d in diams:
                found = -1
                for gi, gd in enumerate(all_diams):
                    if abs(gd - d) < _SHELL_TOL:
                        found = gi
                        break
                if found < 0:
                    all_diams.append(d)
                    found = len(all_diams) - 1
                mapping.append(found)
            global_map.append(mapping)

        return per_body, all_diams, global_map

    def _bin_diameter(self, d: float, bins: list) -> int:
        for k, ref in enumerate(bins):
            if abs(d - ref) < _SHELL_TOL:
                return k
        bins.append(d)
        return len(bins) - 1

    def _bin_signature(self, sig: Tuple[float, ...], bins: list) -> int:
        for k, ref in enumerate(bins):
            if len(ref) != len(sig):
                continue
            if all(abs(a - b) < _SHELL_TOL for a, b in zip(sig, ref)):
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
        engine.objective_mode = _OBJECTIVE_MODE_ATAT
        engine.atat_tol = _ATAT_TOL
        engine.atat_w_dist = 1.0
        engine.atat_w_npts = _WEIGHT_NPTS

        per_body, all_diams, global_map = self._enumerate_clusters()
        d_min = min(all_diams)

        for (n_pts, per_shell, _diams), shell_map in zip(per_body, global_map):
            for local_s, clusters in per_shell.items():
                gs = shell_map[local_s]
                arr = np.array(clusters, dtype=np.int32)
                flat = arr.flatten().tolist()
                engine.add_cluster_body(n_pts, flat, [int(gs)] * len(clusters))

        shell_weights = [float(np.exp(-_WEIGHT_DECAY * (d - d_min))) for d in all_diams]
        engine.shell_weight = shell_weights
        engine.shell_diameter = list(all_diams)
        engine.finalize()

        if self.max_steps <= 0:
            best_types = type_arr.tolist()
            best_corr = engine.correlations(best_types)
            best_obj = engine.objective(best_types)
        else:
            best_types, best_obj, best_corr = engine.run_mc(
                type_arr.tolist(),
                int(self.max_steps),
                float(self.T),
                int(self.n_replicas),
                int(self.seed),
                get_num_threads(),
            )

        from mdapy.system import System

        new_data = self._sys_in.data.clone()
        new_type = np.array(best_types, dtype=np.int32) + 1
        if label_kind == "element":
            new_elem = [labels[t] for t in best_types]
            new_data = new_data.with_columns(pl.Series("element", new_elem))
        new_data = new_data.with_columns(pl.Series("type", new_type))

        self.system = System(data=new_data, box=self._sys_in.box)
        self.objective = float(best_obj)
        self.correlations = np.asarray(best_corr, dtype=np.float64)
        self._engine = engine
        self._best_types = np.asarray(best_types, dtype=np.int64)
        self._species_labels = labels

        infos = []
        for i in range(engine.num_channels()):
            n_pts, shell, funcs, n_inst, target = engine.channel_info(i)
            infos.append(
                {
                    "n_pts": int(n_pts),
                    "shell": int(shell),
                    "diameter": float(all_diams[shell]),
                    "funcs": list(funcs),
                    "n_instances": int(n_inst),
                    "target": float(target),
                    "corr": float(self.correlations[i]),
                }
            )
        self.channel_info = infos
        return self

    # -------------------------------------------------------------- judgment
    def is_sqs(
        self,
        tol: float = 0.03,
        verbose: bool = True,
    ):
        """Verify whether ``self.system`` is a good SQS.

        The verdict follows the formal SQS definition (Zunger 1990;
        van de Walle ATAT 2013): an SQS is a configuration whose cluster
        correlation residuals all sit below tolerance,

            ``max_alpha |pi_alpha - pi_target_alpha| < tol``

        across every cluster channel (pair / triplet / quad) the optimiser
        was given. ``tol=0.03`` is fine for ~100-atom equiatomic cells;
        tighten as ``N`` grows.

        For convenience this method **also** computes Warren-Cowley
        ``max |alpha_ij|`` (full matrix, including the diagonal) at every
        pair shell within ``cutoffs[2]`` and lists them in the report.
        WCP is a derived physical quantity — it's a linear transform of
        the pair correlations — and is *not* part of the verdict, mirroring
        what ATAT mcsqs does. It's there so the user can inspect the SRO,
        especially in non-equiatomic / small-cell cases where the cell may
        not have enough degrees of freedom to drive every alpha to zero
        even when the correlation residuals all pass.

        Parameters
        ----------
        tol : float, default=0.03
            Threshold for the absolute-residual verdict.
        verbose : bool, default=True
            Print a one-screen summary report.

        Returns
        -------
        passed : bool
            True iff ``max |Delta pi| < tol``.
        info : dict
            Diagnostic numbers behind the verdict.
        """
        if self._engine is None:
            raise RuntimeError("call compute() before is_sqs()")

        engine = self._engine
        types = self._best_types.tolist()
        labels = self._species_labels

        # ----- 1) absolute correlation residual ---------------------------
        delta_all = np.array(engine.per_channel_delta(types))
        max_delta = float(np.max(delta_all)) if len(delta_all) else 0.0
        absolute_pass = max_delta < tol

        # ----- 2) Warren-Cowley per pair shell (informational only) -------
        sys_out = self.system
        pair_d = sorted(
            {ci["diameter"] for ci in self.channel_info if ci["n_pts"] == 2}
        )
        per_shell = []
        for s_idx, d_s in enumerate(pair_d):
            rc = d_s + _SHELL_TOL
            sys_out.build_neighbor(rc=rc)
            wcp = sys_out.cal_warren_cowley_parameter(rc=rc)
            mat = wcp.WCP
            mat_off = mat - np.diag(np.diag(mat))
            per_shell.append(
                {
                    "shell": f"NN{s_idx + 1}",
                    "diameter": float(d_s),
                    "rc": float(rc),
                    "max_abs": float(np.max(np.abs(mat))),
                    "max_off_diag": float(np.max(np.abs(mat_off))),
                    "matrix": mat,
                }
            )

        # Verdict follows the formal SQS definition (correlation residual).
        # WCP is reported alongside but does NOT enter the verdict, matching
        # ATAT mcsqs behaviour.
        verdict = absolute_pass

        # ----- composition string for the report --------------------------
        from collections import Counter

        type_counter = Counter(types)
        comp_str = ", ".join(
            f"{labels[t]}({type_counter[t]})" for t in sorted(type_counter)
        )

        # ----- per-body channel counts ------------------------------------
        bcount = Counter(ci["n_pts"] for ci in self.channel_info)
        body_str = "  ".join(
            f"{name}={bcount.get(n, 0)}"
            for n, name in [(2, "pair"), (3, "triplet"), (4, "quad")]
            if bcount.get(n, 0) > 0
        )

        info = {
            "verdict": verdict,
            "absolute": {
                "pass": absolute_pass,
                "max_delta": max_delta,
                "tol": tol,
            },
            "warren_cowley": {
                "tol": tol,
                "per_shell": per_shell,
            },
        }

        if verbose:
            n = self._sys_in.N
            print(f"SQS verification ({n} atoms; species: {comp_str})")
            print("-" * 60)
            print(f"correlations    : {len(self.channel_info)} channels  ({body_str})")
            print(f"objective       : {self.objective:.5f}")
            print()
            ok_abs = "PASS" if absolute_pass else "FAIL"
            print(
                f"absolute residual   max|pi - target| = {max_delta:.4f}"
                f"   tol={tol:.3f}   {ok_abs}    <- decides verdict"
            )
            for s in per_shell:
                print(
                    f"WCP {s['shell']:>3s}  d={s['diameter']:.3f} A    "
                    f"max|alpha|={s['max_abs']:.4f}   tol={tol:.3f}   INFO"
                )
            print()
            print(f"Verdict: {'SQS' if verdict else 'NOT YET'}")

        return verdict, info


if __name__ == "__main__":
    from mdapy import build_hea

    hea = build_hea(
        ("Cr", "Co", "Ni"),
        (1 / 3, 1 / 3, 1 / 3),
        "fcc",
        3.53,
        nx=2,
        ny=2,
        nz=2,
        random_seed=2,
    )
    sqs = SQS(
        hea,
        cutoffs={2: 4.0, 3: 3.0},
        max_steps=20000,
        n_replicas=4,
        seed=1,
        T=2.0,
    ).compute()
    sqs.is_sqs()
    # sqs.system.write_poscar("POSCAR")
