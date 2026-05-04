# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Bond stiffness vs bond length, modeled after ATAT's *fitsvsl*.

Each chemical bond (i,j) between species (A,B) is treated as a 3D harmonic
spring whose force-constant matrix in the bond-aligned frame is
``diag(k_l, k_t, k_t)`` — one **longitudinal** (stretch) constant and two
identical **transverse** (bending) constants. Both are assumed to be
*universal functions of bond length only* for a given element pair (the
"bond stiffness vs bond length" approximation used by ATAT *svsl* /
*fitsvsl*; see van de Walle & Ceder, *Rev. Mod. Phys.* **74**, 11, 2002, and
Pinsook & Ceder, *Phys. Rev. B* **60**, 11997, 1999):

.. math::

    k_l(r) = c^l_0 + c^l_1 r + c^l_2 r^2 + \\dots

    k_t(r) = c^t_0 + c^t_1 r + c^t_2 r^2 + \\dots

Workflow
--------

1. Build the equilibrium neighbor list from the input :class:`mdapy.System`.
   Bonds within ``rc_bond`` are grouped into distance *shells* (NN1,
   NN2, ...) using ``shell_tol``; each shell gets its own ``(k_l, k_t)``
   polynomial. For BCC and similar lattices where the second-neighbor
   shell carries non-negligible stiffness, set ``rc_bond`` large enough
   to include 2NN.
2. Generate :math:`6N` perturbations: each atom displaced by
   :math:`\\pm\\delta` along x, y, z. Repeat at ``n_lattice`` isotropic
   strains symmetrically distributed across
   ``[-max_strain, +max_strain]`` to scan a range of bond lengths
   (needed for ``poly_order >= 1``).
3. Compute forces with the user-supplied ``calculator`` (any
   :class:`mdapy.calculator.CalculatorMP` subclass — typically
   :class:`mdapy.NEP`).
4. Solve a single ordinary-least-squares system whose unknowns are the
   polynomial coefficients ``(c^l_q, c^t_q)`` for every
   ``(element_pair, shell, q)`` triple, and whose rows are the per-atom
   force components for every perturbation. The harmonic-pair model
   gives the force on atom :math:`k` from a perturbation :math:`d_p` of
   atom :math:`p` as

   .. math::

       F_k = \\sum_{(i,j)\\ \\text{containing}\\ p,k}
             \\bigl[k_l\\, (d_{rel}\\!\\cdot\\!\\hat u)\\hat u
             + k_t\\, (d_{rel} - (d_{rel}\\!\\cdot\\!\\hat u)\\hat u)\\bigr]

   with the appropriate sign so that Newton's third law holds. This is
   the same formulation used by ATAT *fitsvsl* and gives a
   well-conditioned, statistically efficient fit.
5. Visualise: :meth:`BondStiffness.plot` reproduces the ATAT-style
   ``stiffness vs bond length`` figure — one panel per shell, with one
   colour per element pair, scatter for raw data, polynomial overlay.
6. Write the result in (multi-shell-extended) ATAT *slspring.out*
   format for cross-validation or downstream use with *svsl*.

Citation
--------

Using this class should cite the ATAT *fitsvsl* method papers:

  A. van de Walle, M. Asta. *Self-driven lattice-model Monte Carlo
  simulations of alloy thermodynamic properties and phase diagrams.*
  Modelling Simul. Mater. Sci. Eng. **10**, 521 (2002).

  E. J. Wu, G. Ceder, A. van de Walle. *Using bond-length-dependent
  transferable force constants to predict vibrational entropies in
  Au-Cu, Au-Pd and Cu-Pd alloys.* Phys. Rev. B **67**, 134103 (2003).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from mdapy.system import System
    from mdapy.calculator import CalculatorMP


class BondStiffness:
    """Fit bond longitudinal and transverse stiffness vs bond length.

    Parameters
    ----------
    system : mdapy.System
        Reference structure (already relaxed; atoms in equilibrium positions).
        Must have an ``element`` column.
    calculator : CalculatorMP
        Force calculator (e.g. :class:`mdapy.NEP`). Must implement
        ``get_forces(data, box) -> (N, 3)``.
    rc_bond : float, optional
        Bond cutoff in Å. All atom pairs with distance below this
        threshold are considered bonds. Pairs are then automatically
        grouped into *distance shells* (NN1, NN2, ...) with tolerance
        ``shell_tol``; each shell is fit independently. If omitted,
        defaults to 1.05 × the smallest non-zero pairwise distance in
        ``system`` — a 1NN-only cutoff. For BCC and similar lattices
        where 2NN contributes meaningfully, pass an ``rc_bond`` large
        enough to include both shells (e.g. ``rc_bond = 1.2 * a`` for
        BCC catches NN1 ≈ a√3/2 and NN2 = a).
    shell_tol : float, default=0.1
        Distance tolerance (Å) for grouping bonds into the same shell.
        Two bonds within ``shell_tol`` of each other share a shell
        index; bonds that move within their shell across strains
        provide the (r, k) data points for a length-dependent fit.
    delta : float, default=0.05
        Magnitude of atomic displacement used to probe forces, in Å.
        Use central differences (``+delta`` and ``-delta`` per axis) so
        the linear-response coefficient is extracted exactly even with a
        small anharmonic part.
    poly_order : int, default=1
        Polynomial order of the ``k(r)`` fit. ``0`` gives a single
        constant per (element pair, shell). For ``poly_order >= 1``
        set ``n_lattice >= 2`` to actually scan bond lengths.
    n_lattice : int, default=3
        Number of isotropic strains at which the perturbation set is
        repeated. Strains are spaced linearly and *symmetrically* across
        ``[-max_strain, +max_strain]`` (so ``n_lattice=1`` gives
        ``[0]``, ``n_lattice=2`` gives ``[-max_strain, +max_strain]``,
        ``n_lattice=3`` gives ``[-max_strain, 0, +max_strain]``, etc.).
        This contrasts with ATAT *fitsvsl*'s ``-ns`` which only
        stretches; we scan symmetrically since extrapolating to
        compressed states is just as physical as expanded ones.
    max_strain : float, default=0.02
        Maximum isotropic strain magnitude. With the default
        ``n_lattice=3`` this scans bond lengths over ``±2 %``.
    central_diff : bool, default=True
        If True, displace each atom by both :math:`+\\delta` and
        :math:`-\\delta` per axis and use central differences. Doubles
        the force calculations but makes the linear response exact in
        the limit of small :math:`\\delta`. If False, displace only
        :math:`+\\delta` (forward differences; cheaper, less accurate).
    rcond : float, default=1e-6
        Relative cutoff for singular values in the OLS pseudo-inverse.
        Singular values below ``rcond * max(sv)`` are treated as zero,
        regularising the fit when polynomial-feature columns are nearly
        collinear (e.g. ``poly_order >= 1`` with a tiny strain range).

    Attributes
    ----------
    bond_table : polars.DataFrame
        One row per bond per perturbation strain: columns
        ``element_a``, ``element_b``, ``shell``, ``r``, ``k_long``,
        ``k_trans``, ``strain``. Set by :meth:`compute`.
    shells : list[float]
        Mean diameter (in Å) of each distance shell, sorted from
        nearest to farthest. ``len(shells)`` is the number of shells
        retained.
    k_long, k_trans : dict[tuple[str, str, int], numpy.ndarray]
        Fitted polynomial coefficients (lowest order first) keyed by
        ``(element_a, element_b, shell)``. Element pair is sorted
        alphabetically so ``("Al","Cr",0)`` and ``("Cr","Al",0)``
        always map to the same key. Set by :meth:`compute`.
    """

    def __init__(
        self,
        system: "System",
        calculator: "CalculatorMP",
        rc_bond: Optional[float] = None,
        shell_tol: float = 0.1,
        delta: float = 0.05,
        poly_order: int = 1,
        n_lattice: int = 3,
        max_strain: float = 0.02,
        central_diff: bool = True,
        rcond: float = 1e-6,
    ):
        if "element" not in system.data.columns:
            raise ValueError("system must have an 'element' column")
        self._sys = system
        self._calc = calculator
        self.delta = float(delta)
        self.poly_order = int(poly_order)
        self.n_lattice = int(n_lattice)
        self.max_strain = float(max_strain)
        self.central_diff = bool(central_diff)
        self.rc_bond = float(rc_bond) if rc_bond is not None else None
        self.shell_tol = float(shell_tol)
        self.rcond = float(rcond)

        # filled in by compute()
        self.bond_table: Optional[pl.DataFrame] = None
        self.shells: List[float] = []
        self.k_long: Dict[Tuple[str, str, int], np.ndarray] = {}
        self.k_trans: Dict[Tuple[str, str, int], np.ndarray] = {}

    # ------------------------------------------------------------------ helpers
    def _auto_cutoff(self, system: "System") -> float:
        """Return 1.05 × the shortest non-zero pairwise distance."""
        # quick estimate via Neighbor with a generous cutoff
        rc_probe = 0.5 * min(
            np.linalg.norm(system.box.box[0]),
            np.linalg.norm(system.box.box[1]),
            np.linalg.norm(system.box.box[2]),
        )
        system.build_neighbor(rc=min(rc_probe, 5.0))
        d = system.distance_list
        d = d[d > 0]
        return float(d.min()) * 1.05

    def _build_bonds(self, system: "System", rc: float):
        """Return the list of unique bonds in ``system``.

        Each bond is a 5-tuple ``(i, j, dx, dy, dz, r)`` where
        ``(i, j)`` are *primary* atom indices (i.e. in the original cell,
        not enlarged-replicated indices), ``(dx,dy,dz)`` is the cartesian
        vector from atom i to atom j (already periodic-image-resolved),
        and ``r = ||(dx,dy,dz)||``.

        Each (i, j, image) pair appears once. mdapy's neighbor list may
        replicate small boxes; we fold ghost indices via
        ``j_real = j_enl % N`` and keep all distinct geometric instances.
        """
        system.build_neighbor(rc=rc)
        verlet = system.verlet_list
        nnum = system.neighbor_number
        dist = system.distance_list
        N = system.N

        # We need bond *vectors*, not just distances — neighbor.distance_list
        # only gives scalars. Reconstruct by looking up positions and
        # computing min-image displacement.
        x = system.data["x"].to_numpy(allow_copy=False)
        y = system.data["y"].to_numpy(allow_copy=False)
        z = system.data["z"].to_numpy(allow_copy=False)
        pos = np.stack([x, y, z], axis=1)
        cell = system.box.box
        inv = np.linalg.inv(cell)

        bonds = []
        for i in range(N):
            for k in range(int(nnum[i])):
                j_enl = int(verlet[i, k])
                if j_enl < 0:
                    continue
                d = float(dist[i, k])
                if d > rc + 1e-9:
                    continue
                j = j_enl % N
                # in the central image we keep only j > i to avoid duplicates;
                # ghost copies (j_enl >= N) keep both directions because they
                # encode different periodic images.
                if j_enl < N and j <= i:
                    continue
                # min-image vector from i to image of j
                dr = pos[j] - pos[i]
                # remove integer multiples of cell vectors so |dr| matches dist
                f = dr @ inv
                f = f - np.round(f)
                dr = f @ cell
                # if mdapy replicated the box, the distance dr above might
                # not match because pos is in the original cell. Recover by
                # matching length: pick the shift of dr that has length d.
                # First check current length:
                if abs(np.linalg.norm(dr) - d) > 1e-3:
                    # try ±cell vectors to find the right image
                    best = dr
                    best_err = abs(np.linalg.norm(dr) - d)
                    for s0 in (-1, 0, 1):
                        for s1 in (-1, 0, 1):
                            for s2 in (-1, 0, 1):
                                cand = (
                                    pos[j]
                                    - pos[i]
                                    + s0 * cell[0]
                                    + s1 * cell[1]
                                    + s2 * cell[2]
                                )
                                err = abs(np.linalg.norm(cand) - d)
                                if err < best_err:
                                    best_err = err
                                    best = cand
                    dr = best
                bonds.append((i, j, dr[0], dr[1], dr[2], d))
        return bonds

    def _scaled_system(self, factor: float) -> "System":
        """Return a copy of ``self._sys`` with isotropic strain ``factor``.

        Both cell vectors and atom positions scale by ``factor``.
        """
        from mdapy.system import System

        data = self._sys.data.clone()
        for col in ("x", "y", "z"):
            data = data.with_columns((pl.col(col) * factor).alias(col))
        new_box = self._sys.box.box * factor
        return System(data=data, box=new_box)

    def _displace_force(
        self, system: "System", atom_idx: int, axis: int, sign: int
    ) -> np.ndarray:
        """Compute force on every atom after displacing one atom by
        ``sign * delta`` along ``axis`` (0=x, 1=y, 2=z)."""
        data = system.data.clone()
        col = "xyz"[axis]
        arr = data[col].to_numpy().copy()
        arr[atom_idx] += sign * self.delta
        data = data.with_columns(pl.Series(col, arr))

        # force calc — re-instantiate System so the calculator sees fresh data
        from mdapy.system import System

        s = System(data=data, box=system.box)
        s.calc = self._calc
        return s.get_force()

    # ------------------------------------------------------------------ compute
    def compute(self) -> "BondStiffness":
        """Run the full pipeline and populate :attr:`k_long`,
        :attr:`k_trans`, :attr:`bond_table`. Returns ``self``.

        The fit is set up exactly as in ATAT *fitsvsl*: a single ordinary
        least-squares system whose unknowns are the polynomial
        coefficients ``(c^l_0,...,c^l_p, c^t_0,...,c^t_p)`` for each
        element pair, NOT per-bond stiffnesses. This is the well-
        conditioned formulation that gives a numerically stable answer
        even when many bonds of the same chemical pair share identical
        lengths.
        """
        rc = self.rc_bond if self.rc_bond is not None else self._auto_cutoff(self._sys)
        self.rc_bond = rc

        # Symmetric strain range: ``n_lattice == 1`` collapses to [0];
        # otherwise linspace(-max_strain, +max_strain, n_lattice).
        if self.n_lattice <= 1:
            strains = [0.0]
        else:
            strains = list(
                np.linspace(-self.max_strain, +self.max_strain, self.n_lattice)
            )

        elements = self._sys.data["element"].to_list()

        # ---- discover element pairs (alphabetic ordering keeps (A,B)≡(B,A))
        pair_set = set()
        for ea in set(elements):
            for eb in set(elements):
                a, b = (ea, eb) if ea <= eb else (eb, ea)
                pair_set.add((a, b))
        pairs = sorted(pair_set)
        pair_idx = {p: k for k, p in enumerate(pairs)}
        n_pairs = len(pairs)
        ncoef = self.poly_order + 1

        # ---- pre-pass: discover distance shells from the unstrained cell.
        # Bonds are binned by their *unstrained* bond length so a given
        # bond keeps the same shell index across all strain samples.
        bonds_by_strain = []  # list aligned with `strains`
        sys_eq = self._scaled_system(1.0)
        eq_bonds = self._build_bonds(sys_eq, rc)
        # cluster equilibrium bond lengths into shells using shell_tol
        eq_lengths = sorted(b[5] for b in eq_bonds)
        shell_centers: List[float] = []
        for L in eq_lengths:
            if not shell_centers or abs(L - shell_centers[-1]) > self.shell_tol:
                shell_centers.append(L)
        # refine each shell center by averaging its members
        shell_members: List[List[float]] = [[] for _ in shell_centers]
        for L in eq_lengths:
            for s, c in enumerate(shell_centers):
                if abs(L - c) < self.shell_tol:
                    shell_members[s].append(L)
                    break
        shell_centers = [float(np.mean(m)) for m in shell_members]
        n_shells = len(shell_centers)
        self.shells = shell_centers

        def _shell_of(L: float) -> int:
            best, best_err = -1, 1e9
            for s, c in enumerate(shell_centers):
                err = abs(L - c)
                if err < best_err:
                    best_err = err
                    best = s
            return best

        # Column layout:  for each (pair, shell), 2*ncoef columns
        #   (k_l_coef0..p, k_t_coef0..p). Total cols = n_pairs * n_shells * 2 * ncoef.
        cols_per_shell = 2 * ncoef
        cols_per_pair = n_shells * cols_per_shell
        n_cols = n_pairs * cols_per_pair

        # ---- build the giant OLS system across ALL strains
        big_A_rows = []
        big_y_rows = []
        bond_records = []

        for strain in strains:
            system = self._scaled_system(1.0 + strain)
            bonds = self._build_bonds(system, rc)

            system.calc = self._calc
            F_eq = system.get_force()

            N = system.N
            atom_bonds: List[List[int]] = [[] for _ in range(N)]
            bond_shells = []
            for b_idx, (i, j, _, _, _, length) in enumerate(bonds):
                atom_bonds[i].append(b_idx)
                atom_bonds[j].append(b_idx)
                # shell index from *unstrained* equilibrium length, found
                # by un-scaling: equivalent to using strained length divided
                # by (1+strain), because the cell is isotropically scaled.
                bond_shells.append(_shell_of(length / (1.0 + strain)))

            signs = (1, -1) if self.central_diff else (1,)

            # Per-bond observation accumulators (used for the bond_table
            # scatter data). For each perturbation that touches a bond,
            # the force on the bond's *other* endpoint comes only from
            # this bond (in the harmonic pair model), so projecting that
            # force onto the longitudinal / transverse displacement
            # directions isolates the bond's individual (k_l, k_t):
            #
            #   F_q = +k_l * d_l + k_t * d_t
            #   k_l_obs = (F_q · d_l) / |d_l|^2
            #   k_t_obs = (F_q · d_t) / |d_t|^2
            #
            # ATAT writes the same per-observation data into its
            # ``f_<A>-<B>.dat`` files; we average across all
            # perturbations touching a bond to get a single (r, k_l,
            # k_t) row per bond per strain.
            kl_sum = np.zeros(len(bonds))
            kt_sum = np.zeros(len(bonds))
            kl_n   = np.zeros(len(bonds), dtype=np.int64)
            kt_n   = np.zeros(len(bonds), dtype=np.int64)

            for atom_idx in range(N):
                for axis in range(3):
                    for sign in signs:
                        F = self._displace_force(system, atom_idx, axis, sign)
                        dF = F - F_eq
                        block = np.zeros((3 * N, n_cols), dtype=np.float64)
                        d_p = np.zeros(3)
                        d_p[axis] = sign * self.delta
                        for b_idx in atom_bonds[atom_idx]:
                            i, j, dx, dy, dz, length = bonds[b_idx]
                            shell_id = bond_shells[b_idx]
                            u = np.array([dx, dy, dz]) / length
                            if atom_idx == i:
                                d_rel = d_p
                                q_atom = j
                            else:
                                d_rel = -d_p
                                q_atom = i
                            d_l = (d_rel @ u) * u
                            d_t = d_rel - d_l

                            # OLS row contribution
                            a = elements[i]
                            b = elements[j]
                            if a > b:
                                a, b = b, a
                            pid = pair_idx[(a, b)]
                            base = pid * cols_per_pair + shell_id * cols_per_shell
                            for q in range(ncoef):
                                col_l = base + q
                                col_t = base + ncoef + q
                                rq = length**q
                                for ax in range(3):
                                    block[3 * i + ax, col_l] += -d_l[ax] * rq
                                    block[3 * i + ax, col_t] += -d_t[ax] * rq
                                    block[3 * j + ax, col_l] += d_l[ax] * rq
                                    block[3 * j + ax, col_t] += d_t[ax] * rq

                            # per-bond projection (raw observation).
                            # Use the *signed* displacement d_p, not
                            # d_rel: F_q = +k_l * d_l_pos + k_t *
                            # d_t_pos always (whether the perturbed
                            # atom is the bond's i or its j endpoint),
                            # because flipping i<->j flips both d_rel
                            # and the role of (i, j), and they cancel.
                            d_l_pos = (d_p @ u) * u
                            d_t_pos = d_p - d_l_pos
                            F_q = dF[q_atom]
                            ldn = float(d_l_pos @ d_l_pos)
                            tdn = float(d_t_pos @ d_t_pos)
                            if ldn > 1e-12:
                                kl_sum[b_idx] += float(F_q @ d_l_pos) / ldn
                                kl_n  [b_idx] += 1
                            if tdn > 1e-12:
                                kt_sum[b_idx] += float(F_q @ d_t_pos) / tdn
                                kt_n  [b_idx] += 1
                        big_A_rows.append(block)
                        big_y_rows.append(dF.reshape(-1))

            for b_idx, (i, j, _, _, _, length) in enumerate(bonds):
                a, b = elements[i], elements[j]
                if a > b:
                    a, b = b, a
                kl_obs = kl_sum[b_idx] / kl_n[b_idx] if kl_n[b_idx] > 0 else np.nan
                kt_obs = kt_sum[b_idx] / kt_n[b_idx] if kt_n[b_idx] > 0 else np.nan
                bond_records.append(
                    {
                        "element_a": a,
                        "element_b": b,
                        "shell":   int(bond_shells[b_idx]),
                        "r":       float(length),
                        "strain":  float(strain),
                        "k_long":  float(kl_obs),
                        "k_trans": float(kt_obs),
                    }
                )

        big_A = np.concatenate(big_A_rows, axis=0)
        big_y = np.concatenate(big_y_rows, axis=0)
        # SVD-based least-squares with a relative cutoff. Polynomial
        # features (1, r, r^2, ...) are highly collinear when all bond
        # lengths are within a tiny range, so we drop near-zero singular
        # directions to avoid coefficients blowing up under noise. ATAT
        # *fitsvsl* achieves the same end via Gram-Schmidt
        # (``list_nonredundant_columns``); the two parameterisations
        # differ when ``poly_order >= 1`` but the polynomial *evaluated
        # at the data range* is the same.
        beta, *_ = np.linalg.lstsq(big_A, big_y, rcond=self.rcond)

        # ---- unpack solution per (pair, shell)
        self.k_long.clear()
        self.k_trans.clear()
        for pair, pid in pair_idx.items():
            for s in range(n_shells):
                base = pid * cols_per_pair + s * cols_per_shell
                key = (pair[0], pair[1], s)
                self.k_long[key] = beta[base : base + ncoef].copy()
                self.k_trans[key] = beta[base + ncoef : base + 2 * ncoef].copy()

        # bond_records already carry the raw per-bond projection-based
        # observations of (k_long, k_trans); the polynomial coefficients
        # in self.k_long / self.k_trans come from the global OLS fit.
        # The two are intentionally separate: the scatter shows the data,
        # the polynomial shows the fit.
        self.bond_table = pl.DataFrame(bond_records)

        return self

    # ------------------------------------------------------------------ outputs
    def write_slspring(self, path: str) -> None:
        """Write fitted ``(k_l, k_t)`` polynomials to ``path`` in ATAT
        ``slspring.out`` format (extended for multi-shell output).

        With one shell, the file is byte-for-byte compatible with what
        ATAT *fitsvsl* writes. With multiple shells, each (element pair,
        shell) becomes a separate block annotated with ``# shell <id>
        d=<diameter>`` so you can tell them apart while keeping the
        section structure ATAT *svsl* parses.

        File layout::

            <element_a> <element_b>     # shell 0 d=<NN1 distance>
            <n_coeffs>
            c_l0
            ...
            <n_coeffs>
            c_t0
            ...
            <element_a> <element_b>     # shell 1 d=<NN2 distance>
            ...
        """
        if not self.k_long:
            raise RuntimeError("call compute() before write_slspring()")
        with open(path, "w") as f:
            for key in sorted(self.k_long):
                ea, eb, shell = key
                kl = self.k_long[key]
                kt = self.k_trans[key]
                d = self.shells[shell]
                if len(self.shells) > 1:
                    f.write(f"{ea} {eb}    # shell {shell} d={d:.4f}\n")
                else:
                    f.write(f"{ea} {eb}\n")
                f.write(f"{len(kl)}\n")
                for c in kl:
                    f.write(f"{c:.5f}\n")
                f.write(f"{len(kt)}\n")
                for c in kt:
                    f.write(f"{c:.5f}\n")

    # ----------------------------------------------------------------- plotting
    def plot(
        self,
        which: str = "both",
        ax=None,
        ncol: Optional[int] = None,
    ):
        """Plot fitted bond stiffness vs bond length.

        Produces **one panel per element pair**, with all distance shells
        (NN1, NN2, ...) drawn on the same axes. Different shells use
        different scatter markers; each shell's polynomial fit is
        overlaid as a continuous curve within that shell's r range.
        This reproduces the canonical "stiffness vs bond length" figure
        of Wu, Ceder, van de Walle (*MSMSE* **10**, 521, 2002), where
        all shells of a chemical pair share a single panel and the
        x-axis spans the full r range covered by the chosen
        ``rc_bond``.

        To get a wider x-range like the published figures (~2.5 to
        5.0 Å), pass an ``rc_bond`` covering NN1..NN4 (e.g. ~1.4 × the
        lattice parameter for fcc).

        Parameters
        ----------
        which : {'both', 'long', 'trans'}, default='both'
            Which stiffness mode to plot. ``'both'`` shows
            longitudinal and transverse on the same panel (different
            line styles).
        ax : matplotlib Axes or list of Axes, optional
            Pre-existing axes to plot into. If omitted, a fresh figure
            is created via :func:`mdapy.set_figure`.
        ncol : int, optional
            Number of subplot columns when multiple element pairs are
            plotted and ``ax`` is None. Defaults to ``min(3, n_pairs)``.

        Returns
        -------
        fig, axes : matplotlib Figure, list of Axes
            One axes per element pair.
        """
        if self.bond_table is None:
            raise RuntimeError("call compute() before plot()")
        if which not in ("both", "long", "trans"):
            raise ValueError("which must be 'both', 'long' or 'trans'")

        import matplotlib.pyplot as plt
        from mdapy.plotset import set_figure

        df = self.bond_table
        pairs = sorted(set(zip(df["element_a"].to_list(), df["element_b"].to_list())))
        n_pairs  = len(pairs)
        n_shells = len(self.shells)
        ncoef    = self.poly_order + 1

        if ax is None:
            ncol = ncol if ncol is not None else min(3, n_pairs)
            nrow = int(np.ceil(n_pairs / ncol))
            fig, axes = set_figure(
                figsize=(8.5 * ncol, 6.5 * nrow), nrow=nrow, ncol=ncol
            )
            if n_pairs == 1:
                axes = [axes]
            else:
                if isinstance(axes, list) and isinstance(axes[0], list):
                    axes = [a for row in axes for a in row]
        else:
            axes = ax if isinstance(ax, (list, tuple)) else [ax]
            fig = axes[0].figure

        cmap = plt.colormaps.get_cmap("tab10")
        long_color  = cmap(0)
        trans_color = cmap(1)
        shell_markers_l = ("o", "^", "v", "D", "p", "h", "*", "X", "8")
        shell_markers_t = ("s", "<", ">", "P", "x", "+", "1", "2", "3")

        for p_idx, (ea, eb) in enumerate(pairs):
            a = axes[p_idx]
            pair_sub = df.filter(
                (pl.col("element_a") == ea) & (pl.col("element_b") == eb)
            )
            if pair_sub.is_empty():
                a.set_visible(False)
                continue

            for s_idx in range(n_shells):
                shell_sub = pair_sub.filter(pl.col("shell") == s_idx)
                if shell_sub.is_empty():
                    continue
                # Average all bonds at the same r within this shell so
                # each r gives a single point (the rest is symmetry-
                # equivalent noise that just clutters the plot).
                # Quantise r to 4 decimals to group floating-point
                # duplicates from PBC reconstruction.
                grouped = (
                    shell_sub
                    .with_columns((pl.col("r") * 1e4).round() / 1e4)
                    .group_by("r")
                    .agg([pl.col("k_long").mean(), pl.col("k_trans").mean()])
                    .sort("r")
                )
                m_l = shell_markers_l[s_idx % len(shell_markers_l)]
                m_t = shell_markers_t[s_idx % len(shell_markers_t)]
                if which in ("both", "long"):
                    a.scatter(
                        grouped["r"], grouped["k_long"],
                        s=22, color=long_color, marker=m_l, alpha=0.85,
                        label=(f"NN{s_idx+1} long" if which == "both"
                               else f"NN{s_idx+1}"),
                    )
                if which in ("both", "trans"):
                    a.scatter(
                        grouped["r"], grouped["k_trans"],
                        s=22, color=trans_color, marker=m_t, alpha=0.85,
                        label=(f"NN{s_idx+1} trans" if which == "both"
                               else f"NN{s_idx+1}"),
                    )

                # polynomial overlay within the shell's r range
                key = (ea, eb, s_idx)
                if key not in self.k_long:
                    continue
                cl = self.k_long [key]
                ct = self.k_trans[key]
                rs = grouped["r"].to_numpy()
                if rs.size == 0:
                    continue
                r_min, r_max = float(rs.min()), float(rs.max())
                pad = 0.02 * (r_max - r_min + 0.01)
                r_lin = np.linspace(r_min - pad, r_max + pad, 50)
                if which in ("both", "long"):
                    a.plot(r_lin,
                           sum(cl[q] * r_lin**q for q in range(ncoef)),
                           "-", color=long_color, linewidth=1.2)
                if which in ("both", "trans"):
                    a.plot(r_lin,
                           sum(ct[q] * r_lin**q for q in range(ncoef)),
                           "--", color=trans_color, linewidth=1.2)

            a.axhline(0.0, color="k", linewidth=0.5, linestyle=":")
            a.set_xlabel("Bond length r (Å)")
            ylab = ("k_long, k_trans" if which == "both"
                    else "k_long" if which == "long" else "k_trans")
            a.set_ylabel(f"{ylab} (eV/Å²)")
            a.set_title(f"{ea} - {eb} bonds")
            a.legend(fontsize=7, loc="best")

        # hide any leftover axes when n_pairs doesn't fill the grid
        for k in range(n_pairs, len(axes)):
            axes[k].set_visible(False)
        return fig, axes

    def generate_perturbed_structures(
        self,
        output_dir: Optional[str] = None,
    ) -> List["System"]:
        """Generate the list of perturbed :class:`mdapy.System` objects
        used internally by :meth:`compute`. If ``output_dir`` is given,
        also dump each as an ATAT-format ``str.out`` file (and a single
        ``str_unpert.out``) ready for an external DFT or potential
        run, plus a ``perturb_index.txt`` describing the perturbation
        applied to each.

        Returns the list of System objects in the same order they are
        used by :meth:`compute`. Note this enumerates only the n_lattice=1
        (unstrained) case; for multi-strain workflows call this method
        with different ``self.max_strain / n_lattice`` settings.
        """
        rc = self.rc_bond if self.rc_bond is not None else self._auto_cutoff(self._sys)
        self.rc_bond = rc

        out: List["System"] = []
        signs = (1, -1) if self.central_diff else (1,)
        index_lines = []
        from mdapy.system import System

        unpert = self._sys
        if output_dir:
            outp = Path(output_dir)
            outp.mkdir(parents=True, exist_ok=True)
            self._write_atat_struct(unpert, outp / "str_unpert.out")
            index_lines.append("# id  atom_idx  axis  sign  delta")

        for atom_idx in range(self._sys.N):
            for axis in range(3):
                for sign in signs:
                    data = self._sys.data.clone()
                    col = "xyz"[axis]
                    arr = data[col].to_numpy().copy()
                    arr[atom_idx] += sign * self.delta
                    data = data.with_columns(pl.Series(col, arr))
                    s = System(data=data, box=self._sys.box)
                    out.append(s)
                    if output_dir:
                        sub = outp / f"p{len(out)-1:05d}"
                        sub.mkdir(exist_ok=True)
                        self._write_atat_struct(s, sub / "str.out")
                        self._write_atat_struct(unpert, sub / "str_ideal.out")
                        self._write_atat_struct(unpert, sub / "str_unpert.out")
                        index_lines.append(
                            f"{len(out)-1}  {atom_idx}  {axis}  {sign:+d}  {self.delta}"
                        )
        if output_dir:
            (outp / "perturb_index.txt").write_text("\n".join(index_lines) + "\n")
        return out

    def _write_atat_struct(self, system: "System", path: Path) -> None:
        """Write ``system`` in ATAT structure-file format.

        Format::

            1 0 0
            0 1 0
            0 0 1
            <ax> <ay> <az>
            <bx> <by> <bz>
            <cx> <cy> <cz>
            <x1> <y1> <z1> <element1>
            ...
        """
        cell = system.box.box
        x = system.data["x"].to_numpy()
        y = system.data["y"].to_numpy()
        z = system.data["z"].to_numpy()
        elem = system.data["element"].to_list()
        with open(path, "w") as f:
            # identity coord system → atomic coords are cartesian
            f.write("1 0 0\n0 1 0\n0 0 1\n")
            for v in cell:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for k in range(system.N):
                f.write(f"{x[k]:.6f} {y[k]:.6f} {z[k]:.6f} {elem[k]}\n")


if __name__ == "__main__":
    import mdapy as mp
    import matplotlib.pyplot as plt

    nep = mp.NEP(
        "/m/home/home2/22/wuy33/unix/Study/mdapy/tests/input_files/UNEP-v1.txt"
    )
    # Au-Cu-Pd ternary FCC HEA, mirroring the chemistry of the
    # canonical Wu/Ceder/van de Walle (PRB 67, 134103, 2003) figures.
    # Lattice constant 3.85 Å sits between pure Au (4.08 Å) and pure
    # Cu (3.61 Å); FIRE relaxation below pulls atoms slightly off the
    # ideal sites so each chemical environment ends up with a
    # slightly different equilibrium bond length — that's where the
    # *within-strain* r-spread comes from.
    sys_ = mp.build_hea(
        ("Au", "Cu", "Pd"),
        (1.0 / 3, 1.0 / 3, 1.0 / 3),
        "fcc",
        a=3.85,
        nx=4,
        ny=4,
        nz=4,
        random_seed=1,
    )

    # FIRE relaxation: positions only (cell fixed). Each Au atom is
    # bigger than each Cu, so the relaxed alloy is no longer perfectly
    # periodic and bonds at the same shell scatter in length around
    # their average.
    sys_.calc = nep
    fire = mp.FIRE(sys_)
    fire.run(steps=300, fmax=1e-3, show_process=False)

    # Wide rc_bond so we sample NN1..NN4 like the PRB-2003 figures;
    # a 9-point symmetric ±5 % volume scan further densifies r along
    # the x-axis.
    bsl = mp.BondStiffness(
        sys_,
        calculator=nep,
        rc_bond=5.6,
        delta=0.01,
        poly_order=1,
        n_lattice=9,
        max_strain=0.05,
        central_diff=True,
    ).compute()

    print(bsl.bond_table)
    print("shells (mean d, Å):", bsl.shells)
    bsl.plot()
    plt.show()
