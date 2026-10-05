# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
except ImportError:
    raise ImportError(
        "One need install phonopy: https://phonopy.github.io/phonopy/install.html"
    )

from mdapy.system import System
from mdapy.calculator import CalculatorMP
from mdapy.data import atomic_numbers
import numpy as np
import polars as pl

from typing import Optional, List, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


class Phonon:
    """
    Wrapper around phonopy to compute phonon band structures, DOS, PDOS and thermal properties.

    Parameters
    ----------
    path : str or list-like, optional
        Band-path specification in reciprocal (fractional) coordinates. Several
        forms are accepted:

        * ``"auto"`` / ``"seekpath"`` (default) -- the standard high-symmetry
          path is generated automatically from ``unitcell`` with seekpath,
          including any path discontinuities (e.g. fcc ``U|K``). ``labels`` is
          then filled in automatically and may be left as ``None``.
        * a single whitespace-separated string of floats (or a flat list /
          ``(n, 3)`` array) -- a single *continuous* path, as before.
        * a list of such strings/arrays -- each item is a continuous sub-path;
          consecutive sub-paths are drawn with a *discontinuity* between them.
          This is how a broken path such as ``... U | K ...`` is specified by
          hand.
    labels : str or List[str], optional
        Labels of the high-symmetry q-points. A single string is split on
        whitespace. For a discontinuous manual path, pass a list with one
        string/list per sub-path. Ignored (and auto-generated) when
        ``path="auto"``.
    unitcell : System
        The primitive/unit cell wrapped in MDAPY `System`. Must have a
        `calc` attribute set to a `CalculatorMP` instance.
    symprec : float, optional
        Symmetry tolerance passed to Phonopy and seekpath (default: 1e-5).
    repeat : list of int, optional
        Supercell repeat vector. If None, computed automatically based on box
        thickness to reach ~15 Å in each direction.
    displacement : float, optional
        Finite displacement distance for generating supercells (default: 0.01).
    cutoff : float, optional
        If set, zero force constants beyond this radius (in same units as cell).
    with_time_reversal : bool, optional
        Passed to seekpath when ``path="auto"`` (default: True).

    Notes
    -----
    With ``path="auto"`` the q-points come from seekpath's standardized
    primitive cell. For the standardized conventional cells produced by
    :func:`mdapy.build_crystal` this matches Phonopy's auto primitive cell, so
    the labels line up with the computed bands.
    """

    def __init__(
        self,
        path: Union[str, List[float], List[List[float]], None] = "auto",
        labels: Union[str, List[str], None] = None,
        unitcell: System = None,
        symprec: float = 1e-5,
        repeat: Optional[List[int]] = None,
        displacement: float = 0.01,
        cutoff: Optional[float] = None,
        with_time_reversal: bool = True,
    ) -> None:
        assert unitcell is not None, "Must provide a unitcell."
        # Resolve the band path into a list of continuous sub-paths
        # (``band_paths``) plus a flat list of point labels (``self.labels``).
        if isinstance(path, str) and path.lower() in ("auto", "seekpath"):
            self.band_paths, self.labels = self._seekpath_band_path(
                unitcell, symprec, with_time_reversal
            )
        else:
            self.band_paths, self.labels = self._parse_manual_path(path, labels)
        assert sum(len(r) for r in self.band_paths) == len(self.labels), (
            "The number of labels should equal the number of path points."
        )
        # Filled by ``compute_band_structure``; one bool per segment, False at a
        # path discontinuity.
        self.connections: Optional[List[bool]] = None
        self.unitcell = unitcell
        assert isinstance(self.unitcell.calc, CalculatorMP), (
            "Must set calculator for unitcell."
        )
        if repeat is None:
            lengths = self.unitcell.box.get_thickness()
            self.repeat = np.ceil(15.0 / lengths).astype(int)
        else:
            self.repeat = repeat
        self.symprec = symprec
        self.displacement = float(displacement)
        self.cutoff = cutoff

        # Containers for results
        self.band_dict = None
        self.dos_dict = None
        self.pdos_dict = None
        self.thermal_dict = None

        # Instantiate Phonopy and generate displaced supercells
        self.phonon = Phonopy(
            unitcell=self._system2phononAtoms(self.unitcell),
            supercell_matrix=self.repeat,
            primitive_matrix="auto",
            symprec=self.symprec,
        )
        self.phonon.generate_displacements(distance=self.displacement)
        self.supercells: List[System] = [
            self._phononAtoms2system(i)
            for i in self.phonon.supercells_with_displacements
        ]
        # Build force constants immediately
        self.get_force_constants()

    # ------------------------------------------------------------------
    # Band-path construction
    # ------------------------------------------------------------------
    @staticmethod
    def _format_label(label: str) -> str:
        """Turn a seekpath label into a matplotlib-friendly string."""
        if label.upper() == "GAMMA":
            return r"$\Gamma$"
        if "_" in label:  # e.g. ``H_2`` -> ``$H_2$``
            return f"${label}$"
        return label

    def _seekpath_band_path(
        self, unitcell: System, symprec: float, with_time_reversal: bool
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Generate the standard high-symmetry band path with seekpath.

        Returns
        -------
        band_paths : list of (n_i, 3) ndarray
            One array of high-symmetry q-points per *continuous* sub-path; a
            break between consecutive arrays marks a path discontinuity.
        labels : list of str
            Flat list of formatted labels, one per q-point across all
            sub-paths.
        """
        try:
            import seekpath
        except ImportError:
            raise ImportError("path='auto' needs seekpath: pip install seekpath")
        cell = np.asarray(unitcell.box.box, float)
        frac = unitcell.get_positions().to_numpy() @ np.linalg.inv(cell)
        numbers = [atomic_numbers[e] for e in unitcell.data["element"].to_numpy()]
        res = seekpath.get_path(
            (cell.tolist(), frac.tolist(), numbers),
            with_time_reversal=with_time_reversal,
            symprec=symprec,
        )
        pc, path = res["point_coords"], res["path"]

        band_paths, labels = [], []
        run_pts, run_lab = [np.array(pc[path[0][0]], float)], [path[0][0]]
        for i, (a, b) in enumerate(path):
            if i > 0 and path[i - 1][1] != a:  # discontinuity: close current run
                band_paths.append(np.array(run_pts, float))
                labels.extend(run_lab)
                run_pts, run_lab = [np.array(pc[a], float)], [a]
            run_pts.append(np.array(pc[b], float))
            run_lab.append(b)
        band_paths.append(np.array(run_pts, float))
        labels.extend(run_lab)
        return band_paths, [self._format_label(l) for l in labels]

    @staticmethod
    def _parse_manual_path(
        path: Union[str, List], labels: Union[str, List[str], None]
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Parse a user-supplied (possibly discontinuous) band path."""
        assert labels is not None, "labels are required for a manual path."
        # Decide whether ``path`` holds several sub-paths (discontinuous) or a
        # single continuous one. Multi-run if the first item is a string or a
        # 2D block of q-points; single run if it is a scalar or a 3-vector.
        multirun = False
        if not isinstance(path, str) and len(path):
            first = path[0]
            multirun = isinstance(first, str) or np.asarray(first).ndim >= 2
        if multirun:
            runs, run_labels = path, labels
        else:  # single continuous path
            runs, run_labels = [path], [labels]

        band_paths, flat_labels = [], []
        for p, lab in zip(runs, run_labels):
            if isinstance(p, str):
                pts = np.array(p.split(), float).reshape(-1, 3)
            else:
                pts = np.array(p, float).reshape(-1, 3)
            lab = lab.split() if isinstance(lab, str) else list(lab)
            assert len(lab) == len(pts), "Each sub-path needs one label per q-point."
            band_paths.append(pts)
            flat_labels.extend(lab)
        return band_paths, flat_labels

    def _system2phononAtoms(self, system: System) -> PhonopyAtoms:
        """
        Convert an mdapy.System into a PhonopyAtoms object.

        Parameters
        ----------
        system : System
            MDAPY System containing `data["element"]` and positions from
            `system.get_positions()` and `system.box.box` for the cell.

        Returns
        -------
        PhonopyAtoms
            PhonopyAtoms instance representing the unit cell.
        """
        return PhonopyAtoms(
            symbols=system.data["element"].to_numpy(),
            cell=system.box.box,
            positions=system.get_positions().to_numpy(),
        )

    def _phononAtoms2system(self, atoms: PhonopyAtoms) -> System:
        """
        Convert a PhonopyAtoms supercell with displacements into an mdapy.System.

        Parameters
        ----------
        atoms : PhonopyAtoms
            PhonopyAtoms instance for the supercell with atomic displacements.

        Returns
        -------
        System
            An mdapy.System object containing the supercell atomic data, box,
            and a calculator copied from the original unitcell (with cleared results).
        """
        data = pl.DataFrame(
            {
                "element": atoms.symbols,
                "x": atoms.positions[:, 0],
                "y": atoms.positions[:, 1],
                "z": atoms.positions[:, 2],
            },
            schema={
                "element": pl.Utf8,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
            },
        )
        box = atoms.cell
        system = System(data=data, box=box)
        system.calc = self.unitcell.calc
        return system

    def get_force_constants(self) -> None:
        """
        Compute force constants from finite-displacement supercells.

        The method gathers forces from each displaced supercell (using the
        System.get_force() method), removes the average rigid-body component
        per supercell, converts the set into a NumPy array and passes it to
        Phonopy's `produce_force_constants`. If `self.cutoff` is set, it will
        zero the force constants beyond that radius.
        """
        set_of_forces = []
        for i in self.supercells:
            forces = i.get_force()
            forces -= np.mean(forces, axis=0)
            set_of_forces.append(forces)
        set_of_forces = np.array(set_of_forces)
        self.phonon.produce_force_constants(forces=set_of_forces)
        if self.cutoff is not None:
            self.phonon.set_force_constants_zero_with_radius(float(self.cutoff))

    def compute_band_structure(self, npoints: int = 101) -> None:
        """
        Compute the phonon band structure along the provided path.

        Parameters
        ----------
        npoints : int, optional
            Number of q-points sampled along each path segment (default 101).

        Notes
        -----
        Results are stored in `self.band_dict` using Phonopy's
        `get_band_structure_dict()`.
        """
        qpoints, connections = get_band_qpoints_and_path_connections(
            self.band_paths, npoints=npoints
        )
        self.connections = connections
        self.phonon.run_band_structure(
            qpoints, path_connections=connections, labels=self.labels
        )
        self.band_dict = self.phonon.get_band_structure_dict()

    def compute_dos(self, mesh: Tuple[int] = (10, 10, 10)) -> None:
        """
        Compute total density of states (DOS).

        Parameters
        ----------
        mesh : tuple of int, optional
            q-point mesh for DOS calculation (default (10,10,10)).

        Notes
        -----
        Uses tetrahedron method for DOS.
        """
        self.phonon.run_mesh(mesh)
        self.phonon.run_total_dos(use_tetrahedron_method=True)
        self.dos_dict = self.phonon.get_total_dos_dict()

    def compute_pdos(self, mesh: Tuple[int] = (10, 10, 10)) -> None:
        """
        Compute projected (partial) density of states (PDOS).

        Parameters
        ----------
        mesh : tuple of int, optional
            q-point mesh; with eigenvectors enabled (default (10,10,10)).

        Notes
        -----
        Stores results in `self.pdos_dict`.
        """
        self.phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
        self.phonon.run_projected_dos()
        self.pdos_dict = self.phonon.get_projected_dos_dict()

    def compute_thermal(
        self, t_min: float, t_step: float, t_max: float, mesh: Tuple[int] = (10, 10, 10)
    ) -> None:
        """
        Compute thermal properties (free energy, entropy, heat capacity).

        Parameters
        ----------
        t_min : float
            Minimum temperature (K).
        t_step : float
            Temperature step (K).
        t_max : float
            Maximum temperature (K).
        mesh : tuple of int, optional
            q-point mesh for thermal property calculation (default (10,10,10)).

        Notes
        -----
        Results stored in `self.thermal_dict`.
        """
        self.phonon.run_mesh(mesh)
        self.phonon.run_thermal_properties(t_min=t_min, t_step=t_step, t_max=t_max)
        self.thermal_dict = self.phonon.get_thermal_properties_dict()

    def plot_dos(
        self,
        fig: Optional["Figure"] = None,
        ax: Optional["Axes"] = None,
    ) -> Tuple["Figure", "Axes"]:
        """
        Plot the total density of states (DOS).

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure to draw on. If None, a new figure and axes are created via
            `mdapy.plotset.set_figure()`.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None and fig is None, a new axes is created.

        Returns
        -------
        fig, ax : tuple
            The matplotlib Figure and Axes used for the plot.

        Raises
        ------
        RuntimeError
            If `compute_dos` has not been called before plotting.
        """
        if fig is None and ax is None:
            from mdapy.plotset import set_figure

            fig, ax = set_figure()
        if self.dos_dict is None:
            raise "call compute_dos before plot_dos."
        x, y = self.dos_dict["frequency_points"], self.dos_dict["total_dos"]
        ax.plot(x, y)
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("Density of states")
        ax.set_ylim(y.min(), y.max() * 1.1)
        return fig, ax

    def plot_pdos(
        self,
        fig: Optional["Figure"] = None,
        ax: Optional["Axes"] = None,
    ) -> Tuple["Figure", "Axes"]:
        """
        Plot projected (partial) density of states (PDOS).

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure to draw on. If None, creates a new figure via `set_figure`.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None and fig is None, a new axes is created.

        Returns
        -------
        fig, ax : tuple
            The matplotlib Figure and Axes used for the plot.

        Raises
        ------
        RuntimeError
            If `compute_pdos` has not been called before plotting.
        """
        if fig is None and ax is None:
            from mdapy.plotset import set_figure

            fig, ax = set_figure()
        if self.pdos_dict is None:
            raise "call compute_pdos before plot_pdos."
        x, y1 = self.pdos_dict["frequency_points"], self.pdos_dict["projected_dos"]
        for i, y in enumerate(y1, start=1):
            ax.plot(x, y, label=f"[{i}]")

        ax.legend()
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("Partial density of states")
        ax.set_ylim(y1.min(), y1.max() * 1.1)
        return fig, ax

    def plot_thermal(
        self,
        fig: Optional["Figure"] = None,
        ax: Optional["Axes"] = None,
    ) -> Tuple["Figure", "Axes"]:
        """
        Plot thermal properties computed by `compute_thermal`.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure to draw on. If None, a new figure and axes are created.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None and fig is None, a new axes is created.

        Returns
        -------
        fig, ax : tuple
            The matplotlib Figure and Axes used for the plot.

        Raises
        ------
        RuntimeError
            If `compute_thermal` has not been called before plotting.
        """
        if fig is None and ax is None:
            from mdapy.plotset import set_figure

            fig, ax = set_figure()
        if self.thermal_dict is None:
            raise "call compute_thermal before plot_thermal."
        temperatures = self.thermal_dict["temperatures"]
        free_energy = self.thermal_dict["free_energy"]
        entropy = self.thermal_dict["entropy"]
        heat_capacity = self.thermal_dict["heat_capacity"]

        ax.plot(temperatures, free_energy, label="Free energy (kJ/mol)")
        ax.plot(temperatures, entropy, label="Entropy (J/K/mol)")
        ax.plot(temperatures, heat_capacity, label="$C_v$ (J/K/mol)")
        ax.legend()
        ax.set_xlabel("Temperature (K)")
        ax.set_xlim(temperatures[0], temperatures[-1])
        return fig, ax

    def plot_band_structure(
        self,
        fig: Optional["Figure"] = None,
        ax: Optional[Union["Axes", List["Axes"]]] = None,
    ) -> Tuple["Figure", Optional[Union["Axes", List["Axes"]]]]:
        """
        Plot computed phonon band structure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure to draw on. If None, a new figure and axes are created.
        ax : matplotlib.axes.Axes or list of Axes, optional
            Axes to draw on. If None and fig is None, a new axes is created.

        Returns
        -------
        fig, ax : tuple
            The matplotlib Figure and Axes (or list of Axes) used for the plot.

        Raises
        ------
        RuntimeError
            If `compute_band_structure` has not been called prior to plotting.
        """
        if self.band_dict is None:
            raise "call compute_ban_structure before plot_band_structure."

        frequencies = self.band_dict["frequencies"]
        distances = self.band_dict["distances"]
        xticks, xlabels = self._xticks_and_labels()
        if fig is None and ax is None:
            from mdapy.plotset import set_figure

            fig, ax = set_figure()

        for d, f in zip(distances, frequencies):
            for band in f.T:
                ax.plot(d, band, c="grey")

        # vertical guide lines at the high-symmetry q-points
        for xt in xticks:
            ax.axvline(xt, c="grey", lw=0.5)
        ax.set_xlim(xticks[0], xticks[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_ylabel("Frequency (THz)")

        return fig, ax

    def _xticks_and_labels(self) -> Tuple[List[float], List[str]]:
        """Tick positions and labels, merging ``X|Y`` at path discontinuities."""
        distances = self.band_dict["distances"]
        connections = self.connections
        labels = self.labels
        if connections is None:  # single continuous path fallback
            connections = [True] * (len(distances) - 1) + [False]

        positions = [distances[0][0]]
        out_labels = [labels[0]]
        li = 1
        for i, connected in enumerate(connections):
            positions.append(distances[i][-1])
            if connected:  # interior high-symmetry point
                out_labels.append(labels[li])
                li += 1
            elif i < len(connections) - 1:  # discontinuity: merge both labels
                out_labels.append(f"{labels[li]}|{labels[li + 1]}")
                li += 2
            else:  # final point
                out_labels.append(labels[li])
                li += 1
        return positions, out_labels


if __name__ == "__main__":
    from mdapy import build_crystal, FIRE, NEP
    import matplotlib.pyplot as plt

    Al = build_crystal("Al", "fcc", 4.05)
    Al.calc = NEP("tests/input_files/UNEP-v1.txt")
    fy = FIRE(Al, optimize_cell=True)
    fy.run(100, show_process=False)
    # path="auto" builds the standard seekpath high-symmetry path
    # (with the fcc U|K discontinuity) automatically.
    pho = Phonon(path="auto", unitcell=Al, symprec=1e-3)
    pho.compute_band_structure()
    pho.plot_band_structure()
    plt.show()
    pho.compute_dos()
    pho.plot_dos()
    plt.show()
    pho.compute_pdos()
    pho.plot_pdos()
    plt.show()
    pho.compute_thermal(300, 100, 1000)
    pho.plot_thermal()
    plt.show()
