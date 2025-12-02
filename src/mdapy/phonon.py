# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
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
    path : str or list-like
        Path specification in reciprocal coordinates. If a string, it is split by
        whitespace and interpreted as a flat list of floats which is then
        reshaped into (1, npoints, 3). If a list is provided it must be
        convertible to shape (1, npoints, 3).
    labels : str or List[str]
        Labels corresponding to q-points on the path. If a single string it is
        split by whitespace.
    unitcell : System
        The primitive/unit cell wrapped in MDAPY `System`. Must have a
        `calc` attribute set to a `CalculatorMP` instance.
    symprec : float, optional
        Symmetry tolerance passed to Phonopy (default: 1e-5).
    repeat : list of int, optional
        Supercell repeat vector. If None, computed automatically based on box
        thickness to reach ~15 Å in each direction.
    displacement : float, optional
        Finite displacement distance for generating supercells (default: 0.01).
    cutoff : float, optional
        If set, zero force constants beyond this radius (in same units as cell).
    """

    def __init__(
        self,
        path: Union[str, List[float], List[List[float]]],
        labels: Union[str, List[str]],
        unitcell: System,
        symprec: float = 1e-5,
        repeat: Optional[List[int]] = None,
        displacement: float = 0.01,
        cutoff: Optional[float] = None,
    ) -> None:
        if isinstance(path, str):
            self.path = np.array(path.split(), float).reshape(1, -1, 3)
        else:
            assert len(path[0]) == 3
            self.path = np.array(path).reshape(1, -1, 3)
        if isinstance(labels, str):
            self.labels = labels.split()
        else:
            self.labels = labels
        assert len(self.labels) == self.path.shape[1], (
            "The length of path should be equal to labels."
        )
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
        system.calc.results = {}
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
            self.path, npoints=npoints
        )
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
        xticks = [distances[0][0]] + [i[-1] for i in distances]
        if fig is None and ax is None:
            from mdapy.plotset import set_figure

            fig, ax = set_figure()

        for d, f in zip(distances, frequencies):
            for band in f.T:
                ax.plot(d, band, c="grey")

        ax.set_xlim(xticks[0], xticks[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(self.labels)
        ax.set_ylabel("Frequency (THz)")

        return fig, ax


if __name__ == "__main__":
    from mdapy import build_crystal, FIRE, NEP
    import matplotlib.pyplot as plt

    Al = build_crystal("Al", "fcc", 4.05)
    Al.calc = NEP("tests/input_files/UNEP-v1.txt")
    fy = FIRE(Al, optimize_cell=True)
    fy.run(100, show_process=False)
    pho = Phonon(
        path="0.0 0.0 0.0 0.5 0.0 0.5 0.625 0.25 0.625 0.375 0.375 0.75 0.0 0.0 0.0 0.5 0.5 0.5",
        labels="$\\Gamma$ X U K $\\Gamma$ L",
        unitcell=Al,
        symprec=1e-3,
    )
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
