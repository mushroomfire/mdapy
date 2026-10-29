# Trajectories

`mdapy.read_trajectory` opens the frames of a file without reading them. A
frame is read when it is used, so a long trajectory never has to fit in
memory, and iterating over it starts at once.

```python
import numpy as np
import mdapy as mp

rng = np.random.default_rng(0)
system = mp.System(rng.random((10, 3)) * 5, mp.Box(5.0), elements="Ar")
frames = (
    system.with_columns(positions=system["positions"] + 0.01 * k).with_info(step=k)
    for k in range(100)
)
mp.write("run.xyz", frames)

with mp.read_trajectory("run.xyz") as trajectory:
    first = trajectory[0]  # reads one frame
    every_tenth = trajectory[::10]  # a trajectory of 10 frames, nothing read
    steps = [frame.info["step"] for frame in every_tenth]
    count = len(trajectory)  # 100
    mp.write("every_tenth.mdapy", every_tenth)  # writes the selected frames
```

## Use

`mdapy.read_trajectory(path, format=None, *, topology=None, **options)`
takes the same arguments as [`mdapy.read`](io.md#reading), without
`frame`; the options of the format apply to every frame. `path` may also
name [several files](#several-files), and `topology` a system whose atoms
the frames [move](#a-topology-for-the-frames). It returns a `Trajectory`:

| Operation | Result |
|---|---|
| `trajectory[i]` | the frame `i`, read from the file; negative indices count from the end |
| `trajectory[a:b:c]` | a `Trajectory` of the selected frames, sharing the open file |
| `for frame in trajectory` | the frames in order, one at a time |
| `len(trajectory)` | the number of frames |
| `trajectory.path`, `trajectory.format` | the file, or the pattern or first file of several, and the format |
| `trajectory.close()`, `with ...` | releases the files |
| `trajectory.map(function)` | the frames through a function, [below](#frames-through-a-function) |

Any format works; a format that holds one frame gives a trajectory of one
frame. A trajectory is an iterable of systems, so `mdapy.write` writes it, or
a slice of it, to any format that holds several frames.

## Reading frames

A frame is read when it is used, and opening a trajectory reads nothing
else: iteration starts at once even for a file of many gigabytes. `len`,
negative indices and slices read through the whole file once, without
keeping its atoms.

A compressed file is decompressed into memory when the trajectory is opened;
an uncompressed one is not. The file is released by `close()`, at the end of
a `with` block, or once no trajectory of it is used any more.

If the file ends in the middle of a frame, the frames before it are read
normally and the incomplete frame raises
[`FileFormatError`](../reference/api.md#mdapy.errors.FileFormatError) when it
is reached.

## Frames through a function

`trajectory.map(function)` gives the frames through a function of a frame,
such as a pipeline of analyses that adds their columns, each computed when it
is used. It is a sequence as the trajectory is: indices, slices (mapped as
well), `len` and iteration, which reads the frames one after the other; its
`map` adds another function. The analyses of trajectories, such as
[`TimeAverage`](../analysis/time-average.md) or
[`TimeSeries`](../analysis/time-series.md), and
[`remove_drift`](../analysis/drift.md) or
[`smooth_trajectory`](../analysis/smooth-trajectory.md) take it as they take
the trajectory.

```python
from mdapy.analysis import TimeSeries


def with_height(frame):
    """The frame with the height of each atom above the middle of the box."""
    return frame.with_columns(height=frame["positions.z"] - 2.5)


with mp.read_trajectory("run.xyz") as trajectory:
    analyzed = trajectory.map(with_height)
    mean = TimeSeries("height", "mean").compute(analyzed)  # one value a frame
    last = analyzed[-1]  # the last frame, with its height when it is read
```

## Several files

A pattern with `*` stands for the files it matches, in the order of the
numbers in their names, so that `dump.10.lammpstrj` follows
`dump.9.lammpstrj`; a list gives the files in its order. The frames of each
file follow those of the file before, and files of one frame and of many
mix:

```python
for k in range(3):
    mp.write(f"part.{k * 50}.xyz", [system.with_info(step=k)] * 2)

parts = mp.read_trajectory("part.*.xyz")  # part.0, part.50, part.100: 6 frames
chosen = mp.read_trajectory(["part.100.xyz", "part.0.xyz"])  # 4 frames
```

The format is that of the first file. At most two files are open at a
time, so a pattern may match thousands of files.

## A topology for the frames

LAMMPS dumps hold positions, data files the bonds, angles, charges and
molecules. With `topology`, each frame takes the atoms of a system, such as
one read from a data file, with all of these:

```python
oxygens = rng.random((4, 3)) * 12 + 1
positions = np.stack(
    [oxygens, oxygens + [0.96, 0, 0], oxygens + [-0.24, 0.93, 0]], axis=1
).reshape(-1, 3)
water = mp.System(
    positions,
    mp.Box(15.0),
    elements=["O", "H", "H"] * 4,
    topology=mp.Topology(
        bonds=[[3 * k, 3 * k + i] for k in range(4) for i in (1, 2)],
        angles=[[3 * k + 1, 3 * k, 3 * k + 2] for k in range(4)],
    ),
).with_columns(
    charge=[-0.8476, 0.4238, 0.4238] * 4, molecule=np.repeat([1, 2, 3, 4], 3)
)
mp.write("water.data", water, atom_style="full")
data = mp.read("water.data")  # ids 1 to 12

# dumps of the positions only, the atoms in any order
for step in range(3):
    shuffled = data.gather(rng.permutation(12)).select("id", "type", "positions")
    mp.write(f"water.{step}.dump", shuffled.with_info(timestep=step))

for frame in mp.read_trajectory("water.*.dump", topology=data):
    frame.topology.bonds  # the eight bonds of the data file
    frame["charge"]  # its charges, in its order
```

Each frame, as OVITO's load trajectory makes it:

- has the atoms of the topology in their order, matched by id, with the
  columns of the frame and those of the topology the frame lacks;
- leaves out the atoms a frame lacks, with their bonds, angles, dihedrals
  and impropers, the others renumbered; frames with every atom share the
  `Topology` object of the system;
- has the box of the frame with the periodicity of the topology, the
  information of both, that of the frame first, and loses the image flags
  of the topology when the frame has none.

Without ids, a frame must have as many atoms as the topology, taken in
order, and the ids of the topology, if any, must be 1 to N in order.
Reading a frame raises
[`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter) when
it has atoms the topology lacks (OVITO adds them), when ids repeat, or when
without ids the numbers of atoms differ.
