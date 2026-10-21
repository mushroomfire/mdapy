# Trajectories

`mdapy.read_trajectory` opens the frames of a file without reading them. A
frame is read when it is used, so a long trajectory never has to fit in
memory, and iterating over it starts at once.

```python
import numpy as np
import mdapy

rng = np.random.default_rng(0)
system = mdapy.System(rng.random((10, 3)) * 5, mdapy.Box(5.0), elements="Ar")
frames = (
    system.with_positions(system.positions + 0.01 * k).with_info(step=k)
    for k in range(100)
)
mdapy.write("run.xyz", frames)

with mdapy.read_trajectory("run.xyz") as trajectory:
    first = trajectory[0]  # reads one frame
    every_tenth = trajectory[::10]  # a trajectory of 10 frames, nothing read
    steps = [frame.info["step"] for frame in every_tenth]
    count = len(trajectory)  # 100
    mdapy.write("every_tenth.mdapy", every_tenth)  # writes the selected frames
```

## Use

`mdapy.read_trajectory(path, format=None, **options)` takes the same
arguments as [`mdapy.read`](io.md#reading), without `frame`; the options of
the format apply to every frame. It returns a `Trajectory`:

| Operation | Result |
|---|---|
| `trajectory[i]` | the frame `i`, read from the file; negative indices count from the end |
| `trajectory[a:b:c]` | a `Trajectory` of the selected frames, sharing the open file |
| `for frame in trajectory` | the frames in order, one at a time |
| `len(trajectory)` | the number of frames |
| `trajectory.path`, `trajectory.format` | the file and its format |
| `trajectory.close()`, `with ...` | releases the file |

Any format works; a format that holds one frame gives a trajectory of one
frame. A trajectory is an iterable of systems, so `mdapy.write` writes it, or
a slice of it, to any format that holds several frames.

## How frames are found

The first time a frame is needed, mdapy scans the file up to that frame and
keeps the position where each frame starts; it never reads the atoms of the
frames it skips. `len`, negative indices and slices scan the whole file once.
Iterating scans and reads frame by frame, so it starts at once even for a
file of many gigabytes. For mdapy files the table of frames is read once.

Uncompressed files are memory-mapped; compressed files are decompressed into
memory when the trajectory is opened. The file is released by `close()`, at
the end of a `with` block, or once no trajectory of it is used any more.

If the file ends in the middle of a frame, the frames before it are read
normally and the incomplete frame raises
[`FileFormatError`](../reference/api.md#mdapy.errors.FileFormatError) when it
is reached.

## Validation

The tests in `tests/io/test_trajectory.py` check, for extended XYZ, LAMMPS
dump, mdapy and compressed files, that lengths, positive and negative
indices, nested slices and iteration give the frames that were written; that
reading the first frame scans only that frame; that iteration stops with an
error at a truncated last frame after the complete frames; that options
reach every frame; that an mdapy file reads its table of frames once; and
that closed trajectories release the file.
