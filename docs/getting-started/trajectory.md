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
