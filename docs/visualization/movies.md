# Movies

`mdapy.visualization.write_movie` writes pictures as a movie with
[FFmpeg](https://ffmpeg.org), which must be on the `PATH`: `conda install ffmpeg`,
`brew install ffmpeg` or `apt install ffmpeg`. MDAPY does not ship it.

<!-- skip-test -->
```python
import mdapy as mp
from mdapy.visualization import render, write_movie

frames = mp.read_trajectory("run.dump")
camera = dict(view="iso", focus=(50.0, 50.0, 50.0), span=150.0)
pictures = (
    render(frame, width=1280, height=720, device="auto", **camera) for frame in frames
)
write_movie("run.mp4", pictures, fps=30)
```

Give the pictures as a generator, as here: each is drawn when the movie needs it, so
a long movie is not held in memory. Fix the frame with `focus` and `span`, or each
picture frames its own atoms and the movie jumps.

## Parameters

`write_movie(path, pictures, *, fps=30)`

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `path` | str or path-like | | | the file; its suffix, such as `.mp4`, tells FFmpeg the format |
| `pictures` | iterable of arrays | | | (H, W, 3) or (H, W, 4) uint8 images of one size, as [`render`](rendering.md) gives them |
| `fps` | float | 30 | 1/s | pictures per second |

The movie is H.264 where FFmpeg offers it, the usual choice for `.mp4`, with
a picture of odd size padded to even sizes for players. Draw the pictures with a
`background`: a movie has no transparency.

## Errors

`write_movie` raises:

- `FileNotFoundError` when FFmpeg is not on the `PATH`;
- [`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter) when `fps` is
  not positive, there are no pictures, or a picture is not an image of the size of
  the first;
- [`CalculationFailed`](../reference/api.md#mdapy.errors.CalculationFailed) when FFmpeg
  fails, with what it says.

The tutorial [A movie that flies through a crystal](../getting-started/tutorials/movie.md)
makes a movie from one system.

## In the application

The Render tab of the MDAPY application writes the frames of a trajectory as a movie,
with the view as you set it. On macOS and Windows it encodes with the system, with
nothing to install; on Linux it needs FFmpeg as above. A camera that moves, or colors
that change on the way, are the work of a script like the one in the tutorial.
