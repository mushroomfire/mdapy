# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Movies from pictures, with the FFmpeg of the user."""

import contextlib
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from os import PathLike, fspath
from pathlib import Path

import numpy as np

from mdapy.errors import CalculationFailed, InvalidParameter


def _picture(picture: np.ndarray) -> np.ndarray:
    image = np.ascontiguousarray(picture)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise InvalidParameter(
            "a picture must be (H, W, 3) or (H, W, 4) uint8, "
            f"not {image.dtype} of shape {image.shape}"
        )
    return image


def _command(ffmpeg: str, path: str, shape: tuple, fps: float) -> list[str]:
    height, width, channels = shape
    return [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24" if channels == 3 else "rgba",
        "-s",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        # movies for players have even sizes
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        path,
    ]


def write_movie(
    path: str | PathLike, pictures: Iterable[np.ndarray], *, fps: float = 30
) -> None:
    """Writes pictures as a movie with FFmpeg, which must be on the PATH.

    Parameters
    ----------
    path : str or path-like
        The file; its suffix, such as ``.mp4``, tells FFmpeg the format.
    pictures : iterable of numpy.ndarray
        (H, W, 3) or (H, W, 4) uint8 images of one size, rows from the top.
        They are taken one at a time, so a generator of pictures writes a long
        movie without holding them in memory.
    fps : float, optional
        Pictures per second. Default 30.

    Raises
    ------
    FileNotFoundError
        If FFmpeg is not on the PATH.
    InvalidParameter
        If ``fps`` is not positive, there are no pictures, or a picture is not
        an image of the size of the first.
    CalculationFailed
        If FFmpeg fails, with what it says.
    """
    if not fps > 0:
        raise InvalidParameter(f"fps must be positive, not {fps!r}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError(
            "movies need FFmpeg on the PATH: install it, for example with "
            "conda install ffmpeg, brew install ffmpeg or apt install ffmpeg"
        )
    process, shape = None, None
    with tempfile.TemporaryFile() as said:
        try:
            for picture in pictures:
                image = _picture(picture)
                if process is None:
                    shape = image.shape
                    process = subprocess.Popen(
                        _command(ffmpeg, fspath(path), shape, fps),
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=said,
                    )
                elif image.shape != shape:
                    raise InvalidParameter(
                        f"a picture of shape {image.shape} after those of {shape}"
                    )
                try:
                    process.stdin.write(image.data)
                except OSError:
                    break  # FFmpeg stopped, and says why below
        except BaseException:
            if process is not None:
                process.kill()
                process.wait()
                with contextlib.suppress(OSError):  # what it did not take
                    process.stdin.close()
                Path(path).unlink(missing_ok=True)
            raise
        if process is None:
            raise InvalidParameter("there are no pictures to write")
        with contextlib.suppress(OSError):
            process.stdin.close()
        if process.wait() != 0:
            said.seek(0)
            text = said.read().decode("utf-8", "replace").strip()
            raise CalculationFailed(f"FFmpeg could not write the movie: {text}")
