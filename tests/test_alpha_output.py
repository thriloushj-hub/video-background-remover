"""4.1 -- the transparent output really carries alpha, and stays readable.

v1 shipped a `transparency.webm` that renders green rather than transparent.
The only proof either way is a round trip: encode with the real writer, decode
the file back, and measure the alpha that comes out.

The second test is the trap that made this look broken when it wasn't.
ffmpeg's *default* vp9 decoder silently discards VP9's alpha side data and
returns a fully opaque frame -- exit code 0, no warning. Decoding a good file
that way is indistinguishable from the v1 defect. Same for `pix_fmt`: VP9
alpha lives in per-block side data, so ffprobe reports the base plane and a
perfectly good transparent file reads as `yuv420p`. The Matroska `alpha_mode`
tag is the field that means anything.
"""
import json
import os
import shutil
import subprocess
import tempfile
from fractions import Fraction

import numpy as np
import pytest

from vbgr import video_io

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None
                                or shutil.which("ffprobe") is None,
                                reason="needs ffmpeg/ffprobe")

W = H = 64
N = 6


def _clip():
    """A hard-edged opaque square on a fully transparent field."""
    f = np.zeros((H, W, 4), np.uint8)
    f[16:48, 16:48] = (0, 0, 255, 255)          # BGRA: opaque red square
    return [f.copy() for _ in range(N)]


def _write(path):
    w = video_io.VideoWriter(path, W, H, Fraction(24000, 1001), alpha=True)
    for f in _clip():
        w.write(f)
    return w.close(expect=N)


def _decode(path, force_libvpx=True):
    cmd = [video_io.FFMPEG, "-v", "error", "-nostdin"]
    if force_libvpx:
        cmd += ["-c:v", "libvpx-vp9"]
    cmd += ["-i", path, "-f", "rawvideo", "-pix_fmt", "rgba", "-"]
    p = subprocess.run(cmd, capture_output=True)
    assert p.returncode == 0, p.stderr.decode()[:300]
    a = np.frombuffer(p.stdout, np.uint8)
    n = a.size // (W * H * 4)
    return a[:n * W * H * 4].reshape(n, H, W, 4)


def test_alpha_survives_the_round_trip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.webm")
        assert _write(p) == N

        dec = _decode(p)

        assert len(dec) == N, "frame count changed in the container"
        a = dec[..., 3]
        assert a.min() == 0, "nothing decoded transparent -- this is the v1 bug"
        assert a.max() == 255
        # the square is 32x32 of 64x64 = a quarter of the frame
        assert abs(float(a.mean()) / 255 - 0.25) < 0.01


def test_the_default_vp9_decoder_silently_drops_alpha():
    """Guards the trap, so nobody re-diagnoses a working file as broken."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.webm")
        _write(p)

        assert _decode(p, force_libvpx=False)[..., 3].min() == 255
        assert _decode(p, force_libvpx=True)[..., 3].min() == 0


def test_pix_fmt_does_not_tell_you_whether_there_is_alpha():
    """`yuv420p` on an alpha file is normal. Read `alpha_mode` instead."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.webm")
        _write(p)

        s = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_streams", "-of", "json", p],
            capture_output=True).stdout.decode())["streams"][0]
        tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}

        assert s["codec_name"] == "vp9"
        assert s["pix_fmt"] == "yuv420p"          # and the file IS transparent
        assert tags.get("alpha_mode") == "1"


def test_the_transparent_region_is_not_painted():
    """v1's failure mode: a green rectangle where transparency should be."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.webm")
        _write(p)

        dec = _decode(p)
        bg = dec[..., 3] < 8
        assert bg.any()
        key_rgb = np.array([119, 253, 151], np.float32)      # v1's key, as RGB
        d_green = np.linalg.norm(dec[..., :3][bg].astype(np.float32) - key_rgb,
                                 axis=-1).mean()
        assert d_green > 60, "the transparent region is carrying v1's green key"
