"""4.2 -- compositing over image and video backgrounds, and 4.4 -- audio.

Both were on the plan as things to verify "by eye" and neither had ever been
measured. The still-image path existed; `composite_over_video` had no code at
all until 29 Aug.

The two that matter are the exactness tests. A composite that is *almost* the
background where alpha is 0 is a composite with the subject's original plate
bleeding through, which is exactly what makes a keyed shot look stuck on.
"""
import os
import shutil
import subprocess
import tempfile
from fractions import Fraction

import numpy as np
import pytest

from vbgr import compose, video_io

H, W = 120, 200


def _alpha_square():
    a = np.zeros((H, W), np.float32)
    a[40:80, 60:140] = 1.0
    return a


def _frame(color):
    return np.full((H, W, 3), color, np.uint8)


def test_transparent_region_is_exactly_the_background():
    fg = _frame((10, 20, 30))
    bg = _frame((200, 100, 50))
    out = compose.composite_over_image(fg, _alpha_square(), bg)

    outside = _alpha_square() < 0.5
    assert np.array_equal(out[outside], bg[outside])


def test_opaque_region_is_exactly_the_foreground():
    fg = _frame((10, 20, 30))
    bg = _frame((200, 100, 50))
    out = compose.composite_over_image(fg, _alpha_square(), bg)

    inside = _alpha_square() > 0.5
    assert np.array_equal(out[inside], fg[inside])


def test_a_wrong_shaped_background_is_cropped_not_stretched():
    """Cover-resize must preserve aspect: a circle stays a circle."""
    import cv2
    bg = np.zeros((480, 1920, 3), np.uint8)
    cv2.circle(bg, (960, 240), 160, (0, 255, 255), -1)

    out = compose.composite_over_image(_frame((0, 0, 0)),
                                       np.zeros((H, W), np.float32), bg)

    m = out[..., 1] > 200
    ys, xs = np.nonzero(m)
    ratio = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
    assert abs(ratio - 1.0) < 0.05, f"circle became an ellipse ({ratio:.3f})"


def test_video_background_loops_rather_than_freezing():
    """A frozen background behind a moving subject reads as a bug."""
    frames = [_frame((0, 0, 0)) for _ in range(7)]
    alphas = [np.zeros((H, W), np.float32) for _ in range(7)]
    bgs = [_frame((i * 30, 0, 0)) for i in range(3)]

    out = compose.composite_over_video(frames, alphas, bgs)

    assert len(out) == 7
    assert np.array_equal(out[0], out[3])          # looped
    assert not np.array_equal(out[0], out[1])      # and it moved


def test_video_background_can_hold_instead_of_looping():
    frames = [_frame((0, 0, 0)) for _ in range(5)]
    alphas = [np.zeros((H, W), np.float32) for _ in range(5)]
    bgs = [_frame((i * 30, 0, 0)) for i in range(3)]

    out = compose.composite_over_video(frames, alphas, bgs, loop=False)

    assert np.array_equal(out[3], out[4]) and np.array_equal(out[4], out[2])


def test_video_background_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        compose.composite_over_video([_frame((0, 0, 0))] * 3,
                                     [np.zeros((H, W), np.float32)] * 2,
                                     [_frame((1, 1, 1))])


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="needs ffmpeg/ffprobe")
def test_mux_audio_leaves_the_video_untouched():
    """`-c:v copy`, so the matte must survive the mux bit for bit."""
    import cv2
    import json
    with tempfile.TemporaryDirectory() as d:
        silent = os.path.join(d, "v.mp4")
        w = video_io.VideoWriter(silent, W, H, Fraction(24000, 1001))
        for i in range(6):
            w.write(_frame((i * 20, 40, 60)))
        w.close(expect=6)

        # a source with a real audio track
        src = os.path.join(d, "a.mp4")
        subprocess.check_call(
            [video_io.FFMPEG, "-v", "error", "-y", "-nostdin",
             "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:d=1",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             "-c:a", "aac", "-shortest", src])

        out = os.path.join(d, "out.mp4")
        video_io.mux_audio(silent, src, out)

        st = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", out],
            capture_output=True).stdout.decode())["streams"]
        assert any(s["codec_type"] == "audio" for s in st), "audio was dropped"

        a, b = cv2.VideoCapture(silent), cv2.VideoCapture(out)
        while True:
            ok1, f1 = a.read()
            ok2, f2 = b.read()
            if not (ok1 and ok2):
                break
            assert np.array_equal(f1, f2), "the mux re-encoded the video"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_source_without_audio_still_produces_a_file():
    with tempfile.TemporaryDirectory() as d:
        silent = os.path.join(d, "v.mp4")
        w = video_io.VideoWriter(silent, W, H, Fraction(24, 1))
        for _ in range(4):
            w.write(_frame((5, 5, 5)))
        w.close(expect=4)

        out = os.path.join(d, "out.mp4")
        video_io.mux_audio(silent, silent, out)

        assert os.path.getsize(out) > 0
