"""Frame-exact video decode and encode.

Why this is its own module and not two lines of cv2
---------------------------------------------------
Auditing the v1 outputs turned up two concrete defects that live entirely in
I/O, not in the model:

* ``ipman.mp4``: 501 source frames in, **325** frames out -- 176 frames (35%)
  silently dropped.  Every other clip round-tripped, so this is a code path
  (almost certainly a shot boundary whose frames were never written), not a
  decoder issue.  It is invisible unless you count frames, which is why it
  survived.
* Frame rate is round-tripped through a float everywhere: 24000/1001 comes back
  as 1199/50, 30000/1001 as 1497/50, 60/1 as 5999/100.  Harmless in isolation
  but it desyncs audio and makes frame-accurate comparison against the source
  impossible.

So: decode with an explicit frame count, assert the count on the way out, and
carry the exact rational frame rate through untouched.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


class FrameCountMismatch(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #

@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: Fraction
    n_frames: int
    has_audio: bool

    @property
    def fps_float(self) -> float:
        return float(self.fps)


def probe(path: str, count_frames: bool = True) -> VideoInfo:
    """Probe a video.  ``count_frames`` actually decodes, which is slower but is
    the only number you can trust -- container ``nb_frames`` metadata was wrong
    on several of the sample clips."""
    cmd = [FFPROBE, "-v", "error", "-print_format", "json",
           "-show_streams", "-select_streams", "v:0"]
    if count_frames:
        cmd += ["-count_frames"]
    cmd += [path]
    st = json.loads(subprocess.check_output(cmd))["streams"][0]

    n = int(st.get("nb_read_frames") or st.get("nb_frames") or 0)
    fr = st.get("r_frame_rate", "25/1")
    num, _, den = fr.partition("/")
    fps = Fraction(int(num), int(den or 1))

    has_audio = bool(json.loads(subprocess.check_output(
        [FFPROBE, "-v", "error", "-print_format", "json",
         "-show_streams", "-select_streams", "a:0", path]
    )).get("streams"))

    return VideoInfo(path, int(st["width"]), int(st["height"]), fps, n, has_audio)


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #

def read_frames(path: str, max_size: Optional[int] = None) -> Iterator[np.ndarray]:
    """Yield BGR uint8 frames via an ffmpeg rawvideo pipe.

    An ffmpeg pipe rather than ``cv2.VideoCapture`` because cv2 will happily
    stop early on a stream with a damaged packet and give you no indication
    that it did.
    """
    info = probe(path, count_frames=False)
    w, h = info.width, info.height
    vf = []
    if max_size and min(w, h) > max_size:
        s = max_size / float(min(w, h))
        # Even dimensions: many encoders and most matting backbones require it.
        w, h = (int(round(w * s)) // 2) * 2, (int(round(h * s)) // 2) * 2
        vf = ["-vf", f"scale={w}:{h}:flags=lanczos"]

    cmd = [FFMPEG, "-v", "error", "-nostdin", "-i", path, *vf,
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    frame_bytes = w * h * 3
    # stderr to DEVNULL: a caller that stops early (``max_frames``, a ``break``)
    # closes the pipe, and ffmpeg then prints "Broken pipe" noise that has
    # nothing to do with the caller's problem.
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, bufsize=frame_bytes * 4)
    try:
        while True:
            buf = p.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        try:
            p.stdout.close()
        except Exception:                       # noqa: BLE001
            pass
        if p.poll() is None:
            p.kill()
        p.wait()


def read_all(path: str, max_size: Optional[int] = None) -> List[np.ndarray]:
    return list(read_frames(path, max_size))


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #

class VideoWriter:
    """ffmpeg rawvideo sink with an exact rational fps and a frame counter."""

    def __init__(self, path: str, width: int, height: int, fps: Fraction,
                 codec: str = "h264", crf: int = 16,
                 pix_fmt_out: str = "yuv420p", alpha: bool = False):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.path, self.n = path, 0
        self.width, self.height = width, height

        in_pix = "bgra" if alpha else "bgr24"
        if alpha:
            # VP9 in WebM is the only widely-supported alpha video format.
            enc = ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
                   "-crf", str(crf), "-b:v", "0", "-row-mt", "1"]
        elif codec == "prores":
            enc = ["-c:v", "prores_ks", "-profile:v", "4444",
                   "-pix_fmt", "yuva444p10le"]
        else:
            enc = ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                   "-pix_fmt", pix_fmt_out]

        cmd = [FFMPEG, "-v", "error", "-y", "-nostdin",
               "-f", "rawvideo", "-pix_fmt", in_pix,
               "-s", f"{width}x{height}",
               "-r", f"{fps.numerator}/{fps.denominator}", "-i", "-",
               *enc,
               # Keep the *exact* rational rate; do not let ffmpeg re-derive it
               # or drop/duplicate frames to hit a rounded rate.  This is what
               # turned 24000/1001 into 1199/50 in the v1 outputs.
               "-vsync", "passthrough",
               path]
        self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(
                f"frame {frame.shape[:2]} != writer {(self.height, self.width)}")
        self._p.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.n += 1

    def close(self, expect: Optional[int] = None, strict: bool = True) -> int:
        self._p.stdin.close()
        self._p.wait()
        if expect is not None and self.n != expect:
            msg = (f"{self.path}: wrote {self.n} frames, expected {expect} "
                   f"({expect - self.n} dropped)")
            if strict:
                raise FrameCountMismatch(msg)
            print("WARNING:", msg)
        return self.n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._p.stdin.close()
        self._p.wait()


def mux_audio(video_path: str, source_with_audio: str, out_path: str) -> str:
    """Copy the source audio track onto a rendered video.

    v1 dropped audio entirely.  For a batch tool that is usually wrong -- the
    clip still needs to cut against the original.
    """
    if not probe(source_with_audio, count_frames=False).has_audio:
        shutil.copyfile(video_path, out_path)
        return out_path
    subprocess.check_call(
        [FFMPEG, "-v", "error", "-y", "-nostdin",
         "-i", video_path, "-i", source_with_audio,
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-shortest", out_path])
    return out_path


def write_png_sequence(frames_rgba: List[np.ndarray], out_dir: str,
                       prefix: str = "frame") -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, f in enumerate(frames_rgba):
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_{i:06d}.png"), f)
