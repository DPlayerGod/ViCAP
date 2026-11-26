"""8 ảnh + audio cho một video, port 1-1 của script đã dựng dataset lúc train (PLAN mục 4.1).

Bản gốc nằm ở ``scripts/reference/``:

- ``extract_frame.py`` → dataset ``tiktok-frames-v1`` (``{video_id}_{k}.jpg``):
  - ``_0``: ảnh bìa; không có thì frame tại 0 s (video) / slide đầu (slideshow).
  - video: ``_k`` (k = 1…7) cắt bằng ``ffmpeg -ss {dur·k/8} -i video -frames:v 1`` (seek trước ``-i``),
    ``dur`` đọc từ header (``ffmpeg -i``), chặn ở ``dur − 0,1``; không ra frame thì thử lại ở
    0,9 / 0,75 / 0,5 / 0,25 / 0 lần mốc đó.
  - slideshow: 7 slide theo ``linspace(0, n−1, 7).round()``; ít hơn 7 slide thì ``arange(7)·n // 7``.
  - cả 8 ảnh: ``convert("RGB")`` → ``thumbnail((640, 640))`` → JPEG ``quality=90``.
- ``extract_audio.py`` → dataset ``tiktok-audio-v1``: ``ffmpeg -i video -vn -ar 16000 -ac 1 -b:a 64k x.mp3``;
  độ dài đo bằng ``mutagen``. Audio < 1 s vẫn giữ file nhưng ASR coi là ``empty_or_music``. Slideshow không có
  audio.

ffmpeg lấy từ ``imageio-ffmpeg`` như hai script gốc (bản build tĩnh, giống nhau trên Windows / Linux); không cài
thì dùng ``ffmpeg`` trên PATH.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

N_FRAMES = 7  # + 1 ảnh bìa = 8 ảnh
LONG_SIDE = 640
JPEG_QUALITY = 90
MIN_AUDIO_SEC = 1.0
_SEEK_RETRY = (1.0, 0.9, 0.75, 0.5, 0.25, 0.0)


class MediaError(RuntimeError):
    pass


@cache
def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise MediaError("không có ffmpeg (pip install imageio-ffmpeg)") from None
        return exe


def probe(path: str | Path) -> tuple[float, float]:
    """``(duration_s, fps)`` đọc từ stderr của ``ffmpeg -i`` (không cần ffprobe)."""
    err = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True).stderr
    text = err.decode("utf-8", "ignore")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", text)
    if not m:
        raise MediaError("không đọc được duration")
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    f = re.search(r"(\d+(?:\.\d+)?)\s*fps", text)
    return dur, float(f.group(1)) if f else 30.0


def grab(path: str | Path, t: float):
    """Một frame tại giây ``t``; header đôi khi dài hơn video thật nên không ra frame thì lùi dần ``t``."""
    from PIL import Image

    for scale in _SEEK_RETRY:
        out = subprocess.run(
            [ffmpeg_exe(), "-v", "error", "-ss", "%.3f" % (t * scale), "-i", str(path),
             "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True,
        ).stdout
        if out:
            return Image.open(io.BytesIO(out)).convert("RGB")
    raise MediaError(f"ffmpeg không trả frame tại {t:.2f}s")


def frame_times(duration: float) -> list[float]:
    last = max(duration - 0.1, 0)  # tránh seek vượt cuối file
    return [min(duration * k / (N_FRAMES + 1), last) for k in range(1, N_FRAMES + 1)]


def pick_slides(n: int) -> list[int]:
    import numpy as np

    if n <= 0:
        raise MediaError("slideshow không có ảnh")
    if n >= N_FRAMES:
        return np.linspace(0, n - 1, N_FRAMES).round().astype(int).tolist()
    return (np.arange(N_FRAMES) * n // N_FRAMES).tolist()  # < 7 slide: lặp đều từng slide


def to_jpeg(im) -> bytes:
    im = im.convert("RGB")
    im.thumbnail((LONG_SIDE, LONG_SIDE))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


@dataclass(frozen=True)
class Frames:
    images: tuple[bytes, ...]  # 8 JPEG, thứ tự _0 … _7
    n_source: int  # video: số frame (dur·fps); slideshow: số slide
    thumb_source: str  # thumbnail / first_frame


def video_frames(video: str | Path, thumbnail: str | Path | None = None) -> Frames:
    from PIL import Image

    dur, fps = probe(video)
    if dur <= 0:
        raise MediaError("video 0 giây")
    frames = [grab(video, t) for t in frame_times(dur)]
    first = Image.open(thumbnail) if thumbnail else grab(video, 0)
    images = tuple(to_jpeg(im) for im in (first, *frames))
    return Frames(images, int(round(dur * fps)), "thumbnail" if thumbnail else "first_frame")


def slideshow_frames(slides: Sequence[str | Path], thumbnail: str | Path | None = None) -> Frames:
    """``slides`` đã theo thứ tự (``img_001``, ``img_002``, …)."""
    from PIL import Image

    picks = [Image.open(slides[i]) for i in pick_slides(len(slides))]
    first = Image.open(thumbnail if thumbnail else slides[0])
    images = tuple(to_jpeg(im) for im in (first, *picks))
    return Frames(images, len(slides), "thumbnail" if thumbnail else "first_frame")


def extract_audio(video: str | Path, dst: str | Path) -> float:
    """Tách audio ra ``dst`` (.mp3), trả về độ dài (giây) theo ``mutagen``."""
    from mutagen.mp3 import MP3

    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
           "-vn", "-ar", "16000", "-ac", "1", "-b:a", "64k", str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise MediaError(result.stderr.strip()[-500:] or "ffmpeg lỗi không rõ nguyên nhân")
    return float(MP3(str(dst)).info.length)
