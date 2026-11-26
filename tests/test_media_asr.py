from __future__ import annotations

import io
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pytest

pytest.importorskip("PIL")
pytest.importorskip("mutagen")

from PIL import Image  # noqa: E402
from pyiceberg.io.pyarrow import schema_to_pyarrow  # noqa: E402

from vicap.features import asr as asr_mod  # noqa: E402
from vicap.features.media import (  # noqa: E402
    MediaError,
    extract_audio,
    ffmpeg_exe,
    frame_times,
    pick_slides,
    slideshow_frames,
    to_jpeg,
    video_frames,
)
from vicap.iceberg.reader import read_table  # noqa: E402
from vicap.iceberg.tables import SILVER_MEDIA, SILVER_TRANSCRIPTS, SILVER_VIDEOS  # noqa: E402
from vicap.iceberg.writer import ensure_table, write_table  # noqa: E402
from vicap.pipelines.asr import run_asr  # noqa: E402
from vicap.pipelines.media import build_media  # noqa: E402

T0 = datetime(2026, 11, 25, 15, tzinfo=UTC)


def test_frame_times_and_slide_picks() -> None:
    assert frame_times(8.0) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert frame_times(0.4)[-1] == pytest.approx(0.3)  # chặn ở dur − 0,1
    assert pick_slides(12) == [0, 2, 4, 6, 7, 9, 11]
    assert pick_slides(7) == list(range(7))
    assert pick_slides(3) == [0, 0, 0, 1, 1, 2, 2]  # < 7 slide: lặp đều
    with pytest.raises(MediaError):
        pick_slides(0)


def test_to_jpeg_rgb_640() -> None:
    im = Image.new("RGBA", (1000, 2000), (255, 0, 0, 128))
    out = Image.open(io.BytesIO(to_jpeg(im)))
    assert (out.format, out.mode, out.size) == ("JPEG", "RGB", (320, 640))
    small = Image.open(io.BytesIO(to_jpeg(Image.new("RGB", (300, 200)))))
    assert small.size == (300, 200)  # chỉ thu nhỏ


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    """mp4 3 giây 1280×720 có tiếng, sinh bằng ffmpeg."""
    path = tmp_path / "clip.mp4"
    subprocess.run([ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=3:size=1280x720:rate=25",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-shortest",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def _size(b: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(b)).size


def test_video_frames_and_audio(clip: Path, tmp_path: Path) -> None:
    fr = video_frames(clip)
    assert len(fr.images) == 8 and fr.thumb_source == "first_frame" and fr.n_source == 75
    assert all(_size(b) == (640, 360) for b in fr.images)
    cover = tmp_path / "cover.png"
    Image.new("RGB", (720, 960), "blue").save(cover)
    fr = video_frames(clip, cover)
    assert fr.thumb_source == "thumbnail" and _size(fr.images[0]) == (480, 640)
    assert extract_audio(clip, tmp_path / "a.mp3") == pytest.approx(3.0, abs=0.1)


def test_slideshow_frames(tmp_path: Path) -> None:
    slides = []
    for i in range(3):
        p = tmp_path / f"img_{i:03d}.jpg"
        Image.new("RGB", (1080, 1440), (i * 80, 0, 0)).save(p)
        slides.append(p)
    fr = slideshow_frames(slides)
    assert len(fr.images) == 8 and fr.n_source == 3 and fr.thumb_source == "first_frame"


def test_build_media_writes_frames_and_audio(clip: Path) -> None:
    raw = {"videos/1/video.mp4": clip.read_bytes()}  # không có ảnh bìa → frame đầu
    written: dict[str, bytes] = {}

    def read_raw(key: str) -> bytes:
        if key not in raw:
            raise FileNotFoundError(key)
        return raw[key]

    row = {"video_id": "1", "media_type": "video", "media_key": "videos/1/video.mp4",
           "thumbnail_key": "videos/1/cover.jpg", "image_keys": [], "batch_id": "b1"}
    out = build_media(row, read_raw=read_raw, write=lambda k, d, _: written.__setitem__(k, d), created_at=T0)
    assert sorted(written) == ["derived/audio/1.mp3", *(f"derived/frames/1_{k}.jpg" for k in range(8))]
    assert out["has_audio"] and out["thumb_source"] == "first_frame" and out["n_source_images"] == 75


# ── ASR ──
def test_status_and_batch_fallback() -> None:
    assert asr_mod.status_of("xin chào", None) == "ok"
    assert asr_mod.status_of("  ", None) == "empty_or_music"
    assert asr_mod.status_of("", "boom") == "error"

    def fake(x, generate_kwargs):
        assert generate_kwargs == asr_mod.GENERATE_KWARGS
        if isinstance(x, list):
            if "bad" in x:
                raise RuntimeError("batch")
            return [{"text": f" {p} "} for p in x]
        if x == "bad":
            raise RuntimeError("hỏng")
        return {"text": x}

    assert asr_mod.transcribe(fake, ["a", "b"]) == [("a", None), ("b", None)]
    assert asr_mod.transcribe(fake, ["a", "bad"]) == [("a", None), ("", "hỏng")]


class FakeFS:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects

    def cat_file(self, path: str) -> bytes:
        return self.objects[path]


def _rows(table, rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=schema_to_pyarrow(table.schema()))


def test_run_asr_statuses_and_skip(sql_catalog, s3_env) -> None:
    silver = ensure_table(sql_catalog, "silver.videos", SILVER_VIDEOS.schema)
    media = ensure_table(sql_catalog, "silver.media", SILVER_MEDIA.schema)
    base = {"username": "u", "media_type": "video", "crawl_ts": T0, "batch_id": "b1", "updated_at": T0}
    write_table(silver, _rows(silver, [{**base, "video_id": v} for v in ("1", "2", "3", "4", "5")]))
    m = {"media_type": "video", "n_source_images": 10, "thumb_source": "thumbnail", "batch_id": "b1",
         "created_at": T0}
    write_table(media, _rows(media, [
        {**m, "video_id": "1", "has_audio": True, "audio_duration": 5.0},
        {**m, "video_id": "2", "has_audio": True, "audio_duration": 0.5},  # < 1 s
        {**m, "video_id": "3", "media_type": "slideshow", "has_audio": False, "audio_duration": None},
        {**m, "video_id": "4", "has_audio": True, "audio_duration": 9.0},
    ]))  # video 5: chưa có media
    fs = FakeFS({"lakehouse/derived/audio/1.mp3": b"one", "lakehouse/derived/audio/4.mp3": b"four"})
    calls: list[list[str]] = []

    def fake(x, generate_kwargs):
        calls.append(x)
        return [{"text": "" if Path(p).read_bytes() == b"four" else " xin chào "} for p in x]

    s = run_asr(catalog=sql_catalog, fs=fs, asr=fake, now=lambda: T0)
    assert (s.counts, s.no_media) == ({"ok": 1, "empty_or_music": 2, "missing": 1}, 1)
    assert len(calls) == 1 and len(calls[0]) == 2  # chỉ chạy model cho video 1 và 4
    out = {r["video_id"]: r for r in read_table(sql_catalog.load_table(SILVER_TRANSCRIPTS.identifier)).to_pylist()}
    assert {v: (r["transcript_raw"], r["transcript_status"], r["has_transcript"]) for v, r in out.items()} == {
        "1": ("xin chào", "ok", True), "2": ("", "empty_or_music", False),
        "3": ("", "missing", False), "4": ("", "empty_or_music", False),
    }
    s = run_asr(catalog=sql_catalog, fs=fs, asr=fake, now=lambda: T0)  # chạy lại: bỏ qua hết
    assert (s.skipped, s.counts, len(calls)) == (4, {}, 1)
