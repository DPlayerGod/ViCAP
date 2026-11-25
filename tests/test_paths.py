from __future__ import annotations

from datetime import UTC, datetime

import pytest

from vicap.storage import paths as nm

# Ba ví dụ ở PLAN mục 4.2
EXAMPLES = [
    ("02_data/raw/videos/1vongthread__20260709_094500__7660442169215487252.mp4",
     "1vongthread", datetime(2026, 7, 9, 9, 45, 0, tzinfo=UTC), "7660442169215487252"),
    ("02_data/raw/videos/_vetthuong_09__20261008_153532__7694321051815185682/",
     "_vetthuong_09", datetime(2026, 10, 8, 15, 35, 32, tzinfo=UTC), "7694321051815185682"),
    ("_vetthuong_09__20261008_153532__7694321051815185682__img_001.jpg",
     "_vetthuong_09", datetime(2026, 10, 8, 15, 35, 32, tzinfo=UTC), "7694321051815185682"),
    ("02_data/raw/thumbnails/1vongthread__20260709_094500__7660442169215487252.jpg",
     "1vongthread", datetime(2026, 7, 9, 9, 45, 0, tzinfo=UTC), "7660442169215487252"),
    # username chứa "__"
    ("a__b__20260101_000000__123.mp4", "a__b", datetime(2026, 1, 1, tzinfo=UTC), "123"),
]


@pytest.mark.parametrize(("name", "username", "posted_at", "video_id"), EXAMPLES)
def test_parse_and_roundtrip(name: str, username: str, posted_at: datetime, video_id: str) -> None:
    parsed = nm.parse_legacy_stem(name)
    assert parsed == nm.LegacyStem(username, posted_at, video_id)
    rebuilt = nm.legacy_stem(parsed.username, parsed.posted_at, parsed.video_id)
    assert rebuilt == nm.parse_legacy_stem(rebuilt).__str__()
    assert rebuilt in name


def test_windows_path() -> None:
    assert nm.video_id_from_legacy(r"C:\data\1vongthread__20260709_094500__7660442169215487252.mp4") == \
        "7660442169215487252"


def test_legacy_stem_drops_dot_like_v5() -> None:
    # Dataset v5: harry.nista → harrynista__20260131_060926__7601403892349275399
    posted = datetime(2026, 1, 31, 6, 9, 26, tzinfo=UTC)
    assert nm.legacy_stem("harry.nista", posted, "7601403892349275399") == \
        "harrynista__20260131_060926__7601403892349275399"


def test_legacy_stem_from_epoch_is_utc() -> None:
    epoch = int(datetime(2026, 7, 9, 9, 45, tzinfo=UTC).timestamp())
    assert nm.legacy_stem("1vongthread", epoch, "7660442169215487252") == \
        "1vongthread__20260709_094500__7660442169215487252"


def test_legacy_stem_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone"):
        nm.legacy_stem("u", datetime(2026, 1, 1), "1")


BAD_NAMES = ["video.mp4", "u__2026_01__1.mp4", "u__20260101_000000__abc.mp4", "__20260101_000000__1"]


@pytest.mark.parametrize("bad", BAD_NAMES)
def test_parse_rejects_bad_names(bad: str) -> None:
    with pytest.raises(ValueError):
        nm.parse_legacy_stem(bad)


def test_slide_image_name_matches_v5() -> None:
    stem = "_vetthuong_09__20261008_153532__7694321051815185682"
    assert nm.legacy_slide_image_name(stem, 1) == f"{stem}__img_001.jpg"


def test_keys_use_video_id_only() -> None:
    vid = "6917211883481074946"
    assert nm.frame_name(vid, 0) == f"{vid}_0.jpg"
    assert nm.audio_name(vid) == f"{vid}.mp3"
    assert nm.raw_video(vid, nm.VIDEO_FILE) == f"videos/{vid}/video.mp4"
    assert nm.raw_video(vid, nm.INFO_FILE) == f"videos/{vid}/info.json"
    assert nm.raw_video(vid, nm.cover_name("jpeg")) == f"videos/{vid}/cover.jpg"
    assert nm.raw_video(vid, nm.slide_name(2, "png")) == f"videos/{vid}/img_002.png"
    assert nm.raw_prefix(vid) == f"videos/{vid}/"
    assert nm.frame(vid, 7) == f"derived/frames/{vid}_7.jpg"
    assert nm.audio(vid) == f"derived/audio/{vid}.mp3"
    assert nm.s3_uri("raw", nm.raw_video(vid, "video.mp4")) == f"s3://raw/videos/{vid}/video.mp4"


def test_quarantine_key() -> None:
    assert nm.quarantine("gold_features", "2026-11-25", "20261125-000000") == (
        "quarantine/gold_features/processing_date=2026-11-25/batch_id=20261125-000000"
    )


@pytest.mark.parametrize("call", [
    lambda: nm.frame_name("1", 8),
    lambda: nm.frame("1", -1),
    lambda: nm.raw_video("abc", "video.mp4"),
    lambda: nm.raw_video("1", "../x.mp4"),
    lambda: nm.raw_video("1", "video.mov"),
    lambda: nm.raw_video("1", "img_1.jpg"),
    lambda: nm.cover_name("gif"),
    lambda: nm.slide_name(0),
    lambda: nm.quarantine("a/b", "2026-11-25", "b"),
    lambda: nm.quarantine("d", "2026-11-25", ""),
])
def test_invalid_inputs(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_image_ext_from_magic_bytes() -> None:
    assert nm.image_ext(b"\xff\xd8\xff\xe0rest") == "jpg"
    assert nm.image_ext(b"\x89PNG\r\n\x1a\nrest") == "png"
    with pytest.raises(ValueError):
        nm.image_ext(b"GIF89a")
