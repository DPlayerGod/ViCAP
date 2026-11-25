"""Crawler: parse itemStruct, message Kafka, luồng tải lên raw/ rồi gửi message (không gọi mạng)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from vicap.ingestion.crawler import CrawlStats, KafkaSink, crawl_creators, crawl_post
from vicap.ingestion.message import MessageError, VideoMessage
from vicap.ingestion.parse import ItemParseError, extract_hashtags, parse_item, strip_hashtags
from vicap.pipelines.crawl import read_creators

JPEG = b"\xff\xd8\xff" + b"j" * 100
PNG = b"\x89PNG\r\n\x1a\n" + b"p" * 100
MP4 = b"\x00\x00\x00\x18ftypmp4" + b"m" * 20_000
TS = datetime(2026, 11, 25, 1, 2, 3, tzinfo=UTC)


def video_item(video_id: str = "7706426861328272641", username: str = "creator.a") -> dict:
    """Rút gọn từ itemStruct thật (video thường), giữ đúng tên và kiểu trường TikTok trả."""
    return {
        "id": video_id,
        "createTime": 1794312000,
        "desc": "Nước súp chua cay​ #LẩuThái #TomYum  👨‍🍳 #LẩuThái",
        "author": {"uniqueId": username, "nickname": "A"},
        "authorStats": {"followerCount": 1300000},
        "authorStatsV2": {"followerCount": "1300000"},
        "stats": {"diggCount": 765, "playCount": 21900, "collectCount": "32"},
        "statsV2": {"diggCount": "765", "shareCount": "8", "commentCount": "4", "playCount": "21900",
                    "collectCount": "32"},
        "challenges": [{"title": "lẩuthái"}, {"title": "tomyum"}],
        "music": {"duration": 99},
        "video": {"duration": 99, "width": 576, "height": 1024, "codecType": "h264",
                  "cover": "https://cdn/cover", "originCover": "https://cdn/origin"},
    }


def slideshow_item(video_id: str = "7591386053089479943") -> dict:
    """Slideshow: video.duration = 0, thời lượng nằm ở music; ảnh trong imagePost.images."""
    return {
        "id": video_id,
        "createTime": "1767225600",
        "desc": "ảnh đẹp",
        "author": {"uniqueId": "creator.b"},
        "statsV2": {"playCount": "10"},
        "challenges": [{"title": "dulich"}],
        "music": {"duration": 60},
        "video": {"duration": 0, "width": 0, "height": 0, "cover": "https://cdn/s-cover"},
        "imagePost": {
            "images": [
                {"imageWidth": 1080, "imageHeight": 1440, "imageURL": {"urlList": ["https://bad/1", "https://cdn/1"]}},
                {"imageWidth": 1080, "imageHeight": 1440, "imageURL": {"urlList": ["https://cdn/2"]}},
            ],
            "cover": {"imageURL": {"urlList": ["https://cdn/ip-cover"]}},
        },
    }


# ─────────────────────────── parse ───────────────────────────


def test_hashtags_from_description_keep_case_and_dedupe():
    assert extract_hashtags("a #LẩuThái b​#TomYum #LẩuThái") == ["LẩuThái", "TomYum"]
    assert strip_hashtags("Nước súp​ #LẩuThái  cay #x") == "Nước súp cay"


def test_parse_video():
    meta = parse_item(video_item())
    assert meta.video_id == "7706426861328272641"
    assert meta.username == "creator.a"
    assert meta.posted_at == datetime.fromtimestamp(1794312000, UTC)
    assert meta.media_type == "video"
    assert meta.hashtags == ("LẩuThái", "TomYum")
    assert (meta.duration, meta.width, meta.height) == (99, 576, 1024)
    assert (meta.play_count, meta.like_count, meta.share_count, meta.comment_count, meta.save_count) == (
        21900, 765, 8, 4, 32,
    )
    assert meta.follower_count == 1300000
    assert meta.cover_url == "https://cdn/cover"
    assert meta.image_urls == ()


def test_parse_slideshow_uses_music_duration_and_challenges_fallback():
    meta = parse_item(slideshow_item())
    assert meta.media_type == "slideshow"
    assert meta.duration == 60
    assert (meta.width, meta.height) == (1080, 1440)
    assert meta.hashtags == ("dulich",)  # desc không có #… → challenges
    assert meta.image_urls == (("https://bad/1", "https://cdn/1"), ("https://cdn/2",))
    assert meta.cover_url == "https://cdn/s-cover"


def test_parse_slideshow_cover_fallbacks():
    item = slideshow_item()
    item["video"].pop("cover")
    assert parse_item(item).cover_url == "https://cdn/ip-cover"
    item["imagePost"].pop("cover")
    assert parse_item(item).cover_url == "https://bad/1"


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(id="abc"),
    lambda d: d.pop("author"),
    lambda d: d.update(createTime=None),
    lambda d: d["imagePost"]["images"][1]["imageURL"].update(urlList=[]),
])
def test_parse_rejects_broken_items(mutate):
    item = slideshow_item()
    mutate(item)
    with pytest.raises(ItemParseError):
        parse_item(item)


# ─────────────────────────── message ───────────────────────────


def ok_message(**kw) -> VideoMessage:
    base = dict(
        video_id="1", username="u", batch_id="20261125-010203", crawl_ts=TS, media_type="video",
        posted_at=TS, hashtags=["a"], duration=5, info_key="videos/1/info.json", media_key="videos/1/video.mp4",
        thumbnail_key="videos/1/cover.jpg",
    )
    return VideoMessage(**{**base, **kw})


def test_message_roundtrip_and_key():
    msg = ok_message(description="Nước #a")
    raw = msg.to_bytes()
    assert msg.key() == b"1"
    assert len(raw) < 2000
    assert "Nước" in raw.decode()  # ensure_ascii=False
    assert VideoMessage.from_bytes(raw) == msg


def test_failed_message_roundtrip():
    msg = VideoMessage.failed("1", "u", batch_id="b", crawl_ts=TS, error="TikTokError: status=10204")
    assert not msg.ok
    assert VideoMessage.from_bytes(msg.to_bytes()) == msg


@pytest.mark.parametrize("raw", [
    b"not json",
    b"[]",
    json.dumps({**ok_message().to_dict(), "v": 99}).encode(),
    json.dumps({**ok_message().to_dict(), "extra": 1}).encode(),
    json.dumps({**ok_message().to_dict(), "crawl_ts": "hôm qua"}).encode(),
    json.dumps({**ok_message().to_dict(), "crawl_ts": "2026-11-25T01:02:03"}).encode(),  # thiếu múi giờ
    json.dumps({k: v for k, v in ok_message().to_dict().items() if k != "username"}).encode(),
    json.dumps({**ok_message().to_dict(), "media_key": "videos/2/video.mp4"}).encode(),
    json.dumps({**ok_message().to_dict(), "media_type": "live"}).encode(),
    json.dumps({**ok_message().to_dict(), "duration": "5"}).encode(),
    json.dumps({**ok_message().to_dict(), "media_type": "slideshow", "media_key": "videos/1/"}).encode(),
    json.dumps({**ok_message().to_dict(), "status": "error", "error": None}).encode(),
])
def test_message_rejects_bad_values(raw):
    with pytest.raises(MessageError):
        VideoMessage.from_bytes(raw)


# ─────────────────────────── crawler ───────────────────────────


class FakeStore:
    def __init__(self, existing: dict[str, bytes] | None = None):
        self.objects: dict[str, bytes] = dict(existing or {})
        self.order: list[str] = []

    def put(self, key, data, content_type):
        self.objects[key] = data
        self.order.append(key)

    def exists(self, key):
        return key in self.objects


class FakeClient:
    def __init__(self, items: dict[str, dict], posts: dict[str, list[str]], files: dict[str, bytes],
                 fail_fetch: dict[str, int] | None = None):
        self.items, self.posts, self.files = items, posts, files
        self.fail_fetch = dict(fail_fetch or {})  # video_id → số lần lỗi trước khi qua
        self.downloads: list[str] = []

    def list_post_ids(self, username, n):
        if username not in self.posts:
            raise RuntimeError("không có creator")
        return self.posts[username][:n]

    def fetch_item(self, username, video_id):
        if self.fail_fetch.get(video_id, 0) > 0:
            self.fail_fetch[video_id] -= 1
            raise RuntimeError("status=10231")
        return self.items[video_id]

    def video_source(self, item, username):
        return f"https://cdn/mp4/{item['id']}", {"Referer": "x"}

    def download(self, url, headers=None):
        self.downloads.append(url)
        if url not in self.files:
            raise OSError(f"404 {url}")
        return self.files[url]


def make_client(**kw) -> FakeClient:
    v, s = video_item("11"), slideshow_item("22")
    return FakeClient(
        items={"11": v, "22": s},
        posts={"creator.a": ["11", "22"]},
        files={"https://cdn/mp4/11": MP4, "https://cdn/cover": PNG, "https://cdn/1": JPEG, "https://cdn/2": PNG,
               "https://cdn/s-cover": JPEG},
        **kw,
    )


def test_crawl_post_video_uploads_then_info_json_last():
    store = FakeStore()
    msg, skipped = crawl_post(make_client(), store, "creator.a", "11", batch_id="b", crawl_ts=TS)
    assert not skipped
    assert store.order == ["videos/11/video.mp4", "videos/11/cover.png", "videos/11/info.json"]
    assert json.loads(store.objects["videos/11/info.json"])["id"] == "11"
    assert (msg.media_key, msg.thumbnail_key, msg.info_key) == (
        "videos/11/video.mp4", "videos/11/cover.png", "videos/11/info.json",
    )
    assert msg.ok and msg.media_type == "video" and msg.image_keys == []


def test_crawl_post_slideshow_tries_next_cdn():
    store, client = FakeStore(), make_client()
    msg, _ = crawl_post(client, store, "creator.a", "22", batch_id="b", crawl_ts=TS)
    assert msg.image_keys == ["videos/22/img_001.jpg", "videos/22/img_002.png"]
    assert msg.media_key == "videos/22/"
    assert msg.thumbnail_key == "videos/22/cover.jpg"
    assert "https://bad/1" in client.downloads  # CDN đầu lỗi, sang CDN thứ hai


def test_crawl_post_skips_existing_mp4_unless_refresh():
    client = make_client()
    store = FakeStore({"videos/11/video.mp4": b"old"})
    _, skipped = crawl_post(client, store, "creator.a", "11", batch_id="b", crawl_ts=TS)
    assert skipped and store.objects["videos/11/video.mp4"] == b"old"
    assert "https://cdn/mp4/11" not in client.downloads

    _, skipped = crawl_post(client, store, "creator.a", "11", batch_id="b", crawl_ts=TS, refresh_media=True)
    assert not skipped and store.objects["videos/11/video.mp4"] == MP4


def test_crawl_post_rejects_tiny_mp4():
    client = make_client()
    client.files["https://cdn/mp4/11"] = b"x" * 10
    store = FakeStore()
    with pytest.raises(RuntimeError, match="byte"):
        crawl_post(client, store, "creator.a", "11", batch_id="b", crawl_ts=TS)
    assert "videos/11/info.json" not in store.objects


def run(client, **kw) -> tuple[CrawlStats, list[VideoMessage], FakeStore]:
    sent: list[VideoMessage] = []
    store = FakeStore()
    stats = crawl_creators(
        kw.pop("creators", ["creator.a"]), n=kw.pop("n", 10), batch_id="20261125-010203", client=client,
        store=store, send=sent.append, now=lambda: TS, pause=lambda s: None, **kw,
    )
    return stats, sent, store


def test_crawl_creators_sends_one_message_per_post_after_upload():
    stats, sent, store = run(make_client())
    assert [m.video_id for m in sent] == ["11", "22"]
    assert all(m.ok and m.batch_id == "20261125-010203" and m.crawl_ts == TS for m in sent)
    assert all(store.exists(m.info_key) for m in sent)
    assert (stats.creators, stats.posts, stats.ok, stats.failed) == (1, 2, 2, 0)


def test_crawl_creators_retries_then_sends_error_message():
    stats, sent, _ = run(make_client(fail_fetch={"11": 1, "22": 99}), retries=2)
    by_id = {m.video_id: m for m in sent}
    assert by_id["11"].ok  # lỗi 1 lần rồi qua
    assert not by_id["22"].ok and "10231" in by_id["22"].error
    assert (stats.ok, stats.failed) == (1, 1)


def test_crawl_creators_records_creator_errors_and_respects_n():
    stats, sent, _ = run(make_client(), creators=["missing", "creator.a"], n=1)
    assert [m.video_id for m in sent] == ["11"]
    assert set(stats.creator_errors) == {"missing"}


class FakeFuture:
    def __init__(self, error: Exception | None = None):
        self.error = error

    def get(self, timeout=None):
        if self.error:
            raise self.error


class FakeProducer:
    def __init__(self, error: Exception | None = None):
        self.sent, self.error, self.closed = [], error, False

    def send(self, topic, key=None, value=None):
        self.sent.append((topic, key, value))
        return FakeFuture(self.error)

    def flush(self, timeout=None):
        pass

    def close(self):
        self.closed = True


def test_kafka_sink_sends_key_and_value(monkeypatch):
    monkeypatch.setenv("KAFKA_TOPIC_VIDEOS", "tiktok.videos")
    producer = FakeProducer()
    with KafkaSink(producer=producer) as sink:
        sink(ok_message())
    topic, key, value = producer.sent[0]
    assert (topic, key) == ("tiktok.videos", b"1")
    assert VideoMessage.from_bytes(value) == ok_message()
    assert producer.closed


def test_kafka_sink_close_raises_on_failed_send():
    producer = FakeProducer(error=RuntimeError("broker down"))
    sink = KafkaSink(producer=producer)
    sink(ok_message())
    with pytest.raises(RuntimeError, match="broker down"):
        sink.close()
    assert producer.closed


def test_read_creators(tmp_path):
    f = tmp_path / "creators.txt"
    f.write_text("# comment\n@a\nb  # note\n\na\n", encoding="utf-8")
    assert read_creators(f) == ["a", "b"]
