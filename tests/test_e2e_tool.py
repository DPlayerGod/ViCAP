from __future__ import annotations

from datetime import UTC, datetime

import pyarrow as pa

from vicap.ingestion.crawler import crawl_creators
from vicap.tools.e2e import BAD_MP4, E2E_BAD, E2E_GOOD, E2E_IDS, E2E_USER, FixtureClient, duplicate_keys, fake_item

ITEM = {
    "id": "7712356767337843975",
    "author": {"uniqueId": "real_user", "nickname": "x"},
    "createTime": 1764000000,
    "desc": "bàn thắng #bongda",
    "video": {"duration": 12, "width": 576, "height": 1024, "cover": "https://cdn/cover.jpg"},
    "stats": {"playCount": 10},
}
JPEG = b"\xff\xd8\xff" + b"0" * 100


def test_duplicate_keys_counts_repeated_keys_with_nulls() -> None:
    t = pa.table({"stage": ["a", "a", "a", "b"], "video_id": ["1", "1", None, None]})
    assert duplicate_keys(t, ("stage", "video_id")) == 1
    t2 = pa.table({"stage": ["a", "a"], "video_id": [None, None]})
    assert duplicate_keys(t2, ("stage", "video_id")) == 1
    assert duplicate_keys(t.slice(0, 0), ("stage",)) == 0


def test_fake_item_rewrites_id_and_author_without_touching_source() -> None:
    out = fake_item(ITEM, E2E_GOOD, has_cover=False)
    assert out["id"] == E2E_GOOD and out["author"]["uniqueId"] == E2E_USER
    assert "cover" not in out["video"]
    assert ITEM["id"] == "7712356767337843975" and ITEM["video"]["cover"]


def test_fixture_client_goes_through_real_crawler() -> None:
    client = FixtureClient({v: fake_item(ITEM, v, has_cover=True) for v in E2E_IDS},
                           {E2E_GOOD: b"\x01" * 20_000, E2E_BAD: BAD_MP4}, JPEG)

    class Store:
        def __init__(self):
            self.files: dict[str, bytes] = {}

        def put(self, key, data, content_type):
            self.files[key] = data

        def exists(self, key):
            return key in self.files

    store, sent = Store(), []
    stats = crawl_creators([E2E_USER], n=2, batch_id="20261126-090000", client=client, store=store,
                           send=sent.append, refresh_media=True, pause=lambda _: None,
                           now=lambda: datetime(2026, 11, 26, tzinfo=UTC))
    assert stats.ok == 2
    assert [m.video_id for m in sent] == list(E2E_IDS)
    assert {m.username for m in sent} == {E2E_USER}
    assert store.files[f"videos/{E2E_BAD}/video.mp4"] == BAD_MP4
    assert f"videos/{E2E_GOOD}/cover.jpg" in store.files
