"""Kiểm tra độ bền ``daily_pipeline`` trên dữ liệu thật (TODO "Chạy lại cùng một lô" và "Lô có 1 video lỗi").

Bơm vào một lô 2 video giả của creator ``vicap_e2e``, dựng từ một video thật đã crawl (copy ``info.json``,
mp4, ảnh bìa; đổi ``id`` và ``author.uniqueId``): ``E2E_GOOD`` giữ nguyên mp4, ``E2E_BAD`` có mp4 là 20 KB
byte rác (qua được kiểm tra kích thước của crawler, hỏng ở bước ``media``). Đi qua ``crawl_creators`` thật
(client giả) nên file lên ``raw/`` và message vào ``tiktok.videos`` giống hệt crawler.

Chạy trong container Airflow (``batch_id`` = logical date của lần ``dags test``)::

    docker compose exec airflow-scheduler python -m vicap.tools.e2e seed --batch-id 20261126-090000
    docker compose exec airflow-scheduler airflow dags test daily_pipeline 2026-11-26T09:00:00+00:00 \\
        -c '{"creators": "hoanganhquan_blv", "n": 1}'
    docker compose exec airflow-scheduler python -m vicap.tools.e2e check --batch-id 20261126-090000
    # chạy lại dags test cùng logical date rồi check lần nữa: số dòng không đổi, không trùng khoá
    docker compose exec airflow-scheduler python -m vicap.tools.e2e cleanup --batch-id 20261126-090000

``check`` thoát mã 1 nếu có lỗi:

- Iceberg: không bảng nào có hai dòng cùng khoá (``bronze.videos_raw`` theo ``(batch_id, video_id)``,
  ``ops.quarantine`` theo ``(stage, batch_id, video_id, username, errors)``, các bảng khác theo ``TableDef.key``).
- ClickHouse: ``count() FINAL`` = số dòng của bảng gold tương ứng.
- ``E2E_BAD`` nằm trong ``ops.quarantine`` (stage ``media``) và không tới ``gold.features``;
  ``E2E_GOOD`` không bị cách ly, có trong ``vicap.video_predictions``; ``vicap_e2e`` có trong
  ``vicap.creator_profiles``; ``quality_check`` của lô báo ``quarantined`` và ``missing_embedding`` ≥ 1.

``cleanup`` xoá mọi dòng / file của ``vicap_e2e`` (Iceberg, ClickHouse, ``raw/``, ``derived/``) và
``ops.quality_checks`` của lô, để dashboard không còn dữ liệu giả.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Sequence
from typing import Any

import pyarrow as pa
from pyiceberg.expressions import EqualTo, In, Or

from vicap.config import ClickHouseConfig, MinioConfig, crawler_minio_config
from vicap.iceberg.reader import read_table
from vicap.iceberg.tables import (
    ALL_TABLES,
    BRONZE_VIDEOS_RAW,
    GOLD_FEATURES,
    OPS_QUALITY_CHECKS,
    OPS_QUARANTINE,
    SILVER_VIDEOS,
)
from vicap.storage import paths
from vicap.storage.objects import s3_filesystem
from vicap.utils.log import setup_logging

E2E_USER = "vicap_e2e"
E2E_GOOD = "1000000000000000001"
E2E_BAD = "1000000000000000002"
E2E_IDS = (E2E_GOOD, E2E_BAD)
BAD_MP4 = b"\x00" * 20_000

KEYS: dict[str, tuple[str, ...]] = {t.identifier: t.key for t in ALL_TABLES if t.key}
KEYS[BRONZE_VIDEOS_RAW.identifier] = ("batch_id", "video_id")
KEYS[OPS_QUARANTINE.identifier] = ("stage", "batch_id", "video_id", "username", "errors")
CH_TABLES = {  # ClickHouse → Iceberg
    "video_predictions": "gold.video_predictions",
    "creator_profiles": "gold.creator_profiles",
    "creator_profile_history": "gold.creator_profile_history",
}


def _catalog():
    from vicap.iceberg.catalog import load_lakehouse_catalog

    return load_lakehouse_catalog()


def _clickhouse():
    from vicap.clickhouse.client import get_clickhouse_client

    return get_clickhouse_client()


# ─────────────────────────── seed ───────────────────────────


def fake_item(item: dict[str, Any], video_id: str, *, has_cover: bool) -> dict[str, Any]:
    out = copy.deepcopy(item)
    out["id"] = video_id
    out["author"] = {**(out.get("author") or {}), "uniqueId": E2E_USER}
    out.pop("imagePost", None)
    if not has_cover:
        for k in ("cover", "originCover"):
            (out.get("video") or {}).pop(k, None)
    return out


class FixtureClient:
    """``crawler.Client`` trả item / bytes có sẵn thay vì gọi TikTok."""

    def __init__(self, items: dict[str, dict], media: dict[str, bytes], cover: bytes | None):
        self.items, self.media, self.cover = items, media, cover

    def list_post_ids(self, username: str, n: int) -> list[str]:
        return list(self.items)[:n]

    def fetch_item(self, username: str, video_id: str) -> dict[str, Any]:
        return self.items[video_id]

    def video_source(self, item: dict[str, Any], username: str) -> tuple[str, dict[str, str]]:
        return f"e2e://{item['id']}", {}

    def download(self, url: str, headers: dict[str, str] | None = None) -> bytes:
        if url.startswith("e2e://"):
            return self.media[url.removeprefix("e2e://")]
        if self.cover is None:
            raise FileNotFoundError(url)
        return self.cover


def pick_source(catalog: Any) -> dict:
    """Một video (không phải slideshow) thật đã qua Silver."""
    silver = read_table(catalog.load_table(SILVER_VIDEOS.identifier), EqualTo("media_type", "video"))
    rows = [r for r in silver.to_pylist() if r["username"] != E2E_USER and r.get("media_key")]
    if not rows:
        raise SystemExit("silver.videos chưa có video nào để làm mẫu (chạy daily_pipeline một lần trước)")
    return rows[0]


def seed(batch_id: str) -> None:
    from vicap.ingestion.crawler import KafkaSink, RawStore, crawl_creators

    src = pick_source(_catalog())
    fs = s3_filesystem(MinioConfig())  # key pipeline: đọc được raw
    raw = MinioConfig().raw_bucket
    item = json.loads(fs.cat_file(f"{raw}/{paths.raw_video(src['video_id'], paths.INFO_FILE)}"))
    mp4 = fs.cat_file(f"{raw}/{src['media_key']}")
    cover = fs.cat_file(f"{raw}/{src['thumbnail_key']}") if src.get("thumbnail_key") else None

    client = FixtureClient(
        {v: fake_item(item, v, has_cover=cover is not None) for v in E2E_IDS},
        {E2E_GOOD: mp4, E2E_BAD: BAD_MP4},
        cover,
    )
    with KafkaSink() as sink:
        stats = crawl_creators([E2E_USER], n=len(E2E_IDS), batch_id=batch_id, client=client, store=RawStore(),
                               send=sink, refresh_media=True, pause=lambda _: None)
    print(f"seed từ video {src['video_id']} ({src['username']}): {stats.summary()}")


# ─────────────────────────── check ───────────────────────────


def duplicate_keys(table: pa.Table, key: Sequence[str]) -> int:
    """Số khoá xuất hiện hơn một lần (null tính là một giá trị)."""
    if not table.num_rows:
        return 0
    t = table.select(list(key))
    t = pa.table({c: pa.array(["\0" if x is None else str(x) for x in t.column(c).to_pylist()]) for c in key})
    counts = t.group_by(list(key)).aggregate([([], "count_all")])
    return sum(n > 1 for n in counts.column("count_all").to_pylist())


def check(batch_id: str) -> list[str]:
    catalog, client = _catalog(), _clickhouse()
    db = ClickHouseConfig().database
    errors: list[str] = []

    rows: dict[str, int] = {}
    for identifier, key in KEYS.items():
        if not catalog.table_exists(identifier):
            continue
        t = read_table(catalog.load_table(identifier))
        rows[identifier] = t.num_rows
        dup = duplicate_keys(t, key)
        print(f"{identifier:32} {t.num_rows:6} dòng, trùng khoá {dup}")
        if dup:
            errors.append(f"{identifier}: {dup} khoá trùng theo {key}")

    for ch, ice in CH_TABLES.items():
        final = client.query(f"SELECT count() FROM {db}.{ch} FINAL").result_rows[0][0]
        total = client.query(f"SELECT count() FROM {db}.{ch}").result_rows[0][0]
        print(f"{db}.{ch:28} {final:6} dòng FINAL ({total} trước khi gộp), Iceberg {rows.get(ice, 0)}")
        if final != rows.get(ice, 0):
            errors.append(f"{db}.{ch}: FINAL {final} ≠ {ice} {rows.get(ice, 0)}")

    q = read_table(catalog.load_table(OPS_QUARANTINE.identifier), EqualTo("batch_id", batch_id)).to_pylist()
    bad_q = [r for r in q if r["video_id"] == E2E_BAD]
    print(f"quarantine lô {batch_id}: {[(r['stage'], r['video_id'], (r['errors'] or [''])[0][:80]) for r in q]}")
    if not any(r["stage"] == "media" for r in bad_q):
        errors.append(f"{E2E_BAD} không có trong ops.quarantine stage media")
    if any(r["video_id"] == E2E_GOOD for r in q):
        errors.append(f"{E2E_GOOD} bị cách ly")
    feats = read_table(catalog.load_table(GOLD_FEATURES.identifier), In("video_id", list(E2E_IDS)))
    if E2E_BAD in feats.column("video_id").to_pylist():
        errors.append(f"{E2E_BAD} vẫn tới gold.features")

    pred = client.query(f"SELECT count() FROM {db}.video_predictions FINAL WHERE video_id = {{v:String}}",
                        parameters={"v": E2E_GOOD}).result_rows[0][0]
    prof = client.query(f"SELECT main_label, n_videos FROM {db}.creator_profiles FINAL WHERE username = {{u:String}}",
                        parameters={"u": E2E_USER}).result_rows
    print(f"dashboard: {E2E_GOOD} trong video_predictions = {pred}; hồ sơ {E2E_USER} = {prof}")
    if not pred:
        errors.append(f"{E2E_GOOD} chưa tới {db}.video_predictions")
    if not prof:
        errors.append(f"{E2E_USER} chưa tới {db}.creator_profiles")

    qc = {r["check_name"]: r for r in read_table(catalog.load_table(OPS_QUALITY_CHECKS.identifier),
                                                  EqualTo("batch_id", batch_id)).to_pylist()}
    for name in ("quarantined", "missing_embedding"):
        r = qc.get(name)
        print(f"quality_check {name}: {None if r is None else (r['value'], r['severity'], r['detail'])}")
        if r is None or not r["value"] or r["severity"] != "warn":
            errors.append(f"quality_check {name} không cảnh báo")

    for e in errors:
        print("LỖI:", e)
    print("OK" if not errors else f"{len(errors)} lỗi")
    return errors


# ─────────────────────────── cleanup ───────────────────────────


def cleanup(batch_id: str | None) -> None:
    catalog, client = _catalog(), _clickhouse()
    db = ClickHouseConfig().database
    ids = list(E2E_IDS)
    for t in ALL_TABLES:
        if not catalog.table_exists(t.identifier):
            continue
        names = {f.name for f in t.schema.fields}
        conds = []
        if "video_id" in names:
            conds.append(In("video_id", ids))
        if "username" in names:
            conds.append(EqualTo("username", E2E_USER))
        if t is OPS_QUALITY_CHECKS and batch_id:
            conds.append(EqualTo("batch_id", batch_id))
        if not conds:
            continue
        flt = conds[0] if len(conds) == 1 else Or(*conds)
        table = catalog.load_table(t.identifier)
        n = read_table(table, flt).num_rows
        if n:
            table.delete(delete_filter=flt)
        print(f"{t.identifier}: xoá {n}")

    for ch in CH_TABLES:
        cond = "username = {u:String}" + (" OR video_id IN {ids:Array(String)}" if ch == "video_predictions" else "")
        client.command(f"ALTER TABLE {db}.{ch} DELETE WHERE {cond} SETTINGS mutations_sync = 2",
                       parameters={"u": E2E_USER, "ids": ids})
        print(f"{db}.{ch}: xoá dòng {E2E_USER}")

    lake = s3_filesystem(MinioConfig())
    cfg = MinioConfig()
    derived = [f"{cfg.bucket}/{paths.frame(v, k)}" for v in ids for k in range(paths.N_FRAMES)]
    derived += [f"{cfg.bucket}/{paths.audio(v)}" for v in ids]
    lake.rm([p for p in derived if lake.exists(p)])
    crawler = crawler_minio_config()
    raw_fs = s3_filesystem(crawler)
    for v in ids:
        prefix = f"{crawler.raw_bucket}/{paths.raw_prefix(v)}"
        if raw_fs.exists(prefix):
            raw_fs.rm(prefix, recursive=True)
    print("xoá file raw/ và derived/ của video giả")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["seed", "check", "cleanup"])
    ap.add_argument("--batch-id", required=True)
    args = ap.parse_args(argv)

    setup_logging()
    if args.action == "seed":
        seed(args.batch_id)
    elif args.action == "check":
        return 1 if check(args.batch_id) else 0
    else:
        cleanup(args.batch_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
