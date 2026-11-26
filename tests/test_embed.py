from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
from pyiceberg.io.pyarrow import schema_to_pyarrow

from vicap.iceberg.reader import read_table
from vicap.iceberg.tables import GOLD_FEATURES, OPS_QUARANTINE, SILVER_MEDIA, SILVER_TRANSCRIPTS, SILVER_VIDEOS
from vicap.iceberg.writer import ensure_table, write_table
from vicap.pipelines.embed import run_embed, to_matrix_column

T0 = datetime(2026, 11, 26, 9, tzinfo=UTC)


def test_to_matrix_column_roundtrip() -> None:
    x = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    col = to_matrix_column(x)
    assert col.type == pa.list_(pa.list_(pa.float32()))
    assert np.array(col.to_pylist(), dtype=np.float32).tolist() == x.tolist()


class FakeText:
    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    def encode_texts(self, texts):
        self.calls.append(("texts", list(texts)))
        return np.ones((len(texts), 768), np.float32)

    def encode_chunked(self, texts):
        self.calls.append(("chunked", list(texts)))
        return np.array([np.full(768, 2.0) if t else np.zeros(768) for t in texts], np.float32)


class FakeImage:
    def encode(self, images):
        out = np.ones((len(images), 8, 768), np.float32)
        for i, ims in enumerate(images):
            for k, im in enumerate(ims):
                if im is None:
                    out[i, k] = 0
        return out


class FakeFS:
    def __init__(self, missing: set[str]):
        self.missing = missing

    def cat_file(self, path: str) -> bytes:
        if path in self.missing:
            raise FileNotFoundError(path)
        return b"jpg"


def _rows(table, rows):
    return pa.Table.from_pylist(rows, schema=schema_to_pyarrow(table.schema()))


def test_run_embed(sql_catalog, s3_env) -> None:
    silver = ensure_table(sql_catalog, SILVER_VIDEOS.identifier, SILVER_VIDEOS.schema)
    media = ensure_table(sql_catalog, SILVER_MEDIA.identifier, SILVER_MEDIA.schema)
    trans = ensure_table(sql_catalog, SILVER_TRANSCRIPTS.identifier, SILVER_TRANSCRIPTS.schema)
    base = {"username": "u", "media_type": "video", "crawl_ts": T0, "batch_id": "b1", "updated_at": T0}
    write_table(silver, _rows(silver, [
        {**base, "video_id": "1", "caption": "Món ngon", "hashtags": ["food", "vn"]},
        {**base, "video_id": "2", "caption": "", "hashtags": ["dance"]},
        {**base, "video_id": "3", "caption": "x", "hashtags": []},  # thiếu frame _5 → cách ly
        {**base, "video_id": "4", "caption": "y", "hashtags": []},  # chưa có transcript
    ]))
    m = {"media_type": "video", "n_source_images": 100, "thumb_source": "thumbnail", "has_audio": True,
         "audio_duration": 5.0, "batch_id": "b1", "created_at": T0}
    write_table(media, _rows(media, [{**m, "video_id": v} for v in ("1", "2", "3", "4")]))
    t = {"asr_model": "vinai/PhoWhisper-medium", "batch_id": "b1", "created_at": T0}
    write_table(trans, _rows(trans, [
        {**t, "video_id": "1", "transcript_raw": "xin chào", "transcript_status": "ok", "has_transcript": True},
        {**t, "video_id": "2", "transcript_raw": "", "transcript_status": "empty_or_music", "has_transcript": False},
        {**t, "video_id": "3", "transcript_raw": "a", "transcript_status": "ok", "has_transcript": True},
    ]))
    text = FakeText()
    fs = FakeFS({"lakehouse/derived/frames/3_5.jpg"})

    s = run_embed(catalog=sql_catalog, fs=fs, text_encoder=text, image_encoder=FakeImage(), now=lambda: T0)
    assert (s.written, s.quarantined, s.not_ready) == (2, 1, 1)
    assert text.calls[0] == ("texts", ["Món ngon food vn", "dance", "x"])  # caption + " " + hashtag
    assert text.calls[1] == ("chunked", ["xin chào", "", "a"])
    out = {r["video_id"]: r for r in read_table(sql_catalog.load_table(GOLD_FEATURES.identifier)).to_pylist()}
    assert sorted(out) == ["1", "2"]
    te = np.array(out["2"]["text_emb"])
    assert te.shape == (2, 768) and te[1].sum() == 0  # không có transcript → vector 0
    assert np.array(out["1"]["image_emb"]).shape == (8, 768) and out["1"]["n_source_images"] == 100
    q = read_table(sql_catalog.load_table(OPS_QUARANTINE.identifier)).to_pylist()
    assert [(r["stage"], r["video_id"]) for r in q] == [("gold_features", "3")]

    s = run_embed(catalog=sql_catalog, fs=fs, text_encoder=FakeText(), image_encoder=FakeImage(), now=lambda: T0)
    assert (s.skipped, s.written) == (2, 0)  # video 3 tính lại (vẫn cách ly)
