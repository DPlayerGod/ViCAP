"""vicap.iceberg.writer: unit test bằng mock + test thật trên SqlCatalog (sqlite, không cần Docker)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest
from pyiceberg.exceptions import NamespaceAlreadyExistsError, NoSuchTableError
from pyiceberg.expressions import In
from pyiceberg.schema import Schema
from pyiceberg.types import LongType, NestedField, StringType

from vicap.iceberg.reader import try_read_table
from vicap.iceberg.tables import ALL_TABLES, GOLD_FEATURES
from vicap.iceberg.writer import ensure_table, replace_by_keys, write_table

SIMPLE_SCHEMA = Schema(
    NestedField(field_id=1, name="video_id", field_type=StringType(), required=True),
    NestedField(field_id=2, name="value", field_type=LongType(), required=False),
)


def _sample() -> pa.Table:
    return pa.table({"video_id": ["1", "2"], "value": [100, 200]})


# ─────────────────────────────── ensure_table ───────────────────────────────


class TestEnsureTable:
    def test_returns_existing_table(self) -> None:
        catalog = MagicMock()
        result = ensure_table(catalog, "gold.t", SIMPLE_SCHEMA)
        assert result is catalog.load_table.return_value
        catalog.create_table.assert_not_called()

    def test_creates_table_when_not_found(self) -> None:
        catalog = MagicMock()
        catalog.load_table.side_effect = NoSuchTableError("gold.t")
        result = ensure_table(catalog, "gold.t", SIMPLE_SCHEMA)
        assert result is catalog.create_table.return_value
        catalog.create_namespace.assert_called_once_with("gold")
        catalog.create_table.assert_called_once_with("gold.t", schema=SIMPLE_SCHEMA)

    def test_other_errors_propagate_and_do_not_create(self) -> None:
        catalog = MagicMock()
        catalog.load_table.side_effect = ConnectionError("catalog down")
        with pytest.raises(ConnectionError):
            ensure_table(catalog, "gold.t", SIMPLE_SCHEMA)
        catalog.create_table.assert_not_called()

    def test_namespace_already_exists_is_ignored(self) -> None:
        catalog = MagicMock()
        catalog.load_table.side_effect = NoSuchTableError("gold.t")
        catalog.create_namespace.side_effect = NamespaceAlreadyExistsError("gold")
        assert ensure_table(catalog, "gold.t", SIMPLE_SCHEMA) is catalog.create_table.return_value

    def test_namespace_other_error_propagates(self) -> None:
        catalog = MagicMock()
        catalog.load_table.side_effect = NoSuchTableError("gold.t")
        catalog.create_namespace.side_effect = PermissionError("denied")
        with pytest.raises(PermissionError):
            ensure_table(catalog, "gold.t", SIMPLE_SCHEMA)

    def test_top_level_table_no_namespace(self) -> None:
        catalog = MagicMock()
        catalog.load_table.side_effect = NoSuchTableError("t")
        ensure_table(catalog, "t", SIMPLE_SCHEMA)
        catalog.create_namespace.assert_not_called()


def test_try_read_table_only_swallows_no_such_table() -> None:
    catalog = MagicMock()
    catalog.load_table.side_effect = NoSuchTableError("x")
    assert try_read_table(catalog, "gold.x") is None
    catalog.load_table.side_effect = ConnectionError("down")
    with pytest.raises(ConnectionError):
        try_read_table(catalog, "gold.x")


# ──────────────────────────────── write (mock) ──────────────────────────────


class TestWriteTable:
    @pytest.fixture
    def table(self) -> MagicMock:
        table = MagicMock()
        table.schema.return_value = SIMPLE_SCHEMA
        return table

    def test_append(self, table: MagicMock) -> None:
        write_table(table, _sample())
        table.append.assert_called_once()
        assert isinstance(table.append.call_args[0][0], pa.Table)

    def test_overwrite_with_filter(self, table: MagicMock) -> None:
        write_table(table, _sample(), mode="overwrite", overwrite_filter="video_id = '1'")
        assert table.overwrite.call_args[1]["overwrite_filter"] == "video_id = '1'"

    def test_invalid_mode(self, table: MagicMock) -> None:
        with pytest.raises(ValueError, match="Unsupported write mode"):
            write_table(table, _sample(), mode="delete")  # type: ignore[arg-type]

    def test_columns_reordered_to_table_schema(self, table: MagicMock) -> None:
        write_table(table, pa.table({"value": [1], "video_id": ["9"]}))
        assert table.append.call_args[0][0].column_names == ["video_id", "value"]

    def test_replace_by_keys_uses_in_filter(self, table: MagicMock) -> None:
        replace_by_keys(table, _sample())
        kwargs = table.overwrite.call_args[1]
        assert kwargs["overwrite_filter"] == In("video_id", ["1", "2"])
        table.append.assert_not_called()

    def test_replace_by_keys_rejects_duplicates_and_skips_empty(self, table: MagicMock) -> None:
        with pytest.raises(ValueError, match="trùng"):
            replace_by_keys(table, pa.table({"video_id": ["1", "1"], "value": [1, 2]}))
        replace_by_keys(table, _sample().slice(0, 0))
        table.overwrite.assert_not_called()


# ───────────────────────── SqlCatalog (sqlite, local) ───────────────────────


# fixture sql_catalog nằm trong conftest.py


def _features(ids: list[str], value: float) -> pa.Table:
    n = len(ids)
    text = np.full((n, 2, 768), value, dtype=np.float32)
    image = np.full((n, 8, 768), value, dtype=np.float32)
    nested = pa.list_(pa.list_(pa.float32()))
    return pa.table({
        "video_id": ids,
        "text_emb": pa.array([t.tolist() for t in text], type=nested),
        "image_emb": pa.array([i.tolist() for i in image], type=nested),
        "n_source_images": pa.array([8] * n, type=pa.int32()),
        "text_encoder": ["phobert"] * n,
        "image_encoder": ["siglip2"] * n,
        "batch_id": ["b"] * n,
        "created_at": pa.array([0] * n, type=pa.timestamp("us", tz="UTC")),
    })


def test_all_table_schemas_can_be_created(sql_catalog) -> None:
    for t in ALL_TABLES:
        ensure_table(sql_catalog, t.identifier, t.schema, t.partition_spec)
    assert sorted(n for (n,) in sql_catalog.list_namespaces()) == ["bronze", "gold", "ops", "silver"]


def test_replace_by_keys_roundtrip_with_embeddings(sql_catalog) -> None:
    table = ensure_table(sql_catalog, GOLD_FEATURES.identifier, GOLD_FEATURES.schema)
    replace_by_keys(table, _features(["1", "2", "3"], 0.1))
    replace_by_keys(table, _features(["2", "4"], 0.9))  # 2 được thay, 4 thêm mới, 1 và 3 giữ nguyên

    result = table.refresh().scan().to_arrow().sort_by("video_id")
    assert result.column("video_id").to_pylist() == ["1", "2", "3", "4"]
    first_values = [row[0][0] for row in result.column("image_emb").to_pylist()]
    assert first_values == pytest.approx([0.1, 0.9, 0.1, 0.9])
    assert np.asarray(result.column("image_emb")[1].as_py()).shape == (8, 768)
