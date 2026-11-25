from __future__ import annotations

import warnings
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc
from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError, NoSuchTableError
from pyiceberg.expressions import BooleanExpression, In
from pyiceberg.io.pyarrow import schema_to_pyarrow
from pyiceberg.partitioning import PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import Table

WriteMode = Literal["append", "overwrite"]


def ensure_table(
    catalog: Catalog,
    identifier: str,
    schema: Schema,
    partition_spec: PartitionSpec | None = None,
) -> Table:
    """Load bảng, chỉ tạo mới khi catalog báo ``NoSuchTableError``.

    HOSE bắt ``Exception`` chung nên lỗi mạng tới catalog cũng dẫn tới tạo bảng mới (PLAN mục 9.2).
    """
    try:
        return catalog.load_table(identifier)
    except NoSuchTableError:
        pass
    namespace = ".".join(identifier.split(".")[:-1])
    if namespace:
        try:
            catalog.create_namespace(namespace)
        except NamespaceAlreadyExistsError:
            pass
    if partition_spec is None:
        return catalog.create_table(identifier, schema=schema)
    return catalog.create_table(identifier, schema=schema, partition_spec=partition_spec)


def write_table(
    table: Table,
    data: pa.Table,
    mode: WriteMode = "append",
    overwrite_filter: str | BooleanExpression | None = None,
) -> None:
    arrow_table = _align_arrow_schema(data, table)
    if mode == "append":
        table.append(arrow_table)
        return
    if mode == "overwrite":
        if overwrite_filter is not None:
            with warnings.catch_warnings():  # lần đầu ghi một lô / khoá: chưa có dòng nào để xoá là bình thường
                warnings.filterwarnings("ignore", "Delete operation did not match any records")
                table.overwrite(arrow_table, overwrite_filter=overwrite_filter)
        else:
            table.overwrite(arrow_table)
        return
    raise ValueError(f"Unsupported write mode: {mode}")


def replace_by_keys(table: Table, data: pa.Table, key: str = "video_id") -> None:
    """Xoá các dòng có ``key`` thuộc ``data`` rồi ghi ``data``, trong cùng một commit (PLAN mục 5).

    Dùng cho bảng có cột mảng (``gold.features``, ``gold.video_predictions``) thay cho
    ``table.upsert``, vì ``upsert`` so từng cột không phải khoá, gặp mảng 8×768 thì lỗi hoặc rất chậm.
    """
    if data.num_rows == 0:
        return
    keys = data.column(key)
    if keys.null_count:
        raise ValueError(f"{key} có giá trị null")
    if len(pc.unique(keys)) != data.num_rows:
        raise ValueError(f"{key} bị trùng trong lô cần ghi")
    write_table(table, data, mode="overwrite", overwrite_filter=In(key, keys.to_pylist()))


def _align_arrow_schema(arrow_table: pa.Table, iceberg_table: Table) -> pa.Table:
    """Cast Arrow table theo schema Arrow mà PyIceberg suy ra từ bảng Iceberg.

    Arrow dựng từ Python / NumPy cho mọi field nullable, còn schema Iceberg có thể khai báo field
    *required*; PyIceberg kiểm tra và từ chối nếu lệch. Cột được sắp lại theo thứ tự của bảng.
    Cast cũng xử lý lệch timestamp (``us`` → ``us, tz=UTC``) và kiểu list lồng nhau của cột embedding.
    """
    target_schema = schema_to_pyarrow(iceberg_table.schema())
    if arrow_table.schema.equals(target_schema):
        return arrow_table
    return arrow_table.select(target_schema.names).cast(target_schema)
