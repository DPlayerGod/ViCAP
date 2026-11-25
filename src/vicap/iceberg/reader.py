from __future__ import annotations

import pyarrow as pa
from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import NoSuchTableError
from pyiceberg.expressions import AlwaysTrue, BooleanExpression
from pyiceberg.table import Table


def read_table(table: Table, row_filter: str | BooleanExpression | None = None) -> pa.Table:
    return table.scan(row_filter=row_filter if row_filter is not None else AlwaysTrue()).to_arrow()


def try_read_table(catalog: Catalog, identifier: str) -> pa.Table | None:
    """Trả ``None`` khi bảng chưa tồn tại. Lỗi khác (mạng, quyền, …) được ném ra."""
    try:
        table = catalog.load_table(identifier)
    except NoSuchTableError:
        return None
    return read_table(table)
