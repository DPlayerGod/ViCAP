from __future__ import annotations

import clickhouse_connect

from vicap.config import ClickHouseConfig


def get_clickhouse_client(config: ClickHouseConfig | None = None):
    config = config or ClickHouseConfig()
    return clickhouse_connect.get_client(
        host=config.host,
        port=config.port,
        username=config.username,
        password=config.password,
        database=config.database,
        secure=config.secure,
    )
