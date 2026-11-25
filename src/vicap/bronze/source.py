"""Đọc ``tiktok.videos`` cho Bronze: đọc tới ``end_offsets`` chốt lúc bắt đầu, commit offset sau khi ghi xong.

- Group ``bronze-writer``, tắt auto commit. Gán thẳng mọi partition (``assign``) thay cho ``subscribe`` để
  không phải chờ rebalance; đọc từ offset đã commit (chưa có thì từ đầu topic).
- Chỉ lấy record có offset < ``end_offsets`` lúc bắt đầu: message tới sau để lần chạy sau, nên mỗi lần chạy
  là một tập xác định.
- ``commit()`` ghi đúng các ``end_offsets`` đó. Ghi Iceberg lỗi thì không commit: lần chạy lại đọc lại từ
  offset cũ, Bronze khử trùng theo ``(batch_id, video_id)`` nên không sinh dòng trùng.
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from vicap.bronze.videos import KafkaRecord
from vicap.config import KafkaConfig


class KafkaSource:
    def __init__(self, config: KafkaConfig | None = None, consumer: Any = None, idle_timeout: float = 60):
        self.config = config or KafkaConfig()
        if consumer is None:
            from kafka import KafkaConsumer

            consumer = KafkaConsumer(
                bootstrap_servers=self.config.bootstrap_servers.split(","),
                group_id=self.config.group_id,
                enable_auto_commit=False,
                auto_offset_reset="earliest",
            )
        self.consumer = consumer
        self.idle_timeout = idle_timeout
        self._end: dict[Any, int] = {}

    def _partitions(self) -> list[Any]:
        from kafka import TopicPartition

        self.consumer.topics()  # nạp metadata của cluster
        parts = self.consumer.partitions_for_topic(self.config.topic)
        if not parts:
            raise RuntimeError(f"topic {self.config.topic!r} không tồn tại (infra/kafka/create_topics.sh)")
        return [TopicPartition(self.config.topic, p) for p in sorted(parts)]

    def read_pending(self) -> list[KafkaRecord]:
        """Mọi record từ offset đã commit tới ``end_offsets`` lúc gọi hàm."""
        tps = self._partitions()
        self.consumer.assign(tps)
        self._end = dict(self.consumer.end_offsets(tps))
        for tp in tps:
            committed = self.consumer.committed(tp)
            if committed is not None:
                self.consumer.seek(tp, committed)
        pending = {tp: self._end[tp] for tp in tps if self.consumer.position(tp) < self._end[tp]}
        todo = {tp.partition: (self.consumer.position(tp), end) for tp, end in pending.items()}
        logger.info("bronze: cần đọc (partition: từ offset, tới offset) {}", todo)

        records: list[KafkaRecord] = []
        last_progress = time.monotonic()
        while pending:
            batch = self.consumer.poll(timeout_ms=1000)
            for tp, recs in batch.items():
                end = self._end.get(tp, 0)
                for r in recs:
                    if r.offset < end:
                        records.append(KafkaRecord(r.partition, r.offset, r.key, r.value))
                if recs:
                    last_progress = time.monotonic()
            pending = {tp: end for tp, end in pending.items() if self.consumer.position(tp) < end}
            if pending and time.monotonic() - last_progress > self.idle_timeout:
                raise TimeoutError(f"không đọc tiếp được sau {self.idle_timeout}s, còn thiếu {pending}")
        return records

    def commit(self) -> None:
        from kafka.structs import OffsetAndMetadata

        if self._end:
            self.consumer.commit({tp: OffsetAndMetadata(end, "", -1) for tp, end in self._end.items()})

    def close(self) -> None:
        self.consumer.close()

    def __enter__(self) -> KafkaSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
