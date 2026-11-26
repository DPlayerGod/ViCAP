"""DockerOperator cho các bước chạy trong image gpu-worker (media, asr, embed, predict) — PLAN mục 9.4.

Airflow không mount Docker socket; nó gọi Docker qua service ``docker-proxy`` (chỉ mở API container /
image). Container GPU chạy trên network ``lakehouse_network`` để gọi được MinIO, Iceberg REST, MLflow.

``GPU_WORKER_DEV_SRC=true`` (docker-compose bật sẵn khi dev): lúc chạy task, đọc đường dẫn host của bind mount
``/opt/airflow/src`` (inspect chính container Airflow qua proxy) rồi mount nó vào ``/app/src`` của container
worker + ``PYTHONPATH=/app/src``, giống ``make asr`` / ``make embed``. Không cần build lại image mỗi lần sửa code.
Tắt biến này thì worker chạy code đã build trong image.
"""

from __future__ import annotations

import os
import socket

from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import DeviceRequest, Mount

AIRFLOW_SRC = "/opt/airflow/src"
WORKER_SRC = "/app/src"

# Biến môi trường chuyển từ Airflow sang container (giá trị lấy lúc parse DAG từ env của Airflow).
_FORWARD_ENV = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "MINIO_ENDPOINT",
    "CATALOG_URI_CLIENT",
    "CATALOG_WAREHOUSE",
    "MLFLOW_TRACKING_URI",
    "TEXT_ENCODER",
    "IMAGE_ENCODER",
)


def _docker_url() -> str:
    return os.getenv("DOCKER_HOST_URL", "tcp://docker-proxy:2375")


def _worker_env(extra: dict[str, str] | None) -> dict[str, str]:
    env = {k: v for k in _FORWARD_ENV if (v := os.getenv(k))}
    env["MLFLOW_S3_ENDPOINT_URL"] = env.get("MINIO_ENDPOINT", "http://minio:9000")
    env.update(extra or {})
    return env


def host_src_path() -> str | None:
    """Đường dẫn trên host Docker của thư mục đang mount vào ``/opt/airflow/src`` (None nếu không có)."""
    import docker

    client = docker.APIClient(base_url=_docker_url())
    try:
        info = client.inspect_container(socket.gethostname())  # hostname mặc định = id container
    finally:
        client.close()
    for m in info.get("Mounts", []):
        if m.get("Destination") == AIRFLOW_SRC and m.get("Type") == "bind":
            return m["Source"]
    return None


class WorkerOperator(DockerOperator):
    def execute(self, context):
        if os.getenv("GPU_WORKER_DEV_SRC", "").lower() == "true":
            src = host_src_path()
            if src:
                self.mounts = [*self.mounts, Mount(target=WORKER_SRC, source=src, type="bind", read_only=True)]
                self.environment = {**(self.environment or {}), "PYTHONPATH": WORKER_SRC}
                self.log.info("mount code từ host %s → %s", src, WORKER_SRC)
        return super().execute(context)


def gpu_task(task_id: str, command: str | list[str], *, gpu: bool = True,
             environment: dict[str, str] | None = None, **kwargs) -> DockerOperator:
    """Một task chạy ``command`` trong ``vicap/gpu-worker``; ``gpu=True`` thì gắn toàn bộ GPU của host."""
    return WorkerOperator(
        task_id=task_id,
        image=os.getenv("GPU_WORKER_IMAGE", "vicap/gpu-worker:dev"),
        command=command,
        docker_url=_docker_url(),
        network_mode="lakehouse_network",
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])] if gpu else None,
        mounts=[Mount(source="vicap_hf_cache", target="/models/hf", type="volume")],
        # private_environment: không hiện trong tab "Rendered Template" của UI (có secret S3)
        private_environment=_worker_env(environment),
        mount_tmp_dir=False,  # thư mục tmp của Airflow không tồn tại trên host Docker
        auto_remove="success",
        force_pull=False,
        **kwargs,
    )
