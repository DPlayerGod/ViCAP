"""DockerOperator cho các bước chạy trong image gpu-worker (asr, embed, predict) — PLAN mục 9.4.

Airflow không mount Docker socket; nó gọi Docker qua service ``docker-proxy`` (chỉ mở API container /
image). Container GPU chạy trên network ``lakehouse_network`` để gọi được MinIO, Iceberg REST, MLflow.
"""

from __future__ import annotations

import os

from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import DeviceRequest, Mount

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


def _worker_env(extra: dict[str, str] | None) -> dict[str, str]:
    env = {k: v for k in _FORWARD_ENV if (v := os.getenv(k))}
    env["MLFLOW_S3_ENDPOINT_URL"] = env.get("MINIO_ENDPOINT", "http://minio:9000")
    env.update(extra or {})
    return env


def gpu_task(task_id: str, command: str | list[str], *, environment: dict[str, str] | None = None,
             **kwargs) -> DockerOperator:
    """Một task chạy ``command`` trong ``vicap/gpu-worker`` với toàn bộ GPU của host."""
    return DockerOperator(
        task_id=task_id,
        image=os.getenv("GPU_WORKER_IMAGE", "vicap/gpu-worker:dev"),
        command=command,
        docker_url=os.getenv("DOCKER_HOST_URL", "tcp://docker-proxy:2375"),
        network_mode="lakehouse_network",
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
        mounts=[Mount(source="vicap_hf_cache", target="/models/hf", type="volume")],
        # private_environment: không hiện trong tab "Rendered Template" của UI (có secret S3)
        private_environment=_worker_env(environment),
        mount_tmp_dir=False,  # thư mục tmp của Airflow không tồn tại trên host Docker
        auto_remove="success",
        force_pull=False,
        **kwargs,
    )
