# GPU worker: PhoWhisper, PhoBERT, SigLIP2, predict P1 (PLAN mục 9.4).
# Airflow gọi bằng DockerOperator (dags/common/gpu.py); không chạy thường trực.
# Driver host 537.70 chạy tới CUDA 12.2 → dùng bản CUDA 11.8. torch >= 2.6 vì transformers chặn torch.load file
# .bin (PhoWhisper-medium chỉ có pytorch_model.bin) trên torch cũ hơn (CVE-2025-32434).
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# torch đã có trong base image nên pip không tải lại.
RUN pip install --no-cache-dir ".[gpu]" \
 && python -c "import torch, transformers, mlflow, vicap; print('torch', torch.__version__)"

RUN useradd --create-home --uid 1000 worker && mkdir -p /models/hf && chown -R worker /models
USER worker
