"""PhoWhisper → ``transcript_raw``, giống hệt ``run_asr`` của notebook (cell 10) đã sinh ``transcript_v2.csv``.

- ``pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-medium", chunk_length_s=30,
  stride_length_s=5, batch_size=8)``, fp16 trên GPU; ``generate_kwargs={"language": "vi", "task": "transcribe",
  "num_beams": 1}``.
- Đưa 16 file một lần; batch lỗi thì chạy lại từng file để biết file nào hỏng.
- Trạng thái: ``ok`` (có chữ), ``empty_or_music`` (rỗng, hoặc audio < 1 s nên không chạy), ``missing``
  (không có audio: slideshow), ``error`` (lần sau thử lại).
- ``transcript_raw`` = ``text.strip()``, không làm sạch (``text_emb[1]`` dùng bản thô, PLAN mục 4.1).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

ASR_MODEL = "vinai/PhoWhisper-medium"
GENERATE_KWARGS = {"language": "vi", "task": "transcribe", "num_beams": 1}
FILE_BATCH = 16
FINAL_STATUS = ("ok", "empty_or_music", "missing")  # không chạy lại; "error" thì chạy lại


def load_asr(model: str = ASR_MODEL, device: str | None = None) -> Any:
    import torch
    from transformers import pipeline

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return pipeline(
        "automatic-speech-recognition", model=model,
        device=0 if device == "cuda" else -1,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        chunk_length_s=30, stride_length_s=5, batch_size=8,
    )


def status_of(text: str, error: str | None) -> str:
    if error:
        return "error"
    return "ok" if text.split() else "empty_or_music"


def transcribe(asr: Callable[..., Any], paths: Sequence[str]) -> list[tuple[str, str | None]]:
    """``[(text, lỗi)]`` theo thứ tự ``paths``, chia nhóm ``FILE_BATCH`` file như notebook."""
    out: list[tuple[str, str | None]] = []
    for g in range(0, len(paths), FILE_BATCH):
        group = list(paths[g:g + FILE_BATCH])
        try:
            out.extend((o["text"].strip(), None) for o in asr(group, generate_kwargs=GENERATE_KWARGS))
        except Exception:  # noqa: BLE001 — batch lỗi → từng file
            for p in group:
                try:
                    out.append((asr(p, generate_kwargs=GENERATE_KWARGS)["text"].strip(), None))
                except Exception as e:  # noqa: BLE001
                    out.append(("", str(e)[:200]))
    return out
