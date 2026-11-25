from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from vicap.model.predict import build_p1_inputs


def _emb(n: int, k: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, k, 768)).astype(np.float32)


def test_text_order_caption_then_asr() -> None:
    t = np.zeros((1, 2, 768), np.float32)
    t[0, 0] = 1.0  # caption+hashtag
    t[0, 1] = 2.0  # ASR
    xt, _ = build_p1_inputs(t, _emb(1, 8, 0))
    assert xt.shape == (1, 1536) and xt.dtype == np.float32
    assert (xt[0, :768] == 1).all() and (xt[0, 768:] == 2).all()


def test_image_mean_full() -> None:
    img = _emb(3, 8, 1)
    _, xv = build_p1_inputs(_emb(3, 2, 2), img)
    assert xv.shape == (3, 768) and xv.dtype == np.float32
    np.testing.assert_allclose(xv, img.mean(1), rtol=1e-5, atol=1e-6)


def test_image_mean_divides_by_present_count() -> None:
    img = _emb(1, 8, 3)
    img[0, 5:] = 0  # chỉ có 5 ảnh
    _, xv = build_p1_inputs(_emb(1, 2, 4), img)
    np.testing.assert_allclose(xv[0], img[0, :5].mean(0), rtol=1e-5, atol=1e-6)


def test_no_image_gives_zero_vector() -> None:
    _, xv = build_p1_inputs(_emb(1, 2, 5), np.zeros((1, 8, 768), np.float32))
    assert not xv.any()


@pytest.mark.parametrize(
    "text_shape, image_shape",
    [((2, 1536), (2, 8, 768)), ((2, 2, 768), (2, 768)), ((2, 2, 768), (3, 8, 768)), ((2, 3, 768), (2, 8, 768))],
)
def test_rejects_wrong_shape(text_shape, image_shape) -> None:
    with pytest.raises(ValueError):
        build_p1_inputs(np.zeros(text_shape, np.float32), np.zeros(image_shape, np.float32))


# Cache của notebook v12 (output kernel cngngquc/mainnotebook, CACHE_TAG v2a). Model cuối dùng bản 2.107 video:
# load_emb chỉ nhận file có đủ mọi video_id của dataset, các hash khác là cache của dataset cũ (1.331 / 2.018 video).
KAGGLE_DIR = Path(os.environ.get("VICAP_P1_CACHE", "data/kaggle/mainnotebook"))
CACHE = {
    "T_cap": "text_caphash_290eedc8__v2a.npz",
    "T_asr": "text_asr_chunked_PhoWhisper_medium_6ba74b77__v2a.npz",
    "V_frames": "siglip_frames8__v2a.npz",
    "V_frames_seq": "siglip_frames8_seq__v2a.npz",
}


@pytest.fixture(scope="module")
def kaggle_cache() -> dict[str, np.ndarray]:
    """Mỗi file ``.npz`` có ``ids`` + ``X``; ghép theo ``ids`` của ``V_frames``."""
    missing = [f for f in CACHE.values() if not (KAGGLE_DIR / f).exists()]
    if missing:
        pytest.skip(f"thiếu cache Kaggle trong {KAGGLE_DIR}: {missing}")
    loaded = {k: np.load(KAGGLE_DIR / f) for k, f in CACHE.items()}
    ids = loaded["V_frames"]["ids"].tolist()
    out: dict[str, np.ndarray] = {}
    for k, z in loaded.items():
        pos = {v: i for i, v in enumerate(z["ids"].tolist())}
        out[k] = z["X"][[pos[v] for v in ids]].astype(np.float32)
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def test_matches_kaggle_v_frames(kaggle_cache) -> None:
    """Trung bình ``V_frames_seq`` theo ``build_p1_inputs`` phải trùng ``V_frames``.

    Notebook kiểm cùng phép này với ngưỡng 0,99."""
    seq, ref = kaggle_cache["V_frames_seq"], kaggle_cache["V_frames"]
    n = len(seq)
    _, xv = build_p1_inputs(np.zeros((n, 2, 768), np.float32), seq)
    keep = np.abs(ref).sum(1) > 0
    assert _cos(xv[keep], ref[keep]).min() > 0.999
    assert not xv[~keep].any()  # video không có ảnh → vector 0 ở cả hai phía


def test_matches_kaggle_t_capasr(kaggle_cache) -> None:
    """``text_emb = [T_cap, T_asr]`` → đúng ``T_capasr = np.hstack([T_cap, T_asr])`` của notebook."""
    t_cap, t_asr = kaggle_cache["T_cap"], kaggle_cache["T_asr"]
    xt, _ = build_p1_inputs(np.stack([t_cap, t_asr], 1), kaggle_cache["V_frames_seq"])
    np.testing.assert_array_equal(xt, np.hstack([t_cap, t_asr]))
