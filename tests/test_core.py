#!/usr/bin/env python3
"""
test_core.py — verifiche di correttezza (eseguibili senza pytest).

    python tests/test_core.py        # stampa PASS/FAIL
    pytest tests/                    # se hai pytest

Coprono: proprietà della SVD, teorema di Eckart–Young, legame SVD↔autovalori,
pseudoinversa/minimi quadrati, e il round-trip embed→extract + codec del segreto.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import svd_core, steganography as steg, metrics as M
from src.steganography import EmbedConfig

rng = np.random.default_rng(0)


def test_svd_reconstruction():
    A = rng.standard_normal((40, 25))
    svd = svd_core.svd_decompose(A)
    assert np.allclose(A, svd.reconstruct(), atol=1e-9)


def test_eckart_young():
    A = rng.standard_normal((60, 45))
    svd = svd_core.svd_decompose(A)
    for k in (1, 5, 10, 20):
        Ak = svd.reconstruct(k)
        ey = svd_core.eckart_young_errors(svd, k)
        assert np.isclose(np.linalg.norm(A - Ak, 2), ey["spectral_sigma_k1"], atol=1e-8)
        assert np.isclose(np.linalg.norm(A - Ak, "fro"), ey["frobenius_tail"], atol=1e-8)


def test_svd_eigendecomposition():
    A = rng.standard_normal((30, 18))
    r = svd_core.verify_svd_eigendecomposition(A)
    assert r["sigma_vs_sqrt_lambda"] < 1e-8
    assert r["reconstruction_rel_err"] < 1e-10


def test_compression_factor():
    # esempio numerico delle slide: ~18% per k=20 su immagine 256x?  (formula generale)
    assert np.isclose(svd_core.compression_factor(100, 100, 10), 10 * 201 / 10000)


def test_pseudoinverse_least_squares():
    A = rng.standard_normal((50, 12))         # sovradeterminato
    b = rng.standard_normal(50)
    x = svd_core.lstsq_via_svd(A, b)
    x_ref, *_ = np.linalg.lstsq(A, b, rcond=None)
    assert np.allclose(x, x_ref, atol=1e-8)


def test_condition_number():
    A = np.diag([5.0, 1.0, 0.5])
    assert np.isclose(svd_core.condition_number(A), 10.0)


def test_secret_codec_roundtrip():
    from src import dataset
    sec = dataset.default_secret(48)
    rec = steg.decompress_secret(steg.compress_secret(sec, 12))
    assert rec.shape == sec.shape
    assert M.normalized_correlation(sec, rec) > 0.85   # compressione con perdita


def test_embed_extract_roundtrip_clean():
    cover = rng.integers(20, 235, (256, 256, 3), dtype=np.uint8)  # evita saturazioni
    bits = rng.integers(0, 2, 800).astype(np.uint8)              # 800*3=2400 ≤ 3072
    ec = EmbedConfig(block=8, delta=32.0, channels=(0, 1, 2), repeat=3)
    res = steg.embed_bits(cover, bits, ec, None)
    out = steg.extract_bits(res.stego, len(bits), ec, None)
    assert M.ber(bits, out) == 0.0          # lossless in chiaro con R=3


def test_hide_reveal_secret_clean():
    from src import dataset
    cover = rng.integers(20, 235, (512, 512, 3), dtype=np.uint8)
    sec = dataset.default_secret(64)
    ec = EmbedConfig(block=8, delta=24.0, channels=(0, 1, 2), repeat=1)
    res, payload = steg.hide_secret(cover, sec, 10, ec, None)
    rec = steg.reveal_secret(res.stego, ec, None)
    assert rec.shape == sec.shape
    assert M.normalized_correlation(sec, rec) > 0.8


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} test superati")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
