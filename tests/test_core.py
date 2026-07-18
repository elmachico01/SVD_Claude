#!/usr/bin/env python3
"""Core correctness tests. Run with ``python tests/test_core.py`` or pytest."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import dataset, metrics as M, steganography as steg, svd_core
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
        residual = A - svd.reconstruct(k)
        ey = svd_core.eckart_young_errors(svd, k)
        assert np.isclose(np.linalg.norm(residual, 2), ey["spectral_sigma_k1"], atol=1e-8)
        assert np.isclose(np.linalg.norm(residual, "fro"), ey["frobenius_tail"], atol=1e-8)


def test_svd_eigendecomposition():
    result = svd_core.verify_svd_eigendecomposition(rng.standard_normal((30, 18)))
    assert result["sigma_vs_sqrt_lambda"] < 1e-8
    assert result["reconstruction_rel_err"] < 1e-10


def test_pseudoinverse_least_squares():
    A, b = rng.standard_normal((50, 12)), rng.standard_normal(50)
    assert np.allclose(svd_core.lstsq_via_svd(A, b),
                       np.linalg.lstsq(A, b, rcond=None)[0], atol=1e-8)


def test_secret_codec_and_crc():
    secret = dataset.default_secret(48)
    payload = steg.compress_secret(secret, 12)
    recovered = steg.decompress_secret(payload)
    assert recovered.shape == secret.shape
    assert M.normalized_correlation(secret, recovered) > 0.85
    corrupted = bytearray(payload)
    corrupted[-1] ^= 1
    try:
        steg.decompress_secret(bytes(corrupted))
    except ValueError as exc:
        assert "CRC" in str(exc)
    else:
        raise AssertionError("CRC corruption not detected")


def test_clean_blind_roundtrip():
    cover = rng.integers(20, 235, (256, 256, 3), dtype=np.uint8)
    bits = rng.integers(0, 2, 800, dtype=np.uint8)
    cfg = EmbedConfig(block=8, delta=32.0, repeat=1)
    result = steg.embed_bits(cover, bits, cfg)
    assert np.array_equal(bits, steg.extract_bits(result.stego, len(bits), cfg))


def test_guided_mode_requires_same_priority_map():
    cover = rng.integers(20, 235, (256, 256, 3), dtype=np.uint8)
    priority = rng.random((32, 32))
    bits = rng.integers(0, 2, 500, dtype=np.uint8)
    cfg = EmbedConfig(block=8, delta=32.0, repeat=1, key=17)
    result = steg.embed_bits(cover, bits, cfg, priority)
    correct = steg.extract_bits(result.stego, len(bits), cfg, priority)
    wrong = steg.extract_bits(result.stego, len(bits), cfg, None)
    assert np.array_equal(bits, correct)
    assert not np.array_equal(bits, wrong)


def test_interleaved_repetition_roundtrip():
    cover = rng.integers(20, 235, (256, 256, 3), dtype=np.uint8)
    bits = rng.integers(0, 2, 700, dtype=np.uint8)
    cfg = EmbedConfig(block=8, delta=40.0, repeat=3, key=21)
    result = steg.embed_bits(cover, bits, cfg)
    assert np.array_equal(bits, steg.extract_bits(result.stego, len(bits), cfg))


def test_hide_reveal_status():
    cover = rng.integers(20, 235, (512, 512, 3), dtype=np.uint8)
    secret = dataset.default_secret(64)
    cfg = EmbedConfig(block=8, delta=24.0, repeat=1)
    result, _ = steg.hide_secret(cover, secret, 10, cfg)
    recovered, status = steg.reveal_secret(result.stego, cfg, return_status=True)
    assert status.header_ok and status.crc_ok
    assert recovered.shape == secret.shape


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
