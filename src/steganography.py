"""SVD-based image steganography.

The module separates two ideas:
1. truncated-SVD compression of a grayscale secret;
2. block-SVD QIM embedding of the resulting bitstream.

Two extraction modes are deliberately distinguished:
- keyed/blind: the carrier order depends only on ``key``;
- guided: the carrier order also depends on a priority map, which is side
  information and must be available unchanged at extraction time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import zlib
import numpy as np

from . import svd_core

MAGIC = b"SV"
VERSION = 2
# magic(2) + version(1) + h(2) + w(2) + k(2) + payload_crc32(4)
HEADER_BYTES = 13
HEADER_BITS = HEADER_BYTES * 8


@dataclass
class DecodeStatus:
    header_ok: bool
    crc_ok: bool
    message: str = ""


# ---------------------------------------------------------------------------
# Secret codec
# ---------------------------------------------------------------------------
def _payload_body(secret_gray: np.ndarray, k: int) -> tuple[bytes, int, int, int]:
    S = np.asarray(secret_gray, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError("secret must be a 2-D grayscale image")
    h, w = S.shape
    k = int(np.clip(k, 1, min(h, w)))
    svd = svd_core.svd_decompose(S)
    Uk, sk, Vtk = svd_core.low_rank_components(svd, k)
    Uq = np.clip(np.rint(Uk * 127.0), -127, 127).astype(np.int8)
    Vq = np.clip(np.rint(Vtk * 127.0), -127, 127).astype(np.int8)
    body = sk.astype("<f2").tobytes() + Uq.tobytes() + Vq.tobytes()
    return body, h, w, k


def compress_secret(secret_gray: np.ndarray, k: int) -> bytes:
    """Serialize a rank-k approximation using int8 factors and float16 values."""
    body, h, w, k = _payload_body(secret_gray, k)
    crc = zlib.crc32(body) & 0xFFFFFFFF
    header = (MAGIC + bytes([VERSION]) + h.to_bytes(2, "big")
              + w.to_bytes(2, "big") + k.to_bytes(2, "big")
              + crc.to_bytes(4, "big"))
    return header + body


def payload_size_bytes(h: int, w: int, k: int) -> int:
    return HEADER_BYTES + 2 * k + h * k + k * w


def parse_header(payload: bytes) -> tuple[int, int, int, int]:
    if len(payload) < HEADER_BYTES or payload[:2] != MAGIC:
        raise ValueError("header magic not found")
    if payload[2] != VERSION:
        raise ValueError(f"unsupported payload version {payload[2]}")
    h = int.from_bytes(payload[3:5], "big")
    w = int.from_bytes(payload[5:7], "big")
    k = int.from_bytes(payload[7:9], "big")
    crc = int.from_bytes(payload[9:13], "big")
    if not (0 < h <= 1024 and 0 < w <= 1024 and 0 < k <= min(h, w)):
        raise ValueError(f"implausible header h={h} w={w} k={k}")
    return h, w, k, crc


def decompress_secret(payload: bytes, verify_crc: bool = True) -> np.ndarray:
    h, w, k, expected_crc = parse_header(payload)
    need = 2 * k + h * k + k * w
    body = bytes(payload[HEADER_BYTES:HEADER_BYTES + need])
    if len(body) != need:
        raise ValueError("truncated payload")
    if verify_crc and (zlib.crc32(body) & 0xFFFFFFFF) != expected_crc:
        raise ValueError("payload CRC mismatch")
    sk = np.frombuffer(body[:2 * k], dtype="<f2").astype(np.float64)
    p = 2 * k
    Uq = np.frombuffer(body[p:p + h * k], dtype=np.int8).astype(np.float64).reshape(h, k)
    p += h * k
    Vq = np.frombuffer(body[p:p + k * w], dtype=np.int8).astype(np.float64).reshape(k, w)
    recon = (Uq / 127.0 * sk) @ (Vq / 127.0)
    return np.clip(np.rint(np.nan_to_num(recon)), 0, 255).astype(np.uint8)


def secret_compression_report(secret_gray: np.ndarray, k: int) -> dict:
    from .metrics import psnr, normalized_correlation
    S = np.asarray(secret_gray, dtype=np.float64)
    payload = compress_secret(S, k)
    rec = decompress_secret(payload)
    h, w = S.shape
    svd = svd_core.svd_decompose(S)
    return {
        "k": int(k), "payload_bytes": len(payload),
        "compression_factor": svd_core.compression_factor(h, w, k),
        "energy_ratio": svd_core.energy_ratio(svd, k),
        "recon_psnr": psnr(S, rec),
        "recon_nc": normalized_correlation(S, rec),
    }


# ---------------------------------------------------------------------------
# Bit helpers
# ---------------------------------------------------------------------------
def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    pad = (-len(bits)) % 8
    if pad:
        bits = np.pad(bits, (0, pad))
    return np.packbits(bits).tobytes()


# ---------------------------------------------------------------------------
# Block-QIM
# ---------------------------------------------------------------------------
@dataclass
class EmbedConfig:
    block: int = 8
    delta: float = 40.0
    channels: tuple[int, ...] = (0, 1, 2)
    key: int = 2024
    repeat: int = 1
    verify_after_quantization: bool = True

    def __post_init__(self):
        if self.block <= 0 or self.delta <= 0:
            raise ValueError("block and delta must be positive")
        if self.repeat <= 0 or self.repeat % 2 == 0:
            raise ValueError("repeat must be a positive odd integer")


@dataclass
class EmbedResult:
    stego: np.ndarray
    num_bits: int
    order: np.ndarray = field(repr=False)
    capacity_bits: int = 0
    corrected_sites: int = 0


def _site_grid(shape, block, channels):
    H, W = shape[:2]
    nby, nbx = H // block, W // block
    sites = [(c, by, bx) for c in channels for by in range(nby) for bx in range(nbx)]
    return np.asarray(sites, dtype=np.int64), (nby, nbx)


def embedding_order(shape, cfg: EmbedConfig, priority_map: np.ndarray | None = None) -> np.ndarray:
    sites, (nby, nbx) = _site_grid(shape, cfg.block, cfg.channels)
    rng = np.random.default_rng(cfg.key)
    tie = rng.random(len(sites))
    if priority_map is None:
        return sites[np.argsort(tie, kind="stable")]
    import cv2
    pm = np.asarray(priority_map, dtype=np.float64)
    if pm.shape != (nby, nbx):
        pm = cv2.resize(pm, (nbx, nby), interpolation=cv2.INTER_AREA)
    prio = pm[sites[:, 1], sites[:, 2]]
    return sites[np.lexsort((tie, prio))]


def _nearest_qim_target(sigma: float, bit: int, delta: float) -> float:
    """Nearest point in the QIM coset b (minimum-distortion quantizer)."""
    offset = (0.25 if bit == 0 else 0.75) * delta
    q = max(0, int(np.rint((sigma - offset) / delta)))
    return q * delta + offset


def _qim_read_sigma1(block: np.ndarray, delta: float) -> int:
    s0 = svd_core.svd_decompose(block).s[0]
    return int((s0 / delta - np.floor(s0 / delta)) >= 0.5)


def _qim_embed_sigma1(block: np.ndarray, bit: int, delta: float,
                      verify: bool = True) -> tuple[np.ndarray, bool]:
    svd = svd_core.svd_decompose(block)
    s = svd.s.copy()
    s[0] = _nearest_qim_target(float(s[0]), bit, delta)
    candidate = (svd.U * s) @ svd.Vt
    quantized = np.clip(np.rint(candidate), 0, 255).astype(np.uint8)
    if not verify or _qim_read_sigma1(quantized, delta) == bit:
        return quantized, False
    # Rare clipping/rounding failure: move one lattice step deeper in the same coset.
    s[0] += delta
    candidate = (svd.U * s) @ svd.Vt
    quantized = np.clip(np.rint(candidate), 0, 255).astype(np.uint8)
    return quantized, True


def _physical_order(order: np.ndarray, repeat: int, key: int) -> np.ndarray:
    """Interleave repetitions so replicas are not adjacent/correlated sites."""
    if repeat == 1:
        return order
    n = len(order) // repeat
    rng = np.random.default_rng(key + 991)
    chunks = []
    for r in range(repeat):
        chunk = order[r * n:(r + 1) * n].copy()
        rng.shuffle(chunk)
        chunks.append(chunk)
    return np.stack(chunks, axis=1).reshape(-1, 3)


def capacity_bits(shape, cfg: EmbedConfig) -> int:
    physical = len(cfg.channels) * (shape[0] // cfg.block) * (shape[1] // cfg.block)
    return physical // cfg.repeat


def embed_bits(cover: np.ndarray, bits: np.ndarray, cfg: EmbedConfig,
               priority_map: np.ndarray | None = None) -> EmbedResult:
    logical = np.asarray(bits, dtype=np.uint8).ravel()
    base_order = embedding_order(cover.shape, cfg, priority_map)
    cap = len(base_order) // cfg.repeat
    if len(logical) > cap:
        raise ValueError(f"payload {len(logical)} bits exceeds capacity {cap}")
    order = _physical_order(base_order, cfg.repeat, cfg.key)
    work = np.asarray(cover, dtype=np.uint8).copy()
    corrected = 0
    B = cfg.block
    for i, bit in enumerate(np.repeat(logical, cfg.repeat)):
        c, by, bx = order[i]
        y0, x0 = by * B, bx * B
        qblk, was_corrected = _qim_embed_sigma1(
            work[y0:y0+B, x0:x0+B, c], int(bit), cfg.delta,
            cfg.verify_after_quantization)
        work[y0:y0+B, x0:x0+B, c] = qblk
        corrected += int(was_corrected)
    return EmbedResult(work, len(logical), order, cap, corrected)


def extract_bits(stego: np.ndarray, num_bits: int, cfg: EmbedConfig,
                 priority_map: np.ndarray | None = None) -> np.ndarray:
    base_order = embedding_order(stego.shape, cfg, priority_map)
    if num_bits > len(base_order) // cfg.repeat:
        raise ValueError("requested bit count exceeds capacity")
    order = _physical_order(base_order, cfg.repeat, cfg.key)
    B = cfg.block
    phys = np.empty(num_bits * cfg.repeat, dtype=np.uint8)
    for i in range(len(phys)):
        c, by, bx = order[i]
        y0, x0 = by * B, bx * B
        phys[i] = _qim_read_sigma1(stego[y0:y0+B, x0:x0+B, c], cfg.delta)
    if cfg.repeat == 1:
        return phys
    return (phys.reshape(num_bits, cfg.repeat).sum(axis=1) > cfg.repeat / 2).astype(np.uint8)


def hide_secret(cover: np.ndarray, secret_gray: np.ndarray, k: int,
                cfg: EmbedConfig, priority_map: np.ndarray | None = None) -> tuple[EmbedResult, bytes]:
    payload = compress_secret(secret_gray, k)
    bits = bytes_to_bits(payload)
    return embed_bits(cover, bits, cfg, priority_map), payload


def reveal_secret(stego: np.ndarray, cfg: EmbedConfig,
                  priority_map: np.ndarray | None = None,
                  return_status: bool = False):
    """Recover the secret. A priority map, when supplied, is side information."""
    try:
        header = bits_to_bytes(extract_bits(stego, HEADER_BITS, cfg, priority_map))
        h, w, k, _ = parse_header(header)
        total_bits = payload_size_bytes(h, w, k) * 8
        payload = bits_to_bytes(extract_bits(stego, total_bits, cfg, priority_map))
        rec = decompress_secret(payload, verify_crc=True)
        status = DecodeStatus(True, True, "ok")
    except Exception as exc:
        status = DecodeStatus(False, False, str(exc))
        if not return_status:
            raise
        rec = None
    return (rec, status) if return_status else rec
