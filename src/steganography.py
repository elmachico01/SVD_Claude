"""
steganography.py
================
The SVD-based steganography engine.

The scheme has two SVD stages, both grounded in the course material:

1. **Secret compression — truncated SVD (Eckart–Young).**
   The secret image S is replaced by its best rank-``k`` approximation and
   stored as the compact factors (U_k, σ_k, V_kᵀ).  This *dimensionality
   reduction* shrinks the payload from m·n numbers to k(m+n+1), exactly the
   compression accounting of the lecture, so that a sizeable secret fits inside
   a single cover image.

2. **Cover embedding — block-wise SVD + QIM.**
   The cover is split into B×B blocks.  Each carrier block A = UΣVᵀ has its
   largest singular value σ₁ quantised (Quantisation Index Modulation) to carry
   one payload bit.  σ₁ concentrates almost all of the block energy
   (Eckart–Young), so it is the most stable place to hide information and the
   change ‖δ·u₁v₁ᵀ‖_F = |δ| is spread over the whole block — imperceptible per
   pixel.

The order in which blocks are filled is decided by a *priority map* (see
``yolo_guidance``): background blocks first, salient/object blocks last.  This
keeps the visually-important and semantically-important regions almost
untouched.  A pseudo-random key breaks ties and provides a keyed permutation
fallback that needs no detector at extraction time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from . import svd_core

MAGIC = b"SV"
VERSION = 1
HEADER_BYTES = 9                 # magic(2) + version(1) + h(2) + w(2) + k(2)
HEADER_BITS = HEADER_BYTES * 8


# ===========================================================================
# Secret codec  (stage 1: truncated SVD compression + serialisation)
# ===========================================================================
def compress_secret(secret_gray: np.ndarray, k: int) -> bytes:
    """
    Compress a grayscale secret image into a self-describing byte payload using
    its rank-``k`` truncated SVD.

    Layout (fixed 9-byte self-describing header, no length prefix):
        magic(2) | version(1) | h(2) | w(2) | k(2) |
        σ_k  : k × float16 |
        U_k  : h·k × int8  (entries scaled by 127) |
        V_kᵀ : k·w × int8
    The payload length is fully determined by (h, w, k), so the decoder reads
    the header first and then exactly the right number of bits.
    """
    S = np.asarray(secret_gray, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError("secret must be a 2-D grayscale image")
    h, w = S.shape
    k = int(np.clip(k, 1, min(h, w)))

    svd = svd_core.svd_decompose(S)
    Uk, sk, Vtk = svd_core.low_rank_components(svd, k)

    Uq = np.clip(np.round(Uk * 127), -127, 127).astype(np.int8)
    Vq = np.clip(np.round(Vtk * 127), -127, 127).astype(np.int8)
    sq = sk.astype(np.float16)

    body = bytearray()
    body += MAGIC
    body += bytes([VERSION])
    body += int(h).to_bytes(2, "big")
    body += int(w).to_bytes(2, "big")
    body += int(k).to_bytes(2, "big")
    body += sq.tobytes()
    body += Uq.tobytes()
    body += Vq.tobytes()
    return bytes(body)


def payload_size_bytes(h: int, w: int, k: int) -> int:
    """Total payload size for a rank-k secret of size h×w (header included)."""
    return HEADER_BYTES + 2 * k + h * k + k * w


def decompress_secret(payload: bytes) -> np.ndarray:
    """
    Inverse of :func:`compress_secret` → recovered grayscale image (uint8).

    The parser is defensive: if the payload is corrupted (e.g. after an attack)
    it validates the header and pads/truncates the body so that it never raises
    on a size mismatch (it raises only on an implausible/garbage header, which
    the caller is expected to catch).
    """
    body = bytes(payload)
    if body[:2] != MAGIC:
        raise ValueError("bad magic — payload corrupted")
    off = 3  # magic(2) + version(1)
    h = int.from_bytes(body[off:off + 2], "big"); off += 2
    w = int.from_bytes(body[off:off + 2], "big"); off += 2
    k = int.from_bytes(body[off:off + 2], "big"); off += 2
    if not (0 < h <= 1024 and 0 < w <= 1024 and 0 < k <= min(h, w)):
        raise ValueError(f"implausible header h={h} w={w} k={k}")

    need = 2 * k + h * k + k * w
    chunk = body[off:off + need]
    if len(chunk) < need:                      # pad if the payload was truncated
        chunk = chunk + b"\x00" * (need - len(chunk))

    sk = np.frombuffer(chunk[:2 * k], dtype=np.float16).astype(np.float64)
    sk = np.nan_to_num(sk, nan=0.0, posinf=0.0, neginf=0.0)   # corrupted σ → 0
    p = 2 * k
    Uq = np.frombuffer(chunk[p:p + h * k], dtype=np.int8).astype(np.float64).reshape(h, k)
    p += h * k
    Vq = np.frombuffer(chunk[p:p + k * w], dtype=np.int8).astype(np.float64).reshape(k, w)

    recon = (Uq / 127.0 * sk) @ (Vq / 127.0)
    recon = np.nan_to_num(recon, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(np.round(recon), 0, 255).astype(np.uint8)


def secret_compression_report(secret_gray: np.ndarray, k: int) -> dict:
    """Compression-factor and reconstruction quality of the rank-k secret."""
    S = np.asarray(secret_gray, dtype=np.float64)
    h, w = S.shape
    payload = compress_secret(S, k)
    recon = decompress_secret(payload)
    store = svd_core.storage_floats(h, w, k)
    svd = svd_core.svd_decompose(S)
    return {
        "k": int(k),
        "payload_bytes": len(payload),
        "compression_factor": svd_core.compression_factor(h, w, k),
        "energy_ratio": svd_core.energy_ratio(svd, k),
        "store_full_floats": store["full"],
        "store_rank_k_floats": store["rank_k"],
        "recon_psnr": _psnr(S, recon),
    }


def _psnr(a, b):
    from .metrics import psnr
    return psnr(a, b)


# ===========================================================================
# Bit helpers
# ===========================================================================
def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


# ===========================================================================
# Block-SVD QIM embedding  (stage 2)
# ===========================================================================
@dataclass
class EmbedConfig:
    block: int = 8                       # block side B
    delta: float = 40.0                  # QIM quantisation step Δ
    channels: tuple[int, ...] = (0, 1, 2)  # cover channels used as carriers
    key: int = 2024                      # PRNG key for the (tie-break) permutation
    repeat: int = 1                      # repetition factor (majority-vote ECC)


@dataclass
class EmbedResult:
    stego: np.ndarray                    # uint8 (H, W, 3)
    num_bits: int                        # bits actually written
    order: np.ndarray = field(repr=False)  # ordered carrier-site indices used
    capacity_bits: int = 0


def _site_grid(shape, block, channels):
    """All candidate carrier sites as an array of (channel, by, bx)."""
    H, W = shape[:2]
    nby, nbx = H // block, W // block
    sites = [(c, by, bx) for c in channels for by in range(nby) for bx in range(nbx)]
    return np.array(sites, dtype=np.int64), (nby, nbx)


def embedding_order(shape, cfg: EmbedConfig,
                    priority_map: np.ndarray | None = None) -> np.ndarray:
    """
    Deterministic ordering of carrier sites, identical at embed and extract.

    Sorting key = (priority value, keyed-random tie-break).  With
    ``priority_map`` low values are filled first (background → objects);
    without it the order is a pure keyed permutation (blind, detector-free).
    """
    sites, (nby, nbx) = _site_grid(shape, cfg.block, cfg.channels)
    rng = np.random.default_rng(cfg.key)
    tie = rng.permutation(len(sites)).astype(np.float64)

    if priority_map is None:
        primary = tie
        order = np.argsort(primary, kind="stable")
    else:
        pm = np.asarray(priority_map, dtype=np.float64)
        if pm.shape != (nby, nbx):
            # resize priority map to the block grid
            import cv2
            pm = cv2.resize(pm, (nbx, nby), interpolation=cv2.INTER_AREA)
        prio = pm[sites[:, 1], sites[:, 2]]
        # stable lexsort: primary = priority, secondary = keyed tie-break
        order = np.lexsort((tie, prio))
    return sites[order]


def _qim_embed_sigma1(block: np.ndarray, bit: int, delta: float) -> np.ndarray:
    svd = svd_core.svd_decompose(block)
    s = svd.s.copy()
    q = np.floor(s[0] / delta)
    s[0] = q * delta + (0.75 if bit else 0.25) * delta
    return (svd.U * s) @ svd.Vt


def _qim_read_sigma1(block: np.ndarray, delta: float) -> int:
    s0 = svd_core.svd_decompose(block).s[0]
    frac = s0 / delta - np.floor(s0 / delta)
    return int(frac >= 0.5)


def embed_bits(cover: np.ndarray, bits: np.ndarray, cfg: EmbedConfig,
               priority_map: np.ndarray | None = None) -> EmbedResult:
    """
    Embed a *logical* bit array into the cover.

    With ``cfg.repeat = R`` each logical bit is written into R consecutive
    carrier sites (repetition code); extraction recovers it by majority vote.
    """
    cover = np.asarray(cover)
    B = cfg.block
    order = embedding_order(cover.shape, cfg, priority_map)
    logical = np.asarray(bits, dtype=np.uint8).ravel()
    phys = np.repeat(logical, cfg.repeat)            # repetition coding
    if len(phys) > len(order):
        raise ValueError(f"payload {len(phys)} physical bits exceeds capacity "
                         f"{len(order)} bits (logical={len(logical)}, R={cfg.repeat})")

    work = cover.astype(np.float64).copy()
    for i, bit in enumerate(phys):
        c, by, bx = order[i]
        y0, x0 = by * B, bx * B
        blk = work[y0:y0 + B, x0:x0 + B, c]
        work[y0:y0 + B, x0:x0 + B, c] = _qim_embed_sigma1(blk, int(bit), cfg.delta)

    stego = np.clip(np.round(work), 0, 255).astype(np.uint8)
    return EmbedResult(stego=stego, num_bits=len(logical),
                       order=order, capacity_bits=len(order) // cfg.repeat)


def extract_bits(stego: np.ndarray, num_bits: int, cfg: EmbedConfig,
                 priority_map: np.ndarray | None = None) -> np.ndarray:
    """Read ``num_bits`` *logical* bits using the same ordering + majority vote."""
    stego = np.asarray(stego)
    B = cfg.block
    R = cfg.repeat
    order = embedding_order(stego.shape, cfg, priority_map)
    work = stego.astype(np.float64)
    phys = np.empty(num_bits * R, dtype=np.uint8)
    for i in range(num_bits * R):
        c, by, bx = order[i]
        y0, x0 = by * B, bx * B
        blk = work[y0:y0 + B, x0:x0 + B, c]
        phys[i] = _qim_read_sigma1(blk, cfg.delta)
    if R == 1:
        return phys
    votes = phys.reshape(num_bits, R).sum(axis=1)
    return (votes > (R / 2.0)).astype(np.uint8)       # ties → 0


# ===========================================================================
# High-level API: hide / reveal a secret image
# ===========================================================================
def capacity_bits(shape, cfg: EmbedConfig) -> int:
    """Number of *logical* bits that can be stored (physical sites // repeat)."""
    H, W = shape[:2]
    physical = len(cfg.channels) * (H // cfg.block) * (W // cfg.block)
    return physical // cfg.repeat


def hide_secret(cover: np.ndarray, secret_gray: np.ndarray, k: int,
                cfg: EmbedConfig,
                priority_map: np.ndarray | None = None) -> tuple[EmbedResult, bytes]:
    """Compress the secret (truncated SVD) then embed it into the cover."""
    payload = compress_secret(secret_gray, k)
    bits = bytes_to_bits(payload)
    cap = capacity_bits(cover.shape, cfg)
    if len(bits) > cap:
        raise ValueError(
            f"secret needs {len(bits)} bits but cover capacity is {cap} bits; "
            f"reduce k, enlarge the cover, or use more channels")
    res = embed_bits(cover, bits, cfg, priority_map)
    return res, payload


def reveal_secret(stego: np.ndarray, cfg: EmbedConfig,
                  priority_map: np.ndarray | None = None) -> np.ndarray:
    """Blindly recover the secret image from a (possibly attacked) stego image."""
    # 1) read the fixed 9-byte header to learn (h, w, k)
    header_bits = extract_bits(stego, HEADER_BITS, cfg, priority_map)
    header = bits_to_bytes(header_bits)
    if header[:2] != MAGIC:
        raise ValueError("header magic not found — wrong key/params or corruption")
    h = int.from_bytes(header[3:5], "big")
    w = int.from_bytes(header[5:7], "big")
    k = int.from_bytes(header[7:9], "big")
    if not (0 < h <= 1024 and 0 < w <= 1024 and 0 < k <= min(h, w)):
        raise ValueError(f"implausible header h={h} w={w} k={k}")
    # 2) read exactly the payload the header describes, then decode
    total_bits = min(payload_size_bytes(h, w, k) * 8, capacity_bits(stego.shape, cfg) * 1)
    all_bits = extract_bits(stego, total_bits, cfg, priority_map)
    return decompress_secret(bits_to_bytes(all_bits))
