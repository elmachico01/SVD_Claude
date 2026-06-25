"""
svd_core.py
===========
Singular Value Decomposition: the mathematical engine of the project.

Every routine here maps directly onto a slide of the course lecture
*"Reducing data dimensionality for AI applications"* (M. Popolizio):

    * SVD definition            A = U Σ Vᵀ
    * Truncated SVD             A_k = Σ_{i=1}^k σ_i u_i v_iᵀ
    * Eckart–Young theorem      ‖A − A_k‖₂ = σ_{k+1},  ‖A − A_k‖_F = (Σ_{i>k} σ_i²)^½
    * Compression footprint     k(m + n + 1)  vs  m·n
    * SVD ↔ eigendecomposition  AᵀA = VΛVᵀ,  σ_i = √λ_i
    * Condition number          κ(A) = σ₁ / σ_r
    * Moore–Penrose pseudoinverse  A⁺ = V Σ⁺ Uᵀ  (least-squares / min-norm)

The functions are intentionally written on top of ``numpy.linalg`` so that the
*mathematics* is explicit and auditable, rather than hidden behind a black box.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# 1. The decomposition
# ---------------------------------------------------------------------------
@dataclass
class SVDResult:
    """Container for a (thin) SVD  A = U Σ Vᵀ."""
    U: np.ndarray          # (m, r)  left singular vectors
    s: np.ndarray          # (r,)    singular values, σ₁ ≥ … ≥ σ_r ≥ 0
    Vt: np.ndarray         # (r, n)  right singular vectors (transposed)

    @property
    def rank(self) -> int:
        """Numerical rank = number of singular values above a tiny tolerance."""
        tol = self.s.max() * max(self.U.shape[0], self.Vt.shape[1]) * np.finfo(float).eps
        return int(np.sum(self.s > tol))

    def reconstruct(self, k: int | None = None) -> np.ndarray:
        """Rebuild the matrix from the first ``k`` triplets (all if ``k`` is None)."""
        if k is None:
            k = len(self.s)
        k = int(np.clip(k, 0, len(self.s)))
        return (self.U[:, :k] * self.s[:k]) @ self.Vt[:k, :]


def svd_decompose(A: np.ndarray) -> SVDResult:
    """Compute the thin SVD ``A = U Σ Vᵀ`` (slide *SVD: Definition*)."""
    A = np.asarray(A, dtype=np.float64)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return SVDResult(U=U, s=s, Vt=Vt)


# ---------------------------------------------------------------------------
# 2. Truncated SVD / best rank-k approximation  (Eckart–Young)
# ---------------------------------------------------------------------------
def truncated_svd(A: np.ndarray, k: int) -> tuple[np.ndarray, SVDResult]:
    """
    Return the best rank-``k`` approximation ``A_k`` and the full SVD.

    By the Eckart–Young theorem ``A_k`` is the optimal rank-k matrix in both
    the spectral and Frobenius norms.
    """
    svd = svd_decompose(A)
    return svd.reconstruct(k), svd


def low_rank_components(svd: SVDResult, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the truncated factors (U_k, σ_k, V_kᵀ) — the *stored* quantities."""
    k = int(np.clip(k, 0, len(svd.s)))
    return svd.U[:, :k].copy(), svd.s[:k].copy(), svd.Vt[:k, :].copy()


def eckart_young_errors(svd: SVDResult, k: int) -> dict[str, float]:
    """
    Theoretical vs. measured approximation error for rank ``k``.

    Theory (Eckart–Young):
        ‖A − A_k‖₂ = σ_{k+1}
        ‖A − A_k‖_F = sqrt(Σ_{i>k} σ_i²)
    """
    s = svd.s
    k = int(np.clip(k, 0, len(s)))
    spectral = float(s[k]) if k < len(s) else 0.0
    frob = float(np.sqrt(np.sum(s[k:] ** 2)))
    return {"spectral_sigma_k1": spectral, "frobenius_tail": frob}


def energy_ratio(svd: SVDResult, k: int) -> float:
    """
    Fraction of squared-Frobenius energy retained by the first k components:
        E(k) = Σ_{i≤k} σ_i² / Σ_i σ_i².
    This is the natural way to *choose* k from the singular-value spectrum.
    """
    s2 = svd.s ** 2
    total = float(np.sum(s2))
    if total == 0.0:
        return 1.0
    k = int(np.clip(k, 0, len(s2)))
    return float(np.sum(s2[:k]) / total)


def rank_for_energy(svd: SVDResult, target: float = 0.95) -> int:
    """Smallest k such that the retained energy ratio ≥ ``target``."""
    s2 = svd.s ** 2
    csum = np.cumsum(s2) / np.sum(s2)
    return int(np.searchsorted(csum, target) + 1)


# ---------------------------------------------------------------------------
# 3. Storage / compression accounting (slide *Reduction of memory occupation*)
# ---------------------------------------------------------------------------
def compression_factor(m: int, n: int, k: int) -> float:
    """
    Ratio between the rank-k footprint k(m+n+1) and the full footprint m·n.

    A value of, e.g., 0.18 means the compressed object needs ~18 % of the
    original memory (the exact example given on the slides).
    """
    return k * (m + n + 1) / (m * n)


def storage_floats(m: int, n: int, k: int) -> dict[str, int]:
    """Number of stored scalars for the full matrix vs. its rank-k SVD."""
    return {"full": m * n, "rank_k": k * (m + n + 1)}


# ---------------------------------------------------------------------------
# 4. SVD ↔ eigendecomposition  (slide *SVD and Eigen-Decomposition*)
# ---------------------------------------------------------------------------
def verify_svd_eigendecomposition(A: np.ndarray) -> dict[str, float]:
    """
    Numerically verify  σ_i = √λ_i(AᵀA)  and that V diagonalises AᵀA.

    Returns the max absolute discrepancies (≈ machine precision when correct).
    """
    A = np.asarray(A, dtype=np.float64)
    svd = svd_decompose(A)
    # Eigendecomposition of the (smaller) Gram matrix.
    gram = A.T @ A if A.shape[1] <= A.shape[0] else A @ A.T
    eigvals = np.linalg.eigvalsh(gram)             # ascending
    eig_sigma = np.sqrt(np.clip(eigvals[::-1], 0, None))
    r = min(len(svd.s), len(eig_sigma))
    sigma_err = float(np.max(np.abs(svd.s[:r] - eig_sigma[:r])))
    # Reconstruction error of A = UΣVᵀ.
    recon_err = float(np.linalg.norm(A - svd.reconstruct()) / (np.linalg.norm(A) + 1e-12))
    return {"sigma_vs_sqrt_lambda": sigma_err, "reconstruction_rel_err": recon_err}


# ---------------------------------------------------------------------------
# 5. Conditioning & pseudoinverse (slides 17–24, rectangular systems)
# ---------------------------------------------------------------------------
def condition_number(A: np.ndarray) -> float:
    """κ(A) = σ₁ / σ_r over the *non-zero* singular values."""
    s = svd_decompose(A).s
    s = s[s > s.max() * 1e-12] if s.size and s.max() > 0 else s
    if s.size == 0:
        return np.inf
    return float(s[0] / s[-1])


def pseudoinverse(A: np.ndarray, k: int | None = None) -> np.ndarray:
    """
    Moore–Penrose pseudoinverse  A⁺ = V Σ⁺ Uᵀ.

    If ``k`` is given, a *truncated* pseudoinverse (TSVD regularisation) is
    returned, discarding the smallest singular values to stabilise the solution
    of ill-conditioned least-squares problems (slide *Overdetermined Systems
    and Conditioning*).
    """
    svd = svd_decompose(A)
    s = svd.s
    if k is not None:
        k = int(np.clip(k, 0, len(s)))
        s = np.concatenate([s[:k], np.zeros(len(s) - k)])
    s_inv = np.array([1.0 / x if x > s.max() * 1e-12 else 0.0 for x in s])
    return (svd.Vt.T * s_inv) @ svd.U.T


def lstsq_via_svd(A: np.ndarray, b: np.ndarray, k: int | None = None) -> np.ndarray:
    """Least-squares / minimum-norm solution  x* = A⁺ b  (slides 20–24)."""
    return pseudoinverse(A, k=k) @ np.asarray(b, dtype=np.float64)
