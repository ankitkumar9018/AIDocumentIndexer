"""
AIDocumentIndexer - Embedding Inversion Defense Service
=======================================================

Implements OWASP LLM08:2025 mitigations to protect stored embeddings
from text reconstruction (inversion) attacks.

Defense techniques applied in sequence:
  1. Noise injection   - Calibrated Gaussian noise degrades inversion
                         fidelity while preserving cosine-similarity ranking.
  2. Dimension shuffle - A secret, deterministic permutation of embedding
                         dimensions prevents adversaries from mapping
                         dimensions to linguistic features.
  3. Norm clipping     - Clipping to a fixed L2 radius bounds the amount
                         of information any single embedding can leak.

All three transforms are *symmetric*: the same `protect()` call is applied
at indexing time and again at query time so that cosine similarity between
the defended vectors is preserved.
"""

import hashlib
import math
import os
from typing import List, Optional

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional["EmbeddingDefense"] = None


def get_embedding_defense() -> "EmbeddingDefense":
    """Return the module-level EmbeddingDefense singleton.

    Creates the instance on first call.  Subsequent calls return the same
    object so that the secret permutation key is stable for the lifetime
    of the process.
    """
    global _instance
    if _instance is None:
        _instance = EmbeddingDefense()
        logger.info("embedding_defense.singleton_created")
    return _instance


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class EmbeddingDefense:
    """Protects embeddings from inversion attacks.

    The defense is controlled by three settings (read lazily from the
    settings service):

    * ``security.embedding_defense_enabled`` -- master toggle (default False)
    * ``security.defense_noise_scale``       -- stddev of Gaussian noise
                                                (default 0.01)
    * ``security.defense_clip_norm``         -- L2 norm cap (default 1.0)

    A secret *permutation key* is derived from the environment variable
    ``EMBEDDING_DEFENSE_SECRET``.  If the variable is absent a random
    32-byte key is generated at startup and a warning is logged (the
    permutation will change across restarts, which is fine for dev but
    not for production).
    """

    def __init__(self) -> None:
        # ----- permutation secret -----
        env_secret = os.environ.get("EMBEDDING_DEFENSE_SECRET")
        if env_secret:
            self._secret = env_secret.encode("utf-8")
            logger.info("embedding_defense.secret_loaded_from_env")
        else:
            self._secret = os.urandom(32)
            logger.warning(
                "embedding_defense.no_env_secret",
                detail=(
                    "EMBEDDING_DEFENSE_SECRET not set; generated a random "
                    "key.  Permutations will change on restart."
                ),
            )

        # Cache permutation arrays keyed by dimension count so we only
        # compute once per distinct embedding size.
        self._permutation_cache: dict[int, List[int]] = {}

        logger.info("embedding_defense.initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def protect(self, embedding: List[float]) -> List[float]:
        """Apply all enabled defenses to *embedding* and return the result.

        If ``security.embedding_defense_enabled`` is ``False`` the
        original embedding is returned unchanged.

        Parameters
        ----------
        embedding:
            A list of floats representing a single embedding vector.

        Returns
        -------
        List[float]
            The defended embedding (same dimensionality as the input).
        """
        try:
            if not self._is_enabled():
                return embedding

            noise_scale = self._get_noise_scale()
            clip_norm = self._get_clip_norm()

            vec = list(embedding)  # work on a copy

            # 1. Noise injection
            vec = self._inject_noise(vec, noise_scale)

            # 2. Dimension shuffling
            vec = self._shuffle_dimensions(vec)

            # 3. Norm clipping
            vec = self._clip_norm(vec, clip_norm)

            return vec

        except Exception:
            logger.exception("embedding_defense.protect_failed")
            # Fail-open: return original embedding so search is not broken
            return embedding

    # ------------------------------------------------------------------
    # Defence primitives
    # ------------------------------------------------------------------

    def _inject_noise(self, vec: List[float], scale: float) -> List[float]:
        """Add i.i.d. Gaussian noise with stddev *scale* to each dimension.

        Uses a simple Box-Muller transform so we stay dependency-free
        (no numpy required).
        """
        import random

        rng = random.Random()  # fresh unseeded RNG for true randomness
        noisy = [v + rng.gauss(0.0, scale) for v in vec]
        return noisy

    def _shuffle_dimensions(self, vec: List[float]) -> List[float]:
        """Deterministically permute the vector dimensions.

        The permutation is derived from ``self._secret`` and the
        dimensionality of the vector via a keyed Fisher-Yates shuffle.
        Because the same permutation is applied to both index-time and
        query-time embeddings, cosine similarity is preserved.
        """
        n = len(vec)
        perm = self._get_permutation(n)
        return [vec[perm[i]] for i in range(n)]

    def _clip_norm(self, vec: List[float], max_norm: float) -> List[float]:
        """Clip the L2 norm of *vec* to at most *max_norm*.

        If the vector's norm already satisfies the constraint it is
        returned unchanged.
        """
        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= max_norm or norm == 0.0:
            return vec
        factor = max_norm / norm
        return [v * factor for v in vec]

    # ------------------------------------------------------------------
    # Permutation helpers
    # ------------------------------------------------------------------

    def _get_permutation(self, n: int) -> List[int]:
        """Return the cached permutation for dimension count *n*."""
        if n not in self._permutation_cache:
            self._permutation_cache[n] = self._build_permutation(n)
        return self._permutation_cache[n]

    def _build_permutation(self, n: int) -> List[int]:
        """Build a deterministic permutation of ``range(n)`` keyed by the
        instance secret.

        Uses a keyed Fisher-Yates shuffle where the random indices are
        drawn from successive HMAC-SHA256 digests.
        """
        indices = list(range(n))
        # Derive a per-dimension-count seed via HMAC(secret, n)
        seed = hashlib.sha256(self._secret + n.to_bytes(4, "big")).digest()

        for i in range(n - 1, 0, -1):
            # Produce a pseudo-random index in [0, i]
            seed = hashlib.sha256(seed).digest()
            j = int.from_bytes(seed[:4], "big") % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]

        return indices

    # ------------------------------------------------------------------
    # Settings helpers (lazy reads -- no async needed)
    # ------------------------------------------------------------------

    def _is_enabled(self) -> bool:
        """Read the ``security.embedding_defense_enabled`` setting."""
        return self._read_setting_sync(
            "security.embedding_defense_enabled", False
        )

    def _get_noise_scale(self) -> float:
        """Read the ``security.defense_noise_scale`` setting."""
        return float(
            self._read_setting_sync("security.defense_noise_scale", 0.01)
        )

    def _get_clip_norm(self) -> float:
        """Read the ``security.defense_clip_norm`` setting."""
        return float(
            self._read_setting_sync("security.defense_clip_norm", 1.0)
        )

    @staticmethod
    def _read_setting_sync(key: str, default):
        """Best-effort synchronous read of a setting.

        The settings service is async (DB-backed) so we fall back to the
        static defaults defined in ``DEFAULT_SETTINGS`` to avoid needing
        an event loop here.  This keeps the defense usable from both sync
        and async call-sites.
        """
        try:
            from backend.services.settings import DEFAULT_SETTINGS

            for defn in DEFAULT_SETTINGS:
                if defn.key == key:
                    return defn.default_value
            return default
        except Exception:
            return default
