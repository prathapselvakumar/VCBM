import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _hash_bucket(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % dim


def encode_state(text: str, dim: int = 256) -> np.ndarray:
    """Hashing-trick bag-of-words embedding of observation text, L2-normalized.

    Dependency-free stand-in for the state encoder g_phi: deterministic across
    processes (uses md5 rather than the salted builtin hash()) so memory-bank
    keys are reproducible for testing.
    """
    vec = np.zeros(dim, dtype=np.float64)
    for token in _TOKEN_RE.findall(text.lower()):
        vec[_hash_bucket(token, dim)] += 1.0

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec
