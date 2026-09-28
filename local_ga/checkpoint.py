"""Local checkpoint persistence.

Checkpoints are written to a regular local file with the standard library
``pickle``. Nothing is ever sent over the network or stored outside the path
the caller chooses.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .optimizer import GeneticOptimizer

CHECKPOINT_VERSION = 1


def save_checkpoint(optimizer: "GeneticOptimizer", path: str) -> str:
    """Atomically save the full optimizer state (population, generation, RNG).

    Writes to a temp file in the same directory and renames into place, so a
    crashed write cannot corrupt an existing checkpoint.
    """
    payload = {"version": CHECKPOINT_VERSION, "optimizer": optimizer}
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".ckpt-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return os.path.abspath(path)


def load_checkpoint(path: str) -> "GeneticOptimizer":
    """Load a checkpoint and return a ready-to-run GeneticOptimizer."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "optimizer" not in payload:
        raise ValueError(f"{path!r} is not a valid local_ga checkpoint")
    version = payload.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {version!r}, expected {CHECKPOINT_VERSION}"
        )
    optimizer = payload["optimizer"]
    from .optimizer import GeneticOptimizer

    if not isinstance(optimizer, GeneticOptimizer):
        raise ValueError("checkpoint payload does not contain a GeneticOptimizer")
    return optimizer
