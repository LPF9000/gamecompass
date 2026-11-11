"""Observed-feedback-only new-user fold-in for pretrained item factors.

No API accepts a population user ID or a population user embedding. Unknown items
are candidates, never negative training labels. Scores are uncalibrated rankings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.special import expit

from .models import sha256


@dataclass(frozen=True)
class FoldInConfig:
    method: str = "ridge"
    normalization: str = "l2"
    regularization: float = 1.0
    negative_weight: float = 1.0
    max_iterations: int = 100

    def __post_init__(self) -> None:
        if self.method not in ("sum", "centroid", "ridge", "logistic"):
            raise ValueError("Unknown fold-in method")
        if self.normalization not in ("none", "l2", "center_l2"):
            raise ValueError("Unknown factor normalization")
        if not np.isfinite(self.regularization) or self.regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        if not np.isfinite(self.negative_weight) or self.negative_weight <= 0:
            raise ValueError("negative_weight must be finite and positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


class FoldInModel:
    """Cache item transformations; infer an anonymous user from supplied ratings.

    Input feedback is signed: positive=liked, negative=disliked, zero=neutral.
    Absolute magnitude weights confidence; no unobserved examples enter the fit.
    An empty/unknown-only support returns the explicitly supplied popularity prior.
    """

    def __init__(self, game_ids: Sequence[str], factors: np.ndarray,
                 config: FoldInConfig, popularity: np.ndarray | None = None):
        self.game_ids = list(game_ids)
        if not self.game_ids or len(set(self.game_ids)) != len(self.game_ids):
            raise ValueError("Item IDs must be nonempty and unique")
        matrix = np.array(factors, dtype=np.float64, copy=True)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.game_ids) or matrix.shape[1] == 0:
            raise ValueError("Item-factor rows must match the item IDs")
        if not np.isfinite(matrix).all():
            raise ValueError("Item factors must be finite")
        self.config = config
        if config.normalization == "center_l2":
            matrix -= matrix.mean(axis=0, keepdims=True)
        if config.normalization in ("l2", "center_l2"):
            matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        self.factors = matrix
        self._idx = {game: i for i, game in enumerate(self.game_ids)}
        self.popularity = np.zeros(len(game_ids), dtype=np.float64) if popularity is None else np.asarray(popularity, dtype=np.float64).copy()
        if self.popularity.shape != (len(game_ids),) or not np.isfinite(self.popularity).all():
            raise ValueError("Popularity must be finite and aligned with item IDs")
        self.logistic_calls = 0
        self.logistic_nonconverged = 0

    def _observed(self, support: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
        observed = []
        for game, value in support.items():
            weight = float(value)
            if not np.isfinite(weight):
                raise ValueError("Feedback weights must be finite")
            if game in self._idx and weight:
                observed.append((self._idx[game], weight))
        if not observed:
            return np.empty(0, dtype=int), np.empty(0, dtype=np.float64)
        indices, weights = zip(*observed)
        return np.asarray(indices, dtype=int), np.asarray(weights, dtype=np.float64)

    def user_vector(self, support: Mapping[str, float]) -> np.ndarray:
        indices, weights = self._observed(support)
        if not len(indices):
            return np.zeros(self.factors.shape[1], dtype=np.float64)
        x = self.factors[indices]
        weights = weights.copy()
        weights[weights < 0] *= self.config.negative_weight
        if self.config.method == "sum":
            return weights @ x
        if self.config.method == "centroid":
            positive, negative = weights > 0, weights < 0
            vector = np.zeros(x.shape[1], dtype=np.float64)
            if positive.any():
                vector += weights[positive] @ x[positive] / int(positive.sum())
            if negative.any():
                # Independent centroids avoid making three positives overwhelm one
                # explicit negative merely because the class counts differ. Average confidence
                # is retained, so a weak implicit signal cannot cancel a strong dislike.
                vector += weights[negative] @ x[negative] / int(negative.sum())
            return vector
        confidence = np.abs(weights)
        labels = np.sign(weights)
        lam = self.config.regularization
        if self.config.method == "ridge":
            scaled = x * np.sqrt(confidence)[:, None]
            target = labels * np.sqrt(confidence)
            # Dual solve is much smaller for the common 1/3/5-rating profiles.
            if len(indices) <= x.shape[1]:
                gram = scaled @ scaled.T + lam * np.eye(len(indices))
                return scaled.T @ solve(gram, target, assume_a="pos", check_finite=False)
            gram = scaled.T @ scaled + lam * np.eye(x.shape[1])
            return solve(gram, scaled.T @ target, assume_a="pos", check_finite=False)

        def objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
            margin = labels * (x @ vector)
            loss = float(np.dot(confidence, np.logaddexp(0.0, -margin)) + 0.5 * lam * np.dot(vector, vector))
            gradient = -(x.T @ (confidence * labels * expit(-margin))) + lam * vector
            return loss, gradient

        result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                          options={"maxiter": self.config.max_iterations, "gtol": 1e-7, "ftol": 1e-12})
        self.logistic_calls += 1
        self.logistic_nonconverged += int(not result.success)
        if not np.isfinite(result.x).all():
            raise ValueError("Logistic fold-in produced nonfinite factors")
        # A finite iteration-limited result is an approximate optimizer, reported
        # in benchmark diagnostics rather than misrepresented as converged.
        return np.asarray(result.x)

    def score(self, support: Mapping[str, float]) -> np.ndarray:
        indices, _ = self._observed(support)
        if not len(indices):
            return self.popularity.copy()
        return self.factors @ self.user_vector(support)


def load_item_factors(directory: Path) -> tuple[list[str], np.ndarray, dict]:
    """Read only a checked item-only NumPy export; never a Torch checkpoint."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / "factors.npz"
    if sha256(path) != manifest.get("factor_sha256"):
        raise ValueError("Item-factor checksum mismatch")
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != {"game_ids", "factors"}:
            raise ValueError("Expected an item-only factor export; unexpected arrays rejected")
        ids = [str(value) for value in stored["game_ids"]]
        factors = np.asarray(stored["factors"], dtype=np.float64)
    # Validate without changing the representation.
    FoldInModel(ids, factors, FoldInConfig(method="sum", normalization="none"))
    return ids, factors, manifest
