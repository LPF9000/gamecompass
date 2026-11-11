"""Portable population models and immediate signed-feedback user fold-in.

No user embedding is loaded for inference: each request builds a representation
from its supplied support. Unknown interactions are not stored as dislikes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.linalg import solve
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd

ALGORITHMS = ("popularity", "itemknn", "ease", "svd", "content", "hybrid")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def game_text(game: Mapping) -> str:
    """No title feature: avoids recommending editions solely by identical names."""
    parts = [str(game.get("description") or "")]
    for key in ("genres", "themes"):
        value = game.get(key, [])
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = []
        parts.extend(str(v) for v in value)
    return " ".join(parts).strip()


class Model:
    """Population item models; arrays are index-aligned by stable canonical IDs."""

    def __init__(self) -> None:
        self.game_ids: list[str] = []
        self.cf_game_ids: list[str] = []
        self.selected = "popularity"
        self.hybrid_alpha = 0.5
        self.hybrid_cf = "itemknn"
        self.hybrid_popularity = 0.0
        self.selection_by_history: dict[str, str] = {}
        self.metadata: dict = {}
        self.knn = sparse.csr_matrix((0, 0), dtype=np.float32)
        self.ease = np.empty((0, 0), dtype=np.float32)
        self.factors = np.empty((0, 0), dtype=np.float32)
        self.content = sparse.csr_matrix((0, 4096), dtype=np.float32)
        self.idf = np.ones(4096, dtype=np.float32)
        self.popularity = np.empty(0, dtype=np.float32)

    def _indexes(self) -> None:
        self._idx = {g: i for i, g in enumerate(self.game_ids)}
        self._cf_idx = {g: i for i, g in enumerate(self.cf_game_ids)}
        self._cf_out = np.array([self._idx[g] for g in self.cf_game_ids])

    @classmethod
    def fit(
        cls, games: Sequence[Mapping], cf_game_ids: Sequence[str],
        positive_matrix: sparse.csr_matrix, seed: int = 42,
        ease_regularization: float = 100.0, neighbors: int = 80,
    ) -> "Model":
        """Fit on explicitly positive population interactions only.

        Explicit negatives remain available to evaluation and support inference;
        they are never inserted in the positive interaction matrix.
        """
        model = cls()
        model.game_ids = [str(g["game_id"]) for g in games]
        if len(set(model.game_ids)) != len(model.game_ids):
            raise ValueError("Duplicate canonical IDs in catalog")
        model.cf_game_ids = list(cf_game_ids)
        if positive_matrix.shape[1] != len(cf_game_ids):
            raise ValueError("Interaction columns do not match game IDs")
        if len(cf_game_ids) > 6000:
            raise ValueError("Dense EASE is capped at 6000 items; use a bounded configuration")
        model._indexes()
        x = positive_matrix.astype(np.float64).tocsr()
        counts = np.asarray(x.sum(axis=0)).ravel()
        model.popularity = np.zeros(len(games), dtype=np.float32)
        model.popularity[model._cf_out] = counts
        gram = (x.T @ x).toarray()
        norm = np.sqrt(np.maximum(np.diag(gram), 1))
        cosine = gram / (norm[:, None] * norm[None, :])
        # Co-count shrinkage controls extremely rare coincident pairs.
        cosine *= gram / (gram + 10.0)
        np.fill_diagonal(cosine, 0)
        k = min(neighbors, max(0, len(cf_game_ids) - 1))
        if k:
            keep = np.argpartition(cosine, -k, axis=1)[:, -k:]
            rows = np.repeat(np.arange(len(cf_game_ids)), k)
            cols = keep.ravel()
            vals = cosine[rows, cols]
            model.knn = sparse.csr_matrix((vals, (rows, cols)), shape=cosine.shape, dtype=np.float32)
        else:
            model.knn = sparse.csr_matrix(cosine, dtype=np.float32)
        gram.flat[:: len(cf_game_ids) + 1] += ease_regularization
        inverse = solve(gram, np.eye(len(cf_game_ids)), assume_a="pos", check_finite=False)
        model.ease = (-inverse / np.diag(inverse)[None, :]).astype(np.float32)
        np.fill_diagonal(model.ease, 0)
        components = min(64, min(x.shape) - 1)
        if components >= 1:
            _, _, vt = randomized_svd(x, n_components=components, random_state=seed)
            model.factors = vt.T.astype(np.float32)
        else:
            model.factors = np.zeros((len(cf_game_ids), 1), dtype=np.float32)
        vectorizer = HashingVectorizer(n_features=4096, alternate_sign=False, norm=None, stop_words="english", ngram_range=(1, 2))
        hashed = vectorizer.transform([game_text(g) for g in games])
        tfidf = TfidfTransformer()
        model.content = tfidf.fit_transform(hashed).astype(np.float32).tocsr()
        model.idf = tfidf.idf_.astype(np.float32)
        model.metadata = {"format_version": 1, "seed": seed, "training_users": x.shape[0],
                          "training_positives": int(x.nnz), "ease_regularization": ease_regularization,
                          "knn_neighbors": neighbors, "content": "4096-bin hashing TF-IDF; descriptions, genres and themes; no title",
                          "score_semantics": "uncalibrated ranking scores, not enjoyment probabilities"}
        return model

    def content_features(self, games: Sequence[Mapping]) -> sparse.csr_matrix:
        vectorizer = HashingVectorizer(n_features=4096, alternate_sign=False, norm=None, stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.transform([game_text(g) for g in games]).multiply(self.idf)
        return normalize(matrix, norm="l2").tocsr().astype(np.float32)

    @staticmethod
    def _scaled(values: np.ndarray) -> np.ndarray:
        scale = np.max(np.abs(values)) if values.size else 0
        return values / scale if scale > 0 else values

    def algorithm_for(self, support: Mapping[str, float]) -> str:
        history = sum(g in self._cf_idx and np.isfinite(w) and w != 0 for g, w in support.items())
        eligible = [int(n) for n in self.selection_by_history if int(n) <= history]
        return self.selection_by_history[str(max(eligible))] if eligible else self.selected

    def score(self, support: Mapping[str, float], algorithm: str | None = None) -> np.ndarray:
        algorithm = algorithm or self.algorithm_for(support)
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown model: {algorithm}")
        known = [(self._idx[g], float(w)) for g, w in support.items() if g in self._idx and np.isfinite(w) and w != 0]
        if not known or algorithm == "popularity":
            return self.popularity.copy()
        profile = np.zeros(len(self.cf_game_ids), dtype=np.float32)
        for g, weight in support.items():
            if g in self._cf_idx and np.isfinite(weight):
                profile[self._cf_idx[g]] = float(weight)
        scores = np.zeros(len(self.game_ids), dtype=np.float32)
        if algorithm == "content":
            indices, weights = zip(*known)
            centroid = np.asarray(self.content[list(indices)].T @ np.array(weights, dtype=np.float32)).ravel()
            norm = np.linalg.norm(centroid)
            if norm:
                centroid /= norm
            return np.asarray(self.content @ centroid).ravel()
        if algorithm == "hybrid":
            return ((1 - self.hybrid_alpha - self.hybrid_popularity) * self._scaled(self.score(support, self.hybrid_cf))
                    + self.hybrid_alpha * self._scaled(self.score(support, "content"))
                    + self.hybrid_popularity * self._scaled(self.popularity))
        indices = np.flatnonzero(profile)
        if algorithm == "itemknn":
            scores[self._cf_out] = np.asarray(profile[indices] @ self.knn[indices]).ravel()
        elif algorithm == "ease":
            scores[self._cf_out] = profile[indices] @ self.ease[indices]
        elif algorithm == "svd":
            user = profile[indices] @ self.factors[indices]
            scores[self._cf_out] = self.factors @ user
        return scores

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=False)
        weights = path / "weights.npz"
        # np.savez appends .npz to a string filename; use a file object for atomic name.
        tmp = path / "weights.npz.tmp"
        with tmp.open("wb") as f:
            np.savez_compressed(f, ease=self.ease, factors=self.factors, popularity=self.popularity, idf=self.idf)
        os.replace(tmp, weights)
        sparse.save_npz(path / "knn.npz", self.knn)
        sparse.save_npz(path / "content.npz", self.content)
        manifest = {**self.metadata, "game_ids": self.game_ids, "cf_game_ids": self.cf_game_ids,
                    "selected": self.selected, "hybrid_alpha": self.hybrid_alpha, "hybrid_cf": self.hybrid_cf,
                    "hybrid_popularity": self.hybrid_popularity, "selection_by_history": self.selection_by_history,
                    "checksums": {name: sha256(path / name) for name in ("weights.npz", "knn.npz", "content.npz")}}
        atomic_json(path / "manifest.json", manifest)

    @classmethod
    def load(cls, path: Path) -> "Model":
        try:
            manifest = json.loads((path / "manifest.json").read_text())
            if manifest.get("format_version") != 1:
                raise ValueError("Unsupported model format")
            for name in ("weights.npz", "knn.npz", "content.npz"):
                if sha256(path / name) != manifest["checksums"][name]:
                    raise ValueError(f"Model integrity check failed: {name}")
            model = cls()
            model.game_ids = manifest["game_ids"]
            model.cf_game_ids = manifest["cf_game_ids"]
            model.selected = manifest["selected"]
            model.hybrid_alpha = manifest["hybrid_alpha"]
            model.hybrid_cf = manifest["hybrid_cf"]
            model.hybrid_popularity = manifest.get("hybrid_popularity", 0.0)
            model.selection_by_history = manifest.get("selection_by_history", {})
            model.metadata = manifest
            with np.load(path / "weights.npz", allow_pickle=False) as data:
                for key in ("ease", "factors", "popularity", "idf"):
                    setattr(model, key, data[key])
            model.knn = sparse.load_npz(path / "knn.npz")
            model.content = sparse.load_npz(path / "content.npz")
            model._indexes()
            n, m = len(model.game_ids), len(model.cf_game_ids)
            if (model.ease.shape != (m, m) or model.knn.shape != (m, m)
                    or model.factors.shape[0] != m or model.popularity.shape != (n,)
                    or model.content.shape != (n, 4096) or model.idf.shape != (4096,)):
                raise ValueError("Model arrays do not match catalog")
            return model
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Cannot load model artifact; retrain or restore a verified version") from exc
