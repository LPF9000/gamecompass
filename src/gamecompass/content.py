"""Portable structured game content and supervised collaborative projection.

Steam tags remain atomic labels (``open world`` is one feature). Description
words have their own vocabulary and IDF. No pickle or executable serialization
is used. Missing/unknown metadata yields zero evidence, never popularity scores.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize

from .models import atomic_json, sha256


def _json_value(value, default):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


def atomic_tags(game: Mapping) -> list[str]:
    """Read the real source tag location, plus available structured genres/themes."""
    metadata = _json_value(game.get('metadata'), {})
    if not isinstance(metadata, dict):
        metadata = {}
    tags = []
    for value in (metadata.get('tags', []), game.get('tags', []), game.get('genres', []), game.get('themes', [])):
        value = _json_value(value, [])
        if isinstance(value, (list, tuple)):
            tags.extend(v for v in value if isinstance(v, str))
    return sorted({re.sub(r'\s+', ' ', v).strip().casefold() for v in tags if v.strip()})


class ContentFeatures:
    """Fitted multi-hot atomic tags and independent sublinear description TF-IDF."""

    def __init__(self) -> None:
        self.tag_vocabulary: dict[str, int] = {}
        self.description_vocabulary: dict[str, int] = {}
        self.idf = np.empty(0, dtype=np.float32)
        self.metadata: dict = {}

    @classmethod
    def fit(cls, games: Sequence[Mapping], max_description_features: int = 12000, min_df: int = 2) -> 'ContentFeatures':
        if not games:
            raise ValueError('Cannot fit content features on an empty catalog')
        if max_description_features < 1 or min_df < 1:
            raise ValueError('Description feature limits must be positive')
        model = cls()
        effective_min_df = min(min_df, len(games))
        tags = sorted({tag for game in games for tag in atomic_tags(game)})
        model.tag_vocabulary = {tag: i for i, tag in enumerate(tags)}
        descriptions = [str(g.get('description') or '') for g in games]
        vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english', min_df=effective_min_df,
                                     max_features=max_description_features, dtype=np.float32)
        try:
            counts = vectorizer.fit_transform(descriptions).tocsr()
            model.description_vocabulary = {str(token): int(index) for token, index in vectorizer.vocabulary_.items()}
            df = np.asarray((counts > 0).sum(axis=0)).ravel()
            model.idf = (np.log((1 + len(games)) / (1 + df)) + 1).astype(np.float32)
        except ValueError as exc:
            if 'empty vocabulary' not in str(exc) and 'no terms remain' not in str(exc).lower():
                raise
        model.metadata = {'description_ngram_range': [1, 2], 'description_stop_words': 'english',
                          'description_sublinear_tf': True, 'description_min_df': effective_min_df,
                          'description_max_features': max_description_features, 'catalog_fit_rows': len(games),
                          'tag_fields': ['metadata.tags', 'tags', 'genres', 'themes'],
                          'tag_semantics': 'deduplicated case-folded atomic multi-hot labels, L2 normalized'}
        return model

    def transform(self, games: Sequence[Mapping]) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        rr, cc = [], []
        for row, game in enumerate(games):
            for tag in atomic_tags(game):
                if tag in self.tag_vocabulary:
                    rr.append(row)
                    cc.append(self.tag_vocabulary[tag])
        tags = sparse.csr_matrix((np.ones(len(rr), dtype=np.float32), (rr, cc)),
                                 shape=(len(games), len(self.tag_vocabulary)))
        tags = normalize(tags, norm='l2').tocsr() if tags.shape[1] else tags
        if self.description_vocabulary:
            vectorizer = CountVectorizer(vocabulary=self.description_vocabulary, ngram_range=(1, 2),
                                         stop_words='english', dtype=np.float32)
            words = vectorizer.transform([str(g.get('description') or '') for g in games]).tocsr()
            if words.nnz:
                words.data = np.log(words.data) + 1
            words = words.multiply(self.idf).tocsr()
            words = normalize(words, norm='l2').tocsr()
        else:
            words = sparse.csr_matrix((len(games), 0), dtype=np.float32)
        return tags.astype(np.float32), words.astype(np.float32)

    @staticmethod
    def combine(tags: sparse.csr_matrix, words: sparse.csr_matrix, tag_weight: float) -> sparse.csr_matrix:
        if not 0 <= tag_weight <= 1 or tags.shape[0] != words.shape[0]:
            raise ValueError('Aligned feature blocks and a tag weight in [0,1] are required')
        combined = sparse.hstack((tags * np.sqrt(tag_weight), words * np.sqrt(1 - tag_weight)), format='csr')
        return normalize(combined, norm='l2').tocsr().astype(np.float32) if combined.shape[1] else combined

    def to_manifest(self) -> dict:
        return {**self.metadata, 'tag_vocabulary': self.tag_vocabulary,
                'description_vocabulary': {token: int(index) for token, index in self.description_vocabulary.items()}}

    @classmethod
    def from_manifest(cls, manifest: Mapping, idf: np.ndarray) -> 'ContentFeatures':
        result = cls()
        result.tag_vocabulary = dict(manifest['tag_vocabulary'])
        result.description_vocabulary = dict(manifest['description_vocabulary'])
        result.idf = np.asarray(idf, dtype=np.float32)
        for vocabulary in (result.tag_vocabulary, result.description_vocabulary):
            if sorted(vocabulary.values()) != list(range(len(vocabulary))):
                raise ValueError('Feature vocabulary is not contiguous and unique')
        if result.idf.shape != (len(result.description_vocabulary),) or not np.isfinite(result.idf).all():
            raise ValueError('Invalid fitted description IDF')
        result.metadata = {k: v for k, v in manifest.items() if not k.endswith('vocabulary')}
        return result


def fit_ridge_projection(features: sparse.csr_matrix, factors: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0 or features.shape[0] != factors.shape[0] or features.shape[1] == 0:
        raise ValueError('Aligned nonempty features and positive ridge alpha required')
    estimator = Ridge(alpha=alpha, solver='lsqr', fit_intercept=False, tol=1e-5)
    estimator.fit(features, factors)
    return np.asarray(estimator.coef_, dtype=np.float32)


def project(features: sparse.csr_matrix, coefficients: np.ndarray) -> np.ndarray:
    if features.shape[1] != coefficients.shape[1]:
        raise ValueError('Projection feature dimension mismatch')
    return normalize(np.asarray(features @ coefficients.T), norm='l2').astype(np.float32)


class ContentModel:
    """Fitted content artifact, including metadata-only representations for new games."""

    def __init__(self) -> None:
        self.features = ContentFeatures()
        self.game_ids: list[str] = []
        self.warm_ids: list[str] = []
        self.warm_factors = np.empty((0, 0), dtype=np.float32)
        self.projection = np.empty((0, 0), dtype=np.float32)
        self.tags = sparse.csr_matrix((0, 0), dtype=np.float32)
        self.words = sparse.csr_matrix((0, 0), dtype=np.float32)
        self.cosine_tag_weight = 1.0
        self.projection_tag_weight = 1.0
        self.projection_alpha = 1.0
        self.selected = 'cosine'
        self.metadata: dict = {}

    def _indexes(self) -> None:
        self._idx = {g: i for i, g in enumerate(self.game_ids)}
        self._warm_idx = {g: i for i, g in enumerate(self.warm_ids)}
        self._cosine = ContentFeatures.combine(self.tags, self.words, self.cosine_tag_weight)
        projection_features = ContentFeatures.combine(self.tags, self.words, self.projection_tag_weight)
        self._projected = project(projection_features, self.projection)
        self._candidate_cache: dict[tuple, object] = {}
        for game, row in self._warm_idx.items():
            if game in self._idx:
                self._projected[self._idx[game]] = self.warm_factors[row]

    def _representation(self, algorithm: str):
        return {'tags': self.tags, 'description': self.words, 'cosine': self._cosine, 'projection': self._projected}[algorithm]

    def _transform_representation(self, games: Sequence[Mapping], algorithm: str):
        tag, words = self.features.transform(games)
        if algorithm == 'tags':
            return tag
        if algorithm == 'description':
            return words
        matrix = ContentFeatures.combine(tag, words, self.cosine_tag_weight if algorithm == 'cosine' else self.projection_tag_weight)
        return project(matrix, self.projection) if algorithm == 'projection' else matrix

    def score(self, support: Mapping[str, float], candidates: Sequence[Mapping] | Sequence[str] | None = None,
              support_games: Sequence[Mapping] | None = None, algorithm: str | None = None) -> np.ndarray:
        algorithm = algorithm or self.selected
        if algorithm not in {'tags', 'description', 'cosine', 'projection'}:
            raise ValueError('Unknown content algorithm')
        representation = self._representation(algorithm)
        if candidates is None:
            candidate_matrix = representation
            count = len(self.game_ids)
        else:
            ids = tuple(g if isinstance(g, str) else str(g.get('game_id', '')) for g in candidates)
            cache_key = (algorithm, ids)
            if all(g in self._idx for g in ids):
                # Repeated evaluation/user requests reuse a small candidate view.
                if cache_key not in self._candidate_cache:
                    if len(self._candidate_cache) >= 8:
                        self._candidate_cache.pop(next(iter(self._candidate_cache)))
                    self._candidate_cache[cache_key] = representation[[self._idx[g] for g in ids]]
                candidate_matrix = self._candidate_cache[cache_key]
            else:
                games = [{'game_id': g} if isinstance(g, str) else g for g in candidates]
                known_positions = [i for i, g in enumerate(ids) if g in self._idx]
                fresh_positions = [i for i, g in enumerate(ids) if g not in self._idx]
                known_rows = representation[[self._idx[ids[i]] for i in known_positions]]
                fresh_rows = self._transform_representation([games[i] for i in fresh_positions], algorithm)
                stacked = np.vstack((known_rows, fresh_rows)) if algorithm == 'projection' else sparse.vstack((known_rows, fresh_rows), format='csr')
                candidate_matrix = stacked[np.argsort(known_positions + fresh_positions)]
            count = len(candidates)
        known = [(self._idx[g], float(w)) for g, w in support.items()
                 if g in self._idx and np.isfinite(w) and w != 0]
        center = np.zeros(representation.shape[1], dtype=np.float32)
        if known:
            indices, weights = zip(*known)
            center += np.asarray(representation[list(indices)].T @ np.array(weights, dtype=np.float32)).ravel()
        extras = [g for g in (support_games or []) if g.get('game_id') not in self._idx and support.get(g.get('game_id'), 0) != 0]
        if extras:
            matrix = self._transform_representation(extras, algorithm)
            weights = np.array([float(support[g['game_id']]) for g in extras], dtype=np.float32)
            center += np.asarray(matrix.T @ weights).ravel()
        length = np.linalg.norm(center)
        if not length:
            return np.zeros(count, dtype=np.float32)
        return np.asarray(candidate_matrix @ (center / length)).ravel()

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=False)
        sparse.save_npz(path / 'tags.npz', self.tags)
        sparse.save_npz(path / 'words.npz', self.words)
        temporary = path / 'weights.npz.tmp'
        with temporary.open('wb') as f:
            np.savez_compressed(f, idf=self.features.idf, warm_factors=self.warm_factors, projection=self.projection)
        os.replace(temporary, path / 'weights.npz')
        atomic_json(path / 'manifest.json', {
            'format_version': 1, 'kind': 'structured-content-v2', 'game_ids': self.game_ids,
            'warm_ids': self.warm_ids, 'features': self.features.to_manifest(),
            'cosine_tag_weight': self.cosine_tag_weight, 'projection_tag_weight': self.projection_tag_weight,
            'projection_alpha': self.projection_alpha, 'selected': self.selected, 'metadata': self.metadata,
            'checksums': {name: sha256(path / name) for name in ('tags.npz', 'words.npz', 'weights.npz')},
        })

    @classmethod
    def load(cls, path: Path) -> 'ContentModel':
        try:
            manifest = json.loads((path / 'manifest.json').read_text())
            if manifest.get('format_version') != 1 or manifest.get('kind') != 'structured-content-v2':
                raise ValueError('Unsupported structured content artifact')
            for name in ('tags.npz', 'words.npz', 'weights.npz'):
                if sha256(path / name) != manifest['checksums'][name]:
                    raise ValueError(f'Content artifact integrity check failed: {name}')
            result = cls()
            result.game_ids = manifest['game_ids']
            result.warm_ids = manifest['warm_ids']
            result.cosine_tag_weight = float(manifest['cosine_tag_weight'])
            result.projection_tag_weight = float(manifest['projection_tag_weight'])
            result.projection_alpha = float(manifest['projection_alpha'])
            result.selected = manifest['selected']
            result.metadata = manifest['metadata']
            with np.load(path / 'weights.npz', allow_pickle=False) as data:
                result.features = ContentFeatures.from_manifest(manifest['features'], data['idf'])
                result.warm_factors = data['warm_factors']
                result.projection = data['projection']
            result.tags = sparse.load_npz(path / 'tags.npz').tocsr()
            result.words = sparse.load_npz(path / 'words.npz').tocsr()
            n = len(result.game_ids)
            dimensions = len(result.features.tag_vocabulary) + len(result.features.description_vocabulary)
            if (len(set(result.game_ids)) != n or len(set(result.warm_ids)) != len(result.warm_ids)
                    or not set(result.warm_ids) <= set(result.game_ids)
                    or result.tags.shape != (n, len(result.features.tag_vocabulary))
                    or result.words.shape != (n, len(result.features.description_vocabulary))
                    or result.warm_factors.ndim != 2 or result.warm_factors.shape[0] != len(result.warm_ids)
                    or result.projection.shape != (result.warm_factors.shape[1], dimensions)
                    or result.selected not in {'cosine', 'projection'}
                    or not np.isfinite(result.warm_factors).all() or not np.isfinite(result.projection).all()):
                raise ValueError('Content artifact dimensions or values are invalid')
            result._indexes()
            return result
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError('Cannot load structured content artifact; restore or retrain it') from exc
