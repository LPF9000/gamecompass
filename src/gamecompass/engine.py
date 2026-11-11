"""UI-independent local recommendation API; never uploads feedback."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
from .store import Store


class Recommender:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.store = Store(self.data_dir)

    def _model(self):
        from .models import Model
        pointer = self.data_dir / 'models' / 'current.json'
        if not pointer.exists():
            raise RuntimeError('No trained population model. Run gamecompass benchmark first; see setup documentation.')
        manifest = json.loads(pointer.read_text())
        version = manifest.get('version') or manifest.get('path')
        if not version:
            raise ValueError('Invalid current model pointer')
        model_path = (self.data_dir / 'models' / version).resolve()
        if not model_path.is_relative_to((self.data_dir / 'models').resolve()):
            raise ValueError('Model path must stay within local models directory')
        model = Model.load(model_path)
        additions = self.store.catalog_additions(set(model.game_ids))
        if additions:
            from scipy import sparse
            model.content = sparse.vstack([model.content, model.content_features(additions)], format='csr')
            model.game_ids.extend(g['game_id'] for g in additions)
            model.popularity = np.concatenate([model.popularity,np.zeros(len(additions),dtype=np.float32)])
            model._indexes()
        return model

    def recommend(self, profile_id: str, limit: int = 20, platform: str | None = None,
                  exclude_played: bool = True, content_explore: bool = False) -> list[dict[str, Any]]:
        if not 1 <= limit <= 200:
            raise ValueError('limit must be between 1 and 200')
        model = self._model()
        feedback = self.store.feedback(profile_id)
        support = {f['game_id']: float(f['rating']) for f in feedback if f.get('rating') is not None and f['rating'] != 0}
        implicit_mode = False
        implicit_algorithm = None
        config = model.metadata.get('implicit')
        if not support and config:
            judged = {f['game_id'] for f in feedback if f.get('rating') is not None or f.get('not_interested')}
            observed = {f['game_id']:max(0, float(f['playtime_minutes']) / 60) for f in self.store.implicit(profile_id)
                        if f['game_id'] not in judged and f['game_id'] in model.game_ids}
            buckets = [int(k) for k in config.get('selection_by_history', {}) if int(k) <= sum(g in set(model.cf_game_ids) for g in observed)]
            route = config['selection_by_history'][str(max(buckets))] if buckets else config
            implicit_algorithm = route['algorithm']
            support = {g: (float(np.log1p(hours)) if route['weighting'] == 'log1p_hours' else 1.0)
                       for g,hours in observed.items()}
            support = {g:w for g,w in support.items() if w != 0}
            implicit_mode = bool(support)
        has_cf_support = any(g in set(model.cf_game_ids) for g in support)
        content_fallback = bool(support) and not has_cf_support
        algorithm = 'content' if content_explore or content_fallback else implicit_algorithm or model.algorithm_for(support)
        if algorithm == 'content' and support:
            valid_indices = [model._idx[g] for g in support if g in model._idx]
            if not valid_indices or model.content[valid_indices].nnz == 0:
                algorithm = 'popularity'
        scores = model.score(support, algorithm)
        unwanted = {f['game_id'] for f in feedback if f.get('not_interested') or
                    (f.get('rating') is not None and f['rating'] < 0) or
                    (exclude_played and f.get('status') in {'played', 'completed', 'abandoned'})}
        rated = {f['game_id'] for f in feedback if f.get('rating') is not None}
        unwanted.update(rated)
        result = []
        positives = [] if implicit_mode else [self.store.game(g) for g, r in support.items() if r > 0]
        positives = [g for g in positives if g]
        for i in np.argsort(-np.nan_to_num(scores, nan=-1e30), kind='stable'):
            game_id = model.game_ids[i]
            if game_id not in model.cf_game_ids and algorithm != 'content' and not (algorithm == 'hybrid' and model.hybrid_alpha > 0):
                continue
            if game_id in unwanted:
                continue
            game = self.store.game(game_id)
            if not game:
                continue
            if platform and platform not in game.get('platforms', []):
                continue
            if not support:
                explanation = 'Population baseline: you have not provided enjoyment ratings yet. This is not personalized.'
            elif implicit_mode and algorithm != 'popularity':
                explanation = (('Content exploration inferred from imported ownership/playtime. '
                                if algorithm == 'content' else
                                'Inferred from imported ownership/playtime using the validation-selected implicit-history model. ')
                               + 'These are weak signals, not explicit enjoyment ratings. The evaluation used reviewed-game histories as a library proxy. '
                               'This ranking is not a probability of enjoyment.')
                if content_fallback:
                    explanation += ' Collaborative evidence is unavailable; metadata-only uncertainty is higher.'
            elif algorithm == 'popularity':
                explanation = 'Population baseline selected by validation. This ranking does not use individual taste; further personalization experiments are needed.'
            else:
                tags = set(game.get('genres', []) + game.get('themes', []))
                shared = sorted(tags.intersection({t for g in positives for t in g.get('genres', []) + g.get('themes', [])}))
                explanation = ('Content-based ranking using descriptions and catalog tags of games you rated.'
                               if algorithm == 'content' else
                               'Ranked using your explicit likes and dislikes with the validation-selected recommendation model.')
                if game_id not in model.cf_game_ids:
                    explanation += ' No population interaction evidence for this title; content-only uncertainty is higher.'
                if content_fallback:
                    explanation += ' Your rated games have no collaborative representation, so this uses the content fallback.'
                if shared:
                    explanation += ' Shares catalog tags with games you liked: ' + ', '.join(shared[:3]) + '.'
                explanation += ' This is an inferred ranking, not a probability of enjoyment.'
            game.update(score=float(scores[i]), explanation=explanation)
            result.append(game)
            if len(result) >= limit:
                break
        return result

    def questions(self, profile_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """Optional diverse elicitation suggestions; information gain is not claimed."""
        return self.recommend(profile_id, limit=limit, exclude_played=False)
