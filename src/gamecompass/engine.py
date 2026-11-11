"""UI-independent local recommendation API; never uploads feedback."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
from .store import Store
from .refinements import RecommendationFilters, release_facts


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
        model.artifact_path = model_path
        return model

    def recommend(self, profile_id: str, limit: int = 20, platform: str | None = None,
                  exclude_played: bool = True, content_explore: bool = False, filters: RecommendationFilters | None = None) -> list[dict[str, Any]]:
        return self._recommend(profile_id,limit,platform,exclude_played,content_explore,filters)

    def similar(self, game_id: str, limit: int = 20, filters: RecommendationFilters | None = None,
                profile_id: str | None = None) -> list[dict[str, Any]]:
        """Context-only seed; does not write a rating or alter a saved profile."""
        if self.store.game(game_id) is None:
            raise ValueError('Reference game is absent from the local catalog')
        return self._recommend(profile_id,limit,None,True,False,filters,seed_game_id=game_id)

    def _recommend(self, profile_id: str | None, limit: int = 20, platform: str | None = None,
                   exclude_played: bool = True, content_explore: bool = False, filters: RecommendationFilters | None = None,
                   seed_game_id: str | None = None) -> list[dict[str, Any]]:
        if not 1 <= limit <= 200:
            raise ValueError('limit must be between 1 and 200')
        model = self._model()
        from .refinements import RecommendationFilters
        filters = filters or RecommendationFilters(platform=platform)
        if platform is not None and filters.platform not in (None,platform):
            raise ValueError('Conflicting platform refinements')
        feedback = self.store.feedback(profile_id) if profile_id else []
        advanced_path = model.artifact_path / 'advanced.json'
        advanced = json.loads(advanced_path.read_text()) if advanced_path.exists() else {}
        artifact_root = advanced_path.parent.resolve()
        def content_model():
            from .content import ContentModel
            target=(artifact_root/advanced['content_directory']).resolve()
            if not target.is_relative_to(artifact_root):raise ValueError('Invalid content artifact path')
            return ContentModel.load(target)
        support = {f['game_id']: float(f['rating']) for f in feedback if f.get('rating') is not None and f['rating'] != 0}
        if seed_game_id:
            support={seed_game_id:1.0}
        mixed_mode = False
        implicit_mode = False
        mixed_config = advanced.get('mixed')
        if support and mixed_config and profile_id and not seed_game_id:
            from .mixed import MixedPersonalizer
            observations={r['game_id']:float(r['playtime_minutes'])/60 for r in self.store.implicit(profile_id) if r['game_id'] in model._cf_idx}
            explicit={f['game_id']:float(f['rating']) for f in feedback if f.get('rating') is not None}
            combined=MixedPersonalizer.combine(explicit,observations,mixed_config['coefficient'],mixed_config['weighting'],
                     not_interested={f['game_id'] for f in feedback if f.get('not_interested')})
            mapped_count=len((set(explicit)|set(observations)) & set(model.cf_game_ids))
            explicit_count=sum(g in model._cf_idx and w!=0 for g,w in explicit.items())
            mixed_mode=any(g not in explicit for g in combined) and 5<=mapped_count<=20 and 1<=explicit_count<=3
            if mixed_mode:support={g:w for g,w in combined.items() if w}
        implicit_algorithm = None
        config = model.metadata.get('implicit')
        if not support and config and profile_id:
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
        algorithm = 'content' if content_explore or content_fallback else (mixed_config['algorithm'] if mixed_mode else implicit_algorithm or model.algorithm_for(support))
        if algorithm == 'content' and support and not advanced.get('content_directory'):
            valid_indices = [model._idx[g] for g in support if g in model._idx]
            if not valid_indices or model.content[valid_indices].nnz == 0:
                algorithm = 'popularity'
        refinement_active = filters != RecommendationFilters()
        catalog = self.store.catalog_games() if refinement_active or (advanced and algorithm=='content') else None
        by_id = {g['game_id']:g for g in catalog} if catalog is not None else {}
        def related_ids(game):
            metadata=game.get('metadata',{}) if game else {}
            record=metadata.get('wikidata',{}) if isinstance(metadata,dict) else {}
            values=record.get('related_steam_game_ids',[]) if isinstance(record,dict) else []
            return {g for g in values if isinstance(g,str)} if isinstance(values,list) else set()
        def with_source_exclusions(ids):
            expanded=set(ids)
            if exclude_played:
                for g in ids:
                    if g.startswith('wikidata:'):
                        expanded.update(related_ids(by_id.get(g) or self.store.game(g)))
            return expanded
        refinement_fallback = False
        if refinement_active and support and advanced and not any(filters.matches(by_id[g]) for g in model.cf_game_ids if g in by_id):
            algorithm='content';refinement_fallback=True
        v2_content = algorithm=='content' and bool(advanced.get('content_directory')) and bool(support)
        if v2_content:
            from .blending import CandidateGenerator
            rich=content_model()
            # Keep candidate IDs aligned with the base model; newly cached games were added in _model.
            candidate_games=[by_id.get(g) or self.store.game(g) for g in model.game_ids]
            candidate_games=[g if g is not None else {'game_id':key} for key,g in zip(model.game_ids,candidate_games)]
            support_games=[self.store.game(g) for g in support if self.store.game(g) is not None]
            scores=rich.score(support,candidates=candidate_games,support_games=support_games)
            blocked={f['game_id'] for f in feedback if f.get('rating') is not None or f.get('not_interested')
                     or (exclude_played and f.get('status') in {'played','completed','abandoned'})}
            if seed_game_id:blocked.add(seed_game_id)
            blocked=with_source_exclusions(blocked)
            profile_blocked=set(blocked)
            if exclude_played:
                blocked.update(g['game_id'] for g in candidate_games if profile_blocked.intersection(related_ids(g)))
            blocked.update(g['game_id'] for g in candidate_games if not filters.matches(g))
            pool=CandidateGenerator(200).generate(model.game_ids,{'selected_content':scores},excluded=blocked)
            pool_set=set(pool.tolist())
        else:
            scores = model.score(support, algorithm)
            pool_set=None
        unwanted = {f['game_id'] for f in feedback if f.get('not_interested') or
                    (f.get('rating') is not None and f['rating'] < 0) or
                    (exclude_played and f.get('status') in {'played', 'completed', 'abandoned'})}
        rated = {f['game_id'] for f in feedback if f.get('rating') is not None}
        unwanted.update(rated)
        if seed_game_id:unwanted.add(seed_game_id)
        unwanted=with_source_exclusions(unwanted)
        result = []
        positives = [] if implicit_mode or seed_game_id else [self.store.game(f['game_id']) for f in feedback if f.get('rating') is not None and f['rating']>0]
        positives = [g for g in positives if g]
        for i in np.argsort(-np.nan_to_num(scores, nan=-1e30), kind='stable'):
            game_id = model.game_ids[i]
            if pool_set is not None and int(i) not in pool_set:continue
            if game_id not in model.cf_game_ids and algorithm != 'content' and not (algorithm == 'hybrid' and model.hybrid_alpha > 0):
                continue
            if game_id in unwanted:
                continue
            game = by_id.get(game_id) or self.store.game(game_id)
            if not game:
                continue
            if not filters.matches(game):
                continue
            linked=related_ids(game)
            if exclude_played and linked.intersection(unwanted):
                continue
            if not support:
                explanation = 'Population baseline: you have not provided enjoyment ratings yet. This is not personalized.'
            elif mixed_mode and algorithm != 'popularity':
                explanation = 'Ranked from your explicit ratings together with weaker imported ownership/playtime evidence. Explicit judgments override activity on the same game. This is not a probability of enjoyment.'
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
            if v2_content:
                from .content import atomic_tags
                positive_tags={tag for g in positives for tag in atomic_tags(g)}
                overlaps=sorted(set(atomic_tags(game)) & positive_tags)
                if overlaps:explanation += ' Shares catalog tags with titles you rated positively: '+', '.join(overlaps[:3])+'.'
                explanation += ' Uses structured gameplay tags and descriptions projected into population preference relationships; cold-item uncertainty remains high.'
            if refinement_fallback:
                explanation += ' No collaborative candidates match these refinements; this is a metadata-based fallback.'
            if seed_game_id:
                explanation = f"If you liked {self.store.game(seed_game_id)['title']}, these titles are ranked using that game as a temporary seed. " + ('Uses gameplay tags, descriptions and learned population relationships; metadata-only uncertainty remains higher.' if v2_content else 'Uses the population model when collaborative evidence is available.') + ' Your saved ratings are unchanged. Scores are not probabilities of enjoyment.'
            if filters.platform and filters.era_basis=='edition':
                game['release_context']=release_facts(game,platform=filters.platform)
            game.update(score=float(scores[i]), explanation=explanation)
            result.append(game)
            if len(result) >= limit:
                break
        return result

    def questions(self, profile_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """Optional diverse elicitation suggestions; information gain is not claimed."""
        return self.recommend(profile_id, limit=limit, exclude_played=False)
