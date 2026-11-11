"""Validated score combination and explicit candidate-generation interfaces.

Channels are uncalibrated ranking evidence. Positive approval rates describe
observed reviews, not an individual's probability of enjoying an unseen game.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import json
import numpy as np
from .models import atomic_json


def normalized_channels(channels: Mapping[str, np.ndarray], names: Sequence[str], reference_indices: np.ndarray | None = None) -> np.ndarray:
    arrays = [np.asarray(channels[name], dtype=np.float32) for name in names]
    if not arrays or any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError('Score channels must be aligned finite one-dimensional arrays')
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite score channel')
    matrix = np.column_stack(arrays)
    reference = matrix if reference_indices is None else matrix[np.asarray(reference_indices,dtype=int)]
    scale = np.max(np.abs(reference),axis=0) if len(reference) else np.ones(len(names))
    return matrix / np.where(scale > 0,scale,1)


@dataclass(frozen=True)
class CandidateGenerator:
    """Union top items per evidence channel, with stable IDs and explicit eligibility."""
    per_channel: int = 200

    def generate(self, game_ids: Sequence[str], channels: Mapping[str,np.ndarray],
                 excluded: set[str] | None = None) -> np.ndarray:
        if self.per_channel < 1 or len(set(game_ids)) != len(game_ids):
            raise ValueError('Unique candidate IDs and positive retrieval depth required')
        ids=np.asarray(game_ids)
        eligible=np.array([i for i,g in enumerate(game_ids) if g not in (excluded or set())],dtype=int)
        selected:set[int]=set()
        for values in channels.values():
            values=np.asarray(values)
            if values.shape != (len(ids),) or not np.isfinite(values).all():
                raise ValueError('Invalid retrieval channel')
            order=np.lexsort((ids[eligible],-values[eligible]))
            selected.update(int(i) for i in eligible[order[:self.per_channel]])
        return np.array(sorted(selected,key=lambda i:game_ids[i]),dtype=int)


@dataclass
class ScoreBlend:
    names: list[str]
    weights: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.weights=np.asarray(self.weights,dtype=np.float32)
        if (self.weights.shape != (len(self.names),) or len(set(self.names)) != len(self.names)
                or not self.names or not np.isfinite(self.weights).all() or (self.weights<0).any()
                or not np.isclose(self.weights.sum(),1,atol=1e-6)):
            raise ValueError('Weights must form a finite nonnegative simplex')

    def score(self,channels:Mapping[str,np.ndarray],reference_indices:np.ndarray|None=None)->np.ndarray:
        return normalized_channels(channels,self.names,reference_indices) @ self.weights

    def save(self,path:Path)->None:
        atomic_json(path,{'format_version':1,'names':self.names,'weights':self.weights.tolist(),'metadata':self.metadata})

    @classmethod
    def load(cls,path:Path)->'ScoreBlend':
        obj=json.loads(path.read_text())
        if obj.get('format_version')!=1:raise ValueError('Unsupported blend format')
        return cls(obj['names'],np.array(obj['weights']),obj.get('metadata',{}))


def weight_proposals(names:Sequence[str],seed:int=20261110)->list[np.ndarray]:
    """Finite predeclared search containing the existing baseline and simple ablations."""
    if not names or names[0]!='baseline':raise ValueError('Baseline must be the first channel')
    n=len(names);proposals=[v for v in np.eye(n,dtype=np.float32)]
    for i in range(1,n):
        for fraction in (.05,.1,.25,.5):
            v=np.zeros(n,dtype=np.float32);v[0]=1-fraction;v[i]=fraction;proposals.append(v)
    for i in range(1,n):
        for j in range(i+1,n):
            for fraction in (.1,.25):
                v=np.zeros(n,dtype=np.float32);v[0]=1-fraction;v[i]=v[j]=fraction/2;proposals.append(v)
    # Seed retained in signature for prospective reproducibility; no random test-driven search.
    return proposals


def validation_objective(features:np.ndarray,weights:np.ndarray,target:np.ndarray,
                         allowed:np.ndarray,game_ids:Sequence[str],k:int=20)->float:
    scores=features@weights
    ids=np.asarray(game_ids)
    values=[]
    for row in range(len(scores)):
        candidates=np.flatnonzero(allowed[row])
        order=candidates[np.lexsort((ids[candidates],-scores[row,candidates]))[:k]]
        hits=target[row,order]
        n=int(target[row].sum())
        if n==0:raise ValueError('Each validation query requires a positive target')
        ideal=np.sum(1/np.log2(np.arange(min(k,n))+2))
        values.append(float(np.sum(hits/np.log2(np.arange(len(order))+2))/ideal))
    if not values:raise ValueError('No validation queries')
    return float(np.mean(values))


def fit_blend(names:Sequence[str],features:np.ndarray,target:np.ndarray,allowed:np.ndarray,
              game_ids:Sequence[str])->tuple[ScoreBlend,list[dict]]:
    if features.ndim!=3 or features.shape[:2]!=target.shape or target.shape!=allowed.shape or features.shape[2]!=len(names):
        raise ValueError('Misaligned validation tensors')
    trials=[]
    for weights in weight_proposals(names):
        value=validation_objective(features,weights,target,allowed,game_ids)
        trials.append({'weights':weights.tolist(),'validation_ndcg@20':value})
    best=max(range(len(trials)),key=lambda i:trials[i]['validation_ndcg@20'])
    return ScoreBlend(list(names),np.array(trials[best]['weights']),
                      {'objective':'validation NDCG@20','validation_ndcg@20':trials[best]['validation_ndcg@20'],
                       'trials':len(trials),'ranking_scores_not_probabilities':True}),trials
