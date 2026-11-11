"""Local withheld-preference checks, distinct from post-play enjoyment."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from .engine import Recommender
from .store import Store


def evaluate_profile(data_dir: Path, profile_id: str, seed: int = 42,
                     holdout_fraction: float = 0.25) -> dict:
    """Hide some known ratings from fold-in; never edits the user's actual ratings."""
    if not 0 < holdout_fraction < 1:
        raise ValueError('holdout_fraction must be strictly between zero and one')
    store = Store(data_dir)
    model = Recommender(data_dir)._model()
    index = {g:i for i,g in enumerate(model.game_ids)}
    feedback = [f for f in store.feedback(profile_id) if f.get('rating') not in (None,0) and f['game_id'] in index]
    if len(feedback) < 5:
        raise ValueError('At least five mapped nonneutral ratings are required for a small local holdout check.')
    rng = np.random.default_rng(seed)
    # Stratify to preserve positive/negative labels when each group is large enough.
    held, support = [], []
    for sign in (1,-1):
        group = [f for f in feedback if (f['rating'] > 0) == (sign > 0)]
        rng.shuffle(group)
        n = min(len(group)-1, max(1, round(len(group)*holdout_fraction))) if len(group)>1 else 0
        held.extend(group[:n]); support.extend(group[n:])
    scores = model.score({f['game_id']:float(f['rating']) for f in support})
    excluded = {f['game_id'] for f in support}
    candidates = [i for i in np.argsort(-scores,kind='stable') if model.game_ids[i] not in excluded]
    ranks = {model.game_ids[i]:r+1 for r,i in enumerate(candidates)}
    positive = [f for f in held if f['rating']>0]
    negative = [f for f in held if f['rating']<0]
    report = {'kind':'predicting_withheld_already_played_preferences', 'seed':seed,
              'support_count':len(support), 'holdout_count':len(held),
              'candidate_count':len(candidates), 'model':model.selected,
              'recall_at_10':sum(ranks[f['game_id']]<=10 for f in positive)/len(positive) if positive else None,
              'positive_beats_negative':float(np.mean([scores[index[p['game_id']]] > scores[index[n['game_id']]]
                       for p in positive for n in negative])) if positive and negative else None,
              'note':'Small local diagnostic. Does not measure enjoyment of unplayed recommendations.',
              'timestamp':time.time()}
    with store.connection() as db:
        db.execute('CREATE TABLE IF NOT EXISTS personal_evaluations(profile_id TEXT REFERENCES profiles(id) ON DELETE CASCADE, report TEXT, created REAL)')
        db.execute('INSERT INTO personal_evaluations VALUES(?,?,?)', (profile_id,json.dumps(report),time.time()))
    return report


def record_outcome(data_dir: Path, profile_id: str, game_id: str, kind: str, value: int) -> None:
    """Separate recommendation relevance from later post-play enjoyment."""
    if kind not in {'recommendation_relevance','post_play_enjoyment'} or value not in {-2,-1,0,1,2}:
        raise ValueError('Invalid outcome kind/value')
    store = Store(data_dir)
    if profile_id not in {p['id'] for p in store.profiles()} or store.game(game_id) is None:
        raise ValueError('Unknown profile or game')
    with store.connection() as db:
        db.execute('''CREATE TABLE IF NOT EXISTS outcomes(profile_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
                       game_id TEXT, kind TEXT, value INTEGER, created REAL)''')
        db.execute('INSERT INTO outcomes VALUES(?,?,?,?,?)',(profile_id,game_id,kind,value,time.time()))
