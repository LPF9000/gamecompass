"""Explicit local-only population snapshot incorporating personal feedback."""
from __future__ import annotations
import json
from pathlib import Path
import sqlite3
import time
from .store import Store
from .models import atomic_json, sha256


def incorporate_feedback(data_dir: Path, output_dir: Path) -> dict:
    """Create a separate immutable training input; never silently mutate defaults."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if output_dir.exists():
        raise ValueError('Use a new output directory for each training snapshot')
    source = data_dir/'population.sqlite'
    if not source.exists():
        raise ValueError('Acquire a population dataset first')
    store=Store(data_dir)
    output_dir.mkdir(parents=True)
    destination=output_dir/'population.sqlite'
    try:
        with sqlite3.connect(source) as original, sqlite3.connect(destination) as combined:
            original.backup(combined)
            count=0
            for profile in store.profiles():
                for feedback in store.feedback(profile['id']):
                    rating=feedback.get('rating')
                    if rating in (None,0) or feedback.get('not_interested'):
                        continue
                    game=store.game(feedback['game_id'])
                    combined.execute('INSERT OR IGNORE INTO games VALUES(?,?,?,?,?,?,?)',
                         (game['game_id'],game['title'],game.get('description',''),json.dumps(game.get('platforms',[])),
                          json.dumps(game.get('genres',[])),json.dumps(game.get('themes',[])),json.dumps(game.get('metadata',{}))))
                    combined.execute('INSERT OR REPLACE INTO interactions VALUES(?,?,?,?,?)',
                         ('local:'+profile['id'],game['game_id'],int(rating>0),int(feedback['updated']),None))
                    count+=1
            combined.commit()
        destination.chmod(0o600)
        report={'created':time.time(),'source_sha256':sha256(source),'sha256':sha256(destination),
                'local_explicit_labels':count,'privacy':'contains local personal preferences; do not publish or upload',
                'semantics':'nonneutral numerical sign only; no playtime, ownership, unjudged or not-interested labels'}
        atomic_json(output_dir/'snapshot.json',report)
        return report
    except BaseException:
        # Leave failed snapshot visibly incomplete rather than activating it.
        raise
