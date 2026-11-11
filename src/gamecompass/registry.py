"""Explicit model activation and rollback; candidate models must pass evaluation gates."""
from __future__ import annotations
import json
from pathlib import Path
import time
from .models import Model, atomic_json


def activate(data_dir: Path, version: str, minimum_validation_ndcg: float | None = None) -> None:
    root = (Path(data_dir)/'models').resolve()
    candidate = (root/version).resolve()
    if candidate.parent != root or not candidate.is_dir():
        raise ValueError('Specify an existing model version directory directly under models/')
    model = Model.load(candidate)  # integrity check before changing the production pointer
    pointer = root/'current.json'
    old = json.loads(pointer.read_text()) if pointer.exists() else None
    if minimum_validation_ndcg is not None:
        report = candidate/'evaluation.json'
        if not report.exists():
            raise ValueError('Evaluation report required for quality-gated activation')
        score = json.loads(report.read_text()).get('validation_ndcg_at_20')
        if score is None or score < minimum_validation_ndcg:
            raise ValueError('Candidate did not pass validation quality gate')
    atomic_json(pointer,{'version':version,'selected':model.selected,'activated':time.time()})
    with (root/'activation-history.jsonl').open('a') as f:
        f.write(json.dumps({'previous':old,'current':version,'timestamp':time.time()})+'\n')
