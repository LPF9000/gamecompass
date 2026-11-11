"""Compare a locally executed semantic encoder with signed TF-IDF cold-user profiles.

Runs on the exact same prepare()/cold_queries() split and eligible item universe
as the native benchmark. It does not tune on test users or activate production
models. Downloaded weights and local artifacts remain outside Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from gamecompass.evaluation import HISTORY_LENGTHS, cold_queries, prepare, ranking_metrics
from gamecompass.models import atomic_json, game_text, sha256

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Immutable official model commit; changed only after an explicit new experiment.
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def signed_centroid_scores(features: np.ndarray | sparse.csr_matrix,
                           support: Mapping[str, float], index: Mapping[str, int]) -> np.ndarray:
    """Signed support is supplied explicitly; unobserved items are never negatives."""
    known = [(index[g], float(w)) for g, w in support.items() if g in index and np.isfinite(w) and w != 0]
    if not known:
        return np.zeros(features.shape[0], dtype=np.float32)
    rows, weights = zip(*known)
    centroid = np.asarray(features[list(rows)].T @ np.array(weights, dtype=np.float32)).ravel()
    norm = np.linalg.norm(centroid)
    if norm:
        centroid /= norm
    return np.asarray(features @ centroid).ravel()


def evaluate_features(features: np.ndarray | sparse.csr_matrix, ids: list[str],
                      queries: list, seed: int = 42) -> dict:
    """Full eligible ranking, excluding every support item; no negative sampling."""
    index = {g:i for i,g in enumerate(ids)}
    id_array = np.array(ids)
    records = []
    covered = set()
    for support, seen, targets, negatives in queries:
        scores = signed_centroid_scores(features, support, index)
        candidates = np.array([i for i,g in enumerate(ids) if g not in seen], dtype=int)
        ranked = candidates[np.lexsort((id_array[candidates], -scores[candidates]))[:20]]
        ranking = [ids[i] for i in ranked]
        records.append({**ranking_metrics(ranking, targets, 10), **ranking_metrics(ranking, targets, 20),
                        "explicit_negative_fraction@20":len(set(ranking) & negatives)/max(1,len(ranking))})
        covered.update(ranking)
    if not records:
        return {"users":0,"status":"insufficient history in this cohort"}
    rng = np.random.default_rng(seed)
    metrics = {}
    for key in records[0]:
        values = np.array([r[key] for r in records], dtype=float)
        boot = np.mean(values[rng.integers(0,len(values),size=(200,len(values)))],axis=1)
        metrics[key] = {"mean":float(values.mean()),"ci95":np.quantile(boot,[0.025,0.975]).tolist(),"n":len(values)}
    return {"users":len(records),"metrics":metrics,"catalog_coverage@20":len(covered)/len(ids)}


def run(data_dir: Path, max_items: int = 1500, max_users: int = 300, seed: int = 42,
        revision: str = MODEL_REVISION, local_files_only: bool = False) -> dict:
    started = time.monotonic()
    source = data_dir / "population.sqlite"
    source_hash = sha256(source)
    prepared = prepare(data_dir, max_items, seed)
    if sha256(source) != source_hash:
        raise RuntimeError("Population database changed during split preparation; rerun on a fixed snapshot")
    ids = prepared["cf_ids"]
    all_texts = [game_text(g) for g in prepared["games"]]
    all_index = {g["game_id"]:i for i,g in enumerate(prepared["games"])}
    texts = [all_texts[all_index[g]] for g in ids]
    key = hashlib.sha256(json.dumps({"ids":ids,"texts":texts,"model":MODEL_ID,"revision":revision},sort_keys=True).encode()).hexdigest()
    artifact_dir = data_dir / "semantic" / key
    artifact_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = artifact_dir / "embeddings.npy"
    embedding_manifest = artifact_dir / "embeddings.json"
    if embedding_path.exists() and embedding_manifest.exists():
        saved = json.loads(embedding_manifest.read_text())
        if saved["sha256"] != sha256(embedding_path):
            raise ValueError("Semantic embedding cache failed its integrity check")
        embeddings = np.load(embedding_path,allow_pickle=False)
        if saved.get("game_ids") != ids:
            saved["game_ids"] = ids
            atomic_json(embedding_manifest,saved)
    else:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(MODEL_ID, revision=revision, device="cpu",
                                      cache_folder=str(data_dir/"model-cache"),
                                      local_files_only=local_files_only, trust_remote_code=False,
                                      model_kwargs={"use_safetensors":True})
        embeddings = encoder.encode(texts,batch_size=32,normalize_embeddings=True,
                                    show_progress_bar=True,convert_to_numpy=True).astype(np.float32)
        # Empty descriptions have no legitimate semantic features; zero their rows.
        embeddings[[i for i,text in enumerate(texts) if not text.strip()]] = 0
        tmp = embedding_path.with_suffix('.npy.tmp')
        with tmp.open('wb') as f:
            np.save(f,embeddings,allow_pickle=False)
        os.replace(tmp,embedding_path)
        atomic_json(embedding_manifest,{"key":key,"model":MODEL_ID,"revision":revision,
                                        "sha256":sha256(embedding_path),"items":len(ids),"game_ids":ids})
    if embeddings.shape != (len(ids),384) or not np.isfinite(embeddings).all():
        raise ValueError("Semantic feature shape or values invalid")
    # Same baseline as Model.fit: fit static catalog TF-IDF, then project to CF scope.
    vectorizer = HashingVectorizer(n_features=4096,alternate_sign=False,norm=None,
                                  stop_words="english",ngram_range=(1,2))
    tfidf = TfidfTransformer().fit_transform(vectorizer.transform(all_texts)).astype(np.float32).tocsr()
    tfidf = tfidf[[all_index[g] for g in ids]]
    validation_queries = [query for history in (1,3,5)
                          for query in cold_queries(prepared["validation_users"],history,max_users)]
    validation = {"minilm":evaluate_features(embeddings,ids,validation_queries,seed),
                  "tfidf":evaluate_features(tfidf,ids,validation_queries,seed)}
    results = {}
    for history in HISTORY_LENGTHS:
        queries = cold_queries(prepared["test_users"],history,max_users)
        results[str(history)] = {"minilm":evaluate_features(embeddings,ids,queries,seed),
                                 "tfidf":evaluate_features(tfidf,ids,queries,seed)}
    report = {"model":MODEL_ID,"revision":revision,"license":"Apache-2.0 (official model card)",
              "dataset_sha256":source_hash,"seed":seed,"eligible_items":len(ids),
              "source_interactions":prepared["source_interactions"],"source_users":prepared["source_users"],
              "max_evaluation_users_per_cohort":max_users,"text_fields":"shared game_text: descriptions, genres and themes; no title",
              "candidate_strategy":"same prepare/cold_queries as native benchmark; full training-user-selected eligible catalog excluding support; no sampled negatives",
              "population_users":"80/10/10 seeded disjoint user buckets; heldout users contribute no population interactions",
              "scope":"development semantic content-only comparison; no model activation or hybrid tuning",
              "zero_history":"zero-vector ties resolve by canonical ID; nonpersonalized content control, not population baseline",
              "validation_pooled_support_1_3_5":validation,"cold_test":results,"duration_seconds":time.monotonic()-started,
              "limitations":["Static metadata can postdate historical reviews", "Pretrained text model may have seen game descriptions; pretraining corpus cannot be fully audited for game text", "Not fine-tuned on gaming preferences", "Bounded popular-item universe omits cold and long-tail catalog items", "Not a calibrated enjoyment probability", "No temporal population experiment in this script"],
              "embedding_key":key}
    atomic_json(artifact_dir/"results.json",report)
    atomic_json(data_dir/"semantic-results.json",report)
    write_report(report,Path(__file__).resolve().parents[1]/"docs"/"semantic.md")
    return report


def write_report(report: dict, path: Path) -> None:
    lines = ["# Measured semantic content experiment", "",
             f"Model: [{report['model']}](https://huggingface.co/{report['model']}) at `{report['revision']}`; official model card declares Apache-2.0.", "",
             f"Real population snapshot: {report['source_interactions']:,} interactions / {report['source_users']:,} users. SHA-256: `{report['dataset_sha256']}`.",
             f"Eligible candidate universe: {report['eligible_items']:,} training-selected games; at most {report['max_evaluation_users_per_cohort']} heldout users per cohort. CPU elapsed {report['duration_seconds']:.1f} seconds.", "",
             "Both methods use signed weighted item centroids, the same metadata fields, disjoint test users, identical chronological support prefixes, and the full identical eligible item universe with support items removed. No test-set tuning or production activation occurred. The TF-IDF method uses the native 4096-bin hashing baseline with IDF fit on static catalog metadata. MiniLM runs locally; only its initial official weights download needs network access.", "",
             "| Support reviews | Users | TF-IDF Recall@20 | MiniLM Recall@20 | TF-IDF NDCG@20 | MiniLM NDCG@20 |", "|---:|---:|---:|---:|---:|---:|"]
    for history, methods in report["cold_test"].items():
        a,b = methods['tfidf'],methods['minilm']
        if not a.get('users'):
            lines.append(f"| {history} | 0 | — | — | — | — |")
            continue
        vals=[a['metrics']['recall@20']['mean'],b['metrics']['recall@20']['mean'],a['metrics']['ndcg@20']['mean'],b['metrics']['ndcg@20']['mean']]
        lines.append(f"| {history} | {a['users']} | " + ' | '.join(f'{v:.4f}' for v in vals)+' |')
    validation = report["validation_pooled_support_1_3_5"]
    lines.extend(["", "## Validation comparison", "",
                  "Pooled support lengths 1, 3 and 5; a validation user may contribute more than one query. No test results are used for choosing an encoder.", "",
                  f"TF-IDF NDCG@20: {validation['tfidf']['metrics']['ndcg@20']['mean']:.4f}; MiniLM NDCG@20: {validation['minilm']['metrics']['ndcg@20']['mean']:.4f}."])
    lines.extend(["", "Zero-history content controls have zero vectors and canonical-ID tie order. They are not the population popularity baseline. Detailed Recall/NDCG/MAP@10/20, explicit-negative diagnostics and bootstrap intervals are saved locally in `data/semantic-results.json`.","", "## Reproduce", "", "```bash", "uv venv .venv-semantic --python 3.11", "uv pip install --python .venv-semantic/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu", "uv pip install --python .venv-semantic/bin/python -r configs/semantic-requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match", "PYTHONPATH=src .venv-semantic/bin/python scripts/benchmark_semantic.py --data-dir data --max-items 1500 --max-users 300", "```", "", "Add `--local-files-only` after the initial successful acquisition. Existing embeddings are reused only when their content key and checksum match.", "", "## Limits", ""])
    lines.extend('- '+s for s in report['limitations'])
    path.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    parser.add_argument('--max-items',type=int,default=1500)
    parser.add_argument('--max-users',type=int,default=300)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--revision',default=MODEL_REVISION)
    parser.add_argument('--local-files-only',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.data_dir,args.max_items,args.max_users,args.seed,args.revision,args.local_files_only),indent=2))
