#!/usr/bin/env python3
"""Train actual RecBole models on the exact population split used by benchmarks.

This is a bounded development experiment, not a full-corpus accuracy claim.
The local application can use exported item embeddings without importing Torch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_resume(checkpoint: Path, checkpoint_dir: Path, expected: dict) -> dict:
    """Validate our registered local checkpoint before any pickle is loaded."""
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_relative_to(checkpoint_dir.resolve()) or checkpoint.suffix != ".pth":
        raise ValueError("Resume only checkpoints created in this model's local checkpoint directory")
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".json")
    provenance = json.loads(sidecar.read_text())
    if provenance.get("signature") != expected:
        raise ValueError("Checkpoint source export/configuration mismatch; retrain instead")
    if provenance.get("checkpoint_sha256") != file_sha256(checkpoint):
        raise ValueError("Checkpoint integrity check failed; use a verified checkpoint")
    return provenance


def export_atomic(rows: Iterable, destination: Path) -> dict:
    """Export only observed positive preference records; never convert dislikes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    users: set[str] = set()
    items: set[str] = set()
    count = 0
    tmp = destination.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        out.write("user_id:token\titem_id:token\ttimestamp:float\n")
        for user, game, positive, timestamp, *_ in rows:
            if not positive:
                continue
            if any(c in str(user) + str(game) for c in "\t\r\n"):
                raise ValueError("Invalid identifier in atomic data")
            out.write(f"{user}\t{game}\t{int(timestamp or 0)}\n")
            users.add(user)
            items.add(game)
            count += 1
    os.replace(tmp, destination)
    return {"users": len(users), "items": len(items), "positive_interactions": count,
            "sha256": file_sha256(destination)}


def train_one(algorithm: str, atomic_dir: Path, output_dir: Path, epochs: int, seed: int, resume_checkpoint: Path | None = None):
    import numpy as np
    import torch
    from recbole.config import Config
    from recbole.data import create_dataset
    from recbole.data.dataloader import TrainDataLoader
    from recbole.sampler import Sampler
    from recbole.trainer import Trainer
    from recbole.utils import get_model, init_seed

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    init_seed(seed, True)
    params = {
        "data_path": str(atomic_dir.parent),
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id", "TIME_FIELD": "timestamp",
        "epochs": epochs, "seed": seed,
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "log_wandb": False, "show_progress": False,
        "save_dataset": False, "save_dataloaders": False,
    }
    if algorithm == "LightGCN":
        params["train_batch_size"] = 1048576  # One full-positive graph update per epoch for this cohort.
    config = Config(model=algorithm, dataset=atomic_dir.name, config_file_list=[str(Path(__file__).resolve().parents[1] / "configs/recbole-development.yaml")], config_dict=params)
    dataset = create_dataset(config)
    train_dataset, _, _ = dataset.build()
    sampler = Sampler(["train"], [train_dataset]).set_phase("train")
    train_loader = TrainDataLoader(config, train_dataset, sampler, shuffle=True)
    model = get_model(algorithm)(config, train_dataset).to(config["device"])
    signature = {"algorithm": algorithm, "seed": seed,
                 "atomic_export_sha256": file_sha256(atomic_dir / f"{atomic_dir.name}.inter"),
                 "parameters": {key: config[key] for key in ("embedding_size", "n_layers", "learning_rate", "train_batch_size", "train_neg_sample_args", "learner", "weight_decay", "reg_weight", "clip_grad_norm", "enable_amp")}}

    class SafeCheckpointTrainer(Trainer):
        def _save_checkpoint(self, epoch, verbose=True, **kwargs):
            import random
            final = Path(self.saved_model_file)
            tmp = final.with_suffix(final.suffix + ".tmp")
            super()._save_checkpoint(epoch, verbose=verbose, saved_model_file=str(tmp))
            os.replace(tmp, final)
            state = np.random.get_state()
            rng = {"numpy": [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
                   "torch": torch.get_rng_state().tolist(), "loader": train_loader.generator.get_state().tolist(), "python": random.getstate()}
            write_json(final.with_suffix(final.suffix + ".json"),
                       {"signature": signature, "epoch": epoch, "rng": rng, "checkpoint_sha256": file_sha256(final)})

    trainer = SafeCheckpointTrainer(config, model)
    if resume_checkpoint is not None:
        import random
        provenance = validate_resume(resume_checkpoint, output_dir / "checkpoints", signature)
        if provenance["epoch"] + 1 >= epochs:
            raise ValueError("Checkpoint already meets the requested epoch budget; use --evaluate-only or increase --epochs")
        # The sidecar/hash check above restricts this to our own registered local
        # Torch checkpoint. Never copy third-party .pth files into this directory.
        trainer.resume_checkpoint(str(resume_checkpoint))
        state = provenance["rng"]["numpy"]
        np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), state[2], state[3], state[4]))
        torch.set_rng_state(torch.tensor(provenance["rng"]["torch"], dtype=torch.uint8))
        if "loader" in provenance["rng"]:
            train_loader.generator.set_state(torch.tensor(provenance["rng"]["loader"], dtype=torch.uint8))
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value
        random.setstate(tuples(provenance["rng"]["python"]))
    started = time.perf_counter()
    # Epoch count is fixed in advance. There is no test-set early stopping.
    trainer.fit(train_loader, valid_data=None, saved=True, show_progress=False)
    model.eval()
    with torch.no_grad():
        if algorithm == "LightGCN":
            _, item_factors = model.forward()
        else:
            item_factors = model.item_embedding.weight
        factors = item_factors.detach().cpu().numpy()
    item_tokens = dataset.field2id_token["item_id"].tolist()
    # RecBole reserves index zero for padding; preserve exact canonical token alignment.
    indices = {str(token): i for i, token in enumerate(item_tokens) if i}
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_factors = output_dir / "factors.npz.tmp"
    with tmp_factors.open("wb") as stream:
        np.savez_compressed(stream, game_ids=np.array(list(indices)), factors=factors[list(indices.values())])
    os.replace(tmp_factors, output_dir / "factors.npz")
    manifest = {"algorithm": algorithm, "epochs": epochs, "seed": seed,
                "training_seconds": time.perf_counter() - started,
                "resumed_from_epoch": int(trainer.start_epoch),
                "hyperparameters": {key: config[key] for key in ("embedding_size", "n_layers", "learning_rate", "train_neg_sample_args")},
                "train_batch_size": int(config["train_batch_size"]), "train_users": int(dataset.user_num - 1), "train_items": int(dataset.item_num - 1),
                "train_interactions": int(len(train_dataset)),
                "train_loss": {str(k): float(v) for k, v in trainer.train_loss_dict.items()},
                "factor_sha256": file_sha256(output_dir / "factors.npz"),
                "backend": str(config["device"]), "recbole_version": "1.2.1", "torch_version": torch.__version__,
                "protocol": "fixed epochs; exact benchmark population training users only; signed mean item-embedding fold-in for unseen users",
                "negative_sampling": "Uniform sampled unobserved items are contrastive alternatives, not observed dislike labels. Explicit dislikes are excluded from positive export.",
                "redistribution": "See source dataset provenance/license report before redistributing weights; no artifacts committed"}
    write_json(output_dir / "manifest.json", manifest)
    return manifest, indices, factors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-items", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-users", type=int, default=300)
    parser.add_argument("--resume-checkpoint", type=Path, help="Continue one algorithm from its own registered local checkpoint; never use third-party .pth files")
    parser.add_argument("--evaluate-only", action="store_true", help="Reuse local factors after verifying the exact training export hash")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithms", nargs="+", default=["BPR", "LightGCN"], choices=["BPR", "LightGCN"])
    args = parser.parse_args()
    if args.epochs < 1 or args.max_items < 2 or args.max_users < 1:
        parser.error("epochs and max-users must be positive; max-items >=2")
    if args.resume_checkpoint and (len(args.algorithms) != 1 or args.evaluate_only):
        parser.error("--resume-checkpoint requires exactly one --algorithms value and cannot combine with --evaluate-only")
    from gamecompass.evaluation import prepare
    prepared = prepare(args.data_dir, max_items=args.max_items, seed=args.seed)
    # Shared benchmark preparation is the sole source of population membership.
    rows = prepared["train_rows"]
    root = args.data_dir / "recbole"
    atomic_dir = root / "gamecompass"
    export_stats = export_atomic(rows, atomic_dir / "gamecompass.inter")
    import numpy as np
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from gamecompass.models import Model, game_text
    from gamecompass.evaluation import cold_queries, evaluate_queries, interaction_matrix, HISTORY_LENGTHS
    base = Model()
    base.game_ids = [g["game_id"] for g in prepared["games"]]
    base.cf_game_ids = prepared["cf_ids"]
    base._indexes()
    base.popularity = np.zeros(len(base.game_ids), dtype=np.float32)
    base.popularity[base._cf_out] = np.asarray(interaction_matrix(rows, base.cf_game_ids).sum(axis=0)).ravel()
    vectorizer = HashingVectorizer(n_features=4096, alternate_sign=False, norm=None, stop_words="english", ngram_range=(1, 2))
    base.content = TfidfTransformer().fit_transform(vectorizer.transform([game_text(g) for g in prepared["games"]])).astype(np.float32).tocsr()
    results = {"scope": "development", "atomic_export": export_stats, "models": {},
               "candidate_strategy": "Same full training-user-selected eligible item universe and complete held-out users as gamecompass benchmark; configured maximum queries per cohort",
               "item_count": len(base.cf_game_ids), "source_users": prepared["source_users"],
               "method": "Fixed epochs chosen before evaluation. No per-user embedding from validation/test exists. Signed sum of learned item embeddings produces unseen-user scores. Not probabilities."}
    source_manifest = json.loads((args.data_dir / "active-population.json").read_text()) if (args.data_dir / "active-population.json").exists() else {}
    results["source"] = source_manifest
    results["max_evaluation_users_per_cohort"] = args.max_users
    previous = json.loads((root / "results.json").read_text()) if args.evaluate_only else None
    if previous and previous["atomic_export"]["sha256"] != export_stats["sha256"]:
        raise ValueError("Current population split differs from existing trained factors; retrain instead")
    for algorithm in args.algorithms:
        if args.evaluate_only:
            manifest = json.loads((root / algorithm / "manifest.json").read_text())
            factor_file = root / algorithm / "factors.npz"
            if file_sha256(factor_file) != manifest["factor_sha256"]:
                raise ValueError("Trained factor checksum mismatch; restore or retrain")
            with np.load(factor_file, allow_pickle=False) as stored:
                indices = {str(g): i for i, g in enumerate(stored["game_ids"])}
                all_factors = stored["factors"].copy()
        else:
            manifest, indices, all_factors = train_one(algorithm, atomic_dir, root / algorithm, args.epochs, args.seed, args.resume_checkpoint)
        manifest["source"] = source_manifest
        manifest["atomic_export_sha256"] = export_stats["sha256"]
        base.factors = np.stack([all_factors[indices[g]] for g in base.cf_game_ids]).astype(np.float32)
        validation_queries = [q for n in (1, 3, 5) for q in cold_queries(prepared["validation_users"], n, args.max_users)]
        manifest["validation"] = evaluate_queries(base, validation_queries, "svd", args.seed)
        manifest["cold_test"] = {str(n): evaluate_queries(base, cold_queries(prepared["test_users"], n, args.max_users), "svd", args.seed) for n in HISTORY_LENGTHS}
        write_json(root / algorithm / "manifest.json", manifest)
        results["models"][algorithm] = manifest
        print(json.dumps({"algorithm": algorithm, **manifest}, indent=2), flush=True)
    write_json(root / "results.json", results)
    lines = ["# Actual RecBole development experiments", "", f"Fixed {args.epochs} epochs; seed {args.seed}; {export_stats['users']:,} population users, {export_stats['items']:,} eligible games, {export_stats['positive_interactions']:,} positive interactions.", "", results["candidate_strategy"], "", results["method"], "", "| Model | Seconds | Validation NDCG@20 | Cold support=1 NDCG@20 | Cold support=3 NDCG@20 |", "|---|---:|---:|---:|---:|"]
    for name, result in results["models"].items():
        lines.append(f"| {name} | {result['training_seconds']:.1f} | {result['validation']['metrics']['ndcg@20']['mean']:.5f} | {result['cold_test']['1']['metrics']['ndcg@20']['mean']:.5f} | {result['cold_test']['3']['metrics']['ndcg@20']['mean']:.5f} |")
    lines += ["", "All metrics and 95% query bootstrap intervals: `data/recbole/results.json`. Full-catalog ranking within the bounded eligible universe; no sampled evaluation negatives. Fixed shallow training is a development baseline, not a tuned neural-model ceiling. Zero-history uses the same population popularity baseline. Cohorts without sufficient held-out history are explicitly reported unavailable.", "", "Local artifacts: `data/recbole/{BPR,LightGCN}/factors.npz` (safe NumPy arrays) and own-training checkpoints under checkpoints/. Check source data provenance before distribution; no pretrained weights are committed.", "", "## Reproduction", "", "```sh", "uv python install 3.11", "uv venv .venv-recbole --python 3.11", "uv pip install --python .venv-recbole/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -r configs/recbole-requirements.lock", f".venv-recbole/bin/python scripts/train_recbole.py --max-items {args.max_items} --epochs {args.epochs} --max-users {args.max_users} --seed {args.seed}", "```", "", "The Kaggle development run uses BPR batch size 32768 and LightGCN full batch size 1048576 (one graph update per epoch for 658217 positives) to bound dense embedding-optimizer and repeated full-graph costs on CPU. This is a fixed-budget baseline, not an extensively tuned setting. RecBole 1.2.1 uses removed SciPy DOK `_update` APIs; this separate environment pins SciPy 1.10.1 and pandas 2.2.3. Main app environment stays independent. Linux CPU execution verified. This lock contains the official Linux torch +cpu build; macOS needs the corresponding platform Torch 2.5.1 wheel rather than +cpu. CUDA/MPS are not tested by these experiments."]
    (Path(__file__).resolve().parents[1] / "docs/recbole.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
