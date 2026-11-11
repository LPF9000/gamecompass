# Resuming RecBole training

New training runs from `scripts/train_recbole.py` save each epoch atomically as an own-training `.pth` checkpoint plus `.pth.json` provenance sidecar. The sidecar records the normalized positive-training export SHA-256, algorithm, seed, effective model/optimizer settings, completed epoch, checkpoint SHA-256, and Python/NumPy/Torch and dataloader random-number generator states.

To continue an interrupted BPR run, choose its checkpoint from `data/recbole/BPR/checkpoints/` and pass it explicitly:

```sh
.venv-recbole/bin/python scripts/train_recbole.py --algorithms BPR --max-items 1500 --max-users 300 --epochs 50 --seed 42 --resume-checkpoint data/recbole/BPR/checkpoints/BPR-TIMESTAMP.pth
```

Replace `BPR-TIMESTAMP.pth` with the filename created by your run. For LightGCN choose `--algorithms LightGCN` and its matching checkpoint directory. `--epochs` is the total target epoch budget, not an additional count. An already completed budget is rejected; use `--evaluate-only` for metric reruns. Checkpoint creation and loss values remain local.

The script refuses a changed source training export, changed effective configuration, an outside-directory checkpoint, missing sidecar, or a mismatched file checksum before invoking RecBole's checkpoint loader. It restores model/optimizer progress and the CPU and dataloader RNG states after loading. Exact bitwise reproducibility across different hardware/library builds is not guaranteed.

RecBole 1.2.1 serializes its own checkpoints with Python pickle. Resume only checkpoints created by this script in your own model directory. Never download or copy third-party `.pth` files into it. Application inference uses safe NumPy item-factor exports, not pickle. The early research runs made before this guarded checkpoint writer existed have no provenance sidecars and cannot use this resume command; retrain them with the current script. Final model selection is independent of checkpoint resumption.

Verified locally with a small, explicitly synthetic **test fixture only**: train one BPR epoch, save own checkpoint, resume to two total epochs, confirm training begins at epoch index 1, and compare against uninterrupted two-epoch training: item-factor arrays were exactly equal (maximum absolute difference 0.0) in the tested Linux CPU environment. The optional automated RecBole resume test reproduces this check; it skips when Torch/RecBole are absent. Ordinary automated tests cover source/config mismatch, checksum corruption, and outside-directory rejection without external services or Torch. No fixture contributes to real-data metrics.
