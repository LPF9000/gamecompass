# Troubleshooting

- **No trained population model:** finish acquisition/activation, then run `gamecompass benchmark`. Smoke fixtures are tests, never a substitute for acquisition.
- **No recommendations for a platform:** the catalog may have no verified platform mapping for that platform. Steam OS flags do not establish console availability. Disable the filter or add verified metadata; do not assume PS5 availability.
- **Missing API key/private library:** set `STEAM_API_KEY` in your shell and provide SteamID64. Steam visibility must permit requested history. Manual ratings work without credentials.
- **RecBole install/runtime errors:** use the isolated Python 3.11 environment and checked lockfile. The app environment intentionally uses a newer SciPy; old RecBole LightGCN needs SciPy's removed DOK API. Do not mutate global packages.
- **Download interruption:** rerun acquisition; partial downloads use Range only when server responses confirm correct offsets. Checksums and archive validation reject corruption. A locally calculated hash detects changed bytes, not independent source authenticity.
- **Corrupted model:** activation/load checks hashes and array shapes. Retrain or activate an intact prior version. Do not disable verification.
- **High RAM/time:** use the 1/32 complete-user cohort and 1500-item development mode. Dense EASE is explicitly capped; `--user-modulus 1` processes the full corpus and is a separate resource decision.
- **Weak personalized results:** check dataset coverage, sparse ratings and chosen history-specific baseline. A measured popularity winner indicates current limitations, not proof of deep preference understanding. Rate more diverse familiar games and compare personal holdout checks.
- **Offline errors:** acquisition, model downloads and account imports inherently require connectivity. After initial setup, search/rating/inference do not.
