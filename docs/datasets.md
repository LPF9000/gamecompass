# Dataset acquisition and reproducibility

The primary application candidate is Anton Kozyriev's **Game Recommendations on Steam**, Kaggle release **28**. Kaggle's public listing API declares **CC0: Public Domain**, last updated **2024-08-14**. Original GameCompass source code licensing does not replace third-party terms. See [data-licenses.md](data-licenses.md) for distinctions and caveats.

The UCSD Australian Steam data is acquired separately for local research. No explicit application/raw-data/derived-model redistribution grant was located on the author page; do not distribute its data, trained weights or catalog in a project release unless permission is resolved. `--research-only` makes this source selection explicit. Never silently mix it into the CC0 production snapshot.

## Commands

Run from the repository after installation:

```bash
# Primary: complete histories for a deterministic development cohort (1/32 source users)
python scripts/acquire_data.py --source kaggle --user-modulus 32 --activate
# Full release: larger disk/time requirements, never implicitly selected
python scripts/acquire_data.py --source kaggle --user-modulus 1
# Separate research source with unresolved redistribution rights
python scripts/acquire_data.py --source ucsd --research-only
```

`--activate` promotes a successfully processed Kaggle snapshot to primary `data/population.sqlite`, preserving an existing UCSD snapshot as `data/ucsd-research.sqlite`. Without it, processing leaves the existing primary untouched. Do not activate while training/evaluation is reading a source that must remain stable.

The pipeline stores raw files under `data/raw/`, source-specific SQLite snapshots, JSON manifests/statistics and RecBole atomic files under `data/recbole/`. All are ignored by Git. Re-running processing builds a new temporary SQLite database and only replaces a successful snapshot atomically. Interrupted processing is restarted from raw source; downloads retain `.part` files and resume where the server supports byte ranges. Retries use bounded backoff. A server that ignores a Range request triggers a fresh overwrite, not a corrupt append.

The default Kaggle mode is explicitly a **development user cohort**, not the full training dataset. Selection is `numeric source user_id % 32 == 0`. It is a systematic deterministic sample, not asserted to be random. Every review from each selected user is retained; every source review is streamed and counted. Use `--user-modulus 1` to retain the full release. Evaluate how this cohort differs from full population before making general accuracy claims.

## Schemas actually inspected

| Source member | Actual fields |
| --- | --- |
| UCSD Australian top level | `user_id`, `user_url`, `reviews` |
| UCSD individual review | `funny`, `posted`, `last_edited`, `item_id`, `helpful`, `recommend`, `review` |
| UCSD game metadata | `app_name`, `developer`, `discount_price`, `early_access`, `genres`, `id`, `metascore`, `price`, `publisher`, `release_date`, `reviews_url`, `sentiment`, `specs`, `tags`, `title`, `url` (not all present per row) |
| Kaggle recommendations.csv | `app_id`, `helpful`, `funny`, `date`, `is_recommended`, `hours`, `user_id`, `review_id` |
| Kaggle games.csv | `app_id`, `title`, `date_release`, `win`, `mac`, `linux`, `rating`, `positive_ratio`, `user_reviews`, `price_final`, `price_original`, `discount`, `steam_deck` |
| Kaggle games_metadata.json | `app_id`, `description`, `tags` |
| Kaggle users.csv | `user_id`, `products`, `reviews` |

The UCSD Australian release has explicit recommendation booleans and often year-bearing review timestamps, but **no review-level playtime**. Reviews missing a year retain NULL timestamps; no year is invented. Kaggle has recommendation booleans, dates and hours, but hours may be measured at collection rather than at review date. Do not use future hours for strict historical prediction without confirming collection semantics.

## Normalization and integrity

Canonical game IDs are exact `steam:<positive integer>` external IDs. Similar titles, editions, DLC and remakes are never merged by title. Game titles without reliable metadata are flagged as missing rather than guessed. Metadata tags are preserved as source tags; they are not silently asserted to be curated gameplay characteristics. Kaggle's `win/mac/linux` flags mean supported operating systems, not PS5/Xbox/Nintendo platforms. No console coverage is inferred.

Users are source-namespaced SHA-256 pseudonyms. Hashing does **not** guarantee anonymity against reidentification; raw datasets must still be handled as potentially personal. User URLs, original user IDs and raw review text do not enter normalized SQLite. Raw source archives still contain source information. Different datasets' users are never merged.

`positive=1` means an explicit recommendation. `positive=0` means explicit non-recommendation. Missing interaction means unobserved. Playtime is stored separately in hours. Duplicate user/game records retain the most recent known timestamp; ties retain the first deterministic source occurrence. Counts and conflicts are recorded for the UCSD research release. Exact deduplication happens in SQLite, so the full source need not fit RAM.

Download manifests record HTTPS source URL, byte count and SHA-256. Source authors do not publish a trusted SHA-256 in the inspected pages; a locally recorded checksum detects later changes but does not independently authenticate the original source. gzip CRC is validated by consuming complete streams. ZIP members consumed by processing are checked by Python's ZIP CRC handling. No unsafe `eval` is used; UCSD literal dictionaries are parsed using `ast.literal_eval`.

RecBole positive files have `user_id:token`, `item_id:token`, `timestamp:float`. Explicit negatives are written separately and must not be accidentally passed to a positive-only sampler as positives. Missing timestamps are exported as `0`; strict temporal evaluation must exclude them, not treat zero as a known date. Dataset export itself does not determine a leakage-safe split; evaluation code owns whole-user and temporal splitting.

## Actual acquisition evidence

- UCSD Australian reviews: **6,935,922 bytes**, SHA-256 `a934bf74d8579bb25bc7bf24e652a87e2ccc91064e429a4cb4af5809178687a8`.
- UCSD games: **2,740,516 bytes**, SHA-256 `717f4653a26585fec8f463f59311a654127c2ed66b07b9f78b4c0ce64961c740`.
- UCSD source: 25,799 user rows and 59,305 reviews. After deduplication: **25,458 users, 3,682 interacted games, 58,431 interactions**; **51,692 positive**, **6,739 negative**, **48,498 known timestamps**, **874 duplicate interactions**. Sparsity over observed users/items: **99.9377%**. Catalog: **32,618 games** including flagged IDs lacking title metadata.
- Kaggle release 28 archive downloaded: **692,315,526 bytes**. Members: `games.csv` **4,863,790**, `games_metadata.json` **17,548,605**, `recommendations.csv` **2,023,223,067**, `users.csv` **193,068,018** uncompressed bytes. SHA-256 `7620a48dbff9fdb53e08c06640fae6060189f458784df646835eff4f69c7995a`.
- Kaggle source scan: **41,154,794 recommendations** (**35,304,398 positive**, **5,850,396 negative**), **14,306,064 user rows**, **50,872 catalog games** and metadata rows.
- Primary Kaggle development cohort: **430,864 users**, **24,868 interacted games**, **1,295,681 deduplicated interactions** (**1,110,813 positive**, **184,868 explicit negative**). One duplicate user/game record removed from 1,295,682 selected source rows. Observed sparsity **99.9879%**; all **50,872 catalog games** available locally. Full source review scan and cohort normalization took **118.139 seconds** in this environment. Final stats are in `data/kaggle-population.stats.json`; active primary `data/population.sqlite` is this CC0 cohort.
- Selected cohort review date span: **2010-10-23 to 2022-12-31**. Recorded playtime ranges **0 to 999.9 hours**; the upper boundary suggests clipping and must not be interpreted as exact uncapped time. Metadata release update 2024 does not make the review histories current to 2024.
- UCSD research snapshot remains at `data/ucsd-research.sqlite`; it is not the primary application catalog/training source.

Sources: [UCSD author dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets.html), [UCSD Australian reviews](https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_user_reviews.json.gz), [UCSD metadata](https://cseweb.ucsd.edu/~wckang/steam_games.json.gz), [Kaggle publisher dataset page](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam), [Kaggle public listing API](https://www.kaggle.com/api/v1/datasets/list?search=game-recommendations-on-steam), [CC0 deed](https://creativecommons.org/publicdomain/zero/1.0/).
