# GameCompass

Free, open-source, local-first game recommendation research and a working local application. Python services, SQLite and a Streamlit UI; no hosted inference, subscription or mandatory gaming account. Original code is MIT. Data, model and artwork rights are separate.

**Status: working research application, not a demonstrated state-of-the-art recommender.** Real Steam data has been acquired, processed and benchmarked. Development evaluations use a deterministic complete-user cohort and a bounded item universe. Read measured results and known limitations before interpreting accuracy claims.

## Current follow-up release

V0.2 adds Platform/Genre/Era refinements, edition and platform-release date handling, contextual “If you liked this game”, richer gameplay-tag content features, and bounded explicit-plus-library personalization. The main collaborative ranking remains the measured control. See [follow-up guide](docs/followup-v2.md) and [fresh-user confirmation](docs/fresh-v2.md).

The updated ready-data bundle includes 1,475 separate Wikidata source entities with their own console facts, in addition to the 50,872-game Steam catalog. These entries are metadata-ranked; no console-specific enjoyment accuracy is established. Original-work dates remain unknown unless supported by positive source evidence. A platform-specific era never uses an unqualified global date as a console port date.

## Quick start

Requires Python 3.12 and about 3 GB free disk for default data/workflow (more for optional research environments). CPU supported; no GPU required for the default pipeline. The measured environment has approximately 10 GiB RAM. Windows/macOS installation has not been verified in this session.

```bash
git clone https://github.com/LPF9000/gamecompass.git
cd gamecompass
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
gamecompass doctor
gamecompass acquire --source kaggle --user-modulus 32 --activate
gamecompass benchmark --max-items 1500 --max-users 300 --seed 42
gamecompass app
```

Open `http://127.0.0.1:8501`. Create a profile, search for familiar games and rate them. Enjoyment and activity are separate: owning or completing a game never automatically means loving it. Feedback updates fold-in immediately. Internet is required for initial acquisition and requested service imports; cached inference/search/ratings work offline.

The default download is the publisher-declared CC0 **Game Recommendations on Steam v28** archive (692,315,526 bytes). The pipeline scans all 41,154,794 recommendation rows but retains complete histories for a deterministic 1/32 user cohort: 430,864 users, 1,295,681 interactions and 50,872 catalog games. This is explicitly a development subset. Use `--user-modulus 1` for full processing; full-scale training is not claimed.

Data and models stay under `./data` or `GAMECOMPASS_DATA`. Nothing private is committed. Raw data and trained artifacts are intentionally downloaded/trained locally rather than bundled in Git. Repeated acquisition reuses validated files; versioned models persist across restarts.

## Application

- Home: recommendations, recent feedback, profile and account status.
- Discover: platform filtering, played/unwanted exclusions, explanations and separate content exploration.
- Search: game search, five enjoyment levels, insufficient experience, not interested, reasons and activity controls.
- My Games: editable history, ratings and imported playtime.
- Settings: profiles, Steam/manual imports, exports, deletion, personal holdout checks and diagnostics.

Platform filters use verified catalog mappings. The acquired Steam catalog establishes Windows/macOS/Linux availability; it does not establish console releases. Optional IGDB enrichment needs credentials and terms review. Covers are not invented or scraped.

## Accounts and manual data

Manual rating works without accounts. Optional Steam:

```bash
export STEAM_API_KEY='YOUR_LOCAL_KEY'
gamecompass profile 'My profile'  # prints its local ID
gamecompass steam PROFILE_ID STEAMID64
```

Steam visibility must permit the requested library. Passwords and session cookies are never requested. Recent activity and game achievements are available on demand. Live authorized imports remain unverified until a real key/account is supplied.

JSON export/import and CLI examples:

```bash
gamecompass export PROFILE_ID my-profile.json
gamecompass import my-profile.json
gamecompass personal-test PROFILE_ID
```

The UI also accepts manual CSV imports. See [application guide](docs/application.md) for fields and rating semantics.

## Evidence and reproducibility

- [Delivery report](docs/delivery.md), [execution plan](PLAN.md) and [acceptance status](docs/STATUS.md)
- [Actual native benchmark](docs/benchmarks.md) and [RecBole BPR/LightGCN](docs/recbole.md)
- [Semantic comparison](docs/semantic.md), [implicit-history adapter](docs/benchmarks-implicit.md), [temporal benchmark](docs/benchmarks-temporal.md), [cold-item checks](docs/benchmarks-cold-items.md) and [supervised projection](docs/content-projection.md)
- [Dataset acquisition/provenance](docs/datasets.md) and [licensing](docs/data-licenses.md)
- [Research](docs/research.md), [architecture](docs/architecture.md), [catalog](docs/catalog.md)
- [Training](docs/training.md), [evaluation](docs/evaluation.md), [integrations](docs/integrations.md)
- [Privacy](docs/privacy.md), [troubleshooting](docs/troubleshooting.md), [limitations and roadmap](docs/limitations.md)

```bash
python -m pytest -q
gamecompass temporal-benchmark --max-items 1500
```

Benchmarks report Recall/NDCG/MAP, coverage, novelty, feature diversity, long-tail and explicit-negative diagnostics with confidence intervals where applicable. Whole users are held out from population fitting. Validation selects settings; test results do not tune them. Scores are rankings, never probabilities. Offline review retrieval does not establish real-world enjoyment.

No purchases or paid services are needed. Optional research dependencies and data use their own licenses; noncommercial intent does not override those terms. A public web service, mobile app and commercial infrastructure are outside the current scope.
