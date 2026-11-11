# Dataset provenance and rights review

Checked 2026-11-10. Source-code licensing does not license data, descriptions, reviews, trademarks, artwork, or trained weights. This is an engineering rights inventory, not a statement that publication alone grants unrestricted rights.

## Candidates

| Dataset | Evidence and expected use | Size and age | Rights / decision |
|---|---|---|---|
| UCSD Steam V1 reviews | Nested user records; explicit boolean recommendations, review text and posted dates verified from downloaded first record | Official compressed link labeled 6.7 MB; actual request returned 6,935,922 bytes; 2017-era research release | Research collection requests citation; no explicit application, raw redistribution or derived-model distribution grant found. Local research only; do not redistribute raw data or weights pending clarification |
| UCSD Steam V1 user/items | Ownership and playtime complement reviews; weaker enjoyment evidence | Official compressed link labeled 71 MB | Same unresolved rights; do not infer that ownership is approval |
| UCSD Steam V2 reviews | Actual first streamed row has username, product_id, hours, date and text, but no recommendation label | Official compressed link labeled 1.3 GB | Same unresolved rights; usernames are not guaranteed stable user identities; do not fabricate vote labels |
| UCSD Steam V2 metadata | Actual first row has id, title/app_name, genres, tags, publisher, developer, release_date, specs, price and URLs | Official compressed link labeled 2.7 MB | Same unresolved rights; artwork rights are separate |
| Anton Kozyriev Game Recommendations on Steam v28 | Recommendation relations plus anonymized users, games.csv and games_metadata.json; stronger candidate for reproducible public pipeline | Kaggle API: updated 2024-08-14; listing totalBytes 692,315,526 vs view totalBytes 2,238,703,480 (different archive/unpacked accounting, not assumed equal); source says >41M reviews | Official Kaggle API declares CC0 Public Domain; preserve source/version/license metadata. Thumbnail belongs to Valve; CC0 declaration does not erase third-party content rights |
| RecBole Steam atomic conversion | Convenient framework input, not independent preference evidence | Depends on converted release | Framework code license is not a dataset license; trace original source and filtering before adoption |
| Large ownership/playtime datasets | Useful for availability and weak implicit evidence, not explicit enjoyment | Many historical snapshots; exact sizes must be verified per acquired release | Not selected merely for volume; verify uploader rights and original terms before use |
| Generic explicit-rating collections | Potentially useful for cross-platform enjoyment prediction | None acquired or verified in this research | No invented counts, availability or license claims |

Full acquired-data counts, sparsity and hashes belong in generated processing statistics. Published counts from different releases must not be substituted for observed processing counts. Do not join user IDs across independent datasets; namespace by source. Steam app IDs can join games when semantics agree; bundles, DLC and remakes remain distinct.

## Official acquisition sources

- [UCSD dataset index](https://cseweb.ucsd.edu/~jmcauley/datasets.html), Steam section. Its aggregate page counts are 7,793,069 reviews, 2,567,538 users, 15,474 items and 615 bundles; these are page-level statistics, not verified counts of each linked file.
- V1 reviews: https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_user_reviews.json.gz
- V1 ownership/playtime: https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_users_items.json.gz
- V2 reviews: https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz
- V2 metadata: https://cseweb.ucsd.edu/~wckang/steam_games.json.gz
- [Kaggle original dataset](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam).
- [Kaggle metadata endpoint](https://www.kaggle.com/api/v1/datasets/view/antonkozyriev/game-recommendations-on-steam), independently requested using Python urllib during research. `licenseName=CC0: Public Domain`, `currentVersionNumber=28`.
- Versioned archive endpoint: https://www.kaggle.com/api/v1/datasets/download/antonkozyriev/game-recommendations-on-steam?datasetVersionNumber=28 . Listing verification alone does not establish successful acquisition; record HTTP response, archive test and hash in the acquisition manifest.
- [RecBole Steam conversion instructions](https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/usage/Steam.md).

## Verified V1 example schema

The downloaded record parses as a Python literal, not strict JSON. Use bounded `ast.literal_eval`, never the source page's unsafe `eval` example. Root keys: `user_id`, `user_url`, `reviews`. Review keys observed: `funny`, `posted`, `last_edited`, `item_id`, `helpful`, `recommend`, `review`. `recommend` was boolean and `posted` included a year. Do not assume every date has a year or every record is valid. V1 review records do not contain per-review playtime; that requires the separate user/items file. Do not preserve profile URLs in deployable model artifacts.

## Verified Kaggle v28 schema

Research requested the versioned per-file archive endpoints and decompressed the first deflate member bytes to inspect real headers (without claiming the complete dataset was acquired by this inspection):

- `recommendations.csv`: `app_id,helpful,funny,date,is_recommended,hours,user_id,review_id`. First inspected record had date `2022-12-12`, boolean text `true`, 36.3 hours, anonymized user ID, and review ID. This is explicit recommendation plus playtime, not an explicit numerical rating. The meaning of `hours` as current-versus-at-review time is not established by the header; do not use it as a strict historical feature without clarification.
- `games.csv`: `app_id,title,date_release,win,mac,linux,rating,positive_ratio,user_reviews,price_final,price_original,discount,steam_deck`. `rating` is aggregate storefront sentiment, not a user's numerical rating. Platform fields concern desktop OS availability and Steam Deck; they do not establish console availability.
- `games_metadata.json`: newline-delimited JSON objects with `app_id`, `description`, `tags`; empty descriptions exist. A description and tags alone are not a comprehensive cross-platform catalog.

Official file listing API reports `games.csv` 4,863,790 bytes; `games_metadata.json` 17,548,605; `recommendations.csv` 2,023,223,067; `users.csv` 193,068,018. A per-file URL appends `/recommendations.csv?datasetVersionNumber=28` to the download endpoint; that response was a ZIP archive with Content-Length 608,515,429. Preserve source compressed bytes and calculate your own SHA256 after complete download; the header inspection is not a full integrity check.

## Rights matrix

| Asset | Research / personal use | Noncommercial app | Distribute raw data | Distribute learned weights | Attribution |
|---|---|---|---|---|---|
| Original project code, MIT | Yes | Yes | N/A | Only original components | Preserve MIT notice |
| UCSD Steam | Research intent evident | Not expressly licensed on inspected page | Unresolved: exclude | Unresolved: exclude from public repo | Cite listed Steam papers |
| Kaggle Kozyriev v28 CC0 declaration | Permitted by declared dedication | Permitted by declared dedication | Declared dedication permits, subject to third-party rights; use downloader | No separate restriction found in dedication; privacy/memorization still relevant | Credit original source for provenance even though CC0 does not require it |
| IGDB data and images | Subject to developer agreement | API docs permit free noncommercial use | No blanket grant verified | Training rights not established | Retain provider notices; verify requirements before packaging |
| MiniLM-L6-v2 weights | Apache-2.0 | Apache-2.0 | Model-weight terms only | Apache terms apply to covered weights | Include license/notices when distributing |

No data purchase is needed for the current implementation. Open-source publication should contain code, configuration and aggregate nonidentifying benchmark outputs; download and train locally. Recheck provider terms before adding any redistribution bundle.
