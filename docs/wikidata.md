# Free Wikidata catalog enrichment

Wikidata structured metadata was acquired from the official SPARQL service and applied to the local game catalog as **separate `wikidata:Q…` source-entity entries**. Existing Steam game platforms, edition dates, population training data and user profiles were not changed. This adds useful console coverage without equating a Steam edition with every platform release of its underlying work.

## Verified primary-source properties and rights

| Property | Official meaning used | Handling |
|---|---|---|
| [P1733](https://www.wikidata.org/wiki/Property:P1733) | Steam application identifier | Exact external-ID relation; never fuzzy title merging |
| [P400](https://www.wikidata.org/wiki/Property:P400) | Platform of a work or specific software version | Apply only to the source entity owning that statement |
| [P577](https://www.wikidata.org/wiki/Property:P577) | First publication/release of the described work/product | Preserve all statements, precision and qualifiers; do not infer original-work identity |
| [P629](https://www.wikidata.org/wiki/Property:P629) | Version, edition or translation of another entity | Preserve explicit edition relationship; do not inherit parent facts |
| [P747](https://www.wikidata.org/wiki/Property:P747) | Has a version, edition or translation | Preserve work/edition distinctions |
| [P31](https://www.wikidata.org/wiki/Property:P31) | Instance/type membership | Require an explicit [video game Q7889](https://www.wikidata.org/wiki/Q7889) statement; unknown narrower class ancestry remains reviewable |
| [P136](https://www.wikidata.org/wiki/Property:P136) | Creative-work genre | Copy source labels as genres; never invent themes |

[Wikidata licensing](https://www.wikidata.org/wiki/Wikidata:Licensing) and [data access documentation](https://www.wikidata.org/wiki/Wikidata:Data_access) explicitly release structured main/Property namespace data under **CC0**; attribution is appreciated and included. Original GameCompass code remains **MIT**. This acquisition does not include article text, covers, images or other media, whose licenses differ. Dataset rights are separate from the code license.

## Commands

```bash
# Cache metadata for up to 1,500 current collaborative candidate Steam IDs
python scripts/enrich_wikidata.py --data-dir data --limit 1500
# Add separate Wikidata source entities to the local catalog
python scripts/enrich_wikidata.py --data-dir data --limit 1500 --apply
```

No account, API key, subscription or payment is required. The cached catalog works offline. Queries use the public repository URL in an identifying User-Agent, at least one second between requests, at most three attempts, bounded exponential backoff/Retry-After, and up to25 exact Steam IDs per query. A persistent denied/unavailable service stops acquisition rather than generating thousands of further requests. Successful batches are atomically cached and reused on restart. Full statement responses remain under ignored `data/wikidata/batches/`; compact public catalog metadata is `data/wikidata/catalog.json`.

Current Wikidata proper-name labels often use multilingual defaults (`mul`), including PlayStation5, NintendoSwitch and many game names. The pipeline requests `en,mul` and resolves missing labels through bounded cached label queries. This follows [official default-label guidance](https://www.wikidata.org/wiki/Help:Default_values_for_labels_and_aliases/en). Labels are never guessed from QIDs. Earlier successful en-only statement caches are reused; only missing labels require additional queries.

## Identity and date policy

A source entity can describe a work, collection, version or edition. An exact Steam app ID does not prove that all platforms or publication dates of that entity apply to the particular Steam edition. Therefore:

- Each source entity gets its own canonical `wikidata:Q…` ID and its **own** P400/P577 facts. No Steam platforms/dates are overwritten.
- Ambiguous Steam-to-multiple-entity relations remain in `mapping-review.json`; they are never silently merged. Separate upstream entities may still be listed using their own source facts.
- P629/P747/type statements and source statement IDs, ranks, qualifiers, reference URLs, query hashes and retrieval timestamps remain attached. Deprecated statements are omitted.
- Original-work release date stays **unknown unless the optional explicit-parent check below succeeds**. Absence of an edition relation is not proof that a record represents the original work. The initial base acquisition asserted zero such dates; the bounded follow-up resolves three relative parent-work dates without identifying an ultimate original.
- `metadata.entity_release_date` means earliest **observed source-entity publication date**, not an independently established original release and not the port date for a particular console. P577 qualifiers are preserved so platform-specific date handling can be added later.
- Year/month/day precision is preserved as ISO partial dates; missing day/month values are never invented.
- A unique exact P1733 relation plus equal normalized source titles can reuse the separately licensed Steam description/tags, with explicit provenance. This text reuse does not prove edition equivalence; genres/platforms/dates are not inherited. The title equality is only a text-reuse safety gate, never an identity matching rule.
- Renamed/reused app IDs, such as historical Counter-Strike:Global Offensive versus present Counter-Strike2, fail that text-reuse gate. The Wikidata entity keeps its own text rather than receiving obsolete release content.

`metadata.wikidata` records scope, related Steam game IDs, raw compact statement facts and confidence **as an evidence description, not a calibrated probability**. `metadata.tags` and `tags_provenance` distinguish copied Steam tags from Wikidata P136 genres. Existing enjoyment ratings/CF factors are never transferred across these relations. Applications may conservatively exclude related Q entries when a linked Steam game was already played, while allowing users to inspect edition differences.

Platform associations are community-maintained assertions, not independently verified current storefront availability. They can describe historical releases, ports or services; qualifiers must be considered before claiming current purchasability or release status. No automatic PS4→PS5 backward-compatibility inference is performed.

## Actual acquisition results

Queried **1,500 Steam IDs** in **60 successful data batches**. Final local catalog additions: **1,475 Wikidata source entities**, all with labeled platform facts; **1,469** have an observed entity publication date. **Three** relative parent-work dates were subsequently resolved by the explicit-relation follow-up below; no ultimate-original claim is made. **851** entries safely reuse uniquely mapped Steam descriptions/tags with explicit provenance.

| Console platform, as labeled by Wikidata | Source entities |
|---|---:|
| PlayStation 5 | 204 |
| Nintendo Switch | 459 |
| PlayStation 4 | 669 |
| Xbox One | 552 |
| Xbox Series X and Series S | 228 |
| Nintendo Switch 2 | 34 |
| PlayStation 3 | 192 |
| Xbox 360 | 194 |
| PlayStation Vita | 54 |

Mapping outcomes: **1,454 unique external relations**, **29 Steam IDs associated with multiple entities**, and **37 entities retained for type review rather than automatically listed**. Seventeen requested Steam IDs had no relation in this query snapshot. These are source coverage results, not recall/accuracy estimates.

Examples actually acquired:

| Source entity | Own platform evidence (selected) | Earliest observed entity publication |
|---|---|---|
| [Final Fantasy X/X-2 HD Remaster Q17013575](https://www.wikidata.org/wiki/Q17013575) | PlayStation3/4/Vita, NintendoSwitch, Windows (among source facts) | 2013-12-26; not the original Final FantasyX date |
| [No Man’s Sky Q15297043](https://www.wikidata.org/wiki/Q15297043) | PlayStation4/5, NintendoSwitch, XboxOne/Series, Windows/macOS | 2016-08-09; not every port’s date |
| [The Witcher3 Q4267401](https://www.wikidata.org/wiki/Q4267401) | PlayStation4/5, NintendoSwitch, XboxOne/Series, Windows | 2015-05-19; not every port’s date |

The population SQLite source hash before/after enrichment remains `e873ed8dc766549b66d69496a6580d38242fa84786f6ce029609241b94e0a76f`. All1,475 final compact entries were compared to the stored local fields and matched. Profile isolation is covered by an automated temporary-database test; fixtures never enter the real local catalog.

Compact public catalog size: **9,032,598 bytes**, SHA-256 `d034ffd4bceae9514359eb3253b1bb19a441958dd2aa3b60d308defd3fe1b300`. Initial60 data batches took **98.741 seconds**; cached reconstruction with default-label completion and final application took **13.810 seconds**. Aggregate machine-readable evidence: `data/wikidata/report.json`. Full data/label queries and raw responses are cached separately and need not be included in the ready bundle.

## External access observations and limitations

The official MediaWiki Action API was first attempted with `maxlag=5`; two attempts returned a `maxlag` error around10 seconds, so it was stopped rather than bypassing its requested backoff. The documented SPARQL data access route succeeded and supplied the actual statements. No private APIs, proxy changes or scraping were used.

Nine offline tests cover exact-ID validation, malformed inputs, bounded retries, denial stopping, default-label fallback, date precision, deprecated/ambiguous statements, the reused-Steam-ID text gate, separate entity scope and unchanged local feedback.

Catalog enrichment improves metadata availability; it does not establish recommendation accuracy for console titles. Those new Q entries have no trained collaborative factors and must use clearly labeled metadata fallback until evaluated. Original-work chronology, exact edition equivalence, platform-specific availability dates and narrower class ancestry remain explicit limitations. Future updates should retain these distinctions and re-check source facts.

## Optional explicit parent-work date follow-up

```bash
python scripts/enrich_wikidata.py --data-dir data --parent-work-dates --apply
```

Exactly **five** acquired entries had a positive P629 relation to **three** parents. One official bounded query acquired17 parent statements. The resolver requires: a single explicit child P629 parent; a positive reciprocal parent P747 child relation; parent P31 explicitly identifying an individual video game; no parent edition/series/compilation evidence; valid publication precision; and no contradictory dates in the same qualifier context. Different platform/region-qualified port dates remain distinct. Unsupported date qualifiers keep the result unknown.

| Entry | Explicit parent | Relative parent-work publication | Outcome |
|---|---|---|---|
| The Witcher: Enhanced Edition Q110752642 | The Witcher Q287637 | 2007-10-26 | Resolved |
| Age of Empires II: HD Edition Q16267780 | Age of Empires II: The Age of Kings Q34852 | 1999-09-27, Windows-qualified source statement | Resolved |
| Age of Empires II: Definitive Edition Q63463133 | Age of Empires II: The Age of Kings Q34852 | 1999-09-27, Windows-qualified source statement | Resolved |
| Cave Story+ Q101076778 | Cave Story Q108370681 | Unknown | Parent is explicitly a video-game series, not an individual work |
| Cave Story Q1051853 | Cave Story Q108370681 | Unknown | Same series-parent ambiguity |

Resolved dates are stored as structured `metadata.wikidata.original_work_release_date`, containing ISO value, precision, parent ID, parent type facts, observed parent edition relations, child P629 statement, reciprocal P747 statement, parent P577 publication statement and qualifiers, query hash, retrieval time and semantics. They mean **relative parent-work earliest known publication**, not proof of an ultimate-original lineage or any console’s port date. The existing entity/edition date stays separate. No P144/adaptation/title-based relationship is used.

The population snapshot and model/test configurations remain unchanged. Only five Q-entry metadata records in the local catalog were updated; source Steam records and user preferences were untouched. Current catalog/report hashes above include this optional follow-up. Machine-readable decisions and statement provenance: [wikidata-parent-work-results.json](wikidata-parent-work-results.json). Re-running the base acquisition intentionally resets unresolved original-work fields; run this explicit follow-up again to restore independently checked parent facts from its cached official query.
