# Optional IGDB enrichment

The default catalog remains usable offline without IGDB. An optional internal API in `gamecompass.catalog` can add verified cross-platform metadata after authorized access. This integration has mocked tests only: credentials were unavailable, so live requests and coverage remain unverified.

Set `IGDB_CLIENT_ID` and `IGDB_ACCESS_TOKEN` in your process environment. The token is an app access token obtained through the documented Twitch process. Neither token nor client secret is saved in the database. Refresh expired tokens outside this integration. Never commit credentials.

```python
from gamecompass.catalog import IGDBClient, enrich_steam_game
from gamecompass.store import Store

store = Store("data")
# The canonical Steam game must already exist in this store.
decision = enrich_steam_game(
    store, "steam:10", IGDBClient(),
    acknowledge_provider_terms=True,
)
print(decision.status, decision.reason)
```

Setting the acknowledgement flag is the user's explicit instruction to cache this provider's response locally after reviewing the current terms. It does not establish unrestricted retention, redistribution, artwork rights or model-training permission. Cached metadata carries a `redistribution_authorized=false` marker. Do not distribute these records or profile exports containing them without appropriate rights. The implementation retains cover URLs, not image files. No bulk caching or automatic startup enrichment occurs.

Identity resolution accepts only a unique exact Steam app-ID mapping in IGDB external-game records. The current source table's Steam ID is discovered by name, rather than assuming the deprecated category enum's value applies to the new table. Legacy expanded records using the documented category=1 can be resolved by the pure resolver. Identical or similar titles never establish identity. Multiple external-ID matches, explicit game-type conflicts, and unresolved candidates remain marked for review. Remakes, remasters, parent games, DLC, expansions, collections and franchises remain distinct records or relationships. The original canonical ID and title are retained.

`resolve_candidates(source_game, candidate_games)` is a pure reusable decision API. It returns `matched`, `review`, or `unmatched`, selected IGDB ID if any, deterministic evidence confidence and retained candidates. Confidence is not a probability of a correct match or enjoyment. Candidate records must contain expanded `external_games` with source names, or legacy category values; bare numeric source references are insufficient evidence without source lookup.

Matched records retain alternative names, release date, modes, franchises, developers/publishers, relationships, external IDs and cover URL inside `metadata.igdb`. Named platforms, genres and themes enrich the searchable catalog. This updates local metadata only; retrain or rebuild content representations to use newly enriched fields in an existing model. Run batches in a background job; this synchronous API uses request pacing and bounded retries, and never creates a public server.

Sources: [IGDB API documentation](https://api-docs.igdb.com/) and [Twitch Developer Services Agreement](https://legal.twitch.com/legal/developer-agreement). The external-games `category` field is deprecated in favor of `external_game_source`; provider schemas may change, so verify a live authorized request before relying on a release.
