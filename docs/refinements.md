# Recommendation refinements and game editions

Discover offers three ordinary refinements: **platform**, **genre**, and **era**. They restrict recommendation candidates, not your preferences. You can search and rate any game in the local catalog, including a different genre, platform or era. Rating a game does not change the active filters.

The same controls apply to **If you liked this game**, available on game cards. This mode uses one selected game as its starting point without saving a positive rating for it. Your profile's played/unwanted exclusions still apply. Return to browsing or select another main page to leave this mode. The engine supplies the actual ranking and explanation; a similar-game ranking is not an enjoyment probability.

## Platforms and genres

Platform choices come only from the cached catalog or explicitly entered verified platforms. Common spellings such as PS5 and PlayStation 5 are normalized, but support for a different console generation is never inferred. A Windows release does not establish PlayStation, Xbox or Nintendo availability. A game with missing platform metadata cannot satisfy a platform filter.

The current Steam dataset primarily verifies desktop operating systems. If there are no verified console platforms, the UI says so and does not fabricate console choices. Console information can be added through authorized, exactly mapped IGDB enrichment or a reviewed manual catalog entry. No new credentials or provider accounts are required for manual entry.

Genre filters use explicit catalog genre fields first. A limited list of recognizable genre labels may also come from catalog tags; these are labeled **inferred from catalog tags** on cards. Tags such as Story Rich, Singleplayer, Anime and Controller are not converted into genres. This fallback reflects the source tagging, not a verified taxonomy or psychological inference.

## Original works versus editions

Era defaults to the **original work's** release date. Under **Edition details**, you can switch to **this edition's release date**. For example, an explicitly documented 1997 original with a 2013 remaster matches the 1990s under the original basis and the 2010s under the edition basis.

Steam/Kaggle `release_date` and `date_release` values describe the catalog listing or edition. They never establish the original work's date. A new PC release of an older console game is not automatically treated as a newly created game.

Original dates are accepted from explicit user-provided metadata or an exact IGDB mapping whose record is identified as a main game without a parent/version relationship. A remake or remaster's IGDB first release date describes that edition, not the original. Lists of remakes/remasters on an original record do not make the original itself a remake. Relationships and original dates are never guessed from titles.

Games with unknown dates are included when no era is selected, and excluded when the requested era requires the missing date. The UI explains this. Original-date coverage can be sparse; changing to the edition date basis is a deliberate choice, not an automatic fallback.

The optional release-type control offers any release, originals only, or remakes/remasters only. Unknown release types fail a specific release-type filter. A plain re-release is recorded separately and is not silently equated with a remake or remaster. Words such as HD, Definitive and Remastered in a title are not enough to establish a type.

## Manual release evidence

Settings → Import & export → Add a missing game or new release accepts an optional original date, edition date and known release type. Dates may be a year or `YYYY-MM-DD`; leave uncertain facts blank and use `unknown`. These fields are labeled user-provided, not independently verified by the software. An exact existing canonical ID requires explicit update selection; titles never trigger automatic merging.

The metadata representation is:

```json
{
  "manual_release": {
    "source": "user-provided",
    "original_date": "1997-01-31",
    "edition_date": "2013-12-05",
    "release_kind": "remaster"
  }
}
```

`RecommendationFilters` is a validated, immutable service value with `platform`, `genre`, `year_min`, `year_max`, `era_basis` (`original` or `edition`) and `edition_policy` (`any`, `originals_only`, `remakes_remasters_only`). Its `matches(game)` method uses the same evidence rules for ordinary and seed-based recommendations. `release_facts(game)` exposes dates, years, release kind and source labels. Filters do not affect saved feedback or retrain the model.

These controls improve usability and candidate eligibility; they do not establish a new recommendation-accuracy result. Offline benchmark results remain tied to their documented original candidate scopes. Strict filters can leave few or no results, especially with incomplete console/original-date metadata.

Wikidata source-entity entries have their own platform facts and earliest known entity publication date, without asserting an original-work date. When an edition-era filter is combined with a platform, only publication statements explicitly qualified for that platform can supply its date; a global entity publication does not stand in for a later console port. Missing platform dates are excluded from that constrained era. The UI labels this release-date context separately.

Three enriched editions additionally have an explicitly sourced P629 parent-work date (The Witcher Enhanced Edition and two Age of Empires II editions). These are earliest observed dates of the directly linked parent work, not proof of an ultimate original across every historical relationship. A series parent or unresolved edition chain does not supply an original-work year.

The UI uses this release’s date for an era selection by default, so available catalog release data remains useful. Edition details offers the stricter original-work date view. The internal filter API keeps its explicit original/edition basis; a store date is never represented as an original date.
