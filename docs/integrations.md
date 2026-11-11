# Account and catalog integrations

Checked 2026-11-10. Never require an account connection for manual ratings. Do not infer dislike from unavailable, hidden or missing library information. Do not log request URLs containing credentials.

## Steam

The documented [IPlayerService](https://partner.steamgames.com/doc/webapi/IPlayerService) offers `GetOwnedGames` and `GetRecentlyPlayedGames`. Owned games require visible game details. `include_appinfo` requests names/icons, and `include_played_free_games` includes eligible played free games. Missing private data is not an empty library. Keep playtime and ownership separate from ratings. Achievements use the documented [ISteamUserStats](https://partner.steamgames.com/doc/webapi/ISteamUserStats) interface; requesting achievements for every title can be slow and needs bounded requests, retries, and progress.

Provide the user's own Steam Web API key and numeric SteamID64 at import time. Prefer environment variables or OS keyring; do not put keys in SQLite, exports, screenshots, committed examples or error logs. The ordinary public API host is `api.steampowered.com`; partner-only endpoints must not be represented as available to arbitrary accounts. A mock passing does not verify live access. Validate authorized live access separately.

Imported game identity is `steam:<appid>`. Do not match a title name automatically to a different ID. Imports should upsert activity, retain manual enjoyment ratings, and report missing titles rather than replacing local preferences.

## IGDB catalog

[Official IGDB API documentation](https://api-docs.igdb.com/) permits free noncommercial use under the Twitch developer agreement. It requires a Twitch account with 2FA, a confidential application, Client ID and Client Secret. Obtain an app token using client credentials; request `/v4/games` server-side with `Client-ID` and bearer headers. Limit requests to four per second and at most eight concurrent requests. There is no need for live IGDB access during local inference after authorized metadata is cached.

Keep this integration opt-in and credential gated. The public documentation did not establish unrestricted permanent caching, dataset redistribution, artwork redistribution, or ML-training rights. The [Twitch Developer Services Agreement](https://legal.twitch.com/legal/developer-agreement) reserves provider rights and does not offer a blanket data sublicense. Confirm applicable current conditions before bulk caching or publishing assets; do not distribute an IGDB dump under MIT.

Request explicit external game IDs and metadata fields needed for recommendation. Resolve by exact external Steam app IDs first. Preserve separate IGDB IDs for remakes, expansions, DLC, collections and editions; franchise relationships are edges, not identity equivalence. A fuzzy name suggestion must retain confidence and need review. Never mark every historical Steam title available on PS5 without a verified platform mapping.

## Other providers

| Provider | Research finding | Current safe path |
|---|---|---|
| PlayStation | [Partner portal](https://partners.playstation.net/) does not expose a verified general public account-library API in the inspected material | Manual ratings and user-supplied supported imports; no cookie extraction or private API |
| Xbox | [Microsoft authentication docs](https://learn.microsoft.com/en-us/gaming/gdk/docs/services/fundamentals/s2s-auth-calls/service-authentication/live-xbox-live-authentication) describe XSTS, signatures and title contexts; docs are not permission for arbitrary cross-title history access | Manual imports until approved integration permissions and documented scopes exist |
| Nintendo | [Developer portal](https://developer.nintendo.com/) is for developing and publishing titles; a general public library-history API was not verified | Manual ratings/imports; no undocumented Nintendo account endpoints |
| GOG | [Official GALAXY integrations Python API](https://github.com/gogcom/galaxy-integrations-python-api) builds integrations *into* GALAXY; it is not a documented standalone API granting arbitrary GOG library access | Supported local user exports/manual imports; future GALAXY plugin evaluated separately |

These findings describe what was verified, not proof that no authorized integration could ever exist. Live integrations beyond Steam need separate access and terms review. Do not claim them implemented because a provider interface exists.
