"""Authorized provider interfaces. Credentials live in the process environment only."""
from __future__ import annotations

import os
import re
import time
from typing import Any, Protocol
import requests


class ProviderError(RuntimeError):
    """Actionable provider failure without credentials or sensitive URLs."""


class GameProvider(Protocol):
    def library(self, account_id: str) -> list[dict[str, Any]]: ...


class SteamProvider:
    """Steam Web API, public/authorized visibility only; no scraping or passwords."""

    BASE = "https://api.steampowered.com"

    def __init__(self, api_key: str | None = None, session: Any = None):
        self._key = api_key or os.environ.get("STEAM_API_KEY", "")
        self._session = session or requests.Session()
        if not self._key:
            raise ProviderError("Set STEAM_API_KEY in the environment before importing Steam.")

    @staticmethod
    def _account(account_id: str) -> str:
        if not re.fullmatch(r"[0-9]{17}", account_id):
            raise ValueError("Enter a 17-digit SteamID64, not a profile URL or password.")
        return account_id

    def _get(self, endpoint: str, **params: Any) -> dict[str, Any]:
        for attempt in range(3):
            try:
                response = self._session.get(
                    self.BASE + endpoint,
                    params={"key": self._key, **params}, timeout=25,
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt < 2:
                        time.sleep(0.5 * 2**attempt)
                        continue
                if response.status_code != 200:
                    raise ProviderError(f"Steam returned HTTP {response.status_code}. Check API authorization and visibility.")
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ProviderError("Steam returned an invalid response.")
                return payload
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise ProviderError("Steam request failed. Check connectivity, API key and visibility.") from None
                time.sleep(0.5 * 2**attempt)
        raise ProviderError("Steam temporarily unavailable.")

    def library(self, account_id: str) -> list[dict[str, Any]]:
        response = self._get("/IPlayerService/GetOwnedGames/v0001/", steamid=self._account(account_id),
                             include_appinfo=1, include_played_free_games=1).get("response", {})
        if "game_count" not in response:
            raise ProviderError("Steam did not expose a library. Set game details visibility appropriately; private histories cannot be imported.")
        games = response.get("games", [])
        return [{"game_id": f"steam:{int(g['appid'])}", "title": g.get("name", f"Steam app {g['appid']}"),
                 "playtime_minutes": max(0, int(g.get("playtime_forever", 0))),
                 "platforms": ["PC"], "source": "steam"} for g in games]

    def recent(self, account_id: str) -> list[dict[str, Any]]:
        return self._get("/IPlayerService/GetRecentlyPlayedGames/v0001/", steamid=self._account(account_id)).get("response", {}).get("games", [])

    def achievements(self, account_id: str, app_id: int) -> list[dict[str, Any]]:
        if app_id <= 0:
            raise ValueError("app_id must be positive")
        stats = self._get("/ISteamUserStats/GetPlayerAchievements/v0001/", steamid=self._account(account_id), appid=app_id).get("playerstats", {})
        if not stats.get("success"):
            raise ProviderError("Achievements unavailable for this game/account.")
        return stats.get("achievements", [])
