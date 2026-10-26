"""Polite client for the public FPL API with an on-disk cache.

Every response is cached under data/cache/ with a time-to-live, and live requests are spaced
at least `MIN_GAP` seconds apart. The machine this was built on re-signs HTTPS through a
corporate proxy, so the OS trust store is used via `truststore` when it is installed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

try:  # use the OS certificate store (needed behind TLS-intercepting proxies)
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover
    pass

BASE = "https://fantasy.premierleague.com/api/"
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
MIN_GAP = 1.5  # seconds between live requests
HEADERS = {"User-Agent": "fpl-optimizer (personal research project)"}

_last_call = 0.0


def get(path: str, ttl: float = 6 * 3600, refresh: bool = False) -> dict | list:
    """GET `BASE + path`, served from cache when younger than `ttl` seconds."""
    global _last_call
    CACHE.mkdir(parents=True, exist_ok=True)
    file = CACHE / (path.strip("/").replace("/", "_") + ".json")
    if file.exists() and not refresh and time.time() - file.stat().st_mtime < ttl:
        return json.loads(file.read_text(encoding="utf-8"))
    wait = MIN_GAP - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    try:
        r = requests.get(BASE + path, headers=HEADERS, timeout=30)
        _last_call = time.time()
        r.raise_for_status()
    except requests.RequestException:
        if file.exists():  # stale cache beats no data
            return json.loads(file.read_text(encoding="utf-8"))
        raise
    data = r.json()
    file.write_text(json.dumps(data), encoding="utf-8")
    return data


def bootstrap(refresh: bool = False) -> dict:
    return get("bootstrap-static/", ttl=3600, refresh=refresh)


def fixtures(refresh: bool = False) -> list:
    return get("fixtures/", ttl=3600, refresh=refresh)


def event_live(gw: int, finished: bool, refresh: bool = False) -> dict:
    # a finished, data-checked gameweek never changes, so cache it for good
    return get(f"event/{gw}/live/", ttl=10 * 365 * 86400 if finished else 1800, refresh=refresh)


def entry(team_id: int) -> dict:
    return get(f"entry/{team_id}/", ttl=1800)


def entry_history(team_id: int) -> dict:
    return get(f"entry/{team_id}/history/", ttl=1800)


def entry_transfers(team_id: int) -> list:
    return get(f"entry/{team_id}/transfers/", ttl=1800)


def entry_picks(team_id: int, gw: int) -> dict:
    return get(f"entry/{team_id}/event/{gw}/picks/", ttl=1800)
