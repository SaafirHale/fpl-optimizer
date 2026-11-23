"""Import a manager's squad by FPL team ID from the public endpoints.

Public data gives the picks, bank, transfer history and chips, but not purchase prices or
free transfers, so both are reconstructed:
    purchase price  the cost on the manager's last transfer in of that player (from
                    /entry/<id>/transfers/), or for players kept since the first squad, the
                    price at the start: now_cost - cost_change_start when the team started
                    in GW1, otherwise the gameweek price from Valuo's nightly FPL snapshot
    free transfers  replayed week by week from /entry/<id>/history/: +1 per gameweek up to
                    the cap, minus transfers made, kept through Wildcard and Free Hit weeks
Both are printed so they can be checked against the FPL app, and `--ft` overrides the count.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import api
from .rules import Rules, selling_price

# Valuo (a sibling repo) commits a nightly FPL snapshot; read-only. Override with VALUO_SNAPSHOT.
VALUO_SNAPSHOT = Path(os.environ.get("VALUO_SNAPSHOT", Path(__file__).resolve().parents[2] / "valuo" / "data" / "fpl-snapshot.json"))


@dataclass
class Team:
    id: int
    name: str
    gw: int  # gameweek the picks are from
    picks: list  # element ids
    purchase: dict  # element -> purchase price (tenths)
    sell: dict  # element -> selling price now
    bank: int
    free_transfers: int
    chips_used: list  # (name, event)
    started_event: int
    notes: list = field(default_factory=list)


def _snapshot_prices() -> dict:
    """element id -> {round: price} from Valuo's snapshot (read-only), if it is on this machine."""
    if not VALUO_SNAPSHOT.exists():
        return {}
    snap = json.loads(VALUO_SNAPSHOT.read_text(encoding="utf-8"))
    return {p["id"]: {g[0]: g[11] for g in p["gw"]} for p in snap["players"]}


def replay_free_transfers(history: list, chips: list, started: int, rules: Rules = Rules()) -> int:
    """Free transfers available for the gameweek after the last one in `history`."""
    chip_at = {c["event"]: c["name"] for c in chips}
    ft = None
    for ev in sorted(history, key=lambda e: e["event"]):
        gw = ev["event"]
        if gw < started:
            continue
        if gw == started:
            ft = 1  # the first squad is free; one free transfer for the next gameweek
            continue
        if chip_at.get(gw) in ("wildcard", "freehit"):
            continue  # kept as they were
        unused = max(ft - ev["event_transfers"], 0)
        ft = min(rules.max_free_transfers, unused + 1)
    return ft if ft is not None else 1


def load_team(team_id: int, bootstrap: dict, rules: Rules = Rules()) -> Team:
    ent = api.entry(team_id)
    hist = api.entry_history(team_id)
    current = [e for e in bootstrap["events"] if e["is_current"] or e["finished"]]
    gw = max(e["id"] for e in current)
    picks = api.entry_picks(team_id, gw)
    els = {e["id"]: e for e in bootstrap["elements"]}
    ids = [p["element"] for p in picks["picks"]]
    started = ent.get("started_event") or 1
    chips = hist.get("chips", [])
    fh_events = {c["event"] for c in chips if c["name"] == "freehit"}
    notes = []

    # purchase prices
    transfers = sorted(api.entry_transfers(team_id), key=lambda t: t["time"])
    last_in = {}
    for t in transfers:
        if t["event"] in fh_events:
            continue
        last_in[t["element_in"]] = t["element_in_cost"]
    snap = _snapshot_prices() if started > 1 else {}
    purchase = {}
    for i in ids:
        if i in last_in:
            purchase[i] = last_in[i]
        elif started == 1:
            purchase[i] = els[i]["now_cost"] - els[i]["cost_change_start"]
        elif i in snap and started in snap[i]:
            purchase[i] = snap[i][started]
        else:
            purchase[i] = els[i]["now_cost"]
            notes.append(f"no purchase price found for {els[i]['web_name']}; using today's price")
    sell = {i: selling_price(purchase[i], els[i]["now_cost"], rules.sell_on_fee) for i in ids}
    ft = replay_free_transfers(hist.get("current", []), chips, started, rules)
    # if the current gameweek is still being played, picks for the next one are not public yet
    if not any(e["id"] == gw and e["finished"] for e in bootstrap["events"]):
        notes.append(f"GW{gw} is in progress: transfers already made for the next deadline are not visible")
    return Team(
        id=team_id,
        name=ent.get("name", str(team_id)),
        gw=gw,
        picks=ids,
        purchase=purchase,
        sell=sell,
        bank=picks["entry_history"]["bank"],
        free_transfers=ft,
        chips_used=[(c["name"], c["event"]) for c in chips],
        started_event=started,
        notes=notes,
    )


def chips_available(bootstrap: dict, used: list, gw: int) -> list:
    """Chip names still playable in `gw`, given (name, event) pairs already used."""
    out = []
    for ch in bootstrap.get("chips", []):
        if not (ch["start_event"] <= gw <= ch["stop_event"]):
            continue
        if any(n == ch["name"] and ch["start_event"] <= e <= ch["stop_event"] for n, e in used):
            continue
        out.append(ch["name"])
    return sorted(set(out))
