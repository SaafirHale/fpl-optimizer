"""Load seasons into one shape: a per player-match table plus players and fixtures.

History comes from the public per-gameweek CSVs in the vaastav/Fantasy-Premier-League
repository (`fplopt fetch-history` downloads them to data/raw/<season>/). The current season
comes from the live FPL API: bootstrap-static, fixtures and one event/<gw>/live call per
finished gameweek, so a full refresh is about ten requests.

rows columns (one row per player per match):
    gw fixture element code team opp home minutes goals assists xg xa cs gc saves bonus
    yc rc og pm ps dc points price fpl_xp
`dc` is the raw defensive-contribution count (CBIT for defenders, CBIRT otherwise), not points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import api
from .rules import POSITIONS, Rules, Scoring

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
HISTORY_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/{file}"
HISTORY_FILES = {"gws/merged_gw.csv": "merged_gw.csv", "players_raw.csv": "players_raw.csv", "teams.csv": "teams.csv"}

ROW_COLS = [
    "gw", "fixture", "element", "code", "team", "opp", "home", "minutes", "goals", "assists", "xg", "xa",
    "cs", "gc", "saves", "bonus", "yc", "rc", "og", "pm", "ps", "dc", "points", "price", "fpl_xp",
]

# Defensive contributions were introduced in 2025/26; earlier seasons scored none.
NO_DC = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
SEASON_SCORING = {
    "2023-24": Scoring(defensive_contribution=NO_DC, goals={"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}),
    "2024-25": Scoring(defensive_contribution=NO_DC, goals={"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}),
    "2025-26": Scoring(),
}


@dataclass
class Season:
    name: str
    rows: pd.DataFrame
    players: pd.DataFrame  # index element: code name pos team price status chance
    fixtures: pd.DataFrame  # gw fixture home_team away_team finished
    teams: dict  # id -> short name
    scoring: Scoring = field(default_factory=Scoring)
    rules: Rules = field(default_factory=Rules)
    current_gw: int | None = None  # live seasons: last gameweek with any data
    next_gw: int | None = None


def fetch_history(season: str) -> None:
    """Download one season of per-gameweek CSVs (about 5 MB) into data/raw/<season>/."""
    out = RAW / season
    out.mkdir(parents=True, exist_ok=True)
    for remote, local in HISTORY_FILES.items():
        r = api.requests.get(HISTORY_URL.format(season=season, file=remote), headers=api.HEADERS, timeout=60)
        r.raise_for_status()
        (out / local).write_bytes(r.content)


def load_history(season: str) -> Season:
    d = RAW / season
    if not (d / "merged_gw.csv").exists():
        raise FileNotFoundError(f"{d}/merged_gw.csv missing: run `fplopt fetch-history {season}`")
    m = pd.read_csv(d / "merged_gw.csv", encoding="utf-8")
    teams_df = pd.read_csv(d / "teams.csv")
    pr = pd.read_csv(d / "players_raw.csv", encoding="utf-8")
    m = m[m.position != "AM"].copy()  # 2024/25 assistant managers
    name_to_id = dict(zip(teams_df.name, teams_df.id))
    code = dict(zip(pr.id, pr.code))
    pos_of = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}

    def col(c, default=0.0):
        return m[c] if c in m.columns else pd.Series(default, index=m.index)

    rows = pd.DataFrame(
        {
            "gw": m.GW.astype(int),
            "fixture": m.fixture.astype(int),
            "element": m.element.astype(int),
            "code": m.element.map(code).fillna(-m.element).astype(int),
            "team": m.team.map(name_to_id).astype(int),
            "opp": m.opponent_team.astype(int),
            "home": m.was_home.astype(str).str.lower().eq("true"),
            "minutes": m.minutes.astype(float),
            "goals": m.goals_scored.astype(float),
            "assists": m.assists.astype(float),
            "xg": m.expected_goals.astype(float),
            "xa": m.expected_assists.astype(float),
            "cs": m.clean_sheets.astype(float),
            "gc": m.goals_conceded.astype(float),
            "saves": m.saves.astype(float),
            "bonus": m.bonus.astype(float),
            "yc": m.yellow_cards.astype(float),
            "rc": m.red_cards.astype(float),
            "og": m.own_goals.astype(float),
            "pm": m.penalties_missed.astype(float),
            "ps": m.penalties_saved.astype(float),
            "dc": col("defensive_contribution", np.nan).astype(float),
            "points": m.total_points.astype(float),
            "price": m.value.astype(int),
            "fpl_xp": m.xP.astype(float),
        }
    )
    # one position per element (positions are fixed within a season)
    pos = m.groupby("element").position.first().map(pos_of)
    last = rows.sort_values("gw").groupby("element").last()
    names = pr.set_index("id")
    players = pd.DataFrame(
        {
            "code": last.code,
            "name": names.web_name.reindex(last.index).fillna(m.groupby("element").name.first()),
            "pos": pos.reindex(last.index),
            "team": last.team,
            "price": last.price,
            "status": "a",
            "chance": np.nan,
        }
    )
    fx = (
        rows[rows.home][["gw", "fixture", "team", "opp"]]
        .drop_duplicates("fixture")
        .rename(columns={"team": "home_team", "opp": "away_team"})
        .assign(finished=True)
        .sort_values(["gw", "fixture"])
        .reset_index(drop=True)
    )
    return Season(
        name=season,
        rows=rows.sort_values(["gw", "fixture", "element"]).reset_index(drop=True),
        players=players,
        fixtures=fx,
        teams=dict(zip(teams_df.id, teams_df.short_name)),
        scoring=SEASON_SCORING.get(season, Scoring()),
        rules=Rules(scoring=SEASON_SCORING.get(season, Scoring())),
    )


LIVE_STATS = {
    "minutes": "minutes", "goals": "goals_scored", "assists": "assists", "xg": "expected_goals",
    "xa": "expected_assists", "cs": "clean_sheets", "gc": "goals_conceded", "saves": "saves",
    "bonus": "bonus", "yc": "yellow_cards", "rc": "red_cards", "og": "own_goals", "pm": "penalties_missed",
    "ps": "penalties_saved", "dc": "defensive_contribution", "points": "total_points",
}


def load_live(refresh: bool = False, season_name: str = "2026-27") -> Season:
    """Build the current season from the live FPL API (cached; ~10 requests on a cold cache)."""
    bs = api.bootstrap(refresh)
    fxs = api.fixtures(refresh)
    rules = Rules.from_bootstrap(bs)
    teams = {t["id"]: t["short_name"] for t in bs["teams"]}
    events = {e["id"]: e for e in bs["events"]}
    current = max((e["id"] for e in bs["events"] if e["is_current"] or e["finished"]), default=0)
    nxt = next((e["id"] for e in bs["events"] if e["is_next"]), None)
    els = bs["elements"]
    players = pd.DataFrame(
        {
            "code": [e["code"] for e in els],
            "name": [e["web_name"] for e in els],
            "pos": [POSITIONS[e["element_type"] - 1] if e["element_type"] <= 4 else None for e in els],
            "team": [e["team"] for e in els],
            "price": [e["now_cost"] for e in els],
            "status": [e["status"] for e in els],
            "chance": [e["chance_of_playing_next_round"] for e in els],
            "news": [e["news"] for e in els],
            "selected": [float(e["selected_by_percent"]) for e in els],
            "ep_next": [float(e["ep_next"] or 0) for e in els],
        },
        index=pd.Index([e["id"] for e in els], name="element"),
    )
    players = players[players.pos.notna()]
    players["chance"] = players.chance.astype(float)
    fixtures = pd.DataFrame(
        [
            {"gw": f["event"], "fixture": f["id"], "home_team": f["team_h"], "away_team": f["team_a"],
             "finished": bool(f["finished"]), "started": bool(f["started"])}
            for f in fxs
            if f["event"] is not None
        ]
    )
    # per player-gameweek rows from event/live; split double gameweeks evenly across matches
    rows = []
    for gw in range(1, current + 1):
        if not any(f["started"] for f in fxs if f["event"] == gw):
            continue
        live = api.event_live(gw, finished=bool(events[gw]["data_checked"]), refresh=refresh)
        gw_fx = fixtures[(fixtures.gw == gw) & fixtures.started]
        by_team: dict[int, list] = {}
        for f in gw_fx.itertuples():
            by_team.setdefault(f.home_team, []).append((f.fixture, f.away_team, True))
            by_team.setdefault(f.away_team, []).append((f.fixture, f.home_team, False))
        for el in live["elements"]:
            pid = el["id"]
            if pid not in players.index:
                continue
            team = players.at[pid, "team"]
            matches = by_team.get(team, [])
            if not matches:
                continue
            st = el["stats"]
            n = len(matches)
            for fixture, opp, home in matches:
                row = {k: float(st.get(v, 0) or 0) / n for k, v in LIVE_STATS.items()}
                row.update(gw=gw, fixture=fixture, element=pid, code=players.at[pid, "code"], team=team,
                           opp=opp, home=home, price=players.at[pid, "price"], fpl_xp=np.nan)
                rows.append(row)
    rows_df = pd.DataFrame(rows, columns=ROW_COLS)
    return Season(
        name=season_name,
        rows=rows_df.sort_values(["gw", "fixture", "element"]).reset_index(drop=True),
        players=players,
        fixtures=fixtures.sort_values(["gw", "fixture"]).reset_index(drop=True),
        teams=teams,
        scoring=rules.scoring,
        rules=rules,
        current_gw=current,
        next_gw=nxt,
    )


def match_points(r: pd.DataFrame, pos: pd.Series, sc: Scoring) -> pd.Series:
    """Recompute FPL points from the components of each row (used to check the scoring tables)."""
    p = pos.values
    m = r.minutes.values
    pts = np.where(m >= 60, sc.long_play, np.where(m > 0, sc.short_play, 0)).astype(float)
    pts += r.goals.values * np.array([sc.goals[x] for x in p])
    pts += r.assists.values * sc.assist
    pts += r.cs.values * np.array([sc.clean_sheet[x] for x in p])
    pts += np.floor(r.gc.values / 2) * np.array([sc.goals_conceded[x] for x in p])
    pts += np.floor(r.saves.values / 3) * sc.saves
    pts += r.bonus.values + r.yc.values * sc.yellow + r.rc.values * sc.red
    pts += r.og.values * sc.own_goal + r.pm.values * sc.pen_missed + r.ps.values * sc.pen_saved
    thr = np.array([sc.dc_threshold[x] for x in p])
    dcp = np.array([sc.defensive_contribution[x] for x in p])
    pts += np.where(np.nan_to_num(r.dc.values) >= thr, dcp, 0)
    return pd.Series(pts, index=r.index)
