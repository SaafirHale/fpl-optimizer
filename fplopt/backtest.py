"""Walk-forward backtest of the projections against actual FPL points.

For each gameweek t of a finished season, the model sees only that season's rows from
gameweeks before t plus the previous season, projects every player registered in t, and is
scored against the points they actually got. The fixture list and prices at t are known
before the deadline, so using them is not lookahead.

Baselines, computed with the same no-lookahead rule:
    form     mean points over the player's previous four gameweeks (GW1: last season per match)
    ppg      season points per match so far (GW1: last season per match)
    fpl_lag  FPL's own expected points as recorded in the previous gameweek's row

The dataset's same-gameweek xP column is not usable as a baseline: it was recorded after the
gameweek and leaks its result (see `leakage_check`). Its correlation with points falls from
0.685 this week to 0.453 next week on 2024/25, while honest forecasts lose about 0.03-0.06.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Season, load_history
from .model import PARTS, Params, project

ROOT = Path(__file__).resolve().parent.parent
PREV_OF = {"2024-25": "2023-24", "2025-26": "2024-25", "2026-27": "2025-26"}


def team_map_between(prev: Season, cur: Season) -> dict:
    """Previous-season team id -> current-season team id, matched on short name."""
    by_short = {v: k for k, v in cur.teams.items()}
    return {pid: by_short[s] for pid, s in prev.teams.items() if s in by_short}


def players_at(season: Season, gw: int) -> pd.DataFrame:
    r = season.rows[season.rows.gw == gw]
    g = r.groupby("element").agg(code=("code", "first"), team=("team", "first"), price=("price", "first"))
    g["pos"] = season.players.pos.reindex(g.index)
    g["status"] = "a"
    g["chance"] = np.nan
    return g


def project_gw(season: Season, prev: Season | None, gw: int, params: Params = Params(), horizon: int = 1,
               players: pd.DataFrame | None = None) -> pd.DataFrame:
    gws = list(range(gw, gw + horizon))
    sched = season.fixtures[season.fixtures.gw.isin(gws)][["gw", "home_team", "away_team"]]
    P = players if players is not None else players_at(season, gw)
    return project(
        P, sched, season.rows, prev.rows if prev else None, prev.players.pos if prev else None,
        team_map_between(prev, season) if prev else {}, season.scoring, params, first_gw=gw,
    )


def baselines(season: Season, prev: Season | None) -> pd.DataFrame:
    """Per (element, gw): actual points and the no-lookahead baselines."""
    r = season.rows
    g = r.groupby(["element", "gw"]).agg(points=("points", "sum"), minutes=("minutes", "sum"),
                                         fpl_xp=("fpl_xp", "first"), code=("code", "first"),
                                         n=("fixture", "size")).reset_index()
    g = g.sort_values(["element", "gw"])
    prior = pd.Series(dtype=float)
    if prev is not None:
        pr = prev.rows.groupby("code").agg(pts=("points", "sum"), n=("fixture", "size"))
        prior = pr.pts / pr.n
    g["prior"] = g.code.map(prior).fillna(0.0)
    # gameweeks where the dataset holds no xP at all (all zero) are missing, not zero
    empty = g.groupby("gw").fpl_xp.transform(lambda s: (s.fillna(0) == 0).all())
    g.loc[empty, "fpl_xp"] = np.nan
    g["fpl_lag"] = g.groupby("element").fpl_xp.shift(1)
    # rows are per gameweek; a gameweek may hold two matches, so baselines are per match x matches
    g["ppm"] = g.points / g.n
    grp = g.groupby("element")
    form = grp.ppm.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    ppg = grp.ppm.transform(lambda s: s.shift(1).expanding().mean())
    g["form"] = form.fillna(g.prior) * g.n
    g["ppg"] = ppg.fillna(g.prior) * g.n
    pos = season.players.pos
    g["pos"] = g.element.map(pos)
    return g


def run(season_name: str = "2025-26", params: Params = Params(), gws: range | None = None,
        verbose: bool = False) -> pd.DataFrame:
    season = load_history(season_name)
    prev_name = PREV_OF.get(season_name)
    prev = load_history(prev_name) if prev_name and (ROOT / "data" / "raw" / prev_name).exists() else None
    base = baselines(season, prev)
    out = []
    gws = gws or range(1, int(season.rows.gw.max()) + 1)
    for gw in gws:
        proj = project_gw(season, prev, gw, params)
        proj = proj[proj.gw == gw]
        out.append(proj[["element", "gw", "xp", "xmins", "p60"] + PARTS])
        if verbose:
            print(f"  GW{gw}: {len(proj)} players")
    pred = pd.concat(out)
    return base.merge(pred, on=["element", "gw"], how="inner")


def _spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


MODELS = {"model": "xp", "form": "form", "ppg": "ppg"}


def metrics(df: pd.DataFrame, models: dict = MODELS) -> dict:
    """Error and ranking metrics for the model and each baseline, all on the same rows."""
    res = {"rows": int(len(df)), "gameweeks": int(df.gw.nunique()), "players": int(df.element.nunique())}
    for name, col in models.items():
        if df[col].isna().all():
            continue
        e = df[col] - df.points
        per_gw = df.groupby("gw").apply(lambda d: _spearman(d[col], d.points), include_groups=False)
        top = []
        cap = []
        for _, d in df.groupby("gw"):
            k = 50
            pred_top = set(d.nlargest(k, col).element)
            act_top = set(d.nlargest(k, "points").element)
            top.append(len(pred_top & act_top) / k)
            cap.append(float(d.loc[d[col].idxmax(), "points"]))
        played = df.minutes > 0
        res[name] = {
            "mae": float(e.abs().mean()),
            "rmse": float(np.sqrt((e**2).mean())),
            "bias": float(e.mean()),
            "pearson": float(df[col].corr(df.points)),
            "spearman_per_gw": float(per_gw.mean()),
            "top50_overlap": float(np.mean(top)),
            "captain_pts": float(np.mean(cap)),
            "mae_played": float(e[played].abs().mean()),
            "mae_by_pos": {p: float(e[df.pos == p].abs().mean()) for p in ["GKP", "DEF", "MID", "FWD"]},
        }
    return res


def head_to_head_fpl(df: pd.DataFrame) -> dict:
    """Model vs FPL's previous-gameweek xP on the rows where that snapshot exists."""
    sub = df.dropna(subset=["fpl_lag"])
    if sub.empty:
        return {}
    return metrics(sub, {"model": "xp", "fpl_lag": "fpl_lag", "form": "form"})


def leakage_check(df: pd.DataFrame) -> dict:
    """Correlation of each forecast with last week's, this week's and next week's points.

    An honest forecast for week t loses only a little accuracy on week t+1. A column recorded
    after week t correlates with week t far better than with week t+1.
    """
    d = df.sort_values(["element", "gw"]).copy()
    g = d.groupby("element").points
    d["prev"], d["next"] = g.shift(1), g.shift(-1)
    d = d.dropna(subset=["prev", "next"])
    out = {}
    for c in ["xp", "fpl_xp", "form", "ppg"]:
        s = d.dropna(subset=[c])
        if len(s):
            out[c] = {k: float(s[c].corr(s[w])) for k, w in [("prev", "prev"), ("this", "points"), ("next", "next")]}
    return out


def calibration(df: pd.DataFrame, col: str = "xp") -> pd.DataFrame:
    bins = [-np.inf, 0.5, 1, 2, 3, 4, 5, 6, 8, np.inf]
    labels = ["<0.5", "0.5-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-8", "8+"]
    b = pd.cut(df[col], bins=bins, labels=labels)
    t = df.groupby(b, observed=False).agg(n=("points", "size"), projected=(col, "mean"), actual=("points", "mean"))
    return t


def save(df: pd.DataFrame, season: str, params: Params, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = metrics(df)
    m["vs_fpl_lag"] = head_to_head_fpl(df)
    m["leakage_check"] = leakage_check(df)
    m["season"] = season
    m["params"] = asdict(params)
    m["calibration"] = calibration(df).reset_index().rename(columns={"xp": "bucket"}).to_dict("records")
    (out_dir / f"backtest-{season}.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    return m
