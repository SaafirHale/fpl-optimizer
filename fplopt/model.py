"""Expected FPL points per player per gameweek.

The method follows Valuo's per-90 model (valuo/src/lib/model/valuation.ts): per-90 rates are
shrunk toward a prior with SHRINK_MINUTES of prior weight, and last season's stats count as
up to half a season of evidence, fading as current minutes accrue (Valuo's `blendedValue`).
Two changes for points rather than value: the prior is price-implied within position (fit on
the previous season, so a GBP 10m forward starts from a GBP 10m forward's rates, not from a
league-wide average), and each rate feeds an FPL scoring component instead of a z-score.

Per fixture:
    minutes   P(60+), P(1-59) from recency-weighted appearances, prior = last season / price
    attack    goals and assists: shrunk xG/xA per 90 (blended with actual G/A), scaled by the
              opponent's xG conceded per match and home advantage
    defence   clean sheet and goals conceded: Poisson on own-team xG conceded x opponent attack
    keepers   saves per 90 scaled by opponent attack; 1 point per 3 saves
    other     bonus per 90, defensive-contribution hit rate per 60+ appearance, cards
Nothing here looks at a gameweek at or after the one being projected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .rules import POSITIONS, Scoring

SEASON_MINUTES = 38 * 90


@dataclass(frozen=True)
class Params:
    shrink_minutes: float = 900.0  # Valuo SHRINK_MINUTES: prior weight in each per-90 rate
    prev_fade_max: float = 0.5  # last season counts as up to half a season...
    prev_fade_min: float = 0.15  # ...and never less than this (Valuo blendedValue)
    goal_xg_weight: float = 0.85  # goal rate = w*xG + (1-w)*goals
    assist_xa_weight: float = 0.85
    minutes_decay: float = 0.6  # recency weight per match back (tuned on 2024/25)
    minutes_prior_weight: float = 1.0  # prior counts as this many recent matches (tuned on 2024/25)
    prev_minutes_k: float = 5.0  # shrink last season's minute shares toward the price fit
    team_prior_matches: float = 8.0  # team ratings: prior counts as this many matches
    dc_prior_apps: float = 6.0
    availability_recovery: float = 0.6  # flagged players: (1 - chance) shrinks by this per GW ahead


def _expected_floor_div(lam: np.ndarray, d: int, kmax: int = 25) -> np.ndarray:
    """E[floor(X/d)] for X ~ Poisson(lam), vectorised."""
    lam = np.asarray(lam, dtype=float)[:, None]
    k = np.arange(kmax + 1)[None, :]
    logp = -lam + k * np.log(np.maximum(lam, 1e-12)) - np.cumsum(np.log(np.maximum(k, 1)), axis=1)
    return (np.exp(logp) * (k // d)).sum(axis=1)


def _fit_linear(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float]:
    if len(x) < 3:
        return (float(np.mean(y)) if len(y) else 0.0, 0.0)
    w = np.ones_like(x) if w is None else w
    A = np.vstack([np.ones_like(x), x]).T * np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)
    return float(coef[0]), float(coef[1])


def _aggregate(r: pd.DataFrame, key: str, scoring: Scoring) -> pd.DataFrame:
    """Season totals per player (by element or code) with appearance splits. Needs a `pos` column."""
    if r.empty:
        return pd.DataFrame()
    r = r.copy()
    m = r.minutes
    r["a60"] = (m >= 60).astype(float)
    r["asub"] = ((m > 0) & (m < 60)).astype(float)
    r["m60"] = np.where(m >= 60, m, 0.0)
    r["msub"] = np.where((m > 0) & (m < 60), m, 0.0)
    thr = r.pos.map(scoring.dc_threshold).fillna(999)
    known = r.dc.notna() & (m >= 60)
    r["dc_known"] = known.astype(float)
    r["dc_hit"] = (known & (r.dc.fillna(0) >= thr)).astype(float)
    g = r.groupby(key)
    out = g[["minutes", "xg", "xa", "goals", "assists", "saves", "bonus", "yc", "rc", "a60", "asub", "m60",
             "msub", "dc_known", "dc_hit"]].sum()
    out["matches"] = g.size()
    out["price0"] = g.price.first()
    out["points"] = g.points.sum()
    out["pos"] = g.pos.first()
    return out


def team_ratings(cur: pd.DataFrame, prev: pd.DataFrame | None, team_map: dict, teams_now: list,
                 params: Params) -> tuple[pd.DataFrame, float, float]:
    """Attack (xG for) and defence (xG against) per match, shrunk toward last season.

    `team_map` maps previous-season team ids to current ids. Promoted teams start at the mean of
    the previous season's relegated teams. Returns (ratings, league mean, home factor).
    """

    def matches(rows):
        if rows is None or rows.empty:
            return pd.DataFrame(columns=["fixture", "team", "home", "xgf", "xga"])
        t = rows.groupby(["fixture", "team"]).agg(xgf=("xg", "sum"), home=("home", "first"), opp=("opp", "first"))
        t = t.reset_index()
        opp = t.set_index(["fixture", "team"]).xgf
        t["xga"] = [opp.get((f, o), np.nan) for f, o in zip(t.fixture, t.opp)]
        return t.dropna(subset=["xga"])

    pm = matches(prev)
    cm = matches(cur)
    both = pd.concat([pm, cm])
    league = float(both.xgf.mean()) if len(both) else 1.35
    if len(both) and both[~both.home.astype(bool)].xgf.mean() > 0:
        home_factor = float(np.sqrt(both[both.home.astype(bool)].xgf.mean() / both[~both.home.astype(bool)].xgf.mean()))
    else:
        home_factor = 1.0
    prior = {}
    if len(pm):
        pt = pm.groupby("team")[["xgf", "xga"]].mean()
        stay = {team_map[t] for t in pt.index if t in team_map}
        relegated = pt.loc[[t for t in pt.index if t not in team_map]]
        promoted_prior = relegated.mean() if len(relegated) else pt.mean()
        for t in teams_now:
            src = next((p for p, c in team_map.items() if c == t), None)
            prior[t] = pt.loc[src] if src is not None and src in pt.index else promoted_prior
            _ = stay
    k = params.team_prior_matches
    out = []
    for t in teams_now:
        mine = cm[cm.team == t]
        n = len(mine)
        pa, pd_ = (prior[t].xgf, prior[t].xga) if t in prior else (league, league)
        att = (mine.xgf.sum() + k * pa) / (n + k)
        dfn = (mine.xga.sum() + k * pd_) / (n + k)
        out.append({"team": t, "att": att, "def": dfn, "n": n})
    return pd.DataFrame(out).set_index("team"), league, home_factor


def project(
    players: pd.DataFrame,
    schedule: pd.DataFrame,
    cur: pd.DataFrame,
    prev: pd.DataFrame | None,
    prev_pos: pd.Series | None,
    team_map: dict,
    scoring: Scoring,
    params: Params = Params(),
    first_gw: int | None = None,
) -> pd.DataFrame:
    """Project every player for every fixture in `schedule`.

    players:  index element; columns code pos team price [status chance]
    schedule: columns gw home_team away_team (only gameweeks to project)
    cur:      this season's rows strictly before the first projected gameweek
    prev:     last season's rows (any length) or None; matched to players by `code`
    Returns one row per (element, gw) with xp, the components, and minutes probabilities.
    """
    P = players.copy()
    pos = P.pos
    first_gw = first_gw if first_gw is not None else int(schedule.gw.min())
    cur_all = cur[cur.gw < first_gw]  # every player's rows, for team ratings
    cur = cur_all[cur_all.element.isin(P.index)].assign(pos=lambda d: d.element.map(pos))
    has_prev_rows = prev is not None and len(prev) > 0
    if has_prev_rows:
        prev = prev.assign(pos=prev.element.map(prev_pos))

    # ---- evidence: this season (by element) + faded last season (by code)
    cols = ["minutes", "xg", "xa", "goals", "assists", "saves", "bonus", "yc", "rc", "a60", "asub", "m60", "msub",
            "dc_known", "dc_hit", "matches", "price0", "points"]
    zero = pd.DataFrame(0.0, index=P.index, columns=cols)
    ca = _aggregate(cur, "element", scoring)[cols].reindex(P.index).fillna(0.0) if len(cur) else zero
    fit_src = _aggregate(prev, "code", scoring) if has_prev_rows else None
    pa = fit_src[cols].reindex(P.code.values).set_axis(P.index).fillna(0.0) if has_prev_rows else zero
    team_matches_so_far = cur.groupby("team").fixture.nunique() if len(cur) else pd.Series(dtype=float)
    cur_possible = P.team.map(team_matches_so_far).fillna(0) * 90
    fade = np.maximum(params.prev_fade_min, params.prev_fade_max * (1 - cur_possible / SEASON_MINUTES))

    # ---- price-implied priors per position, fit on last season's player-seasons
    rates = ["xg", "xa", "goals", "assists", "bonus", "saves", "yc", "rc"]
    n = len(P)
    prior = {r: np.zeros(n) for r in rates}
    prior_p60 = np.full(n, 0.3)
    prior_psub = np.full(n, 0.15)
    pooled_dc = {p: 0.0 for p in POSITIONS}
    if fit_src is not None:
        for p in POSITIONS:
            sel = fit_src[(fit_src.pos == p)]
            reg = sel[sel.minutes >= 900]
            mask = (pos == p).values
            price = P.price.values[mask] / 10.0
            for r in rates:
                a, b = _fit_linear(reg.price0.values / 10.0, (reg[r] / reg.minutes * 90).values)
                prior[r][mask] = np.maximum(0.0, a + b * price)
            live = sel[sel.matches >= 10]
            a, b = _fit_linear(live.price0.values / 10.0, (live.a60 / live.matches).values)
            prior_p60[mask] = np.clip(a + b * price, 0.0, 0.95)
            a, b = _fit_linear(live.price0.values / 10.0, (live.asub / live.matches).values)
            prior_psub[mask] = np.clip(a + b * price, 0.0, 0.6)
    # defensive contributions: pooled hit rate per position from every row with DC data before now
    dc_src = [x for x in (prev if has_prev_rows else None, cur) if x is not None and len(x)]
    if dc_src:
        d = pd.concat(dc_src)
        d = d[(d.minutes >= 60) & d.dc.notna()]
        for p in POSITIONS:
            dp = d[d.pos == p]
            if len(dp):
                pooled_dc[p] = float((dp.dc >= scoring.dc_threshold[p]).mean())
    prior = {r: pd.Series(a, index=P.index) for r, a in prior.items()}
    prior_p60 = pd.Series(prior_p60, index=P.index)
    prior_psub = pd.Series(prior_psub, index=P.index)

    # ---- shrunk per-90 rates
    K = params.shrink_minutes
    M = ca.minutes + fade * pa.minutes
    rate = {}
    for r in rates:
        S = ca[r] + fade * pa[r]
        rate[r] = (S + prior[r] * K / 90) / ((M + K) / 90)
    g90 = params.goal_xg_weight * rate["xg"] + (1 - params.goal_xg_weight) * rate["goals"]
    a90 = params.assist_xa_weight * rate["xa"] + (1 - params.assist_xa_weight) * rate["assists"]
    dc_prior = pos.map(pooled_dc).astype(float)
    dc_n = ca.dc_known + fade * pa.dc_known
    dc_rate = (ca.dc_hit + fade * pa.dc_hit + params.dc_prior_apps * dc_prior) / (dc_n + params.dc_prior_apps)

    # ---- minutes: recency-weighted shares this season, prior = last season shrunk to the price fit
    has_prev = pa.matches > 0
    base60 = np.where(has_prev, (pa.a60 + params.prev_minutes_k * prior_p60) / (pa.matches + params.prev_minutes_k), prior_p60)
    basesub = np.where(has_prev, (pa.asub + params.prev_minutes_k * prior_psub) / (pa.matches + params.prev_minutes_k), prior_psub)
    w60 = pd.Series(0.0, index=P.index)
    wsub = pd.Series(0.0, index=P.index)
    wsum = pd.Series(0.0, index=P.index)
    if len(cur):
        c = cur[cur.element.isin(P.index)].sort_values("gw")
        # matches back from the most recent one, per player (DGW halves count as matches)
        c = c.assign(back=c.groupby("element").cumcount(ascending=False))
        wt = params.minutes_decay ** c.back
        w60 = (wt * (c.minutes >= 60)).groupby(c.element).sum().reindex(P.index).fillna(0)
        wsub = (wt * ((c.minutes > 0) & (c.minutes < 60))).groupby(c.element).sum().reindex(P.index).fillna(0)
        wsum = wt.groupby(c.element).sum().reindex(P.index).fillna(0)
    kp = params.minutes_prior_weight
    p60 = (w60 + kp * base60) / (wsum + kp)
    psub = (wsub + kp * basesub) / (wsum + kp)
    a60_all = ca.a60 + pa.a60
    m60 = (ca.m60 + pa.m60 + 5 * 87) / (a60_all + 5)
    msub = (ca.msub + pa.msub + 3 * 25) / (ca.asub + pa.asub + 3)

    # ---- team strength and schedule
    teams_now = sorted(set(schedule.home_team) | set(schedule.away_team) | set(P.team))
    tr, league, hf = team_ratings(cur_all, prev, team_map, teams_now, params)
    sched = pd.concat([
        schedule.assign(team=schedule.home_team, opp=schedule.away_team, home=True),
        schedule.assign(team=schedule.away_team, opp=schedule.home_team, home=False),
    ])[["gw", "team", "opp", "home"]]
    X = P.reset_index().rename(columns={"index": "element"}).merge(sched, on="team", how="inner")
    idx = X.element.values

    def v(s):
        return pd.Series(s, index=P.index).reindex(idx).values if not isinstance(s, pd.Series) else s.reindex(idx).values

    att_o = tr.att.reindex(X.opp).values
    def_o = tr["def"].reindex(X.opp).values
    def_t = tr["def"].reindex(X.team).values
    home = X.home.values
    hmult = np.where(home, hf, 1 / hf)
    # availability for flagged players, recovering toward 1 for later gameweeks
    chance = X.chance.values if "chance" in X else np.full(len(X), np.nan)
    status = X.status.values if "status" in X else np.full(len(X), "a")
    ahead = X.gw.values - first_gw
    c0 = np.where(np.isnan(chance.astype(float)), 1.0, chance.astype(float) / 100.0)
    avail = 1 - (1 - c0) * params.availability_recovery ** ahead
    avail = np.where(np.isin(status, ["u", "n"]), 0.0, avail)

    P60 = v(p60) * avail
    PSUB = v(psub) * avail
    M60 = v(m60)
    MSUB = v(msub)
    emin = P60 * M60 + PSUB * MSUB
    posv = X.pos.values
    sc = scoring
    goal_pts = np.array([sc.goals[p] for p in posv])
    cs_pts = np.array([sc.clean_sheet[p] for p in posv])
    gc_pts = np.array([sc.goals_conceded[p] for p in posv])
    dc_pts = np.array([sc.defensive_contribution[p] for p in posv])

    att_mult = (def_o / league) * hmult
    lam_conc = def_t * (att_o / league) / hmult  # opponent's expected goals against this team
    comp = pd.DataFrame({"element": idx, "gw": X.gw.values, "opp": X.opp.values, "home": home})
    comp["appearance"] = sc.long_play * P60 + sc.short_play * PSUB
    comp["goals"] = goal_pts * v(g90) * emin / 90 * att_mult
    comp["assists"] = sc.assist * v(a90) * emin / 90 * att_mult
    comp["clean_sheet"] = cs_pts * P60 * np.exp(-lam_conc * M60 / 90)
    comp["conceded"] = gc_pts * (P60 * _expected_floor_div(lam_conc * M60 / 90, 2)
                                 + PSUB * _expected_floor_div(lam_conc * MSUB / 90, 2))
    is_gk = posv == "GKP"
    save_mu = v(rate["saves"]) * M60 / 90 * (att_o / league) / hmult
    comp["saves"] = np.where(is_gk, sc.saves * P60 * _expected_floor_div(save_mu, 3), 0.0)
    comp["bonus"] = v(rate["bonus"]) * emin / 90
    comp["defcon"] = dc_pts * P60 * v(dc_rate)
    comp["cards"] = (sc.yellow * v(rate["yc"]) + sc.red * v(rate["rc"])) * emin / 90
    parts = ["appearance", "goals", "assists", "clean_sheet", "conceded", "saves", "bonus", "defcon", "cards"]
    comp["xp"] = comp[parts].sum(axis=1)
    comp["p60"] = P60
    comp["xmins"] = emin
    # sum double gameweeks; keep the opponents for display
    agg = comp.groupby(["element", "gw"]).agg({**{p: "sum" for p in parts + ["xp", "p60", "xmins"]},
                                               "opp": lambda s: list(s), "home": lambda s: list(s)})
    return agg.reset_index()


PARTS = ["appearance", "goals", "assists", "clean_sheet", "conceded", "saves", "bonus", "defcon", "cards"]
