"""Season replay: run the optimizer week by week through a finished season and score it.

Each gameweek t:
  1. project gameweeks t..t+H-1 from data before t (the walk-forward model in backtest.py)
  2. solve with the squad, bank, selling prices and free transfers carried from last week
  3. play week t only (a rolling horizon), then score it on the points players really got:
     automatic substitutions in bench order when a starter gets no minutes (formation rules
     respected), vice-captain doubles when the captain does not play, -4 per hit
GW1 is a free draft on the GBP 100m budget. Prices are each player's real price that week;
selling prices follow the keep-half-the-rise rule. No chips are played.

`strategy="form"` swaps the model for the naive projection (mean points over the last four
gameweeks, used for every planned week) with the same solver, which isolates what the
projections are worth.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .backtest import PREV_OF, baselines, players_at, project_gw
from .data import load_history
from .model import Params
from .rules import Rules, selling_price, valid_formation
from .solver import Candidate, Settings, solve

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Week:
    gw: int
    points: int
    hits: int
    transfers: int
    captain: str
    autosubs: int
    bank: int


@dataclass
class Result:
    season: str
    strategy: str
    total: int
    weeks: list = field(default_factory=list)


def score_week(squad_order: list, captain: int, vice: int, pts: dict, mins: dict, pos: dict,
               rules: Rules = Rules()) -> tuple[int, int]:
    """FPL scoring of one week: lineup = first 11 of `squad_order`, bench = last 4 (GK first)."""
    xi, bench = list(squad_order[:11]), list(squad_order[11:])
    subs = 0
    played = lambda i: mins.get(i, 0) > 0  # noqa: E731
    gk_bench = bench[0]
    for k, i in enumerate(xi):
        if pos[i] == "GKP" and not played(i) and played(gk_bench):
            xi[k] = gk_bench
            subs += 1
    for b in bench[1:]:
        if not played(b):
            continue
        for k, i in enumerate(xi):
            if pos[i] == "GKP" or played(i):
                continue
            trial = xi[:k] + [b] + xi[k + 1:]
            counts = {p: sum(pos[j] == p for j in trial) for p in ("GKP", "DEF", "MID", "FWD")}
            if valid_formation(counts, rules):
                xi = trial
                subs += 1
                break
    total = sum(pts.get(i, 0) for i in xi)
    if played(captain) and captain in xi:
        total += pts.get(captain, 0)
    elif played(vice) and vice in xi:
        total += pts.get(vice, 0)
    return int(total), subs


def run(season_name: str = "2025-26", strategy: str = "model", horizon: int = 3, params: Params = Params(),
        verbose: bool = True, time_limit: int = 30, until: int | None = None, hit_margin: float = 0.0,
        ft_value: float = 1.5) -> Result:
    season = load_history(season_name)
    prev = load_history(PREV_OF[season_name])
    rules = season.rules
    last_gw = int(season.rows.gw.max())
    stop = until or last_gw
    base = baselines(season, prev) if strategy == "form" else None
    pos_all = season.players.pos.to_dict()
    names = season.players.name.to_dict()
    latest = season.rows.sort_values("gw").groupby("element")  # for owned players missing a row
    owned: dict = {}  # element -> purchase price
    bank, ft = rules.budget, 0
    res = Result(season_name, strategy, 0)
    for gw in range(1, stop + 1):
        t0 = time.time()
        gws = [g for g in range(gw, gw + horizon) if g <= last_gw]
        P = players_at(season, gw)
        missing = [i for i in owned if i not in P.index]
        if missing:  # owned but no row this week: blank gameweek or left the league
            prior_rows = season.rows[(season.rows.gw < gw) & season.rows.element.isin(missing)]
            last = prior_rows.sort_values("gw").groupby("element").last()
            playing = set(season.fixtures[season.fixtures.gw == gw][["home_team", "away_team"]].values.ravel())
            gone = last.team.isin(playing)  # his club plays this week but he has no row: he has left
            extra = pd.DataFrame({"code": last.code, "team": last.team, "price": last.price,
                                  "pos": [pos_all[i] for i in last.index],
                                  "status": gone.map({True: "u", False: "a"}), "chance": gone.map({True: 0.0, False: float("nan")})})
            P = pd.concat([P, extra])
        if strategy == "model":
            proj = project_gw(season, prev, gw, params, horizon=len(gws), players=P)
            xp = proj.pivot_table(index="element", columns="gw", values="xp", aggfunc="sum")
        else:
            f = base[base.gw < gw].sort_values("gw").groupby("element").last()
            # form known before this deadline: last week's form value already excludes week gw itself
            form_now = base[base.gw == gw].set_index("element").form
            per = (form_now / base[base.gw == gw].set_index("element").n).reindex(P.index)
            per = per.fillna(f.form.reindex(P.index)).fillna(0.0)
            fixtures = season.fixtures[season.fixtures.gw.isin(gws)]
            nfx = {g: pd.concat([fixtures[fixtures.gw == g].home_team, fixtures[fixtures.gw == g].away_team]).value_counts()
                   for g in gws}
            xp = pd.DataFrame({g: per * P.team.map(nfx[g]).fillna(0).values for g in gws}, index=P.index)
        xp = xp.reindex(P.index).fillna(0.0)
        xp.loc[P.index[P.status.eq("u").values]] = 0.0
        cands = []
        for i, r in P.iterrows():
            o = i in owned
            cands.append(Candidate(int(i), names.get(i, str(i)), r.pos, int(r.team), int(r.price),
                                   {g: float(xp.at[i, g]) if g in xp.columns else 0.0 for g in gws}, owned=o,
                                   sell=selling_price(owned[i], int(r.price), rules.sell_on_fee) if o else None))
        # keep the pool manageable: owned + top players per position by horizon xp + cheapest
        tot = {c.id: sum(c.xp.values()) for c in cands}
        keep = set(owned)
        for p, n in {"GKP": 12, "DEF": 35, "MID": 40, "FWD": 22}.items():
            cs = [c for c in cands if c.pos == p]
            keep |= {c.id for c in sorted(cs, key=lambda c: -tot[c.id])[:n]}
            keep |= {c.id for c in sorted(cs, key=lambda c: c.buy)[:4]}
        pool = [c for c in cands if c.id in keep]
        st = Settings(gws=gws, bank=bank, free_transfers=ft, draft=(gw == 1), time_limit=time_limit, gap=0.001,
                      hit_margin=hit_margin, ft_value=ft_value)
        plan = solve(pool, st, rules)
        w = plan.first
        for i in w.transfers_out:
            owned.pop(i)
        price_now = {c.id: c.buy for c in pool}
        for i in (w.squad if gw == 1 else w.transfers_in):
            owned[i] = price_now[i]
        assert set(owned) == set(w.squad)
        bank = w.bank
        n_in = len(w.transfers_in)
        ft = 1 if gw == 1 else min(rules.max_free_transfers, max(ft - n_in, 0) + 1)
        week_rows = season.rows[season.rows.gw == gw]
        pts = week_rows.groupby("element").points.sum().to_dict()
        mins = week_rows.groupby("element").minutes.sum().to_dict()
        order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
        xi = sorted(w.lineup, key=lambda i: order[pos_all[i]])
        got, subs = score_week(xi + w.bench, w.captain, w.vice, pts, mins, pos_all, rules)
        got -= rules.hit_cost * w.hits
        res.total += got
        res.weeks.append(Week(gw, got, w.hits, n_in, names.get(w.captain, ""), subs, bank))
        if verbose:
            print(f"  {strategy} GW{gw:2d}: {got:3d} pts  (transfers {n_in}, hits {w.hits}, cap {names.get(w.captain)}, "
                  f"autosubs {subs}, bank {bank / 10:.1f})  total {res.total}  [{time.time() - t0:.0f}s]", flush=True)
    return res


def save(results: list[Result], out: Path = ROOT / "reports") -> None:
    out.mkdir(exist_ok=True)
    for r in results:
        data = {"season": r.season, "strategy": r.strategy, "total": r.total,
                "weeks": [w.__dict__ for w in r.weeks]}
        (out / f"simulation-{r.season}-{r.strategy}.json").write_text(json.dumps(data, indent=2, default=int))


if __name__ == "__main__":
    import sys

    season = sys.argv[1] if len(sys.argv) > 1 else "2025-26"
    strategies = sys.argv[2:] or ["model", "form"]
    rs = []
    for s in strategies:
        rs.append(run(season, s))
        save(rs)
    for r in rs:
        print(r.strategy, r.total, "hits", sum(w.hits for w in r.weeks))
