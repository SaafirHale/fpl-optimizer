"""Command line: fplopt <command> [options]. Run `fplopt -h` for the list."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import api
from .backtest import PREV_OF, team_map_between
from .data import RAW, Season, fetch_history, load_history, load_live
from .model import Params, project
from .planner import candidates, horizon_table, shortlist, why
from .report import money, plan_html, plan_text, table_text, why_text
from .solver import Settings, solve
from .team import chips_available, load_team

ROOT = Path(__file__).resolve().parent.parent
CHIP_ALIASES = {"wc": "wildcard", "wildcard": "wildcard", "fh": "freehit", "freehit": "freehit",
                "bb": "bboost", "bboost": "bboost", "tc": "3xc", "3xc": "3xc"}


@dataclass
class Context:
    season: Season
    gws: list
    table: pd.DataFrame


def live_context(horizon: int, refresh: bool = False) -> Context:
    season = load_live(refresh)
    prev_name = PREV_OF.get(season.name)
    if not (RAW / prev_name / "merged_gw.csv").exists():
        print(f"downloading {prev_name} history (one-off, ~6 MB)...", file=sys.stderr)
        fetch_history(prev_name)
    prev = load_history(prev_name)
    start = season.next_gw or (season.current_gw or 0) + 1
    last = int(season.fixtures.gw.max())
    gws = list(range(start, min(start + horizon, last + 1)))
    sched = season.fixtures[season.fixtures.gw.isin(gws)][["gw", "home_team", "away_team"]]
    proj = project(season.players, sched, season.rows, prev.rows, prev.players.pos,
                   team_map_between(prev, season), season.scoring, Params(), first_gw=start)
    table = horizon_table(proj, season.players, gws, season.teams)
    return Context(season, gws, table)


def parse_chips(specs: list[str]) -> dict:
    out = {}
    for spec in specs or []:
        name, _, gw = spec.partition("@")
        if name.lower() not in CHIP_ALIASES or not gw.isdigit():
            raise SystemExit(f"bad --chip {spec!r}: use e.g. wildcard@10, fh@12, bb@14, tc@11")
        out[int(gw)] = CHIP_ALIASES[name.lower()]
    return out


def write_html(path: str | None, page: str) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(page, encoding="utf-8")
        print(f"\nwrote {path}")


def cmd_project(a) -> None:
    ctx = live_context(a.horizon, a.refresh)
    t = ctx.table.copy()
    if a.pos:
        t = t[t.pos == a.pos.upper()]
    t = t.nlargest(a.top, "total")
    t["team_s"] = t.team.map(ctx.season.teams)
    t["£"] = t.price.map(money)
    for gw in ctx.gws:
        t[f"GW{gw}"] = t[gw].map(lambda v: f"{v:.2f}")
    t["xP"] = t.total.map(lambda v: f"{v:.1f}")
    t["mins"] = t.xmins_next.map(lambda v: f"{v:.0f}")
    t["flag"] = t.chance.map(lambda c: "" if pd.isna(c) else f"{c:.0f}%")
    cols = ["name", "pos", "team_s", "£", "mins", "flag"] + [f"GW{g}" for g in ctx.gws] + ["xP"]
    print(f"Projections for GW{ctx.gws[0]}-{ctx.gws[-1]} (data to GW{ctx.season.current_gw})\n")
    print(table_text(t, cols, ["player", "pos", "club", "£m", "xMins", "flag"] + [f"GW{g}" for g in ctx.gws] + ["total"]))


def _settings(a, ctx, **kw) -> Settings:
    return Settings(gws=ctx.gws, chips=parse_chips(a.chip), decay=a.decay, ft_value=a.ft_value,
                    hit_margin=a.hit_margin, time_limit=a.time_limit, **kw)


def cmd_squad(a) -> None:
    ctx = live_context(a.horizon, a.refresh)
    pool = shortlist(ctx.table, set())
    cands = candidates(pool, ctx.gws)
    st = _settings(a, ctx, draft=True)
    rules = ctx.season.rules
    if a.budget:
        rules = type(rules)(**{**rules.__dict__, "budget": int(round(a.budget * 10))})
    t0 = time.time()
    plan = solve(cands, st, rules)
    print(f"Best squad from scratch, budget {money(rules.budget)}m, planned over GW{ctx.gws[0]}-{ctx.gws[-1]} "
          f"({len(cands)} candidates, solved in {time.time() - t0:.1f}s)\n")
    print(plan_text(plan, ctx.table, ctx.season.teams))
    whys = None
    if not a.no_why:
        print("\nScoring each pick against its best replacement (15 re-solves)...")
        whys = why(plan, cands, st, rules, plan.first.squad)
        print(why_text(whys, ctx.table, ctx.season.teams))
    write_html(a.html, plan_html(f"Best FPL squad, GW{ctx.gws[0]}-{ctx.gws[-1]}",
                                 f"Wildcard-style draft on a {money(rules.budget)}m budget", plan, ctx.table,
                                 ctx.season.teams, whys))


def cmd_team(a) -> None:
    ctx = live_context(a.horizon, a.refresh)
    bs = api.bootstrap()
    team = load_team(a.team_id, bs, ctx.season.rules)
    ft = a.ft if a.ft is not None else team.free_transfers
    print(f"{team.name} (ID {team.id}): picks from GW{team.gw}, bank {money(team.bank)}m, "
          f"{ft} free transfer(s){' (set by --ft)' if a.ft is not None else ' (replayed from history)'}")
    for n in team.notes:
        print(f"  note: {n}")
    avail = chips_available(bs, team.chips_used, ctx.gws[0])
    print(f"  chips available in GW{ctx.gws[0]}: {', '.join(avail) or 'none'}")
    missing = [i for i in team.picks if i not in ctx.table.index]
    if missing:
        raise SystemExit(f"players {missing} have no projection")
    pool = shortlist(ctx.table, set(team.picks))
    cands = candidates(pool, ctx.gws, team.sell)
    st = _settings(a, ctx, bank=team.bank, free_transfers=ft, draft=False)
    rules = ctx.season.rules
    plan = solve(cands, st, rules)
    hold = solve(cands, _settings(a, ctx, bank=team.bank, free_transfers=ft, transfer_limits={ctx.gws[0]: 0}), rules)
    print()
    print(plan_text(plan, ctx.table, ctx.season.teams))
    gain = plan.objective - hold.objective
    print(f"\nPlan vs rolling the transfer this week: {gain:+.2f} objective points over the horizon")
    whys = None
    if not a.no_why:
        w = plan.first
        ids = w.transfers_in + [i for i in w.lineup if i not in w.transfers_in][: 0 if a.quick else 11]
        whys = why(plan, cands, st, rules, ids)
        print(why_text(whys, ctx.table, ctx.season.teams))
    extra = (f"<div class='card'>Selling prices use the FPL rule (keep half of any rise). Purchase prices and "
             f"free transfers are reconstructed from public data: {ft} free transfer(s), bank {money(team.bank)}m."
             f" Plan vs rolling the transfer: {gain:+.2f}.</div>")
    write_html(a.html, plan_html(f"{team.name}: GW{ctx.gws[0]} plan", f"FPL team {team.id}", plan, ctx.table,
                                 ctx.season.teams, whys, extra))


def cmd_chips(a) -> None:
    """Every available chip in every planned week, against the no-chip plan."""
    ctx = live_context(a.horizon, a.refresh)
    bs = api.bootstrap()
    if a.team_id:
        team = load_team(a.team_id, bs, ctx.season.rules)
        pool = shortlist(ctx.table, set(team.picks))
        cands = candidates(pool, ctx.gws, team.sell)
        base_kw = dict(bank=team.bank, free_transfers=a.ft if a.ft is not None else team.free_transfers, draft=False)
        used = team.chips_used
    else:
        pool = shortlist(ctx.table, set())
        cands = candidates(pool, ctx.gws)
        base_kw = dict(draft=True)
        used = []
    rules = ctx.season.rules
    base = solve(cands, _settings(a, ctx, **base_kw), rules)
    rows = []
    for gw in ctx.gws:
        for chip in chips_available(bs, used, gw):
            if base_kw.get("draft") and gw == ctx.gws[0] and chip in ("freehit", "wildcard"):
                continue
            a.chip = [f"{chip}@{gw}"]
            p = solve(cands, _settings(a, ctx, **base_kw), rules)
            w = next(x for x in p.weeks if x.gw == gw)
            rows.append({"chip": chip, "gw": gw, "gain": p.objective - base.objective, "week_xp": w.xp,
                         "moves": len(w.transfers_in)})
    if not rows:
        print(f"No chips left to play in GW{ctx.gws[0]}-{ctx.gws[-1]}.")
        return
    df = pd.DataFrame(rows).sort_values("gain", ascending=False)
    df["gain"] = df.gain.map(lambda v: f"{v:+.2f}")
    df["week_xp"] = df.week_xp.map(lambda v: f"{v:.1f}")
    print(f"Chip scenarios over GW{ctx.gws[0]}-{ctx.gws[-1]} (gain = objective vs the same plan with no chip)\n")
    print(table_text(df, ["chip", "gw", "gain", "week_xp", "moves"], ["chip", "GW", "gain", "week xP", "moves"]))


def cmd_backtest(a) -> None:
    from . import backtest

    out = ROOT / "reports"
    for season in a.seasons:
        live = season == "live"
        prev = "2025-26" if live else PREV_OF.get(season)
        for s in ([prev] if live else [season, prev]):
            if s and not (RAW / s / "merged_gw.csv").exists():
                print(f"downloading {s} history...")
                fetch_history(s)
        t0 = time.time()
        if live:
            cur, done = backtest.live_season()
            season = f"{cur.name}-gw{done[0]}-{done[-1]}"
            df = backtest.run_season(cur, load_history(prev), Params())
        else:
            df = backtest.run(season, Params())
        m = backtest.save(df, season, Params(), out)
        print(f"{season}: {m['rows']} player-gameweeks, {time.time() - t0:.0f}s -> reports/backtest-{season}.json")
        for name in backtest.MODELS:
            r = m[name]
            print(f"  {name:6s} MAE {r['mae']:.3f}  RMSE {r['rmse']:.3f}  r {r['pearson']:.3f}  "
                  f"rank/GW {r['spearman_per_gw']:.3f}  top-50 {r['top50_overlap']:.3f}  captain {r['captain_pts']:.2f}")


def cmd_fetch(a) -> None:
    load_live(refresh=True)
    print("live FPL data refreshed (data/cache)")
    for s in a.history or []:
        fetch_history(s)
        print(f"downloaded {s} history to data/raw/{s}")


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:  # pragma: no cover
        pass
    ap = argparse.ArgumentParser(prog="fplopt", description="FPL expected points + integer-programming optimizer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, horizon=5):
        p.add_argument("--horizon", type=int, default=horizon, help="gameweeks to plan (default %(default)s)")
        p.add_argument("--refresh", action="store_true", help="re-fetch live FPL data now")

    def solver_opts(p):
        p.add_argument("--chip", action="append", help="play a chip, e.g. wildcard@10, fh@12, bb@14, tc@11")
        p.add_argument("--decay", type=float, default=0.9, help="weight of each later gameweek")
        p.add_argument("--ft-value", type=float, default=1.5, help="points per free transfer banked at the end")
        p.add_argument("--hit-margin", type=float, default=8.0,
                       help="extra objective cost per -4 hit (default chosen by replaying 2024/25)")
        p.add_argument("--time-limit", type=int, default=60)
        p.add_argument("--no-why", action="store_true", help="skip the counterfactual re-solves")
        p.add_argument("--html", help="also write an HTML report to this path")

    p = sub.add_parser("project", help="top projected players")
    common(p)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--pos", choices=["GKP", "DEF", "MID", "FWD", "gkp", "def", "mid", "fwd"])
    p.set_defaults(fn=cmd_project)

    p = sub.add_parser("squad", help="best 15 from scratch (draft / wildcard what-if)")
    common(p)
    solver_opts(p)
    p.add_argument("--budget", type=float, help="budget in GBP m (default 100.0)")
    p.set_defaults(fn=cmd_squad)

    p = sub.add_parser("team", help="import an FPL team by ID and plan its transfers")
    common(p)
    solver_opts(p)
    p.add_argument("team_id", type=int)
    p.add_argument("--ft", type=int, help="override the reconstructed free-transfer count")
    p.add_argument("--quick", action="store_true", help="explain transfers only, not the whole XI")
    p.set_defaults(fn=cmd_team)

    p = sub.add_parser("chips", help="compare every available chip in every planned week")
    common(p, horizon=4)
    solver_opts(p)
    p.add_argument("team_id", type=int, nargs="?")
    p.add_argument("--ft", type=int)
    p.set_defaults(fn=cmd_chips)

    p = sub.add_parser("backtest", help="walk-forward backtest of the projections ('live' = this season so far)")
    p.add_argument("seasons", nargs="*", default=["2024-25", "2025-26"])
    p.set_defaults(fn=cmd_backtest)

    p = sub.add_parser("fetch", help="refresh live data and optionally download history")
    p.add_argument("--history", nargs="*", help="seasons to download, e.g. 2024-25 2025-26")
    p.set_defaults(fn=cmd_fetch)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
