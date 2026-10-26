"""Text and HTML rendering of plans, projections and the "why" table."""

from __future__ import annotations

import html
from datetime import datetime, timezone

import pandas as pd

from .model import PARTS
from .planner import Why
from .solver import Plan

COMPONENT_LABEL = {
    "appearance": "mins", "goals": "goals", "assists": "assists", "clean_sheet": "CS", "conceded": "conceded",
    "saves": "saves", "bonus": "bonus", "defcon": "DC", "cards": "cards",
}


def money(t: int) -> str:
    return f"{t / 10:.1f}"


def drivers(row: pd.Series, n: int = 3) -> str:
    """The biggest positive scoring components over the horizon, e.g. 'goals 4.1, CS 2.0'."""
    vals = sorted(((row.get(p, 0.0), COMPONENT_LABEL[p]) for p in PARTS if p != "appearance"), reverse=True)
    return ", ".join(f"{lbl} {v:.1f}" for v, lbl in vals[:n] if v >= 0.05)


def table_text(df: pd.DataFrame, cols: list, headers: list | None = None) -> str:
    headers = headers or cols
    cells = [[str(h) for h in headers]] + [[str(v) for v in r] for r in df[cols].itertuples(index=False)]
    widths = [max(len(r[i]) for r in cells) for i in range(len(cols))]
    lines = []
    for k, r in enumerate(cells):
        lines.append("  ".join(c.ljust(w) if j < 2 else c.rjust(w) for j, (c, w) in enumerate(zip(r, widths))))
        if k == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def week_text(plan: Plan, table: pd.DataFrame, teams: dict, gw_index: int = 0) -> str:
    w = plan.weeks[gw_index]
    t = table
    out = [f"GW{w.gw}" + (f"  chip: {w.chip}" if w.chip else "") + f"  expected {w.xp:.1f} pts"
           f"  bank {money(w.bank)}  free transfers {w.free_transfers}" + (f"  hits {w.hits} (-{4 * w.hits})" if w.hits else "")]
    if w.transfers_out or w.transfers_in:
        for o, i in zip(sorted(w.transfers_out, key=lambda x: t.at[x, "pos"]), sorted(w.transfers_in, key=lambda x: t.at[x, "pos"])):
            out.append(f"  OUT {t.at[o, 'name']} ({money(t.at[o, 'price'])})  ->  IN {t.at[i, 'name']} ({money(t.at[i, 'price'])})")
    order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    xi = sorted(w.lineup, key=lambda i: (order[t.at[i, "pos"]], -t.at[i, w.gw]))
    for label, ids in (("XI", xi), ("Bench", w.bench)):
        out.append(f"  {label}:")
        for i in ids:
            tag = " (C)" if i == w.captain else " (V)" if i == w.vice else ""
            out.append(f"    {t.at[i, 'pos']:3s} {t.at[i, 'name'] + tag:24s} {teams.get(t.at[i, 'team'], ''):4s} "
                       f"{money(t.at[i, 'price']):>5s}  {t.at[i, w.gw]:5.2f}")
    return "\n".join(out)


def plan_text(plan: Plan, table: pd.DataFrame, teams: dict) -> str:
    parts = [week_text(plan, table, teams, k) for k in range(len(plan.weeks))]
    total = sum(w.xp - 4 * w.hits for w in plan.weeks)
    parts.append(f"Horizon: {total:.1f} expected points after hits over GW{plan.weeks[0].gw}-{plan.weeks[-1].gw}")
    return "\n\n".join(parts)


def why_frame(whys: list[Why], table: pd.DataFrame, teams: dict) -> pd.DataFrame:
    rows = []
    for w in sorted(whys, key=lambda w: -w.loss):
        r = table.loc[w.id]
        rows.append({
            "player": r["name"], "pos": r.pos, "team": teams.get(r.team, ""), "price": money(r.price),
            "xp": f"{r.total:.1f}", "per_m": f"{r.total / (r.price / 10):.2f}",
            "loss": "must" if w.loss == float("inf") else f"{w.loss:.2f}",
            "drivers": drivers(r),
            "fixtures": r.fixtures,
            "instead": ", ".join(table.at[j, "name"] for j in w.replacement[:3]) or "-",
        })
    return pd.DataFrame(rows)


def why_text(whys: list[Why], table: pd.DataFrame, teams: dict) -> str:
    df = why_frame(whys, table, teams)
    head = ("Why each pick won: 'cost' is the drop in the plan's objective if the player is removed and the\n"
            "solver re-optimises around the gap; 'instead' is who it would bring in.\n")
    return head + table_text(df, ["player", "pos", "team", "price", "xp", "per_m", "loss", "drivers", "instead"],
                             ["player", "pos", "team", "price", "xP", "xP/m", "cost", "drivers", "instead"])


# ------------------------------------------------------------------------------------- HTML

CSS = """
:root{--bg:#f7f8fa;--fg:#14171c;--muted:#5b6472;--card:#fff;--line:#e2e5ea;--accent:#1f6feb;--pitch:#e9f5ec}
@media (prefers-color-scheme:dark){:root{--bg:#0f1216;--fg:#e8ebf0;--muted:#9aa4b2;--card:#171b21;--line:#2a3039;--accent:#6ea8ff;--pitch:#15241a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:12px 0}
.pitch{background:var(--pitch);border-radius:10px;padding:12px}.row{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;margin:8px 0}
.p{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 10px;min-width:118px;text-align:center}
.p b{display:block;font-size:14px}.p span{font-size:12px;color:var(--muted)}.cap{border-color:var(--accent);border-width:2px}
.scroll{overflow-x:auto;position:relative}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}td.n,th.n{text-align:right}
.kpi{display:flex;flex-wrap:wrap;gap:12px}.kpi div{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px}
.kpi b{display:block;font-size:20px}
"""


def _player_box(i, w, t, teams):
    cls = "p cap" if i == w.captain else "p"
    tag = " (C)" if i == w.captain else " (V)" if i == w.vice else ""
    return (f'<div class="{cls}"><b>{html.escape(str(t.at[i, "name"]))}{tag}</b>'
            f'<span>{teams.get(t.at[i, "team"], "")} · {money(t.at[i, "price"])} · {t.at[i, w.gw]:.1f} xP</span></div>')


def plan_html(title: str, subtitle: str, plan: Plan, table: pd.DataFrame, teams: dict,
              whys: list[Why] | None = None, extra: str = "") -> str:
    e = html.escape
    w = plan.first
    body = [f"<h1>{e(title)}</h1><p class='muted'>{e(subtitle)}</p>"]
    total = sum(x.xp - 4 * x.hits for x in plan.weeks)
    body.append("<div class='kpi'>"
                f"<div><span class='muted'>GW{w.gw} expected</span><b>{w.xp:.1f}</b></div>"
                f"<div><span class='muted'>Horizon (after hits)</span><b>{total:.1f}</b></div>"
                f"<div><span class='muted'>Bank</span><b>{money(w.bank)}</b></div>"
                f"<div><span class='muted'>Hits this week</span><b>{w.hits}</b></div></div>")
    if w.transfers_in:
        moves = "".join(f"<li>{e(table.at[o, 'name'])} → <b>{e(table.at[i, 'name'])}</b></li>"
                        for o, i in zip(w.transfers_out, w.transfers_in))
        body.append(f"<div class='card'><b>Transfers for GW{w.gw}</b><ul>{moves}</ul></div>")
    order = ["GKP", "DEF", "MID", "FWD"]
    rows = "".join(
        "<div class='row'>" + "".join(_player_box(i, w, table, teams) for i in w.lineup if table.at[i, "pos"] == p) + "</div>"
        for p in order)
    bench = "".join(_player_box(i, w, table, teams) for i in w.bench)
    body.append(f"<h2>GW{w.gw} team" + (f" · {e(w.chip)}" if w.chip else "") + "</h2>"
                f"<div class='pitch'>{rows}<p class='muted' style='text-align:center;margin:8px 0 0'>Bench, in order</p>"
                f"<div class='row'>{bench}</div></div>")
    if len(plan.weeks) > 1:
        lines = []
        for x in plan.weeks:
            moves = ", ".join(f"{table.at[o, 'name']} → {table.at[i, 'name']}" for o, i in zip(x.transfers_out, x.transfers_in))
            lines.append(f"<tr><td>GW{x.gw}</td><td>{e(x.chip or '')}</td><td>{e(moves or 'hold')}</td>"
                         f"<td class='n'>{x.hits}</td><td class='n'>{x.free_transfers}</td><td>{e(table.at[x.captain, 'name'])}</td>"
                         f"<td class='n'>{x.xp:.1f}</td></tr>")
        body.append("<h2>Plan over the horizon</h2><div class='scroll'><table><tr><th>GW</th><th>Chip</th><th>Moves</th>"
                    "<th class='n'>Hits</th><th class='n'>Free transfers</th><th>Captain</th><th class='n'>xP</th></tr>"
                    + "".join(lines) + "</table></div>")
    if whys:
        df = why_frame(whys, table, teams)
        cells = "".join(
            f"<tr><td>{e(r.player)}</td><td>{r.pos}</td><td>{e(r.team)}</td><td class='n'>{r.price}</td>"
            f"<td class='n'>{r.xp}</td><td class='n'>{r.per_m}</td><td class='n'>{r.loss}</td><td>{e(r.drivers)}</td>"
            f"<td>{e(r.fixtures)}</td><td>{e(r.instead)}</td></tr>" for r in df.itertuples())
        body.append("<h2>Why each pick won</h2><p class='muted'>Cost = points the plan loses if the player is "
                    "removed and the solver re-optimises around the gap. Drivers = largest expected scoring "
                    "components over the horizon.</p><div class='scroll'><table><tr><th>Player</th><th>Pos</th>"
                    "<th>Club</th><th class='n'>Price</th><th class='n'>xP</th><th class='n'>xP/m</th>"
                    "<th class='n'>Cost</th><th>Drivers</th><th>Fixtures</th><th>Instead</th></tr>" + cells + "</table></div>")
    body.append(extra)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body.append(f"<p class='muted'>Generated {stamp} by fpl-optimizer from the public FPL API.</p>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{e(title)}</title><style>{CSS}</style></head>"
            f"<body><main>{''.join(body)}</main></body></html>")
