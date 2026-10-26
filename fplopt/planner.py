"""Glue between projections and the solver: candidate pools, plans and "why this pick won".

The "why" is counterfactual: for each player in the plan, the solver is re-run with that
player banned, and the drop in the objective is what he is worth to this squad over the
horizon after the best possible replacement. A player who costs 0.1 to remove was a coin
flip; one who costs 6 was the pick the plan is built around.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from .model import PARTS
from .rules import Rules
from .solver import Candidate, InfeasibleError, Plan, Settings, solve

SHORTLIST = {"GKP": 14, "DEF": 40, "MID": 45, "FWD": 25}


def horizon_table(proj: pd.DataFrame, players: pd.DataFrame, gws: list, teams: dict) -> pd.DataFrame:
    """One row per player: xp per gameweek, horizon total, components and fixtures."""
    p = proj[proj.gw.isin(gws)]
    wide = p.pivot_table(index="element", columns="gw", values="xp", aggfunc="sum").reindex(columns=gws).fillna(0.0)
    comp = p.groupby("element")[PARTS + ["xmins"]].sum()
    out = players.join(wide, how="inner").join(comp, how="left")
    out["total"] = wide.sum(axis=1).reindex(out.index)
    fx = p.groupby("element").apply(
        lambda d: " ".join(
            f"{teams.get(o, o)}{'(H)' if h else '(a)'}" for os_, hs in zip(d.opp, d.home) for o, h in zip(os_, hs)
        ),
        include_groups=False,
    )
    out["fixtures"] = fx.reindex(out.index).fillna("-")
    out["xmins_next"] = p[p.gw == gws[0]].set_index("element").xmins.reindex(out.index).fillna(0.0)
    return out


def shortlist(table: pd.DataFrame, keep: set, sizes: dict = SHORTLIST) -> pd.DataFrame:
    """Top players per position by horizon xp and by xp per GBP, the cheapest few, plus `keep`."""
    picks = set(keep)
    for pos, n in sizes.items():
        t = table[table.pos == pos]
        picks |= set(t.nlargest(n, "total").index)
        picks |= set((t.total / t.price).nlargest(n // 3).index)
        picks |= set(t[t.xmins_next > 30].nsmallest(4, "price").index)
        picks |= set(t.nsmallest(3, "price").index)
    return table.loc[sorted(i for i in picks if i in table.index)]


def candidates(table: pd.DataFrame, gws: list, owned_sell: dict | None = None) -> list[Candidate]:
    owned_sell = owned_sell or {}
    out = []
    for i, r in table.iterrows():
        out.append(
            Candidate(
                id=int(i), name=str(r["name"]), pos=r.pos, team=int(r.team), buy=int(r.price),
                xp={gw: float(r[gw]) for gw in gws}, owned=i in owned_sell,
                sell=owned_sell.get(i),
            )
        )
    return out


@dataclass
class Why:
    id: int
    loss: float  # objective points lost if this player is banned
    replacement: list  # who the re-solved plan brings in instead


def why(plan: Plan, cands: list[Candidate], st: Settings, rules: Rules, ids: list) -> list[Why]:
    """Counterfactual value of each player in `ids` (re-solve with him banned)."""
    out = []
    base = set(plan.first.squad)
    owned = {c.id for c in cands if c.owned} if not st.draft else set()
    pos = {c.id: c.pos for c in cands}
    for i in ids:
        if i in owned:  # an owned player cannot vanish: make the plan sell him this week instead
            alt_st = replace(st, sold=set(st.sold) | {i}, forced=set(st.forced) - {i})
        else:
            alt_st = replace(st, banned=set(st.banned) | {i}, forced=set(st.forced) - {i})
        try:
            alt = solve(cands, alt_st, rules)
        except InfeasibleError:
            out.append(Why(i, float("inf"), []))
            continue
        new = set(alt.first.squad) - base
        # same-position replacements first; other changes are the solver rebalancing the budget
        out.append(Why(i, plan.objective - alt.objective, sorted(new, key=lambda j: (pos[j] != pos[i], j))))
    return out
