"""Multi-gameweek FPL squad and transfer optimizer as a mixed-integer linear program (PuLP + CBC).

Decision variables, for each candidate player i and planned gameweek t:
    x[i,t]   in the squad after this week's transfers          (binary)
    s[i,t]   in the starting XI                                  (binary)
    c[i,t]   captain, v[i,t] vice-captain                        (binary)
    b[i,t,k] k-th outfield substitute, k = 1..3                  (binary)
    tin/tout transfers in / out                                  (binary)
    y[i,t]   Free Hit squad for a Free Hit week                  (binary)
    ft[t]    free transfers available, pen[t] paid transfers      (integer)

Constraints: 15 players (2/5/5/3), at most 3 per club, budget through the bank with selling
prices for players already owned, a legal formation, one captain and one vice in the XI, an
ordered bench, and free transfers that roll over up to the cap with -4 per extra transfer.
Wildcard: transfers are free that week. Free Hit: a one-week squad on the same budget, then
the old squad returns. Bench Boost: the bench scores in full. Triple Captain: captain x3.

Assumptions, stated rather than hidden: prices stay flat over the horizon; a player sold from
the starting squad is not bought back inside the horizon; free transfers are kept (not
increased) through a Wildcard or Free Hit week; bench weights approximate the chance that a
substitute comes on and are tie-breakers, not forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pulp

from .rules import POSITIONS, Rules

CHIPS = ("wildcard", "freehit", "bboost", "3xc")


class InfeasibleError(RuntimeError):
    pass


@dataclass
class Candidate:
    id: int
    name: str
    pos: str
    team: int
    buy: int  # price to buy now, tenths
    xp: dict  # gw -> expected points
    owned: bool = False
    sell: int | None = None  # selling price if owned


@dataclass
class Settings:
    gws: list
    bank: int = 0  # tenths; ignored in draft mode (the budget is rules.budget)
    free_transfers: int = 1
    draft: bool = False  # pick a squad from scratch (season start / what-if); first week is free
    chips: dict = field(default_factory=dict)  # gw -> chip name
    decay: float = 0.9  # weight of each later gameweek
    bench_weights: tuple = (0.15, 0.05, 0.01)
    bench_gk_weight: float = 0.02
    vice_weight: float = 0.05
    ft_value: float = 1.5  # points credited per free transfer still banked after the horizon
    transfer_friction: float = 0.01  # tiny cost per transfer so equal-value swaps are skipped
    hit_margin: float = 0.0  # extra objective cost per hit on top of the real -4 (projection noise guard)
    banned: set = field(default_factory=set)
    forced: set = field(default_factory=set)  # must be in the squad every planned week
    sold: set = field(default_factory=set)  # owned players that must go in the first week
    max_transfers: int | None = None  # per week, outside Wildcard / Free Hit
    transfer_limits: dict = field(default_factory=dict)  # gw -> max transfers that week
    time_limit: int = 60
    gap: float = 0.0


@dataclass
class WeekPlan:
    gw: int
    chip: str | None
    squad: list
    lineup: list
    captain: int
    vice: int
    bench: list  # [bench GK, sub 1, sub 2, sub 3]
    transfers_in: list
    transfers_out: list
    free_transfers: int
    hits: int
    bank: int
    xp: float  # expected points of this week's XI with captaincy and chip


@dataclass
class Plan:
    weeks: list
    objective: float
    status: str

    @property
    def first(self) -> WeekPlan:
        return self.weeks[0]


def solve(cands: list[Candidate], st: Settings, rules: Rules = Rules()) -> Plan:
    gws = list(st.gws)
    T = range(len(gws))
    for gw, chip in st.chips.items():
        if chip not in CHIPS:
            raise ValueError(f"unknown chip {chip!r}; use one of {CHIPS}")
        if gw not in gws:
            raise ValueError(f"chip {chip} is set for GW{gw}, outside the horizon {gws}")
    if st.draft and st.chips.get(gws[0]) == "freehit":
        raise ValueError("a Free Hit needs an existing squad to return to")
    cands = [c for c in cands if c.id not in st.banned]
    by_id = {c.id: c for c in cands}
    ids = list(by_id)
    owned = {i for i in ids if by_id[i].owned} if not st.draft else set()
    if not st.draft and len(owned) != rules.squad_size:
        raise ValueError(f"expected {rules.squad_size} owned players, got {len(owned)}")
    sell = {i: (by_id[i].sell if i in owned and by_id[i].sell is not None else by_id[i].buy) for i in ids}
    buy = {i: by_id[i].buy for i in ids}
    xp = {(i, t): float(by_id[i].xp.get(gws[t], 0.0)) for i in ids for t in T}
    outfield = [i for i in ids if by_id[i].pos != "GKP"]
    gks = [i for i in ids if by_id[i].pos == "GKP"]
    teams = sorted({by_id[i].team for i in ids})
    chip = {t: st.chips.get(gws[t]) for t in T}
    fh = [t for t in T if chip[t] == "freehit"]

    prob = pulp.LpProblem("fpl", pulp.LpMaximize)
    B = pulp.LpVariable.dicts
    x = B("x", [(i, t) for i in ids for t in T], cat="Binary")
    s = B("s", [(i, t) for i in ids for t in T], cat="Binary")
    c = B("c", [(i, t) for i in ids for t in T], cat="Binary")
    v = B("v", [(i, t) for i in ids for t in T], cat="Binary")
    b = B("b", [(i, t, k) for i in outfield for t in T for k in range(3)], cat="Binary")
    tin = B("tin", [(i, t) for i in ids for t in T], cat="Binary")
    tout = B("tout", [(i, t) for i in ids for t in T], cat="Binary")
    y = B("y", [(i, t) for i in ids for t in fh], cat="Binary")
    z = B("z", [(i, t) for i in owned for t in fh], cat="Binary")
    bank = B("bank", list(T), lowBound=0)
    ft = B("ft", list(range(len(gws) + 1)), lowBound=0, upBound=rules.max_free_transfers, cat="Integer")
    pen = B("pen", list(T), lowBound=0, cat="Integer")
    q = B("q", list(T), cat="Binary")  # ft[t+1] hits the cap
    r = B("r", list(T), cat="Binary")  # paid transfers this week (so no free ones left unused)
    nt = {t: pulp.lpSum(tin[i, t] for i in ids) for t in T}

    def squad(i, t):
        return y[i, t] if t in fh else x[i, t]

    obj = []
    for t in T:
        w = st.decay**t
        bb = chip[t] == "bboost"
        cap_extra = 2 if chip[t] == "3xc" else 1
        bw = (1.0, 1.0, 1.0) if bb else st.bench_weights
        gkw = 1.0 if bb else st.bench_gk_weight
        obj += [w * xp[i, t] * (s[i, t] + cap_extra * c[i, t] + st.vice_weight * v[i, t]) for i in ids]
        obj += [w * xp[i, t] * bw[k] * b[i, t, k] for i in outfield for k in range(3)]
        obj += [w * xp[i, t] * gkw * (squad(i, t) - s[i, t]) for i in gks]
        obj += [-(rules.hit_cost + st.hit_margin) * pen[t] - st.transfer_friction * nt[t]]
    obj.append(st.ft_value * ft[len(gws)])
    prob += pulp.lpSum(obj)

    for t in T:
        for sq in ([x] + ([y] if t in fh else [])):
            for p in POSITIONS:
                prob += pulp.lpSum(sq[i, t] for i in ids if by_id[i].pos == p) == rules.squad_count[p]
            for tm in teams:
                prob += pulp.lpSum(sq[i, t] for i in ids if by_id[i].team == tm) <= rules.team_limit
        prob += pulp.lpSum(s[i, t] for i in ids) == rules.starters
        for p in POSITIONS:
            n = pulp.lpSum(s[i, t] for i in ids if by_id[i].pos == p)
            prob += n >= rules.play_min[p]
            prob += n <= rules.play_max[p]
        prob += pulp.lpSum(c[i, t] for i in ids) == 1
        prob += pulp.lpSum(v[i, t] for i in ids) == 1
        for i in ids:
            prob += s[i, t] <= squad(i, t)
            prob += c[i, t] <= s[i, t]
            prob += v[i, t] <= s[i, t]
            prob += c[i, t] + v[i, t] <= 1
        for k in range(3):
            prob += pulp.lpSum(b[i, t, k] for i in outfield) == 1
        for i in outfield:
            prob += pulp.lpSum(b[i, t, k] for k in range(3)) <= squad(i, t) - s[i, t]
        for i in st.forced:
            if i in by_id:
                prob += squad(i, t) == 1
        if gws[t] in st.transfer_limits:
            prob += nt[t] <= st.transfer_limits[gws[t]]
    for i in st.sold:
        if i in owned:
            prob += x[i, 0] == 0

    # ---- squad continuity, budget and transfers
    for t in T:
        first_draft = st.draft and t == 0
        for i in ids:
            prev = (1 if i in owned else 0) if t == 0 else x[i, t - 1]
            if first_draft:
                prob += tout[i, t] == 0
                prob += tin[i, t] == x[i, t]
            else:
                prob += x[i, t] == prev + tin[i, t] - tout[i, t]
                if i in owned:
                    prob += tin[i, t] == 0  # no buy-back of a sold starting-squad player
                if t in fh:
                    prob += tin[i, t] == 0
                    prob += tout[i, t] == 0
        if first_draft:
            prob += bank[t] == rules.budget - pulp.lpSum(buy[i] * x[i, t] for i in ids)
        else:
            prev_bank = st.bank if t == 0 else bank[t - 1]
            prob += bank[t] == prev_bank + pulp.lpSum(sell[i] * tout[i, t] - buy[i] * tin[i, t] for i in ids)
        if t in fh:
            prev_bank = st.bank if t == 0 else bank[t - 1]
            value_prev = pulp.lpSum(sell[i] * ((1 if i in owned else 0) if t == 0 else x[i, t - 1]) for i in ids)
            cost = pulp.lpSum(buy[i] * y[i, t] for i in ids) - pulp.lpSum((buy[i] - sell[i]) * z[i, t] for i in owned)
            prob += cost <= prev_bank + value_prev
            for i in owned:
                prob += z[i, t] <= y[i, t]
                prob += z[i, t] <= (1 if t == 0 else x[i, t - 1])

        # ---- free transfers: bank up to the cap, pay for extras, keep them through WC / FH
        if t == 0:
            prob += ft[0] == (0 if st.draft else min(max(st.free_transfers, 0), rules.max_free_transfers))
        M = rules.squad_size + rules.max_free_transfers
        if first_draft:
            prob += pen[t] == 0
            prob += ft[t + 1] == 1
        elif chip[t] in ("wildcard", "freehit"):
            prob += pen[t] == 0
            prob += ft[t + 1] == ft[t]
        else:
            if st.max_transfers is not None:
                prob += nt[t] <= st.max_transfers
            unused = ft[t] - nt[t] + pen[t]
            prob += pen[t] >= nt[t] - ft[t]
            prob += pen[t] <= M * r[t]
            prob += unused <= rules.max_free_transfers * (1 - r[t])
            prob += ft[t + 1] <= unused + 1
            prob += ft[t + 1] >= unused + 1 - M * q[t]
            prob += ft[t + 1] >= rules.max_free_transfers * q[t]

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=st.time_limit, gapRel=st.gap))
    if pulp.LpStatus[status] != "Optimal":
        raise InfeasibleError(f"no feasible plan ({pulp.LpStatus[status]}): check budget, bans and forced players")

    def on(var):
        return var.value() is not None and var.value() > 0.5

    weeks = []
    for t in T:
        sq = [i for i in ids if on(squad(i, t))]
        lineup = [i for i in ids if on(s[i, t])]
        cap = next(i for i in ids if on(c[i, t]))
        vice = next(i for i in ids if on(v[i, t]))
        bench_gk = next(i for i in gks if i in sq and i not in lineup)
        subs = [next(i for i in outfield if on(b[i, t, k])) for k in range(3)]
        mult = 3 if chip[t] == "3xc" else 2
        week_xp = sum(xp[i, t] for i in lineup) + (mult - 1) * xp[cap, t]
        if chip[t] == "bboost":
            week_xp += sum(xp[i, t] for i in [bench_gk] + subs)
        weeks.append(
            WeekPlan(
                gw=gws[t],
                chip=chip[t],
                squad=sq,
                lineup=lineup,
                captain=cap,
                vice=vice,
                bench=[bench_gk] + subs,
                transfers_in=[] if st.draft and t == 0 else [i for i in ids if on(tin[i, t])],
                transfers_out=[i for i in ids if on(tout[i, t])],
                free_transfers=int(round(ft[t].value())),
                hits=int(round(pen[t].value())),
                bank=int(round(bank[t].value())),
                xp=week_xp,
            )
        )
    return Plan(weeks=weeks, objective=float(pulp.value(prob.objective)), status="Optimal")
