"""Constraint validity and optimality of the ILP on edge cases."""

import itertools
import random

import pytest

from fplopt.rules import POSITIONS, Rules, selling_price
from fplopt.solver import Candidate, InfeasibleError, Settings, solve
from tests.conftest import check_week, make_pool

QUIET = dict(bench_weights=(0, 0, 0), bench_gk_weight=0, vice_weight=0, ft_value=0, transfer_friction=0)


def owned_squad(pool, rules=Rules()):
    """A legal 15 from the pool: the cheapest players per position, max 3 per club."""
    sq, teams = [], {}
    for p in POSITIONS:
        for c in sorted((c for c in pool if c.pos == p), key=lambda c: c.buy):
            if sum(x.pos == p for x in sq) < rules.squad_count[p] and teams.get(c.team, 0) < rules.team_limit:
                sq.append(c)
                teams[c.team] = teams.get(c.team, 0) + 1
    return sq


def own(pool, squad, sell=None):
    ids = {c.id for c in squad}
    for c in pool:
        c.owned = c.id in ids
        c.sell = (sell or {}).get(c.id, c.buy) if c.owned else None
    return pool


# ---------------------------------------------------------------- draft (squad from scratch)

def test_draft_squad_obeys_every_rule(pool):
    plan = solve(pool, Settings(gws=[1], draft=True))
    by_id = {c.id: c for c in pool}
    check_week(plan.first, by_id, budget=1000)
    assert plan.first.bank == 1000 - sum(by_id[i].buy for i in plan.first.squad)


def test_draft_matches_brute_force_on_small_pool():
    """Enumerate every legal squad of a 19-player pool and its best XI + captain."""
    rng = random.Random(7)
    spec = {"GKP": 3, "DEF": 6, "MID": 6, "FWD": 4}
    pool, pid = [], 1
    for p, n in spec.items():
        for _ in range(n):
            # four strong players share club 1 so the 3-per-club limit binds
            team = 1 if pid in (4, 5, 10, 11) else pid + 1
            price = rng.randint(40, 120)
            pool.append(Candidate(pid, f"P{pid}", p, team, price, {1: round(rng.uniform(0, 10), 2)}))
            pid += 1
    by_pos = {p: [c for c in pool if c.pos == p] for p in POSITIONS}
    need = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    lo = sum(sum(sorted(c.buy for c in by_pos[p])[:n]) for p, n in need.items())
    hi = sum(sum(sorted((c.buy for c in by_pos[p]), reverse=True)[:n]) for p, n in need.items())
    budget = (lo + hi) // 2  # binding: the best squad on xp alone does not fit
    rules = Rules(budget=budget)
    feasible = 0
    best = -1.0
    for g in itertools.combinations(by_pos["GKP"], 2):
        for d in itertools.combinations(by_pos["DEF"], 5):
            for m in itertools.combinations(by_pos["MID"], 5):
                for f in itertools.combinations(by_pos["FWD"], 3):
                    sq = g + d + m + f
                    if sum(c.buy for c in sq) > budget:
                        continue
                    clubs = [c.team for c in sq]
                    if max(clubs.count(t) for t in set(clubs)) > 3:
                        continue
                    srt = {p: sorted((c.xp[1] for c in sq if c.pos == p), reverse=True) for p in POSITIONS}
                    xi = max(
                        srt["GKP"][0] + sum(srt["DEF"][:nd]) + sum(srt["MID"][:nm]) + sum(srt["FWD"][:nf])
                        for nd in range(3, 6) for nm in range(2, 6) for nf in range(1, 4) if nd + nm + nf == 10
                    )
                    feasible += 1
                    best = max(best, xi + max(c.xp[1] for c in sq))
    assert feasible > 10
    plan = solve(pool, Settings(gws=[1], draft=True, **QUIET), rules)
    check_week(plan.first, {c.id: c for c in pool}, rules, budget=budget)
    assert plan.first.xp == pytest.approx(best, abs=1e-6)


def test_club_limit_binds_when_one_club_has_all_the_stars(pool):
    for c in pool:
        if c.team == 1:
            c.xp = {1: 50.0}
    plan = solve(pool, Settings(gws=[1], draft=True))
    picked = [c for c in pool if c.id in plan.first.squad]
    assert sum(c.team == 1 for c in picked) == 3


def test_tight_budget_and_infeasible_budget(pool):
    cheapest = sum(c.buy for c in owned_squad(pool))
    plan = solve(pool, Settings(gws=[1], draft=True), Rules(budget=cheapest))
    check_week(plan.first, {c.id: c for c in pool}, budget=cheapest)
    with pytest.raises(InfeasibleError):
        solve(pool, Settings(gws=[1], draft=True), Rules(budget=cheapest - 1))


def test_formation_goes_to_the_limits(pool):
    # defenders worth a lot, midfielders and forwards nearly nothing: 5 DEF, 2 MID (min), 3 FWD? no: FWD min 1
    for c in pool:
        c.xp = {1: {"GKP": 3.0, "DEF": 6.0, "MID": 0.1, "FWD": 0.2}[c.pos] + c.buy / 1000}
    plan = solve(pool, Settings(gws=[1], draft=True))
    counts = check_week(plan.first, {c.id: c for c in pool})
    assert counts == {"GKP": 1, "DEF": 5, "MID": 2, "FWD": 3}


def test_bench_order_follows_expected_points(pool):
    plan = solve(pool, Settings(gws=[1], draft=True))
    by_id = {c.id: c for c in pool}
    subs = [by_id[i].xp[1] for i in plan.first.bench[1:]]
    assert subs == sorted(subs, reverse=True)
    starters = [by_id[i].xp[1] for i in plan.first.lineup]
    assert by_id[plan.first.captain].xp[1] == max(starters)


def test_banned_and_forced_players(pool):
    plan = solve(pool, Settings(gws=[1], draft=True))
    top = plan.first.captain
    forced = min(pool, key=lambda c: c.xp[1] if c.pos == "MID" else 99).id
    plan2 = solve(pool, Settings(gws=[1], draft=True, banned={top}, forced={forced}))
    assert top not in plan2.first.squad and forced in plan2.first.squad


# ---------------------------------------------------------------- transfers

def transfer_setup(gains, gws=(1,)):
    """Owned = cheapest legal squad with xp 1; `gains` upgrades on new clubs at the same price."""
    pool = make_pool(seed=3, gws=gws)
    for c in pool:
        c.xp = {gw: 0.0 for gw in gws}
    squad = owned_squad(pool)
    for c in squad:
        c.xp = {gw: 1.0 for gw in gws}
    own(pool, squad)
    targets = []
    outfield = [c for c in squad if c.pos == "MID"]
    for k, g in enumerate(gains):
        old = outfield[k]
        new = Candidate(9000 + k, f"Up{k}", "MID", 900 + k, old.buy, {gw: 1.0 + g for gw in gws})
        pool.append(new)
        targets.append(new.id)
    return pool, targets


def test_one_free_transfer_takes_the_upgrade():
    pool, targets = transfer_setup([10])
    plan = solve(pool, Settings(gws=[1], free_transfers=1, **QUIET))
    w = plan.first
    assert w.transfers_in == targets and len(w.transfers_out) == 1 and w.hits == 0
    check_week(w, {c.id: c for c in pool})


def test_hit_only_when_it_pays():
    pool, targets = transfer_setup([10, 6, 3])  # +6 beats a -4 hit, +3 does not
    plan = solve(pool, Settings(gws=[1], free_transfers=1, **QUIET))
    assert sorted(plan.first.transfers_in) == sorted(targets[:2])
    assert plan.first.hits == 1


def test_banked_transfers_cover_several_moves():
    pool, targets = transfer_setup([3, 3, 3])
    plan = solve(pool, Settings(gws=[1], free_transfers=3, **QUIET))
    assert sorted(plan.first.transfers_in) == sorted(targets) and plan.first.hits == 0


def test_free_transfers_roll_over_and_cap():
    pool, _ = transfer_setup([], gws=(1, 2, 3))
    plan = solve(pool, Settings(gws=[1, 2, 3], free_transfers=4, **QUIET))
    assert [w.free_transfers for w in plan.weeks] == [4, 5, 5]
    assert all(not w.transfers_in for w in plan.weeks)


def test_selling_price_limits_what_the_bank_affords():
    pool, targets = transfer_setup([10])
    old = next(c for c in pool if c.owned and c.pos == "MID")
    up = next(c for c in pool if c.id == targets[0])
    up.buy = old.buy + 3
    # bought 0.2 cheaper: the manager keeps half the 0.2 rise, so he sells for 0.1 below today's price
    old.sell = selling_price(old.buy - 2, old.buy)
    assert old.sell == old.buy - 1
    for c in pool:
        if c.owned and c.pos == "MID" and c is not old:
            c.xp = {1: 20.0}  # keep the other midfielders so `old` is the only sale
    # bank 3 would cover it at today's price (3 + buy - (buy + 3) = 0) but not at the selling price
    plan = solve(pool, Settings(gws=[1], bank=3, free_transfers=1, **QUIET))
    assert up.id not in plan.first.transfers_in
    plan = solve(pool, Settings(gws=[1], bank=4, free_transfers=1, **QUIET))
    assert plan.first.transfers_in == [up.id] and plan.first.bank == 0


def test_no_buy_back_of_a_sold_player():
    pool, _ = transfer_setup([], gws=(1, 2))
    sq = [c for c in pool if c.owned]
    sq[5].xp = {1: 0.0, 2: 30.0}  # tempting to sell in GW1 and rebuy in GW2: not allowed
    plan = solve(pool, Settings(gws=[1, 2], free_transfers=2, **QUIET))
    sold = {i for w in plan.weeks for i in w.transfers_out}
    bought = {i for w in plan.weeks for i in w.transfers_in}
    assert not (sold & bought)


def test_selling_price_rule():
    assert selling_price(50, 53) == 51
    assert selling_price(50, 55) == 52
    assert selling_price(50, 51) == 50
    assert selling_price(50, 48) == 48
    assert selling_price(50, 50) == 50


# ---------------------------------------------------------------- chips

def test_wildcard_makes_many_free_moves_and_keeps_free_transfers():
    pool, targets = transfer_setup([2, 2, 2, 2], gws=(1, 2))
    plan = solve(pool, Settings(gws=[1, 2], free_transfers=2, chips={1: "wildcard"}, **QUIET))
    w1, w2 = plan.weeks
    assert set(targets) <= set(w1.transfers_in) and w1.hits == 0
    assert w2.free_transfers == 2


def test_free_hit_squad_reverts_next_week():
    pool, _ = transfer_setup([], gws=(1, 2))
    owned_ids = {c.id for c in pool if c.owned}
    for c in pool:
        c.xp = {1: (0.0 if c.owned else 5.0), 2: (1.0 if c.owned else 0.0)}
    plan = solve(pool, Settings(gws=[1, 2], bank=600, free_transfers=1, chips={1: "freehit"}, **QUIET))
    w1, w2 = plan.weeks
    assert not (set(w1.squad) & owned_ids)  # a brand-new squad for the Free Hit week
    assert set(w2.squad) == owned_ids and not w1.transfers_in and not w2.transfers_in
    check_week(w1, {c.id: c for c in pool})


def test_free_hit_budget_uses_selling_prices():
    pool, _ = transfer_setup([], gws=(1,))
    for c in pool:
        if c.owned:
            c.sell = c.buy - 5  # every owned player is worth 0.5 less when sold
    owned_value = sum(c.sell for c in pool if c.owned)
    plan = solve(pool, Settings(gws=[1], bank=0, chips={1: "freehit"}))
    by_id = {c.id: c for c in pool}
    cost = sum(by_id[i].sell if by_id[i].owned else by_id[i].buy for i in plan.first.squad)
    assert cost <= owned_value


def test_bench_boost_scores_the_bench(pool):
    plan = solve(pool, Settings(gws=[1], draft=True, chips={1: "bboost"}))
    by_id = {c.id: c for c in pool}
    w = plan.first
    assert w.xp == pytest.approx(sum(by_id[i].xp[1] for i in w.squad) + by_id[w.captain].xp[1])


def test_triple_captain(pool):
    plan = solve(pool, Settings(gws=[1], draft=True, chips={1: "3xc"}))
    by_id = {c.id: c for c in pool}
    w = plan.first
    assert w.xp == pytest.approx(sum(by_id[i].xp[1] for i in w.lineup) + 2 * by_id[w.captain].xp[1])


def test_bad_chip_requests_are_rejected(pool):
    with pytest.raises(ValueError):
        solve(pool, Settings(gws=[1], draft=True, chips={1: "doublecaptain"}))
    with pytest.raises(ValueError):
        solve(pool, Settings(gws=[1], draft=True, chips={1: "freehit"}))
    with pytest.raises(ValueError):
        solve(pool, Settings(gws=[1], draft=True, chips={5: "wildcard"}))


def test_multi_week_plan_stays_legal_every_week():
    pool = make_pool(seed=11, gws=(1, 2, 3))
    plan = solve(pool, Settings(gws=[1, 2, 3], draft=True))
    by_id = {c.id: c for c in pool}
    for w in plan.weeks:
        check_week(w, by_id)
        assert w.bank >= 0
    assert plan.weeks[1].free_transfers == 1  # draft week gives way to 1 free transfer
