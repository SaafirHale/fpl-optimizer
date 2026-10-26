"""Projection model, scoring tables, team reconstruction and replay scoring."""

import numpy as np
import pandas as pd
import pytest

from fplopt.data import RAW, load_history, match_points
from fplopt.rules import Rules
from fplopt.simulate import score_week
from fplopt.team import replay_free_transfers

HAVE = all((RAW / s / "merged_gw.csv").exists() for s in ("2024-25", "2025-26"))
needs_data = pytest.mark.skipif(not HAVE, reason="run `fplopt fetch --history 2024-25 2025-26` first")


@pytest.fixture(scope="module")
def seasons():
    return load_history("2025-26"), load_history("2024-25")


@needs_data
@pytest.mark.parametrize("season", ["2024-25", "2025-26"])
def test_scoring_tables_reproduce_every_actual_score(season):
    s = load_history(season)
    calc = match_points(s.rows, s.rows.element.map(s.players.pos), s.scoring)
    assert (calc == s.rows.points).mean() == 1.0


@needs_data
def test_no_lookahead(seasons):
    """Rewriting everything from GW t onward must not change the projection for GW t."""
    from fplopt.backtest import project_gw

    cur, prev = seasons
    gw = 12
    a = project_gw(cur, prev, gw, horizon=2)
    noisy = cur.rows.copy()
    future = noisy.gw >= gw
    rng = np.random.default_rng(0)
    for col in ["minutes", "xg", "xa", "goals", "assists", "points", "bonus", "saves", "dc"]:
        noisy.loc[future, col] = rng.integers(0, 90, future.sum())
    cur.rows, saved = noisy, cur.rows
    try:
        b = project_gw(cur, prev, gw, horizon=2)
    finally:
        cur.rows = saved
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


@needs_data
def test_double_gameweek_sums_and_flags_zero_out(seasons):
    from fplopt.backtest import players_at, project_gw

    cur, prev = seasons
    gw = 10
    P = players_at(cur, gw)
    fx = cur.fixtures[cur.fixtures.gw == gw]
    team = int(fx.iloc[0].home_team)
    star = P[P.team == team].index[0]
    single = project_gw(cur, prev, gw, players=P)
    # give that team a second fixture in the same week
    extra = fx.iloc[[0]].assign(fixture=99999, home_team=int(fx.iloc[0].away_team), away_team=team)
    saved = cur.fixtures
    cur.fixtures = pd.concat([saved, extra])
    try:
        double = project_gw(cur, prev, gw, players=P)
    finally:
        cur.fixtures = saved
    xs = single.set_index("element").xp
    xd = double.set_index("element").xp
    assert xd[star] > xs[star] * 1.6
    # injury flag: 0% chance -> nothing next week, partial recovery the week after
    P2 = P.copy()
    P2.loc[star, "chance"] = 0.0
    hurt = project_gw(cur, prev, gw, horizon=2, players=P2).set_index(["element", "gw"]).xp
    fine = project_gw(cur, prev, gw, horizon=2, players=P).set_index(["element", "gw"]).xp
    assert hurt[(star, gw)] == 0.0
    assert 0 < hurt[(star, gw + 1)] < fine[(star, gw + 1)]
    P2.loc[star, "status"] = "u"
    gone = project_gw(cur, prev, gw, horizon=2, players=P2).set_index(["element", "gw"]).xp
    assert gone[(star, gw)] == 0.0 and gone[(star, gw + 1)] == 0.0


@needs_data
def test_projected_totals_are_in_the_range_of_reality(seasons):
    from fplopt.backtest import project_gw

    cur, prev = seasons
    pr = project_gw(cur, prev, 20)
    actual = cur.rows[cur.rows.gw == 20].points.sum()
    assert 0.8 < pr.xp.sum() / actual < 1.25


def test_free_transfer_replay():
    hist = [{"event": e, "event_transfers": t} for e, t in [(1, 0), (2, 0), (3, 0), (4, 3), (5, 0), (6, 1), (7, 10), (8, 0)]]
    # GW1 squad -> 1 FT; GW2 roll -> 2; GW3 roll -> 3; GW4 three moves -> 1; GW5 roll -> 2;
    # GW6 one move -> 2; GW7 wildcard keeps 2; GW8 roll -> 3 for GW9
    chips = [{"name": "wildcard", "event": 7}]
    assert replay_free_transfers(hist, chips, started=1) == 3
    assert replay_free_transfers(hist[:3], [], started=1) == 3
    many = [{"event": e, "event_transfers": 0} for e in range(1, 12)]
    assert replay_free_transfers(many, [], started=1) == Rules().max_free_transfers
    # joined in GW5 -> 1 for GW6; GW6 uses it -> 1; GW7 ten moves (hits) -> 1; GW8 roll -> 2
    assert replay_free_transfers(hist, [], started=5) == 2


POS = {1: "GKP", 2: "DEF", 3: "DEF", 4: "DEF", 5: "MID", 6: "MID", 7: "MID", 8: "MID", 9: "FWD", 10: "FWD",
       11: "FWD", 12: "GKP", 13: "DEF", 14: "MID", 15: "FWD"}
ORDER = list(range(1, 16))  # XI = 1..11 (3-4-3), bench = 12 (GK), 13, 14, 15


def test_autosub_replaces_a_non_playing_starter():
    pts = {i: 2 for i in ORDER}
    mins = {i: 90 for i in ORDER}
    mins[2], pts[2], pts[13] = 0, 0, 6
    total, subs = score_week(ORDER, captain=9, vice=5, pts=pts, mins=mins, pos=POS)
    assert subs == 1 and total == 10 * 2 + 6 + 2  # ten starters, the sub's 6, captain bonus


def test_vice_captain_doubles_when_captain_blanks():
    pts = {i: 2 for i in ORDER}
    mins = {i: 90 for i in ORDER}
    mins[2], pts[2], pts[13] = 0, 0, 6
    mins[9], pts[9] = 0, 0
    total, subs = score_week(ORDER, captain=9, vice=5, pts=pts, mins=mins, pos=POS)
    # DEF 13 on for DEF 2, MID 14 on for FWD 9 (3-5-2 is legal), vice 5 doubles
    assert subs == 2 and total == 9 * 2 + 6 + 2 + 2


def test_autosub_respects_formation():
    pts = {i: 2 for i in ORDER}
    mins = {i: 90 for i in ORDER}
    mins[3] = 0
    mins[13] = 0  # the bench defender blanks too: a MID or FWD for DEF 3 would leave two defenders
    _, subs = score_week(ORDER, captain=9, vice=5, pts=pts, mins=mins, pos=POS)
    assert subs == 0
