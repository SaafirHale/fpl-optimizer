import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fplopt.rules import POSITIONS, Rules  # noqa: E402
from fplopt.solver import Candidate  # noqa: E402

SHAPE = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}


def make_pool(n_teams=20, seed=1, gws=(1,), per_team=SHAPE):
    """A full-sized synthetic league: per team 2 GK, 5 DEF, 5 MID, 3 FWD with random prices/xp."""
    rng = random.Random(seed)
    base_price = {"GKP": 40, "DEF": 40, "MID": 45, "FWD": 45}
    pool = []
    pid = 1
    for team in range(1, n_teams + 1):
        for pos in POSITIONS:
            for _ in range(per_team[pos]):
                price = base_price[pos] + rng.randint(0, 80 if pos in ("MID", "FWD") else 25)
                quality = (price - base_price[pos]) / 20 + rng.random() * 2
                pool.append(Candidate(pid, f"P{pid}", pos, team, price, {gw: round(quality + rng.random(), 2) for gw in gws}))
                pid += 1
    return pool


def check_week(week, by_id, rules=Rules(), budget=None):
    """Assert every FPL rule for one planned week; returns per-position XI counts."""
    sq = week.squad
    assert len(sq) == rules.squad_size == len(set(sq))
    for p in POSITIONS:
        assert sum(by_id[i].pos == p for i in sq) == rules.squad_count[p]
    teams = {}
    for i in sq:
        teams[by_id[i].team] = teams.get(by_id[i].team, 0) + 1
    assert max(teams.values()) <= rules.team_limit
    assert len(week.lineup) == rules.starters and set(week.lineup) <= set(sq)
    counts = {p: sum(by_id[i].pos == p for i in week.lineup) for p in POSITIONS}
    for p in POSITIONS:
        assert rules.play_min[p] <= counts[p] <= rules.play_max[p]
    assert week.captain in week.lineup and week.vice in week.lineup and week.captain != week.vice
    assert len(week.bench) == 4 and len(set(week.bench)) == 4
    assert set(week.bench) == set(sq) - set(week.lineup)
    assert by_id[week.bench[0]].pos == "GKP"
    assert all(by_id[i].pos != "GKP" for i in week.bench[1:])
    if budget is not None:
        assert sum(by_id[i].buy for i in sq) <= budget
    return counts


@pytest.fixture
def pool():
    return make_pool()
