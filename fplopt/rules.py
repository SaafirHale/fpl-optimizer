"""FPL game rules: squad shape, budget, scoring and the selling-price rule.

Defaults match the 2025/26 and 2026/27 rules. `Rules.from_bootstrap` reads the live values
from bootstrap-static (`game_config`, `element_types`) so a rule change upstream flows through.
Prices are in FPL tenths throughout (55 = GBP 5.5m).
"""

from __future__ import annotations

from dataclasses import dataclass, field

POSITIONS = ("GKP", "DEF", "MID", "FWD")


@dataclass(frozen=True)
class Scoring:
    long_play: int = 2  # 60+ minutes
    short_play: int = 1  # 1-59 minutes
    goals: dict = field(default_factory=lambda: {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4})
    assist: int = 3
    clean_sheet: dict = field(default_factory=lambda: {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0})
    goals_conceded: dict = field(default_factory=lambda: {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0})  # per 2 conceded
    saves: int = 1  # per 3 saves
    yellow: int = -1
    red: int = -3
    own_goal: int = -2
    pen_missed: int = -2
    pen_saved: int = 5
    defensive_contribution: dict = field(default_factory=lambda: {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2})
    # Defensive-contribution thresholds are not in game_config; these are the published 2025/26 rules:
    # defenders need 10 clearances/blocks/interceptions/tackles, everyone else 12 incl. recoveries.
    dc_threshold: dict = field(default_factory=lambda: {"GKP": 999, "DEF": 10, "MID": 12, "FWD": 12})


@dataclass(frozen=True)
class Rules:
    squad_size: int = 15
    squad_count: dict = field(default_factory=lambda: {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3})
    play_min: dict = field(default_factory=lambda: {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1})
    play_max: dict = field(default_factory=lambda: {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3})
    starters: int = 11
    team_limit: int = 3
    budget: int = 1000
    hit_cost: int = 4
    max_free_transfers: int = 5  # 1 + max_extra_free_transfers
    sell_on_fee: float = 0.5  # share of a price rise the manager keeps
    scoring: Scoring = field(default_factory=Scoring)

    @classmethod
    def from_bootstrap(cls, bs: dict) -> "Rules":
        cfg = bs.get("game_config", {})
        r, s = cfg.get("rules", {}), cfg.get("scoring", {})
        types = {t["singular_name_short"]: t for t in bs.get("element_types", [])}
        d = cls()
        sc = Scoring(
            long_play=s.get("long_play", d.scoring.long_play),
            short_play=s.get("short_play", d.scoring.short_play),
            goals=s.get("goals_scored", d.scoring.goals),
            assist=s.get("assists", d.scoring.assist),
            clean_sheet=s.get("clean_sheets", d.scoring.clean_sheet),
            goals_conceded=s.get("goals_conceded", d.scoring.goals_conceded),
            saves=s.get("saves", d.scoring.saves),
            yellow=s.get("yellow_cards", d.scoring.yellow),
            red=s.get("red_cards", d.scoring.red),
            own_goal=s.get("own_goals", d.scoring.own_goal),
            pen_missed=s.get("penalties_missed", d.scoring.pen_missed),
            pen_saved=s.get("penalties_saved", d.scoring.pen_saved),
            defensive_contribution=s.get("defensive_contribution", d.scoring.defensive_contribution),
        )
        return cls(
            squad_size=r.get("squad_squadsize", d.squad_size),
            squad_count={p: types[p]["squad_select"] for p in POSITIONS} if types else d.squad_count,
            play_min={p: types[p]["squad_min_play"] for p in POSITIONS} if types else d.play_min,
            play_max={p: types[p]["squad_max_play"] for p in POSITIONS} if types else d.play_max,
            starters=r.get("squad_squadplay", d.starters),
            team_limit=r.get("squad_team_limit", d.team_limit),
            budget=r.get("squad_total_spend", d.budget),
            max_free_transfers=1 + r.get("max_extra_free_transfers", d.max_free_transfers - 1),
            sell_on_fee=r.get("transfers_sell_on_fee", d.sell_on_fee),
            scoring=sc,
        )


def selling_price(purchase: int, current: int, sell_on_fee: float = 0.5) -> int:
    """FPL selling price in tenths: keep half of any rise (rounded down), absorb every fall."""
    if current <= purchase:
        return current
    return purchase + int((current - purchase) * sell_on_fee)


def valid_formation(counts: dict, rules: Rules = Rules()) -> bool:
    """True if a starting XI with these per-position counts is legal."""
    if sum(counts.get(p, 0) for p in POSITIONS) != rules.starters:
        return False
    return all(rules.play_min[p] <= counts.get(p, 0) <= rules.play_max[p] for p in POSITIONS)
