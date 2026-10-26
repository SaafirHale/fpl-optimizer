"""Coordinate search of model hyperparameters on the validation season only (2024/25).

The test season (2025/26) is never touched here. Objective: RMSE of projected vs actual
points per player-gameweek. Results go to reports/tuning-<season>.json.

    py scripts/tune.py [season]
"""

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fplopt.backtest import metrics, run  # noqa: E402
from fplopt.model import Params  # noqa: E402

GRID = {
    "minutes_decay": [0.5, 0.6, 0.7, 0.8],
    "minutes_prior_weight": [1.0, 2.0, 3.0],
    "shrink_minutes": [450.0, 900.0, 1800.0],
    "goal_xg_weight": [0.7, 0.85, 1.0],
    "team_prior_matches": [4.0, 8.0, 16.0],
    "prev_fade_max": [0.35, 0.5, 0.75],
}


def score(p: Params, season: str) -> dict:
    m = metrics(run(season, p))["model"]
    return {"rmse": m["rmse"], "mae": m["mae"], "pearson": m["pearson"], "spearman_per_gw": m["spearman_per_gw"]}


def main(season: str = "2024-25") -> None:
    best = Params()
    log = []
    cur = score(best, season)
    log.append({"params": asdict(best), **cur})
    print("start", cur, flush=True)
    for _round in range(2):
        improved = False
        for key, values in GRID.items():
            for v in values:
                if getattr(best, key) == v:
                    continue
                cand = replace(best, **{key: v})
                s = score(cand, season)
                log.append({"params": asdict(cand), **s})
                print(key, v, {k: round(x, 4) for k, x in s.items()}, flush=True)
                if s["rmse"] < cur["rmse"] - 1e-4:
                    best, cur, improved = cand, s, True
            print("best so far", asdict(best), cur, flush=True)
        if not improved:
            break
    out = Path(__file__).resolve().parent.parent / "reports"
    out.mkdir(exist_ok=True)
    (out / f"tuning-{season}.json").write_text(json.dumps({"best": asdict(best), "best_score": cur, "log": log}, indent=2))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
