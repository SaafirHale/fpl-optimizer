"""Pick the transfer policy (extra hit aversion, value of a banked free transfer) by replaying
the validation season (2024/25) only. Writes reports/policy-2024-25.json.

    py scripts/tune_policy.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fplopt.simulate import run  # noqa: E402

POLICIES = [
    {"hit_margin": 0.0, "ft_value": 1.5},
    {"hit_margin": 2.0, "ft_value": 1.5},
    {"hit_margin": 4.0, "ft_value": 1.5},
    {"hit_margin": 8.0, "ft_value": 1.5},
    {"hit_margin": 4.0, "ft_value": 3.0},
]


def main(season: str = "2024-25") -> None:
    out = []
    for pol in POLICIES:
        r = run(season, "model", verbose=False, **pol)
        row = {**pol, "total": r.total, "hits": sum(w.hits for w in r.weeks), "transfers": sum(w.transfers for w in r.weeks)}
        out.append(row)
        print(row, flush=True)
    best = max(out, key=lambda r: r["total"])
    path = Path(__file__).resolve().parent.parent / "reports" / f"policy-{season}.json"
    path.write_text(json.dumps({"season": season, "best": best, "runs": out}, indent=2))
    print("best", best)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
