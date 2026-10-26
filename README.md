# fpl-optimizer

Expected points for every Fantasy Premier League player over the next few gameweeks, and an
integer linear program that picks the best squad, transfers, captain, bench order and chip
timing under FPL's rules. It also explains each pick by re-solving without that player.

Projections are backtested walk-forward on a hold-out season. See
[reports/BACKTEST.md](reports/BACKTEST.md).

## Results (2025/26 hold-out, 29,338 player-gameweeks)

| Projection | RMSE | Pearson r | Top pick's actual points |
|---|---:|---:|---:|
| This model | 1.960 | 0.572 | 5.74 |
| Form (last 4 gameweeks) | 2.157 | 0.499 | 4.66 |
| Points per match so far | 2.088 | 0.509 | 6.26 |

Against FPL's own expected points (the snapshot recorded before each deadline, on the 7,452
rows of 2025/26 where the dataset has one): RMSE 1.936 vs 2.197, r 0.575 vs 0.478.

**Season replay.** The optimizer played every week of 2025/26, using only data from before
each deadline, with squad, bank, selling prices and free transfers carried forward and real
points scored. It finished on 2,005 points with the model, against 1,894 for the same solver
fed form projections. It used the transfer policy chosen on 2024/25 (an extra 8-point
penalty per hit inside the objective). The untuned policy scored higher on 2025/26 (2,059
with 10 hits), and the report says so.

The model does not win everything. On 2025/26 its single top pick per gameweek averaged 5.74
real points, against 6.26 for "highest points per match so far". MAE is close for every
method (1.048 vs 1.058 and 1.075), because most player-gameweeks score 0 or 1. The report
also shows why the dataset's same-week FPL xP column can't be used as a baseline: it was
recorded after the gameweek, so it already reflects the result.

## How it works

**Projection** ([fplopt/model.py](fplopt/model.py)) adapts the per-90 method from Valuo's
valuation model:

- Per-90 xG, xA, bonus, saves and cards are shrunk toward a prior worth 900 minutes. Last
  season counts as up to half a season of evidence and fades as minutes accrue.
- The prior depends on price within each position, fitted on last season, so a GBP 10m
  forward starts from what GBP 10m forwards produced.
- Minutes: P(60+) and P(1-59) from recency-weighted appearances, scaled by FPL's
  `chance_of_playing_next_round` in live runs.
- Fixtures: opponent xG conceded and home advantage scale attacking returns. Clean sheets
  and goals conceded come from a Poisson model on own-team xG conceded and opponent attack.
- Scoring uses the live rules from `bootstrap-static` (`game_config.scoring`), including the
  2025/26 defensive-contribution points.

**Optimizer** ([fplopt/solver.py](fplopt/solver.py)) is a mixed-integer program solved with
PuLP and CBC. It covers 15 players (2/5/5/3), 3 per club, the GBP 100m budget through the
bank, selling prices (keep half of any rise), legal formations, captain and vice, an ordered
bench, free transfers banked up to 5, -4 per extra transfer, and Wildcard, Free Hit (squad
reverts), Bench Boost and Triple Captain. It plans several gameweeks at once with a decay
on later weeks.

**Why each pick won** ([fplopt/planner.py](fplopt/planner.py)): every player in the plan is
banned in turn (or, if already owned, forced out), and the solver re-optimises. The drop in
the objective is what that player is worth to this squad after the best replacement. A
player who costs 0.06 to remove was a coin flip. In the GW9 plan of 2026/27, Haaland cost 9.45: the pick the plan is
built around.

## Use

Python 3.11+. PuLP is pinned below 3 because later versions no longer ship the CBC binary.

```bash
pip install -e .[test]
```

```bash
fplopt fetch --history 2024-25 2025-26
```

```bash
fplopt project --horizon 5 --top 30
```

```bash
fplopt squad --horizon 5 --html out/squad.html
```

```bash
fplopt team 1234567 --horizon 5 --html out/team.html
```

```bash
fplopt chips 1234567 --horizon 4
```

```bash
fplopt backtest 2024-25 2025-26
```

- `squad` drafts the best 15 from scratch (a Wildcard what-if) and prints the plan for each
  week plus the "why" table.
- `team <id>` imports a public FPL team. It rebuilds purchase prices from the manager's
  transfer history and replays free transfers week by week; `--ft` overrides the count.
  Chips play as scenarios: `--chip wildcard@10`, `fh@12`, `bb@14`, `tc@11`.
- `chips` solves every available chip in every planned week and ranks the gain over the
  no-chip plan. Banked free transfers are modelled as kept, not increased, through a Wildcard
  or Free Hit week. That forgoes the week's extra transfer, so a Wildcard or Free Hit that
  changes little can show a small negative gain.

Data is cached in `data/cache/` with a time-to-live, and live requests are spaced 1.5 s
apart. A cold refresh is about ten requests: `bootstrap-static`, `fixtures`, and one
`event/<gw>/live` per gameweek played.

## Tests

```bash
pytest
```

- `tests/test_solver.py` covers every FPL rule in each planned week. The draft matches
  brute-force enumeration of a 19-player pool exactly. It also covers the 3-per-club limit
  with one club holding every star, a budget exactly at the cheapest legal squad (and one
  below it, which must fail), formations at their limits, free-transfer roll-over and the
  cap, a hit only when it pays more than 4, selling prices, no buy-back, and each chip,
  including a Free Hit squad reverting and a Free Hit budget at selling prices.
- `tests/test_model.py` rewrites every row from gameweek t onward and requires an identical
  projection for t (no lookahead). It also covers double gameweeks, injury flags, the
  scoring tables reproducing every actual 2024/25 and 2025/26 points total, the
  free-transfer replay, and automatic substitutions with the vice-captain rule.

## Layout

| Path | What |
|---|---|
| `fplopt/data.py` | history CSVs and the live API into one per player-match table |
| `fplopt/model.py` | expected points per player per gameweek |
| `fplopt/solver.py` | the integer program |
| `fplopt/planner.py` | candidate shortlist and counterfactual explanations |
| `fplopt/team.py` | team import, purchase prices, free-transfer replay, chips left |
| `fplopt/backtest.py` | walk-forward projection backtest and the leakage check |
| `fplopt/simulate.py` | season replay with the optimizer |
| `scripts/` | hyperparameter and policy tuning (validation season only), report builder |
| `reports/` | JSON results and BACKTEST.md |

## Data sources

- Live: the public FPL API (no key).
- History: per-gameweek CSVs from
  [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League), which
  hold every player's points, minutes, xG and price per gameweek. FPL's API keeps only
  season totals for past seasons.
- The method follows Valuo's valuation model. When Valuo's nightly snapshot is on the same
  machine, the team importer reads its per-gameweek prices (read-only).
  [PROPOSAL-valuo.md](PROPOSAL-valuo.md) sketches a fuller integration.
