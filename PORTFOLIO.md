# Portfolio entry (draft)

- **slug:** fpl-optimizer
- **category:** quant
- **repo:** https://github.com/SaafirHale/fpl-optimizer
- **title:** FPL Optimizer
- **blurb:** Expected points for every Fantasy Premier League player, and an integer program that
  picks the squad, transfers, captain and chips under the game's rules. Walk-forward
  backtested on a hold-out season.

## Body

Fantasy Premier League is a constrained portfolio problem. You have GBP 100m, 15 slots in a
fixed shape, at most three players per club, one free transfer a week, and a 4-point charge
for each extra one. I built two pieces: a model that projects every player's points for the
next few gameweeks, and a mixed-integer program that turns those projections into decisions.
The program handles squad, starting XI, captain, bench order, free transfers that bank up to
five, selling prices that keep only half of a price rise, and the Wildcard, Free Hit, Bench
Boost and Triple Captain chips.

The projection reuses the per-90 method from my soccer value tracker, Valuo. Rates shrink
toward a prior worth 900 minutes, and last season fades out as this season accrues. The prior
depends on price within each position. Fixture strength comes from team xG for and against,
and clean sheets from a Poisson model. To explain each pick, the solver re-runs with that
player removed. The points lost after the best replacement are his value to this squad. In the plan
for GW9 of 2026/27, removing Haaland cost 9.45 points; removing Rice cost 0.06, a coin flip.

I tuned on 2024/25 and tested once on 2025/26, walk-forward with no lookahead (a test rewrites
all future data and requires an identical projection). On 29,338 player-gameweeks the model's
RMSE was 1.960, against 2.157 for recent form and 2.088 for points per match so far. Its
correlation with actual points was 0.572, against 0.499 and 0.509. Against FPL's own expected
points recorded before each deadline, it scored RMSE 1.936 vs 2.197 on the 7,452 rows where
FPL's number exists. It does not win everything: its top pick each week averaged 5.74 real
points, behind 6.26 for "highest points per match so far". The lead held on the first ten
gameweeks of 2026/27, scored as they were played: RMSE 2.112 vs 2.390 for form.

I also replayed the whole of 2025/26 through the optimizer, deadline by deadline, carrying
the squad, bank, selling prices and free transfers forward and scoring real points with
automatic substitutions. The squad finished on 2,005 points, against 1,894 for the same solver
fed recent-form projections. The transfer policy was chosen on 2024/25. It turned out
slightly worse on the hold-out season than the untuned one (2,059), and the report shows both.

One finding mattered more than any tuning. The public dataset's same-week FPL xP column looks
like the obvious benchmark, but it was recorded after each gameweek. Its correlation with
points on 2024/25 drops from 0.731 for the week it is labelled with to 0.479 for the next week. Honest
forecasts lose 0.025 to 0.059 over the same step. Benchmarking against it would have made
any model look worse than it is, so the report documents the test and uses the pre-deadline
snapshot instead.

## Stack

Python, pandas, NumPy, PuLP + CBC (mixed-integer programming), pytest, public FPL API,
vaastav/Fantasy-Premier-League per-gameweek data.
