# Proposal: bring the optimizer into Valuo

Nothing here has been changed in the Valuo repo. This file describes how the integration could
work, for whoever next works in the Valuo repo.

## What Valuo already has

- `src/lib/model/valuation.ts`: per-90 rates shrunk toward a prior (`SHRINK_MINUTES = 900`)
  and last season blended in with a fade (`blendedValue`). This optimizer reuses that method
  and points it at FPL points instead of a value index.
- `data/fpl-snapshot.json`: nightly per-gameweek minutes, xG, xA, goals, assists, clean sheets,
  goals conceded, saves, defensive contributions and price for every current player. The team
  importer here already reads it (read-only) to price players a manager has held since joining
  after GW1.
- `/fpl-advisor`: sell flags and the best-value replacement the bank can afford.
- `/squad-builder` (`src/lib/optimizer.ts`): an exact knapsack over Valuo Index for a starting XI.

## What is missing for FPL decisions

1. **Points, not value.** The snapshot has no `total_points`, `bonus`, cards, opponent or
   home/away per gameweek, so expected points cannot be computed from it alone. Add these to
   the `gw` tuple in `scripts/sync-fpl.ts`. They come from the same `element-summary` call the
   sync already makes, so it adds no requests.
2. **Fixtures.** Store the `fixtures/` endpoint (one request) in the snapshot for opponent
   strength and blank and double gameweeks.
3. **A solver that handles FPL's rules.** The knapsack in `optimizer.ts` cannot express 3 per
   club, formations, captaincy, free transfers or chips.

## Suggested shape

- **Nightly precompute, not request-time solving.** Add a step to the existing GitHub Action
  after `sync:fpl`: run this Python package (`fplopt project --horizon 5` writing JSON) and
  commit `data/fpl-projections.json` (about 700 players x 5 gameweeks with components, about
  250 KB). Pages import it at build time, as they do the snapshot. No Python on Vercel.
- **Projections page** (`/fpl-projections`): sortable table of xP per gameweek, with the
  component breakdown already produced here (goals, assists, CS, saves, bonus, DC).
- **Advisor upgrade:** `/fpl-advisor` already imports a team by ID. For transfer plans in the
  browser, use the `highs` npm package (HiGHS compiled to WebAssembly): the model in
  `fplopt/solver.py` is plain linear algebra and ports directly. A shortlist of about 150
  candidates over 3 to 5 gameweeks is small enough for the client. Alternatively, keep the
  solver in a Vercel Python function behind the existing rate limits.
- **Backtest gate:** add the 2025/26 hold-out RMSE from `reports/backtest-2025-26.json` to
  `tests/backtest-gate.test.ts` thresholds, like the value-model gate, so a model change that
  makes projections worse fails the deploy.

## Open questions for Saafir

- Python in the nightly Action (simplest) or a TypeScript port of `model.py` (one language)?
- Should projections ship on the public API v1 (`/api/v1/players/{id}` gaining `xp`)? The
  rule there is "add fields, never rename", so it would be additive.
