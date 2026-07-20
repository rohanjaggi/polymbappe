# FIFA World Cup 2026 — Tournament Retrospective

How the model actually did, scored against every completed match. Regenerate with `polymbappe retrospective` (trajectory via `polymbappe trajectory`).

## Headline numbers

Scored over **102 matches** (group stage through the knockout rounds):

| Metric | Value | Skill vs uniform | What it measures |
|--------|------:|-----------------:|------------------|
| Top-pick accuracy | 71.6% | vs 33.3% random | Share of matches where the model's pick matched the result |
| RPS | 0.1353 | 43.3% | Ranked probability score over ordered H/D/A — lower is better |
| Brier score | 0.443 | 33.6% | Mean squared probability error — lower is better |
| Log loss | 0.766 | 30.3% | Surprise at realized outcomes — punishes confident misses |

## Accuracy by round

| Round | Matches | Top-pick accuracy | Avg P(actual) |
|-------|--------:|------------------:|--------------:|
| Group stage | 72 | 66.7% | 51.1% |
| Round of 32 | 16 | 81.2% | 51.6% |
| Round of 16 | 8 | 75.0% | 46.5% |
| Quarter-finals | 4 | 100.0% | 51.1% |
| Semi-finals | 2 | 100.0% | 43.2% |

## The title race, replayed honestly

Each column re-simulates the tournament using only information available on that date (Dixon-Coles refit on pre-date history, played results locked, real bracket walked — no hindsight). Full daily resolution lives in `data/outputs/champion_trajectory.parquet` and on the dashboard's Tournament Retrospective page.

| Team | Pre-tournament | Mid-tournament | Pre-final | Final |
|------|---:|---:|---:|---:|
| Spain | 12.8% | 12.8% | 59.0% | 55.5% |
| Argentina | 16.2% | 30.1% | 24.6% | 44.5% |
| France | 4.2% | 9.8% | 0.0% | 0.0% |
| England | 8.9% | 5.5% | 16.4% | 0.0% |
| Brazil | 7.1% | 6.1% | 0.0% | 0.0% |
| Morocco | 5.5% | 3.1% | 0.0% | 0.0% |

## Upsets the model didn't see coming

Results given under 25% probability:

| Fixture | Score | Model pick | Actual | P(actual) |
|---------|-------|-----------|--------|----------:|
| Qatar vs Switzerland | 1 – 1 | Switzerland (90%) | Draw | 8% |
| Spain vs Cape Verde | 0 – 0 | Spain (83%) | Draw | 13% |
| Japan vs Sweden | 1 – 1 | Japan (62%) | Draw | 22% |
| Germany vs Ecuador | 1 – 2 | Germany (44%) | Ecuador | 23% |
| Iran vs New Zealand | 2 – 2 | Iran (62%) | Draw | 24% |
| England vs Ghana | 0 – 0 | England (66%) | Draw | 24% |
| Ivory Coast vs Ecuador | 1 – 0 | Draw (47%) | Ivory Coast | 25% |

## Model vs bookmaker favorites

On the 72 matches shared with the bookmaker favorite tracker:

- Model top-pick accuracy: **66.7%**
- Bookmaker favorite accuracy: **62.5%**
- McNemar's test on disagreements: p = 0.453

The workbook tracks only the shortest-odds favorite (no full 1X2 prices), so this is an accuracy comparison, not a probability-scoring one.

## Trading the champion market

_No Polymarket price history was available for the resolved champion market, so no P&L backtest was run._
