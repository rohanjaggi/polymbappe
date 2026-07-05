# Architecture

Polymbappe is an end-to-end probabilistic forecasting system for the 2026 FIFA
World Cup. It ingests data from 6+ sources, fits a multi-model ensemble,
simulates the full 48-team tournament 100,000 times, and surfaces predictions
through a live dashboard and Polymarket edge detector.

```
Data Sources → Feature Engineering → 4 Base Models → Ensemble Stack
    → Contextual Adjuster → Dual Pipelines (Calibration + Edge)
        → Monte Carlo Simulation → Dashboard / Market Edges
```

The full model flow is diagrammed in
[model-flow.mmd](superpowers/specs/model-flow.mmd).

## Data layer

**49,500+ international matches** spanning 1872 to present, covering 336 teams.

Sources:

| Source | What it provides | Auth |
|--------|-----------------|------|
| Kaggle international results | Match results (goals, competition, venue) | None (public mirror) |
| FBref / StatsBomb | xG, PPDA, match-level expected goals | None (open data) |
| Polymarket Gamma API | Live betting market probabilities | None (public API) |
| openfootball | WC2026 schedule, groups, venues | None (public JSON) |
| Transfermarkt via Kaggle mirror | Squad market valuations | None (public dataset) |
| GeoNames | Venue coordinates for travel/fatigue features | None (CC-BY dump) |
| Football-Data.co.uk | Historical bookmaker odds for backtesting | None (public CSVs) |

The data pipeline resolves team name aliases across sources and decades of
records (e.g. "Korea Republic" vs "South Korea" vs "Republic of Korea"),
enforces point-in-time feature construction to prevent data leakage, and
stores everything in Polars-native parquet with a DuckDB state layer.

## Base models

Four independent models produce Home/Draw/Away probability estimates, each
capturing different aspects of team strength:

**Dixon-Coles (MLE)** — A bivariate Poisson model with the tau correction for
low-score correlation (the 0-0 / 1-0 / 0-1 / 1-1 adjustment that separates
this from naive independent Poissons). Fits per-team attack and defense
parameters with exponential time decay, competition-specific weighting
(friendlies down-weighted), L2 regularization, goals capping, and corrections
for AFC qualifier inflation and altitude effects.

**Bayesian Dixon-Coles (PyMC)** — A hierarchical extension where team
strengths are partially pooled toward confederation-level means via NUTS
posterior inference. Produces credible intervals on probabilities rather than
point estimates — the property the edge pipeline relies on to determine whether
a market divergence is genuine or within model uncertainty.

**Elo ratings** — Standard Elo with K-factor tuning and home advantage,
converted to 3-way probabilities. Provides a complementary view of team
strength that updates incrementally per match rather than being refit on the
full history.

**Market-implied odds** — When available, bookmaker or Polymarket prices are
ingested as a fourth probability source. Markets aggregate information the
models cannot observe (insider knowledge, public sentiment, liquidity).

## Ensemble

The base models are combined through a two-stage stacking architecture:

**LightGBM stacked model** — A 3-class gradient-boosted classifier that takes
all base model probabilities plus Tier 1 features (squad value ratio, rolling
form over 5 and 10 matches, head-to-head record, rest days) as input. It
captures non-linear interactions the Poisson framework misses. Produces
leakage-safe out-of-fold predictions via stratified K-fold so the meta-learner
never sees in-sample fits.

**Meta-learner** — Combines the base model and GBM outputs into final
calibrated H/D/A probabilities. Three calibrator families are available
(selected by the autotuner): L2-regularized multinomial logistic regression,
per-outcome isotonic calibration, or a learned convex blend of the base model
triples.

**Dual pipeline** — The same architecture runs twice: a *calibration* pipeline
(with market odds, for the dashboard and simulation) and an *edge* pipeline
(without market odds, for genuine market-vs-model divergence detection). You
can't find edges against a model that already includes the market.

## Contextual intelligence

A LightGBM residual adjuster sits on top of the calibrated ensemble and learns
what the core models systematically miss. It trains on the signed residuals
(actual outcome minus base prediction) and adds a correction vector capped at
+/-3 percentage points per outcome — bounding worst-case damage from features
that have limited historical signal.

Six feature groups, each independently toggleable with kill criteria:

| Group | Features | Signal |
|-------|----------|--------|
| xG overperformance | Rolling goals-minus-xG | Teams outperforming underlying quality |
| Draw pressure | Elo-gap interaction in group stage | Close-strength matchups draw more |
| Squad cohesion | Club clustering index, median age | How well the squad plays as a unit |
| Manager pedigree | Knockout win rate, tournament depth | Manager's tournament experience |
| Travel fatigue | Distance from team base to venue | Jet lag and travel load |
| PPDA | Passes per defensive action | Pressing intensity as a style signal |

Groups that fail their kill criterion (no RPS improvement >0.003 or
insufficient tournament coverage) are automatically disabled.

## Simulation engine

100,000 Monte Carlo runs of the full 48-team 2026 World Cup bracket:

1. **Group stage** — Each match samples a scoreline from the strength model's
   score matrix (optionally adjusted by the contextual layer). Full FIFA 2026
   tiebreakers are applied (points, goal difference, goals scored, head-to-head,
   fair play, drawing of lots).
2. **Correlated strength updates** — Each team carries a latent strength delta
   that adjusts as group results land, modelling that group form reveals true
   tournament-level ability and propagates into the knockouts.
3. **Best third-placed** — Selects 8 of 12 third-placed teams per FIFA rules.
4. **Knockout bracket** — Seeds the round of 32 with pathway constraints.
   Each tie resolves through regulation, extra time, and penalties, with an
   upset floor on lopsided R32 matchups.

Outputs: per-team stage-reaching probabilities (R32 through champion),
group-finish probabilities, and per-match H/D/A predictions with expected
goals.

## Autotuner

Two-phase automated hyperparameter and architecture search:

**Phase 1 — LLM-guided structural search**: An LLM (Qwen via Ollama, with a
deterministic fallback) proposes qualitatively different experiments —
feature inclusion, meta-learner family, training scope — that a numeric
optimizer cannot search. Each proposal is backtested and gated.

**Phase 2 — Optuna TPE**: Tree-structured Parzen Estimator optimizes the
numeric hyperparameters (Dixon-Coles decay rate, friendly weight, Elo
K-factor, GBM leaves/learning rate, meta-learner regularization) within the
structure Phase 1 selected.

Acceptance gate: a candidate is accepted only if it improves mean RPS by
>0.003 **and** wins on at least 3 individual tournaments out of the 11-tournament
LOTO validation set. 231 experiments were run.

## LangGraph agent

A 5-node state machine that monitors live news for material team changes:

**Scan** — Pulls from BBC Sport RSS and (optionally) Reddit via PRAW.
**Assess** — Classifies each finding by severity (out / doubt / minor /
non-issue) and confidence (confirmed / likely / rumor) using Ollama or a
deterministic keyword heuristic. **Cross-reference** — Deduplicates against
known player statuses and a cooling window. **Act** — Updates player status
and re-triggers simulation when a top-tier player's availability changes.
**Reflect** — Logs the probability shift and flags significant changes for the
dashboard.

## Polymarket integration

The Polymarket adapter reads live market prices from the Gamma API and
compares them against the edge pipeline's market-blind model. Edges are
flagged where the model-vs-market divergence exceeds a threshold *and* the
model's Bayesian credible interval does not overlap the market price —
filtering out cases where the model is simply uncertain. Kelly criterion
sizing produces a stake fraction for each edge.

## Dashboard

A 7-page Streamlit application:

1. **Tournament Overview** — Championship odds leaderboard, stage-reaching probabilities
2. **Team Deep Dive** — Per-team stage waterfall, group outlook, fixture list
3. **Match Predictor** — Per-fixture H/D/A probabilities with expected goals
4. **Predictions vs Actuals** — Live scoring of predictions against results (accuracy, Brier, calibration)
5. **Market Edges** — Model-vs-market divergences with Kelly sizing
6. **Upset Watch** — Matches where the model sees value against the favourite
7. **Agent Activity** — LangGraph agent changelog and player status tracker
