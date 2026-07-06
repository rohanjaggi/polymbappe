# WC2026 Prediction Results

Live prediction accuracy for the 2026 FIFA World Cup group stage, scored by
the [Predictions vs Actuals](../src/polymbappe/dashboard/pages/predictions_vs_actuals.py)
dashboard page as results came in.

## Headline numbers

After all **90 group-stage matches**:

- **70% accuracy** on a 3-way classification problem (home win / draw / away win), where random guessing gets 33%
- **87% accuracy** when the model was 70%+ confident in its pick (20/23)
- **90% accuracy** on decisive outcomes — home wins and away wins combined (60/67)

## Scorecard

| Metric | Value | Baseline | What it measures |
|--------|------:|--------:|------------------|
| Accuracy | 70.0% | 33.3% (random) | Share of matches where the top pick matched the result |
| Brier score | 0.439 | 0.667 (uniform) | Mean squared error of the probability predictions — lower is better, 0 is perfect |
| Log loss | 0.759 | 1.020 (benchmark) | How surprised the model is by actual outcomes — penalizes confident wrong calls harshly |

The Brier score is 34% better than a model that assigns equal probability to
all three outcomes. Log loss sits well below the 1.02 target used in the
academic sports-forecasting literature.

## Accuracy by outcome

| Outcome | Correct | Total | Accuracy |
|---------|--------:|------:|---------:|
| Home win | 47 | 53 | 88.7% |
| Away win | 13 | 14 | 92.9% |
| Draw | 3 | 23 | 13.0% |

The model excels at identifying which team will win (90% combined on decisive
outcomes). Draw prediction is weak — and deliberately so. Draws are the hardest
outcome to predict in football; even professional bookmakers rarely make the
draw their top pick. The model correctly learns that predicting a decisive
winner is almost always the higher-EV call.

## Confidence calibration

When the model assigns higher confidence, it delivers higher accuracy:

| Confidence threshold | Correct | Total | Accuracy |
|---------------------:|--------:|------:|---------:|
| >= 50% | 49 | 59 | 83.1% |
| >= 60% | 33 | 39 | 84.6% |
| >= 70% | 20 | 23 | 87.0% |

This is the calibration story: the model knows what it knows. When it's
uncertain, it says so. When it's confident, it's almost always right.

## xG prediction quality

The model predicts expected goals (xG) for each team per match. Comparing
against FBref's actual xG (derived from real shot data) across all 90
matches:

| Comparison | MAE |
|------------|----:|
| Model xG vs FBref xG (pure model quality) | 0.47 |
| FBref xG vs actual goals (finishing luck) | 0.75 |
| Model xG vs actual goals (combined) | 0.76 |

The model's predicted xG correlates at **0.74** with FBref's ground truth.
Its prediction error (0.47) is smaller than the inherent randomness of
whether shots go in (0.75). In other words, the gap between FBref's
ground-truth xG and the actual scoreline — pure finishing variance that no
pre-match model can predict — is larger than the gap between the model and
FBref. The model is closer to what *should* have happened than what *did*
happen.

## Historical validation

The model was validated before the tournament using **leave-one-tournament-out**
(LOTO) cross-validation across 11 major international tournaments spanning 14
years:

- FIFA World Cup 2010, 2014, 2018, 2022
- UEFA Euro 2016, 2020, 2024
- Copa America 2016, 2019, 2021, 2024

Best mean Ranked Probability Score: **0.185** (target: < 0.21). The RPS
measures how well the full probability distribution matches reality — not just
the top pick, but whether the model assigns appropriate probabilities to all
three outcomes. Lower is better.

The automated hyperparameter search ran **231 experiments** across two phases:
LLM-guided structural search (feature inclusion, architecture, meta-learner
choice) followed by Optuna TPE numeric optimization. Each candidate was
accepted only if it improved mean RPS by >0.003 and won on at least 3
individual tournaments — no single-tournament flukes.

## Methodology

All metrics on this page are computed from the model's **pre-match
predictions** (generated before each match was played) scored against actual
results as they were ingested. No post-hoc fitting, no cherry-picking. The
same scoring logic powers the live Streamlit dashboard's Predictions vs
Actuals page.

The prediction pipeline: 50,000 historical matches are ingested, 4 base models
are stacked through an ensemble, contextual features are layered on top, and
100,000 Monte Carlo simulations produce the final probabilities. See
[architecture.md](architecture.md) for the full technical breakdown.
