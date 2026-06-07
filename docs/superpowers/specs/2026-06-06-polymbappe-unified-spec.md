# Polymbappe — Unified System Architecture

A probabilistic forecasting engine for the 2026 FIFA World Cup that produces calibrated match-level and tournament-level predictions, identifies edges against prediction markets, updates live during the tournament via an autonomous monitoring agent, and surfaces everything through an interactive dashboard.

**Goals:**
1. Best-calibrated probability outputs (target RPS < 0.21, competitive with betting markets)
2. Edge detection against Polymarket/bookmaker odds using Bayesian uncertainty quantification
3. Literature-backed methodology building on Dixon-Coles (1997), Baio & Blangiardo (2010), and ensemble stacking consensus
4. Comprehensive contextual intelligence layer capturing signals quantitative models structurally miss
5. Autonomous live-updating system that reacts to squad news without manual intervention

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
│  Kaggle Results · EloRatings.net · Transfermarkt · FBref · Polymarket       │
│  EA FC (Kaggle) · FM25 · Reddit API · BBC Sport RSS                        │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FEATURE ENGINEERING                                    │
│                                                                              │
│  ┌─── Core Features (Tier 1-3) ───┐    ┌─── Contextual Features ────────┐  │
│  │ Elo ratings                     │    │ PPDA difference                │  │
│  │ Squad market value              │    │ Squad cohesion & chemistry     │  │
│  │ Market-implied probabilities    │    │ Manager tournament pedigree    │  │
│  │ Rolling xG                      │    │ Fatigue & schedule modeling    │  │
│  │ Recent form                     │    │ Multi-source sentiment         │  │
│  │ Head-to-head record             │    │                                │  │
│  │ Neutral site flag                │    │                                │  │
│  │ Home/host advantage             │    │                                │  │
│  │ Tournament stage                │    │                                │  │
│  └─────────────────────────────────┘    └────────────────────────────────┘  │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MODEL LAYER                                          │
│                                                                              │
│  Base Model 1: MLE Dixon-Coles (time decay, tau correction)                 │
│  Base Model 2: Bayesian Hierarchical Dixon-Coles (PyMC, posterior draws)    │
│  Base Model 3: LightGBM (all core features + base model outputs)            │
│       ↓ [out-of-fold H/D/A probabilities]                                   │
│  Meta-Learner: Logistic Regression (L2-regularized calibration)             │
│       ↓ [calibrated base probabilities]                                     │
│  Contextual Adjuster: LightGBM on residuals (contextual features only)      │
│       ↓ [final calibrated H/D/A probabilities]                              │
│                                                                              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TOURNAMENT SIMULATION                                     │
│  100,000 Monte Carlo iterations                                              │
│  Group stage → Best third-place → R32 → R16 → QF → SF → Final              │
│  Full FIFA 2026 tiebreakers · Extra time · Penalties                        │
│  Contextual features injected per-match (PPDA, fatigue, draw pressure)      │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ↓
┌────────────────────────────┐  ┌─────────────────────────────────────────────┐
│     EDGE DETECTION         │  │           LIVE UPDATE SYSTEM                 │
│  Separate market-blind     │  │  LangGraph Agent (5-node state machine)     │
│  pipeline vs. Polymarket   │  │  Scan → Assess → Cross-Ref → Act → Reflect │
│  Confidence intervals      │  │  Every 6h pre-tournament, 2h during         │
└────────────────────────────┘  └─────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STREAMLIT DASHBOARD                                      │
│  Overview · Team Deep Dive · Match Predictor · Market Edges                 │
│  Upset Watch · Agent Activity                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Data Layer

### 1.1 Data Sources

| Source | Provides | Ingestion Method |
|--------|----------|-----------------|
| Kaggle international results (`martj42/international_results`) | 49K+ matches 1872-2026, scores, tournament, venue | CSV download via `requests`, parse with Polars |
| EloRatings.net | Pre-computed Elo ratings per team per date | Scrape HTML with BeautifulSoup |
| Transfermarkt | Squad market values, squad lists (club affiliations, ages) for all 48 nations | Scrape with BeautifulSoup, proper headers |
| FBref / StatsBomb | Player-level xG, xAG, progressive actions, PPDA pressing stats, formation data, minutes played | Via `soccerdata` package |
| EA FC / FIFA ratings (Kaggle, `stefanoleone992`) | Player technical/physical/mental attributes across FIFA 15–FC25 (every year). ~18K players, 30+ attributes. Primary source for backtesting (consistent schema across all editions matching 2014-2024 tournament eras) | CSV download from Kaggle |
| FM25 (Football Manager) | Tournament-specific mental attributes unavailable in EA FC: ImportantMatches, Pressure, Consistency, Teamwork, Leadership, Dirtiness, Temperament, Adaptability. 159K+ players, 89 attributes. Used for 2026 live predictions only (sourced from public GitHub repo) | CSV from RXGUL/WC2026-AI-PREDICTOR |
| Polymarket + betting odds | Market-implied H/D/A probabilities, line movements | Polymarket CLOB API + Football-Data.co.uk CSVs |
| Reddit API | Post-match thread sentiment from r/soccer | PRAW library |
| BBC Sport RSS + sports news feeds | Pre-tournament headline sentiment per team | RSS feed parsing + LLM classification |
| Tournament venue data | 16 host city coordinates, schedule, venue assignments | Static JSON (FIFA published data) |
| Manager career history | Tournament knockout records, tenure data | Match database cross-referenced with Wikipedia |
| EA FC historical editions (FIFA 18, 20, 22, via Kaggle) | Player attributes for backtest periods. FM25-exclusive attributes (ImportantMatches, Pressure, etc.) use FM25 data as a proxy for 2022 era since personality attributes are stable year-to-year for established players | CSV download from Kaggle |

### 1.2 Storage

All data lands in DuckDB. Normalized into Pydantic schemas:

**Existing schemas:** `Match`, `Team`, `Player`, `Squad`

**New schemas:**
- `EloSnapshot` — team, date, rating
- `SquadValuation` — team, tournament, total_value, median_value, player_count
- `MarketOdds` — match_id, source, home_win_prob, draw_prob, away_win_prob, timestamp
- `PlayerStatus` — player, team, status (fit/doubt/injured/out), last_updated, source, confidence
- `ManagerRecord` — manager, team, tournament, stage_reached, knockout_matches, knockout_wins
- `SentimentSnapshot` — team, date, xg_overperformance, reddit_score, news_tone

### 1.3 Live Update Flow

During pre-tournament: LangGraph agent triggers `polymbappe ingest --live` every 6 hours, pulling latest news + Polymarket odds, appending to DuckDB, triggering feature re-computation and simulation re-run.

During tournament: Agent frequency increases to every 2 hours. Additionally, `polymbappe ingest --live` can be manually triggered after each matchday to lock actual results, re-estimate team strengths (Bayesian posterior update), and re-simulate remaining matches.

---

## 2. Feature Engineering

### 2.1 Core Features (Tier 1-3) — Fed to Base Models

**Tier 1 — Core strength signals:**

| Feature | Source | Construction |
|---------|--------|-------------|
| Elo rating (home & away) | EloRatings.net or self-computed | `EloRatings` class, K=60 for WC, goal-diff multiplier |
| Elo difference | Derived | `elo_home - elo_away` |
| Squad market value ratio | Transfermarkt | `log(value_home / value_away)` |
| Market-implied probabilities | Polymarket/bookmakers | Overround-removed H/D/A probabilities |

**Tier 2 — Form and context:**

| Feature | Source | Construction |
|---------|--------|-------------|
| Rolling xG (last 10 matches) | FBref | Team-level international xG from FBref (2018+). Pre-2018: opponent-strength-adjusted goals scored as proxy |
| Recent form (last 5/10/15) | Kaggle results | Goals scored/conceded rolling windows, opponent-strength-adjusted |
| Days since last match | Kaggle results | Fatigue/rust indicator |
| Head-to-head record | Kaggle results | Win rate in last 5 meetings |

**Tier 3 — Structural features:**

| Feature | Source | Construction |
|---------|--------|-------------|
| Home/host advantage | Tournament metadata | Binary: is team a host nation? |
| Tournament stage | Simulation context | Group vs. knockout indicator |
| Neutral site flag | Match metadata | Already in `Match` schema |

### 2.2 Contextual Features — Fed to Contextual Adjuster Only

**Group A — Tactical Profile:**

| Feature | Source | Construction |
|---------|--------|-------------|
| Pressing intensity (PPDA difference) | FBref | `PPDA_home - PPDA_away`. Raw pressing difference between the two teams — captures high-press vs. low-block dynamic without requiring a full archetype taxonomy. Available 2018+ |

**Group B — Squad Cohesion & Chemistry:**

| Feature | Source | Construction |
|---------|--------|-------------|
| Club cluster index | Transfermarkt squad lists | For each club with `n` called-up players: `n*(n-1)/2`. Sum across all clubs. Computable from same Transfermarkt scrape as squad value |
| Median squad age | Transfermarkt | Median age of expected XI. Simple proxy for experience vs. athleticism tradeoff. International peak is ~27-30 |

**Group C — Manager Tournament Pedigree:**

| Feature | Source | Construction |
|---------|--------|-------------|
| Manager knockout win rate | Match data | Win rate in knockout-stage tournament matches (WC, Euros, Copa) |
| Manager deepest run (weighted recency) | Match data | Furthest stage reached, exponentially decayed (lambda=0.3 per tournament) |
| Manager knockout conversion rate | Match data | % of knockout matches won (group stage excluded) |
| Manager tenure matches | Match data | Competitive matches managed for current national team |

**Group D — Fatigue & Schedule:**

| Feature | Source | Construction |
|---------|--------|-------------|
| Rest days between matches | Tournament schedule | Calendar difference. <4 days = fatigued flag |
| Travel distance between venues | Venue coordinates (16 cities) | Haversine distance between consecutive match venues |
| Club season minutes load | FBref | Total minutes played by expected XI in preceding club season. Captures "players are tired from a 60-game season" |

**Group E — Sentiment:**

| Feature | Source | Construction |
|---------|--------|-------------|
| xG overperformance (permanent) | FBref | `goals_scored - xG` over last 10 matches. The quantitative "winning ugly/well" signal. Backtestable (2018+), mechanistically clean |
| Reddit match sentiment (additive, Tier B) | Reddit API (r/soccer) | NLP scoring (VADER + LLM reranking) on top comments from last 5-10 post-match threads. Additive signal on top of xG overperformance — captures narratives that numbers miss. Known noisy due to sarcasm. Forward-test only |
| News headline sentiment (additive, Tier B) | BBC Sport RSS | Classify positive/negative/neutral using local LLM (Qwen 9B via Ollama), aggregate to net score. Forward-test only, bounded by ±3pp cap |

**Group F — Draw Pressure Indicators:**

Draws are the hardest outcome to predict (~25-28% of group matches) and systematically underpredicted by Poisson-based models. These features specifically target draw probability adjustment.

| Feature | Source | Construction |
|---------|--------|-------------|
| Mutual qualification incentive | Group standings + simulation state | Binary: would a draw qualify both teams? Historically increases draw probability by 5-8pp in final group matches |
| PPDA similarity | Derived | Two teams with similar PPDA values (both high-press or both low-block) draw more. `1 - |PPDA_home - PPDA_away| / max_PPDA_range` |
| Low-scoring match probability | Dixon-Coles output | P(total goals ≤ 1) from the scoreline matrix. Low-scoring matches have higher draw probability mechanically |
| Tournament stage × Elo gap interaction | Derived | Group stage + small Elo gap (<100) → draw more likely. Knockout stage → draw less likely (extra time resolves) |

### 2.3 Squad Selection Uncertainty

Pre-tournament predictions require computing features (tactical, cohesion, fatigue) for a starting XI that isn't known until matchday. This introduces uncertainty that must be handled explicitly.

**Expected XI inference:**
- Build from qualifying campaign + recent friendlies: most frequent starters per position (weighted by recency)
- Maintain a "likely XI" and "rotation candidates" (players who started >30% of recent matches)
- For each position, track probability of each player starting (based on start frequency)

**Feature computation:**
- Primary: compute all features using the expected XI
- Sensitivity: for each key player (top-3 by FM25 overall rating), compute features with that player removed and replaced by their backup. Report the max prediction shift as "squad uncertainty range"
- During tournament: once actual lineups are announced (~1h pre-match), recompute and re-simulate

**Impact on simulation:**
- Pre-tournament simulations use expected XI features (no uncertainty sampling — adds noise without improving calibration)
- Live simulations after squad announcement use actual XI

### 2.4 Feature Builder Pattern

Each feature group gets its own builder module. All builders implement:

```python
def build(matches: pl.DataFrame, as_of_date: date) -> pl.DataFrame
```

Returns a team-date-level feature table. A `FeaturePipeline` orchestrator joins them into the final training matrix. Features computed using only data available before the prediction date (`as_of_date` enforces no leakage).

**Match-pair features:** Draw pressure indicators (mutual qualification incentive) and PPDA difference are match-pair level (require known opponent). For group stage (known opponents), pre-computed before simulation. For knockout matches (unknown until simulated), computed dynamically per path inside the Monte Carlo loop.

### 2.5 Manager Pedigree — Small Sample Mitigation

Apply Bayesian shrinkage for managers with thin tournament records:

```
effective_rate = (n * observed_rate + prior_n * prior_rate) / (n + prior_n)
```

Where `prior_rate` is global average manager performance. Managers with 1-2 tournaments get pulled toward the mean; experienced managers' records dominate.

---

## 3. Model Architecture

### 3.1 Base Model 1: MLE Dixon-Coles (implemented)

Existing `DixonColesModel` with time decay (xi=0.0019), tau correction, competition weighting. Produces full scoreline probability matrix, H/D/A probabilities, expected goals. Fast (~seconds to fit). Captures the core generative structure of football scores.

### 3.2 Base Model 2: Bayesian Hierarchical Dixon-Coles (PyMC)

**Hierarchical structure:**
- `attack_i ~ Normal(mu_confederation[c], sigma_confederation[c])` — confederation-level priors
- `defense_i ~ Normal(mu_confederation[c], sigma_confederation[c])`
- `home_advantage ~ Normal(0.25, 0.1)` — weakly informative prior (literature value)
- `rho ~ Uniform(-0.25, 0.25)` — Dixon-Coles correlation parameter

**Time-varying strengths:** Random walk at per-match granularity: `alpha_i,t = alpha_i,t-1 + epsilon`, `epsilon ~ Normal(0, sigma_walk)`. `sigma_walk` estimated from data.

**Likelihood:** Bivariate Poisson with tau correction, parameters are latent variables with priors.

**Inference:** PyMC NUTS sampler, 2000 tuning + 2000 draws, 4 chains.

**Output:** Posterior predictive draws giving distribution over H/D/A probabilities with credible intervals. Key advantage: principled uncertainty quantification.

**Literature basis:** Baio & Blangiardo (2010), Alan Turing Institute WorldCupPrediction (6th in 2022 Futbolmetrix contest, outperforming multiple professional forecasters).

### 3.3 Base Model 3: LightGBM Stacked Model

**Input features:** All Tier 1-3 core features + outputs of Models 1 and 2 (MLE H/D/A probs, Bayesian posterior means).

**Target:** 3-class classification (home win / draw / away win).

**Hyperparameters:** `num_leaves=31`, `learning_rate=0.05`, `n_estimators=300`, `min_child_samples=20` — conservative to limit overfitting on sparse international data.

**Training:** Cross-validated with out-of-fold predictions to prevent leakage when feeding the meta-learner.

**Role:** Captures non-linear feature interactions the Poisson framework misses.

### 3.4 Meta-Learner: Stacked Ensemble

**Input:** Out-of-fold predictions from all 3 base models (9 features: 3 probabilities x 3 models).

**Default method:** Logistic regression (multinomial, L2-regularized). With only 9 features and ~300 training samples, a parametric calibrator with strong regularization is more appropriate than non-parametric methods.

**Upgrade path:** Isotonic regression per outcome class is available as an autotuner option, but only promoted if it outperforms logistic regression on held-out tournament data by >0.003 RPS. At this sample size, isotonic will likely overfit.

**Output:** Calibrated H/D/A probabilities.

**Fallback:** If both underperform, fall back to optimized weighted average (convex combination of base model probabilities, weights found via grid search).

### 3.5 Contextual Adjustment Layer

Sits between the meta-learner output and final predictions. Trained on **residuals** between calibrated base predictions and actual outcomes. Learns only what the core model systematically misses.

**Model:** LightGBM (small: `num_leaves=15`, `n_estimators=100`)

**Target:** 3-class residual. Actual outcomes are one-hot (`[1,0,0]` for home win) minus base predicted probabilities, giving a signed error vector. The model learns: "when these contextual features are present, the base model tends to under/over-predict in this direction."

**Output:** Adjustment vector applied to base probs, softmax-normalized to valid probability simplex.

**Training:** Same leave-one-tournament-out CV protocol. Match-pair features (PPDA difference, draw pressure) are computed for both group-stage (known opponents) and historical knockout matches. In simulation, knockout matchups compute these dynamically per path.

**Toggle:** The autotuner can disable this entire layer to measure marginal RPS contribution.

**Final calibration:** An optional final isotonic pass is applied only if the adjuster degrades calibration on the validation set.

### 3.6 Dual Pipeline: Calibration vs. Edge Detection

The system runs two parallel pipelines from the same base models:

**Calibration pipeline (primary — used for tournament simulation + dashboard):**
- Includes market-implied probabilities as a Tier 1 feature in LightGBM
- Optimizes for lowest RPS — best possible probability estimates regardless of source
- This is what feeds the Monte Carlo simulation and dashboard probabilities

**Edge detection pipeline (separate — used for market edge identification):**
- Excludes ALL market odds from features (both Tier 1 market-implied probabilities AND betting line movements from contextual layer)
- Trained on the same data with the same architecture, minus market inputs
- Produces "market-blind" probabilities that reflect only the model's own assessment of team strength
- Edges are flagged where: `|edge_model_prob - market_prob| > 0.05` (5pp) AND the Bayesian posterior 90% credible interval from the edge pipeline doesn't overlap the market's implied probability

**Why separate:** If market odds are a feature, the model learns to trust them (correctly — they're the strongest single predictor). But then "edges" are just noise in the model's reproduction of its own input. Genuine edges require a model that has never seen the market's opinion.

---

## 4. Tournament Simulation

### 4.1 Monte Carlo Engine (100,000 iterations)

**Group stage (72 matches per sim):**
- Sample scoreline from ensemble's predicted score matrix for every group match (6 matches/group x 12 groups)
- Scoreline sampling from Bayesian posterior predictive (propagates model uncertainty)
- Contextual adjuster applied per-match using pre-computed group-stage contextual features
- Existing `resolve_group_table` handles full FIFA 2026 tiebreakers (points, GD, GS, H2H, fair play, lots)

**Best third-place ranking:**
- Existing `third_place.py` ranks 12 third-placed teams to find best 8 qualifiers

**Knockout stage (R32 through Final):**
- Existing `seed_round_of_32` handles pathway constraints
- **Per-match contextual injection:** For each knockout match, compute PPDA difference, fatigue features (rest days, travel distance), and draw pressure dynamically based on the simulation path
- Draw after 90 min: extra time with adjusted expected goals (x 30/90 x 0.85 fatigue discount)
- Penalty shootout: team-level penalty win rate with Bayesian shrinkage toward 50% (same shrinkage approach as manager pedigree). First-shooter advantage (+0.5pp per Apesteguia & Palacios-Huerta 2010) applied on top
- No home advantage in knockout (neutral venue) except reduced host bonus for USA/MEX/CAN (tunable parameter, initialized at ~0.15 goals based on literature on home WC advantage; tri-host tournament is unprecedented so this is uncertain — autotuner can adjust)

**Correlated within-simulation updates:**
Each simulation run maintains a latent "true strength" adjustment per team that updates as group-stage results are generated. If Team A massively outperforms expectations in Match 1 (e.g., wins 4-0 when expected ~1.5 xG), their latent strength shifts upward for remaining group matches and knockout rounds within that same simulation. Implemented as a lightweight Bayesian update: `posterior_strength = prior_strength + learning_rate * (observed_GD - expected_GD)`. The learning rate is small (~0.05) to avoid overcorrecting from single-match variance. This captures the reality that group-stage results reveal information about a team's true tournament-level form, which should propagate forward through the bracket.

### 4.2 48-Team Format Adaptation

The model trains on 32-team WCs (2010-2022) but predicts a 48-team tournament with fundamentally different structure. Key differences and mitigations:

**Structural changes:**
- 12 groups of 4 (vs. 8 groups of 4): more groups, same group size. Group dynamics are similar, but best-third-place calculation is more complex (8 of 12 qualify vs. 4 of 8).
- Round of 32 is new: creates more early mismatches (group winners vs. third-place qualifiers). No historical analogue at WC level, but Euros 2016/2020/2024 (24 teams, best-third qualifying) provide some signal on mismatch dynamics.
- Wider Elo spread: expanded confederation quotas mean weaker qualifiers. Groups will have larger Elo gaps than 32-team WCs.

**Mitigations:**
- Group-stage model parameters (home advantage, draw probability baseline) are learned from ALL international tournaments (including Euros, Copa), not just 32-team WCs. This dilutes format-specific overfitting.
- Correlated within-simulation updates (Section 4.1) handle mismatch revelation naturally — if a weak team overperforms in a mismatch, their strength adjusts.
- Down-weight group-stage tactical dynamics learned from 32-team WCs: the "dead rubber" effect (teams already qualified playing less intensely) is calibrated from Euros group stages where 3rd place also qualifies.
- Knockout mismatch handling: for R32 matches with >300 Elo gap, apply a floor on underdog win probability of 8% (based on historical WC upset rate for similar gaps).

**What we cannot model:** Game-theory effects of the new format (e.g., does knowing 3rd place qualifies change team behavior?) are unobservable until the tournament happens. This is accepted uncertainty.

### 4.3 Simulation Outputs

| Output | Description |
|--------|-------------|
| Win probability per team | % of 100K sims each team wins |
| Stage-reaching probability | % reaching R32, R16, QF, SF, Final, Champion |
| Group advancement probability | % finishing 1st, 2nd, 3rd (qualifying), 3rd (eliminated), 4th |
| Expected goals per match | Mean predicted scoreline for each fixture |
| Edge report | Matches/outrights where model diverges from market |
| Contextual adjustment attribution | Per-team breakdown of how much each contextual factor shifted probabilities |

### 4.4 Live Update Behavior

After each real matchday: lock actual results, re-estimate team strengths (Bayesian posterior update), re-simulate remaining matches. CLI: `polymbappe simulate --tournament 2026 --n-sims 100000 --live`.

### 4.5 Model Staleness Detection

During the live tournament, track cumulative surprise: the running sum of `|actual_outcome - predicted_probability|` across all completed matches. If cumulative surprise exceeds a threshold (calibrated from historical tournaments — roughly: "more surprising than 90% of past matchday sequences"), the system flags that the pre-tournament model assumptions may be fundamentally wrong.

**Trigger levels:**
- **Yellow (advisory):** Cumulative surprise > 75th percentile of historical baselines. Dashboard displays a warning. No automatic action.
- **Red (intervention):** Cumulative surprise > 90th percentile. System triggers a full model re-estimation: re-fit Bayesian model using tournament results as additional observations (not just posterior updates on existing parameters, but re-running MCMC with the new data appended). This is computationally expensive (~5-10 min) so only triggered by the red threshold.

**What constitutes "surprise":** A result where the predicted probability was < 15% (e.g., Saudi Arabia beating Argentina when the model gave Saudi Arabia 8% win probability). Single upsets are expected — it's the accumulation of multiple surprises that signals model failure.

---

## 5. LangGraph Live Monitoring Agent

### 5.1 Architecture

A LangGraph state machine with 5 specialized nodes, conditional routing, and persistent state:

```
┌────────────────────────────────────────────────────────────────┐
│                  LangGraph Agent State Machine                  │
│                                                                │
│  [Scan] → [Assess] → [Cross-Reference] → [Act] → [Reflect]  │
│     ↑         │              │              │          │       │
│     │    (not material)  (already known)    │     (not sig.)  │
│     │         ↓              ↓              │          ↓       │
│     │      [Skip]         [Skip]            │      [Log Only] │
│     │                                       │                  │
│     └──────────── next cycle ───────────────┘                  │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 Nodes

**Scan Node:**
- Pulls from BBC Sport RSS, sports news feeds (per team keywords), Reddit r/soccer (new posts), official FIFA/confederation announcements
- Uses tool-calling to search for each of the 48 teams
- Output: list of raw news items with source, timestamp, content snippet

**Assess Node:**
- Classifies each finding via Qwen 9B (structured JSON output):
  - Player importance tier (1-3 based on FM25 rating for that national team)
  - Confidence level (confirmed / likely / rumor)
  - Category (injury / suspension / retirement / tactical change / squad selection)
  - Severity (out for tournament / doubt / minor / non-issue)
- Only passes forward items that are: tier 1-2 player AND confirmed/likely AND severity >= doubt

**False positive mitigation (critical for Qwen 9B reliability):**
- Confidence threshold: only act on items classified as "confirmed" by the LLM. "Likely" items are logged but require corroboration from a second source within 12h before triggering action.
- Cooling period: same player cannot be re-assessed within 12h unless a new distinct source appears (prevents flip-flopping on ambiguous cases like "rested from training")
- Impact threshold: only trigger simulation re-run if the affected player is in the expected starting XI AND their removal shifts cohesion/tactical features by >0.5 standard deviations
- Post-tournament quality metric: track agent false positive rate (acted on → turned out wrong) to calibrate confidence thresholds for future tournaments

**Cross-Reference Node:**
- Checks against agent's persistent state (DuckDB table: `agent_player_statuses`)
- Deduplicates: same player + same status = skip
- Output: only net-new material changes

**Act Node:**
- Updates `agent_player_statuses` table in DuckDB
- Triggers feature recomputation for affected team (squad value, cohesion, FM25 aggregates)
- Kicks off simulation re-run: `polymbappe simulate --tournament 2026 --n-sims 100000`
- Generates human-readable changelog entry

**Reflect Node:**
- Compares new trophy probabilities to previous run
- If any team shifted by > 0.5pp: flag as significant (dashboard notification)
- If shift < 0.5pp: log only (visible in agent activity feed)

### 5.3 State Management

Persistent state in DuckDB:
- `agent_runs` — timestamp, duration, items scanned, items acted on
- `agent_player_statuses` — player, team, status, last_updated, source, confidence
- `agent_changelog` — timestamped log of all changes with reasoning chain
- `agent_decisions` — full trace of each node's decision for dashboard transparency

### 5.4 Scheduling

- APScheduler triggers every 6 hours during pre-tournament phase
- During tournament: every 2 hours
- Manual trigger: `polymbappe agent --run-now`

### 5.5 Tech Stack

- **LangGraph** — agent orchestration, state machine, conditional routing
- **Ollama + Qwen 9B** — local LLM for all agent reasoning (Assess, Cross-Reference, Reflect nodes). Structured JSON output via Ollama's JSON mode. Zero API cost, no rate limits, works offline.
- **APScheduler** — scheduling triggers
- **DuckDB** — agent state persistence
- **BBC Sport RSS + sports news feeds + Reddit API** — data sources

---

## 6. Streamlit Dashboard

### 6.1 Pages

**Page 1: Tournament Overview**
- Trophy probability leaderboard (bar chart, sortable table)
- Group-by-group advancement probabilities (heatmap: team x finish position)
- Bracket visualization showing most likely paths to the final
- Key metrics: model RPS on backtests (with ±SE confidence interval), last simulation timestamp, data freshness

**Page 2: Team Deep Dive**
- Team selector dropdown
- Elo trajectory over time (line chart)
- Feature radar chart: core strength, form, cohesion, fatigue, xG overperformance, PPDA
- Contextual adjustments explained: "Base probability: X% → After adjustments: Y% (manager pedigree +0.8%, fatigue -0.3%, ...)"
- Stage-reaching probability waterfall

**Page 3: Match Predictor**
- Select two teams
- H/D/A probability bars
- Score distribution heatmap (0-0 through 5-5)
- Key factors driving the prediction (SHAP-style feature importance for this specific match)
- Comparison with Polymarket odds (if available)

**Page 4: Market Edges**
- Table: match, model prob, market prob, edge magnitude, confidence interval
- Sorted by: `edge_magnitude * confidence`
- Historical edge accuracy: "In backtesting, flagged edges hit at X% rate"
- Filter by: tournament stage, edge direction

**Page 5: Upset Watch**
- Matches where underdog probability is unusually high relative to Elo gap
- Historical comps: "similar Elo gaps in past WCs produced upsets X% of the time"
- Risk factors: PPDA mismatch, team volatility, fatigue asymmetry

**Page 6: Agent Activity**
- Live feed of LangGraph agent cycles
- Decision trace: scanned → assessed → acted → prediction shifted
- Node-by-node reasoning transparency (collapsible detail)
- Player status board: all monitored players with current status
- Notification log: significant shifts flagged to user

### 6.2 Data Flow

```
DuckDB (source of truth)
    ↓
Streamlit reads on page load (cached with TTL)
    ↓
Agent writes to DuckDB → Streamlit picks up changes on next refresh
```

---

## 7. Backtesting and Evaluation

### 7.1 Leave-One-Tournament-Out Validation

| Tournament | Year | Matches |
|-----------|------|---------|
| World Cup | 2010 | 64 |
| World Cup | 2014 | 64 |
| World Cup | 2018 | 64 |
| World Cup | 2022 | 64 |
| Euros | 2016, 2020, 2024 | ~51 each |
| Copa America | 2016, 2019, 2021, 2024 | ~26 each |

Protocol: Train on all international matches before each test tournament. Features computed with pre-tournament data only. Predict every match. Aggregate metrics.

### 7.2 Metrics

| Metric | Target Benchmark |
|--------|-----------------|
| Ranked Probability Score (RPS) — primary | < 0.21 |
| Brier Score | < 0.22 |
| Multiclass Log Loss | < 1.02 |
| Calibration curves | Tight diagonal fit |

### 7.3 Benchmark Comparisons

For each test tournament, compare ensemble against:
1. **Minimum viable model (MVM):** MLE Dixon-Coles + Elo + market odds + logistic calibration. Built and measured FIRST — this is the baseline everything else must beat. Expected ~0.205-0.215 RPS.
2. Elo-only logistic regression (simplest baseline)
3. Uniform prior (33/33/33) — sanity floor
4. Market odds alone (2018+ from Football-Data.co.uk) — ceiling to beat
5. Full ensemble without contextual layer — measures contextual layer's marginal value
6. Edge pipeline (market-blind) vs. calibration pipeline — measures market information value

### 7.4 Ablation Study

- Ensemble minus Bayesian → value of uncertainty quantification
- Ensemble minus GBM → value of non-linear interactions
- Ensemble minus market features → can we compete without market data
- Ensemble minus contextual layer → value of contextual signals
- Feature tiers: Tier 1 only → +Tier 2 → +Tier 3 → +Contextual → incremental RPS
- Contextual layer ablation: each feature group toggled independently

### 7.5 Complexity Kill Criteria

Every component beyond the minimum viable model must justify its existence with measurable improvement. No component ships enabled without meeting its threshold.

| Component | Kill criterion | Action if failed |
|-----------|---------------|-----------------|
| Contextual layer (entire) | Must improve mean RPS by >0.005 in leave-one-tournament-out CV | Ship disabled. Contextual features still computed and logged for post-tournament analysis, but not applied to predictions |
| Any individual feature | Must improve RPS by >0.002 when toggled on (all other features held constant) | Remove from feature set. Autotuner will not re-enable |
| Isotonic meta-learner (vs. logistic) | Must beat logistic regression by >0.003 RPS on held-out data | Keep logistic as default |
| Bayesian model (vs. MLE-only ensemble) | Must improve RPS by >0.003 vs. ensemble without it | Drop Bayesian model, run 2-model ensemble |
| Autotuner config acceptance | Must beat current best by >0.003 mean RPS AND improve on ≥3/4 individual tournament backtests | Reject config, keep current best |

**Minimum viable model (baseline to beat):** MLE Dixon-Coles + Elo + market odds + logistic calibration. This is built and measured FIRST. All subsequent components are evaluated as incremental improvements over this baseline. Expected baseline RPS: ~0.205-0.215.

### 7.6 Feature Backtesting Tiers

Features are divided into two tiers based on historical data availability:

**Tier A — Fully backtestable (autotuner operates on these):**

| Feature Group | 2010 WC | 2014 WC | 2018 WC | 2022 WC | Euros 2016-24 |
|---------------|---------|---------|---------|---------|---------------|
| Elo, form, H2H, market odds | Full | Full | Full | Full | Full |
| Squad value (Transfermarkt) | Full | Full | Full | Full | Full |
| xG + xG overperformance (FBref) | No | No | Full | Full | Full (2018+) |
| PPDA difference (FBref) | No | No | Full | Full | Full (2018+) |
| Club cluster index | Full | Full | Full | Full | Full |
| Median squad age | Full | Full | Full | Full | Full |
| Manager pedigree | Full | Full | Full | Full | Full |
| Fatigue (rest days, travel, season load) | Full | Full | Full | Full | Full |
| Draw pressure indicators | Full | Full | Full | Full | Full |

**Tier B — Forward-test only (bounded influence, validated during 2026 WC):**

| Feature | Why not backtestable | Mitigation |
|---------|---------------------|-----------|
| Reddit sentiment (additive) | Pushshift shut down 2023. Historical threads not recoverable | Runs alongside xG overperformance (which IS backtestable and permanent). Reddit adds incremental signal; if it adds noise, the model learns to zero its weight |
| News headline sentiment | BBC Sport RSS is current-only. Historical reconstruction impractical | No proxy. Forward-test only. Bounded by ±3pp cap |

### 7.7 Backtesting Strategy

**Core model (Tier A features):** Full leave-one-tournament-out backtesting on 2010-2022. The autotuner optimizes hyperparameters and feature selection using only Tier A features. This is where RPS < 0.21 is targeted.

**Contextual adjuster:** Trained on Tier A contextual features (xG overperformance, PPDA difference, club cluster, median age, manager pedigree, fatigue, draw pressure) using 2018+2022 data. These are all fully backtestable.

**Tier B features (live only, additive):** During 2026, Reddit sentiment and news sentiment are added as EXTRA features alongside the permanent xG overperformance. The contextual adjuster already learned the structure from Tier A alone; Tier B features provide incremental signal. If they're pure noise, the model's learned Tier A weights still carry the load.

**Bounded adjustment magnitude:** To prevent untested Tier B features from blowing up predictions, the contextual adjuster has a hard constraint: no single contextual feature can shift any probability by more than ±3 percentage points. This means even if Tier B features are pure noise, maximum damage is bounded.

**Post-tournament validation:** After the 2026 WC completes (~64 matches), run ablation:
- Full system (Tier A + Tier B) vs. Tier A only → did Tier B help?
- Full system vs. proxy-only version → did real signals beat their proxies?
- Per-feature attribution: which Tier B features had positive vs. negative contribution?

This is the real validation of the contextual layer. Pre-tournament, we trust the architecture based on proxy backtesting. Post-tournament, we measure whether the real signals delivered.

**Player attribute data strategy:**
- **EA FC / FIFA (Kaggle):** Not used directly as model features (aggregating player attributes into team-level features proved too abstract). Used only for player importance tiering in the LangGraph agent (which player's injury matters most).
- **FM25:** Same as EA FC — used for agent player importance classification only, not as model features. The tournament-pressure mental attributes (ImportantMatches, Pressure) inform the agent's severity assessment, not the prediction model.

### 7.8 Backtest Report

Jupyter notebook `04_backtest_report.ipynb`: per-tournament metrics, calibration plots, edge analysis, surprise analysis, contextual layer attribution, proxy vs. real feature comparison (post-tournament).

---

## 8. Automated Hyperparameter Tuning (Autoresearch Loop)

Inspired by Karpathy's autoresearch: an autonomous experiment loop that systematically explores the model's configuration space to minimize RPS.

### 8.1 Two-Phase Loop

**Phase 1 — Structural search (LLM-guided, ~10-15 experiments):**

The LLM proposes qualitatively different modeling decisions that Optuna can't search because they're not in a continuous space:
- Feature inclusion/exclusion decisions ("try removing market odds from GBM")
- Training data scope ("try weighting friendlies at 0.0 instead of 0.2")
- Architecture choices ("try 2-model ensemble without GBM", "try stacking the contextual layer differently")
- Meta-learner selection ("try weighted average instead of logistic")
- Novel feature constructions ("try Elo velocity instead of raw Elo")

Each structural experiment runs a full backtest. Results inform the *design* of the system. The LLM receives all prior results and proposes the next structural experiment based on patterns.

**Phase 2 — Numeric tuning (Optuna TPE only, 100-150 trials):**

Once the winning structure is locked from Phase 1, Optuna TPE optimizes numeric hyperparameters within the fixed architecture. No LLM in this loop — TPE handles 30-parameter numeric spaces efficiently.

**Acceptance gate (both phases):**
```
for each experiment:
    1. Run full backtest pipeline (fit → simulate → compute RPS per tournament)
    2. Compare against current best:
       - Must improve mean RPS by >0.003
       - Must improve on ≥3/4 individual tournament backtests
       - If both conditions met: accept, update best config
       - If marginal (<0.003 either direction): log as inconclusive, keep current
       - If clearly worse (>0.003 degradation): reject
    3. Top-5 configs re-evaluated 3x with different MCMC seeds to separate real gains from sampling noise
```

Each iteration takes ~2-3 minutes. Phase 1 budget: ~30-45 minutes. Phase 2 budget: 2-8 hours (100-150 trials).

### 8.2 Search Space

Defined in `configs/autotuner_search_space.yaml`:

**Dixon-Coles MLE:**
- `xi` (time decay): [0.0005, 0.005]
- `friendly_weight`: [0.1, 0.5]
- `max_goals`: {8, 10, 12}

**Bayesian model:**
- `confederation_sigma_prior`: [0.1, 1.0]
- `sigma_walk`: [0.001, 0.05]
- `home_advantage_prior_mean`: [0.1, 0.4]
- `n_tune` / `n_draws`: {1000, 2000, 4000}

**Feature engineering:**
- `elo_k_factor`: [10, 40]
- `form_window_sizes`: subsets of {3, 5, 10, 15, 20}
- `xg_window`: [5, 20]
- `include_features`: toggle each Tier 2/3 feature on/off

**GBM (base model):**
- `num_leaves`: [15, 63]
- `learning_rate`: [0.01, 0.1]
- `n_estimators`: [100, 500]
- `min_child_samples`: [10, 50]

**Contextual adjuster:**
- `context_num_leaves`: [7, 31]
- `context_n_estimators`: [50, 200]
- `toggle_ppda`: on/off
- `toggle_cohesion`: on/off (club cluster + median age)
- `toggle_manager`: on/off
- `toggle_fatigue`: on/off
- `toggle_xg_overperformance`: on/off
- `toggle_draw_pressure`: on/off
- `enable_contextual_layer`: on/off (entire layer)

**Ensemble:**
- `meta_learner`: {isotonic, logistic, weighted_average}
- `base_model_subset`: all combinations of the 3 base models

### 8.3 Search Strategy

**Phase 1 — LLM as researcher, not optimizer:**

Qwen 9B (via Ollama) receives the current system architecture, prior experiment results, and proposes the next *structural* experiment. The key insight from Karpathy's autoresearch: the LLM replaces the researcher (proposing qualitatively different directions), NOT the optimizer (sampling numbers from ranges). Structured JSON output specifies what to change and a hypothesis for why it should help.

The LLM is valuable here because structural decisions have unbounded search space — there's no YAML that enumerates "all possible feature engineering ideas." But the LLM should NOT be asked to guess that `num_leaves=23` is better than `num_leaves=31` — that's noise at this sample size, and TPE handles it better.

**Phase 2 — Optuna TPE (numeric optimization):**

Standard TPE with the search space from Section 8.2. Early pruning: if first 2 tournament backtests (2010, 2014) are >0.01 RPS worse than baseline, skip remaining tournaments.

**Repeated evaluation for top configs:** Final top-5 configs from Phase 2 are each re-run 3 times with different MCMC random seeds (Bayesian model) and different GBM random states. Only configs whose improvement survives across seeds are accepted — this separates real signal from MCMC sampling noise.

### 8.4 CLI

```
polymbappe autotune --budget 2h --metric rps
polymbappe autotune --budget 8h --metric rps --resume
polymbappe autotune --leaderboard
polymbappe autotune --apply-best
```

### 8.5 Leaderboard

Results in `data/outputs/autotune_leaderboard.parquet`: experiment ID, timestamp, config diff, RPS per tournament, mean RPS, Brier score, log loss. Best config serialized to `configs/best_config.yaml`.

---

## 9. CLI Interface

```
polymbappe ingest                          # Pull all data sources
polymbappe ingest --live                   # Pull latest results + market odds
polymbappe features --as-of 2026-06-11     # Build core feature matrix
polymbappe features --contextual           # Build contextual feature table
polymbappe train                           # Fit all base models + meta-learner + contextual adjuster
polymbappe train --model bayesian          # Fit single model
polymbappe simulate --tournament 2026 --n-sims 100000
polymbappe simulate --tournament 2026 --with-context
polymbappe simulate --tournament 2026 --live
polymbappe backtest --tournaments 2018,2022
polymbappe edges --tournament 2026
polymbappe report --tournament 2026
polymbappe autotune --budget 2h --metric rps
polymbappe autotune --leaderboard
polymbappe autotune --apply-best
polymbappe agent --run-now
polymbappe agent --status
polymbappe agent --history
polymbappe agent --schedule 6h
polymbappe dashboard
```

---

## 10. Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | EDA: match volume, Elo distributions, squad value trends |
| `02_dixon_coles_baseline.ipynb` | MLE fit, team parameters, prediction sanity checks |
| `03_bayesian_model.ipynb` | PyMC fit, trace diagnostics (R-hat, ESS, divergences), posterior predictive checks |
| `04_backtest_report.ipynb` | Full backtesting: metrics, calibration, ablation, contextual layer attribution |
| `05_tournament_predictions.ipynb` | 2026 predictions: group/bracket/win probs, market edges |
| `06_live_dashboard.ipynb` | During tournament: updated predictions, evolving strengths, edge tracking |

---

## 11. Output Artifacts

Saved to `data/outputs/` as Parquet:
- `group_probabilities.parquet` — team x group rank probabilities
- `stage_probabilities.parquet` — team x stage-reaching probabilities
- `match_predictions.parquet` — per-match H/D/A probs + expected scoreline
- `edges.parquet` — model vs. market divergences with confidence
- `simulation_log.parquet` — raw 100K simulation results
- `contextual_attribution.parquet` — per-team contextual adjustment breakdown
- `agent_changelog.parquet` — agent activity history

---

## 12. Project Structure

```
src/polymbappe/
├── __init__.py
├── cli.py                          # Click CLI entry point
├── config.py                       # Configuration management
├── data/                           # Data ingestion and storage
│   ├── __init__.py
│   ├── ingest.py                   # Data source fetchers
│   ├── schema.py                   # Pydantic schemas
│   ├── sources.py                  # Source definitions
│   └── store.py                    # DuckDB storage layer
├── features/                       # Core feature builders (Tier 1-3)
│   ├── __init__.py
│   ├── context.py                  # Match context features
│   ├── elo.py                      # Elo rating computation
│   ├── squad.py                    # Squad value features
│   └── xg.py                       # xG-based features
├── context/                        # Contextual adjustment layer
│   ├── __init__.py
│   ├── adjuster.py                 # LightGBM residual model
│   ├── ppda.py                     # PPDA difference feature
│   ├── cohesion.py                 # Club cluster + median age
│   ├── manager.py                  # Manager pedigree features
│   ├── fatigue.py                  # Rest days, travel, season load
│   ├── draw_pressure.py            # Draw incentive features
│   └── sentiment.py                # xG overperformance + Reddit + news
├── models/                         # Prediction models
│   ├── __init__.py
│   ├── base.py                     # Abstract base model
│   ├── bayesian_dc.py              # Bayesian Hierarchical Dixon-Coles (PyMC)
│   ├── dixon_coles.py              # MLE Dixon-Coles
│   └── gbm.py                      # LightGBM stacked model
├── simulate/                       # Tournament simulation engine
│   ├── __init__.py
│   ├── bracket.py                  # Knockout bracket seeding
│   ├── group.py                    # Group stage simulation
│   ├── match.py                    # Single match simulation
│   ├── third_place.py              # Best third-place ranking
│   └── tournament.py               # Full tournament Monte Carlo
├── eval/                           # Evaluation and backtesting
│   ├── __init__.py
│   ├── backtest.py                 # Leave-one-tournament-out protocol
│   ├── market.py                   # Market comparison utilities
│   └── metrics.py                  # RPS, Brier, log loss, calibration
├── polymarket/                     # Polymarket integration
│   ├── __init__.py
│   └── adapter.py                  # CLOB API client
├── agent/                          # LangGraph monitoring agent
│   ├── __init__.py
│   ├── graph.py                    # LangGraph state machine definition
│   ├── nodes.py                    # Scan, Assess, CrossRef, Act, Reflect
│   ├── sources.py                  # News/Reddit/RSS data fetchers
│   ├── state.py                    # Agent state persistence (DuckDB)
│   └── scheduler.py                # APScheduler integration
└── dashboard/                      # Streamlit app
    ├── app.py                      # Main entry point
    ├── pages/
    │   ├── overview.py
    │   ├── team_deep_dive.py
    │   ├── match_predictor.py
    │   ├── market_edges.py
    │   ├── upset_watch.py
    │   └── agent_activity.py
    └── components/                 # Shared chart/UI components

tests/
├── test_bracket.py
├── test_dixon_coles.py
├── test_group_tiebreakers.py
└── test_third_place_ranking.py

notebooks/
├── 01_data_exploration.ipynb
├── 02_dixon_coles_baseline.ipynb
├── 03_bayesian_model.ipynb
├── 04_backtest_report.ipynb
├── 05_tournament_predictions.ipynb
└── 06_live_dashboard.ipynb

configs/
├── autotuner_search_space.yaml
└── best_config.yaml

data/
├── raw/                            # Raw ingested data
├── processed/                      # Feature tables
└── outputs/                        # Simulation results, edges, reports
```

---

## 13. Dependencies

```toml
[project]
dependencies = [
    "polars>=1.0",
    "duckdb>=1.0",
    "numpy>=1.26",
    "scipy>=1.12",
    "pydantic>=2.6",
    "click>=8.1",
    "requests>=2.31",
    "beautifulsoup4>=4.12",
    "soccerdata>=1.0",
]

[project.optional-dependencies]
modeling = ["pymc>=5.17", "lightgbm>=4.5", "scikit-learn>=1.4", "optuna>=3.6", "shap>=0.45"]
context = ["langgraph>=0.2", "langchain-community>=0.3", "apscheduler>=3.10", "praw>=7.7", "vaderSentiment>=3.3", "ollama>=0.4"]
dashboard = ["streamlit>=1.38", "plotly>=5.22"]
dev = ["pytest>=8.0", "ruff>=0.3"]
```

---

## 14. Key Literature References

| Reference | Contribution |
|-----------|-------------|
| Dixon & Coles (1997) | Bivariate Poisson with tau correction and time decay — foundational match model |
| Baio & Blangiardo (2010) | Bayesian hierarchical framework for football prediction — core model structure |
| Karlis & Ntzoufras (2003) | Bivariate Poisson with explicit covariance — theoretical basis for goal correlation |
| Rue & Salvesen (2000) | "Surprise" component for upset modeling |
| Koopman & Lit (2015) | State-space Dixon-Coles with Kalman filter — inspiration for time-varying strengths |
| Groll et al. (2019) | Random forest with team ability parameters + club-level indicators for WC prediction |
| Hubacek et al. (2019) | Bookmaker odds as dominant feature in ML football models |
| Wheatcroft (2020) | Stacking Poisson + market odds for calibrated profitable predictions |
| Constantinou et al. (2019) | Dolores hybrid Bayesian network — ensemble competition winner |
| Apesteguia & Palacios-Huerta (2010) | Penalty shootout first-mover advantage |
| Goddard (2005) | Regression of football outcomes on rest days, travel distance |
| Leitner et al. (2010) | Manager quality as independent predictor beyond team strength |
| Palacios-Huerta (2014) | "Beautiful Game Theory" — tactical style interactions and outcome effects |
| Brechot & Flepp (2020) | Manager dismissal and performance — evidence manager identity matters |
| Brown et al. (2018) | Altitude effects on football performance |
| Alan Turing Institute WorldCupPrediction | Bayesian Dixon-Coles via numpyro, 6th in 2022 Futbolmetrix contest |
| FiveThirtyEight SPI (archived) | Offensive/defensive ratings + Monte Carlo simulation methodology (site shut down 2024, methodology documented) |

---

## 15. Technical Stack Summary

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Data wrangling | Polars |
| Storage | DuckDB |
| Bayesian modeling | PyMC 5.17+ |
| Gradient boosting | LightGBM 4.5+ |
| Optimization | SciPy (MLE), Optuna (autotuner) |
| Agent orchestration | LangGraph |
| LLM reasoning | Ollama + Qwen 9B (local, zero cost) |
| Scheduling | APScheduler |
| Sentiment NLP | VADER + Qwen 9B (local) |
| Data ingestion | BeautifulSoup, soccerdata, PRAW |
| Dashboard | Streamlit + Plotly |
| CLI | Click |
| Testing | pytest |
| Linting | ruff |
