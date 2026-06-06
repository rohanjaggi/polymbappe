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
│  EA FC (Kaggle) · FM25 · Reddit API · Google News RSS · Google Trends      │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FEATURE ENGINEERING                                    │
│                                                                              │
│  ┌─── Core Features (Tier 1-3) ───┐    ┌─── Contextual Features ────────┐  │
│  │ Elo ratings                     │    │ Tactical matchup modeling      │  │
│  │ Squad market value              │    │ Squad cohesion & chemistry     │  │
│  │ Market-implied probabilities    │    │ Manager tournament pedigree    │  │
│  │ Rolling xG                      │    │ Fatigue & schedule modeling    │  │
│  │ Recent form                     │    │ Multi-source sentiment         │  │
│  │ Head-to-head record             │    │                                │  │
│  │ Confederation strength          │    │                                │  │
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
│  Meta-Learner: Isotonic Regression (per-class calibration)                  │
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
│  Contextual features injected per-match (tactical matchups, fatigue)        │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ↓
┌────────────────────────────┐  ┌─────────────────────────────────────────────┐
│     EDGE DETECTION         │  │           LIVE UPDATE SYSTEM                 │
│  Model vs. Polymarket      │  │  LangGraph Agent (5-node state machine)     │
│  Confidence intervals      │  │  Scan → Assess → Cross-Ref → Act → Reflect │
│  Kelly criterion sizing    │  │  Every 6h pre-tournament, 2h during         │
└────────────────────────────┘  └─────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STREAMLIT DASHBOARD                                      │
│  Overview · Team Deep Dive · Match Predictor · Market Edges                 │
│  What-If Simulator · Upset Watch · Agent Activity                           │
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
| Google News RSS | Pre-tournament headline sentiment per team | RSS feed parsing + LLM classification |
| Google Trends API | Search interest volume per team | `pytrends` library |
| Expert power rankings | Aggregated professional pre-tournament rankings | Web scrape (BBC, Guardian, ESPN, FiveThirtyEight, The Athletic) |
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
- `TacticalProfile` — team, tournament, archetype, ppda, formation_variance
- `SentimentSnapshot` — team, date, reddit_score, expert_rank, line_movement, trends_volume, news_tone, composite

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
| Rolling xG (last 10 matches) | FBref | Sum player-level club xG for national team squad, weighted by minutes |
| Recent form (last 5/10/15) | Kaggle results | Goals scored/conceded rolling windows, opponent-strength-adjusted |
| Days since last match | Kaggle results | Fatigue/rust indicator |
| Head-to-head record | Kaggle results | Win rate in last 5 meetings |

**Tier 3 — Structural features:**

| Feature | Source | Construction |
|---------|--------|-------------|
| Confederation strength | Derived from Elo | Mean Elo of confederation |
| Home/host advantage | Tournament metadata | Binary: is team a host nation? |
| Tournament stage | Simulation context | Group vs. knockout indicator |
| Neutral site flag | Match metadata | Already in `Match` schema |

### 2.2 Contextual Features — Fed to Contextual Adjuster Only

**Group A — Tactical Matchup:**

| Feature | Source | Construction |
|---------|--------|-------------|
| Formation archetype (each team) | EA FC/FM25 + FBref | Classify into: high-press possession, counter-attack, direct/physical, hybrid flexible. Based on PPDA + formation frequency + player attribute profiles |
| Archetype matchup advantage | Historical results (2010+) | Win rate delta when archetype A faces B vs. Elo-predicted baseline |
| Pressing intensity index | FBref PPDA | Passes allowed per defensive action |
| Defensive compactness | EA FC (backtest) / FM25 (2026) | Aggregate of positional discipline, teamwork, concentration for starting XI |
| Style clash indicator | Derived | Binary: similar vs. opposing styles. Two possession teams → more draws historically |

**Group B — Squad Cohesion & Chemistry:**

| Feature | Source | Construction |
|---------|--------|-------------|
| Club cluster index | Transfermarkt squad lists | For each club with `n` called-up players: `n*(n-1)/2`. Sum across all clubs |
| Core continuity | Historical squad data | % of starting XI that also started in last 3 major tournaments |
| Tournament experience depth | Kaggle + squad lists | Average major tournament caps per player in expected XI |
| Age profile maturity | Transfermarkt | Squad age distribution distance from international peak (27-30) |
| Leadership spine age | Squad data | Average age of GK, CB leader, midfield anchor, captain |

**Group C — Manager Tournament Pedigree:**

| Feature | Source | Construction |
|---------|--------|-------------|
| Manager knockout win rate | Match data | Win rate in knockout-stage tournament matches (WC, Euros, Copa) |
| Manager deepest run (weighted recency) | Match data | Furthest stage reached, exponentially decayed (lambda=0.3 per tournament) |
| Manager knockout conversion rate | Match data | % of knockout matches won (group stage excluded) |
| Manager tenure matches | Match data | Competitive matches managed for current national team |
| Manager tactical flexibility | FBref | Variance in formation usage across last 10 matches |

**Group D — Fatigue & Schedule:**

| Feature | Source | Construction |
|---------|--------|-------------|
| Rest days between matches | Tournament schedule | Calendar difference. <4 days = fatigued flag |
| Travel distance between venues | Venue coordinates (16 cities) | Haversine distance between consecutive match venues |
| Club season minutes load | FBref | Total minutes played by expected XI in preceding club season |
| Match intensity accumulation | Simulation state | Cumulative tournament minutes + extra-time matches so far |
| Fixture congestion index | FBref | Matches played in final 2 months of club season |

**Group E — Multi-Source Sentiment:**

| Feature | Source | Construction |
|---------|--------|-------------|
| Reddit match sentiment | Reddit API (r/soccer) | NLP scoring (VADER + LLM reranking) on top comments from last 5-10 post-match threads |
| Expert power rankings | Web scrape (10-15 sources) | Normalize to 0-1, take median rank across BBC, Guardian, ESPN, FiveThirtyEight, The Athletic |
| Betting line movement | Polymarket + odds portals | Direction and magnitude of odds movement in 30 days pre-tournament |
| Google Trends interest | Google Trends API | Relative search volume for "[team] world cup 2026" |
| News headline sentiment | Google News RSS | Classify positive/negative/neutral using Claude Haiku, aggregate to net score |
| Composite sentiment score | Derived | Learned weighted combination: `w1*reddit + w2*expert + w3*lines + w4*trends + w5*news` |

### 2.3 Feature Builder Pattern

Each feature group gets its own builder module. All builders implement:

```python
def build(matches: pl.DataFrame, as_of_date: date) -> pl.DataFrame
```

Returns a team-date-level feature table. A `FeaturePipeline` orchestrator joins them into the final training matrix. Features computed using only data available before the prediction date (`as_of_date` enforces no leakage).

**Exception:** Tactical matchup features are match-pair level (require known opponent). These use a different interface:

```python
TacticalMatchupBuilder.compute(team_a: str, team_b: str) -> dict
```

Called inside the Monte Carlo simulation loop per match. For group stage (known opponents), pre-computed before simulation.

### 2.4 Manager Pedigree — Small Sample Mitigation

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

**Literature basis:** Baio & Blangiardo (2010), Alan Turing Institute WorldCupPrediction (6th in 2022 Futbolmetrix contest, beating FiveThirtyEight, Opta, and Betfair).

### 3.3 Base Model 3: LightGBM Stacked Model

**Input features:** All Tier 1-3 core features + outputs of Models 1 and 2 (MLE H/D/A probs, Bayesian posterior means).

**Target:** 3-class classification (home win / draw / away win).

**Hyperparameters:** `num_leaves=31`, `learning_rate=0.05`, `n_estimators=300`, `min_child_samples=20` — conservative to limit overfitting on sparse international data.

**Training:** Cross-validated with out-of-fold predictions to prevent leakage when feeding the meta-learner.

**Role:** Captures non-linear feature interactions the Poisson framework misses.

### 3.4 Meta-Learner: Stacked Ensemble

**Input:** Out-of-fold predictions from all 3 base models (9 features: 3 probabilities x 3 models).

**Method:** Isotonic regression per outcome class (non-parametric monotonic calibration).

**Output:** Calibrated H/D/A probabilities.

**Fallback:** If isotonic overfits (possible with small validation sets), fall back to logistic regression or optimized weighted average.

### 3.5 Contextual Adjustment Layer

Sits between the meta-learner output and final predictions. Trained on **residuals** between calibrated base predictions and actual outcomes. Learns only what the core model systematically misses.

**Model:** LightGBM (small: `num_leaves=15`, `n_estimators=100`)

**Target:** 3-class residual. Actual outcomes are one-hot (`[1,0,0]` for home win) minus base predicted probabilities, giving a signed error vector. The model learns: "when these contextual features are present, the base model tends to under/over-predict in this direction."

**Output:** Adjustment vector applied to base probs, softmax-normalized to valid probability simplex.

**Training:** Same leave-one-tournament-out CV protocol. For tactical matchup features (which require known opponents), only group-stage matches and historical knockout matches are used for training; in simulation, knockout matchups are computed dynamically per path.

**Toggle:** The autotuner can disable this entire layer to measure marginal RPS contribution.

**Final calibration:** An optional final isotonic pass is applied only if the adjuster degrades calibration on the validation set.

### 3.6 Edge Detection Layer

Compare final ensemble probabilities against Polymarket/bookmaker implied probabilities. Flag edges where: `|model_prob - market_prob| > 0.05` (5pp) AND the Bayesian posterior 90% credible interval doesn't overlap the market's implied probability.

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
- **Per-match contextual injection:** For each knockout match, compute tactical matchup features (TacticalMatchupBuilder) and fatigue features (ScheduleBuilder) dynamically based on the simulation path
- Draw after 90 min: extra time with adjusted expected goals (x 30/90 x 0.85 fatigue discount)
- Penalty shootout: weighted coin flip (~50.5/49.5 first-shooter advantage per Apesteguia & Palacios-Huerta 2010)
- No home advantage in knockout (neutral venue) except reduced host bonus (~0.15 goals for USA/MEX/CAN)

### 4.2 Simulation Outputs

| Output | Description |
|--------|-------------|
| Win probability per team | % of 100K sims each team wins |
| Stage-reaching probability | % reaching R32, R16, QF, SF, Final, Champion |
| Group advancement probability | % finishing 1st, 2nd, 3rd (qualifying), 3rd (eliminated), 4th |
| Expected goals per match | Mean predicted scoreline for each fixture |
| Edge report | Matches/outrights where model diverges from market |
| Contextual adjustment attribution | Per-team breakdown of how much each contextual factor shifted probabilities |

### 4.3 Live Update Behavior

After each real matchday: lock actual results, re-estimate team strengths (Bayesian posterior update), re-simulate remaining matches. CLI: `polymbappe simulate --tournament 2026 --n-sims 100000 --live`.

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
- Pulls from Google News RSS (per team keywords), Reddit r/soccer (new posts), official FIFA/confederation announcements
- Uses tool-calling to search for each of the 48 teams
- Output: list of raw news items with source, timestamp, content snippet

**Assess Node:**
- Classifies each finding via Claude API (structured output):
  - Player importance tier (1-3 based on FM25 rating for that national team)
  - Confidence level (confirmed / likely / rumor)
  - Category (injury / suspension / retirement / tactical change / squad selection)
  - Severity (out for tournament / doubt / minor / non-issue)
- Only passes forward items that are: tier 1-2 player AND confirmed/likely AND severity >= doubt

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
- **Claude API (Haiku)** — Assess and Reflect nodes (cheap, fast classification)
- **Claude API (Sonnet)** — Cross-Reference node when ambiguity is high
- **APScheduler** — scheduling triggers
- **DuckDB** — agent state persistence
- **Google News RSS + Reddit API** — data sources

---

## 6. Streamlit Dashboard

### 6.1 Pages

**Page 1: Tournament Overview**
- Trophy probability leaderboard (bar chart, sortable table)
- Group-by-group advancement probabilities (heatmap: team x finish position)
- Bracket visualization showing most likely paths to the final
- Key metrics: model RPS on backtests, last simulation timestamp, data freshness

**Page 2: Team Deep Dive**
- Team selector dropdown
- Elo trajectory over time (line chart)
- Feature radar chart: core strength, form, cohesion, fatigue, sentiment, tactical edge
- Contextual adjustments explained: "Base probability: X% → After adjustments: Y% (manager pedigree +0.8%, fatigue -0.3%, ...)"
- Stage-reaching probability waterfall
- Key players and their FM25 contribution ratings

**Page 3: Match Predictor**
- Select two teams
- H/D/A probability bars
- Score distribution heatmap (0-0 through 5-5)
- Tactical matchup analysis: "This is a [style clash type], historically favoring [team] by +X%"
- Key factors driving the prediction (SHAP-style feature importance for this specific match)
- Comparison with Polymarket odds (if available)

**Page 4: Market Edges**
- Table: match, model prob, market prob, edge magnitude, confidence interval
- Sorted by: `edge_magnitude * confidence`
- Historical edge accuracy: "In backtesting, flagged edges hit at X% rate"
- Filter by: tournament stage, edge direction

**Page 5: What-If Simulator**
- Toggle player availability (injury/fit)
- Adjust formation archetype
- See real-time impact on trophy probabilities across all 48 teams
- "Impact chain" visualization: player out → feature changes → probability shifts

**Page 6: Upset Watch**
- Matches where underdog probability is unusually high relative to Elo gap
- Historical comps: "similar Elo gaps in past WCs produced upsets X% of the time"
- Risk factors: style clash, team volatility, fatigue asymmetry

**Page 7: Agent Activity**
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
User interactions (What-If) trigger lightweight re-simulation
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
1. MLE Dixon-Coles alone
2. Elo-only logistic regression (simplest baseline)
3. Uniform prior (33/33/33) — sanity floor
4. Market odds (2018+ from Football-Data.co.uk) — ceiling to beat
5. Core ensemble without contextual layer — measures contextual layer's marginal value

### 7.4 Ablation Study

- Ensemble minus Bayesian → value of uncertainty quantification
- Ensemble minus GBM → value of non-linear interactions
- Ensemble minus market features → can we compete without market data
- Ensemble minus contextual layer → value of contextual signals
- Feature tiers: Tier 1 only → +Tier 2 → +Tier 3 → +Contextual → incremental RPS
- Contextual layer ablation: each feature group toggled independently

### 7.5 Contextual Feature Backtesting Coverage

| Feature Group | 2010 WC | 2014 WC | 2018 WC | 2022 WC | Euros 2016-24 |
|---------------|---------|---------|---------|---------|---------------|
| Tactical matchups | EA FC only | EA FC only | EA FC + FBref | EA FC + FBref | EA FC + FBref (2020+) |
| Squad cohesion | Full | Full | Full | Full | Full |
| Manager pedigree | Full | Full | Full | Full | Full |
| Fatigue/schedule | Full | Full | Full | Full | Full |
| Sentiment (Reddit) | No | No | Full | Full | No |
| Sentiment (betting lines) | Full | Full | Full | Full | Full |
| Sentiment (expert rankings) | Partial | Partial | Full | Full | Full |
| FM25-exclusive mentals | No | No | Proxy (FM25 data) | Proxy (FM25 data) | No |

The contextual adjuster is trained only on tournaments where all its input features are available (2018+). For pre-2018 backtesting, the contextual layer passes through base predictions unchanged.

**Player attribute data strategy:**
- **EA FC / FIFA (Kaggle):** Primary attribute source for backtesting. FIFA 18 data for 2018 WC, FIFA 22 for 2022 WC, etc. Covers technical, physical, and basic mental attributes with consistent schema across all editions.
- **FM25:** Supplements EA FC for 2026 live predictions with tournament-pressure attributes (ImportantMatches, Pressure, Consistency, Teamwork, Leadership, Dirtiness, Temperament). These have no EA FC equivalent.
- **FM25 as backtest proxy:** For 2018/2022 backtesting of FM-exclusive attributes, FM25 ratings are used as a proxy. Rationale: personality attributes (Pressure, Consistency, ImportantMatches) are stable across FM editions for established international players — a player rated 18/20 for ImportantMatches in FM25 was likely similar in the 2022 era. This is imperfect but acceptable for these slow-changing attributes.

### 7.6 Backtest Report

Jupyter notebook `04_backtest_report.ipynb`: per-tournament metrics, calibration plots, edge analysis, surprise analysis, contextual layer attribution.

---

## 8. Automated Hyperparameter Tuning (Autoresearch Loop)

Inspired by Karpathy's autoresearch: an autonomous experiment loop that systematically explores the model's configuration space to minimize RPS.

### 8.1 Core Loop

```
for each experiment in budget:
    1. Sample a configuration change from the search space
    2. Run full backtest pipeline (fit models → simulate tournaments → compute RPS)
    3. If RPS improved: accept change, update best config
    4. If RPS worsened: revert change
    5. Log result to leaderboard
```

Each iteration takes ~2-3 minutes. A 2-hour budget yields ~40-60 experiments; overnight (8h) yields ~160-240.

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
- `toggle_tactical`: on/off
- `toggle_cohesion`: on/off
- `toggle_manager`: on/off
- `toggle_fatigue`: on/off
- `toggle_sentiment`: on/off
- `enable_contextual_layer`: on/off (entire layer)

**Ensemble:**
- `meta_learner`: {isotonic, logistic, weighted_average}
- `base_model_subset`: all combinations of the 3 base models

### 8.3 Search Strategy

Sequential model-based optimization (SMBO) via Optuna:
- Tree-structured Parzen Estimator (TPE) for continuous/integer hyperparameters
- Random sampling for categorical choices
- Pruning: if early tournament backtests are significantly worse than baseline, skip remaining

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
│   ├── tactical.py                 # TacticalMatchupBuilder
│   ├── cohesion.py                 # Squad cohesion features
│   ├── manager.py                  # Manager pedigree features
│   ├── fatigue.py                  # Fatigue/schedule features
│   └── sentiment.py                # Multi-source sentiment aggregator
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
    │   ├── what_if.py
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
context = ["langgraph>=0.2", "langchain-anthropic>=0.3", "apscheduler>=3.10", "praw>=7.7", "pytrends>=4.9", "vaderSentiment>=3.3"]
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
| FiveThirtyEight SPI | Offensive/defensive ratings + Monte Carlo simulation methodology |

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
| LLM reasoning | Claude API (Haiku + Sonnet) |
| Scheduling | APScheduler |
| Sentiment NLP | VADER + Claude Haiku |
| Data ingestion | BeautifulSoup, soccerdata, PRAW, pytrends |
| Dashboard | Streamlit + Plotly |
| CLI | Click |
| Testing | pytest |
| Linting | ruff |
