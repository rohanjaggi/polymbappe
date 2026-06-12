# polymbappe

Probabilistic forecasting toolkit for the 2026 FIFA World Cup: ingests
international results, ratings, squads, xG/PPDA and market odds; fits a
Dixon-Coles / GBM / Bayesian ensemble; simulates the tournament by Monte Carlo;
and compares model probabilities against live Polymarket prices to surface
edges. An optional LangGraph agent watches news/Reddit for late team-news and
re-simulates, and a Streamlit dashboard visualizes the output.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]          # core only — runs ingest/train/simulate/backtest offline
pre-commit install
pytest
```

Core CLI (works offline, no credentials, no optional extras):

```bash
polymbappe ingest                                   # results + self-computed Elo
polymbappe features                                 # build the feature matrix
polymbappe train                                    # Dixon-Coles (+ GBM/Bayesian with extras)
polymbappe simulate --tournament 2026 --n-sims 50000
polymbappe backtest --format-version 2018
polymbappe edges --tournament 2026                  # model vs market (needs odds — see below)
polymbappe report
```

The base install runs the full results→Elo→Dixon-Coles→simulate→backtest path
with zero setup. Everything else below is **opt-in**: each feature needs either
an optional dependency extra, a credential, an opt-in input file, or `--live`.

## Optional dependency extras

Each feature group is a separate extra in `pyproject.toml`. Install only what
you need, or all at once:

```bash
pip install -e .[modeling]    # GBM, Bayesian DC, meta-learner, autotuner, SHAP
pip install -e .[context]     # LangGraph live-news agent (RSS / Reddit / Ollama)
pip install -e .[dashboard]   # Streamlit + Plotly dashboard
pip install -e .[kaggle]      # EA FC / FM player-attribute ingest (kagglehub)
pip install -e .[modeling,context,dashboard,kaggle,dev]   # everything
```

| Extra | Unlocks | Key packages |
|-------|---------|--------------|
| (core) | results, Elo, Dixon-Coles, simulate, backtest, Polymarket edges, scrapers | polars, requests, duckdb, scipy, soccerdata |
| `modeling` | `train --bayesian`, GBM, meta-stacking, `autotune`, SHAP | lightgbm, pymc, scikit-learn, optuna, shap |
| `context` | `polymbappe agent` live monitoring | langgraph, praw, feedparser, vaderSentiment, ollama, apscheduler |
| `dashboard` | `polymbappe dashboard` | streamlit, plotly |
| `kaggle` | player-attribute ingest (agent importance tiers) | kagglehub |

## Configuration & credentials

Copy `.env.example` to `.env` and fill in what you need. App settings use the
`POLYMBAPPE_*` prefix and are loaded automatically by pydantic-settings; the
third-party credentials below (`KAGGLE_*`, `REDDIT_*`) are **not** read from
`.env` automatically — export them into your shell first:

```bash
set -a; . .env; set +a
```

| Variable | Used by | Required? |
|----------|---------|-----------|
| `POLYMBAPPE_RANDOM_SEED` | reproducibility | optional (default 20260611) |
| `POLYMBAPPE_DATA_DIR` | data root | optional (default `data`) |
| `POLYMBAPPE_FRIENDLY_WEIGHT` | friendly-match down-weighting | optional (default 0.3) |
| `POLYMBAPPE_DIXON_COLES_XI` | Dixon-Coles time-decay | optional (default 0.0019) |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | player-attribute ingest | optional (public dataset works anonymously) |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | agent Reddit scan | only if you enable Reddit in the agent |

### Opt-in input files (under `data/raw/`)

Several sources are enabled simply by dropping a file in `data/raw/`. They are
git-ignored except the small tracked config stubs. None are required for the
core pipeline.

| File | Enables | Format |
|------|---------|--------|
| `polymarket_query.txt` *(tracked)* | Polymarket market slug filter | one slug, e.g. `fifa-world-cup-2026` |
| `player_attributes_kaggle.txt` *(tracked)* | Kaggle dataset for player attributes | dataset slug + optional `file=` line |
| `player_attributes.csv` | local player attributes (skips Kaggle) | `team,player,overall` |
| `football_data_urls.txt` | Football-Data.co.uk bookmaker odds | one CSV URL per line (`#` comments ok) |
| `football_data/*.csv` | local Football-Data.co.uk odds | raw Football-Data CSVs |
| `squads_manifest.csv` | per-team Transfermarkt squad pages | `tournament,team[,tm_id,saison_id,url,wiki_page]` |
| `elo_url.txt` | live published Elo (EloRatings.net `World.tsv`) | a URL, or empty file to use the default |
| `elo_world.tsv` + `elo_teams.tsv` | local published Elo snapshot | EloRatings.net `World.tsv` / `en.teams.tsv` |

`configs/team_aliases.yaml` maps source-specific team spellings onto canonical
names — extend it whenever ingest/`edges` reports an unmatched team so cross-source
joins (results, squads, market odds) line up.

## Features & setup

### Live Polymarket edges
No credentials and no extra dependency required — reads the public Polymarket
Gamma API over `requests`. The market slug filter lives in
`data/raw/polymarket_query.txt` (default `fifa-world-cup-2026`).

```bash
polymbappe simulate --live                          # pull live odds during simulate
polymbappe simulate --refresh-odds                  # re-pull odds before computing edges
polymbappe edges                                    # per-match model-vs-market edges
polymbappe edges --outright --market world-cup-winner   # futures edges (champion, reach-stage)
```

Supported futures slugs (champion, reach final/SF/QF/R16/R32) are listed in
`polymbappe/polymarket/adapter.py::WORLD_CUP_FUTURES`. If a market team doesn't
join a fixture, add the spelling to `configs/team_aliases.yaml`.

### EA FC / FM player attributes (Kaggle)
Feeds the agent's player-importance tiers (not model features). Requires the
`kaggle` extra; the default dataset
(`stefanoleone992/ea-sports-fc-24-complete-player-dataset`, configured in
`data/raw/player_attributes_kaggle.txt`) is public and downloads anonymously.

```bash
pip install -e .[kaggle]
polymbappe ingest
```

A Kaggle API token is only needed for private/competition datasets or to lift
anonymous rate limits (kaggle.com → Settings → API → Create New Token):

```bash
# Option A — credentials file (persistent)
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
# Option B — env vars (take precedence; export into the shell before `ingest`)
export KAGGLE_USERNAME=your_username KAGGLE_KEY=your_key
```

To skip Kaggle entirely, drop `data/raw/player_attributes.csv` (`team,player,overall`).

### Squads & squad valuations (Transfermarkt / Wikipedia)
No credentials — uses built-in browser headers and an on-disk HTTP cache
(`data/raw/.http_cache`). Transfermarkt is anti-bot-sensitive and self-throttled.
An optional `data/raw/squads_manifest.csv` pins each team's Transfermarkt page
(`tm_id`/`url`); without it, ingest falls back to the per-tournament Wikipedia
"squads" pages. Runs as part of `polymbappe ingest`.

### Team xG & PPDA (StatsBomb Open Data)
Real xG and zonal PPDA are derived from StatsBomb's free event data (pinned
commit, no auth). This pull is heavy (~260 event files) so it only runs under
`--live`; offline runs use a local CSV if present, otherwise record zero. Note
StatsBomb open data is released **after** a tournament — a live in-tournament
xG/PPDA feed for 2026 is a documented TODO.

```bash
polymbappe ingest --live
```

### Published Elo (opt-in)
By default Elo is self-computed from match results (offline, reproducible). To
use live published EloRatings.net values instead, create `data/raw/elo_url.txt`
(empty file uses the default `World.tsv` URL) or drop local
`elo_world.tsv` + `elo_teams.tsv`.

### Bookmaker odds (Football-Data.co.uk)
Free, no auth. Add CSV URLs to `data/raw/football_data_urls.txt` (one per line)
and/or drop CSVs in `data/raw/football_data/`. Used as a market baseline in the
backtest.

### Live monitoring agent (LangGraph)
Watches news for late team-news (injuries/suspensions), classifies materiality,
and re-simulates when a top-tier player's status changes.

```bash
pip install -e .[context]
polymbappe agent --run-now        # one Scan→Assess→Cross-Ref→Act→Reflect cycle
polymbappe agent --status         # current player statuses
polymbappe agent --history        # changelog
polymbappe agent --schedule 30m   # interval scheduling
```

- **BBC Sport RSS** — no credentials (feedparser).
- **Reddit** — optional; set `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` (Reddit
  app, read-only — no username/password needed). Without them the Reddit scan is
  skipped silently.
- **Headline classification** — a deterministic keyword heuristic ships by
  default; `assess_node` accepts a pluggable classifier callable, so an LLM
  classifier can be supplied without code changes.

### Autotuner
Hyperparameter / structural search over the ensemble (needs the `modeling`
extra). The structural-experiment proposer optionally uses Ollama `qwen2.5:7b`,
falling back to a deterministic experiment list.

```bash
pip install -e .[modeling]
polymbappe autotune --budget 2h --metric rps
polymbappe autotune --leaderboard
polymbappe autotune --apply-best          # writes configs/best_config.yaml
```

### Dashboard
Six-page Streamlit app (overview, team deep-dive, match predictor, market edges,
upset watch, agent activity).

```bash
pip install -e .[dashboard]
polymbappe dashboard
```

## Project layout

- `src/polymbappe/data/` — source adapters, normalization, ingest, DuckDB store
- `src/polymbappe/features/` — Elo, xG, squad, context feature builders + pipeline
- `src/polymbappe/models/` — Dixon-Coles, GBM, Bayesian DC, ensemble, meta-stacker
- `src/polymbappe/simulate/` — Monte Carlo match & tournament simulation
- `src/polymbappe/eval/` — walk-forward backtest, market comparison, reporting
- `src/polymbappe/polymarket/` — Gamma API adapter & market alignment
- `src/polymbappe/agent/` — LangGraph live-monitoring agent
- `src/polymbappe/tune/` — autotuner (Optuna + optional LLM search)
- `src/polymbappe/dashboard/` — Streamlit pages
- `configs/` — tournament structure, team aliases, autotuner search space
- `notebooks/` — backtest report, tournament predictions, live dashboard
- `docs/superpowers/specs/` — unified spec & data-ingestion requirements

See `docs/superpowers/specs/` for the full design and per-source ingestion
requirements (auth, headers, rate limits, gotchas).
