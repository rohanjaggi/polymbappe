# polymbappe

Probabilistic forecasting toolkit for the 2026 FIFA World Cup.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pre-commit install
pytest
```

Run the CLI:

```bash
polymbappe ingest
polymbappe train
polymbappe simulate --tournament 2026 --n-sims 50000
polymbappe backtest --format-version 2018
polymbappe edges --tournament 2026
```

## Kaggle credentials (player attributes)

`polymbappe ingest` pulls EA FC / FM player attributes (for the agent's
player-importance tiers) from the Kaggle dataset configured in
`data/raw/player_attributes_kaggle.txt`. Install the extra first:

```bash
pip install -e .[kaggle]
```

The default dataset is public and downloads anonymously. To authenticate —
required for private/competition datasets, and recommended to avoid anonymous
rate limits — provide a Kaggle API token (kaggle.com → Settings → API → Create
New Token) in either of these ways:

```bash
# Option A: credentials file (persistent)
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Option B: environment variables (take precedence over the file)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

See `.env.example` for the env-var form. Because pydantic-settings only reads
`POLYMBAPPE_*` keys, export the `KAGGLE_*` vars into your shell (e.g.
`set -a; . .env; set +a`) before running `polymbappe ingest`. To skip Kaggle
entirely, drop a `data/raw/player_attributes.csv` (`team,player,overall`) and the
ingester uses that instead.

## Structure

Core implemented components:
- Pydantic settings config
- Data schemas
- Elo feature builder
- Dixon-Coles MLE baseline
- 2026 group tiebreakers
- Third-place ranking (best 8 of 12)
- R32 bracket pathway seeding constraints
- Evaluation metrics

Non-trivial model/data integrations are scaffolded with typed stubs and `NotImplementedError`.
