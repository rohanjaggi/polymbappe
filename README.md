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
