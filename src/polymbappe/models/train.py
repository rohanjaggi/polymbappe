"""Model training orchestration.

Fits the base models, the stacked meta-learner / dual pipelines, and (optionally) the
contextual adjuster, persisting fitted artifacts for simulation and backtesting.

Two artifacts the simulator consumes are produced:

* a Dixon-Coles model fit on *all* available history (the generative scoreline engine);
* the calibration and edge :class:`~polymbappe.models.ensemble.Ensemble` pair, fit on the
  stacked per-tournament base-probability frames (true out-of-fold stacking).

Artifacts are pickled under ``data/processed`` and indexed by name so the simulator and
reporter can load them without re-fitting.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import date

import polars as pl
import structlog

from polymbappe.config import Settings
from polymbappe.eval.backtest import DEFAULT_TOURNAMENTS, Tournament, select_fixtures
from polymbappe.eval.base_probs import BaseProbConfig, compute_tournament_base_probs
from polymbappe.models.dixon_coles import DixonColesModel
from polymbappe.models.ensemble import Ensemble, EnsembleConfig, build_dual_pipelines

logger = structlog.get_logger(__name__)

_ARTIFACT_FILES = {
    "dixon_coles": "model_dixon_coles.pkl",
    "ensemble_calibration": "ensemble_calibration.pkl",
    "ensemble_edge": "ensemble_edge.pkl",
    "contextual_adjuster": "contextual_adjuster.pkl",
}


@dataclass(slots=True)
class TrainArtifacts:
    """In-memory handles to the fitted artifacts (also persisted to disk)."""

    dixon_coles: DixonColesModel
    calibration: Ensemble
    edge: Ensemble
    adjuster: object | None = None
    stacked_frame: pl.DataFrame = field(default_factory=pl.DataFrame)


def assemble_stacked_frame(
    matches: pl.DataFrame,
    tournaments: tuple[Tournament, ...] = DEFAULT_TOURNAMENTS,
    *,
    base_config: BaseProbConfig | None = None,
    market_odds: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Concatenate per-tournament base-probability frames into one stacking matrix.

    Each tournament's base probabilities are computed from history strictly before it
    (no leakage), then stacked. Used both for fitting the ensembles and as the training
    matrix for the meta-learner.
    """

    base_config = base_config or BaseProbConfig()
    frames: list[pl.DataFrame] = []
    for tournament in tournaments:
        fixtures = select_fixtures(matches, tournament)
        if fixtures.is_empty():
            continue
        history = matches.filter(pl.col("date") < tournament.start)
        if history.is_empty():
            continue
        frames.append(
            compute_tournament_base_probs(
                history,
                fixtures,
                tournament=tournament.name,
                config=base_config,
                market_odds=market_odds,
            )
        )
    if not frames:
        raise ValueError("No tournaments with both fixtures and history were found.")
    return pl.concat(frames, how="vertical_relaxed")


def _all_history_dixon_coles(
    matches: pl.DataFrame, base_config: BaseProbConfig
) -> DixonColesModel:
    from polymbappe.eval.base_probs import matches_to_observations

    reference = matches["date"].max()
    assert isinstance(reference, date)
    obs = matches_to_observations(matches, reference)
    return DixonColesModel(base_config.dixon_coles).fit(matches=obs)


def _tournament_context_features(
    matches: pl.DataFrame, tournaments: tuple[Tournament, ...]
) -> pl.DataFrame:
    """Per-fixture contextual features (keyed by match_id) for the training tournaments.

    For each tournament, computes xG-overperformance and Elo as of its start (history
    only), then the per-fixture feature row — the same columns the simulation builds at
    prediction time (:mod:`polymbappe.context.runtime`).
    """

    from polymbappe.context.runtime import (
        SIM_CONTEXT_FEATURES,
        fixture_feature_row,
        latest_overperformance,
    )
    from polymbappe.features.elo import build_elo_snapshots

    rows: list[dict[str, object]] = []
    for tournament in tournaments:
        fixtures = select_fixtures(matches, tournament)
        if fixtures.is_empty():
            continue
        history = matches.filter(pl.col("date") < tournament.start)
        if history.is_empty():
            continue
        overperf = latest_overperformance(history)
        snaps = build_elo_snapshots(history).sort(["team", "date"]).group_by("team").agg(
            pl.col("rating").last()
        )
        elo = {r["team"]: float(r["rating"]) for r in snaps.iter_rows(named=True)}
        for fx in fixtures.iter_rows(named=True):
            feats = fixture_feature_row(fx["home_team"], fx["away_team"], overperf, elo)
            rows.append({"match_id": fx["match_id"], **feats})
    cols = {"match_id": pl.Utf8, **{c: pl.Float64 for c in SIM_CONTEXT_FEATURES}}
    return pl.DataFrame(rows, schema=cols)


def _fit_contextual_adjuster(
    frame: pl.DataFrame, calibration: Ensemble, context_features: pl.DataFrame
) -> object | None:
    """Fit the contextual adjuster on per-fixture context features vs base residuals."""

    from polymbappe.context.adjuster import ContextualAdjuster, ContextualAdjusterConfig
    from polymbappe.context.runtime import FEATURE_GROUPS

    joined = frame.join(context_features, on="match_id", how="inner")
    if joined.height < 20:  # too few labeled rows to learn a residual signal
        return None
    base_probs = calibration.predict_proba(joined)
    adjuster = ContextualAdjuster(FEATURE_GROUPS, ContextualAdjusterConfig())
    adjuster.fit(joined, base_probs)
    return adjuster


def train_full_stack(
    matches: pl.DataFrame,
    *,
    tournaments: tuple[Tournament, ...] = DEFAULT_TOURNAMENTS,
    base_config: BaseProbConfig | None = None,
    ensemble_config: EnsembleConfig | None = None,
    market_odds: pl.DataFrame | None = None,
    fit_contextual: bool = True,
) -> TrainArtifacts:
    """Fit the Dixon-Coles engine, the dual ensembles, and the contextual adjuster."""

    base_config = base_config or BaseProbConfig()
    frame = assemble_stacked_frame(
        matches, tournaments, base_config=base_config, market_odds=market_odds
    )
    has_market = all(c in frame.columns for c in ("mkt_home", "mkt_draw", "mkt_away")) and (
        int(frame.select(["mkt_home", "mkt_draw", "mkt_away"]).null_count().sum_horizontal()[0])
        == 0
    )

    cfg = ensemble_config or EnsembleConfig(
        base_groups=("dc", "elo", "mkt") if has_market else ("dc", "elo"),
        use_gbm=False,  # base-prob frame has no core GBM features by default
    )
    calibration, edge = build_dual_pipelines(cfg)
    calibration.fit(frame)
    edge.fit(frame)
    dc = _all_history_dixon_coles(matches, base_config)

    adjuster: object | None = None
    if fit_contextual:
        try:
            context_features = _tournament_context_features(matches, tournaments)
            adjuster = _fit_contextual_adjuster(frame, calibration, context_features)
        except Exception as exc:  # noqa: BLE001 - contextual layer is optional, never fatal
            logger.warning("train.context_skip", error=str(exc))

    return TrainArtifacts(
        dixon_coles=dc, calibration=calibration, edge=edge, adjuster=adjuster, stacked_frame=frame
    )


def persist_artifacts(artifacts: TrainArtifacts, settings: Settings | None = None) -> None:
    """Pickle fitted artifacts under ``data/processed``."""

    settings = settings or Settings()
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "dixon_coles": artifacts.dixon_coles,
        "ensemble_calibration": artifacts.calibration,
        "ensemble_edge": artifacts.edge,
    }
    if artifacts.adjuster is not None:
        mapping["contextual_adjuster"] = artifacts.adjuster
    for key, obj in mapping.items():
        path = settings.processed_data_dir / _ARTIFACT_FILES[key]
        with path.open("wb") as fh:
            pickle.dump(obj, fh)
        logger.info("train.persisted", artifact=key, path=str(path))


def load_artifact(name: str, settings: Settings | None = None) -> object:
    """Load a single pickled artifact by logical name."""

    settings = settings or Settings()
    path = settings.processed_data_dir / _ARTIFACT_FILES[name]
    with path.open("rb") as fh:
        return pickle.load(fh)


def train_models(model: str | None = None) -> None:
    """CLI entrypoint: fit the full stack over stored matches and persist artifacts.

    Args:
        model: Optional single model to fit (currently ``"dixon_coles"`` only); when
            ``None`` the full dual-pipeline stack is fit.
    """

    from polymbappe.data.store import read_table, table_exists
    from polymbappe.data.tables import Table

    logger.info("train.start", model=model)
    settings = Settings()
    matches = read_table(Table.MATCHES, settings)
    market_odds = (
        read_table(Table.MARKET_ODDS, settings)
        if table_exists(Table.MARKET_ODDS, settings)
        else None
    )

    if model == "dixon_coles":
        dc = _all_history_dixon_coles(matches, BaseProbConfig())
        settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
        path = settings.processed_data_dir / _ARTIFACT_FILES["dixon_coles"]
        with path.open("wb") as fh:
            pickle.dump(dc, fh)
        logger.info("train.persisted", artifact="dixon_coles", path=str(path))
        return

    artifacts = train_full_stack(matches, market_odds=market_odds)
    persist_artifacts(artifacts, settings)
    logger.info("train.done", rows=artifacts.stacked_frame.height)
