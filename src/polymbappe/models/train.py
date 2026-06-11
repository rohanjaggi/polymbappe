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

import importlib.util
import pickle
from dataclasses import dataclass, field
from datetime import date

import polars as pl
import structlog

from polymbappe.config import Settings
from polymbappe.eval.backtest import DEFAULT_TOURNAMENTS, Tournament, select_fixtures
from polymbappe.eval.base_probs import BaseProbConfig, compute_tournament_base_probs
from polymbappe.models.dixon_coles import DixonColesModel
from polymbappe.models.ensemble import (
    BASE_GROUPS,
    Ensemble,
    EnsembleConfig,
    build_dual_pipelines,
)

logger = structlog.get_logger(__name__)

_ARTIFACT_FILES = {
    "dixon_coles": "model_dixon_coles.pkl",
    "bayesian": "model_bayesian.pkl",
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
    bayesian: object | None = None
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


#: Core-feature dtypes the GBM stacker accepts (all cast to Float64 before fitting).
_GBM_NUMERIC = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64, pl.Boolean,
)


def _attach_core_features(
    frame: pl.DataFrame, matches: pl.DataFrame, team_xg: pl.DataFrame | None
) -> tuple[pl.DataFrame, list[str]]:
    """Join the leakage-safe core (Tier 1-3) features onto the stacked frame by ``match_id``.

    These are the non-linear inputs the GBM stacker needs (Elo diffs, rolling form, rest,
    H2H, xG, host flags). Every core builder is point-in-time, so joining by ``match_id``
    adds no leakage. Market columns are intentionally excluded so the edge pipeline's GBM
    stays market-blind. Returns the enriched frame and the list of GBM feature columns.
    """

    from polymbappe.features.pipeline import _ID_COLUMNS, FeaturePipeline

    core = FeaturePipeline().build_core_matrix(matches, team_xg=team_xg)
    drop = set(_ID_COLUMNS) | {"home_goals", "away_goals", "label"}
    feature_cols = [
        c
        for c, dtype in zip(core.columns, core.dtypes, strict=True)
        if c not in drop and not c.endswith("_right") and dtype in _GBM_NUMERIC
    ]
    core_sel = core.select("match_id", *(pl.col(c).cast(pl.Float64) for c in feature_cols))
    return frame.join(core_sel, on="match_id", how="left"), feature_cols


def _all_history_dixon_coles(
    matches: pl.DataFrame, base_config: BaseProbConfig
) -> DixonColesModel:
    from polymbappe.eval.base_probs import matches_to_observations

    reference = matches["date"].max()
    assert isinstance(reference, date)
    obs = matches_to_observations(matches, reference)
    return DixonColesModel(base_config.dixon_coles).fit(matches=obs)


def _all_history_bayesian(matches: pl.DataFrame, base_config: BaseProbConfig) -> object:
    """Fit the Bayesian hierarchical DC model on *all* history (the simulator's CI source).

    Forward-looking (2026), not a backtest fold, so — like ``_all_history_dixon_coles`` —
    there is no leakage concern. Persisted as ``model_bayesian.pkl`` so the simulator can
    emit per-fixture credible intervals for the edge pipeline.
    """

    from polymbappe.eval.base_probs import matches_to_observations
    from polymbappe.models.bayesian_dc import BayesianDixonColesModel

    reference = matches["date"].max()
    assert isinstance(reference, date)
    obs = matches_to_observations(matches, reference)
    return BayesianDixonColesModel(base_config.bayesian).fit(matches=obs)


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
    team_xg: pl.DataFrame | None = None,
    fit_contextual: bool = True,
) -> TrainArtifacts:
    """Fit the Dixon-Coles engine, the dual ensembles, and the contextual adjuster.

    The ensembles stack a LightGBM base model over the core (Tier 1-3) features whenever
    ``lightgbm`` is installed; without it the stack degrades gracefully to the linear
    meta-learner over the base-probability groups.
    """

    base_config = base_config or BaseProbConfig()
    frame = assemble_stacked_frame(
        matches, tournaments, base_config=base_config, market_odds=market_odds
    )
    has_market = all(c in frame.columns for c in ("mkt_home", "mkt_draw", "mkt_away")) and (
        int(frame.select(["mkt_home", "mkt_draw", "mkt_away"]).null_count().sum_horizontal()[0])
        == 0
    )
    has_bay = (
        base_config.use_bayesian
        and all(c in frame.columns for c in BASE_GROUPS["bay"])
        and int(frame.select(list(BASE_GROUPS["bay"])).null_count().sum_horizontal()[0]) == 0
    )

    # Attach core features and stack a GBM over them + the base-probability groups. The
    # edge pipeline drops market columns (see ``Ensemble._gbm_columns``) for a true
    # market-blind GBM. Falls back to the linear stack if lightgbm is unavailable.
    frame, core_cols = _attach_core_features(frame, matches, team_xg)
    base_groups = tuple(
        g
        for g, present in (("dc", True), ("bay", has_bay), ("elo", True), ("mkt", has_market))
        if present
    )
    gbm_cols = [c for g in base_groups for c in BASE_GROUPS[g] if c in frame.columns] + core_cols
    gbm_available = importlib.util.find_spec("lightgbm") is not None
    if core_cols and not gbm_available:
        logger.warning("train.gbm_skip", reason="lightgbm not installed")

    cfg = ensemble_config or EnsembleConfig(
        base_groups=base_groups,
        use_gbm=gbm_available and bool(gbm_cols),
    )
    calibration, edge = build_dual_pipelines(cfg, gbm_feature_columns=gbm_cols)
    calibration.fit(frame)
    edge.fit(frame)
    dc = _all_history_dixon_coles(matches, base_config)
    bayesian = _all_history_bayesian(matches, base_config) if base_config.use_bayesian else None

    adjuster: object | None = None
    if fit_contextual:
        try:
            from polymbappe.context.runtime import build_tournament_context_features

            context_features = build_tournament_context_features(matches, tournaments)
            adjuster = _fit_contextual_adjuster(frame, calibration, context_features)
        except Exception as exc:  # noqa: BLE001 - contextual layer is optional, never fatal
            logger.warning("train.context_skip", error=str(exc))

    return TrainArtifacts(
        dixon_coles=dc,
        calibration=calibration,
        edge=edge,
        adjuster=adjuster,
        bayesian=bayesian,
        stacked_frame=frame,
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
    if artifacts.bayesian is not None:
        mapping["bayesian"] = artifacts.bayesian
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


def train_models(model: str | None = None, *, bayesian: bool = False) -> None:
    """CLI entrypoint: fit the full stack over stored matches and persist artifacts.

    Args:
        model: Optional single model to fit (currently ``"dixon_coles"`` only); when
            ``None`` the full dual-pipeline stack is fit.
        bayesian: Opt-in to fit and stack the (expensive) Bayesian hierarchical DC model,
            persisting ``model_bayesian.pkl`` so the simulator can emit credible intervals.
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
    team_xg = (
        read_table(Table.TEAM_XG, settings) if table_exists(Table.TEAM_XG, settings) else None
    )

    if model == "dixon_coles":
        dc = _all_history_dixon_coles(matches, BaseProbConfig())
        settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
        path = settings.processed_data_dir / _ARTIFACT_FILES["dixon_coles"]
        with path.open("wb") as fh:
            pickle.dump(dc, fh)
        logger.info("train.persisted", artifact="dixon_coles", path=str(path))
        return

    base_config = BaseProbConfig(use_bayesian=bayesian)
    artifacts = train_full_stack(
        matches, base_config=base_config, market_odds=market_odds, team_xg=team_xg
    )
    persist_artifacts(artifacts, settings)
    logger.info("train.done", rows=artifacts.stacked_frame.height, bayesian=bayesian)
