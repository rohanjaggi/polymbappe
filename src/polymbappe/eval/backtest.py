"""Leave-one-tournament-out backtest for the minimum viable model.

Protocol (spec section 7.1): for each test tournament, Dixon-Coles trains on all matches
before it; base probabilities (DC + Elo + market) are precomputed once per tournament;
the meta-learner is then fit on every *other* tournament's base probabilities (true
out-of-fold stacking) and evaluated on the held-out tournament.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np
import polars as pl
import structlog

from polymbappe.context.adjuster import ContextualAdjusterConfig
from polymbappe.eval.base_probs import BaseProbConfig, compute_tournament_base_probs
from polymbappe.eval.metrics import multiclass_log_loss, ranked_probability_score
from polymbappe.models.dixon_coles import DixonColesModel
from polymbappe.models.ensemble import Ensemble, EnsembleConfig
from polymbappe.models.meta import OUTCOMES

logger = structlog.get_logger(__name__)

_LABEL_TO_IDX = {label: idx for idx, label in enumerate(OUTCOMES)}

_DC_COLS = ["dc_home", "dc_draw", "dc_away"]
_BAY_COLS = ["bay_home", "bay_draw", "bay_away"]
_ELO_COLS = ["elo_home", "elo_draw", "elo_away"]
_MKT_COLS = ["mkt_home", "mkt_draw", "mkt_away"]
#: Tier-1 squad-value core feature stacked into the GBM (mirrors train._attach_core_features).
_SQUAD_COLS = ["squad_value_ratio"]
#: Tier-1 rolling form raw columns (per-team); pivoted to home_*/away_* in the backtest.
_FORM_RAW_COLS = ["gs_5", "ga_5", "pts_5", "gs_10", "ga_10", "pts_10"]
_FORM_COLS = [f"{side}_{c}" for side in ("home", "away") for c in _FORM_RAW_COLS]
_H2H_COLS = ["h2h_home_winrate", "h2h_meetings"]
_REST_COLS = ["home_rest_days", "away_rest_days"]


@dataclass(frozen=True, slots=True)
class Tournament:
    """A test tournament defined by competition name and date window."""

    name: str
    competition: str
    start: date
    end: date


#: Leave-one-tournament-out evaluation set (spec section 7.1). Competition strings match
#: the Kaggle international results `tournament` field.
DEFAULT_TOURNAMENTS: tuple[Tournament, ...] = (
    Tournament("WC2010", "FIFA World Cup", date(2010, 6, 11), date(2010, 7, 11)),
    Tournament("WC2014", "FIFA World Cup", date(2014, 6, 12), date(2014, 7, 13)),
    Tournament("WC2018", "FIFA World Cup", date(2018, 6, 14), date(2018, 7, 15)),
    Tournament("WC2022", "FIFA World Cup", date(2022, 11, 20), date(2022, 12, 18)),
    Tournament("EU2016", "UEFA Euro", date(2016, 6, 10), date(2016, 7, 10)),
    Tournament("EU2020", "UEFA Euro", date(2021, 6, 11), date(2021, 7, 11)),
    Tournament("EU2024", "UEFA Euro", date(2024, 6, 14), date(2024, 7, 14)),
    Tournament("CA2016", "Copa América", date(2016, 6, 3), date(2016, 6, 26)),
    Tournament("CA2019", "Copa América", date(2019, 6, 14), date(2019, 7, 7)),
    Tournament("CA2021", "Copa América", date(2021, 6, 13), date(2021, 7, 10)),
    Tournament("CA2024", "Copa América", date(2024, 6, 20), date(2024, 7, 14)),
)


@dataclass(slots=True)
class BacktestResult:
    """Per-tournament and aggregate metrics from a backtest run."""

    per_tournament: dict[str, dict[str, float]] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)

    @property
    def mean_rps(self) -> float:
        values = [m["rps"] for m in self.per_tournament.values()]
        return float(np.mean(values)) if values else float("nan")

    def to_frame(self) -> pl.DataFrame:
        rows = [{"tournament": name, **metrics} for name, metrics in self.per_tournament.items()]
        return pl.DataFrame(rows)


def select_fixtures(matches: pl.DataFrame, tournament: Tournament) -> pl.DataFrame:
    """Select a tournament's matches by competition name and date window."""

    return matches.filter(
        (pl.col("competition") == tournament.competition)
        & (pl.col("date") >= tournament.start)
        & (pl.col("date") <= tournament.end)
    )


def _has_market(df: pl.DataFrame) -> bool:
    """Whether a base-prob frame carries complete (non-null) market columns."""

    if not set(_MKT_COLS).issubset(df.columns):
        return False
    return int(df.select(_MKT_COLS).null_count().sum_horizontal()[0]) == 0


def _has_bay(df: pl.DataFrame) -> bool:
    """Whether a base-prob frame carries complete (non-null) Bayesian columns."""

    if not set(_BAY_COLS).issubset(df.columns):
        return False
    return int(df.select(_BAY_COLS).null_count().sum_horizontal()[0]) == 0


def _metrics(labels: list[str], y_prob: np.ndarray) -> dict[str, float]:
    idx = np.array([_LABEL_TO_IDX[label] for label in labels])
    one_hot = np.zeros_like(y_prob)
    one_hot[np.arange(len(idx)), idx] = 1.0
    return {
        "rps": ranked_probability_score(idx, y_prob),
        "log_loss": multiclass_log_loss(idx, y_prob),
        "brier": float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1))),
        "n": float(len(idx)),
    }


def _pivot_team_features(
    team_frame: pl.DataFrame,
    fixtures: pl.DataFrame,
    feature_cols: list[str],
) -> pl.DataFrame:
    """Pivot a ``(match_id, team, *features)`` frame to ``(match_id, home_feat, away_feat)``."""

    fixture_ids = set(fixtures["match_id"].to_list())
    filtered = team_frame.filter(pl.col("match_id").is_in(fixture_ids))
    if filtered.is_empty():
        return pl.DataFrame(schema={"match_id": pl.Utf8})

    fx_map = fixtures.select("match_id", "home_team", "away_team")
    joined = filtered.join(fx_map, on="match_id", how="left")

    home_feat = (
        joined.filter(pl.col("team") == pl.col("home_team"))
        .select(["match_id"] + feature_cols)
        .rename({c: f"home_{c}" for c in feature_cols})
    )
    away_feat = (
        joined.filter(pl.col("team") == pl.col("away_team"))
        .select(["match_id"] + feature_cols)
        .rename({c: f"away_{c}" for c in feature_cols})
    )
    return home_feat.join(away_feat, on="match_id", how="outer_coalesce")


def _prepare_rolling_form(
    matches: pl.DataFrame,
    sorted_tournaments: list[Tournament],
    per_tournament_probs: dict[str, pl.DataFrame],
) -> list[str]:
    """Compute rolling form features and join into each tournament frame, in-place.

    Returns the list of home_*/away_* column names successfully added (empty on failure).
    """

    try:
        from polymbappe.features.context import build_form_features

        for tournament in sorted_tournaments:
            if tournament.name not in per_tournament_probs:
                continue
            history = matches.filter(pl.col("date") < tournament.start)
            fixtures = select_fixtures(matches, tournament)
            if history.is_empty() or fixtures.is_empty():
                continue
            combined = pl.concat([history, fixtures], how="diagonal_relaxed")
            form = build_form_features(combined)
            pivot = _pivot_team_features(form, fixtures, _FORM_RAW_COLS)
            if pivot.is_empty():
                continue
            per_tournament_probs[tournament.name] = per_tournament_probs[tournament.name].join(
                pivot, on="match_id", how="left"
            )

        sample = next(iter(per_tournament_probs.values()))
        return [c for c in _FORM_COLS if c in sample.columns]
    except Exception as exc:  # noqa: BLE001
        logger.warning("backtest.form_skip", error=str(exc))
        return []


def _prepare_h2h(
    matches: pl.DataFrame,
    sorted_tournaments: list[Tournament],
    per_tournament_probs: dict[str, pl.DataFrame],
) -> list[str]:
    """Compute H2H features and join into each tournament frame, in-place."""

    try:
        from polymbappe.features.context import build_h2h_features

        for tournament in sorted_tournaments:
            if tournament.name not in per_tournament_probs:
                continue
            history = matches.filter(pl.col("date") < tournament.start)
            fixtures = select_fixtures(matches, tournament)
            if history.is_empty() or fixtures.is_empty():
                continue
            combined = pl.concat([history, fixtures], how="diagonal_relaxed")
            h2h = build_h2h_features(combined)
            fixture_ids = set(fixtures["match_id"].to_list())
            h2h_filt = h2h.filter(pl.col("match_id").is_in(fixture_ids))
            if h2h_filt.is_empty():
                continue
            per_tournament_probs[tournament.name] = per_tournament_probs[tournament.name].join(
                h2h_filt, on="match_id", how="left"
            )

        sample = next(iter(per_tournament_probs.values()))
        return [c for c in _H2H_COLS if c in sample.columns]
    except Exception as exc:  # noqa: BLE001
        logger.warning("backtest.h2h_skip", error=str(exc))
        return []


def _prepare_rest_days(
    matches: pl.DataFrame,
    sorted_tournaments: list[Tournament],
    per_tournament_probs: dict[str, pl.DataFrame],
) -> list[str]:
    """Compute rest-days features and join into each tournament frame, in-place."""

    try:
        from polymbappe.features.context import build_rest_features

        for tournament in sorted_tournaments:
            if tournament.name not in per_tournament_probs:
                continue
            history = matches.filter(pl.col("date") < tournament.start)
            fixtures = select_fixtures(matches, tournament)
            if history.is_empty() or fixtures.is_empty():
                continue
            combined = pl.concat([history, fixtures], how="diagonal_relaxed")
            rest = build_rest_features(combined)
            pivot = _pivot_team_features(rest, fixtures, ["rest_days"])
            if pivot.is_empty():
                continue
            per_tournament_probs[tournament.name] = per_tournament_probs[tournament.name].join(
                pivot, on="match_id", how="left"
            )

        sample = next(iter(per_tournament_probs.values()))
        return [c for c in _REST_COLS if c in sample.columns]
    except Exception as exc:  # noqa: BLE001
        logger.warning("backtest.rest_skip", error=str(exc))
        return []


def run_leave_one_tournament_out(
    matches: pl.DataFrame,
    tournaments: tuple[Tournament, ...] = DEFAULT_TOURNAMENTS,
    *,
    base_config: BaseProbConfig | None = None,
    ensemble_config: EnsembleConfig | None = None,
    contextual_config: ContextualAdjusterConfig | None = None,
    market_odds: pl.DataFrame | None = None,
    squad_valuations: pl.DataFrame | None = None,
    toggle_rolling_form: bool = True,
    toggle_h2h: bool = True,
    toggle_rest_days: bool = True,
) -> BacktestResult:
    """Run the leave-one-tournament-out backtest over the stacked ensemble.

    For each held-out tournament the stacking :class:`~polymbappe.models.ensemble.Ensemble`
    (base-probability groups -> optional LightGBM -> meta-learner) is fit on every *other*
    tournament's base probabilities and evaluated on the held-out one. When
    ``contextual_config`` enables the layer, a :class:`ContextualAdjuster` is fit on the
    training fold's residuals and applied to the held-out predictions (capped at ±3pp).

    The full search space flows through here: ``ensemble_config`` carries the meta-learner
    choice / regularization and the GBM hyperparameters, ``contextual_config`` the
    contextual toggles. Returns per-tournament RPS / log loss / Brier and the feature set
    used. Requires at least two tournaments with fixtures present in ``matches``.
    """

    base_config = base_config or BaseProbConfig()
    ensemble_config = ensemble_config or EnsembleConfig()

    dc_model = DixonColesModel(base_config.dixon_coles)
    sorted_tournaments = sorted(tournaments, key=lambda t: t.start)

    per_tournament_probs: dict[str, pl.DataFrame] = {}
    for tournament in sorted_tournaments:
        fixtures = select_fixtures(matches, tournament)
        if fixtures.is_empty():
            continue
        history = matches.filter(pl.col("date") < tournament.start)
        if history.is_empty():
            logger.warning("backtest.no_history", tournament=tournament.name)
            continue
        per_tournament_probs[tournament.name] = compute_tournament_base_probs(
            history,
            fixtures,
            tournament=tournament.name,
            config=base_config,
            market_odds=market_odds,
            dc_model=dc_model,
        )

    if len(per_tournament_probs) < 2:
        raise ValueError(
            "Leave-one-tournament-out needs >=2 tournaments with fixtures and history; "
            f"found {len(per_tournament_probs)}."
        )

    # Use the market features only if every tournament has non-null market odds; likewise
    # stack the Bayesian group only when ``use_bayesian`` produced complete bay_* columns.
    use_market = all(_has_market(df) for df in per_tournament_probs.values())
    use_bay = base_config.use_bayesian and all(
        _has_bay(df) for df in per_tournament_probs.values()
    )
    feature_cols = (
        _DC_COLS
        + (_BAY_COLS if use_bay else [])
        + _ELO_COLS
        + (_MKT_COLS if use_market else [])
    )
    base_groups = tuple(
        g
        for g, present in (("dc", True), ("bay", use_bay), ("elo", True), ("mkt", use_market))
        if present
    )
    # GBM stacks over the base-model H/D/A outputs (the only features the base-prob frame
    # carries); its OOF predictions then enter the meta-learner.
    ensemble_config = replace(ensemble_config, base_groups=base_groups, market_blind=False)
    gbm_cols = feature_cols if ensemble_config.use_gbm else None

    # Stack the Tier-1 squad-value ratio into the GBM (mirrors train._attach_core_features) when
    # valuations are supplied. It is a continuous core feature, so it only has a path through the
    # GBM — with the GBM off it cannot enter (the meta-learner sees base-group probs only).
    use_squad = _prepare_squad(matches, sorted_tournaments, per_tournament_probs, squad_valuations)
    if use_squad and gbm_cols is not None:
        gbm_cols = gbm_cols + _SQUAD_COLS

    # Tier-1 backtestable features: rolling form, H2H, rest days. Each only has a path through
    # the GBM (meta-learner sees base-group probs only), so they are no-ops when GBM is off.
    tier1_cols: list[str] = []
    if gbm_cols is not None:
        if toggle_rolling_form:
            tier1_cols += _prepare_rolling_form(matches, sorted_tournaments, per_tournament_probs)
        if toggle_h2h:
            tier1_cols += _prepare_h2h(matches, sorted_tournaments, per_tournament_probs)
        if toggle_rest_days:
            tier1_cols += _prepare_rest_days(matches, sorted_tournaments, per_tournament_probs)
        if tier1_cols:
            gbm_cols = gbm_cols + tier1_cols

    use_contextual, feature_groups = _prepare_contextual(
        matches, sorted_tournaments, per_tournament_probs, contextual_config
    )

    reported_cols = (
        feature_cols
        + (_SQUAD_COLS if (use_squad and gbm_cols is not None) else [])
        + tier1_cols
    )
    result = BacktestResult(feature_columns=reported_cols)
    names = list(per_tournament_probs)
    for held_out in names:
        train = pl.concat(
            [per_tournament_probs[n] for n in names if n != held_out], how="vertical_relaxed"
        )
        ensemble = Ensemble(ensemble_config, gbm_feature_columns=gbm_cols).fit(train)
        test = per_tournament_probs[held_out]
        y_prob = ensemble.predict_proba(test)
        if use_contextual:
            from polymbappe.context.adjuster import ContextualAdjuster

            adjuster = ContextualAdjuster(feature_groups, contextual_config).fit(
                train, ensemble.predict_proba(train)
            )
            y_prob = adjuster.adjust(test, y_prob)
        result.per_tournament[held_out] = _metrics(test["label"].to_list(), y_prob)

    logger.info(
        "backtest.done",
        mean_rps=round(result.mean_rps, 4),
        features=reported_cols,
        meta=ensemble_config.meta.learner,
        gbm=ensemble_config.use_gbm,
        squad=use_squad and gbm_cols is not None,
        tier1_extra=tier1_cols,
        contextual=use_contextual,
        tournaments=names,
    )
    return result


def _squad_value_ratio(
    matches: pl.DataFrame,
    valuations: pl.DataFrame,
    tournaments: tuple[Tournament, ...],
) -> pl.DataFrame:
    """Per-match ``squad_value_ratio`` (= home_log_value − away_log_value), point-in-time.

    Mirrors :func:`~polymbappe.features.pipeline.FeaturePipeline.build_core_matrix`'s squad
    derivation but emits one ``(match_id, squad_value_ratio)`` row per fixture instead of the
    home/away team-table form. Each fixture uses *its own* tournament's snapshot (leakage-safe);
    only fixtures where both teams have a snapshot value get a row (others stay null on join).
    """

    from polymbappe.features.squad import build_squad_features

    per_team = build_squad_features(valuations)
    rows: list[dict[str, object]] = []
    for tournament in tournaments:
        snapshot = per_team.filter(pl.col("tournament") == tournament.name)
        if snapshot.is_empty():
            continue
        fixtures = select_fixtures(matches, tournament)
        if fixtures.is_empty():
            continue
        value = {r["team"]: r["log_total_value"] for r in snapshot.iter_rows(named=True)}
        for fx in fixtures.iter_rows(named=True):
            home, away = value.get(fx["home_team"]), value.get(fx["away_team"])
            if home is not None and away is not None:
                rows.append({"match_id": fx["match_id"], "squad_value_ratio": home - away})
    return pl.DataFrame(rows, schema={"match_id": pl.Utf8, "squad_value_ratio": pl.Float64})


def _prepare_squad(
    matches: pl.DataFrame,
    sorted_tournaments: list[Tournament],
    per_tournament_probs: dict[str, pl.DataFrame],
    squad_valuations: pl.DataFrame | None,
) -> bool:
    """Join ``squad_value_ratio`` into each tournament frame by ``match_id``, in-place.

    Returns False (leaving frames untouched) when no valuations are supplied or none of the
    fixtures can be valued; never fatal — a failure degrades to "no squad feature".
    """

    if squad_valuations is None or squad_valuations.is_empty():
        return False
    try:
        ratio = _squad_value_ratio(matches, squad_valuations, tuple(sorted_tournaments))
        if ratio.is_empty():
            return False
        for name, df in per_tournament_probs.items():
            per_tournament_probs[name] = df.join(ratio, on="match_id", how="left")
        return True
    except Exception as exc:  # noqa: BLE001 - squad feature is optional, never fatal
        logger.warning("backtest.squad_skip", error=str(exc))
        return False


def _prepare_contextual(
    matches: pl.DataFrame,
    sorted_tournaments: list[Tournament],
    per_tournament_probs: dict[str, pl.DataFrame],
    contextual_config: ContextualAdjusterConfig | None,
) -> tuple[bool, dict[str, list[str]]]:
    """Join contextual feature columns into each tournament frame, in-place.

    Returns ``(enabled, feature_groups)``: ``enabled`` is False (and the frames are left
    untouched) unless a contextual layer is requested and its features build successfully.
    """

    if contextual_config is None or not contextual_config.enable_contextual_layer:
        return False, {}
    try:
        from polymbappe.context.runtime import (
            FEATURE_GROUPS,
            build_tournament_context_features,
        )

        context = build_tournament_context_features(matches, sorted_tournaments)
        for name, df in per_tournament_probs.items():
            per_tournament_probs[name] = df.join(context, on="match_id", how="left")
        return True, FEATURE_GROUPS
    except Exception as exc:  # noqa: BLE001 - contextual layer is optional, never fatal
        logger.warning("backtest.context_skip", error=str(exc))
        return False, {}


def run_walk_forward_backtest() -> None:
    """CLI entrypoint: run the LOTO backtest over stored matches and print the report."""

    from polymbappe.config import Settings
    from polymbappe.data.store import read_table, table_exists
    from polymbappe.data.tables import Table

    settings = Settings()
    matches = read_table(Table.MATCHES, settings)
    market_odds = (
        read_table(Table.MARKET_ODDS, settings)
        if table_exists(Table.MARKET_ODDS, settings)
        else None
    )
    squad_valuations = (
        read_table(Table.SQUAD_VALUATIONS, settings)
        if table_exists(Table.SQUAD_VALUATIONS, settings)
        else None
    )
    result = run_leave_one_tournament_out(
        matches, market_odds=market_odds, squad_valuations=squad_valuations
    )
    print(f"Features: {', '.join(result.feature_columns)}")
    print(result.to_frame().sort("tournament"))
    print(f"Mean RPS: {result.mean_rps:.4f}")


@dataclass(slots=True)
class BayesianABResult:
    """Outcome of the Bayesian-vs-MLE-only kill-criterion A/B (spec 8.1 table)."""

    without_bayesian: BacktestResult
    with_bayesian: BacktestResult
    delta: float  # positive => Bayesian improves mean RPS
    min_delta: float
    wins: int  # tournaments where the Bayesian ensemble beats the MLE-only one
    n_tournaments: int
    keep_bayesian: bool  # spec rule: improves mean RPS by > min_delta
    gate_decision: str  # stricter autotuner gate (delta AND >=3 tournament wins)


def compare_bayesian_ab(
    matches: pl.DataFrame,
    tournaments: tuple[Tournament, ...] = DEFAULT_TOURNAMENTS,
    *,
    base_config: BaseProbConfig | None = None,
    ensemble_config: EnsembleConfig | None = None,
    contextual_config: ContextualAdjusterConfig | None = None,
    market_odds: pl.DataFrame | None = None,
    gate: object | None = None,
) -> BayesianABResult:
    """Measure the Bayesian kill criterion: run the LOTO backtest with and without it.

    Runs the leave-one-tournament-out backtest exactly twice — ``use_bayesian=False`` then
    ``True`` — and applies the +0.003-RPS acceptance gate (the single source of truth in
    :class:`~polymbappe.tune.leaderboard.AcceptanceGate`). ``keep_bayesian`` implements the
    spec 8.1 rule directly (drop the Bayesian model unless it improves mean RPS by more than
    ``min_delta``); ``gate_decision`` reports the stricter autotuner gate that also requires
    winning >=3 individual tournaments. This is a standalone harness and never runs inside
    the autotuner's per-trial TPE loop, so the 2 h budget is untouched.
    """

    from polymbappe.tune.leaderboard import AcceptanceGate
    from polymbappe.tune.objective import ExperimentMetrics

    base_config = base_config or BaseProbConfig()
    gate = gate or AcceptanceGate()

    without = run_leave_one_tournament_out(
        matches,
        tournaments,
        base_config=replace(base_config, use_bayesian=False),
        ensemble_config=ensemble_config,
        contextual_config=contextual_config,
        market_odds=market_odds,
    )
    with_bay = run_leave_one_tournament_out(
        matches,
        tournaments,
        base_config=replace(base_config, use_bayesian=True),
        ensemble_config=ensemble_config,
        contextual_config=contextual_config,
        market_odds=market_odds,
    )

    delta = without.mean_rps - with_bay.mean_rps  # positive => Bayesian is better
    wins = sum(
        1
        for name, metrics in with_bay.per_tournament.items()
        if name in without.per_tournament
        and metrics["rps"] < without.per_tournament[name]["rps"]
    )

    def _metrics_of(result: BacktestResult) -> ExperimentMetrics:
        return ExperimentMetrics(
            mean_rps=result.mean_rps,
            per_tournament={k: v["rps"] for k, v in result.per_tournament.items()},
            feature_columns=result.feature_columns,
        )

    decision = gate.decide(_metrics_of(with_bay), _metrics_of(without))
    return BayesianABResult(
        without_bayesian=without,
        with_bayesian=with_bay,
        delta=delta,
        min_delta=gate.min_delta,
        wins=wins,
        n_tournaments=len(with_bay.per_tournament),
        keep_bayesian=delta > gate.min_delta,
        gate_decision=decision,
    )


def run_bayesian_ab() -> None:
    """CLI entrypoint: run the Bayesian kill-criterion A/B over stored matches and report."""

    from polymbappe.config import Settings
    from polymbappe.data.store import read_table, table_exists
    from polymbappe.data.tables import Table

    settings = Settings()
    matches = read_table(Table.MATCHES, settings)
    market_odds = (
        read_table(Table.MARKET_ODDS, settings)
        if table_exists(Table.MARKET_ODDS, settings)
        else None
    )
    ab = compare_bayesian_ab(matches, market_odds=market_odds)
    print(f"Mean RPS without Bayesian: {ab.without_bayesian.mean_rps:.4f}")
    print(f"Mean RPS with Bayesian:    {ab.with_bayesian.mean_rps:.4f}")
    print(f"Delta (improvement):       {ab.delta:+.4f}  (threshold > {ab.min_delta})")
    print(f"Tournament wins:           {ab.wins}/{ab.n_tournaments}")
    print(f"Keep Bayesian (spec 8.1):  {ab.keep_bayesian}")
    print(f"Autotuner gate decision:   {ab.gate_decision}")
