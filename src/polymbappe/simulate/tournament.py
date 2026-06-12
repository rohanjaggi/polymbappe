"""Monte Carlo tournament engine (spec section 4).

Wires the existing group-table, best-third-place, and bracket-seeding logic into a full
48-team 2026 World Cup simulation and runs it ``n_sims`` times (100K in production).

Per simulation:

1. **Group stage** — every group match samples a scoreline from the strength model's
   (optionally contextually-adjusted) score matrix; :func:`resolve_group_table` applies the
   full FIFA 2026 tiebreakers.
2. **Correlated within-sim strength updates** (spec 4.1) — each team carries a latent
   strength delta that nudges up/down as group results land
   (``delta += lr * (observed_GD - expected_GD)``) and propagates into the knockout rounds,
   modelling that group form reveals true tournament level.
3. **Best third-placed** — :func:`select_best_third_placed` picks the 8 of 12 qualifiers.
4. **Knockout** — :func:`seed_round_of_32` seeds R32 with pathway constraints; each tie is
   resolved by :func:`knockout_home_winprob` (regulation -> extra time -> penalties), with
   the 48-team upset floor applied to lopsided R32 matchups (spec 4.2).

Aggregated outputs give per-team stage-reaching and group-finish probabilities. Live
staleness detection (spec 4.5) is provided separately by :class:`StalenessMonitor`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from math import exp
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from polymbappe.data.schema import GroupStanding, Match

if TYPE_CHECKING:
    from polymbappe.context.runtime import FixtureContext
from polymbappe.simulate.bracket import seed_round_of_32
from polymbappe.simulate.group import resolve_group_table
from polymbappe.simulate.match import (
    ET_GOAL_SCALE,
    hda_marginals,
    knockout_home_winprob,
    reweight_matrix_to_hda,
    sample_scoreline,
    score_matrix_from_rates,
)
from polymbappe.simulate.third_place import select_best_third_placed

#: Per-match contextual hook: ``(home, away, base_hda) -> adjusted_hda`` (both length-3
#: H/D/A vectors). Returned by the contextual adjuster wiring; reweights the score matrix.
ContextHook = Callable[[str, str, np.ndarray], np.ndarray]


def _contextualize(
    matrix: np.ndarray, home: str, away: str, hook: ContextHook | None
) -> np.ndarray:
    """Apply a contextual H/D/A adjustment to a score matrix via reweighting."""

    if hook is None:
        return matrix
    base_hda = np.array(hda_marginals(matrix))
    adjusted = hook(home, away, base_hda)
    return reweight_matrix_to_hda(matrix, adjusted)


#: Round-robin pairings for a 4-team group; matchday 3 (last two) is the "final matchday".
_GROUP_SCHEDULE: tuple[tuple[int, int, int], ...] = (
    (0, 1, 1),
    (2, 3, 1),
    (0, 2, 2),
    (1, 3, 2),
    (0, 3, 3),
    (1, 2, 3),
)

#: Stage-reaching keys, broadest to narrowest.
STAGES: tuple[str, ...] = ("R32", "R16", "QF", "SF", "FINAL", "champion")

#: 48-team mitigation: floor on the underdog's R32 win probability for >300 Elo gaps.
R32_UPSET_FLOOR: float = 0.08
R32_ELO_GAP: float = 300.0

#: The live 2026 World Cup, used by the contextual hook to build point-in-time
#: cohesion / manager lookups (see :func:`_load_context_hook`). ``name`` is the data
#: contract: the ``squads`` / ``manager_records`` tables carry their 2026 rows under
#: ``tournament == "WC2026"``. ``start`` bounds the match history used for those lookups.
WC2026_NAME: str = "WC2026"
WC2026_START: date = date(2026, 6, 11)


@dataclass(slots=True)
class StrengthModel:
    """Per-team attack/defense strengths driving the Poisson score matrices."""

    attack: dict[str, float]
    defense: dict[str, float]
    home_advantage: float = 0.25
    rho: float = 0.0
    max_goals: int = 8
    hosts: frozenset[str] = frozenset()
    host_bonus: float = 0.15

    @classmethod
    def from_dixon_coles(
        cls,
        model: object,
        hosts: frozenset[str] = frozenset(),
        host_bonus: float = 0.15,
    ) -> StrengthModel:
        """Build from a fitted :class:`~polymbappe.models.dixon_coles.DixonColesModel`."""

        idx = model.team_to_index  # type: ignore[attr-defined]
        attack = {t: float(model.attack[i]) for t, i in idx.items()}  # type: ignore[attr-defined]
        defense = {t: float(model.defense[i]) for t, i in idx.items()}  # type: ignore[attr-defined]
        return cls(
            attack=attack,
            defense=defense,
            home_advantage=float(model.home_advantage),  # type: ignore[attr-defined]
            rho=float(model.rho),  # type: ignore[attr-defined]
            max_goals=model.config.max_goals,  # type: ignore[attr-defined]
            hosts=hosts,
            host_bonus=host_bonus,
        )

    def rates(
        self, home: str, away: str, neutral: bool, dh: float = 0.0, da: float = 0.0
    ) -> tuple[float, float]:
        """Expected (home, away) goals including latent strength deltas ``dh``/``da``.

        A host nation (USA/MEX/CAN for 2026) carries a reduced ``host_bonus`` log-rate
        boost on top of its attack — applied to whichever side it is on and even at
        neutral venues, since the World Cup is played in the hosts' countries (spec 4.1).
        Group and knockout matches both pass ``neutral=True``, so this is the only
        home-like effect in the tournament.
        """

        home_term = 0.0 if neutral else self.home_advantage
        if home in self.hosts:
            home_term += self.host_bonus
        away_term = self.host_bonus if away in self.hosts else 0.0
        lam = exp(home_term + self.attack.get(home, 0.0) + dh + self.defense.get(away, 0.0))
        mu = exp(away_term + self.attack.get(away, 0.0) + da + self.defense.get(home, 0.0))
        return lam, mu

    def score_matrix(
        self, home: str, away: str, neutral: bool, dh: float = 0.0, da: float = 0.0
    ) -> np.ndarray:
        lam, mu = self.rates(home, away, neutral, dh, da)
        return score_matrix_from_rates(lam, mu, self.rho, self.max_goals)


@dataclass(slots=True)
class TournamentStructure:
    """Static 2026 tournament inputs."""

    groups: dict[str, list[str]]
    elo: dict[str, float] = field(default_factory=dict)
    penalty_rate: dict[str, float] = field(default_factory=dict)
    n_qualify_thirds: int = 8

    @property
    def teams(self) -> list[str]:
        return [t for members in self.groups.values() for t in members]


@dataclass(slots=True)
class SimulationResult:
    """Aggregated Monte Carlo outputs."""

    n_sims: int
    stage_counts: dict[str, dict[str, int]]
    group_finish_counts: dict[str, dict[int, int]]

    def stage_probabilities(self) -> pl.DataFrame:
        rows = [
            {"team": team, **{s: counts.get(s, 0) / self.n_sims for s in STAGES}}
            for team, counts in self.stage_counts.items()
        ]
        return pl.DataFrame(rows).sort("champion", descending=True)

    def group_probabilities(self) -> pl.DataFrame:
        rows = [
            {"team": team, **{f"finish_{r}": counts.get(r, 0) / self.n_sims for r in (1, 2, 3, 4)}}
            for team, counts in self.group_finish_counts.items()
        ]
        return pl.DataFrame(rows)


class _LatentStrength:
    """Per-simulation latent strength deltas with a small correlated-update learning rate."""

    def __init__(self, learning_rate: float = 0.05) -> None:
        self.lr = learning_rate
        self._delta: dict[str, float] = {}

    def get(self, team: str) -> float:
        return self._delta.get(team, 0.0)

    def update(self, team: str, observed_gd: float, expected_gd: float) -> None:
        self._delta[team] = self.get(team) + self.lr * (observed_gd - expected_gd)


def _simulate_group(
    group: str,
    teams: list[str],
    model: StrengthModel,
    latent: _LatentStrength,
    rng: np.random.Generator,
    context_hook: ContextHook | None = None,
) -> list[GroupStanding]:
    """Play one group's six matches and return the resolved standings."""

    matches: list[Match] = []
    for k, (i, j, _matchday) in enumerate(_GROUP_SCHEDULE):
        home, away = teams[i], teams[j]
        dh, da = latent.get(home), latent.get(away)
        matrix = _contextualize(
            model.score_matrix(home, away, neutral=True, dh=dh, da=da), home, away, context_hook
        )
        hg, ag = sample_scoreline(matrix, rng)
        lam, mu = model.rates(home, away, neutral=True, dh=dh, da=da)
        latent.update(home, hg - ag, lam - mu)
        latent.update(away, ag - hg, mu - lam)
        matches.append(
            Match(
                match_id=f"{group}-{k}",
                date=date(2026, 6, 11),
                home_team=home,
                away_team=away,
                home_goals=hg,
                away_goals=ag,
                competition="FIFA World Cup",
                group=group,
                neutral_site=True,
            )
        )
    return resolve_group_table(group, teams, matches, rng)


def _knockout_winner(
    home: str,
    away: str,
    model: StrengthModel,
    latent: _LatentStrength,
    structure: TournamentStructure,
    rng: np.random.Generator,
    apply_upset_floor: bool,
    context_hook: ContextHook | None = None,
) -> str:
    """Resolve one knockout tie, returning the advancing team."""

    dh, da = latent.get(home), latent.get(away)
    lam, mu = model.rates(home, away, neutral=True, dh=dh, da=da)
    matrix_reg = _contextualize(
        score_matrix_from_rates(lam, mu, model.rho, model.max_goals), home, away, context_hook
    )
    matrix_et = score_matrix_from_rates(
        lam * ET_GOAL_SCALE, mu * ET_GOAL_SCALE, model.rho, model.max_goals
    )
    p_home = knockout_home_winprob(
        matrix_reg,
        matrix_et,
        home_pen_rate=structure.penalty_rate.get(home, 0.5),
        away_pen_rate=structure.penalty_rate.get(away, 0.5),
        first_shooter_home=bool(rng.random() < 0.5),
    )

    if apply_upset_floor and structure.elo:
        gap = structure.elo.get(home, 1500.0) - structure.elo.get(away, 1500.0)
        if gap > R32_ELO_GAP:  # home heavy favorite -> floor away upset chance
            p_home = min(p_home, 1.0 - R32_UPSET_FLOOR)
        elif gap < -R32_ELO_GAP:  # away heavy favorite -> floor home upset chance
            p_home = max(p_home, R32_UPSET_FLOOR)

    return home if rng.random() < p_home else away


def simulate_tournament(
    structure: TournamentStructure,
    model: StrengthModel,
    *,
    n_sims: int = 100_000,
    rng: np.random.Generator | None = None,
    learning_rate: float = 0.05,
    context_hook: ContextHook | None = None,
) -> SimulationResult:
    """Run the full Monte Carlo tournament simulation.

    ``context_hook`` (optional) applies a per-match contextual H/D/A adjustment by
    reweighting each score matrix (spec 4.1 per-match contextual injection).
    """

    rng = rng or np.random.default_rng()
    teams = structure.teams
    stage_counts: dict[str, dict[str, int]] = {t: {} for t in teams}
    group_finish: dict[str, dict[int, int]] = {t: {} for t in teams}
    # The winner of each round reaches the next stage (R16 round -> QF teams, etc.).
    later_rounds = ("QF", "SF", "FINAL", "champion")

    for _ in range(n_sims):
        latent = _LatentStrength(learning_rate)
        winners: list[GroupStanding] = []
        runners_up: list[str] = []
        thirds: list[GroupStanding] = []
        for group, members in structure.groups.items():
            standings = _simulate_group(group, members, model, latent, rng, context_hook)
            for rank, row in enumerate(standings, start=1):
                group_finish[row.team][rank] = group_finish[row.team].get(rank, 0) + 1
            winners.append(standings[0])
            runners_up.append(standings[1].team)
            thirds.append(standings[2])

        ranked_winners = sorted(
            winners,
            key=lambda s: (s.points, s.goal_difference, s.goals_scored),
            reverse=True,
        )
        best_thirds = [s.team for s in select_best_third_placed(thirds, structure.n_qualify_thirds)]
        other_qualifiers = runners_up + best_thirds

        ties = seed_round_of_32(ranked_winners, other_qualifiers, rng)
        # Everyone in a tie reached R32.
        for tie in ties:
            for team in (tie.home_team, tie.away_team):
                stage_counts[team]["R32"] = stage_counts[team].get("R32", 0) + 1

        # R32 round: winners reach R16. The upset floor applies only here (spec 4.2).
        current = [
            _knockout_winner(
                tie.home_team, tie.away_team, model, latent, structure, rng,
                apply_upset_floor=True, context_hook=context_hook,
            )
            for tie in ties
        ]
        for w in current:
            stage_counts[w]["R16"] = stage_counts[w].get("R16", 0) + 1

        # Later rounds: each round's winners reach the next stage.
        for stage_name in later_rounds:
            winners_next: list[str] = []
            for i in range(0, len(current), 2):
                winner = _knockout_winner(
                    current[i], current[i + 1], model, latent, structure, rng,
                    apply_upset_floor=False, context_hook=context_hook,
                )
                stage_counts[winner][stage_name] = stage_counts[winner].get(stage_name, 0) + 1
                winners_next.append(winner)
            current = winners_next

    return SimulationResult(
        n_sims=n_sims, stage_counts=stage_counts, group_finish_counts=group_finish
    )


def compute_match_predictions(
    structure: TournamentStructure,
    model: StrengthModel,
    context_hook: ContextHook | None = None,
    bayesian_model: object | None = None,
    credible_level: float = 0.9,
) -> pl.DataFrame:
    """Per group-stage fixture H/D/A probabilities and expected scoreline (spec 11).

    Iterates each group's round-robin pairings, deriving the contextually-adjusted score
    matrix and its H/D/A marginals plus expected goals. ``match_id`` follows the
    ``2026__home__away`` convention so externally-ingested 2026 market odds keyed the same
    way join cleanly for edge detection.

    When ``bayesian_model`` is supplied, six ``ci_*_low``/``ci_*_high`` credible-interval
    columns are appended from its posterior, feeding the edge pipeline's credible-interval
    test (spec 3.6). The point ``model_*`` probabilities stay the production point estimate;
    the credible band is the Bayesian base model's posterior uncertainty — a principled
    model-based proxy, not a literal interval around the point estimate.

    Returns columns ``match_id, group, home_team, away_team, model_home, model_draw,
    model_away, exp_home_goals, exp_away_goals`` (plus ``ci_*`` when Bayesian is supplied).
    """

    grid = np.arange(model.max_goals + 1)
    rows: list[dict[str, object]] = []
    for group, members in structure.groups.items():
        for i, j, _matchday in _GROUP_SCHEDULE:
            home, away = members[i], members[j]
            matrix = _contextualize(
                model.score_matrix(home, away, neutral=True), home, away, context_hook
            )
            h, d, a = hda_marginals(matrix)
            row: dict[str, object] = {
                "match_id": f"2026__{home}__{away}",
                "group": group,
                "home_team": home,
                "away_team": away,
                "model_home": h,
                "model_draw": d,
                "model_away": a,
                "exp_home_goals": float((matrix.sum(axis=1) * grid).sum()),
                "exp_away_goals": float((matrix.sum(axis=0) * grid).sum()),
            }
            if bayesian_model is not None:
                ci = bayesian_model.credible_interval(
                    home, away, neutral_site=True, level=credible_level
                )
                row.update(
                    {
                        "ci_home_low": ci["home_win"][0],
                        "ci_home_high": ci["home_win"][1],
                        "ci_draw_low": ci["draw"][0],
                        "ci_draw_high": ci["draw"][1],
                        "ci_away_low": ci["away_win"][0],
                        "ci_away_high": ci["away_win"][1],
                    }
                )
            rows.append(row)
    return pl.DataFrame(rows)


# -- staleness detection (spec 4.5) -------------------------------------------


def surprise_increment(predicted_prob: float, occurred: bool) -> float:
    """Per-match surprise: ``|actual - predicted|`` for the realized outcome."""

    return abs((1.0 if occurred else 0.0) - predicted_prob)


@dataclass(slots=True)
class StalenessMonitor:
    """Tracks cumulative forecast surprise against historical-baseline thresholds.

    ``yellow`` and ``red`` are cumulative-surprise thresholds calibrated from historical
    matchday sequences (75th / 90th percentile, spec 4.5). Crossing yellow is advisory;
    crossing red signals a full model re-estimation is warranted.
    """

    yellow: float
    red: float
    cumulative: float = 0.0
    n: int = 0

    def observe(self, predicted_prob: float, occurred: bool) -> str:
        """Record one completed match and return the current level."""

        self.cumulative += surprise_increment(predicted_prob, occurred)
        self.n += 1
        return self.level

    @property
    def level(self) -> str:
        if self.cumulative > self.red:
            return "red"
        if self.cumulative > self.yellow:
            return "yellow"
        return "green"


def refresh_market_odds(settings: object, logger: object) -> int:
    """Re-pull the latest market odds before edge computation (live-agent hook).

    Calls :func:`polymbappe.data.ingest.ingest_market_odds` in append mode so each
    simulation run can incorporate fresh Polymarket / Football-Data odds. Isolated: a
    network/source failure logs and returns -1 rather than aborting the simulation.
    """

    from polymbappe.data.ingest import ingest_market_odds

    try:
        n = ingest_market_odds(settings, live=True)  # type: ignore[arg-type]
        logger.info("simulate.odds_refreshed", rows=n)  # type: ignore[attr-defined]
        return n
    except Exception as exc:  # noqa: BLE001 - odds refresh must never break the sim
        logger.warning("simulate.odds_refresh_failed", error=str(exc))  # type: ignore[attr-defined]
        return -1


def run_tournament_simulation(
    n_sims: int = 100_000,
    with_context: bool = False,
    live: bool = False,
    refresh_odds: bool = False,
) -> None:
    """CLI entrypoint: load fitted artifacts + 2026 structure and simulate.

    Requires a trained Dixon-Coles artifact (``polymbappe train``); writes per-team
    stage-reaching and group-finish probabilities to ``data/outputs``. When ``refresh_odds``
    (or ``live``) is set, the latest market odds are re-pulled after fixtures are written so
    the edge artifact reflects the current market — the hook the live agent uses on re-runs.
    """

    import structlog

    from polymbappe.config import Settings
    from polymbappe.data.store import read_table, table_exists
    from polymbappe.data.tables import Table
    from polymbappe.features.context import HOSTS_2026
    from polymbappe.models.train import load_artifact
    from polymbappe.simulate.structure import (
        load_structure_2026,
        placeholder_structure_2026,
        structure_from_strengths,
        team_strengths,
    )

    logger = structlog.get_logger(__name__)
    settings = Settings()
    try:
        dc = load_artifact("dixon_coles", settings)
    except FileNotFoundError as exc:  # pragma: no cover - depends on prior train run
        raise FileNotFoundError(
            "No trained Dixon-Coles artifact found. Run `polymbappe train` first."
        ) from exc

    # Latest Elo per team (for pot-seeding + the knockout upset floor) when ingested.
    elo: dict[str, float] | None = None
    if table_exists(Table.ELO_SNAPSHOTS, settings):
        latest = (
            read_table(Table.ELO_SNAPSHOTS, settings)
            .sort("date")
            .group_by("team")
            .agg(pl.col("rating").last())
        )
        elo = {r["team"]: float(r["rating"]) for r in latest.iter_rows(named=True)}

    # Structure resolution: real published draw > trained-strength draw > placeholder.
    if (settings.configs_dir / "tournament_2026.yaml").exists():
        structure = load_structure_2026(settings)
        logger.info("simulate.structure", source="config")
    elif len(team_strengths(dc)) >= 48:
        structure = structure_from_strengths(dc, elo)
        logger.info(
            "simulate.structure", source="trained_strengths", elo_seeded=elo is not None
        )
    else:
        structure = placeholder_structure_2026()
        logger.warning("simulate.structure", source="placeholder", reason="<48 trained teams")

    model = StrengthModel.from_dixon_coles(dc, hosts=HOSTS_2026)

    # Optional Bayesian credible intervals for the edge pipeline (spec 3.6): used when a
    # ``model_bayesian.pkl`` artifact exists (written by `train --bayesian`).
    bayesian_model: object | None = None
    try:
        bayesian_model = load_artifact("bayesian", settings)
        logger.info("simulate.bayesian", status="loaded; emitting credible intervals")
    except FileNotFoundError:
        logger.info("simulate.bayesian", status="no artifact; point edges only")

    # Optional contextual injection: load the fitted adjuster and build a per-match hook.
    context_hook: ContextHook | None = None
    if with_context:
        context_hook = _load_context_hook(settings, structure, elo, logger)

    result = simulate_tournament(structure, model, n_sims=n_sims, context_hook=context_hook)
    predictions = compute_match_predictions(
        structure, model, context_hook, bayesian_model=bayesian_model
    )

    settings.outputs_data_dir.mkdir(parents=True, exist_ok=True)
    out = settings.outputs_data_dir
    result.stage_probabilities().write_parquet(out / "stage_probabilities.parquet")
    result.group_probabilities().write_parquet(out / "group_probabilities.parquet")
    predictions.write_parquet(out / "match_predictions.parquet")
    # Refresh odds AFTER fixtures are written so Polymarket can align to them, then edges.
    if refresh_odds or live:
        refresh_market_odds(settings, logger)
    _write_edges(predictions, settings, logger).write_parquet(out / "edges.parquet")
    print(result.stage_probabilities().head(15))


class _LiveTournament:
    """Minimal ``Tournament``-shaped object for the live 2026 cohesion / manager lookups.

    The runtime lookups only need ``.name`` (to select the 2026 snapshot / derive the
    leakage cutoff) and ``.start`` (to bound the match history). A tiny local shim avoids
    importing the heavier :class:`~polymbappe.eval.backtest.Tournament` into the hot path.
    """

    __slots__ = ("name", "start")

    def __init__(self, name: str = WC2026_NAME, start: date = WC2026_START) -> None:
        self.name = name
        self.start = start


def _live_fixture_context(
    settings: object,
    elo: dict[str, float] | None,
    logger: object,
) -> FixtureContext:
    """Build the per-pair :class:`FixtureContext` bundle for the live 2026 simulation.

    **Live data contract.** Cohesion / manager features 0-fill gracefully unless their
    2026 rows are present (the hook never hard-requires these tables):

    * ``squads`` must contain the 2026 pre-tournament call-up snapshot under
      ``tournament == "WC2026"`` (per-player ``team``/``club``/``age`` rows). Absent → all
      cohesion columns 0-fill.
    * ``manager_records`` must contain a 2026 row per nation identifying that nation's
      *current* manager (knockout stats may be 0). The leakage cutoff comes from the records'
      own ``tournament_order``; since ``"WC2026"`` is not among the historical
      ``tournament_order`` values, the cutoff falls back to ``+inf`` and every (strictly
      pre-2026) record is used for pedigree. Absent 2026 identity rows → manager columns
      0-fill.

    The bundle is built from history only via the same ``cohesion_lookup`` /
    ``manager_lookup`` the fit path uses, so the live per-pair frame is column-identical to
    the fit frame.
    """

    from polymbappe.context.runtime import (
        FixtureContext,
        cohesion_lookup,
        latest_overperformance,
        manager_lookup,
    )
    from polymbappe.data.store import read_table, table_exists
    from polymbappe.data.tables import Table

    matches = read_table(Table.MATCHES, settings)  # type: ignore[arg-type]
    team_xg = read_table(Table.TEAM_XG, settings) if table_exists(Table.TEAM_XG, settings) else None  # type: ignore[arg-type]
    overperf = latest_overperformance(matches, team_xg)

    tournament = _LiveTournament()
    cohesion: dict[str, tuple[float, float]] = {}
    if table_exists(Table.SQUADS, settings):  # type: ignore[arg-type]
        squads = read_table(Table.SQUADS, settings)  # type: ignore[arg-type]
        cohesion = cohesion_lookup(squads, tournament)
    manager: dict[str, dict[str, float]] = {}
    if table_exists(Table.MANAGER_RECORDS, settings):  # type: ignore[arg-type]
        records = read_table(Table.MANAGER_RECORDS, settings)  # type: ignore[arg-type]
        manager = manager_lookup(records, tournament)
    logger.info(  # type: ignore[attr-defined]
        "simulate.context.tables",
        squads_teams=len(cohesion),
        manager_teams=len(manager),
    )
    return FixtureContext(overperf=overperf, elo=elo or {}, cohesion=cohesion, manager=manager)


def _context_feature_frame(
    teams: list[str], ctx: FixtureContext
) -> tuple[list[tuple[str, str]], pl.DataFrame]:
    """Per-ordered-pair contextual feature frame for the live teams (testable seam).

    Returns the ordered ``(home, away)`` pairs and the matching feature frame whose columns
    are exactly :data:`~polymbappe.context.runtime.SIM_CONTEXT_FEATURES`, built through the
    same :func:`~polymbappe.context.runtime.fixture_feature_row` the fit path uses.
    """

    from polymbappe.context.runtime import fixture_feature_row

    pairs = [(h, a) for h in teams for a in teams if h != a]
    rows = [fixture_feature_row(h, a, ctx) for h, a in pairs]
    return pairs, pl.DataFrame(rows)


def _load_context_hook(
    settings: object, structure: TournamentStructure, elo: dict[str, float] | None, logger: object
) -> ContextHook | None:
    """Build a per-match contextual hook from the fitted adjuster artifact, if present.

    The contextual adjustment depends only on the matchup (team features), not on the
    in-sim state, so the raw adjustment is precomputed once per ordered team pair in a
    single batched prediction. The returned hook is then an O(1) lookup plus a cheap capped
    re-projection — fast enough for 100K sims.

    Cohesion / manager features are assembled from point-in-time lookups for the 2026
    tournament (``WC2026``) via :func:`_live_fixture_context`; see its docstring for the
    live data contract. They 0-fill when the ``squads`` / ``manager_records`` tables (or
    their 2026 rows) are absent, so the hook never hard-requires them.
    """

    from polymbappe.context.adjuster import apply_adjustment
    from polymbappe.data.store import table_exists
    from polymbappe.data.tables import Table
    from polymbappe.models.train import load_artifact

    try:
        adjuster = load_artifact("contextual_adjuster", settings)  # type: ignore[arg-type]
    except FileNotFoundError:
        logger.warning("simulate.context", status="no adjuster artifact; run `train`")  # type: ignore[attr-defined]
        return None
    if not table_exists(Table.MATCHES, settings):  # type: ignore[arg-type]
        logger.warning("simulate.context", status="no matches table for features")  # type: ignore[attr-defined]
        return None

    ctx = _live_fixture_context(settings, elo, logger)
    pairs, frame = _context_feature_frame(structure.teams, ctx)
    raw = adjuster.predict_adjustment(frame)  # type: ignore[attr-defined]  # one batched call
    cache = {pair: raw[i] for i, pair in enumerate(pairs)}
    cap = adjuster.config.cap  # type: ignore[attr-defined]
    logger.info("simulate.context", status="applied", pairs=len(pairs))  # type: ignore[attr-defined]

    def hook(home: str, away: str, base_hda: np.ndarray) -> np.ndarray:
        adjustment = cache.get((home, away))
        if adjustment is None:
            return base_hda
        return apply_adjustment(base_hda.reshape(1, 3), adjustment.reshape(1, 3), cap)[0]

    return hook


_CI_COLS = (
    "ci_home_low", "ci_home_high", "ci_draw_low", "ci_draw_high", "ci_away_low", "ci_away_high",
)


def _write_edges(
    predictions: pl.DataFrame, settings: object, logger: object
) -> pl.DataFrame:
    """Compute market edges for the fixtures, or an empty edge table if no market odds.

    When the predictions carry Bayesian credible-interval columns, the stricter
    credible-interval edge test (spec 3.6) is applied — a divergence is flagged only if the
    Bayesian credible interval also excludes the market probability — and the output gains
    ``ci_low``/``ci_high`` columns. Otherwise the point-only edge test is used.
    """

    from polymbappe.data.store import read_table, table_exists
    from polymbappe.data.tables import Table
    from polymbappe.eval.market import compute_credible_edges, compute_edges

    has_ci = all(c in predictions.columns for c in _CI_COLS)
    if has_ci:
        empty = pl.DataFrame(
            schema={
                "match_id": pl.Utf8, "outcome": pl.Utf8, "model_prob": pl.Float64,
                "market_prob": pl.Float64, "ci_low": pl.Float64, "ci_high": pl.Float64,
                "edge": pl.Float64, "edge_bps": pl.Float64, "kelly_fraction": pl.Float64,
            }
        )
    else:
        empty = pl.DataFrame(
            schema={
                "match_id": pl.Utf8, "outcome": pl.Utf8, "model_prob": pl.Float64,
                "market_prob": pl.Float64, "edge": pl.Float64, "edge_bps": pl.Float64,
                "kelly_fraction": pl.Float64,
            }
        )
    if not table_exists(Table.MARKET_ODDS, settings):  # type: ignore[arg-type]
        logger.info("simulate.edges", status="no market_odds ingested; empty edges")  # type: ignore[attr-defined]
        return empty
    market = read_table(Table.MARKET_ODDS, settings)  # type: ignore[arg-type]
    matched = predictions.join(market, on="match_id", how="inner").select("match_id").unique()
    hint = None if matched.height else "no fixture match_ids joined market odds (check ids/aliases)"
    logger.info(  # type: ignore[attr-defined]
        "simulate.edges.coverage",
        fixtures=predictions.height,
        market_rows=market.height,
        matched_fixtures=matched.height,
        hint=hint,
    )
    if has_ci:
        edges = compute_credible_edges(
            predictions.select("match_id", "model_home", "model_draw", "model_away", *_CI_COLS),
            market,
        )
    else:
        edges = compute_edges(
            predictions.select("match_id", "model_home", "model_draw", "model_away"), market
        )
    logger.info("simulate.edges", status="computed", n=edges.height, credible=has_ci)  # type: ignore[attr-defined]
    return edges
