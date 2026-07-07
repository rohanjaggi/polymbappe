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
    r32_matchup_counts: dict[tuple[str, str], int]

    def stage_probabilities(
        self, eliminated: set[str] | None = None
    ) -> pl.DataFrame:
        """Per-team stage-reaching probabilities.

        When ``eliminated`` is supplied, those teams are zeroed out and the
        remaining probabilities are renormalised so each stage sums to 1.
        """

        rows = [
            {"team": team, **{s: counts.get(s, 0) / self.n_sims for s in STAGES}}
            for team, counts in self.stage_counts.items()
        ]
        df = pl.DataFrame(rows)
        if eliminated:
            mask = pl.col("team").is_in(list(eliminated))
            df = df.with_columns(
                [pl.when(mask).then(0.0).otherwise(pl.col(s)).alias(s) for s in STAGES]
            )
            for s in STAGES:
                total = df[s].sum()
                if total > 0:
                    df = df.with_columns((pl.col(s) / total).alias(s))
        return df.sort("champion", descending=True)

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


def build_eliminated_teams(
    matches: pl.DataFrame,
    schedule: pl.DataFrame | None = None,
) -> set[str]:
    """Derive the set of teams eliminated from WC 2026 knockout rounds.

    A team that played a knockout match but does not appear in any future scheduled
    fixture is eliminated. Uses the upcoming schedule (null-score fixtures from the
    upstream feed) as the source of truth for who is still alive.
    """

    import structlog

    logger = structlog.get_logger(__name__)
    ko = matches.filter(
        (pl.col("competition") == "FIFA World Cup")
        & pl.col("is_knockout")
        & (pl.col("date") >= WC2026_START)
    )
    if ko.is_empty():
        return set()

    # All teams that have played at least one knockout match.
    ko_teams = set(ko["home_team"].to_list() + ko["away_team"].to_list())

    # Teams in upcoming scheduled fixtures are still alive.
    alive_in_schedule: set[str] = set()
    if schedule is not None and not schedule.is_empty():
        future = schedule.filter(
            (pl.col("competition") == "FIFA World Cup")
            & (pl.col("date") >= WC2026_START)
        )
        alive_in_schedule = set(
            future["home_team"].to_list() + future["away_team"].to_list()
        )

    # A team is eliminated if it entered the knockout rounds but has no future fixture.
    eliminated = ko_teams - alive_in_schedule

    logger.info("simulate.eliminated_teams", count=len(eliminated), teams=sorted(eliminated))
    return eliminated


def build_played_group_results(
    matches: pl.DataFrame, structure: TournamentStructure
) -> dict[str, dict[frozenset[str], dict[str, int]]]:
    """Map already-played 2026 group-stage results onto their groups.

    Returns ``group -> {frozenset(home, away): {team: goals}}`` so the Monte Carlo can lock
    in real scorelines instead of re-sampling them. The ingested results carry no group tag,
    so each match is assigned to a group by team membership in ``structure.groups``; a match
    whose two teams are not both in one configured group (a name mismatch, or a knockout
    fixture) is skipped — degrading to "simulate it" rather than corrupting a table.
    """

    import structlog

    logger = structlog.get_logger(__name__)
    team_group = {t: g for g, members in structure.groups.items() for t in members}
    played: dict[str, dict[frozenset[str], dict[str, int]]] = {}
    if matches.is_empty():
        return played
    wc = matches.filter(
        (pl.col("competition") == "FIFA World Cup")
        & (~pl.col("is_knockout"))
        & (pl.col("date") >= WC2026_START)
    )
    skipped = 0
    for r in wc.sort("date").iter_rows(named=True):
        home, away = r["home_team"], r["away_team"]
        group = team_group.get(home)
        if group is None or team_group.get(away) != group:
            skipped += 1
            continue
        played.setdefault(group, {})[frozenset((home, away))] = {
            home: int(r["home_goals"]),
            away: int(r["away_goals"]),
        }
    logger.info(
        "simulate.played_results",
        locked=sum(len(v) for v in played.values()),
        skipped=skipped,
    )
    return played


def _simulate_group(
    group: str,
    teams: list[str],
    model: StrengthModel,
    latent: _LatentStrength,
    rng: np.random.Generator,
    context_hook: ContextHook | None = None,
    played: dict[frozenset[str], dict[str, int]] | None = None,
) -> list[GroupStanding]:
    """Play one group's six matches and return the resolved standings.

    Fixtures present in ``played`` (real, ingested results) are locked to their actual
    scoreline instead of being sampled, so mid-tournament standings reflect what has really
    happened; the remaining fixtures are simulated as usual. A locked result still feeds the
    latent-strength update, so real group form propagates into the knockout rounds.
    """

    matches: list[Match] = []
    for k, (i, j, _matchday) in enumerate(_GROUP_SCHEDULE):
        home, away = teams[i], teams[j]
        dh, da = latent.get(home), latent.get(away)
        lam, mu = model.rates(home, away, neutral=True, dh=dh, da=da)
        locked = played.get(frozenset((home, away))) if played else None
        if locked is not None:
            hg, ag = locked[home], locked[away]
        else:
            matrix = _contextualize(
                model.score_matrix(home, away, neutral=True, dh=dh, da=da),
                home,
                away,
                context_hook,
            )
            hg, ag = sample_scoreline(matrix, rng)
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
    played_results: dict[str, dict[frozenset[str], dict[str, int]]] | None = None,
) -> SimulationResult:
    """Run the full Monte Carlo tournament simulation.

    ``context_hook`` (optional) applies a per-match contextual H/D/A adjustment by
    reweighting each score matrix (spec 4.1 per-match contextual injection).

    ``played_results`` (optional, keyed by group from :func:`build_played_group_results`)
    locks already-played group fixtures to their real scoreline, so a mid-tournament re-run
    reflects the current standings instead of re-rolling the whole group stage.
    """

    rng = rng or np.random.default_rng()
    teams = structure.teams
    stage_counts: dict[str, dict[str, int]] = {t: {} for t in teams}
    group_finish: dict[str, dict[int, int]] = {t: {} for t in teams}
    matchup_counts: dict[tuple[str, str], int] = {}
    # The winner of each round reaches the next stage (R16 round -> QF teams, etc.).
    later_rounds = ("QF", "SF", "FINAL", "champion")

    for _ in range(n_sims):
        latent = _LatentStrength(learning_rate)
        winners: list[GroupStanding] = []
        runners_up: list[str] = []
        thirds: list[GroupStanding] = []
        for group, members in structure.groups.items():
            standings = _simulate_group(
                group, members, model, latent, rng, context_hook,
                played=played_results.get(group) if played_results else None,
            )
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
            key = (tie.home_team, tie.away_team)
            matchup_counts[key] = matchup_counts.get(key, 0) + 1

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
        n_sims=n_sims,
        stage_counts=stage_counts,
        group_finish_counts=group_finish,
        r32_matchup_counts=matchup_counts,
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


def compute_knockout_fixture_predictions(
    model: StrengthModel,
    matches: pl.DataFrame,
    context_hook: ContextHook | None = None,
    exclude_pairs: set[tuple[str, str]] | None = None,
) -> pl.DataFrame:
    """H/D/A predictions for known knockout fixtures (played + scheduled).

    Reads the ingested matches table for WC 2026 knockout fixtures and generates
    model predictions in the same schema as :func:`compute_match_predictions`.
    Pairings in ``exclude_pairs`` (e.g. group-stage fixtures misclassified as
    knockout) are skipped to avoid duplicates.
    """

    _empty = pl.DataFrame(
        schema={
            "match_id": pl.Utf8, "group": pl.Utf8, "home_team": pl.Utf8,
            "away_team": pl.Utf8, "model_home": pl.Float64, "model_draw": pl.Float64,
            "model_away": pl.Float64, "exp_home_goals": pl.Float64,
            "exp_away_goals": pl.Float64,
        }
    )

    ko = matches.filter(
        (pl.col("competition") == "FIFA World Cup")
        & (pl.col("is_knockout"))
        & (pl.col("date") >= WC2026_START)
    )
    if ko.is_empty():
        return _empty

    exclude = exclude_pairs or set()
    grid = np.arange(model.max_goals + 1)
    rows: list[dict[str, object]] = []
    seen: set[frozenset[str]] = set()
    for r in ko.iter_rows(named=True):
        home, away = r["home_team"], r["away_team"]
        pair = frozenset((home, away))
        if pair in seen:
            continue
        if (home, away) in exclude or (away, home) in exclude:
            continue
        seen.add(pair)
        matrix = _contextualize(
            model.score_matrix(home, away, neutral=True), home, away, context_hook
        )
        h, d, a = hda_marginals(matrix)
        rows.append({
            "match_id": f"2026__{home}__{away}",
            "group": "KO",
            "home_team": home,
            "away_team": away,
            "model_home": h,
            "model_draw": d,
            "model_away": a,
            "exp_home_goals": float((matrix.sum(axis=1) * grid).sum()),
            "exp_away_goals": float((matrix.sum(axis=0) * grid).sum()),
        })
    return pl.DataFrame(rows) if rows else _empty


# -- staleness detection (spec 4.5) -------------------------------------------


def surprise_increment(predicted_prob: float, occurred: bool) -> float:
    """Per-match surprise: ``|actual - predicted|`` for the realized outcome."""

    return abs((1.0 if occurred else 0.0) - predicted_prob)


def _load_scheduled_fixtures(settings: object, logger: object) -> pl.DataFrame | None:
    """Load upcoming (null-score) WC fixtures from the raw results CSV.

    The upstream Kaggle mirror includes scheduled fixtures before scores are filled in.
    Normalization drops these, but they're useful for inferring knockout-draw winners
    (a team scheduled in a later round must have advanced).
    """

    import io
    from pathlib import Path

    raw_path = Path(getattr(settings, "raw_data_dir", "data/raw")) / "results.csv"
    try:
        if raw_path.exists():
            raw = pl.read_csv(io.BytesIO(raw_path.read_bytes()), null_values=["NA"])
        else:
            from polymbappe.data.sources import KAGGLE_RESULTS_RAW_URL
            import requests
            resp = requests.get(KAGGLE_RESULTS_RAW_URL, timeout=30)
            resp.raise_for_status()
            raw = pl.read_csv(io.BytesIO(resp.content), null_values=["NA"])
    except Exception:
        return None

    score_col = "home_score" if "home_score" in raw.columns else "home_goals"
    scheduled = raw.filter(pl.col(score_col).is_null())
    if scheduled.is_empty():
        return None

    rename = {"home_score": "home_goals", "away_score": "away_goals"}
    rename = {k: v for k, v in rename.items() if k in scheduled.columns}
    if rename:
        scheduled = scheduled.rename(rename)

    from polymbappe.data.aliases import normalize_team_expr

    return scheduled.with_columns(
        pl.col("date").cast(pl.Utf8).str.to_date(strict=False).alias("date"),
        normalize_team_expr("home_team").alias("home_team"),
        normalize_team_expr("away_team").alias("away_team"),
        pl.col("tournament").cast(pl.Utf8).alias("competition"),
    )


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
    historical_context: bool = False,
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
    if with_context or historical_context:
        context_hook = _load_context_hook(
            settings, structure, elo, logger, historical=historical_context
        )

    # Lock already-played 2026 results so the sim reflects the live tournament.
    matches_df = read_table(Table.MATCHES, settings) if table_exists(Table.MATCHES, settings) else pl.DataFrame()
    played_results = (
        build_played_group_results(matches_df, structure)
        if not matches_df.is_empty()
        else None
    )
    # Build the set of teams eliminated from knockout rounds so they can't advance.
    scheduled_fixtures = _load_scheduled_fixtures(settings, logger)
    eliminated_teams = (
        build_eliminated_teams(matches_df, schedule=scheduled_fixtures)
        if not matches_df.is_empty()
        else None
    )

    result = simulate_tournament(
        structure, model, n_sims=n_sims, context_hook=context_hook,
        played_results=played_results,
    )
    predictions = compute_match_predictions(
        structure, model, context_hook, bayesian_model=bayesian_model
    )

    # Append predictions for known knockout fixtures (played + scheduled),
    # excluding any pairing already covered by the group-stage predictions.
    if not matches_df.is_empty():
        group_pairs = set(
            zip(predictions["home_team"].to_list(), predictions["away_team"].to_list())
        )
        ko_preds = compute_knockout_fixture_predictions(
            model, matches_df, context_hook, exclude_pairs=group_pairs
        )
        if not ko_preds.is_empty():
            predictions = pl.concat(
                [predictions, ko_preds.select(predictions.columns)], how="diagonal_relaxed"
            )

    settings.outputs_data_dir.mkdir(parents=True, exist_ok=True)
    out = settings.outputs_data_dir
    result.stage_probabilities(eliminated=eliminated_teams).write_parquet(out / "stage_probabilities.parquet")
    result.group_probabilities().write_parquet(out / "group_probabilities.parquet")
    predictions.write_parquet(out / "match_predictions.parquet")
    from polymbappe.simulate.knockout_predictions import compute_knockout_predictions
    compute_knockout_predictions(result.r32_matchup_counts, n_sims, model).write_parquet(
        out / "knockout_predictions.parquet"
    )
    # Knockout bracket forecast, anchored to the *real* draw: the ingested schedule gives the
    # bracket structure (R32/R16 fixtures + W##/L## slot placeholders) and the matches table
    # locks results already played, so future rounds are projected from the current bracket.
    from polymbappe.simulate.knockout_bracket import compute_knockout_bracket

    schedule_df = (
        read_table(Table.SCHEDULE, settings) if table_exists(Table.SCHEDULE, settings)
        else pl.DataFrame()
    )
    compute_knockout_bracket(schedule_df, matches_df, model, structure).write_parquet(
        out / "knockout_bracket.parquet"
    )
    # Refresh odds AFTER fixtures are written so Polymarket can align to them, then edges.
    if refresh_odds or live:
        refresh_market_odds(settings, logger)
    _write_edges(predictions, settings, logger).write_parquet(out / "edges.parquet")
    print(result.stage_probabilities(eliminated=eliminated_teams).head(15))


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


def _wc2026_team_travel(settings: object) -> dict[str, float]:
    """Mean per-match travel km for each WC2026 team, derived from the ingested schedule.

    Computes haversine city-to-city travel across each team's group-stage matches (sorted by
    date) using the ingested ``city_coords`` gazetteer. Returns ``{team: mean_travel_km}``.
    Falls back to an empty dict (all-zero default) when schedule or coords are missing.
    """

    from polymbappe.context.fatigue import build_city_coord_lookup, build_match_travel_features
    from polymbappe.data.store import read_table, table_exists
    from polymbappe.data.tables import Table

    if not (
        table_exists(Table.SCHEDULE, settings)  # type: ignore[arg-type]
        and table_exists(Table.CITY_COORDS, settings)  # type: ignore[arg-type]
    ):
        return {}

    schedule = read_table(Table.SCHEDULE, settings)  # type: ignore[arg-type]
    city_coords = read_table(Table.CITY_COORDS, settings)  # type: ignore[arg-type]
    coords = build_city_coord_lookup(city_coords)
    if not coords:
        return {}

    group_fixtures = schedule.filter(
        pl.col("stage").str.contains("Matchday") & pl.col("city").is_not_null()
    )
    if group_fixtures.is_empty():
        return {}

    travel = build_match_travel_features(group_fixtures, coords)
    mean_travel = (
        travel.group_by("team")
        .agg(pl.col("travel_km").mean().alias("mean_travel_km"))
    )
    return {r["team"]: float(r["mean_travel_km"]) for r in mean_travel.iter_rows(named=True)}


def _context_feature_frame(
    teams: list[str], ctx: FixtureContext, team_travel: dict[str, float] | None = None
) -> tuple[list[tuple[str, str]], pl.DataFrame]:
    """Per-ordered-pair contextual feature frame for the live teams (testable seam).

    Returns the ordered ``(home, away)`` pairs and the matching feature frame whose columns
    are exactly :data:`~polymbappe.context.runtime.SIM_CONTEXT_FEATURES`, built through the
    same :func:`~polymbappe.context.runtime.fixture_feature_row` the fit path uses.

    ``team_travel`` maps each team to its mean within-tournament travel km (computed from
    the WC2026 schedule). When provided, it is passed as ``home_travel_km`` /
    ``away_travel_km`` so the contextual adjuster sees realistic travel values rather than
    the all-zero default — preventing the adjuster from incorrectly treating every away team
    as "local" (0 km) and inflating their win probability.
    """

    from polymbappe.context.runtime import fixture_feature_row

    travel = team_travel or {}
    pairs = [(h, a) for h in teams for a in teams if h != a]
    rows = [
        fixture_feature_row(
            h, a, ctx,
            home_travel_km=travel.get(h, 0.0),
            away_travel_km=travel.get(a, 0.0),
        )
        for h, a in pairs
    ]
    return pairs, pl.DataFrame(rows)


def _load_context_hook(
    settings: object,
    structure: TournamentStructure,
    elo: dict[str, float] | None,
    logger: object,
    *,
    historical: bool = False,
) -> ContextHook | None:
    """Build a per-match contextual hook from live adaptive weights.

    By default (``historical=False``) only the adaptive hook is used: weights earned from
    live WC2026 results via ``contextual-monitor --apply``. When no weights are active yet
    (early in the tournament, or before the first ``--apply`` run), returns ``None`` so the
    simulation runs without any contextual adjustment — which is correct, because the
    historically-trained adjuster is known to hurt the LOTO backtest (contextual features
    are near-zero for all pre-2026 tournaments) and should not fire by default.

    Pass ``historical=True`` (via ``simulate --historical-context``) to fall back to the
    LightGBM residual adjuster artifact from ``train`` when no adaptive weights are active.
    This is preserved as an explicit opt-in for diagnostic use.
    """

    from polymbappe.context.adaptive import load_adaptive_weights
    from polymbappe.data.store import table_exists
    from polymbappe.data.tables import Table

    if not table_exists(Table.MATCHES, settings):  # type: ignore[arg-type]
        logger.warning("simulate.context", status="no matches table for features")  # type: ignore[attr-defined]
        return None

    ctx = _live_fixture_context(settings, elo, logger)
    team_travel = _wc2026_team_travel(settings)

    # Adaptive hook: weights earned from live WC2026 results (default path).
    adaptive_state = load_adaptive_weights(settings)
    if adaptive_state.is_active():
        from polymbappe.context.wc2026_hook import build_adaptive_hook

        adaptive_hook = build_adaptive_hook(adaptive_state, structure.teams, ctx, team_travel)
        if adaptive_hook is not None:
            active = [g for g, w in adaptive_state.weights.items() if w != 0.0]
            logger.info(  # type: ignore[attr-defined]
                "simulate.context",
                status="adaptive",
                active_groups=active,
                n_matches=adaptive_state.n_matches,
            )
            return adaptive_hook  # type: ignore[return-value]

    # No adaptive weights yet. Use historical adjuster only when explicitly requested.
    if not historical:
        logger.info("simulate.context", status="no_adaptive_weights; run contextual-monitor --apply")  # type: ignore[attr-defined]
        return None

    from polymbappe.context.adjuster import apply_adjustment
    from polymbappe.models.train import load_artifact

    try:
        adjuster = load_artifact("contextual_adjuster", settings)  # type: ignore[arg-type]
    except FileNotFoundError:
        logger.warning("simulate.context", status="no adjuster artifact; run `train`")  # type: ignore[attr-defined]
        return None

    pairs, frame = _context_feature_frame(structure.teams, ctx, team_travel=team_travel)
    raw = adjuster.predict_adjustment(frame)  # type: ignore[attr-defined]
    cache = {pair: raw[i] for i, pair in enumerate(pairs)}
    cap = adjuster.config.cap  # type: ignore[attr-defined]
    logger.info("simulate.context", status="historical_adjuster", pairs=len(pairs))  # type: ignore[attr-defined]

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
