"""Core data schemas."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Team(BaseModel):
    """National team."""

    model_config = ConfigDict(extra="forbid")

    name: str
    fifa_code: str
    confederation: str | None = None


class Player(BaseModel):
    """Player entity."""

    model_config = ConfigDict(extra="forbid")

    name: str
    team: str
    club: str | None = None
    rating: float | None = None


class Squad(BaseModel):
    """Tournament squad."""

    model_config = ConfigDict(extra="forbid")

    team: str
    players: list[Player] = Field(default_factory=list)


class Match(BaseModel):
    """Match event."""

    model_config = ConfigDict(extra="forbid")

    match_id: str
    date: date
    home_team: str
    away_team: str
    home_goals: int = Field(ge=0)
    away_goals: int = Field(ge=0)
    competition: str
    is_knockout: bool = False
    neutral_site: bool = False
    group: str | None = None
    fair_play_home: int = 0
    fair_play_away: int = 0


class GroupStanding(BaseModel):
    """Single group table row."""

    model_config = ConfigDict(extra="forbid")

    group: str
    team: str
    points: int
    goal_difference: int
    goals_scored: int
    goals_against: int
    fair_play_score: int = 0
    lots_rank: int = 0


class KnockoutTie(BaseModel):
    """Knockout pairing."""

    model_config = ConfigDict(extra="forbid")

    round_name: Literal["R32", "R16", "QF", "SF", "THIRD", "FINAL"]
    home_team: str
    away_team: str
    pathway: Literal["A", "B"] | None = None
    slot: int | None = None


class EloSnapshot(BaseModel):
    """Team Elo rating at a point in time."""

    model_config = ConfigDict(extra="forbid")

    team: str
    date: date
    rating: float


class SquadValuation(BaseModel):
    """Transfermarkt squad market valuation for a tournament."""

    model_config = ConfigDict(extra="forbid")

    team: str
    tournament: str
    total_value: float = Field(ge=0.0)
    median_value: float = Field(ge=0.0)
    player_count: int = Field(ge=0)


class MarketOdds(BaseModel):
    """Overround-removed market-implied match probabilities."""

    model_config = ConfigDict(extra="forbid")

    match_id: str
    source: str
    home_win_prob: float = Field(ge=0.0, le=1.0)
    draw_prob: float = Field(ge=0.0, le=1.0)
    away_win_prob: float = Field(ge=0.0, le=1.0)
    timestamp: datetime


class PlayerStatus(BaseModel):
    """Availability status for a player, as tracked by the live agent."""

    model_config = ConfigDict(extra="forbid")

    player: str
    team: str
    status: Literal["fit", "doubt", "injured", "out"]
    last_updated: datetime
    source: str
    confidence: float = Field(ge=0.0, le=1.0)


class ManagerRecord(BaseModel):
    """Manager tournament pedigree record."""

    model_config = ConfigDict(extra="forbid")

    manager: str
    team: str
    tournament: str
    stage_reached: str
    knockout_matches: int = Field(default=0, ge=0)
    knockout_wins: int = Field(default=0, ge=0)


class SentimentSnapshot(BaseModel):
    """Per-team sentiment and overperformance signals at a point in time."""

    model_config = ConfigDict(extra="forbid")

    team: str
    date: date
    xg_overperformance: float | None = None
    reddit_score: float | None = None
    news_tone: float | None = None
