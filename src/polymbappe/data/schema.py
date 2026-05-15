"""Core data schemas."""

from datetime import date
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
