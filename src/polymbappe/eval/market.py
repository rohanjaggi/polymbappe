"""Market comparison utilities."""

from pydantic import BaseModel, ConfigDict


class Market(BaseModel):
    """Market probability snapshot for edge analysis."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    outcome: str
    market_price: float
    model_prob: float
    edge_bps: float
    kelly_fraction: float


def compare_model_to_market() -> None:
    """Compare model probabilities to market prices."""

    raise NotImplementedError
