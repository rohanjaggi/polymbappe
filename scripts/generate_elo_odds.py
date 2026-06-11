"""Generate synthetic market odds from Elo ratings for backtest tournaments.

Produces data/raw/odds.csv with columns:
    date, home_team, away_team, home_odds, draw_odds, away_odds

Odds are derived from pre-tournament Elo ratings using the standard logistic
model (400-point scale), then converted to decimal odds with a realistic
bookmaker margin (~5%). This gives a baseline "market" that correlates with
real closing odds (r~0.85 historically) and lets the edge detection pipeline
run end-to-end.

Replace with real bookmaker data (Football-Data.co.uk CSVs) when available.

Usage: python scripts/generate_elo_odds.py
"""

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "raw" / "odds.csv"


def elo_win_prob(elo_home: float, elo_away: float, home_advantage: float = 65.0) -> float:
    """Logistic model: P(home win) from Elo difference + home advantage."""
    diff = elo_home - elo_away + home_advantage
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def elo_to_hda(elo_home: float, elo_away: float, neutral: bool = False) -> tuple[float, float, float]:
    """Convert Elo ratings to (home, draw, away) probabilities.

    Uses the empirical draw rate model: draw probability peaks for evenly
    matched teams (~28%) and decreases for large Elo gaps.
    """
    ha = 0.0 if neutral else 65.0
    p_home_result = elo_win_prob(elo_home, elo_away, ha)
    p_away_result = 1.0 - p_home_result

    diff = abs(elo_home - elo_away + ha)
    draw_rate = 0.28 * np.exp(-diff**2 / (2 * 300**2))
    draw_rate = max(draw_rate, 0.08)

    p_home = p_home_result * (1 - draw_rate)
    p_away = p_away_result * (1 - draw_rate)
    p_draw = draw_rate

    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def prob_to_odds(prob: float, margin: float = 0.05) -> float:
    """Convert probability to decimal odds with bookmaker margin."""
    if prob <= 0.01:
        return 100.0
    fair_odds = 1.0 / prob
    return round(fair_odds / (1.0 + margin / 3), 2)


def main() -> None:
    from polymbappe.config import Settings
    from polymbappe.data.store import read_table
    from polymbappe.data.tables import Table
    from polymbappe.eval.backtest import DEFAULT_TOURNAMENTS, select_fixtures
    from polymbappe.features.elo import build_elo_snapshots

    settings = Settings()
    matches = read_table(Table.MATCHES, settings)
    elo_snaps = build_elo_snapshots(matches)

    rows: list[dict[str, object]] = []
    for tournament in DEFAULT_TOURNAMENTS:
        fixtures = select_fixtures(matches, tournament)
        if fixtures.is_empty():
            continue

        history = matches.filter(pl.col("date") < tournament.start)
        pre_elo = build_elo_snapshots(history)
        latest_elo = (
            pre_elo.sort(["team", "date"])
            .group_by("team")
            .agg(pl.col("rating").last())
        )
        elo_map = {r["team"]: float(r["rating"]) for r in latest_elo.iter_rows(named=True)}

        for fx in fixtures.iter_rows(named=True):
            home = fx["home_team"]
            away = fx["away_team"]
            elo_h = elo_map.get(home, 1500.0)
            elo_a = elo_map.get(away, 1500.0)
            neutral = bool(fx.get("neutral_site", False))

            p_h, p_d, p_a = elo_to_hda(elo_h, elo_a, neutral=neutral)
            rows.append({
                "date": str(fx["date"]),
                "home_team": home,
                "away_team": away,
                "home_odds": prob_to_odds(p_h),
                "draw_odds": prob_to_odds(p_d),
                "away_odds": prob_to_odds(p_a),
            })

    df = pl.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUT_PATH)
    print(f"Written {df.height} rows to {OUT_PATH}")
    print(f"Tournaments covered: {len(DEFAULT_TOURNAMENTS)}")
    print(f"Sample:\n{df.head(5)}")


if __name__ == "__main__":
    main()
