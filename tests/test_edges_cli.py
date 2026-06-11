"""Tests for the `polymbappe edges` entrypoint (compare_model_to_market)."""

from __future__ import annotations

import polars as pl
import pytest

from polymbappe.eval.market import compare_model_to_market

_EDGES_SCHEMA = {
    "match_id": pl.Utf8, "outcome": pl.Utf8, "model_prob": pl.Float64,
    "market_prob": pl.Float64, "edge": pl.Float64, "edge_bps": pl.Float64,
    "kelly_fraction": pl.Float64,
}


def _outputs(tmp_path):
    out = tmp_path / "data" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_reads_precomputed_edges(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    pl.DataFrame(
        {
            "match_id": ["2026__A__B"], "outcome": ["H"], "model_prob": [0.6],
            "market_prob": [0.5], "edge": [0.1], "edge_bps": [1000.0], "kelly_fraction": [0.2],
        }
    ).write_parquet(_outputs(tmp_path) / "edges.parquet")
    compare_model_to_market()
    assert "2026__A__B" in capsys.readouterr().out


def test_empty_edges_gives_guidance(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    pl.DataFrame(schema=_EDGES_SCHEMA).write_parquet(_outputs(tmp_path) / "edges.parquet")
    compare_model_to_market()
    assert "No market edges" in capsys.readouterr().out


def test_missing_prerequisites_named(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _outputs(tmp_path)  # exists but empty (no edges.parquet, no predictions, no odds)
    with pytest.raises(FileNotFoundError) as exc:
        compare_model_to_market()
    msg = str(exc.value)
    assert "match_predictions.parquet" in msg and "market_odds" in msg
