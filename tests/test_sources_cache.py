from pathlib import Path

from polymbappe.config import Settings
from polymbappe.data.sources import (
    _BROWSER_HEADERS,
    _throttle,
    cached_get,
    http_cache_dir,
)
from polymbappe.data.tables import TABLE_COLUMNS, Table, table_path


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path)


# --- new tables -------------------------------------------------------------


def test_squads_table_registered() -> None:
    assert Table.SQUADS == "squads"
    assert TABLE_COLUMNS[Table.SQUADS] == ("team", "tournament", "player", "club", "age")


def test_manager_records_table_registered() -> None:
    assert Table.MANAGER_RECORDS == "manager_records"
    assert TABLE_COLUMNS[Table.MANAGER_RECORDS] == (
        "manager",
        "team",
        "tournament",
        "stage_reached",
        "knockout_matches",
        "knockout_wins",
        "tournament_order",
    )


def test_new_table_paths_resolve(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    processed = settings.processed_data_dir
    assert table_path(Table.SQUADS, settings) == processed / "squads.parquet"
    assert table_path(Table.MANAGER_RECORDS, settings) == processed / "manager_records.parquet"


# --- HTTP cache -------------------------------------------------------------


class _StubResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:  # pragma: no cover - trivial
        return None


def test_cached_get_misses_then_hits(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    calls = {"n": 0}

    def fetcher(url: str, **kwargs: object) -> _StubResponse:
        calls["n"] += 1
        return _StubResponse(b"payload")

    url = "https://example.com/page?team=BRA"

    first = cached_get(url, settings=settings, min_interval=0, _fetcher=fetcher)
    assert first == b"payload"
    assert calls["n"] == 1

    # A file landed under the cache dir.
    cache_dir = http_cache_dir(settings)
    cached_files = list(cache_dir.glob("*.bin"))
    assert len(cached_files) == 1

    # Second call for the same URL is served from disk: no re-fetch.
    second = cached_get(url, settings=settings, min_interval=0, _fetcher=fetcher)
    assert second == b"payload"
    assert calls["n"] == 1


def test_cached_get_force_refresh_refetches(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    calls = {"n": 0}

    def fetcher(url: str, **kwargs: object) -> _StubResponse:
        calls["n"] += 1
        return _StubResponse(b"v%d" % calls["n"])

    url = "https://example.com/page"
    cached_get(url, settings=settings, min_interval=0, _fetcher=fetcher)
    out = cached_get(
        url, settings=settings, min_interval=0, force_refresh=True, _fetcher=fetcher
    )
    assert calls["n"] == 2
    assert out == b"v2"


def test_cached_get_sends_browser_headers(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    captured: dict[str, object] = {}

    def fetcher(url: str, **kwargs: object) -> _StubResponse:
        captured.update(kwargs)
        return _StubResponse(b"x")

    cached_get(
        "https://example.com/x", settings=settings, min_interval=0, _fetcher=fetcher
    )
    assert captured["headers"] == _BROWSER_HEADERS


def test_throttle_no_sleep_with_zero_interval() -> None:
    # Must be callable with min_interval=0 without sleeping.
    _throttle("throttle-test.example", 0)
