"""News / Reddit / RSS data sources for the Scan node (spec section 5.2).

Each fetcher degrades gracefully: optional dependencies (``feedparser``, ``praw``) and
network access may be absent (CI, offline), in which case the fetcher returns an empty
list rather than raising. Callers (the Scan node) can also be handed a pre-built list of
:class:`NewsItem` for deterministic testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

#: Default BBC Sport football RSS feed (current-only; spec 1.1).
BBC_FOOTBALL_RSS = "https://feeds.bbci.co.uk/sport/football/rss.xml"


@dataclass(slots=True)
class NewsItem:
    """A raw scanned news item."""

    source: str
    timestamp: datetime
    title: str
    snippet: str
    team: str | None = None


def fetch_rss(url: str = BBC_FOOTBALL_RSS, limit: int = 50) -> list[NewsItem]:
    """Fetch headlines from an RSS feed; empty list if feedparser/network unavailable."""

    try:
        import feedparser
    except ImportError:
        return []
    try:
        parsed = feedparser.parse(url)
    except Exception:  # noqa: BLE001 - network/parse errors degrade to empty
        return []
    items: list[NewsItem] = []
    for entry in parsed.get("entries", [])[:limit]:
        items.append(
            NewsItem(
                source="bbc_rss",
                timestamp=datetime.now(),
                title=entry.get("title", ""),
                snippet=entry.get("summary", ""),
            )
        )
    return items


def fetch_reddit(
    subreddit: str = "soccer", limit: int = 25, query: str | None = None
) -> list[NewsItem]:
    """Fetch recent r/soccer posts via PRAW; empty list if praw/credentials unavailable."""

    try:
        import praw  # noqa: F401
    except ImportError:
        return []
    try:  # pragma: no cover - requires live credentials
        import os

        reddit = praw.Reddit(
            client_id=os.environ.get("REDDIT_CLIENT_ID", ""),
            client_secret=os.environ.get("REDDIT_CLIENT_SECRET", ""),
            user_agent="polymbappe-agent/0.1",
        )
        sub = reddit.subreddit(subreddit)
        posts = sub.search(query, limit=limit) if query else sub.new(limit=limit)
        return [
            NewsItem(
                source="reddit",
                timestamp=datetime.fromtimestamp(p.created_utc),
                title=p.title,
                snippet=(p.selftext or "")[:500],
            )
            for p in posts
        ]
    except Exception:  # noqa: BLE001
        return []


def scan_sources(
    teams: list[str],
    *,
    rss_url: str = BBC_FOOTBALL_RSS,
    include_reddit: bool = False,
    injected: list[NewsItem] | None = None,
) -> list[NewsItem]:
    """Aggregate raw news items across sources (spec 5.2 Scan node).

    ``injected`` lets callers/tests supply items directly. Items whose title mentions a
    team name are tagged with that team.
    """

    items: list[NewsItem] = list(injected or [])
    items.extend(fetch_rss(rss_url))
    if include_reddit:
        items.extend(fetch_reddit())
    for item in items:
        if item.team is None:
            for team in teams:
                if team.lower() in item.title.lower() or team.lower() in item.snippet.lower():
                    item.team = team
                    break
    return items
