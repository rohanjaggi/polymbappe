# Data Ingestion Requirements Specification

## Purpose

This specification defines source-by-source ingestion requirements for the Polymbappe data pipeline, including package dependencies, authentication requirements, request headers, rate limits, expected output formats, and source-specific ingestion risks.

It also captures cross-cutting requirements that must be implemented before integrating source-specific pipelines.

---

## Match results & Elo

### Kaggle international results (martj42)
- **Packages:** `requests`, `polars` (via raw GitHub CSV path). No Kaggle auth required if using raw CSVs.
- **Auth:** None via GitHub raw. If using Kaggle dataset pages directly, require `kaggle`/`kagglehub` and API token at `~/.kaggle/kaggle.json` with `chmod 600`.
- **Headers / rate limit:** No meaningful constraints on `raw.githubusercontent.com`.
- **Format:** UTF-8 CSV files: `results.csv`, `shootouts.csv`, `goalscorers.csv`.
- **Primary gotcha:** Historical team names (for example, `West Germany`) must be normalized to a canonical key before joins. Preserve original source names in raw data for historical traceability, and store normalized names in standardized columns used for cross-source joins.

### EloRatings.net
- **Packages:** `requests`, `beautifulsoup4`, `lxml`.
- **Auth:** None.
- **Headers:** Use a realistic `User-Agent`.
- **Rate limit:** No published limit; self-throttle to ~1 request every 2–3 seconds.
- **Format:** HTML (plus JS-loaded backend data source).
- **Primary gotcha:** Ratings tables may be populated via JavaScript from a backend data blob; plain BeautifulSoup on landing pages can return empty tables.

---

## Player attributes

### EA FC / FIFA (stefanoleone992)
- **Packages:** `kagglehub` (preferred) or `kaggle` CLI; `polars`/`pandas`.
- **Auth:** Kaggle API token required.
- **Format:** CSV (`male_players.csv` in FC24 is very large).
- **Primary gotchas:**
  - `sofifa_id` renamed to `player_id` in FC24; reconcile across editions.
  - Explicitly map FIFA editions to tournament eras (for example, FIFA 22 ↔ 2022 World Cup squads).

### FM data (FM25 / alternatives)
- **Packages:** `kagglehub`/`kaggle` for Kaggle sources, or `requests` for raw GitHub sources once URL is validated.
- **Auth:** Kaggle token if Kaggle-sourced.
- **Format:** CSV.
- **Primary gotchas:**
  - Confirm source validity before coding against unverified repositories.
  - FM attribute names are not standardized across exports and require column reconciliation against EA FC schema.

### FBref / StatsBomb via soccerdata
- **Packages:** `soccerdata` (and transitive `pandas`, `lxml`, `requests-cache`).
- **Python:** 3.9+.
- **Auth:** None for FBref.
- **Rate limit:** FBref throttles aggressively. Keep soccerdata delays and disk cache enabled.
- **Format:** pandas DataFrame (convert via `pl.from_pandas` as needed).
- **Primary gotcha:** FBref page structure changes can break scrapers; pin known-good `soccerdata` versions and monitor breakage.

---

## Market / odds

### Polymarket
- **Packages:** `requests` for read-only market-implied probability ingestion. `py-clob-client` only if order placement is needed.
- **Auth:** None for public read endpoints (Gamma + CLOB market data).
- **Rate limits (read-only):**
  - CLOB market-data (`/book`, `/price`, `/midprice`): **1,500 requests / 10 seconds**
  - Gamma `/markets`: **300 requests / 10 seconds**
  - Gamma `/events`: **500 requests / 10 seconds**
  - Limits are Cloudflare-enforced and over-limit traffic is delayed/queued.
  - CLOB responses include rate-limit headers; use them for adaptive pacing.
- **Format:** JSON.
- **Primary gotcha:** Discover markets in Gamma, then query CLOB prices using **token ID** (`clobTokenIds`), not condition ID.

### Football-Data.co.uk
- **Packages:** `requests` + `polars`/`pandas` (or via `soccerdata`).
- **Auth:** None (free source).
- **Format:** Per-league, per-season CSVs.
- **Primary gotcha:** Schema drift across seasons and inconsistent bookmaker columns; parser must select by column name rather than index.

---

## Sentiment

### Reddit (PRAW)
- **Packages:** `praw`.
- **Auth:** Reddit app credentials: `client_id`, `client_secret`, descriptive `user_agent`. Username/password not required for read-only public content.
- **Rate limit:** OAuth approximately 100 requests/minute; PRAW handles throttling/backoff.
- **Format:** PRAW objects (`Submission`, `Comment`) mapped to dictionaries/tables.
- **Primary gotcha:** Very large match threads can exhaust quota quickly; cap comment depth/count.

### BBC Sport RSS
- **Packages:** `feedparser` (and optionally `requests`), plus an LLM client for classification.
- **Auth:** None for RSS feed; classifier API key required for downstream LLM enrichment.
- **Rate limit:** No meaningful feed-side constraints; poll on intervals and cache results.
- **Format:** RSS/XML parsed entries.
- **Primary gotcha:** Feed usually contains headline + summary only, not full article text.

---

## Structure / venues / managers

### Venue + schedule (openfootball)
- **Packages:** `requests` + stdlib `json`.
- **Auth:** None (CC0 source).
- **Format:** Raw JSON from `raw.githubusercontent.com`.
- **Primary gotcha:** Stadium data may omit coordinates; provide separate geocoding/static city→coordinate enrichment.

### Transfermarkt
- **Packages:** `requests`, `beautifulsoup4`, `lxml` (or pre-scraped datasets/wrappers where feasible).
- **Auth:** None.
- **Headers:** Use browser-like `User-Agent`, `Accept-Language`, and `Accept` to avoid `403` responses.
- **Rate limit:** Aggressive anti-bot protections; self-throttle, cache heavily, and expect possible IP blocking.
- **Format:** HTML.
- **Primary gotcha:** Market values are localized strings (`€80.00m`, `€500k`) and must be parsed to normalized numeric units.

### Manager career history (Wikipedia)
- **Packages:** `requests` + `beautifulsoup4`, or MediaWiki API (`.../w/api.php`), or `wikipedia-api`.
- **Auth:** None.
- **Headers:** MediaWiki expects a descriptive `User-Agent` with application/contact context.
- **Rate limit:** Follow etiquette with a minimum 1-second delay between requests (no bursts).
- **Format:** HTML or JSON (API).
- **Primary gotcha:** No canonical knockout-record field exists; derive from manager tenure windows joined against match results.

---

## Consolidated dependency set

```text
polars
requests
beautifulsoup4
lxml
soccerdata
kagglehub
praw
feedparser
py-clob-client   # only if placing Polymarket orders later
anthropic        # BBC headline classifier client
```

---

## Cross-cutting requirements (must-build first)

1. **Team-name normalization map**
   - Different sources use different national team naming conventions.
   - This mapping is the core join backbone across results, ratings, squads, and markets.

2. **Shared on-disk request cache**
   - Persisted caching is required to avoid repeated fetches and reduce rate-limit/blocking risk for FBref and Transfermarkt.
   - Cache must persist across runs and be shared across ingestion jobs.
