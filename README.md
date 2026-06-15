# ⚽ Premier League Analytics Dashboard

A Squawka-style football analytics platform covering all 20 Premier League teams across **three seasons (2023/24 – 2025/26)**. Built to answer **"unlucky, or just bad?"** through Expected Goals (xG), and extended into player scouting, transfer analysis and squad comparison.

**Live demo:** _add your Streamlit Cloud URL_

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-app-red) ![Tests](https://img.shields.io/badge/tests-24%20passing-brightgreen) ![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Features

**League overview** — xG-based "deserved" league table, an attack-vs-defence league map, and league-wide leaderboards (most clinical finishers, top creators, best value). Exportable to CSV.

**Team analysis** — pick any club: actual vs expected points, match-by-match xG, rolling form, press intensity (PPDA), home/away splits, a **multi-season trajectory** (points & xG across three seasons), and a data-driven verdict.

**Player profiles** — goals vs xG, position-aware percentile radars (defenders judged on tackles/interceptions/aerials via FBref, forwards on finishing), head-to-head comparison, market value & contract status, and a league-wide similarity finder.

**Compare** — overlay any two teams, the same team across seasons, two players, or one player across seasons. Includes a **player development line** across all three seasons and a cumulative xPts "race".

**Transfer fit** — ranks league players by fit to a target system role using weighted per-90 z-scores, layered with Transfermarkt market value, contract years and age, plus a "fit per £m" value score. Exportable shortlist.

**Shot map** — shot-by-shot locations from Understat, sized by xG, with player hover detail and a per-player filter.

**Squad & roles** — unsupervised **k-means clustering** redefines players into data-driven style archetypes from their statistical fingerprint (with a PCA style-map), surfaces players who play unlike their listed position, and builds a best XI for any formation rated on position-appropriate percentiles.

**Models & insights** — the statistical layer: a regression showing how strongly xG difference predicts points (R²), a split-half analysis testing whether beating xG is repeatable skill or luck that regresses, an end-of-season points projection, and auto-generated narrative insights.

**Methodology** — an in-app explainer of every metric and model, demonstrating the analysis is understood, not just plotted.

---

## Tech stack

Python · pandas · NumPy · scikit-learn · Plotly · Streamlit · soccerdata (Understat + FBref) · BeautifulSoup (Transfermarkt) · pytest

---

## Engineering notes (for reviewers)

- **Machine learning:** k-means clustering + PCA assign data-driven player roles (`analysis/roles.py`).
- **Statistical modelling:** ordinary least-squares regression, split-half reliability analysis, and forecasting (`analysis/models.py`) — moving beyond description into inference.
- **Data validation:** `scripts/validate_data.py` runs quality gates (nulls, ranges, schema) on fetched data.
- **Tested:** a `pytest` suite (`tests/`) covers analytical invariants — percentile bounds, monotonic cumulative series, league-table integrity, and a regression test for NA market-value handling. Run `pytest -q`.
- **Cached:** expensive league-wide computations are memoised with Streamlit's `@st.cache_data` to keep interaction snappy.
- **Config-driven:** seasons, league and palette live in `config.py`.
- **Graceful degradation:** the dashboard runs even if optional data (shot events, market values) is absent.
- **Two data sources, fuzzy-joined:** Understat player names are matched to Transfermarkt via normalised token-set similarity (`analysis/matching.py`).

---

## Setup

```bash
pip install -r requirements.txt

python scripts/fetch_data.py            # matches + players, 3 seasons, all teams
python scripts/fetch_data.py --shots    # (optional) shot events for shot maps
python scripts/fetch_market_data.py     # Transfermarkt market values & contracts

python scripts/validate_data.py         # data-quality checks
streamlit run app.py                    # launch
pytest -q                               # run the test suite
```

Each fetch caches to `data/processed/`.

---

## Putting it on GitHub & going live

```bash
# from the project folder
git init
git add .
git commit -m "Premier League analytics dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/pl-analytics.git
git push -u origin main
```

Make sure `data/processed/*.parquet` is committed (it is by default) so the
deployed app has data without re-fetching. Then:

1. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub.
2. Pick the repo, set the main file to `app.py`, and deploy.
3. Copy the live URL into the badge/links at the top of this README.

The included GitHub Actions workflow (`.github/workflows/tests.yml`) runs the
test suite automatically on every push.

## Project structure

```
spurs-analytics/
├── app.py                      # Multi-page Streamlit dashboard
├── config.py                   # Seasons, league, palette
├── scripts/
│   ├── fetch_data.py           # Understat matches, players, shot events
│   ├── fetch_market_data.py    # Transfermarkt market values & contracts
│   └── validate_data.py        # Data-quality checks
├── analysis/
│   ├── xg.py                   # Match xG, xPts, verdict (any team)
│   ├── league_table.py         # xG-based deserved table
│   ├── players.py              # Per-90 metrics, position radars, similarity
│   ├── transfers.py            # System-fit scoring + value adjustment
│   ├── compare.py              # Team/player comparison engine
│   ├── leaderboards.py         # League-wide leaderboards
│   ├── models.py               # Regression, RTM analysis, projection, insights
│   ├── roles.py                # k-means role clustering + best-XI builder
│   ├── shotmap.py              # Shot-event preparation
│   ├── matching.py             # Fuzzy name matching (Understat ↔ Transfermarkt)
│   ├── shots.py                # Rolling trend helpers
│   └── home_away.py            # Venue splits
├── visuals/charts.py           # All Plotly figures
├── tests/                      # pytest suite
└── data/processed/             # Cached parquet (commit for deploy)
```

---

## How the models work

**xPts gap** — Understat's expected points minus actual points; a large negative gap = unlucky.

**Fit score** — per-90 metrics z-scored against positional peers, weighted by a role profile (`analysis/transfers.py`), squashed to 0–100 with a logistic.

**Similarity** — cosine similarity on z-scored per-90 feature vectors within a position group.

See the in-app **Methodology** page for the full write-up.

---

## Data sources

[Understat](https://understat.com) via [soccerdata](https://github.com/probberechts/soccerdata), and [Transfermarkt](https://www.transfermarkt.com). Free for personal and educational use.
