"""
analysis/models.py
------------------
The analytical/statistical layer — moves the project beyond description into
inference. Three pillars that rigorously underpin "unlucky or just bad?":

1. xG difference predicts points        -> is xG meaningful? (it is, strongly)
2. Finishing overperformance doesn't persist -> is beating xG skill or luck?
3. End-of-season points projection      -> simple forecasting from xPts/game

All stats are computed with numpy only (no sklearn) and the methods are simple
enough to explain in an interview.
"""

import numpy as np
import pandas as pd

from analysis.league_table import build_tables

FULL_SEASON_GAMES = 38


# ── Per-(team, season) aggregates across all seasons ───────────────────────
def team_season_aggregates(matches_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in matches_all["season_label"].unique():
        t = build_tables(matches_all, season)
        t = t.assign(season=season)
        rows.append(t)
    agg = pd.concat(rows, ignore_index=True)
    agg["finishing"] = agg["gf"] - agg["xgf"]          # goals minus xG (attack)
    agg["defending"] = agg["xga"] - agg["ga"]          # xGA minus goals against
    return agg


# ── Simple OLS helper (returns slope, intercept, r, r²) ────────────────────
def ols_fit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"slope": np.nan, "intercept": np.nan, "r": np.nan, "r2": np.nan, "n": len(x)}
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else np.nan
    return {"slope": float(slope), "intercept": float(intercept),
            "r": r, "r2": float(r**2) if not np.isnan(r) else np.nan, "n": int(len(x))}


# ── 1. Does xG difference predict points? ──────────────────────────────────
def xgd_points_model(matches_all: pd.DataFrame):
    agg = team_season_aggregates(matches_all)
    fit = ols_fit(agg["xgd"], agg["points"])
    return agg, fit


# ── 2. Does finishing overperformance persist? (split-half) ────────────────
def finishing_persistence(matches_all: pd.DataFrame):
    """
    For each complete team-season, split games by date into first/second half,
    compute finishing overperformance (goals - xG) per game in each half, then
    correlate. Low correlation => overperformance is largely non-repeatable (luck).
    """
    from analysis.xg import get_xg_summary, list_teams
    pts = []
    for season in matches_all["season_label"].unique():
        for team in list_teams(matches_all):
            s = get_xg_summary(matches_all, team, season)
            if len(s) < 10:                 # need enough games to split
                continue
            half = len(s) // 2
            h1, h2 = s.iloc[:half], s.iloc[half:]
            f1 = (h1["goals_for"] - h1["xg_for"]).mean()
            f2 = (h2["goals_for"] - h2["xg_for"]).mean()
            pts.append({"team": team, "season": season, "h1_finishing": f1, "h2_finishing": f2})
    df = pd.DataFrame(pts)
    fit = ols_fit(df["h1_finishing"], df["h2_finishing"]) if not df.empty else {}
    return df, fit


# ── 3. End-of-season projection (current/partial season) ───────────────────
def project_points(matches_all: pd.DataFrame, season_label: str,
                   total_games: int = FULL_SEASON_GAMES) -> pd.DataFrame:
    t = build_tables(matches_all, season_label)
    if t.empty:
        return t
    played = t["played"].max()
    if played == 0:
        return pd.DataFrame()
    remaining = max(total_games - played, 0)
    out = t[["team", "played", "points", "xpts"]].copy()
    out["ppg"] = out["points"] / out["played"]
    out["xpts_pg"] = out["xpts"] / out["played"]
    # Projection: points so far + remaining games at xPts-per-game pace
    out["proj_on_xg"] = (out["points"] + out["xpts_pg"] * remaining).round(0)
    out["proj_on_form"] = (out["points"] + out["ppg"] * remaining).round(0)
    out = out.sort_values("proj_on_xg", ascending=False).reset_index(drop=True)
    out["proj_rank"] = range(1, len(out) + 1)
    return out


# ── Auto-generated narrative insights ──────────────────────────────────────
def league_insights(matches_all: pd.DataFrame, season_label: str) -> list[str]:
    t = build_tables(matches_all, season_label)
    if t.empty:
        return []
    ins = []
    unlucky = t.sort_values("pts_gap").iloc[0]
    lucky = t.sort_values("pts_gap", ascending=False).iloc[0]
    best_att = t.sort_values("xgf", ascending=False).iloc[0]
    best_def = t.sort_values("xga").iloc[0]

    ins.append(f"**{unlucky['team']}** are the unluckiest side — {abs(unlucky['pts_gap']):.0f} "
               f"points below what their performances (xPts) merit.")
    ins.append(f"**{lucky['team']}** are overperforming most, {lucky['pts_gap']:+.0f} points "
               f"above their expected total — a warning sign for regression.")
    ins.append(f"**{best_att['team']}** create the most ({best_att['xgf']:.1f} xG for), while "
               f"**{best_def['team']}** concede the fewest chances ({best_def['xga']:.1f} xG against).")

    biggest_mover = t.reindex(t["rank_delta"].abs().sort_values(ascending=False).index).iloc[0]
    if abs(biggest_mover["rank_delta"]) >= 2:
        direction = "higher" if biggest_mover["rank_delta"] > 0 else "lower"
        ins.append(f"Luck has lifted/dropped **{biggest_mover['team']}** "
                   f"{abs(int(biggest_mover['rank_delta']))} places {direction} than an xPts table.")
    return ins
