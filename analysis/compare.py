"""
analysis/compare.py
-------------------
Comparison engine: team-vs-team, team-across-seasons, and player-across-seasons.

Team identity is summarised as per-game metrics, then percentile-ranked against
all teams in the same season so two profiles can be overlaid on one radar even
if they're from different seasons.
"""

import pandas as pd
import numpy as np


def _pts(gf, ga):
    return 3 if gf > ga else (1 if gf == ga else 0)


def all_team_metrics(matches_all: pd.DataFrame, season_label: str) -> pd.DataFrame:
    """Per-team, per-game metrics for one season."""
    df = matches_all[matches_all["season_label"] == season_label]
    teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    rows = []
    for team in teams:
        if not isinstance(team, str):
            continue
        gf=ga=xgf=xga=ppda=deep=pts=played=0.0
        for _, r in df.iterrows():
            if r["home_team"] == team:
                gf+=r["home_goals"]; ga+=r["away_goals"]; xgf+=r["home_xg"]; xga+=r["away_xg"]
                ppda+=float(r.get("home_ppda",0) or 0); deep+=float(r.get("home_deep_completions",0) or 0)
                pts+=_pts(r["home_goals"],r["away_goals"]); played+=1
            elif r["away_team"] == team:
                gf+=r["away_goals"]; ga+=r["home_goals"]; xgf+=r["away_xg"]; xga+=r["home_xg"]
                ppda+=float(r.get("away_ppda",0) or 0); deep+=float(r.get("away_deep_completions",0) or 0)
                pts+=_pts(r["away_goals"],r["home_goals"]); played+=1
        if played == 0:
            continue
        rows.append({
            "team": team, "played": int(played),
            "xg_for_pg": xgf/played, "xg_against_pg": xga/played,
            "ppda": ppda/played, "deep_pg": deep/played,
            "finishing_pg": (gf-xgf)/played, "def_overperf_pg": (xga-ga)/played,
            "ppg": pts/played,
        })
    return pd.DataFrame(rows)


# Radar axes: (column, label, higher_is_better)
TEAM_RADAR_AXES = [
    ("xg_for_pg", "Attack (xG for)", True),
    ("xg_against_pg", "Defence (xG against)", False),
    ("ppda", "Press intensity", False),     # lower PPDA = more press
    ("deep_pg", "Territory", True),
    ("finishing_pg", "Finishing", True),
    ("def_overperf_pg", "Keeper/defence", True),
]


def team_profile_percentiles(matches_all, team, season_label):
    """Return [{label, value 0-100}] radar for a team vs its season's peers."""
    metrics = all_team_metrics(matches_all, season_label)
    if metrics.empty or team not in metrics["team"].values:
        return []
    row = metrics[metrics["team"] == team].iloc[0]
    out = []
    for col, label, higher in TEAM_RADAR_AXES:
        series = metrics[col]
        if higher:
            pct = (series < row[col]).mean() * 100
        else:
            pct = (series > row[col]).mean() * 100   # lower value = higher percentile
        out.append({"metric": col, "label": label, "value": round(pct)})
    return out


def team_season_metrics(matches_all, team, season_label) -> dict:
    metrics = all_team_metrics(matches_all, season_label)
    if metrics.empty or team not in metrics["team"].values:
        return {}
    r = metrics[metrics["team"] == team].iloc[0]
    return {
        "xG for /game": round(r["xg_for_pg"], 2),
        "xG against /game": round(r["xg_against_pg"], 2),
        "PPDA": round(r["ppda"], 2),
        "Deep /game": round(r["deep_pg"], 1),
        "Points /game": round(r["ppg"], 2),
    }


def player_across_seasons(players_league, player_name) -> pd.DataFrame:
    """All season rows for a player (raw, per-season)."""
    df = players_league[players_league["player"] == player_name].copy()
    return df.sort_values("season_label")
