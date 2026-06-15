"""
analysis/xg.py
--------------
Match-level xG analysis using Understat data. Works for ANY team.
"""

import pandas as pd
import numpy as np


def team_matches(matches_all: pd.DataFrame, team: str,
                 season_label: str | None = None) -> pd.DataFrame:
    df = matches_all.copy()
    if season_label:
        df = df[df["season_label"] == season_label]
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    return df[mask].sort_values("date").reset_index(drop=True)


def get_xg_summary(matches_all: pd.DataFrame, team: str,
                   season_label: str | None = None) -> pd.DataFrame:
    df = team_matches(matches_all, team, season_label)
    rows = []
    for _, r in df.iterrows():
        is_home = r["home_team"] == team
        rows.append({
            "date": pd.to_datetime(r["date"]),
            "season_label": r.get("season_label", ""),
            "opponent": r["away_team"] if is_home else r["home_team"],
            "venue": "Home" if is_home else "Away",
            "goals_for": float(r["home_goals"] if is_home else r["away_goals"]),
            "goals_against": float(r["away_goals"] if is_home else r["home_goals"]),
            "xg_for": float(r["home_xg"] if is_home else r["away_xg"]),
            "xg_against": float(r["away_xg"] if is_home else r["home_xg"]),
            "xpts": float(r.get("home_expected_points" if is_home else "away_expected_points", np.nan)),
            "ppda": float(r.get("home_ppda" if is_home else "away_ppda", np.nan)),
            "deep": float(r.get("home_deep_completions" if is_home else "away_deep_completions", np.nan)),
        })
    out = pd.DataFrame(rows)
    if out.empty or "xg_for" not in out.columns:
        return pd.DataFrame()
    out = out.dropna(subset=["xg_for"]).sort_values("date").reset_index(drop=True)
    if out.empty:
        return out
    out["xg_diff"] = out["xg_for"] - out["xg_against"]
    out["actual_diff"] = out["goals_for"] - out["goals_against"]
    out["finishing_overperf"] = out["goals_for"] - out["xg_for"]
    out["defensive_overperf"] = out["xg_against"] - out["goals_against"]
    out["match_num"] = range(1, len(out) + 1)
    return out


def _pts(diff):
    return 3 if diff > 0 else (1 if diff == 0 else 0)


def get_cumulative_xg(xg_df: pd.DataFrame) -> pd.DataFrame:
    df = xg_df.sort_values("date").copy()
    df["cum_xg_for"] = df["xg_for"].cumsum()
    df["cum_goals_for"] = df["goals_for"].cumsum()
    df["cum_xg_against"] = df["xg_against"].cumsum()
    df["cum_goals_against"] = df["goals_against"].cumsum()
    df["actual_result"] = df["actual_diff"].apply(_pts)
    df["cum_actual_points"] = df["actual_result"].cumsum()
    df["cum_xpts"] = df["xpts"].cumsum()
    df["points_gap"] = df["cum_actual_points"] - df["cum_xpts"]
    return df


def get_verdict(cum_df: pd.DataFrame, team: str = "The team") -> dict:
    actual = float(cum_df["cum_actual_points"].iloc[-1])
    xpts = float(cum_df["cum_xpts"].iloc[-1])
    gap = actual - xpts
    if gap <= -8:
        label, expl = "Seriously unlucky", f"{team} deserved ~{abs(gap):.0f} more points on xPts. Performances far outstrip results."
    elif gap <= -4:
        label, expl = "Somewhat unlucky", f"A {abs(gap):.0f}-point shortfall vs xPts — poor finishing or harsh results."
    elif gap <= 3:
        label, expl = "About right", f"Only a {abs(gap):.0f}-point gap vs xPts — the table reflects performances."
    else:
        label, expl = "Overperforming", f"{team} sits {gap:.0f} points above xPts — results flatter the underlying numbers."
    return {"actual_pts": round(actual), "xpts": round(xpts), "gap": round(gap),
            "label": label, "explanation": expl}


def season_comparison(matches_all: pd.DataFrame, team: str) -> pd.DataFrame:
    rows = []
    for label in sorted(matches_all["season_label"].unique()):
        s = get_xg_summary(matches_all, team, label)
        if s.empty:
            continue
        rows.append({
            "season": label, "matches": len(s),
            "xg_for": round(s["xg_for"].sum(), 1), "goals_for": int(s["goals_for"].sum()),
            "xg_against": round(s["xg_against"].sum(), 1), "goals_against": int(s["goals_against"].sum()),
            "xpts": round(s["xpts"].sum(), 1),
            "actual_pts": int(s["actual_diff"].apply(_pts).sum()),
            "avg_ppda": round(s["ppda"].mean(), 2),
        })
    return pd.DataFrame(rows)


def list_teams(matches_all: pd.DataFrame) -> list[str]:
    teams = pd.concat([matches_all["home_team"], matches_all["away_team"]]).unique()
    return sorted(t for t in teams if isinstance(t, str))
