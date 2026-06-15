"""
analysis/shotmap.py
-------------------
Prepare Understat shot-event data for shot maps.

Understat coords are normalised 0-1 (x along pitch length, y across width).
We scale to a 100x100 half-pitch for plotting (attacking right).
"""

import pandas as pd
import numpy as np


def team_shots(shots: pd.DataFrame, team: str, season_label: str | None = None) -> pd.DataFrame:
    if shots is None or shots.empty:
        return pd.DataFrame()
    df = shots.copy()
    if season_label and "season_label" in df.columns:
        df = df[df["season_label"] == season_label]
    df = df[df["team"] == team].copy()
    if df.empty:
        return df
    df["x"] = pd.to_numeric(df["location_x"], errors="coerce") * 100
    df["y"] = pd.to_numeric(df["location_y"], errors="coerce") * 100
    df["xg"] = pd.to_numeric(df["xg"], errors="coerce").fillna(0)
    df["is_goal"] = df["result"].astype(str).str.lower().eq("goal")
    return df.dropna(subset=["x", "y"])


def shot_summary(team_shot_df: pd.DataFrame) -> dict:
    if team_shot_df.empty:
        return {}
    return {
        "shots": len(team_shot_df),
        "goals": int(team_shot_df["is_goal"].sum()),
        "total_xg": round(team_shot_df["xg"].sum(), 1),
        "avg_xg": round(team_shot_df["xg"].mean(), 3),
        "big_chances": int((team_shot_df["xg"] >= 0.3).sum()),
    }
