"""analysis/home_away.py — venue split."""
import pandas as pd

def _pts(d): return 3 if d>0 else (1 if d==0 else 0)

def get_home_away_summary(xg_df: pd.DataFrame) -> pd.DataFrame:
    df = xg_df.copy()
    df["points"] = df["actual_diff"].apply(_pts)
    s = df.groupby("venue").agg(
        matches=("goals_for","count"), goals_for=("goals_for","sum"),
        goals_against=("goals_against","sum"), xg_for=("xg_for","sum"),
        xg_against=("xg_against","sum"), points=("points","sum"),
        avg_xg_for=("xg_for","mean"), avg_xg_against=("xg_against","mean"),
        avg_goals_for=("goals_for","mean"), avg_goals_against=("goals_against","mean"),
        avg_ppda=("ppda","mean")).reset_index()
    s["ppg"] = (s["points"]/s["matches"]).round(2)
    s["xg_diff_per_game"] = ((s["xg_for"]-s["xg_against"])/s["matches"]).round(2)
    for c in ["avg_xg_for","avg_xg_against","avg_goals_for","avg_goals_against","avg_ppda"]:
        s[c] = s[c].round(2)
    return s
