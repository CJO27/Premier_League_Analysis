"""analysis/shots.py — rolling xG / PPDA / deep-completion trends."""
import pandas as pd

ROLLING = 5

def get_shot_trends(xg_df: pd.DataFrame) -> pd.DataFrame:
    df = xg_df.sort_values("date").copy()
    for src, dst in [("xg_for","xg_roll"),("xg_against","xga_roll"),
                     ("ppda","ppda_roll"),("deep","deep_roll"),("xg_diff","xg_diff_roll")]:
        df[dst] = df[src].rolling(ROLLING, min_periods=1).mean()
    return df

def get_shot_summary(xg_df: pd.DataFrame) -> dict:
    return {"avg_xg_for": round(xg_df["xg_for"].mean(),2),
            "avg_xg_against": round(xg_df["xg_against"].mean(),2),
            "avg_ppda": round(xg_df["ppda"].mean(),2),
            "avg_deep": round(xg_df["deep"].mean(),1)}
