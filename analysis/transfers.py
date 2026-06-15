"""
analysis/transfers.py
---------------------
Transfer fit: scores league players against weighted positional system
profiles, then layers in Transfermarkt market value + contract context to
surface fits that also represent good value.
"""

import pandas as pd
import numpy as np

SYSTEM_PROFILES = {
    "Winger (RW/LW)": {"position_filter": "F|M", "weights": {
        "npxg_per90": 1.0, "shots_per90": 0.8, "xa_per90": 0.9,
        "key_passes_per90": 0.7, "xgchain_per90": 0.6}},
    "Attacking mid (AM)": {"position_filter": "M|F", "weights": {
        "xa_per90": 1.0, "key_passes_per90": 1.0, "xgchain_per90": 0.9,
        "npxg_per90": 0.6}},
    "Striker (ST)": {"position_filter": "F|S", "weights": {
        "npxg_per90": 1.2, "goals_per90": 0.9, "shots_per90": 0.8,
        "conversion": 0.5, "xgchain_per90": 0.4}},
    "Deep playmaker (CM/DM)": {"position_filter": "M|D", "weights": {
        "xgchain_per90": 1.0, "xgbuildup_per90": 0.9, "key_passes_per90": 0.8,
        "xa_per90": 0.6}},
    "Ball-playing defender": {"position_filter": "D", "weights": {
        "xgbuildup_per90": 1.0, "xgchain_per90": 0.7, "key_passes_per90": 0.5}},
}


def _z(s):
    sd = s.std(ddof=0)
    return pd.Series(0, index=s.index) if (sd == 0 or np.isnan(sd)) else (s - s.mean()) / sd


def score_candidates(league_prepped, profile_name, exclude_team="Tottenham",
                     top_n=12, min_minutes=600, max_value=None, max_age=None):
    profile = SYSTEM_PROFILES[profile_name]
    weights, pos_filter = profile["weights"], profile["position_filter"]

    df = league_prepped.copy()
    df = df[df["minutes"] >= min_minutes]
    df = df[df["position"].str.contains(pos_filter, case=False, na=False)]
    if exclude_team:
        df = df[df["team"] != exclude_team]
    if max_value is not None and "market_value_m" in df.columns:
        df = df[(df["market_value_m"].isna()) | (df["market_value_m"] <= max_value)]
    if max_age is not None and "age" in df.columns:
        df = df[(df["age"].isna()) | (df["age"] <= max_age)]
    if df.empty:
        return pd.DataFrame()

    score = pd.Series(0.0, index=df.index)
    total_w = sum(abs(w) for w in weights.values())
    for metric, w in weights.items():
        if metric in df.columns:
            score += _z(df[metric]) * w
    score = score / total_w if total_w else score
    df["fit_score"] = (100 / (1 + np.exp(-1.6 * score))).round(0)

    # Value score: fit per £m (only where value known)
    if "market_value_m" in df.columns:
        mv = pd.to_numeric(df["market_value_m"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            vs = df["fit_score"] / mv
        df["value_score"] = vs.where(mv > 0).round(1)

    keep = ["player","team","position","fit_score","value_score","market_value_m",
            "years_left","age","minutes","goals","xg","assists","xa"] + list(weights.keys())
    keep = [c for c in keep if c in df.columns]
    return df.sort_values("fit_score", ascending=False).head(top_n)[keep].reset_index(drop=True)


def list_profiles():
    return list(SYSTEM_PROFILES.keys())
