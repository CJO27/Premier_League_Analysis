"""
analysis/league_table.py
------------------------
Builds the actual league table AND an xG-based "deserved" table from
match data, then shows how far each team is over/under-performing.
"""

import pandas as pd
import numpy as np


def _result_points(gf, ga):
    return 3 if gf > ga else (1 if gf == ga else 0)


def build_tables(matches_all: pd.DataFrame, season_label: str) -> pd.DataFrame:
    df = matches_all[matches_all["season_label"] == season_label].copy()
    teams = pd.concat([df["home_team"], df["away_team"]]).unique()

    rows = []
    for team in teams:
        if not isinstance(team, str):
            continue
        actual_pts = xpts = gf = ga = xgf = xga = played = 0.0
        for _, r in df.iterrows():
            if r["home_team"] == team:
                actual_pts += _result_points(r["home_goals"], r["away_goals"])
                xpts += float(r.get("home_expected_points", 0) or 0)
                gf += r["home_goals"]; ga += r["away_goals"]
                xgf += r["home_xg"]; xga += r["away_xg"]; played += 1
            elif r["away_team"] == team:
                actual_pts += _result_points(r["away_goals"], r["home_goals"])
                xpts += float(r.get("away_expected_points", 0) or 0)
                gf += r["away_goals"]; ga += r["home_goals"]
                xgf += r["away_xg"]; xga += r["home_xg"]; played += 1
        rows.append({
            "team": team, "played": int(played),
            "points": int(actual_pts), "xpts": round(xpts, 1),
            "pts_gap": round(actual_pts - xpts, 1),
            "gf": int(gf), "ga": int(ga), "gd": int(gf - ga),
            "xgf": round(xgf, 1), "xga": round(xga, 1),
            "xgd": round(xgf - xga, 1),
        })

    table = pd.DataFrame(rows)
    table["actual_rank"] = table["points"].rank(ascending=False, method="min").astype(int)
    table["xg_rank"] = table["xpts"].rank(ascending=False, method="min").astype(int)
    table["rank_delta"] = table["xg_rank"] - table["actual_rank"]  # + = lucky position
    return table.sort_values("points", ascending=False).reset_index(drop=True)
