"""
analysis/leaderboards.py
------------------------
League-wide leaderboards for quick 'best in the division' comparisons.
Operates on a prepared (per-90, market-joined) league player frame and on
match data for team-level luck.
"""

import pandas as pd
from analysis.league_table import build_tables


def clinical_finishers(league_prepped, top_n=10):
    d = league_prepped.sort_values("g_minus_xg", ascending=False)
    cols = [c for c in ["player","team","goals","xg","g_minus_xg","shots"] if c in d.columns]
    return d.head(top_n)[cols].reset_index(drop=True)


def top_creators(league_prepped, top_n=10):
    sort_col = "xa" if "xa" in league_prepped.columns else "assists"
    d = league_prepped.sort_values(sort_col, ascending=False)
    cols = [c for c in ["player","team","assists","xa","key_passes"] if c in d.columns]
    return d.head(top_n)[cols].reset_index(drop=True)


def best_value(league_prepped, top_n=10):
    if "market_value_m" not in league_prepped.columns:
        return pd.DataFrame()
    d = league_prepped[league_prepped["market_value_m"].notna() & (league_prepped["market_value_m"] > 0)].copy()
    if d.empty:
        return pd.DataFrame()
    d["contrib"] = d["goals"] + d["assists"]
    d["contrib_per_m"] = (d["contrib"] / d["market_value_m"]).round(2)
    d = d[d["contrib"] >= 3].sort_values("contrib_per_m", ascending=False)
    cols = [c for c in ["player","team","contrib","market_value_m","contrib_per_m"] if c in d.columns]
    return d.head(top_n)[cols].reset_index(drop=True)


def team_luck(matches_all, season_label, top_n=None):
    table = build_tables(matches_all, season_label)
    d = table.sort_values("pts_gap")  # most unlucky first
    cols = ["team","points","xpts","pts_gap","rank_delta"]
    out = d[cols].reset_index(drop=True)
    return out


# ── Output leaderboards (raw totals) ────────────────────────────────────────
def _top(df, col, top_n, extra=None):
    base = ["player", "team", col]
    cols = base + [c for c in (extra or []) if c in df.columns and c not in base]
    cols = [c for c in cols if c in df.columns]
    return df.sort_values(col, ascending=False).head(top_n)[cols].reset_index(drop=True)


def top_goals(lp, top_n=10):
    return _top(lp, "goals", top_n, ["xg", "g_minus_xg"])


def top_assists(lp, top_n=10):
    return _top(lp, "assists", top_n, ["xa", "a_minus_xa"])


def top_goal_contributions(lp, top_n=10):
    d = lp.copy()
    d["goal_contributions"] = d["goals"] + d["assists"]
    d["xgi"] = (d["xg"] + d["xa"]).round(2)               # expected goal involvement
    d["gi_minus_xgi"] = (d["goal_contributions"] - d["xgi"]).round(2)
    return _top(d, "goal_contributions", top_n, ["goals", "assists", "xgi", "gi_minus_xgi"])


# ── Over- and under-performance leaderboards (vs expected) ──────────────────
def with_performance_ratios(lp, min_xg=4.0, min_xa=2.5):
    """
    Add G/xG, A/xA and combined GI/xGI ratios.
    Thresholds are deliberately meaningful: a player needs ~4.0 xG (or 2.5 xA)
    before a ratio is informative — otherwise the board fills with low-volume
    players whose 0/low output isn't really 'underperformance'.
    """
    import numpy as np
    d = lp.copy()
    d["goal_contributions"] = d["goals"] + d["assists"]
    d["xgi"] = d["xg"] + d["xa"]
    d["g_vs_xg_ratio"] = np.where(d["xg"] >= min_xg, (d["goals"] / d["xg"]).round(2), np.nan)
    d["a_vs_xa_ratio"] = np.where(d["xa"] >= min_xa, (d["assists"] / d["xa"]).round(2), np.nan)
    d["gi_vs_xgi_ratio"] = np.where(d["xgi"] >= (min_xg + min_xa),
                                    (d["goal_contributions"] / d["xgi"]).round(2), np.nan)
    return d


def top_overperformers(lp, metric="g_minus_xg", top_n=10):
    d = with_performance_ratios(lp)
    if metric not in d.columns:
        return pd.DataFrame()
    return _top(d.dropna(subset=[metric]), metric, top_n,
                ["goals", "xg", "assists", "xa"])


def top_underperformers(lp, metric="g_vs_xg_ratio", top_n=10):
    """Lowest ratio = biggest underperformers, filtered to players with enough volume."""
    d = with_performance_ratios(lp)
    if metric not in d.columns:
        return pd.DataFrame()
    d = d.dropna(subset=[metric]).sort_values(metric, ascending=True)
    extra = {"g_vs_xg_ratio": ["goals", "xg"],
             "a_vs_xa_ratio": ["assists", "xa"],
             "gi_vs_xgi_ratio": ["goal_contributions", "xgi"]}.get(metric, [])
    cols = ["player", "team", metric] + [c for c in extra if c in d.columns]
    return d.head(top_n)[cols].reset_index(drop=True)
