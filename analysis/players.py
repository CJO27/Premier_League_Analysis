"""
analysis/players.py
-------------------
Player profiles, position-aware comparison, and similarity finder.

Understat player columns (flattened): player, team, position, matches,
minutes, goals, xg, np_goals, np_xg, assists, xa, shots, key_passes,
yellow_cards, red_cards, xg_chain, xg_buildup, season_label
"""

import pandas as pd
import numpy as np

# ── Position groups & the metrics that matter for each ──────────────────────
# Understat position codes: F (forward), S (sub/striker), M (mid), D (def), GK
POSITION_GROUPS = {
    "Forward":    {"filter": ["F", "S"]},
    "Midfielder": {"filter": ["M"]},
    "Defender":   {"filter": ["D"]},
    "Goalkeeper": {"filter": ["GK"]},
}

# Radar metric sets tuned per position. Defensive metrics (def_*_per90) are
# populated only when FBref 'misc' data has been merged in; otherwise the
# defender/midfielder radars fall back to build-up metrics automatically.
POSITION_RADAR = {
    "Forward": [
        ("npxg_per90", "Non-pen xG"), ("goals_per90", "Goals"),
        ("shots_per90", "Shots"), ("xa_per90", "xA"),
        ("xgchain_per90", "xG chain"), ("conversion", "Conversion"),
    ],
    "Midfielder": [
        ("xa_per90", "xA"), ("key_passes_per90", "Key passes"),
        ("xgchain_per90", "xG chain"), ("xgbuildup_per90", "Build-up"),
        ("def_tackles_per90", "Tackles"), ("def_interceptions_per90", "Interceptions"),
    ],
    "Defender": [
        ("def_tackles_per90", "Tackles"), ("def_interceptions_per90", "Interceptions"),
        ("def_recoveries_per90", "Recoveries"), ("def_aerials_won_per90", "Aerials won"),
        ("xgbuildup_per90", "Build-up"), ("def_blocks_per90", "Blocks"),
    ],
    "Goalkeeper": [
        ("xgbuildup_per90", "Build-up"), ("xgchain_per90", "xG chain"),
    ],
}

# Fallback radar (attacking/build-up only) used when defensive data is absent
POSITION_RADAR_FALLBACK = {
    "Defender": [
        ("xgbuildup_per90", "Build-up"), ("xgchain_per90", "Progression"),
        ("key_passes_per90", "Key passes"), ("xa_per90", "xA"),
        ("npxg_per90", "Non-pen xG"), ("goals_per90", "Goals"),
    ],
    "Midfielder": [
        ("xa_per90", "xA"), ("key_passes_per90", "Key passes"),
        ("xgchain_per90", "xG chain"), ("xgbuildup_per90", "Build-up"),
        ("npxg_per90", "Non-pen xG"), ("goals_per90", "Goals"),
    ],
}

DEFAULT_RADAR = POSITION_RADAR_FALLBACK["Midfielder"]

# Mapping from our metric names to likely FBref 'misc' column fragments
DEF_SOURCE = {
    "def_tackles_per90":       ["tkl", "tackles", "performance_tkl"],
    "def_interceptions_per90": ["int", "interceptions", "performance_int"],
    "def_recoveries_per90":    ["recov", "recoveries", "performance_recov"],
    "def_aerials_won_per90":   ["aerial_won", "won", "aerial_duels_won", "performance_won"],
    "def_blocks_per90":        ["blocks", "blk", "performance_blocks"],
}


def position_group(pos: str) -> str:
    pos = str(pos).upper()
    if "GK" in pos:
        return "Goalkeeper"
    if "D" in pos and "M" not in pos:
        return "Defender"
    if "F" in pos or pos.strip() == "S":
        return "Forward"
    if "M" in pos:
        return "Midfielder"
    if "F" in pos or "S" in pos:
        return "Forward"
    return "Midfielder"


def _per90(df, col):
    mins = df["minutes"].replace(0, np.nan)
    return (df[col] / mins * 90).fillna(0)


def prepare_players(players: pd.DataFrame, min_minutes: int = 200) -> pd.DataFrame:
    df = players.copy()
    for c in ["goals","xg","np_goals","np_xg","assists","xa","shots",
              "key_passes","xg_chain","xg_buildup","minutes","matches"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df = df[df["minutes"] >= min_minutes].copy()
    if df.empty:
        return df

    df["goals_per90"]      = _per90(df, "goals")
    df["npxg_per90"]       = _per90(df, "np_xg")
    df["xa_per90"]         = _per90(df, "xa")
    df["shots_per90"]      = _per90(df, "shots")
    df["key_passes_per90"] = _per90(df, "key_passes")
    df["xgchain_per90"]    = _per90(df, "xg_chain")
    df["xgbuildup_per90"]  = _per90(df, "xg_buildup")
    df["conversion"]       = np.where(df["shots"] > 0, df["goals"] / df["shots"], 0)

    df["g_minus_xg"]   = df["goals"] - df["xg"]
    df["a_minus_xa"]   = df["assists"] - df["xa"]
    df["xg_per_shot"]  = np.where(df["shots"] > 0, df["xg"] / df["shots"], 0)
    df["pos_group"]    = df["position"].apply(position_group)

    def lab(d):
        return "Overperforming" if d >= 2 else ("Underperforming" if d <= -2 else "On track")
    df["efficiency_label"] = df["g_minus_xg"].apply(lab)
    return df.sort_values("xg", ascending=False).reset_index(drop=True)


def get_team_players(players_league: pd.DataFrame, team: str,
                     season_label: str | None = None, min_minutes: int = 200) -> pd.DataFrame:
    df = players_league.copy()
    if season_label and "season_label" in df.columns:
        df = df[df["season_label"] == season_label]
    df = df[df["team"] == team]
    return prepare_players(df, min_minutes)


def _find_col(df, fragments):
    """Return first column whose name contains any fragment (case-insensitive)."""
    for frag in fragments:
        for c in df.columns:
            if frag == c:
                return c
    for frag in fragments:
        for c in df.columns:
            if frag in c:
                return c
    return None


def merge_defensive(prepped: pd.DataFrame, fbref_def: pd.DataFrame) -> pd.DataFrame:
    """
    Join FBref 'misc' defensive stats onto the prepared (per-90) player frame
    by fuzzy name match, producing def_*_per90 columns. Degrades gracefully:
    if fbref_def is None/empty or columns can't be found, returns input unchanged.
    """
    if fbref_def is None or fbref_def.empty:
        return prepped

    from analysis.matching import _norm  # reuse normaliser
    fb = fbref_def.copy()

    name_col = _find_col(fb, ["player", "name"])
    mins_col = _find_col(fb, ["min", "playing_time_min", "90s", "minutes"])
    if not name_col:
        return prepped

    # Resolve source columns
    src = {dst: _find_col(fb, frags) for dst, frags in DEF_SOURCE.items()}
    fb["_norm"] = fb[name_col].apply(_norm)

    # If FBref gives 90s played, use it; else approximate from minutes
    if mins_col and "90" in mins_col:
        nineties = pd.to_numeric(fb[mins_col], errors="coerce")
    elif mins_col:
        nineties = pd.to_numeric(fb[mins_col], errors="coerce") / 90
    else:
        nineties = pd.Series(1, index=fb.index)
    nineties = nineties.replace(0, np.nan)

    # Build a per-90 lookup keyed by normalised name
    lookup = {}
    for i, row in fb.iterrows():
        rec = {}
        for dst, col in src.items():
            if col:
                val = pd.to_numeric(row[col], errors="coerce")
                rec[dst] = (val / nineties[i]) if pd.notna(val) and pd.notna(nineties[i]) else np.nan
        lookup[row["_norm"]] = rec

    out = prepped.copy()
    for dst in DEF_SOURCE:
        out[dst] = np.nan
    for idx, prow in out.iterrows():
        key = _norm(prow["player"])
        if key in lookup:
            for dst, val in lookup[key].items():
                out.at[idx, dst] = val
    return out


def _radar_metrics_for(pos_group, league_prepped):
    """Pick defensive radar if def_* columns exist & have data, else fallback."""
    primary = POSITION_RADAR.get(pos_group, DEFAULT_RADAR)
    needs_def = any(m[0].startswith("def_") for m in primary)
    if needs_def:
        have_def = all(
            (m[0] in league_prepped.columns and league_prepped[m[0]].notna().any())
            for m in primary if m[0].startswith("def_"))
        if not have_def:
            return POSITION_RADAR_FALLBACK.get(pos_group, DEFAULT_RADAR)
    return primary


def radar_for_position(player_row, league_prepped, pos_group: str | None = None):
    """Percentile-rank a player's role-appropriate metrics vs same-position peers."""
    pos_group = pos_group or player_row.get("pos_group") or "Midfielder"
    metrics = _radar_metrics_for(pos_group, league_prepped)
    peers = league_prepped[league_prepped["pos_group"] == pos_group]
    if peers.empty:
        peers = league_prepped

    out = []
    for col, label in metrics:
        if col not in league_prepped.columns:
            continue
        val = player_row.get(col, 0)
        if pd.isna(val):
            val = 0
        series = peers[col].dropna()
        pct = (series < val).mean() * 100 if len(series) else 50
        out.append({"metric": col, "label": label, "value": round(pct)})
    return out


def compare_players(players_df, name_a, name_b):
    a = players_df[players_df["player"] == name_a]
    b = players_df[players_df["player"] == name_b]
    if a.empty or b.empty:
        return pd.DataFrame()
    a, b = a.iloc[0], b.iloc[0]
    metrics = ["goals","xg","assists","xa","shots","key_passes",
               "goals_per90","npxg_per90","xa_per90","xg_per_shot","conversion","minutes"]
    rows = [{"metric": m, name_a: round(float(a[m]),2), name_b: round(float(b[m]),2)}
            for m in metrics if m in players_df.columns]
    return pd.DataFrame(rows)


# ── Similarity finder ───────────────────────────────────────────────────────
SIM_FEATURES = ["npxg_per90","goals_per90","xa_per90","shots_per90",
                "key_passes_per90","xgchain_per90","xgbuildup_per90","conversion"]


def find_similar(league_prepped: pd.DataFrame, player_name: str,
                 same_position: bool = True, top_n: int = 8) -> pd.DataFrame:
    """Cosine similarity on z-scored per-90 features within position group."""
    if player_name not in league_prepped["player"].values:
        return pd.DataFrame()

    target = league_prepped[league_prepped["player"] == player_name].iloc[0]
    pool = league_prepped.copy()
    if same_position:
        pool = pool[pool["pos_group"] == target["pos_group"]]

    feats = [f for f in SIM_FEATURES if f in pool.columns]
    X = pool[feats].astype(float)
    # z-score
    mu, sd = X.mean(), X.std(ddof=0).replace(0, 1)
    Z = (X - mu) / sd
    tvec = (target[feats].astype(float) - mu) / sd

    # cosine similarity
    dot = Z.values @ tvec.values
    norms = np.linalg.norm(Z.values, axis=1) * np.linalg.norm(tvec.values)
    norms = np.where(norms == 0, 1, norms)
    pool = pool.assign(similarity=(dot / norms))

    out = pool[pool["player"] != player_name].sort_values("similarity", ascending=False)
    cols = ["player","team","position","similarity","goals","xg","assists","xa","minutes"]
    cols = [c for c in cols if c in out.columns]
    res = out.head(top_n)[cols].copy()
    res["similarity"] = (res["similarity"] * 100).round(0)
    return res.reset_index(drop=True)
