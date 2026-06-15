"""
analysis/roles.py
-----------------
Data-driven player roles via unsupervised clustering, and a best-XI builder.

We can't draw true positional heatmaps (no free touch-level tracking data —
Understat only gives x,y for shots). But we CAN redefine a player's *real*
role from their statistical fingerprint, which is the same insight: a nominal
"midfielder" who shoots and scores like a forward gets clustered with forwards.

Method:
  1. Standardise per-90 attacking (+ defensive, if merged) features.
  2. KMeans clusters players into playing-style archetypes.
  3. Each cluster is auto-named from its most distinctive features.
  4. PCA projects to 2D for a style map.

This demonstrates feature engineering, standardisation, clustering and
dimensionality reduction — core applied-ML / data-analytics skills.
"""

import numpy as np
import pandas as pd

ATTACK_FEATURES = ["npxg_per90", "goals_per90", "xa_per90", "shots_per90",
                   "key_passes_per90", "xgchain_per90", "xgbuildup_per90", "conversion"]
DEF_FEATURES = ["def_tackles_per90", "def_interceptions_per90",
                "def_recoveries_per90", "def_aerials_won_per90", "def_blocks_per90"]

# Human-readable label from the two most distinctive (highest-z) features
FEATURE_LABELS = {
    "npxg_per90": "goal threat", "goals_per90": "finishing", "xa_per90": "creativity",
    "shots_per90": "shot volume", "key_passes_per90": "chance creation",
    "xgchain_per90": "involvement", "xgbuildup_per90": "build-up",
    "conversion": "clinical", "def_tackles_per90": "tackling",
    "def_interceptions_per90": "interceptions", "def_recoveries_per90": "recoveries",
    "def_aerials_won_per90": "aerial duels", "def_blocks_per90": "blocking",
}


def _features(df):
    feats = [f for f in ATTACK_FEATURES if f in df.columns]
    feats += [f for f in DEF_FEATURES if f in df.columns and df[f].notna().any()]
    return feats


def assign_roles(league_prepped: pd.DataFrame, k: int = 6, min_minutes: int = 500,
                 random_state: int = 42):
    """
    Cluster players into k style archetypes. Returns (df_with_roles, pca_coords, info).
    Degrades gracefully if scikit-learn isn't available (returns position groups).
    """
    df = league_prepped[league_prepped["minutes"] >= min_minutes].copy()
    feats = _features(df)
    if len(df) < k or len(feats) < 3:
        df["role"] = df.get("pos_group", "Unknown")
        return df, None, {"method": "fallback", "features": feats}

    X = df[feats].fillna(0).astype(float).values

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
    except ImportError:
        df["role"] = df.get("pos_group", "Unknown")
        return df, None, {"method": "fallback", "features": feats}

    Xz = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    clusters = km.fit_predict(Xz)
    df["cluster"] = clusters

    # Name each cluster from its top-2 distinctive standardised centroid features
    centroids = km.cluster_centers_
    names = {}
    for c in range(k):
        order = np.argsort(centroids[c])[::-1]
        top = [feats[i] for i in order[:2]]
        label = " + ".join(FEATURE_LABELS.get(t, t) for t in top)
        names[c] = label.capitalize()
    # De-duplicate identical names
    seen = {}
    for c, nm in list(names.items()):
        if nm in seen.values():
            nm = f"{nm} ({c})"
        seen[c] = nm
    df["role"] = df["cluster"].map(seen)

    # PCA to 2D for the style map
    coords = PCA(n_components=2, random_state=random_state).fit_transform(Xz)
    df["pca_x"], df["pca_y"] = coords[:, 0], coords[:, 1]

    info = {"method": "kmeans", "k": k, "features": feats,
            "role_names": list(seen.values())}
    return df, coords, info


def role_mismatches(roles_df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Players whose data-driven role differs from their nominal position group —
    i.e. they play unlike their label. The interesting 'redefined' cases.
    """
    if "role" not in roles_df.columns or "pos_group" not in roles_df.columns:
        return pd.DataFrame()
    df = roles_df.copy()

    # crude expected-group keyword per role for flagging
    def group_of_role(role):
        r = role.lower()
        if any(w in r for w in ["tackl", "block", "intercept", "aerial", "recover"]):
            return "Defender"
        if any(w in r for w in ["finish", "goal threat", "clinical", "shot"]):
            return "Forward"
        return "Midfielder"

    df["data_group"] = df["role"].apply(group_of_role)
    mism = df[df["data_group"] != df["pos_group"]].copy()
    cols = [c for c in ["player","team","position","pos_group","role","goals","assists","xg","xa"]
            if c in mism.columns]
    return mism[cols].head(top_n).reset_index(drop=True)


# ── Best XI ─────────────────────────────────────────────────────────────────
FORMATIONS = {
    "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "4-2-3-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
    "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
    "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
}

# Pitch coordinates (x along width 0-100, y along length 0-100, GK at bottom)
FORMATION_COORDS = {
    "4-3-3": {"GK":[(50,6)], "DEF":[(15,25),(38,22),(62,22),(85,25)],
              "MID":[(28,50),(50,46),(72,50)], "FWD":[(22,78),(50,82),(78,78)]},
    "4-2-3-1": {"GK":[(50,6)], "DEF":[(15,25),(38,22),(62,22),(85,25)],
                "MID":[(35,42),(65,42),(22,62),(50,60),(78,62)], "FWD":[(50,84)]},
    "3-5-2": {"GK":[(50,6)], "DEF":[(28,22),(50,20),(72,22)],
              "MID":[(12,48),(35,44),(50,52),(65,44),(88,48)], "FWD":[(38,80),(62,80)]},
    "4-4-2": {"GK":[(50,6)], "DEF":[(15,25),(38,22),(62,22),(85,25)],
              "MID":[(15,52),(38,48),(62,48),(85,52)], "FWD":[(38,80),(62,80)]},
}

# Which position groups can fill each slot type
SLOT_POS = {"GK": ["Goalkeeper"], "DEF": ["Defender"],
            "MID": ["Midfielder"], "FWD": ["Forward"]}


def _player_rating(row, radar_fn, league_prepped):
    """Overall rating = mean of the player's position-radar percentiles."""
    rd = radar_fn(row, league_prepped, row.get("pos_group"))
    return np.mean([d["value"] for d in rd]) if rd else 0.0


def best_xi(league_prepped, formation, radar_fn, team=None, min_minutes=600):
    """
    Pick the best XI for a formation. If team given, restrict to that squad.
    Returns list of dicts: {slot, x, y, player, team, rating}.
    """
    pool = league_prepped[league_prepped["minutes"] >= min_minutes].copy()
    if team:
        pool = pool[pool["team"] == team]
    if pool.empty:
        return []

    pool["rating"] = pool.apply(lambda r: _player_rating(r, radar_fn, league_prepped), axis=1)
    slots = FORMATIONS[formation]
    coords = FORMATION_COORDS[formation]

    xi, used = [], set()
    for slot, n in slots.items():
        groups = SLOT_POS[slot]
        cands = pool[pool["pos_group"].isin(groups) & ~pool["player"].isin(used)]
        cands = cands.sort_values("rating", ascending=False)
        picks = cands.head(n)
        # If not enough specialists (e.g. GK data sparse), backfill by rating
        if len(picks) < n:
            extra = pool[~pool["player"].isin(used) & ~pool["player"].isin(picks["player"])]
            extra = extra.sort_values("rating", ascending=False).head(n - len(picks))
            picks = pd.concat([picks, extra])
        for (xy, (_, p)) in zip(coords[slot], picks.iterrows()):
            used.add(p["player"])
            xi.append({"slot": slot, "x": xy[0], "y": xy[1], "player": p["player"],
                       "team": p.get("team",""), "rating": round(float(p["rating"]), 0),
                       "role": p.get("role","")})
    return xi
