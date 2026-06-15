"""
fetch_data.py
-------------
Fetches Premier League data for the analytics dashboard (all 20 teams).

Sources (free):
  - Understat: match-level xG/xPts/PPDA for EVERY team, full-league player
               season stats, and shot-by-shot events (for shot maps).

Outputs (data/processed/):
  - matches_all.parquet     every PL match, both seasons (xG, xPts, PPDA, deep)
  - players_league.parquet  every PL player, both seasons (xG, xA, xGChain...)
  - shots.parquet           shot-by-shot events (optional; --shots flag)

Run:
  python scripts/fetch_data.py            # matches + players (fast)
  python scripts/fetch_data.py --shots    # also fetch shot events (slower)
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

LEAGUE   = "ENG-Premier League"
SEASONS  = ["2025", "2024", "2023"]   # Understat start-year: 3 seasons

FETCH_SHOTS = "--shots" in sys.argv


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(c).strip().lower() for c in col
                      if str(c).strip() and str(c).lower() != "nan") for col in df.columns]
    else:
        df.columns = [str(c).lower().strip() for c in df.columns]
    return df


def clean(df):
    for c in df.columns:
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() >= df[c].notna().sum() * 0.5:
                df[c] = conv
            else:
                df[c] = df[c].astype(str).replace("nan", "")
    return df


def fetch_all_matches(seasons):
    import soccerdata as sd
    print("\n── Understat: ALL team match stats ──")
    out = []
    for season in seasons:
        print(f"  Season {season}...")
        us = sd.Understat(leagues=LEAGUE, seasons=season)
        df = us.read_team_match_stats().reset_index()
        df = flatten(df)
        df["season_label"] = f"{season}/{int(season[2:])+1}"
        out.append(df)
        print(f"    {len(df)} matches")
    combined = pd.concat(out, ignore_index=True).sort_values("date").reset_index(drop=True)
    combined = clean(combined)
    combined.to_parquet(PROCESSED / "matches_all.parquet")
    print(f"  Saved matches_all.parquet — {len(combined)} matches")
    return combined


def fetch_players(seasons):
    import soccerdata as sd
    print("\n── Understat: player season stats (full league) ──")
    out = []
    for season in seasons:
        print(f"  Season {season}...")
        us = sd.Understat(leagues=LEAGUE, seasons=season)
        df = us.read_player_season_stats().reset_index()
        df = flatten(df)
        df["season_label"] = f"{season}/{int(season[2:])+1}"
        out.append(df)
        print(f"    {len(df)} players")
    combined = clean(pd.concat(out, ignore_index=True).reset_index(drop=True))
    combined.to_parquet(PROCESSED / "players_league.parquet")
    print(f"  Saved players_league.parquet — {len(combined)} rows")
    return combined


def fetch_shots(seasons):
    import soccerdata as sd
    print("\n── Understat: shot events (slow — one request per match) ──")
    out = []
    for season in seasons:
        print(f"  Season {season}... (this can take several minutes)")
        us = sd.Understat(leagues=LEAGUE, seasons=season)
        try:
            df = us.read_shot_events().reset_index()
            df = flatten(df)
            df["season_label"] = f"{season}/{int(season[2:])+1}"
            out.append(df)
            print(f"    {len(df)} shots")
        except Exception as e:
            print(f"    WARNING: shot events failed for {season} — {e}")
    if out:
        combined = clean(pd.concat(out, ignore_index=True).reset_index(drop=True))
        combined.to_parquet(PROCESSED / "shots.parquet")
        print(f"  Saved shots.parquet — {len(combined)} shots")


def fetch_fbref_defensive(seasons_fbref):
    """
    FBref 'misc' player stats — tackles, interceptions, recoveries, aerials.
    Understat has no defensive data, so this powers proper defender radars.
    Joined to Understat players later by fuzzy name match.
    """
    import soccerdata as sd
    print("\n── FBref: player defensive (misc) stats ──")
    out = []
    for season in seasons_fbref:
        print(f"  Season {season}...")
        try:
            fb = sd.FBref(leagues=LEAGUE, seasons=season)
            df = fb.read_player_season_stats(stat_type="misc").reset_index()
            df = flatten(df)
            df["season_fbref"] = season
            out.append(clean(df))
            print(f"    {len(df)} players")
        except Exception as e:
            print(f"    WARNING: FBref misc failed for {season} — {e}")
    if out:
        combined = pd.concat(out, ignore_index=True)
        combined.to_parquet(PROCESSED / "fbref_defensive.parquet")
        print(f"  Saved fbref_defensive.parquet — {len(combined)} rows")
        print(f"  Columns: {[c for c in combined.columns][:30]}")
    else:
        print("  No FBref defensive data saved")


def main():
    try:
        import soccerdata  # noqa
    except ImportError:
        print("ERROR: pip install soccerdata")
        sys.exit(1)

    fetch_all_matches(SEASONS)
    fetch_players(SEASONS)
    # FBref uses combined-year season codes: 2025/26 -> "2526"
    fbref_seasons = [f"{s[2:]}{int(s[2:])+1}" for s in SEASONS]
    fetch_fbref_defensive(fbref_seasons)
    if FETCH_SHOTS:
        fetch_shots(SEASONS)
    else:
        print("\n  (Skipping shot events — run with --shots to enable shot maps)")

    print("\n✅ Done!")
    print("   Next: python scripts/fetch_market_data.py   (Transfermarkt values)")
    print("   Then: streamlit run app.py")


if __name__ == "__main__":
    main()
