"""
validate_data.py
----------------
Data-quality checks on the fetched parquet files. Run after fetch_data.py to
confirm the data is sane before building on it. Exits non-zero on hard failures
so it can gate a pipeline.

Run:  python scripts/validate_data.py
"""

import sys
from pathlib import Path
import pandas as pd

PROCESSED = Path("data/processed")

EXPECTED = {
    "matches_all.parquet": ["home_team","away_team","home_xg","away_xg",
                            "home_goals","away_goals","season_label"],
    "players_league.parquet": ["player","team","minutes","goals","xg","season_label"],
}


def check(name, msg, ok):
    flag = "✓" if ok else "✗"
    print(f"  {flag} {msg}")
    return ok


def main():
    print("Validating data/processed/ ...\n")
    all_ok = True

    for fname, cols in EXPECTED.items():
        path = PROCESSED / fname
        print(fname)
        if not path.exists():
            all_ok &= check(fname, "file exists", False); continue
        df = pd.read_parquet(path)
        all_ok &= check(fname, f"has rows ({len(df)})", len(df) > 0)
        all_ok &= check(fname, "expected columns present",
                        all(c in df.columns for c in cols))

        if "matches" in fname:
            xg_ok = df[["home_xg","away_xg"]].notna().all().all()
            all_ok &= check(fname, "no missing xG", xg_ok)
            neg = ((df["home_xg"] < 0) | (df["away_xg"] < 0)).any()
            all_ok &= check(fname, "no negative xG", not neg)
            seasons = df["season_label"].nunique()
            check(fname, f"seasons covered: {seasons}", seasons >= 1)
        if "players" in fname:
            dup = df.duplicated(["player","team","season_label"]).sum()
            check(fname, f"duplicate player rows: {dup}", True)
            neg = (df["minutes"] < 0).any()
            all_ok &= check(fname, "no negative minutes", not neg)
        print()

    # Optional files — warn only
    for opt in ["shots.parquet","market_values.parquet","fbref_defensive.parquet"]:
        p = PROCESSED / opt
        print(f"  {'✓' if p.exists() else '–'} optional: {opt} "
              f"{'present' if p.exists() else 'not fetched'}")

    print("\n" + ("✅ All critical checks passed." if all_ok else "❌ Some checks failed."))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
