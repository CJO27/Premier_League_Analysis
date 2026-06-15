"""
analysis/matching.py
--------------------
Fuzzy-match Transfermarkt player names to Understat player names.
Names differ in order and accents ("Heung-Min Son" vs "Son Heung-min"),
so we normalise and use token-set + difflib similarity.
"""

import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd


def _norm(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    n = re.sub(r"[^a-z\s]", " ", n.lower())
    return " ".join(sorted(n.split()))  # token-sort so order doesn't matter


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def attach_market_values(players: pd.DataFrame, market: pd.DataFrame,
                         threshold: float = 0.72) -> pd.DataFrame:
    """
    Left-join market data onto players by best fuzzy name match within the
    same team where possible, else league-wide.
    Adds: market_value_m, years_left, age, contract_until (NaN if no match).
    """
    df = players.copy()
    for col in ["market_value_m", "years_left", "age", "contract_until"]:
        df[col] = pd.NA

    if market is None or market.empty:
        return df

    market = market.copy()
    market["_norm"] = market["tm_name"].apply(_norm)

    for idx, row in df.iterrows():
        pname = row["player"]
        pteam = row.get("team", "")
        # Prefer same-team candidates
        pool = market[market["team"] == pteam]
        if pool.empty:
            pool = market

        best, best_score = None, 0.0
        pn = _norm(pname)
        for _, m in pool.iterrows():
            score = SequenceMatcher(None, pn, m["_norm"]).ratio()
            if score > best_score:
                best_score, best = score, m

        if best is not None and best_score >= threshold:
            df.at[idx, "market_value_m"] = best["market_value_m"]
            df.at[idx, "years_left"] = best["years_left"]
            df.at[idx, "age"] = best["age"]
            df.at[idx, "contract_until"] = best["contract_until"]

    return df
