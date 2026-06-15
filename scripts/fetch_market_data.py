"""
fetch_market_data.py
--------------------
Scrapes Transfermarkt for Premier League squad market data:
  market value, contract expiry (-> years left), age, nationality,
  position, height, preferred foot.

Why a separate offline script (not live in the app):
  Transfermarkt has no official API and rate-limits aggressively. The clean,
  deploy-safe pattern is to harvest into a cached parquet offline and commit
  it, then have the dashboard read the cache. Refresh by re-running this.

Run:  python scripts/fetch_market_data.py

NOTE: Transfermarkt occasionally changes their HTML. If columns come back
empty, the CSS selectors below may need a tweak — the script prints what it
finds per team so you can see immediately.
"""

import sys
import time
import re
from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

# Transfermarkt internal team IDs for 2025/26 Premier League.
# (Stable identifiers; the slug before /kader doesn't matter.)
PL_TEAMS = {
    "Arsenal": 11, "Aston Villa": 405, "Bournemouth": 989, "Brentford": 1148,
    "Brighton": 1237, "Chelsea": 631, "Crystal Palace": 873, "Everton": 29,
    "Fulham": 931, "Liverpool": 31, "Manchester City": 281,
    "Manchester United": 985, "Newcastle United": 762,
    "Nottingham Forest": 703, "Tottenham": 148, "West Ham": 379,
    "Wolverhampton Wanderers": 543, "Leeds": 399, "Burnley": 1132,
    "Sunderland": 289,
}

SEASON_ID = 2025  # 2025/26
BASE = "https://www.transfermarkt.com"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}


def parse_value(text: str) -> float:
    """'€85.00m' -> 85.0 (in € millions). '€800k' -> 0.8. Returns NaN if blank."""
    if not text:
        return float("nan")
    t = text.replace("€", "").strip().lower()
    try:
        if t.endswith("m"):
            return round(float(t[:-1]), 2)
        if t.endswith("k"):
            return round(float(t[:-1]) / 1000, 3)
        return float(t)
    except ValueError:
        return float("nan")


def years_left(contract_text: str) -> float:
    """'Jun 30, 2028' -> years from now. Returns NaN if unparseable."""
    if not contract_text:
        return float("nan")
    m = re.search(r"(\d{4})", contract_text)
    if not m:
        return float("nan")
    end_year = int(m.group(1))
    from datetime import date
    today = date.today()
    # Contracts typically expire June 30
    return round((end_year + 0.5) - (today.year + today.month / 12), 1)


def scrape_team(team_name: str, team_id: int) -> list[dict]:
    import requests
    from bs4 import BeautifulSoup

    url = f"{BASE}/-/kader/verein/{team_id}/saison_id/{SEASON_ID}/plus/1"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    rows = []
    table = soup.select_one("table.items")
    if not table:
        return rows

    for tr in table.select("tbody > tr"):
        # Player name
        name_el = tr.select_one("td.hauptlink a")
        if not name_el:
            continue
        name = name_el.get("title") or name_el.text.strip()

        # Position (small text under name)
        pos_el = tr.select_one("td.posrela tr:nth-of-type(2) td")
        position = pos_el.text.strip() if pos_el else ""

        # Age — the zentriert cells vary; grab the one that looks like "(age)"
        age = float("nan")
        for td in tr.select("td.zentriert"):
            m = re.search(r"\((\d{2})\)", td.text)
            if m:
                age = int(m.group(1)); break

        # Contract expiry — usually a zentriert cell with a date
        contract = ""
        for td in tr.select("td.zentriert"):
            if re.search(r"\b(19|20)\d{2}\b", td.text) and "," in td.text:
                contract = td.text.strip(); break

        # Market value — last hauptlink/right cell with €
        val_el = tr.select_one("td.rechts.hauptlink a") or tr.select_one("td.rechts a")
        value_m = parse_value(val_el.text.strip()) if val_el else float("nan")

        rows.append({
            "tm_name": name,
            "team": team_name,
            "tm_position": position,
            "age": age,
            "contract_until": contract,
            "years_left": years_left(contract),
            "market_value_m": value_m,
        })
    return rows


def main():
    try:
        import requests  # noqa
        from bs4 import BeautifulSoup  # noqa
    except ImportError:
        print("Installing scraper deps...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "requests", "beautifulsoup4", "lxml"], check=True)

    all_rows = []
    for team, tid in PL_TEAMS.items():
        try:
            rows = scrape_team(team, tid)
            print(f"  {team}: {len(rows)} players, "
                  f"{sum(1 for r in rows if pd.notna(r['market_value_m']))} with value")
            all_rows.extend(rows)
            time.sleep(2)  # be polite to Transfermarkt
        except Exception as e:
            print(f"  {team}: FAILED — {e}")

    if not all_rows:
        print("\nNo data scraped. Transfermarkt may have changed their HTML, "
              "or blocked the request. Try again later or adjust selectors.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    df.to_parquet(PROCESSED / "market_values.parquet")
    print(f"\n✅ Saved market_values.parquet — {len(df)} players, "
          f"{df['market_value_m'].notna().sum()} with market value")


if __name__ == "__main__":
    main()
