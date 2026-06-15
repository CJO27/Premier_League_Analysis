"""Shared synthetic fixtures so tests run without live network data."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEAMS = ["Arsenal","Chelsea","Liverpool","Tottenham","Everton","Fulham",
         "Brentford","Brighton","Wolverhampton Wanderers","Leeds"]


@pytest.fixture(scope="session")
def matches():
    rng = np.random.default_rng(0)
    rows = []
    for season in ["2025/2026", "2024/2025", "2023/2024"]:
        for i, h in enumerate(TEAMS):
            for j, a in enumerate(TEAMS):
                if h == a or (i + j) % 3:
                    continue
                hxg, axg = round(rng.uniform(0.5, 2.8), 2), round(rng.uniform(0.4, 2.2), 2)
                rows.append(dict(home_team=h, away_team=a,
                    home_goals=int(rng.poisson(hxg)), away_goals=int(rng.poisson(axg)),
                    home_xg=hxg, away_xg=axg,
                    home_expected_points=round(rng.uniform(0.5, 2.5), 2),
                    away_expected_points=round(rng.uniform(0.5, 2.5), 2),
                    home_ppda=round(rng.uniform(7, 15), 2), away_ppda=round(rng.uniform(7, 15), 2),
                    home_deep_completions=int(rng.integers(2, 12)),
                    away_deep_completions=int(rng.integers(2, 12)),
                    date=pd.Timestamp("2025-08-01") + pd.Timedelta(days=len(rows)),
                    season_label=season))
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def players():
    rng = np.random.default_rng(1)
    pos = ["F S", "F", "M", "M S", "D M", "D"]
    rows = []
    for season in ["2025/2026", "2024/2025", "2023/2024"]:
        for t in TEAMS:
            for k in range(15):
                name = f"SHARED_{k}" if k < 6 else f"{t[:3]}_{k}_{season[:4]}"
                mins = int(rng.integers(300, 3000))
                xg = round(rng.uniform(1, 16) * mins / 2400, 2)
                rows.append(dict(team=t, player=name, position=rng.choice(pos),
                    minutes=mins, matches=mins // 90, goals=int(rng.poisson(xg)), xg=xg,
                    np_goals=int(rng.poisson(xg * 0.9)), np_xg=round(xg * 0.9, 2),
                    assists=int(rng.poisson(xg * 0.5)), xa=round(xg * 0.5, 2),
                    shots=int(xg * 8) + 1, key_passes=int(xg * 6),
                    xg_chain=round(xg * 1.8, 2), xg_buildup=round(xg * 1.2, 2),
                    season_label=season))
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def market():
    rng = np.random.default_rng(2)
    rows = [dict(tm_name=f"SHARED_{k}", team=TEAMS[k % 10], tm_position="CF",
                 age=int(rng.integers(18, 34)), contract_until="Jun 30, 2028",
                 years_left=round(float(rng.uniform(0.5, 4)), 1),
                 market_value_m=round(float(rng.uniform(2, 120)), 1)) for k in range(6)]
    return pd.DataFrame(rows)
