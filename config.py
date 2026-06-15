"""
config.py — central configuration for the dashboard.
Keeping constants in one place is cleaner than scattering them across modules.
"""

LEAGUE = "ENG-Premier League"

# Understat uses the season START year. 2025 = 2025/26.
# Three seasons gives genuine year-on-year trajectory analysis.
SEASONS_UNDERSTAT = ["2025", "2024", "2023"]

def season_label(start_year: str) -> str:
    return f"{start_year}/{int(start_year[2:]) + 1}"

SEASON_LABELS = [season_label(s) for s in SEASONS_UNDERSTAT]

# Palette (kept in sync with visuals/charts.py)
COLORS = {
    "bg": "#0D0D1A", "navy": "#132257", "cyan": "#00D4FF",
    "white": "#FFFFFF", "red": "#FF4B4B", "gold": "#FFD700", "green": "#1D9E75",
}
