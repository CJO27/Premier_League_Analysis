"""
Smoke + invariant tests for the analytics layer.
Run:  pytest -q
"""
import numpy as np
import pandas as pd

from analysis.xg import (get_xg_summary, get_cumulative_xg, get_verdict,
                         season_comparison, list_teams)
from analysis.players import (prepare_players, radar_for_position,
                              find_similar, position_group)
from analysis.transfers import score_candidates, list_profiles
from analysis.league_table import build_tables
from analysis.compare import (all_team_metrics, team_profile_percentiles,
                              player_across_seasons)
from analysis.leaderboards import clinical_finishers, best_value
from analysis.matching import attach_market_values


# ── xG ──────────────────────────────────────────────────────────────────────
def test_xg_summary_columns(matches):
    df = get_xg_summary(matches, "Tottenham", "2025/2026")
    assert not df.empty
    for col in ["xg_for", "xg_against", "xg_diff", "venue", "match_num"]:
        assert col in df.columns
    assert df["venue"].isin(["Home", "Away"]).all()


def test_cumulative_is_monotonic(matches):
    cum = get_cumulative_xg(get_xg_summary(matches, "Arsenal", "2025/2026"))
    assert cum["cum_actual_points"].is_monotonic_increasing
    assert cum["cum_xpts"].is_monotonic_increasing


def test_verdict_keys(matches):
    cum = get_cumulative_xg(get_xg_summary(matches, "Chelsea", "2025/2026"))
    v = get_verdict(cum, "Chelsea")
    assert set(["actual_pts", "xpts", "gap", "label", "explanation"]) <= v.keys()


def test_three_seasons_present(matches):
    assert matches["season_label"].nunique() == 3
    comp = season_comparison(matches, "Liverpool")
    assert len(comp) == 3


# ── League table ────────────────────────────────────────────────────────────
def test_league_table_ranks_unique_and_complete(matches):
    t = build_tables(matches, "2025/2026")
    assert len(t) == 10
    assert t["points"].is_monotonic_decreasing
    # pts_gap should equal points - xpts
    assert np.allclose(t["pts_gap"], (t["points"] - t["xpts"]), atol=0.05)


# ── Players ─────────────────────────────────────────────────────────────────
def test_per90_non_negative(players):
    lp = prepare_players(players[players["season_label"] == "2025/2026"], 300)
    for c in ["npxg_per90", "xa_per90", "shots_per90"]:
        assert (lp[c] >= 0).all()


def test_radar_values_bounded(players):
    lp = prepare_players(players[players["season_label"] == "2025/2026"], 300)
    row = lp.iloc[0]
    rd = radar_for_position(row, lp, row["pos_group"])
    assert all(0 <= d["value"] <= 100 for d in rd)


def test_similarity_sorted_and_bounded(players):
    lp = prepare_players(players[players["season_label"] == "2025/2026"], 300)
    name = lp.iloc[0]["player"]
    sim = find_similar(lp, name, same_position=True, top_n=5)
    if not sim.empty:
        assert sim["similarity"].is_monotonic_decreasing
        assert (sim["similarity"] <= 100).all()
        assert name not in sim["player"].values


def test_position_group_mapping():
    assert position_group("F S") == "Forward"
    assert position_group("D") == "Defender"
    assert position_group("M") == "Midfielder"
    assert position_group("GK") == "Goalkeeper"


# ── Transfers (incl. the NA market-value regression) ────────────────────────
def test_fit_scores_bounded(players):
    lp = prepare_players(players[players["season_label"] == "2025/2026"], 300)
    for prof in list_profiles():
        c = score_candidates(lp, prof, exclude_team="Tottenham", top_n=10, min_minutes=300)
        if not c.empty:
            assert c["fit_score"].between(0, 100).all()


def test_score_candidates_handles_na_market_value(players):
    """Regression: NA market values must not crash value_score rounding."""
    lp = prepare_players(players[players["season_label"] == "2025/2026"], 300)
    lp["market_value_m"] = pd.array(
        [np.nan if i % 3 else 50.0 for i in range(len(lp))], dtype="Float64")
    c = score_candidates(lp, "Striker (ST)", exclude_team=None, top_n=10, min_minutes=300)
    # Should build with a value_score column and not raise
    assert "value_score" in c.columns


# ── Compare ─────────────────────────────────────────────────────────────────
def test_team_percentiles_bounded(matches):
    rd = team_profile_percentiles(matches, "Tottenham", "2025/2026")
    assert rd and all(0 <= d["value"] <= 100 for d in rd)


def test_player_across_seasons(players):
    hist = player_across_seasons(players, "SHARED_0")
    assert hist["season_label"].nunique() >= 2


# ── Market matching ─────────────────────────────────────────────────────────
def test_market_attach_adds_columns(players, market):
    lp = prepare_players(players[players["season_label"] == "2025/2026"], 300)
    out = attach_market_values(lp, market)
    for c in ["market_value_m", "years_left", "age"]:
        assert c in out.columns
    # at least the SHARED_ players should match
    assert out["market_value_m"].notna().any()


# ── Defensive radar (v7) ────────────────────────────────────────────────────
def _fbref_misc(players_df):
    import numpy as np
    rng = np.random.default_rng(9)
    names = players_df["player"].unique()
    rows = []
    for n in names:
        rows.append({"player": n, "90s": round(float(rng.uniform(5, 34)), 1),
                     "performance_tkl": int(rng.integers(10, 80)),
                     "performance_int": int(rng.integers(5, 60)),
                     "performance_recov": int(rng.integers(40, 200)),
                     "performance_won": int(rng.integers(10, 120)),
                     "performance_blocks": int(rng.integers(5, 50))})
    return pd.DataFrame(rows)


def test_defender_radar_falls_back_without_def_data(players):
    lp = prepare_players(players[players["season_label"] == "2025/2026"], 300)
    dfn = lp[lp["pos_group"] == "Defender"]
    if dfn.empty:
        return
    rd = radar_for_position(dfn.iloc[0], lp, "Defender")
    labels = [d["label"] for d in rd]
    assert "Tackles" not in labels  # no def data => build-up fallback


def test_defender_radar_uses_defensive_metrics_when_merged(players):
    from analysis.players import merge_defensive
    lp = prepare_players(players[players["season_label"] == "2025/2026"], 300)
    lp = merge_defensive(lp, _fbref_misc(lp))
    assert "def_tackles_per90" in lp.columns
    assert lp["def_tackles_per90"].notna().any()
    dfn = lp[lp["pos_group"] == "Defender"]
    if dfn.empty:
        return
    rd = radar_for_position(dfn.iloc[0], lp, "Defender")
    labels = [d["label"] for d in rd]
    assert "Tackles" in labels and "Interceptions" in labels
    assert all(0 <= d["value"] <= 100 for d in rd)


def test_merge_defensive_graceful_when_empty(players):
    from analysis.players import merge_defensive
    lp = prepare_players(players[players["season_label"] == "2025/2026"], 300)
    out = merge_defensive(lp, None)
    assert out.equals(lp)  # unchanged when no FBref data


# ── Models (v8) ──────────────────────────────────────────────────────────────
def test_ols_fit_recovers_known_line():
    from analysis.models import ols_fit
    import numpy as np
    x = np.arange(20.0)
    y = 2.5 * x + 4 + np.random.default_rng(0).normal(0, 0.01, 20)
    fit = ols_fit(x, y)
    assert abs(fit["slope"] - 2.5) < 0.05
    assert fit["r2"] > 0.99


def test_xgd_points_model_runs(matches):
    from analysis.models import xgd_points_model
    agg, fit = xgd_points_model(matches)
    assert not agg.empty
    assert 0 <= fit["r2"] <= 1


def test_projection_at_least_current_points(matches):
    from analysis.models import project_points
    proj = project_points(matches, "2025/2026", total_games=38)
    if not proj.empty:
        assert (proj["proj_on_xg"] >= proj["points"]).all()
        assert (proj["proj_on_form"] >= proj["points"]).all()


def test_insights_generated(matches):
    from analysis.models import league_insights
    ins = league_insights(matches, "2025/2026")
    assert len(ins) >= 3
    assert all(isinstance(s, str) and s for s in ins)


# ── Regressions (v9) ─────────────────────────────────────────────────────────
def test_xg_summary_empty_for_absent_team(matches):
    """A team that didn't play a given season must return empty, not KeyError."""
    out = get_xg_summary(matches, "NonExistentFC", "2025/2026")
    assert out.empty


def test_finishing_persistence_handles_varying_teams(matches):
    """Regression: finishing_persistence iterates all teams × seasons; teams
    absent from a season previously raised KeyError on 'xg_for'."""
    from analysis.models import finishing_persistence
    persist, fit = finishing_persistence(matches)
    assert isinstance(persist, pd.DataFrame)  # ran without raising


def test_over_and_under_performer_leaderboards(players):
    from analysis.leaderboards import (top_goals, top_assists,
        top_goal_contributions, top_overperformers, top_underperformers)
    lp = prepare_players(players[players["season_label"] == "2025/2026"], 300)
    assert top_goals(lp, 5)["goals"].is_monotonic_decreasing
    assert top_assists(lp, 5)["assists"].is_monotonic_decreasing
    gc = top_goal_contributions(lp, 5)
    assert gc["goal_contributions"].is_monotonic_decreasing
    over = top_overperformers(lp, "g_minus_xg", 5)
    assert over["g_minus_xg"].is_monotonic_decreasing
    under = top_underperformers(lp, "g_vs_xg_ratio", 5)
    if not under.empty:
        assert under["g_vs_xg_ratio"].is_monotonic_increasing  # lowest first


# ── Roles & Best XI (v10) ────────────────────────────────────────────────────
def test_clustering_assigns_roles(players):
    from analysis.roles import assign_roles
    lp = prepare_players(players[players["season_label"] == "2025/2026"], 300)
    roles_df, coords, info = assign_roles(lp, k=5, min_minutes=300)
    assert "role" in roles_df.columns
    if info["method"] == "kmeans":
        assert "pca_x" in roles_df.columns
        assert roles_df["role"].nunique() >= 2


def test_best_xi_fills_eleven(players):
    from analysis.roles import best_xi, FORMATIONS
    from analysis.players import radar_for_position
    lp = prepare_players(players[players["season_label"] == "2025/2026"], 300)
    for form in FORMATIONS:
        xi = best_xi(lp, form, radar_for_position, min_minutes=300)
        assert len(xi) == 11
        assert all("player" in p and "rating" in p for p in xi)


def test_best_xi_no_duplicate_players(players):
    from analysis.roles import best_xi
    from analysis.players import radar_for_position
    lp = prepare_players(players[players["season_label"] == "2025/2026"], 300)
    xi = best_xi(lp, "4-3-3", radar_for_position, min_minutes=300)
    names = [p["player"] for p in xi]
    assert len(names) == len(set(names))  # no player appears twice
