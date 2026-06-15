"""
app.py — Premier League Analytics Dashboard (Squawka-style)
Any team · player profiles · transfer fit with market data · shot maps ·
xG league table · player similarity.

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="PL Analytics", page_icon="⚽",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background-color:#0D0D1A;color:#FFF;}
[data-testid="metric-container"]{background:#132257;border:1px solid #1e3580;border-radius:10px;padding:14px 18px;}
[data-testid="stMetricLabel"]{color:#00D4FF!important;font-size:0.8rem;}
[data-testid="stMetricValue"]{color:#FFF!important;font-size:1.5rem;}
h1{color:#FFF!important;font-weight:700;} h2{color:#00D4FF!important;font-weight:600;} h3{color:#AACCFF!important;}
.verdict-box{background:linear-gradient(135deg,#132257 0%,#0D0D1A 100%);border:2px solid #00D4FF;border-radius:12px;padding:22px 26px;margin-top:14px;}
.verdict-label{font-size:1.3rem;font-weight:700;color:#00D4FF;margin-bottom:6px;}
.verdict-text{font-size:1rem;color:#CCDDFF;line-height:1.6;}
section[data-testid="stSidebar"]{background-color:#0a0a14;}
hr{border-color:#1e3580;} #MainMenu,footer{visibility:hidden;}
</style>""", unsafe_allow_html=True)

PROCESSED = Path("data/processed")

@st.cache_data
def load(name):
    p = PROCESSED / name
    return pd.read_parquet(p) if p.exists() else None

if not (PROCESSED / "matches_all.parquet").exists():
    st.title("⚽ PL Analytics")
    st.error("No data found. Run:  `python scripts/fetch_data.py`")
    st.stop()

from analysis.xg import (get_xg_summary, get_cumulative_xg, get_verdict,
                         season_comparison, list_teams)
from analysis.shots import get_shot_trends
from analysis.players import (get_team_players, prepare_players,
                              radar_for_position, compare_players, find_similar,
                              position_group, POSITION_GROUPS)
from analysis.home_away import get_home_away_summary
from analysis.transfers import score_candidates, list_profiles
from analysis.league_table import build_tables
from analysis.shotmap import team_shots, shot_summary
from analysis.matching import attach_market_values
from analysis.compare import (all_team_metrics, team_profile_percentiles,
    team_season_metrics, player_across_seasons)
from analysis.leaderboards import (clinical_finishers, top_creators,
    best_value, team_luck)
from visuals.charts import (cumulative_points_chart, xg_vs_actual_bar,
    rolling_xg_diff_chart, ppda_chart, home_away_bars, player_xg_scatter,
    player_radar, transfer_fit_chart, season_comparison_chart,
    league_table_scatter, shot_map, similarity_chart,
    radar_compare, attack_defence_quadrant, cumulative_compare,
    leaderboard_bar, metric_compare_bars,
    trajectory_chart, xg_trajectory_chart, player_development_chart,
    scatter_with_fit, projection_chart, role_style_map, best_xi_pitch)
from analysis.models import (xgd_points_model, finishing_persistence,
    project_points, league_insights)
from analysis.roles import assign_roles, role_mismatches, best_xi, FORMATIONS

matches = load("matches_all.parquet")
players_league = load("players_league.parquet")
shots = load("shots.parquet")
market = load("market_values.parquet")
fbref_def = load("fbref_defensive.parquet")

seasons = sorted(matches["season_label"].unique(), reverse=True)
teams = list_teams(matches)

# Attach market values + defensive stats to league players once (cached)
@st.cache_data
def league_with_market(season_label):
    lp = prepare_players(players_league[players_league["season_label"] == season_label], min_minutes=300)
    if market is not None and not market.empty:
        lp = attach_market_values(lp, market)
    if fbref_def is not None and not fbref_def.empty:
        from analysis.players import merge_defensive
        lp = merge_defensive(lp, fbref_def)
    return lp

# Cache expensive league-wide computations (row-wise iteration over matches)
@st.cache_data
def cached_tables(season_label):
    return build_tables(matches, season_label)

@st.cache_data
def cached_team_metrics(season_label):
    return all_team_metrics(matches, season_label)

@st.cache_data
def cached_roles(season_label, k):
    return assign_roles(league_with_market(season_label), k=k, min_minutes=500)

with st.sidebar:
    st.markdown("### ⚽ PL Analytics")
    page = st.radio("Section", ["League overview", "Team analysis",
                                "Player profiles", "Compare", "Transfer fit",
                                "Squad & roles", "Shot map", "Models & insights",
                                "Methodology"])
    st.markdown("---")
    season = st.selectbox("Season", seasons, index=0)
    if page in ("Team analysis", "Player profiles", "Shot map"):
        team = st.selectbox("Team", teams,
            index=teams.index("Tottenham") if "Tottenham" in teams else 0)
    st.caption("Data: Understat · Transfermarkt")
    if market is None:
        st.warning("Market data not loaded — run fetch_market_data.py", icon="⚠️")
    if shots is None and page == "Shot map":
        st.warning("Shot data not loaded — run fetch_data.py --shots", icon="⚠️")

league_prepped = league_with_market(season)


# ═══════════════════════ LEAGUE OVERVIEW (default) ═════════════════════════
if page == "League overview":
    st.markdown(f"# Premier League · {season}")
    st.markdown("#### Who's overperforming, who's been unlucky?")
    st.markdown("---")
    table = cached_tables(season)

    # Auto-generated narrative insights
    insights = league_insights(matches, season)
    if insights:
        st.markdown("## Key insights")
        for line in insights:
            st.markdown(f"- {line}")

    st.markdown("## xG-based 'deserved' table")
    st.caption("Above the diagonal = earning more points than performances merit (lucky). Below = unlucky.")
    st.plotly_chart(league_table_scatter(table), use_container_width=True)

    st.markdown("## Full table — actual vs deserved")
    show = table[["actual_rank","team","played","points","xpts","pts_gap",
                  "gf","ga","xgf","xga","xgd","xg_rank","rank_delta"]].copy()
    show.columns = ["#","Team","P","Pts","xPts","Gap","GF","GA","xGF","xGA","xGD","xG#","Δ#"]
    st.dataframe(show, use_container_width=True, hide_index=True, height=740)
    st.download_button("⬇ Download table (CSV)", show.to_csv(index=False),
                       f"pl_table_{season.replace('/','-')}.csv", "text/csv")
    st.caption("Gap = actual − expected points. Δ# = how many places luck has moved them vs an xPts table.")

    st.markdown("## League map — attack vs defence")
    st.caption("Top-right = strong both ways. Bottom-left = struggling both ends.")
    metrics = cached_team_metrics(season)
    st.plotly_chart(attack_defence_quadrant(metrics), use_container_width=True)

    st.markdown("## League leaderboards")

    from analysis.leaderboards import (top_goals, top_assists, top_goal_contributions,
        top_overperformers, top_underperformers)

    st.markdown("### Output — top producers")
    o1, o2, o3 = st.columns(3)
    with o1:
        st.markdown("**Most goals**")
        st.plotly_chart(leaderboard_bar(top_goals(league_prepped, 10), "goals", "player",
            "Goals", fmt="%{x:.0f} goals"), use_container_width=True)
    with o2:
        st.markdown("**Most assists**")
        st.plotly_chart(leaderboard_bar(top_assists(league_prepped, 10), "assists", "player",
            "Assists", color="#FFD700", fmt="%{x:.0f} assists"), use_container_width=True)
    with o3:
        st.markdown("**Goals + assists**")
        st.plotly_chart(leaderboard_bar(top_goal_contributions(league_prepped, 10),
            "goal_contributions", "player", "Goal involvements", color="#1D9E75",
            fmt="%{x:.0f} G+A"), use_container_width=True)

    st.markdown("### Overperformers — beating their expected output")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("**Goals over xG**")
        st.plotly_chart(leaderboard_bar(top_overperformers(league_prepped, "g_minus_xg", 10),
            "g_minus_xg", "player", "Goals − xG", fmt="%{x:+.1f}"), use_container_width=True)
    with p2:
        st.markdown("**Assists over xA**")
        st.plotly_chart(leaderboard_bar(top_overperformers(league_prepped, "a_minus_xa", 10),
            "a_minus_xa", "player", "Assists − xA", color="#FFD700",
            fmt="%{x:+.1f}"), use_container_width=True)
    with p3:
        st.markdown("**G+A over expected (xGI)**")
        st.plotly_chart(leaderboard_bar(top_overperformers(league_prepped, "gi_vs_xgi_ratio", 10),
            "gi_vs_xgi_ratio", "player", "G+A ÷ xG+xA", color="#1D9E75",
            fmt="%{x:.2f}×"), use_container_width=True)

    st.markdown("### Underperformers — falling short of expected (lowest ratio)")
    st.caption("Ratios below 1.0 mean a player is converting fewer than their chances suggest. "
               "Filtered to players with meaningful expected volume.")
    u1, u2, u3 = st.columns(3)
    with u1:
        st.markdown("**Lowest goals ÷ xG**")
        st.plotly_chart(leaderboard_bar(top_underperformers(league_prepped, "g_vs_xg_ratio", 10),
            "g_vs_xg_ratio", "player", "Goals ÷ xG", color="#FF4B4B",
            fmt="%{x:.2f}×"), use_container_width=True)
    with u2:
        st.markdown("**Lowest assists ÷ xA**")
        st.plotly_chart(leaderboard_bar(top_underperformers(league_prepped, "a_vs_xa_ratio", 10),
            "a_vs_xa_ratio", "player", "Assists ÷ xA", color="#FF4B4B",
            fmt="%{x:.2f}×"), use_container_width=True)
    with u3:
        st.markdown("**Lowest G+A ÷ xGI**")
        st.plotly_chart(leaderboard_bar(top_underperformers(league_prepped, "gi_vs_xgi_ratio", 10),
            "gi_vs_xgi_ratio", "player", "G+A ÷ xG+xA", color="#FF4B4B",
            fmt="%{x:.2f}×"), use_container_width=True)

    bv = best_value(league_prepped, 10)
    if not bv.empty:
        st.markdown("### Best value (goal contributions per £m)")
        st.plotly_chart(leaderboard_bar(bv, "contrib_per_m", "player",
            "Contributions per £m", color="#1D9E75",
            fmt="%{x:.2f} per £m"), use_container_width=True)

    st.markdown("### Luckiest & unluckiest teams (xPts gap)")
    luck = team_luck(matches, season)
    st.dataframe(luck.rename(columns={"team":"Team","points":"Pts","xpts":"xPts",
        "pts_gap":"Gap","rank_delta":"Δ rank"}), use_container_width=True, hide_index=True)


# ═══════════════════════════ TEAM ANALYSIS ═════════════════════════════════
elif page == "Team analysis":
    xg_df = get_xg_summary(matches, team, season)
    if xg_df.empty:
        st.warning(f"No data for {team} in {season}."); st.stop()
    cum_df = get_cumulative_xg(xg_df)
    trend_df = get_shot_trends(xg_df)
    verdict = get_verdict(cum_df, team)

    st.markdown(f"# {team} · {season}")
    # Dynamic tagline based on the verdict, not a hardcoded question
    _tag = {
        "Seriously unlucky": "The performances deserved far more — genuinely unlucky.",
        "Somewhat unlucky": "Slightly short of what the underlying numbers merit.",
        "About right": "Results are a fair reflection of performances.",
        "Overperforming": "Punching above the underlying numbers — riding their luck.",
    }.get(verdict["label"], "")
    st.markdown(f"#### {verdict['label']} — {_tag}")
    st.markdown("---")
    c = st.columns(5)
    c[0].metric("Actual points", verdict["actual_pts"], f"{verdict['gap']:+d} vs xPts")
    c[1].metric("Expected points", verdict["xpts"])
    c[2].metric("xG for", f"{xg_df['xg_for'].sum():.1f}", f"{xg_df['goals_for'].sum()-xg_df['xg_for'].sum():+.1f} finishing")
    c[3].metric("xG against", f"{xg_df['xg_against'].sum():.1f}", f"{xg_df['goals_against'].sum()-xg_df['xg_against'].sum():+.1f} vs xG", delta_color="inverse")
    c[4].metric("Matches", len(xg_df))

    st.markdown("---"); st.markdown("## Points: deserved vs reality")
    st.plotly_chart(cumulative_points_chart(cum_df), use_container_width=True)
    st.markdown("## xG vs goals, match by match")
    st.plotly_chart(xg_vs_actual_bar(xg_df), use_container_width=True)
    a,b = st.columns(2)
    a.plotly_chart(rolling_xg_diff_chart(trend_df), use_container_width=True)
    b.plotly_chart(ppda_chart(trend_df), use_container_width=True)

    st.markdown("## Home vs away")
    ha = get_home_away_summary(xg_df); hc = st.columns(len(ha))
    for i,(_,r) in enumerate(ha.iterrows()):
        hc[i].metric(f"{r['venue']} PPG", r["ppg"], f"xG diff/g {r['xg_diff_per_game']:+.2f}")
    st.plotly_chart(home_away_bars(ha), use_container_width=True)

    if len(seasons) > 1:
        st.markdown("## Multi-season trajectory")
        comp = season_comparison(matches, team)
        tcol1, tcol2 = st.columns(2)
        tcol1.plotly_chart(trajectory_chart(comp, f"{team} — points by season"), use_container_width=True)
        tcol2.plotly_chart(xg_trajectory_chart(comp, f"{team} — xG by season"), use_container_width=True)
        st.plotly_chart(season_comparison_chart(comp), use_container_width=True)
        # Cumulative xPts race vs other seasons
        other = [s for s in seasons if s != season]
        if other:
            xb = get_cumulative_xg(get_xg_summary(matches, team, other[0]))
            if not xb.empty:
                st.plotly_chart(cumulative_compare(cum_df, season, xb, other[0],
                    metric="cum_xpts", ytitle="Cumulative xPts"), use_container_width=True)

    st.markdown("---"); st.markdown("## Verdict")
    st.markdown(f"""<div class="verdict-box"><div class="verdict-label">{verdict['label']}</div>
        <div class="verdict-text">{verdict['explanation']}</div></div>""", unsafe_allow_html=True)


# ═══════════════════════════ PLAYER PROFILES ═══════════════════════════════
elif page == "Player profiles":
    st.markdown(f"# {team} players · {season}")
    st.markdown("---")
    sp = get_team_players(players_league, team, season, min_minutes=200)
    if market is not None and not market.empty:
        sp = attach_market_values(sp, market)
    if fbref_def is not None and not fbref_def.empty:
        from analysis.players import merge_defensive
        sp = merge_defensive(sp, fbref_def)
    if sp.empty:
        st.warning("No player data."); st.stop()

    st.markdown("## Goals vs expected goals")
    st.plotly_chart(player_xg_scatter(sp), use_container_width=True)

    st.markdown("## Position-based profile & comparison")
    pos_groups = sorted(sp["pos_group"].unique())
    pos_sel = st.selectbox("Position group", pos_groups)
    pool = sp[sp["pos_group"] == pos_sel]
    names = pool["player"].tolist()
    cc = st.columns(2)
    p1 = cc[0].selectbox("Player", names, index=0)
    p2 = cc[1].selectbox("Compare with", ["None"]+names, index=0)

    row1 = pool[pool["player"]==p1].iloc[0]
    rd1 = radar_for_position(row1, league_prepped, pos_sel)
    rd2 = nm2 = None
    if p2 != "None":
        row2 = pool[pool["player"]==p2].iloc[0]
        rd2 = radar_for_position(row2, league_prepped, pos_sel); nm2 = p2

    cr, ct = st.columns([3,2])
    cr.plotly_chart(player_radar(rd1, p1, rd2, nm2), use_container_width=True)
    if pos_sel in ("Defender", "Midfielder"):
        if fbref_def is not None and not fbref_def.empty:
            cr.caption("Defensive metrics (tackles, interceptions, recoveries, aerials) from FBref.")
        else:
            cr.caption("Showing build-up metrics — run fetch_data.py for FBref defensive stats to enable a true defensive radar.")
    with ct:
        st.markdown(f"#### {p1}")
        st.metric("Goals", int(row1["goals"]), f"{row1['g_minus_xg']:+.1f} vs xG")
        st.metric("Assists", int(row1["assists"]), f"{row1['a_minus_xa']:+.1f} vs xA")
        if pd.notna(row1.get("market_value_m")):
            st.metric("Market value", f"£{row1['market_value_m']:.0f}m")
        if pd.notna(row1.get("years_left")):
            st.metric("Contract left", f"{row1['years_left']:.1f} yrs")

    if p2 != "None":
        st.markdown("## Head to head")
        st.dataframe(compare_players(pool, p1, p2), use_container_width=True, hide_index=True)

    st.markdown("## Find similar players (league-wide)")
    samepos = st.checkbox("Same position only", value=True)
    sim = find_similar(league_prepped, p1, same_position=samepos, top_n=8)
    if not sim.empty:
        st.plotly_chart(similarity_chart(sim, p1), use_container_width=True)
        st.dataframe(sim, use_container_width=True, hide_index=True)
    else:
        st.info(f"{p1} not found in the league-wide pool (needs 300+ minutes).")


# ═══════════════════════════════ COMPARE ═══════════════════════════════════
elif page == "Compare":
    st.markdown("# Compare")
    st.markdown("---")
    mode = st.radio("Compare", ["Two teams (same season)", "One team across seasons",
                                "Two players (same season)", "One player across seasons"],
                    horizontal=True)

    # ---- Two teams, same season ----
    if mode == "Two teams (same season)":
        cc = st.columns(2)
        ta = cc[0].selectbox("Team A", teams, index=teams.index("Tottenham") if "Tottenham" in teams else 0)
        tb = cc[1].selectbox("Team B", teams, index=1 if teams[0]==ta else 0)
        rda = team_profile_percentiles(matches, ta, season)
        rdb = team_profile_percentiles(matches, tb, season)
        st.plotly_chart(radar_compare(rda, ta, rdb, tb,
            f"Team identity — percentile vs league ({season})"), use_container_width=True)
        ma = team_season_metrics(matches, ta, season)
        mb = team_season_metrics(matches, tb, season)
        if ma and mb:
            st.plotly_chart(metric_compare_bars(ma, ta, mb, tb,
                f"Per-game metrics — {ta} vs {tb}"), use_container_width=True)

    # ---- One team across seasons ----
    elif mode == "One team across seasons":
        team_c = st.selectbox("Team", teams, index=teams.index("Tottenham") if "Tottenham" in teams else 0)
        if len(seasons) < 2:
            st.info("Only one season of data available."); st.stop()
        cc = st.columns(2)
        sa = cc[0].selectbox("Season A", seasons, index=0)
        sb = cc[1].selectbox("Season B", seasons, index=1)
        rda = team_profile_percentiles(matches, team_c, sa)
        rdb = team_profile_percentiles(matches, team_c, sb)
        st.plotly_chart(radar_compare(rda, f"{team_c} {sa}", rdb, f"{team_c} {sb}",
            f"{team_c} — identity by season (percentile vs each season's league)"),
            use_container_width=True)

        # Cumulative xPts race across the two seasons
        xa = get_cumulative_xg(get_xg_summary(matches, team_c, sa))
        xb = get_cumulative_xg(get_xg_summary(matches, team_c, sb))
        if not xa.empty and not xb.empty:
            st.plotly_chart(cumulative_compare(xa, sa, xb, sb,
                metric="cum_xpts", ytitle="Cumulative xPts"), use_container_width=True)
        ma = team_season_metrics(matches, team_c, sa)
        mb = team_season_metrics(matches, team_c, sb)
        if ma and mb:
            st.plotly_chart(metric_compare_bars(ma, sa, mb, sb,
                f"{team_c} — per-game metrics by season"), use_container_width=True)

    # ---- Two players, same season ----
    elif mode == "Two players (same season)":
        pool = league_prepped.sort_values("xg", ascending=False)
        names = pool["player"].tolist()
        cc = st.columns(2)
        pa = cc[0].selectbox("Player A", names, index=0)
        pb = cc[1].selectbox("Player B", names, index=min(1, len(names)-1))
        ra = pool[pool["player"]==pa].iloc[0]
        rb = pool[pool["player"]==pb].iloc[0]
        from analysis.players import radar_for_position
        rda = radar_for_position(ra, league_prepped, ra["pos_group"])
        rdb = radar_for_position(rb, league_prepped, rb["pos_group"])
        st.plotly_chart(radar_compare(rda, pa, rdb, pb,
            f"Player percentile profiles ({season})"), use_container_width=True)
        from analysis.players import compare_players
        st.dataframe(compare_players(pool, pa, pb), use_container_width=True, hide_index=True)

    # ---- One player across seasons ----
    else:
        all_names = sorted(players_league["player"].unique())
        pc = st.selectbox("Player", all_names)
        hist = player_across_seasons(players_league, pc)
        if hist["season_label"].nunique() < 2:
            st.info(f"{pc} only appears in one season of the data."); st.stop()

        # Development line across ALL seasons the player features in
        st.markdown("### Development over time")
        st.plotly_chart(player_development_chart(hist, pc), use_container_width=True)

        from analysis.players import prepare_players, radar_for_position
        avail = sorted(hist["season_label"].unique())
        st.markdown("### Percentile profile — pick two seasons to overlay")
        pcols = st.columns(2)
        s_a = pcols[0].selectbox("Season A", avail, index=len(avail)-1)
        s_b = pcols[1].selectbox("Season B", avail, index=0)

        radars = {}
        metrics_by_season = {}
        for sl in [s_a, s_b]:
            lp_season = prepare_players(players_league[players_league["season_label"]==sl], min_minutes=200)
            prow = lp_season[lp_season["player"]==pc]
            if prow.empty:
                continue
            prow = prow.iloc[0]
            radars[sl] = radar_for_position(prow, lp_season, prow["pos_group"])
            metrics_by_season[sl] = {
                "Goals": float(prow["goals"]), "xG": round(float(prow["xg"]),1),
                "Assists": float(prow["assists"]), "xA": round(float(prow["xa"]),1),
                "Shots": float(prow["shots"]), "Minutes": float(prow["minutes"]),
            }
        if s_a in radars and s_b in radars and s_a != s_b:
            st.plotly_chart(radar_compare(radars[s_a], f"{pc} {s_a}",
                radars[s_b], f"{pc} {s_b}",
                f"{pc} — percentile profile by season"), use_container_width=True)
            st.plotly_chart(metric_compare_bars(metrics_by_season[s_a], s_a,
                metrics_by_season[s_b], s_b,
                f"{pc} — output by season"), use_container_width=True)
        st.dataframe(hist[[c for c in ["season_label","team","position","minutes",
            "goals","xg","assists","xa","shots"] if c in hist.columns]],
            use_container_width=True, hide_index=True)


# ═══════════════════════════ TRANSFER FIT ══════════════════════════════════
elif page == "Transfer fit":
    st.markdown("# Transfer fit analysis")
    st.markdown("League players ranked by fit to a target role, with market value & contract context.")
    st.markdown("---")
    profile = st.selectbox("Target role / system profile", list_profiles())
    cc = st.columns(3)
    min_mins = cc[0].slider("Min minutes", 300, 2500, 600, 100)
    max_val = cc[1].slider("Max value (£m)", 5, 150, 150, 5)
    max_age = cc[2].slider("Max age", 18, 38, 38, 1)
    exclude = st.selectbox("Exclude club (your team)", ["None"]+teams,
        index=(teams.index("Tottenham")+1) if "Tottenham" in teams else 0)

    cands = score_candidates(league_prepped, profile,
        exclude_team=None if exclude=="None" else exclude,
        top_n=12, min_minutes=min_mins,
        max_value=max_val if max_val<150 else None,
        max_age=max_age if max_age<38 else None)
    if cands.empty:
        st.warning("No candidates match those filters."); st.stop()

    st.markdown("## Candidate ranking")
    st.plotly_chart(transfer_fit_chart(cands), use_container_width=True)

    st.markdown("## Detail")
    cols = ["player","team","position","fit_score","value_score","market_value_m",
            "years_left","age","minutes","goals","xg","assists","xa"]
    cols = [c for c in cols if c in cands.columns]
    nice = {"fit_score":"Fit","value_score":"Fit/£m","market_value_m":"Value £m",
            "years_left":"Yrs left","age":"Age"}
    st.dataframe(cands[cols].rename(columns=nice), use_container_width=True, hide_index=True)
    st.download_button("⬇ Download shortlist (CSV)", cands[cols].to_csv(index=False),
                       f"shortlist_{profile.split()[0].lower()}.csv", "text/csv")
    st.caption("Fit = weighted z-score of per-90 metrics vs positional peers (0–100). "
               "Fit/£m highlights value. Weights live in analysis/transfers.py. "
               "Market data via Transfermarkt.")


# ═══════════════════════════════ SQUAD & ROLES ═════════════════════════════
elif page == "Squad & roles":
    st.markdown("# Squad & roles")
    st.markdown("Data-driven playing styles and a best XI — what the *numbers* say "
                "a player's role is, regardless of their listed position.")
    st.markdown("---")

    st.info("Note: true positional heatmaps need touch-level tracking data, which "
            "isn't available from free sources. Instead, roles here are derived from "
            "each player's **statistical fingerprint** via clustering — arguably a more "
            "objective way to define how someone actually plays.", icon="📊")

    from analysis.players import radar_for_position

    tab1, tab2 = st.tabs(["🧬 Data-driven roles", "⚽ Best XI"])

    with tab1:
        k = st.slider("Number of style archetypes (clusters)", 4, 10, 6)
        roles_df, coords, info = cached_roles(season, k)
        if info.get("method") == "kmeans":
            st.markdown("### Player style map")
            st.caption("Each player positioned by their statistical profile (PCA of per-90 "
                       "metrics), coloured by the archetype k-means assigned them. Players "
                       "close together play similarly, whatever their nominal position.")
            all_players_in = sorted(roles_df["player"].unique())
            hl = st.selectbox("Highlight a player", ["None"]+all_players_in)
            st.plotly_chart(role_style_map(roles_df, None if hl=="None" else hl),
                            use_container_width=True)

            st.markdown("### Archetypes found")
            st.caption("Each cluster is auto-named from its most distinctive metrics.")
            summary = (roles_df.groupby("role")
                       .agg(players=("player","count"),
                            avg_goals=("goals","mean"), avg_assists=("assists","mean"))
                       .round(1).reset_index().sort_values("players", ascending=False))
            st.dataframe(summary.rename(columns={"role":"Archetype","players":"Players",
                "avg_goals":"Avg goals","avg_assists":"Avg assists"}),
                use_container_width=True, hide_index=True)

            st.markdown("### Redefined roles — players who play unlike their label")
            st.caption("Where the data-driven archetype disagrees with the nominal position group.")
            mism = role_mismatches(roles_df, top_n=15)
            if not mism.empty:
                st.dataframe(mism.rename(columns={"pos_group":"Listed as","role":"Plays like"}),
                             use_container_width=True, hide_index=True)
            else:
                st.write("No notable mismatches at this cluster count.")
        else:
            st.warning("Clustering unavailable (scikit-learn not installed or too few players). "
                       "Showing nominal position groups instead.")

    with tab2:
        cc = st.columns(2)
        formation = cc[0].selectbox("Formation", list(FORMATIONS.keys()))
        scope = cc[1].selectbox("Pick from", ["Whole league"]+teams)
        team_filter = None if scope == "Whole league" else scope
        xi = best_xi(league_prepped, formation, radar_for_position,
                     team=team_filter, min_minutes=600)
        if xi:
            title = f"Best XI ({formation}) — {'league' if not team_filter else team_filter} · {season}"
            st.plotly_chart(best_xi_pitch(xi, title), use_container_width=True)
            st.caption("Rating = average of each player's position-appropriate percentile radar "
                       "(0–100). Players placed at formation positions.")
            xi_df = pd.DataFrame(xi)[["slot","player","team","role","rating"]]
            st.dataframe(xi_df.rename(columns={"slot":"Slot","player":"Player","team":"Team",
                "role":"Style","rating":"Rating"}), use_container_width=True, hide_index=True)
            st.download_button("⬇ Download XI (CSV)", xi_df.to_csv(index=False),
                               f"best_xi_{formation}.csv", "text/csv")
        else:
            st.warning("Not enough players meet the minutes threshold for this selection.")


# ═══════════════════════════════ SHOT MAP ══════════════════════════════════
elif page == "Shot map":
    st.markdown(f"# {team} shot map · {season}")
    st.markdown("---")
    if shots is None:
        st.warning("No shot data. Run `python scripts/fetch_data.py --shots` to enable.")
        st.stop()
    ts = team_shots(shots, team, season)
    if ts.empty:
        st.warning(f"No shots found for {team} in {season}."); st.stop()

    # Player filter
    player_opts = (ts.groupby("player")["xg"].count()
                   .sort_values(ascending=False).index.tolist())
    picked = st.multiselect("Filter by player (leave empty for all)", player_opts)
    view = ts[ts["player"].isin(picked)] if picked else ts

    s = shot_summary(view)
    c = st.columns(5)
    c[0].metric("Shots", s["shots"]); c[1].metric("Goals", s["goals"])
    c[2].metric("Total xG", s["total_xg"]); c[3].metric("Avg xG/shot", s["avg_xg"])
    c[4].metric("Big chances", s["big_chances"])

    map_title = (f"{team} — {', '.join(picked)} ({season})" if picked
                 else f"{team} — all shots ({season})")
    st.plotly_chart(shot_map(view, map_title), use_container_width=True)
    st.caption("Gold = goals, cyan = other shots. Marker size ∝ xG. Hover for player & detail. Attacking right.")

    st.markdown("## By player")
    by_player = ts.groupby("player").agg(shots=("xg","count"),goals=("is_goal","sum"),
        xg=("xg","sum")).reset_index().sort_values("xg",ascending=False)
    by_player["xg"]=by_player["xg"].round(2)
    by_player["xg_per_shot"]=(by_player["xg"]/by_player["shots"]).round(3)
    st.dataframe(by_player, use_container_width=True, hide_index=True)


# ═══════════════════════════ MODELS & INSIGHTS ═════════════════════════════
elif page == "Models & insights":
    st.markdown("# Models & insights")
    st.markdown("Putting statistics behind the central question — is beating xG "
                "skill or luck, and does xG actually predict success?")
    st.markdown("---")

    # 1. xG difference predicts points
    st.markdown("## 1. Does xG difference predict points?")
    agg, fit1 = xgd_points_model(matches)
    st.plotly_chart(scatter_with_fit(agg["xgd"], agg["points"], agg["team"]+" "+agg["season"],
        fit1, "Points vs xG difference (all team-seasons)",
        "xG difference (xGF − xGA)", "League points"), use_container_width=True)
    if fit1.get("r2") == fit1.get("r2"):  # not NaN
        st.markdown(f"**Finding:** xG difference explains **{fit1['r2']*100:.0f}%** of the "
                    f"variation in points (R² = {fit1['r2']:.2f}, n = {fit1['n']}). "
                    "Underlying performance is a strong driver of results — xG is meaningful, not noise.")

    st.markdown("---")
    # 2. Finishing persistence
    st.markdown("## 2. Is beating xG repeatable, or luck?")
    st.caption("Each point is a team-season: finishing overperformance (goals − xG) per game "
               "in the first half of the season vs the second half. If finishing were a durable "
               "skill, points would line up on the diagonal.")
    persist, fit2 = finishing_persistence(matches)
    if not persist.empty and fit2:
        st.plotly_chart(scatter_with_fit(persist["h1_finishing"], persist["h2_finishing"],
            persist["team"]+" "+persist["season"], fit2,
            "First-half vs second-half finishing overperformance",
            "1st-half (goals − xG)/game", "2nd-half (goals − xG)/game"), use_container_width=True)
        rr = fit2.get("r2", float("nan"))
        verdict_txt = ("weak" if rr < 0.2 else ("moderate" if rr < 0.5 else "strong"))
        st.markdown(f"**Finding:** the correlation is **{verdict_txt}** "
            f"(R² = {rr:.2f}, n = {fit2['n']}). A low value means a team beating its xG in the "
            "first half tells you little about the second — supporting the idea that "
            "over/under-performance is largely **luck that regresses**, the heart of the "
            "'unlucky or just bad' question.")
    else:
        st.info("Not enough games per team in the loaded data to split seasons reliably.")

    st.markdown("---")
    # 3. Projection
    st.markdown(f"## 3. End-of-season projection — {season}")
    proj = project_points(matches, season)
    if not proj.empty and proj["played"].max() < 38:
        st.caption("Projects final points two ways: holding current form vs holding xPts pace. "
                   "Where they differ, regression toward the xPts line is the more likely path.")
        st.plotly_chart(projection_chart(proj), use_container_width=True)
        show = proj[["proj_rank","team","played","points","proj_on_form","proj_on_xg"]].copy()
        show.columns = ["Proj #","Team","P","Pts now","Proj (form)","Proj (xPts)"]
        st.dataframe(show, use_container_width=True, hide_index=True, height=560)
        st.download_button("⬇ Download projection (CSV)", show.to_csv(index=False),
                           f"projection_{season.replace('/','-')}.csv", "text/csv")
    else:
        st.info("Projection applies to in-progress seasons — this season looks complete.")

    st.caption("Methods use ordinary least-squares (numpy) and are intentionally simple "
               "and explainable. Small samples (a few dozen team-seasons) mean these are "
               "illustrative rather than definitive — see Methodology for caveats.")


# ═══════════════════════════════ METHODOLOGY ═══════════════════════════════
else:
    st.markdown("# Methodology")
    st.markdown("---")
    st.markdown("""
This dashboard is built to answer one question across every Premier League team:
**are results a fair reflection of performance, or is luck involved?**

### Expected Goals (xG)
Every shot is assigned a probability of becoming a goal based on its location and
type. Summed across a match, xG estimates how many goals a team *should* have scored
from the chances created. Comparing xG to actual goals reveals finishing quality and
luck. Data comes from **Understat**.

### Expected Points (xPts)
Understat simulates each match many times from the two teams' shot quality to estimate
how many league points the performance deserved. Summed across a season, the gap
between **actual points and xPts** is the clearest single measure of over- or
under-performance — the backbone of the *deserved* league table.

### PPDA (press intensity)
*Passes allowed per defensive action* — how many passes the opponent completes before
the team makes a tackle, interception or foul. **Lower = a more aggressive press.**

### Per-90 metrics & percentile radars
Player output is normalised per 90 minutes, then ranked as a percentile against
**positional peers across the whole league**, so each role is judged on the metrics
that matter for it: forwards on finishing and chance creation, **defenders on tackles,
interceptions, recoveries, aerials and blocks**, midfielders on a blend of both. This
makes radars genuinely role-appropriate rather than one-size-fits-all.

Defensive metrics come from **FBref** (player 'misc' stats), fuzzy-matched onto the
Understat attacking data by player name. Where FBref data isn't loaded, defender radars
fall back to build-up/progression metrics and say so.

### Transfer fit score
Each candidate's per-90 metrics are z-scored against positional peers, weighted by a
**system profile** (the demands of a role in a high-press, vertical setup), and squashed
to 0–100 with a logistic function. The weights encode a footballing model and live in
`analysis/transfers.py` — they're transparent and adjustable. Market value, contract
length and age come from **Transfermarkt** and feed a *fit-per-£m* value score.

### Player similarity
Cosine similarity on z-scored per-90 feature vectors within a position group — i.e.
which players have the most statistically similar style and output.

### Data-driven roles (clustering)
The *Squad & roles* page standardises each player's per-90 metrics and applies
**k-means clustering** to group players into playing-style archetypes, with **PCA**
projecting the high-dimensional profiles to a 2D style map. This redefines roles
from behaviour rather than nominal position — a more objective lens than a team
sheet. True positional heatmaps would need touch-level tracking data, which no
free source provides; the statistical fingerprint captures the same idea of *how*
a player actually plays.

### Statistical models
The *Models & insights* page adds inference on top of the descriptive views:
**(1)** an OLS regression of league points on xG difference quantifies how much
performance drives results (R²); **(2)** a split-half analysis correlates a team's
finishing overperformance in the first vs second half of a season — a low
correlation is evidence that beating xG is largely non-repeatable luck that
regresses; **(3)** a simple projection extrapolates final points from xPts-per-game.
All use ordinary least-squares (numpy) and are kept deliberately explainable.

### Data sources
**Understat** (xG, xPts, PPDA, shots), **FBref** (defensive stats), **Transfermarkt** (market value, contracts),
covering three seasons (2023/24 – 2025/26). All free for personal and educational use.

### Limitations
xG models differ between providers; Understat's is one of several. Expected points
assume shot-quality independence. Market values are point-in-time estimates. The fit
model is opinionated by design — it's a scouting *aid*, not an oracle.
""")

st.markdown("<br><p style='color:#445577;font-size:0.78rem;text-align:center;'>"
            "Data: Understat & Transfermarkt · Python · pandas · Plotly · Streamlit</p>",
            unsafe_allow_html=True)
