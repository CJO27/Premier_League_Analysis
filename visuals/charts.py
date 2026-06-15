"""visuals/charts.py — all Plotly figures for the dashboard."""
import plotly.graph_objects as go
import pandas as pd
import numpy as np

BG="#0D0D1A"; NAVY="#132257"; CYAN="#00D4FF"; WHITE="#FFFFFF"
RED="#FF4B4B"; LIGHT_RED="#FF9999"; GREY="#555577"; GOLD="#FFD700"; GREEN="#1D9E75"

_L = dict(paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(color=WHITE, family="'Inter', sans-serif", size=13),
    margin=dict(l=48,r=32,t=60,b=44),
    legend=dict(bgcolor="rgba(19,34,87,0.6)", bordercolor=NAVY, borderwidth=1, font=dict(size=11)),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor=GREY),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor=GREY))

def _lay(fig, title, **extra):
    lay = dict(**_L, title=dict(text=title, font=dict(size=16, color=CYAN))); lay.update(extra)
    fig.update_layout(**lay); return fig

def cumulative_points_chart(c):
    f=go.Figure()
    f.add_trace(go.Scatter(x=c["match_num"],y=c["cum_xpts"],mode="lines",line=dict(color=CYAN,width=0),showlegend=False,hoverinfo="skip"))
    f.add_trace(go.Scatter(x=c["match_num"],y=c["cum_actual_points"],mode="lines",line=dict(color=CYAN,width=0),fill="tonexty",fillcolor="rgba(255,75,75,0.15)",showlegend=False,hoverinfo="skip"))
    f.add_trace(go.Scatter(x=c["match_num"],y=c["cum_xpts"],name="Expected pts (xPts)",mode="lines",line=dict(color=CYAN,width=2.5,dash="dash"),hovertemplate="Match %{x}<br>xPts %{y:.1f}<extra></extra>"))
    f.add_trace(go.Scatter(x=c["match_num"],y=c["cum_actual_points"],name="Actual pts",mode="lines+markers",line=dict(color=WHITE,width=3),marker=dict(size=5),hovertemplate="Match %{x}<br>Pts %{y}<extra></extra>"))
    return _lay(f,"Actual points vs expected points (xPts)",xaxis_title="Match",yaxis_title="Cumulative points")

def xg_vs_actual_bar(x):
    f=go.Figure()
    f.add_trace(go.Bar(x=x["match_num"],y=x["xg_for"],name="xG for",marker_color=CYAN,opacity=0.55,customdata=x[["opponent","venue"]],hovertemplate="%{customdata[0]} (%{customdata[1]})<br>xG for %{y:.2f}<extra></extra>"))
    f.add_trace(go.Scatter(x=x["match_num"],y=x["goals_for"],name="Goals",mode="lines+markers",line=dict(color=WHITE,width=2),marker=dict(size=6)))
    f.add_trace(go.Bar(x=x["match_num"],y=-x["xg_against"],name="xG against",marker_color=RED,opacity=0.55,customdata=x[["opponent","venue"]],hovertemplate="%{customdata[0]} (%{customdata[1]})<br>xG against %{y:.2f}<extra></extra>"))
    f.add_trace(go.Scatter(x=x["match_num"],y=-x["goals_against"],name="Conceded",mode="lines+markers",line=dict(color=LIGHT_RED,width=2),marker=dict(size=6)))
    return _lay(f,"xG vs goals — match by match",barmode="overlay",xaxis_title="Match",yaxis_title="xG / goals (− = against)",shapes=[dict(type="line",x0=0,x1=len(x)+1,y0=0,y1=0,line=dict(color=GREY,width=1))])

def rolling_xg_diff_chart(t,window=5):
    colors=t["xg_diff_roll"].apply(lambda v: CYAN if v>=0 else RED)
    f=go.Figure(); f.add_hline(y=0,line=dict(color=GREY,width=1,dash="dot"))
    f.add_trace(go.Bar(x=t["match_num"],y=t["xg_diff_roll"],marker_color=colors,opacity=0.85,name="xG diff",hovertemplate="Match %{x}<br>%{y:.2f}<extra></extra>"))
    return _lay(f,f"Form — {window}-match rolling xG differential",xaxis_title="Match",yaxis_title="xG differential")

def ppda_chart(t):
    f=go.Figure()
    f.add_trace(go.Scatter(x=t["match_num"],y=t["ppda_roll"],mode="lines",line=dict(color=GOLD,width=2.5),name="PPDA",hovertemplate="Match %{x}<br>PPDA %{y:.2f}<extra></extra>"))
    return _lay(f,"Press intensity — PPDA (lower = more aggressive)",xaxis_title="Match",yaxis_title="Passes per defensive action")

def home_away_bars(s):
    metrics=[("avg_xg_for","xG for",CYAN,0.6),("avg_goals_for","Goals for",WHITE,0.9),("avg_xg_against","xG against",RED,0.6),("avg_goals_against","Goals against",LIGHT_RED,0.9)]
    f=go.Figure()
    for col,label,color,op in metrics:
        if col in s.columns:
            f.add_trace(go.Bar(name=label,x=s["venue"],y=s[col],marker_color=color,opacity=op,hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>"))
    return _lay(f,"Home vs away — per-game averages",barmode="group",yaxis_title="Goals / xG per game")

def player_xg_scatter(p):
    if p.empty: return _lay(go.Figure(),"Player goals vs xG (no data)")
    mx=max(p["xg"].max(),p["goals"].max())*1.15
    colors=p["g_minus_xg"].apply(lambda d: GOLD if d>=2 else (RED if d<=-2 else CYAN))
    f=go.Figure(); f.add_shape(type="line",x0=0,y0=0,x1=mx,y1=mx,line=dict(color=GREY,dash="dot",width=1.5))
    f.add_trace(go.Scatter(x=p["xg"],y=p["goals"],mode="markers+text",text=p["player"],textposition="top center",textfont=dict(size=10,color=WHITE),marker=dict(size=p["shots"].clip(lower=5)/2,color=colors,line=dict(width=1,color=NAVY),opacity=0.85),customdata=p["shots"],hovertemplate="<b>%{text}</b><br>xG %{x:.1f} · goals %{y}<br>shots %{customdata}<extra></extra>"))
    return _lay(f,"Player goals vs xG (bubble = shots)",xaxis=dict(**_L["xaxis"],range=[0,mx],title="xG"),yaxis=dict(**_L["yaxis"],range=[0,mx],title="Goals"))

def player_radar(rd,name,rd2=None,name2=None):
    labels=[d["label"] for d in rd]+[rd[0]["label"]]; vals=[d["value"] for d in rd]+[rd[0]["value"]]
    f=go.Figure()
    f.add_trace(go.Scatterpolar(r=vals,theta=labels,fill="toself",name=name,line=dict(color=CYAN,width=2),fillcolor="rgba(0,212,255,0.2)"))
    if rd2 and name2:
        cv=[d["value"] for d in rd2]+[rd2[0]["value"]]
        f.add_trace(go.Scatterpolar(r=cv,theta=labels,fill="toself",name=name2,line=dict(color=GOLD,width=2),fillcolor="rgba(255,215,0,0.15)"))
    f.update_layout(**_L,title=dict(text=f"Percentile profile — {name}",font=dict(size=15,color=CYAN)),polar=dict(bgcolor=BG,radialaxis=dict(range=[0,100],gridcolor="rgba(255,255,255,0.1)",tickfont=dict(size=9)),angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")))
    return f

def transfer_fit_chart(c):
    if c.empty: return _lay(go.Figure(),"Transfer fit (no candidates)")
    d=c.sort_values("fit_score").copy()
    colors=d["fit_score"].apply(lambda s: GREEN if s>=70 else (GOLD if s>=55 else GREY))
    if "market_value_m" in d.columns:
        def _lbl(row):
            v=row.get("market_value_m")
            return f"{row['team']}  £{v:.0f}m" if pd.notna(v) else f"{row['team']}"
        label=d.apply(_lbl, axis=1)
    else:
        label=d["team"]
    f=go.Figure(go.Bar(x=d["fit_score"],y=d["player"],orientation="h",marker_color=colors,text=label,textposition="inside",insidetextanchor="start",textfont=dict(color=WHITE,size=10),hovertemplate="<b>%{y}</b><br>Fit %{x:.0f}/100<extra></extra>"))
    return _lay(f,"System fit score by candidate",xaxis=dict(**_L["xaxis"],range=[0,100],title="Fit score"),yaxis_title="",height=max(300,42*len(d)))

def season_comparison_chart(c):
    f=go.Figure()
    f.add_trace(go.Bar(name="xPts",x=c["season"],y=c["xpts"],marker_color=CYAN,opacity=0.6))
    f.add_trace(go.Bar(name="Actual",x=c["season"],y=c["actual_pts"],marker_color=WHITE,opacity=0.9))
    return _lay(f,"Season-on-season: expected vs actual points",barmode="group",yaxis_title="Points")

# ── NEW: xG-based league table scatter ─────────────────────────────────────
def league_table_scatter(table):
    mx=max(table["xpts"].max(),table["points"].max())*1.1
    colors=table["pts_gap"].apply(lambda g: GREEN if g>3 else (RED if g<-3 else CYAN))
    f=go.Figure(); f.add_shape(type="line",x0=0,y0=0,x1=mx,y1=mx,line=dict(color=GREY,dash="dot",width=1.5))
    f.add_trace(go.Scatter(x=table["xpts"],y=table["points"],mode="markers+text",text=table["team"],textposition="top center",textfont=dict(size=9,color=WHITE),marker=dict(size=12,color=colors,line=dict(width=1,color=NAVY)),hovertemplate="<b>%{text}</b><br>xPts %{x:.1f} · Pts %{y}<extra></extra>"))
    return _lay(f,"Deserved vs actual points — above line = overperforming",xaxis=dict(**_L["xaxis"],range=[0,mx],title="Expected points (xPts)"),yaxis=dict(**_L["yaxis"],range=[0,mx],title="Actual points"),height=560)

# ── NEW: shot map ───────────────────────────────────────────────────────────
def shot_map(shots_df, title="Shot map"):
    base = {k:v for k,v in _L.items() if k not in ("xaxis","yaxis")}
    if shots_df.empty:
        f=go.Figure(); f.update_layout(**base,title=dict(text=title+" (no shot data)",font=dict(size=15,color=CYAN))); return f
    goals=shots_df[shots_df["is_goal"]]; misses=shots_df[~shots_df["is_goal"]]
    f=go.Figure()
    # pitch box (attacking half, attacking to the right)
    f.add_shape(type="rect",x0=50,y0=0,x1=100,y1=100,line=dict(color=GREY,width=1))
    f.add_shape(type="rect",x0=83,y0=21,x1=100,y1=79,line=dict(color=GREY,width=1))
    f.add_shape(type="rect",x0=94.2,y0=36,x1=100,y1=64,line=dict(color=GREY,width=1))
    def _cd(df):
        cols=["player","xg"]
        if "result" in df.columns: cols.append("result")
        if "minute" in df.columns: cols.append("minute")
        return df[cols]
    miss_cd=_cd(misses); goal_cd=_cd(goals)
    miss_tmpl="<b>%{customdata[0]}</b><br>xG %{customdata[1]:.2f}"+("<br>%{customdata[2]}" if "result" in misses.columns else "")+("<br>%{customdata[3]}'" if "minute" in misses.columns else "")+"<extra></extra>"
    goal_tmpl="<b>%{customdata[0]} — GOAL</b><br>xG %{customdata[1]:.2f}"+("<br>%{customdata[3]}'" if "minute" in goals.columns else "")+"<extra></extra>"
    f.add_trace(go.Scatter(x=misses["x"],y=misses["y"],mode="markers",name="Shot",marker=dict(size=misses["xg"]*30+4,color=CYAN,opacity=0.45,line=dict(width=0.5,color=NAVY)),customdata=miss_cd,hovertemplate=miss_tmpl))
    f.add_trace(go.Scatter(x=goals["x"],y=goals["y"],mode="markers",name="Goal",marker=dict(size=goals["xg"]*30+6,color=GOLD,opacity=0.9,line=dict(width=1,color=WHITE)),customdata=goal_cd,hovertemplate=goal_tmpl))
    f.update_layout(**base,title=dict(text=title,font=dict(size=15,color=CYAN)),xaxis=dict(range=[48,101],showgrid=False,zeroline=False,visible=False),yaxis=dict(range=[-2,102],showgrid=False,zeroline=False,visible=False,scaleanchor="x"),height=460)
    return f

# ── NEW: similarity bars ────────────────────────────────────────────────────
def similarity_chart(sim_df, target):
    if sim_df.empty: return _lay(go.Figure(),"Similar players (none found)")
    d=sim_df.sort_values("similarity")
    f=go.Figure(go.Bar(x=d["similarity"],y=d["player"],orientation="h",marker_color=CYAN,opacity=0.8,text=d["team"],textposition="inside",insidetextanchor="start",textfont=dict(color=WHITE,size=10),hovertemplate="<b>%{y}</b><br>%{x:.0f}% similar<extra></extra>"))
    return _lay(f,f"Players most similar to {target}",xaxis=dict(**_L["xaxis"],range=[0,100],title="Similarity %"),yaxis_title="",height=max(280,40*len(d)))


# ── NEW v4: generic radar compare (teams or seasons) ───────────────────────
def radar_compare(series_a, name_a, series_b=None, name_b=None, title="Profile"):
    if not series_a:
        return _lay(go.Figure(), title + " (no data)")
    labels=[d["label"] for d in series_a]+[series_a[0]["label"]]
    va=[d["value"] for d in series_a]+[series_a[0]["value"]]
    f=go.Figure()
    f.add_trace(go.Scatterpolar(r=va,theta=labels,fill="toself",name=name_a,
        line=dict(color=CYAN,width=2),fillcolor="rgba(0,212,255,0.2)"))
    if series_b and name_b:
        vb=[d["value"] for d in series_b]+[series_b[0]["value"]]
        f.add_trace(go.Scatterpolar(r=vb,theta=labels,fill="toself",name=name_b,
            line=dict(color=GOLD,width=2),fillcolor="rgba(255,215,0,0.15)"))
    f.update_layout(**_L,title=dict(text=title,font=dict(size=15,color=CYAN)),
        polar=dict(bgcolor=BG,radialaxis=dict(range=[0,100],gridcolor="rgba(255,255,255,0.1)",
        tickfont=dict(size=9)),angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")),
        showlegend=True)
    return f


# ── NEW v4: attack vs defence quadrant map ─────────────────────────────────
def attack_defence_quadrant(metrics_df, highlight=None):
    if metrics_df.empty:
        return _lay(go.Figure(), "Attack vs defence (no data)")
    avg_att=metrics_df["xg_for_pg"].mean(); avg_def=metrics_df["xg_against_pg"].mean()
    colors=metrics_df["team"].apply(lambda t: GOLD if t==highlight else CYAN)
    sizes=metrics_df["team"].apply(lambda t: 16 if t==highlight else 11)
    f=go.Figure()
    f.add_hline(y=avg_def,line=dict(color=GREY,width=1,dash="dot"))
    f.add_vline(x=avg_att,line=dict(color=GREY,width=1,dash="dot"))
    f.add_trace(go.Scatter(x=metrics_df["xg_for_pg"],y=metrics_df["xg_against_pg"],
        mode="markers+text",text=metrics_df["team"],textposition="top center",
        textfont=dict(size=9,color=WHITE),
        marker=dict(size=sizes,color=colors,line=dict(width=1,color=NAVY)),
        hovertemplate="<b>%{text}</b><br>xG for/g %{x:.2f}<br>xG against/g %{y:.2f}<extra></extra>"))
    # Note: y reversed so 'good defence' (low xGA) is at TOP
    return _lay(f,"League map — attack (right) vs defence (top)",
        xaxis=dict(**_L["xaxis"],title="xG for per game →"),
        yaxis=dict(**_L["yaxis"],title="← xG against per game (less = better)",autorange="reversed"),
        height=600,showlegend=False)


# ── NEW v4: cumulative overlay (two seasons / teams) ───────────────────────
def cumulative_compare(cum_a, label_a, cum_b, label_b, metric="cum_xpts", ytitle="Cumulative xPts"):
    f=go.Figure()
    f.add_trace(go.Scatter(x=cum_a["match_num"],y=cum_a[metric],name=label_a,mode="lines",
        line=dict(color=CYAN,width=2.5),hovertemplate="MD %{x}<br>%{y:.1f}<extra></extra>"))
    f.add_trace(go.Scatter(x=cum_b["match_num"],y=cum_b[metric],name=label_b,mode="lines",
        line=dict(color=GOLD,width=2.5),hovertemplate="MD %{x}<br>%{y:.1f}<extra></extra>"))
    return _lay(f,f"{ytitle} — {label_a} vs {label_b}",xaxis_title="Matchday",yaxis_title=ytitle)


# ── NEW v4: leaderboard horizontal bar ─────────────────────────────────────
def leaderboard_bar(df, value_col, label_col, title, color=CYAN, fmt="%{x:.1f}"):
    if df.empty:
        return _lay(go.Figure(), title+" (no data)")
    d=df.sort_values(value_col)
    sub=list(d["team"]) if "team" in d.columns else [""]*len(d)
    vals=d[value_col].astype(float)
    vmin, vmax = float(vals.min()), float(vals.max())
    # Robust axis: never collapse to plotly's default [-1,1] when all values equal
    if vmin >= 0:
        x0, x1 = 0, (vmax*1.20 if vmax > 0 else 1)
    else:
        pad=(vmax-vmin)*0.20 or 1
        x0, x1 = vmin-pad, vmax+pad
    # Value labels OUTSIDE the bar so even zero-width bars stay readable
    labels=[f"{v:.2f}" for v in vals]
    f=go.Figure(go.Bar(x=vals,y=d[label_col],orientation="h",marker_color=color,
        opacity=0.85,customdata=sub,text=labels,textposition="outside",
        textfont=dict(color=WHITE,size=10),cliponaxis=False,
        hovertemplate="<b>%{y}</b> (%{customdata})<br>"+fmt+"<extra></extra>"))
    return _lay(f,title,xaxis=dict(**_L["xaxis"],range=[x0,x1]),
        yaxis_title="",height=max(280,38*len(d)))


# ── NEW v4: grouped season/team metric bars ────────────────────────────────
def metric_compare_bars(metrics_a, name_a, metrics_b, name_b, title="Comparison"):
    labels=list(metrics_a.keys())
    f=go.Figure()
    f.add_trace(go.Bar(name=name_a,x=labels,y=[metrics_a[k] for k in labels],
        marker_color=CYAN,opacity=0.8))
    f.add_trace(go.Bar(name=name_b,x=labels,y=[metrics_b.get(k,0) for k in labels],
        marker_color=GOLD,opacity=0.8))
    return _lay(f,title,barmode="group",yaxis_title="")


# ── NEW v6: multi-season trajectory (points & xPts over seasons) ───────────
def trajectory_chart(comp_df, title="Season trajectory"):
    if comp_df.empty or len(comp_df) < 2:
        return _lay(go.Figure(), title + " (need 2+ seasons)")
    d = comp_df.sort_values("season")
    f=go.Figure()
    f.add_trace(go.Scatter(x=d["season"],y=d["actual_pts"],name="Actual points",mode="lines+markers",
        line=dict(color=WHITE,width=3),marker=dict(size=9)))
    f.add_trace(go.Scatter(x=d["season"],y=d["xpts"],name="Expected points (xPts)",mode="lines+markers",
        line=dict(color=CYAN,width=3,dash="dash"),marker=dict(size=9)))
    return _lay(f,title,xaxis_title="Season",yaxis_title="Points")


def xg_trajectory_chart(comp_df, title="xG trajectory"):
    if comp_df.empty or len(comp_df) < 2:
        return _lay(go.Figure(), title + " (need 2+ seasons)")
    d = comp_df.sort_values("season")
    f=go.Figure()
    f.add_trace(go.Scatter(x=d["season"],y=d["xg_for"],name="xG for",mode="lines+markers",
        line=dict(color=GREEN,width=3),marker=dict(size=9)))
    f.add_trace(go.Scatter(x=d["season"],y=d["xg_against"],name="xG against",mode="lines+markers",
        line=dict(color=RED,width=3),marker=dict(size=9)))
    return _lay(f,title,xaxis_title="Season",yaxis_title="Expected goals (season total)")


# ── NEW v6: player development across seasons ──────────────────────────────
def player_development_chart(dev_df, player_name):
    """dev_df: per-season rows with goals, xg, assists, xa."""
    if dev_df.empty or len(dev_df) < 2:
        return _lay(go.Figure(), f"{player_name} development (need 2+ seasons)")
    d = dev_df.sort_values("season_label")
    f=go.Figure()
    f.add_trace(go.Scatter(x=d["season_label"],y=d["goals"],name="Goals",mode="lines+markers",line=dict(color=WHITE,width=2.5),marker=dict(size=8)))
    f.add_trace(go.Scatter(x=d["season_label"],y=d["xg"],name="xG",mode="lines+markers",line=dict(color=CYAN,width=2.5,dash="dash"),marker=dict(size=8)))
    f.add_trace(go.Scatter(x=d["season_label"],y=d["assists"],name="Assists",mode="lines+markers",line=dict(color=GOLD,width=2.5),marker=dict(size=8)))
    f.add_trace(go.Scatter(x=d["season_label"],y=d["xa"],name="xA",mode="lines+markers",line=dict(color="#FFB347",width=2.5,dash="dash"),marker=dict(size=8)))
    return _lay(f,f"{player_name} — output by season",xaxis_title="Season",yaxis_title="Goals / xG / Assists / xA")


# ── NEW v8: scatter with OLS fit line + R² annotation ──────────────────────
def scatter_with_fit(x, y, labels, fit, title, xlabel, ylabel):
    import numpy as np
    f=go.Figure()
    f.add_trace(go.Scatter(x=x,y=y,mode="markers",name="Teams",
        text=labels,marker=dict(size=10,color=CYAN,line=dict(width=1,color=NAVY)),
        hovertemplate="<b>%{text}</b><br>%{x:.2f}, %{y:.1f}<extra></extra>"))
    if fit and not np.isnan(fit.get("slope", np.nan)):
        xs=np.linspace(min(x),max(x),50); ys=fit["slope"]*xs+fit["intercept"]
        f.add_trace(go.Scatter(x=xs,y=ys,mode="lines",name="Best fit",
            line=dict(color=GOLD,width=2,dash="dash"),hoverinfo="skip"))
        f.add_annotation(x=0.05,y=0.95,xref="paper",yref="paper",showarrow=False,
            text=f"R² = {fit['r2']:.2f}  ·  r = {fit['r']:.2f}  ·  n = {fit['n']}",
            font=dict(color=GOLD,size=13),align="left",
            bgcolor="rgba(19,34,87,0.7)",borderpad=6)
    return _lay(f,title,xaxis_title=xlabel,yaxis_title=ylabel,height=520)


# ── NEW v8: projection chart (current vs projected points) ─────────────────
def projection_chart(proj_df):
    if proj_df.empty:
        return _lay(go.Figure(),"Projection (no data)")
    d=proj_df.sort_values("proj_on_xg")
    f=go.Figure()
    f.add_trace(go.Bar(y=d["team"],x=d["points"],orientation="h",name="Points so far",
        marker_color=NAVY,opacity=0.9,hovertemplate="%{y}: %{x}<extra></extra>"))
    f.add_trace(go.Scatter(y=d["team"],x=d["proj_on_xg"],mode="markers",name="Projected (xPts pace)",
        marker=dict(color=CYAN,size=10,symbol="diamond"),
        hovertemplate="%{y}: proj %{x:.0f}<extra></extra>"))
    f.add_trace(go.Scatter(y=d["team"],x=d["proj_on_form"],mode="markers",name="Projected (current form)",
        marker=dict(color=GOLD,size=9,symbol="circle-open"),
        hovertemplate="%{y}: proj %{x:.0f}<extra></extra>"))
    return _lay(f,"Projected final points — current pace vs xPts pace",
        xaxis_title="Points",yaxis_title="",height=max(360,30*len(d)),barmode="overlay")


# ── NEW v9: role style-map (PCA scatter coloured by cluster) ───────────────
ROLE_PALETTE = ["#00D4FF","#FFD700","#1D9E75","#FF4B4B","#B388FF","#FF9F1C",
                "#4ECDC4","#FF6B9D","#A0E548","#7FB3FF"]

def role_style_map(roles_df, highlight=None):
    if "pca_x" not in roles_df.columns:
        return _lay(go.Figure(), "Style map (clustering unavailable)")
    f=go.Figure()
    roles=sorted(roles_df["role"].unique())
    for i,role in enumerate(roles):
        sub=roles_df[roles_df["role"]==role]
        f.add_trace(go.Scatter(x=sub["pca_x"],y=sub["pca_y"],mode="markers",name=role,
            marker=dict(size=8,color=ROLE_PALETTE[i%len(ROLE_PALETTE)],
                line=dict(width=0.5,color=NAVY),opacity=0.8),
            text=sub["player"]+" ("+sub["team"]+")",
            hovertemplate="<b>%{text}</b><br>"+role+"<extra></extra>"))
    if highlight is not None and highlight in roles_df["player"].values:
        h=roles_df[roles_df["player"]==highlight]
        f.add_trace(go.Scatter(x=h["pca_x"],y=h["pca_y"],mode="markers+text",
            text=h["player"],textposition="top center",textfont=dict(color=WHITE,size=11),
            marker=dict(size=16,color="white",symbol="star",line=dict(width=1,color=NAVY)),
            showlegend=False,hoverinfo="skip"))
    base={k:v for k,v in _L.items() if k not in ("xaxis","yaxis")}
    f.update_layout(**base,title=dict(text="Player style map — clustered by statistical profile",
        font=dict(size=15,color=CYAN)),
        xaxis=dict(title="Style axis 1 (PCA)",showgrid=False,zeroline=False,
            gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Style axis 2 (PCA)",showgrid=False,zeroline=False,
            gridcolor="rgba(255,255,255,0.06)"),height=620)
    return f


# ── NEW v9: best XI on a pitch ─────────────────────────────────────────────
def best_xi_pitch(xi, title="Best XI"):
    base={k:v for k,v in _L.items() if k not in ("xaxis","yaxis")}
    f=go.Figure()
    # pitch outline (vertical, attacking up)
    f.add_shape(type="rect",x0=0,y0=0,x1=100,y1=100,line=dict(color=GREY,width=1.2))
    f.add_shape(type="line",x0=0,y0=50,x1=100,y1=50,line=dict(color=GREY,width=0.8))
    f.add_shape(type="circle",x0=38,y0=40,x1=62,y1=60,line=dict(color=GREY,width=0.8))
    f.add_shape(type="rect",x0=22,y0=0,x1=78,y1=16,line=dict(color=GREY,width=0.8))
    f.add_shape(type="rect",x0=22,y0=84,x1=78,y1=100,line=dict(color=GREY,width=0.8))
    if not xi:
        f.update_layout(**base,title=dict(text=title+" (no data)",font=dict(size=15,color=CYAN)),
            xaxis=dict(range=[-5,105],visible=False),yaxis=dict(range=[-5,105],visible=False),height=620)
        return f
    xs=[p["x"] for p in xi]; ys=[p["y"] for p in xi]
    names=[f"{p['player']}<br><span style='font-size:9px'>{p['rating']:.0f}</span>" for p in xi]
    hover=[f"<b>{p['player']}</b> ({p['team']})<br>Rating {p['rating']:.0f}"+
           (f"<br>{p['role']}" if p.get('role') else "") for p in xi]
    f.add_trace(go.Scatter(x=xs,y=ys,mode="markers+text",text=names,textposition="bottom center",
        textfont=dict(color=WHITE,size=10),
        marker=dict(size=26,color=CYAN,line=dict(width=1.5,color=WHITE),opacity=0.9),
        customdata=hover,hovertemplate="%{customdata}<extra></extra>"))
    f.update_layout(**base,title=dict(text=title,font=dict(size=15,color=CYAN)),
        xaxis=dict(range=[-5,105],showgrid=False,zeroline=False,visible=False),
        yaxis=dict(range=[-8,108],showgrid=False,zeroline=False,visible=False),
        height=680,showlegend=False)
    return f
