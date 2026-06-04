import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ITX · Food Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ───────────────────────────────────────────────────────────────
COLORS = {
    "bg":         "#0A0A0F",
    "surface":    "#111118",
    "surface2":   "#1A1A24",
    "border":     "#2A2A3A",
    "accent":     "#FF6B35",
    "accent2":    "#00C9A7",
    "accent3":    "#845EF7",
    "text":       "#F0EEF6",
    "muted":      "#8B899E",
    "positive":   "#00C9A7",
    "negative":   "#FF4757",
    "warning":    "#FFA502",
}

CHART_PALETTE = ["#FF6B35", "#00C9A7", "#845EF7", "#FFD93D", "#FF6B9D",
                 "#4ECDC4", "#A8EDEA", "#FF8B94", "#6C5CE7", "#FDCB6E"]

# ── CSS ─────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {{
    background-color: {COLORS['bg']};
    color: {COLORS['text']};
    font-family: 'DM Sans', sans-serif;
  }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background: {COLORS['surface']};
    border-right: 1px solid {COLORS['border']};
  }}
  section[data-testid="stSidebar"] .block-container {{
    padding-top: 2rem;
  }}

  /* Hide default header */
  header[data-testid="stHeader"] {{ display: none; }}
  .block-container {{ padding-top: 0rem; padding-bottom: 0rem; }}
  footer {{ display: none; }}

  /* Metrics */
  [data-testid="metric-container"] {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    transition: border-color 0.2s;
  }}
  [data-testid="metric-container"]:hover {{
    border-color: {COLORS['accent']};
  }}
  [data-testid="stMetricLabel"] {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {COLORS['muted']};
  }}
  [data-testid="stMetricValue"] {{
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: {COLORS['text']};
  }}
  [data-testid="stMetricDelta"] {{
    font-size: 0.8rem;
  }}

  /* Selectbox & slider */
  .stSelectbox > div > div {{
    background: {COLORS['surface2']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    color: {COLORS['text']};
  }}
  .stMultiSelect > div > div {{
    background: {COLORS['surface2']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
  }}
  .stSlider [data-baseweb="slider"] {{
    margin-top: 0.5rem;
  }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    background: {COLORS['surface']};
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid {COLORS['border']};
  }}
  .stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 8px;
    color: {COLORS['muted']};
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.875rem;
    padding: 0.5rem 1.25rem;
    border: none;
  }}
  .stTabs [aria-selected="true"] {{
    background: {COLORS['surface2']};
    color: {COLORS['text']};
  }}

  /* Buttons */
  .stButton > button {{
    background: {COLORS['accent']};
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.65rem 1.75rem;
    letter-spacing: 0.02em;
    transition: all 0.2s;
    width: 100%;
  }}
  .stButton > button:hover {{
    background: #ff8055;
    transform: translateY(-1px);
  }}

  /* Dividers */
  hr {{
    border-color: {COLORS['border']};
    margin: 1.5rem 0;
  }}

  /* Labels */
  label, .stMarkdown p {{
    color: {COLORS['muted']};
    font-size: 0.82rem;
  }}

  /* Sidebar labels */
  section[data-testid="stSidebar"] label {{
    color: {COLORS['muted']};
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }}

  /* Scrollbar */
  ::-webkit-scrollbar {{ width: 5px; }}
  ::-webkit-scrollbar-track {{ background: {COLORS['bg']}; }}
  ::-webkit-scrollbar-thumb {{ background: {COLORS['border']}; border-radius: 10px; }}

  /* Prediction card */
  .pred-card {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1rem;
  }}
  .pred-result {{
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0;
  }}
  .pred-prob {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: {COLORS['muted']};
    margin-top: 0.25rem;
  }}

  /* Section header */
  .section-header {{
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: {COLORS['text']};
    letter-spacing: -0.01em;
    margin-bottom: 0.1rem;
  }}
  .section-sub {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    color: {COLORS['muted']};
    margin-bottom: 1rem;
  }}

  /* Top bar */
  .top-bar {{
    background: {COLORS['surface']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 1rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 999;
  }}
  .brand {{
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
  }}
  .brand span {{
    color: {COLORS['accent']};
  }}
  .badge {{
    background: rgba(255,107,53,0.12);
    color: {COLORS['accent']};
    border: 1px solid rgba(255,107,53,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
  }}
  .stat-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: {COLORS['surface2']};
    border: 1px solid {COLORS['border']};
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.75rem;
    color: {COLORS['muted']};
    font-family: 'JetBrains Mono', monospace;
  }}
  .dot-live {{
    width: 6px; height: 6px;
    background: {COLORS['positive']};
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s infinite;
  }}
  @keyframes pulse {{
    0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }}
  }}
</style>
""", unsafe_allow_html=True)

# ── Helper ───────────────────────────────────────────────────────────────────────
def plotly_theme(fig, height=360):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=COLORS["muted"], size=12),
        height=height,
        margin=dict(l=0, r=0, t=32, b=0),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=COLORS["border"],
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(
            gridcolor=COLORS["border"],
            tickcolor=COLORS["muted"],
            linecolor=COLORS["border"],
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=COLORS["border"],
            tickcolor=COLORS["muted"],
            linecolor=COLORS["border"],
            zeroline=False,
        ),
        colorway=CHART_PALETTE,
    )
    return fig

def metric_delta(val, ref, fmt="{:.1f}"):
    delta = val - ref
    pct = (delta / ref * 100) if ref != 0 else 0
    return f"{'+' if pct>=0 else ''}{pct:.1f}%"

# ── Data loading ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists("data/food_delivery_data.csv"):
        st.error("Run generate_data.py first.")
        st.stop()
    df = pd.read_csv("data/food_delivery_data.csv", parse_dates=["order_date"])
    return df

@st.cache_resource
def load_model():
    model = joblib.load("models/reorder_model.pkl")
    le_cuisine = joblib.load("models/le_cuisine.pkl")
    le_time = joblib.load("models/le_time.pkl")
    le_payment = joblib.load("models/le_payment.pkl")
    le_city = joblib.load("models/le_city.pkl")
    with open("models/model_meta.json") as f:
        meta = json.load(f)
    return model, le_cuisine, le_time, le_payment, le_city, meta

df_raw = load_data()
model, le_cuisine, le_time, le_payment, le_city, meta = load_model()

# ── Top bar ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-bar">
  <div style="display:flex;align-items:center;gap:1rem">
    <div class="brand">Pulse<span>AI</span></div>
    <div class="badge">FOOD INTELLIGENCE</div>
  </div>
  <div style="display:flex;align-items:center;gap:0.75rem">
    <div class="stat-pill"><span class="dot-live"></span> Live · 15,000 orders</div>
    <div class="stat-pill">Model acc · {meta['accuracy']}%</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='margin-bottom:1.5rem'>
      <div style='font-family:Syne;font-size:0.9rem;font-weight:700;
                  color:{COLORS["text"]};margin-bottom:0.25rem'>
        Dashboard Filters
      </div>
      <div style='font-size:0.75rem;color:{COLORS["muted"]}'>Refine the analysis</div>
    </div>
    """, unsafe_allow_html=True)

    all_cities = sorted(df_raw["city"].unique())
    sel_cities = st.multiselect("Cities", all_cities, default=all_cities,
                                 key="cities_filter")

    all_cuisines = sorted(df_raw["cuisine"].unique())
    sel_cuisines = st.multiselect("Cuisines", all_cuisines, default=all_cuisines,
                                   key="cuisine_filter")

    order_range = st.slider("Order value (₹)", 50, 1000,
                             (50, 1000), step=25, key="order_slider")

    time_opts = sorted(df_raw["time_of_day"].unique())
    sel_times = st.multiselect("Time of day", time_opts, default=time_opts,
                                key="time_filter")

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.72rem;color:{COLORS["muted"]};line-height:1.6'>
      <div style='font-family:Syne;font-size:0.8rem;font-weight:700;
                  color:{COLORS["text"]};margin-bottom:0.4rem'>About</div>
      AI-powered decision dashboard built on 15K synthetic food delivery orders.
      Model: Gradient Boosting · Accuracy: {meta['accuracy']}%
    </div>
    """, unsafe_allow_html=True)

# ── Filtered data ────────────────────────────────────────────────────────────────
df = df_raw[
    df_raw["city"].isin(sel_cities if sel_cities else all_cities) &
    df_raw["cuisine"].isin(sel_cuisines if sel_cuisines else all_cuisines) &
    df_raw["order_value"].between(order_range[0], order_range[1]) &
    df_raw["time_of_day"].isin(sel_times if sel_times else time_opts)
].copy()

if df.empty:
    st.warning("No data matches current filters. Adjust the sidebar.")
    st.stop()

# ── KPI Row ──────────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total_orders = len(df)
total_rev = df["order_value"].sum()
avg_delivery = df["delivery_time_mins"].mean()
avg_rating = df["rating"].mean()
reorder_rate = df["will_reorder"].mean() * 100

# Baseline from full dataset for deltas
base = df_raw
b_orders = len(base)
b_rev = base["order_value"].sum()
b_del = base["delivery_time_mins"].mean()
b_rat = base["rating"].mean()
b_reorder = base["will_reorder"].mean() * 100

with col1:
    st.metric("Total orders", f"{total_orders:,}",
              metric_delta(total_orders, b_orders))
with col2:
    st.metric("Revenue", f"₹{total_rev/1e6:.2f}M",
              metric_delta(total_rev, b_rev))
with col3:
    st.metric("Avg delivery", f"{avg_delivery:.0f} min",
              metric_delta(avg_delivery, b_del))
with col4:
    st.metric("Avg rating", f"{avg_rating:.2f} ★",
              metric_delta(avg_rating, b_rat))
with col5:
    st.metric("Reorder rate", f"{reorder_rate:.1f}%",
              metric_delta(reorder_rate, b_reorder))

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  📊  Overview  ",
    "  🗺️  City & Cuisine  ",
    "  ⏱️  Operations  ",
    "  🤖  AI Predictor  ",
])

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — OVERVIEW                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    row1c1, row1c2 = st.columns([3, 2])

    # Revenue by month
    with row1c1:
        st.markdown('<div class="section-header">Monthly revenue trend</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Total order value · all cuisines</div>', unsafe_allow_html=True)

        monthly = df.groupby("month")["order_value"].sum().reset_index()
        monthly["month_name"] = monthly["month"].apply(
            lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["month_name"], y=monthly["order_value"],
            mode="lines+markers",
            line=dict(color=COLORS["accent"], width=2.5, shape="spline"),
            marker=dict(size=6, color=COLORS["accent"],
                        line=dict(color=COLORS["bg"], width=2)),
            fill="tozeroy",
            fillcolor=f"rgba(255,107,53,0.08)",
            name="Revenue",
        ))
        fig = plotly_theme(fig, height=310)
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="", tickprefix="₹", tickformat=".2s")
        st.plotly_chart(fig, use_container_width=True)

    # Order distribution by cuisine
    with row1c2:
        st.markdown('<div class="section-header">Cuisine share</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Orders by food category</div>', unsafe_allow_html=True)

        cuisine_cnt = df.groupby("cuisine").size().reset_index(name="count").sort_values("count", ascending=False)
        fig2 = go.Figure(go.Pie(
            labels=cuisine_cnt["cuisine"],
            values=cuisine_cnt["count"],
            hole=0.62,
            marker=dict(colors=CHART_PALETTE, line=dict(color=COLORS["bg"], width=2)),
            textinfo="percent",
            textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>%{value:,} orders<br>%{percent}<extra></extra>",
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color=COLORS["muted"]),
            height=310,
            margin=dict(l=0, r=0, t=32, b=0),
            legend=dict(
                bgcolor="rgba(0,0,0,0)", font=dict(size=10),
                orientation="v", x=1.0,
            ),
            annotations=[dict(
                text=f"<b>{total_orders:,}</b><br>orders",
                x=0.5, y=0.5, font=dict(size=13, family="Syne", color=COLORS["text"]),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    row2c1, row2c2, row2c3 = st.columns(3)

    # Top restaurants
    with row2c1:
        st.markdown('<div class="section-header">Top restaurants</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">By total revenue</div>', unsafe_allow_html=True)

        top_rest = (df.groupby("restaurant")["order_value"]
                    .sum().reset_index()
                    .sort_values("order_value", ascending=True)
                    .tail(8))
        fig3 = go.Figure(go.Bar(
            y=top_rest["restaurant"], x=top_rest["order_value"],
            orientation="h",
            marker=dict(
                color=top_rest["order_value"],
                colorscale=[[0, COLORS["surface2"]], [1, COLORS["accent"]]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}<extra></extra>",
        ))
        fig3 = plotly_theme(fig3, height=320)
        fig3.update_xaxes(title_text="", tickprefix="₹", tickformat=".2s")
        fig3.update_yaxes(title_text="")
        st.plotly_chart(fig3, use_container_width=True)

    # Rating distribution
    with row2c2:
        st.markdown('<div class="section-header">Rating distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Customer satisfaction spread</div>', unsafe_allow_html=True)

        fig4 = go.Figure(go.Histogram(
            x=df["rating"], nbinsx=20,
            marker=dict(
                color=COLORS["accent3"],
                line=dict(color=COLORS["bg"], width=1),
            ),
            hovertemplate="Rating: %{x}<br>Count: %{y}<extra></extra>",
        ))
        fig4 = plotly_theme(fig4, height=320)
        fig4.update_xaxes(title_text="Rating")
        fig4.update_yaxes(title_text="Orders")
        st.plotly_chart(fig4, use_container_width=True)

    # Payment method
    with row2c3:
        st.markdown('<div class="section-header">Payment methods</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Transaction type split</div>', unsafe_allow_html=True)

        pay_cnt = df.groupby("payment_method").size().reset_index(name="count").sort_values("count")
        clrs = [COLORS["accent3"], COLORS["accent2"], COLORS["accent"], COLORS["warning"]]
        fig5 = go.Figure(go.Bar(
            x=pay_cnt["payment_method"], y=pay_cnt["count"],
            marker=dict(color=clrs, line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>%{y:,} orders<extra></extra>",
        ))
        fig5 = plotly_theme(fig5, height=320)
        fig5.update_xaxes(title_text="")
        fig5.update_yaxes(title_text="Orders")
        st.plotly_chart(fig5, use_container_width=True)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — CITY & CUISINE                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Revenue by city</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Total order value per market</div>', unsafe_allow_html=True)

        city_rev = (df.groupby("city").agg(
            revenue=("order_value", "sum"),
            orders=("order_id", "count"),
            avg_rating=("rating", "mean"),
        ).reset_index().sort_values("revenue", ascending=False))

        fig = go.Figure(go.Bar(
            x=city_rev["city"], y=city_rev["revenue"],
            marker=dict(
                color=city_rev["revenue"],
                colorscale=[[0, COLORS["surface2"]], [1, COLORS["accent2"]]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
        ))
        fig = plotly_theme(fig, height=320)
        fig.update_yaxes(tickprefix="₹", tickformat=".2s")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Avg order value by cuisine</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Spend per order category</div>', unsafe_allow_html=True)

        cuisine_aov = (df.groupby("cuisine")["order_value"]
                       .mean().reset_index()
                       .sort_values("order_value", ascending=True))
        fig2 = go.Figure(go.Bar(
            y=cuisine_aov["cuisine"], x=cuisine_aov["order_value"],
            orientation="h",
            marker=dict(
                color=cuisine_aov["order_value"],
                colorscale=[[0, COLORS["surface2"]], [1, COLORS["accent3"]]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{y}</b><br>AOV: ₹%{x:,.0f}<extra></extra>",
        ))
        fig2 = plotly_theme(fig2, height=320)
        fig2.update_xaxes(tickprefix="₹")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">City × Cuisine heatmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Order volume across markets and food categories</div>', unsafe_allow_html=True)

    heatmap_data = df.groupby(["city", "cuisine"]).size().reset_index(name="count")
    heatmap_pivot = heatmap_data.pivot(index="city", columns="cuisine", values="count").fillna(0)

    fig3 = go.Figure(go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns.tolist(),
        y=heatmap_pivot.index.tolist(),
        colorscale=[[0, COLORS["surface2"]], [0.5, COLORS["accent3"]], [1, COLORS["accent"]]],
        hovertemplate="<b>%{y} · %{x}</b><br>Orders: %{z:,.0f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            thickness=12, len=0.8,
            tickfont=dict(color=COLORS["muted"], size=10),
            bgcolor="rgba(0,0,0,0)",
            bordercolor=COLORS["border"],
        ),
    ))
    fig3 = plotly_theme(fig3, height=340)
    fig3.update_layout(margin=dict(l=0, r=60, t=16, b=0))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-header">Reorder rate by city</div>', unsafe_allow_html=True)
        city_reorder = (df.groupby("city")["will_reorder"]
                        .mean().mul(100).reset_index()
                        .sort_values("will_reorder", ascending=False))
        fig4 = go.Figure(go.Bar(
            x=city_reorder["city"], y=city_reorder["will_reorder"],
            marker=dict(
                color=city_reorder["will_reorder"],
                colorscale=[[0, COLORS["surface2"]], [1, COLORS["positive"]]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{x}</b><br>Reorder rate: %{y:.1f}%<extra></extra>",
        ))
        fig4 = plotly_theme(fig4, height=280)
        fig4.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig4, use_container_width=True)

    with c4:
        st.markdown('<div class="section-header">Avg rating by cuisine</div>', unsafe_allow_html=True)
        cuisine_rat = (df.groupby("cuisine")["rating"]
                       .mean().reset_index()
                       .sort_values("rating", ascending=False))
        fig5 = go.Figure(go.Bar(
            x=cuisine_rat["cuisine"], y=cuisine_rat["rating"],
            marker=dict(
                color=cuisine_rat["rating"],
                colorscale=[[0, COLORS["surface2"]], [1, COLORS["warning"]]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{x}</b><br>Rating: %{y:.2f}★<extra></extra>",
        ))
        fig5 = plotly_theme(fig5, height=280)
        fig5.update_yaxes(range=[3.5, 5])
        fig5.update_xaxes(tickangle=-30)
        st.plotly_chart(fig5, use_container_width=True)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — OPERATIONS                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab3:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Delivery time by city</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Median delivery minutes · market comparison</div>', unsafe_allow_html=True)

        del_city = (df.groupby("city")["delivery_time_mins"]
                    .agg(["median", "mean", "std"]).reset_index()
                    .sort_values("median"))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=del_city["city"], y=del_city["median"],
            name="Median", marker=dict(color=COLORS["accent2"], line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>Median: %{y:.0f} min<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=del_city["city"], y=del_city["mean"],
            name="Mean", mode="markers",
            marker=dict(color=COLORS["accent"], size=8, symbol="diamond"),
            hovertemplate="<b>%{x}</b><br>Mean: %{y:.1f} min<extra></extra>",
        ))
        fig = plotly_theme(fig, height=320)
        fig.update_yaxes(title_text="Minutes")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Orders by time of day</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Demand concentration · peak discovery</div>', unsafe_allow_html=True)

        order_by = ["Morning", "Lunch", "Evening", "Late Night"]
        time_orders = df.groupby("time_of_day").size().reindex(order_by, fill_value=0).reset_index(name="count")
        clrs_t = [COLORS["warning"], COLORS["accent2"], COLORS["accent"], COLORS["accent3"]]
        fig2 = go.Figure(go.Bar(
            x=time_orders["time_of_day"], y=time_orders["count"],
            marker=dict(color=clrs_t, line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>%{y:,} orders<extra></extra>",
        ))
        fig2 = plotly_theme(fig2, height=320)
        fig2.update_yaxes(title_text="Orders")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)

    with c3:
        st.markdown('<div class="section-header">Weekend vs weekday</div>', unsafe_allow_html=True)
        wknd = df.groupby("is_weekend").agg(
            orders=("order_id","count"),
            revenue=("order_value","sum"),
            avg_val=("order_value","mean"),
        ).reset_index()
        wknd["label"] = wknd["is_weekend"].map({0:"Weekday", 1:"Weekend"})

        fig3 = go.Figure(go.Bar(
            x=wknd["label"], y=wknd["orders"],
            marker=dict(color=[COLORS["surface2"], COLORS["accent"]], line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>%{y:,} orders<extra></extra>",
        ))
        fig3 = plotly_theme(fig3, height=270)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<div class="section-header">Promo impact</div>', unsafe_allow_html=True)
        promo = df.groupby("promo_used").agg(
            avg_val=("order_value","mean"),
            reorder=("will_reorder","mean"),
        ).reset_index()
        promo["label"] = promo["promo_used"].map({0:"No promo", 1:"With promo"})

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=promo["label"], y=promo["avg_val"],
            name="Avg order value",
            marker=dict(color=[COLORS["surface2"], COLORS["accent2"]], line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>AOV: ₹%{y:,.0f}<extra></extra>",
        ))
        fig4 = plotly_theme(fig4, height=270)
        fig4.update_yaxes(tickprefix="₹")
        st.plotly_chart(fig4, use_container_width=True)

    with c5:
        st.markdown('<div class="section-header">Items per order</div>', unsafe_allow_html=True)
        items = df.groupby("num_items").size().reset_index(name="count")
        fig5 = go.Figure(go.Scatter(
            x=items["num_items"], y=items["count"],
            mode="lines+markers",
            line=dict(color=COLORS["accent3"], width=2.5, shape="spline"),
            marker=dict(size=6, color=COLORS["accent3"],
                        line=dict(color=COLORS["bg"], width=2)),
            fill="tozeroy", fillcolor="rgba(132,94,247,0.08)",
            hovertemplate="<b>%{x} items</b><br>%{y:,} orders<extra></extra>",
        ))
        fig5 = plotly_theme(fig5, height=270)
        fig5.update_xaxes(title_text="No. of items")
        fig5.update_yaxes(title_text="Orders")
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Delivery time vs rating scatter</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How speed correlates with customer satisfaction</div>', unsafe_allow_html=True)

    sample = df.sample(min(2000, len(df)), random_state=1)
    fig6 = go.Figure(go.Scatter(
        x=sample["delivery_time_mins"], y=sample["rating"],
        mode="markers",
        marker=dict(
            size=4, opacity=0.5,
            color=sample["order_value"],
            colorscale=[[0, COLORS["accent3"]], [0.5, COLORS["accent2"]], [1, COLORS["accent"]]],
            showscale=True,
            colorbar=dict(
                title="₹ Value", thickness=10,
                tickfont=dict(size=10, color=COLORS["muted"]),
                bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["border"],
            ),
        ),
        hovertemplate="Delivery: %{x} min<br>Rating: %{y}★<extra></extra>",
    ))
    fig6 = plotly_theme(fig6, height=330)
    fig6.update_xaxes(title_text="Delivery time (min)")
    fig6.update_yaxes(title_text="Rating")
    fig6.update_layout(margin=dict(l=0, r=80, t=16, b=0))
    st.plotly_chart(fig6, use_container_width=True)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — AI PREDICTOR                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab4:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    pred_col, info_col = st.columns([1, 1], gap="large")

    with pred_col:
        st.markdown(f"""
        <div class="section-header">Customer reorder predictor</div>
        <div class="section-sub">Fill in order details · AI predicts if customer will come back</div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        p_city = st.selectbox("City", sorted(df_raw["city"].unique()), key="p_city")
        p_cuisine = st.selectbox("Cuisine", sorted(df_raw["cuisine"].unique()), key="p_cuisine")

        col_a, col_b = st.columns(2)
        with col_a:
            p_order_val = st.slider("Order value (₹)", 50, 1000, 300, 10, key="p_val")
            p_rating = st.slider("Rating given", 3.0, 5.0, 4.2, 0.1, key="p_rat")
            p_delivery = st.slider("Delivery time (min)", 10, 90, 30, 5, key="p_del")
        with col_b:
            p_items = st.slider("No. of items", 1, 7, 2, 1, key="p_items")
            p_time = st.selectbox("Time of day", ["Morning", "Lunch", "Evening", "Late Night"], index=2, key="p_time")
            p_payment = st.selectbox("Payment", ["UPI", "Card", "Cash", "Wallet"], key="p_pay")

        col_c, col_d = st.columns(2)
        with col_c:
            p_weekend = st.toggle("Weekend order", value=False, key="p_wknd")
        with col_d:
            p_promo = st.toggle("Promo used", value=False, key="p_promo")

        p_month = st.slider("Month", 1, 12, 6, key="p_month")

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  Run AI prediction", key="pred_btn")

    with info_col:
        st.markdown(f"""
        <div class="section-header">Feature importance</div>
        <div class="section-sub">What drives the model's decisions</div>
        """, unsafe_allow_html=True)

        fi = meta["feature_importances"]
        fi_df = pd.DataFrame(list(fi.items()), columns=["feature", "importance"])
        fi_df = fi_df.sort_values("importance", ascending=True)
        fi_df["feature"] = fi_df["feature"].str.replace("_enc", "").str.replace("_", " ").str.title()

        fig_fi = go.Figure(go.Bar(
            y=fi_df["feature"], x=fi_df["importance"],
            orientation="h",
            marker=dict(
                color=fi_df["importance"],
                colorscale=[[0, COLORS["surface2"]], [1, COLORS["accent2"]]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig_fi = plotly_theme(fig_fi, height=360)
        fig_fi.update_xaxes(title_text="")
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown(f"""
        <div style='background:{COLORS["surface"]};border:1px solid {COLORS["border"]};
                    border-radius:14px;padding:1rem 1.25rem;margin-top:0.5rem'>
          <div style='font-family:Syne;font-size:0.85rem;font-weight:700;
                      color:{COLORS["text"]};margin-bottom:0.5rem'>Model details</div>
          <div style='font-size:0.78rem;color:{COLORS["muted"]};line-height:2'>
            Algorithm: Gradient Boosting Classifier<br>
            Training size: 12,000 orders<br>
            Test accuracy: <b style='color:{COLORS["positive"]}'>{meta["accuracy"]}%</b><br>
            Features: {len(meta["features"])} variables<br>
            Estimators: 150 · Max depth: 4
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Prediction output
    if predict_btn:
        try:
            cuisine_enc = le_cuisine.transform([p_cuisine])[0]
            time_enc = le_time.transform([p_time])[0]
            payment_enc = le_payment.transform([p_payment])[0]
            city_enc = le_city.transform([p_city])[0]

            X_pred = np.array([[
                p_order_val, p_delivery, p_rating, p_items,
                int(p_weekend), int(p_promo),
                cuisine_enc, time_enc, payment_enc, city_enc, p_month
            ]])

            prob = model.predict_proba(X_pred)[0]
            pred_class = model.predict(X_pred)[0]
            reorder_prob = prob[1] * 100
            no_reorder_prob = prob[0] * 100

            if pred_class == 1:
                result_txt = "Will reorder ✓"
                result_color = COLORS["positive"]
                advice = "High retention signal. Consider a loyalty reward or personalised follow-up to lock in the next order."
            else:
                result_txt = "Won't reorder ✗"
                result_color = COLORS["negative"]
                advice = "Churn risk detected. A targeted promo or delivery time improvement for this segment could shift this outcome."

            st.markdown(f"""
            <div class="pred-card" style="border-color:{result_color}30">
              <div style="display:flex;align-items:flex-start;justify-content:space-between">
                <div>
                  <div style="font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;
                              color:{COLORS['muted']};font-family:'DM Sans',sans-serif;margin-bottom:0.4rem">
                    Prediction
                  </div>
                  <div class="pred-result" style="color:{result_color}">{result_txt}</div>
                  <div class="pred-prob">
                    Reorder probability: <b style="color:{result_color}">{reorder_prob:.1f}%</b>
                    &nbsp;·&nbsp; Churn probability: {no_reorder_prob:.1f}%
                  </div>
                </div>
                <div style="font-size:3rem;opacity:0.6">{'🔄' if pred_class==1 else '⚠️'}</div>
              </div>

              <div style="margin-top:1.25rem">
                <div style="background:{result_color}18;border-radius:8px;height:8px;overflow:hidden">
                  <div style="height:100%;width:{reorder_prob:.1f}%;
                              background:{result_color};border-radius:8px;
                              transition:width 0.5s ease"></div>
                </div>
                <div style="display:flex;justify-content:space-between;
                            font-size:0.72rem;color:{COLORS['muted']};
                            margin-top:0.35rem;font-family:'JetBrains Mono',monospace">
                  <span>0%</span><span>50%</span><span>100%</span>
                </div>
              </div>

              <div style="margin-top:1rem;padding:0.85rem;
                          background:{COLORS['surface2']};border-radius:10px;
                          font-size:0.82rem;color:{COLORS['muted']};line-height:1.6">
                💡 {advice}
              </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
