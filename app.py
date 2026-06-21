"""
app.py
======

The Hyperscale Bio-Digital Twin - executive control room.

Launch with:
    streamlit run app.py

A dark, mission-control style Streamlit application that fuses synthetic
geospatial climate stress signals with data centre telemetry, predicts
next-hour PUE degradation with an ML model, and prescribes sustainability-
aware AI workload redistribution across four global regions.

Structure
---------
1. Page config + cached bootstrap (simulate -> engineer -> train -> score)
2. Global CSS injection (glass panels, metric cards, restrained typography)
3. Header and executive abstract
4. Executive KPI row
5. Macro view: pydeck global stress map
6. Regional drill-down: dual-axis climate/plant time series
7. ML intelligence: model performance + feature importance
8. Prescriptive action: workload redistribution command
9. Footer strategic insight
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from analytics_engine import BioDigitalTwinEngine, ModelReport
from data_generator import DataCenterSimulator

# ----------------------------------------------------------------------
# 1. Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Hyperscale Bio-Digital Twin",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ACCENT = "#27d3a2"      # mint - primary accent
AMBER = "#f5b942"
RED = "#ef5b6e"
INK = "#e8edf4"        # main text
MUTED = "#8b97a8"      # secondary text
PANEL = "rgba(20, 28, 41, 0.78)"
BORDER = "rgba(120, 150, 190, 0.16)"

RISK_COLOURS: Dict[str, str] = {"Low": ACCENT, "Moderate": AMBER, "High": RED}


# ----------------------------------------------------------------------
# Cached bootstrap: heavy work runs once, UI re-runs stay instant.
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Initialising digital twin...")
def load_telemetry() -> pd.DataFrame:
    """Deterministic synthetic telemetry (30 days x 4 regions x 1h)."""
    return DataCenterSimulator().generate()


@st.cache_resource(show_spinner="Training thermal-risk model...")
def build_engine() -> Tuple[BioDigitalTwinEngine, ModelReport]:
    """Engine + trained model, cached as a resource (not re-pickled)."""
    engine = BioDigitalTwinEngine(load_telemetry())
    report = engine.train_model()
    return engine, report


@st.cache_data(show_spinner=False)
def latest_scores() -> pd.DataFrame:
    engine, _ = build_engine()
    return engine.score_latest_conditions()


@st.cache_data(show_spinner=False)
def optimisation_plan() -> Dict:
    engine, _ = build_engine()
    return engine.optimize_workload_distribution()


# ----------------------------------------------------------------------
# 2. Global styling - glass-panel control-room aesthetic
# ----------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        html, body, [class*="css"], .stApp {{
            font-family: 'Inter', sans-serif;
        }}
        .stApp {{
            background:
                radial-gradient(1100px 500px at 15% -10%, rgba(39,211,162,0.07), transparent 60%),
                radial-gradient(900px 500px at 85% -10%, rgba(66,135,245,0.08), transparent 60%),
                linear-gradient(180deg, #0a0f18 0%, #0c1220 55%, #0a0f18 100%);
            color: {INK};
        }}
        /* Hide default Streamlit chrome */
        #MainMenu, footer, header {{ visibility: hidden; }}
        .block-container {{ padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1380px; }}

        .hero-title {{
            font-size: 2.35rem; font-weight: 700; letter-spacing: -0.02em;
            background: linear-gradient(90deg, #f2f6fb 30%, {ACCENT} 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.1rem;
        }}
        .hero-sub {{
            color: {MUTED}; font-size: 1.02rem; font-weight: 400;
            letter-spacing: 0.01em; margin-bottom: 0.9rem;
        }}
        .abstract {{
            background: {PANEL}; border: 1px solid {BORDER};
            border-left: 3px solid {ACCENT};
            border-radius: 10px; padding: 0.95rem 1.2rem;
            color: #c3ccd9; font-size: 0.93rem; line-height: 1.55;
            backdrop-filter: blur(8px);
        }}
        .section-head {{
            font-size: 0.78rem; font-weight: 600; letter-spacing: 0.18em;
            text-transform: uppercase; color: {ACCENT};
            border-bottom: 1px solid {BORDER};
            padding-bottom: 0.45rem; margin: 1.7rem 0 0.9rem 0;
        }}
        .kpi-card {{
            background: {PANEL}; border: 1px solid {BORDER};
            border-radius: 12px; padding: 0.95rem 1.05rem;
            backdrop-filter: blur(8px);
            min-height: 118px;
        }}
        .kpi-label {{
            font-size: 0.68rem; font-weight: 600; letter-spacing: 0.14em;
            text-transform: uppercase; color: {MUTED}; margin-bottom: 0.35rem;
        }}
        .kpi-value {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.45rem; font-weight: 500; color: {INK};
            line-height: 1.15;
        }}
        .kpi-delta {{ font-size: 0.78rem; color: {MUTED}; margin-top: 0.3rem; }}
        .risk-chip {{
            display: inline-block; padding: 0.14rem 0.6rem; border-radius: 999px;
            font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
        }}
        .narrative {{
            background: {PANEL}; border: 1px solid {BORDER};
            border-radius: 10px; padding: 0.85rem 1.1rem;
            color: #b9c3d2; font-size: 0.9rem; line-height: 1.55;
        }}
        .command-card {{
            background: linear-gradient(135deg, rgba(39,211,162,0.10), rgba(20,28,41,0.85));
            border: 1px solid rgba(39,211,162,0.35);
            border-radius: 12px; padding: 1.15rem 1.3rem;
            font-size: 1.0rem; line-height: 1.6; color: {INK};
        }}
        .footer-insight {{
            margin-top: 2.2rem; padding: 1.0rem 1.3rem;
            border-top: 1px solid {BORDER};
            color: {MUTED}; font-size: 0.95rem; font-style: italic;
            text-align: center;
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
        .stTabs [data-baseweb="tab"] {{
            background: {PANEL}; border: 1px solid {BORDER};
            border-radius: 8px 8px 0 0; padding: 0.4rem 1.1rem;
            color: {MUTED}; font-size: 0.86rem;
        }}
        .stTabs [aria-selected="true"] {{
            color: {ACCENT} !important; border-bottom: 2px solid {ACCENT};
        }}
        div[data-testid="stExpander"] {{
            background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str = "",
             value_colour: str = INK) -> str:
    """Styled HTML metric card (replaces default st.metric)."""
    return (
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{value_colour}">{value}</div>'
        f'<div class="kpi-delta">{delta}</div></div>'
    )


def section(title: str) -> None:
    st.markdown(f'<div class="section-head">{title}</div>',
                unsafe_allow_html=True)


def risk_chip(risk: str) -> str:
    colour = RISK_COLOURS.get(risk, MUTED)
    return (f'<span class="risk-chip" style="background:{colour}22;'
            f'color:{colour};border:1px solid {colour}55">{risk}</span>')


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#aeb9c8", size=12),
    margin=dict(l=10, r=10, t=80, b=10),
    hoverlabel=dict(bgcolor="#141c29", font_color="#e8edf4"),
)


# ----------------------------------------------------------------------
# View builders
# ----------------------------------------------------------------------
def render_header() -> None:
    st.markdown('<div class="hero-title">The Hyperscale Bio-Digital Twin</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Climate Risk, Thermal Integrity, and '
        'Sustainable AI Workload Orchestration</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="abstract">
        AI data centre growth is no longer constrained by compute and power
        alone. Thermal physics, water availability, grid carbon intensity,
        and regional climate volatility now set the binding limits on where
        the next megawatt of AI training can responsibly run. This digital
        twin fuses macro climate stress signals with asset telemetry across
        four strategic regions, predicts next-hour thermal efficiency
        degradation, and prescribes workload shifts that reduce risk,
        emissions, and water draw - before the hour arrives.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(scores: pd.DataFrame, plan: Dict) -> None:
    section("Executive Signal")
    worst = scores.loc[scores["environmental_stress_index"].idxmax()]
    safest = scores.loc[scores["predicted_degradation"].idxmin()]
    avg_pue = scores["pue"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi_card(
        "Highest Stress Region", worst["region"],
        f"ESI {worst['environmental_stress_index']:.2f} - "
        f"wet-bulb {worst['wet_bulb_temp_c']:.1f}&deg;C", RED),
        unsafe_allow_html=True)
    c2.markdown(kpi_card(
        "Lowest Next-Hour PUE Risk", safest["region"],
        f"&Delta;PUE {safest['predicted_degradation']:+.4f} - "
        f"{safest['risk_class']} risk", ACCENT),
        unsafe_allow_html=True)
    c3.markdown(kpi_card(
        "Global Average PUE", f"{avg_pue:.3f}",
        "Fleet mean, current hour"),
        unsafe_allow_html=True)
    c4.markdown(kpi_card(
        "Potential CO2 Avoidance",
        f"{plan['total_co2_saved_kg'] / 1000:.1f} t",
        "Per 4-hour redistribution window", ACCENT),
        unsafe_allow_html=True)
    c5.markdown(kpi_card(
        "Potential Water Savings",
        f"{plan['total_water_saved_litres'] / 1000:.1f} kL",
        "Stressed-basin draw avoided", ACCENT),
        unsafe_allow_html=True)


def render_map(scores: pd.DataFrame) -> None:
    section("Macro View - Global Thermal Stress")

    df = scores.copy()

    def colour_for(esi: float) -> list:
        # Green -> amber -> red as stress rises; thresholds chosen on the
        # observed ESI distribution so all three bands actually appear.
        if esi < 0.40:
            return [39, 211, 162, 220]
        if esi < 0.60:
            return [245, 185, 66, 220]
        return [239, 91, 110, 230]

    df["colour"] = df["environmental_stress_index"].apply(colour_for)
    df["radius"] = 40000 + df["chiller_power_mw"] * 9000
    df["pue_s"] = df["pue"].round(3)
    df["wb_s"] = df["wet_bulb_temp_c"].round(1)
    df["ws_s"] = df["watershed_stress_index"].round(2)
    df["ci_s"] = df["grid_carbon_intensity_gco2_kwh"].round(0)

    glow = pdk.Layer(
        "ScatterplotLayer", data=df,
        get_position=["longitude", "latitude"],
        get_radius="radius * 2.2", get_fill_color="colour",
        opacity=0.12, pickable=False,
    )
    core = pdk.Layer(
        "ScatterplotLayer", data=df,
        get_position=["longitude", "latitude"],
        get_radius="radius", get_fill_color="colour",
        get_line_color=[230, 240, 250, 120], line_width_min_pixels=1,
        stroked=True, pickable=True,
    )
    tooltip = {
        "html": (
            "<b>{region}</b><br/>"
            "PUE: {pue_s}<br/>"
            "Wet-bulb: {wb_s} &deg;C<br/>"
            "Water stress: {ws_s}<br/>"
            "Grid: {ci_s} gCO2/kWh<br/>"
            "Risk: {risk_class}"
        ),
        "style": {"backgroundColor": "#141c29", "color": "#e8edf4",
                  "fontSize": "12px", "borderRadius": "8px"},
    }
    deck = pdk.Deck(
        layers=[glow, core],
        initial_view_state=pdk.ViewState(
            latitude=44.0, longitude=-35.0, zoom=2.1, pitch=28),
        map_style="dark", tooltip=tooltip,
    )
    st.pydeck_chart(deck, height=430)
    st.markdown(
        '<div class="kpi-delta" style="margin-top:0.4rem">Marker colour = '
        'Environmental Stress Index (green low / amber medium / red high). '
        'Marker size = current chiller strain.</div>',
        unsafe_allow_html=True)


def render_regional(telemetry: pd.DataFrame, scores: pd.DataFrame) -> None:
    section("Regional Drill-Down")
    regions = sorted(scores["region"].unique())
    region = st.selectbox("Region", regions, label_visibility="collapsed")

    row = scores.loc[scores["region"] == region].iloc[0]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.markdown(kpi_card("Current PUE", f"{row['pue']:.3f}"),
                unsafe_allow_html=True)
    c2.markdown(kpi_card("Current WUE", f"{row['wue']:.2f} L/kWh"),
                unsafe_allow_html=True)
    c3.markdown(kpi_card("Compute Load",
                         f"{row['it_compute_load_mw']:.0f} MW"),
                unsafe_allow_html=True)
    c4.markdown(kpi_card("Wet-Bulb", f"{row['wet_bulb_temp_c']:.1f} &deg;C"),
                unsafe_allow_html=True)
    c5.markdown(kpi_card("Predicted Next-Hour PUE",
                         f"{row['predicted_next_pue']:.3f}",
                         f"&Delta; {row['predicted_degradation']:+.4f}"),
                unsafe_allow_html=True)
    c6.markdown(kpi_card("Risk Class", row["risk_class"], "",
                         RISK_COLOURS[row["risk_class"]]),
                unsafe_allow_html=True)

    # Dual-axis time series: wet-bulb (left) vs chiller power (right),
    # last 7 days for readability, with high-stress shading.
    sub = telemetry[telemetry["region"] == region].tail(24 * 7)
    wb_high = sub["wet_bulb_temp_c"].quantile(0.90)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub["wet_bulb_temp_c"],
        name="Wet-bulb temp (degC)", line=dict(color="#4aa8ff", width=1.7),
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub["chiller_power_mw"],
        name="Chiller power (MW)", line=dict(color=ACCENT, width=1.7),
        yaxis="y2",
    ))
    # Shaded bands where wet-bulb sits in its top decile = high-risk hours.
    in_band = (sub["wet_bulb_temp_c"] >= wb_high).to_numpy()
    ts = sub["timestamp"].to_numpy()
    start = None
    for i, flag in enumerate(in_band):
        if flag and start is None:
            start = ts[i]
        elif not flag and start is not None:
            fig.add_vrect(x0=start, x1=ts[i], fillcolor=RED, opacity=0.10,
                          line_width=0)
            start = None
    if start is not None:
        fig.add_vrect(x0=start, x1=ts[-1], fillcolor=RED, opacity=0.10,
                      line_width=0)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"{region} - wet-bulb vs chiller response (7 days)",
                   font=dict(size=14, color="#dde5ef"),
                   x=0, xanchor="left", y=0.97, yanchor="top"),
        height=400,
        yaxis=dict(title="Wet-bulb (degC)", gridcolor="rgba(120,150,190,0.10)"),
        yaxis2=dict(title="Chiller (MW)", overlaying="y", side="right",
                    showgrid=False),
        xaxis=dict(gridcolor="rgba(120,150,190,0.07)"),
        # Legend anchored top-right, on its own band above the plot area,
        # so it never collides with the left-aligned title.
        legend=dict(orientation="h", x=1.0, xanchor="right",
                    y=1.04, yanchor="bottom"),
    )
    st.plotly_chart(fig, use_container_width=True)

    lag_corr = sub["wet_bulb_temp_c"].corr(sub["chiller_power_mw"].shift(-1))
    st.markdown(
        f'<div class="narrative"><b>Operational read:</b> in {region}, '
        f'chiller power tracks wet-bulb temperature with visible thermal '
        f'inertia (lead correlation {lag_corr:.2f}). Shaded bands mark '
        f'top-decile wet-bulb hours - the windows where mechanical cooling '
        f'carries the full load, PUE drifts upward, and marginal AI '
        f'workload is most expensive in both energy and water terms.</div>',
        unsafe_allow_html=True)

    with st.expander("Inspect raw telemetry (last 48 hours)"):
        st.dataframe(
            sub.tail(48).drop(columns=["latitude", "longitude"]).round(3),
            use_container_width=True, hide_index=True)


def render_ml(report: ModelReport) -> None:
    section("Machine Intelligence - Next-Hour PUE Model")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("RMSE", f"{report.rmse:.4f}", "PUE units"),
                unsafe_allow_html=True)
    c2.markdown(kpi_card("MAE", f"{report.mae:.4f}", "PUE units"),
                unsafe_allow_html=True)
    c3.markdown(kpi_card("R&sup2;", f"{report.r2:.3f}",
                         "Chronological hold-out"),
                unsafe_allow_html=True)
    c4.markdown(kpi_card("Training Regime",
                         f"{report.n_train:,} / {report.n_test:,}",
                         "Train / test hours, time-ordered split"),
                unsafe_allow_html=True)

    imp = report.feature_importances.head(10)[::-1]
    fig = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation="h",
        marker=dict(color=imp.values, colorscale=[[0, "#1d3a4f"],
                                                  [1, ACCENT]]),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="What drives next-hour thermal degradation",
                   font=dict(size=14, color="#dde5ef"),
                   x=0, xanchor="left", y=0.97, yanchor="top"),
        height=360,
        xaxis=dict(title="Feature importance",
                   gridcolor="rgba(120,150,190,0.10)"),
        yaxis=dict(tickfont=dict(family="IBM Plex Mono", size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)

    top = report.feature_importances.index[0].replace("_", " ")
    st.markdown(
        f'<div class="narrative"><b>Interpretation:</b> the model\'s '
        f'strongest signal is <b>{top}</b>, confirming that near-term '
        f'efficiency is governed by persistence and environmental load '
        f'rather than IT intensity alone. Lagged wet-bulb and chiller '
        f'features rank highly because thermal plants respond to climate '
        f'with one-to-three hours of inertia - exactly the window in which '
        f'workload orchestration can pre-empt degradation rather than '
        f'react to it.</div>',
        unsafe_allow_html=True)


def render_prescriptive(plan: Dict) -> None:
    section("Prescriptive Action - Workload Orchestration Command")
    ws, we = plan["target_window"]
    st.markdown(
        f'<div class="command-card"><b>ORCHESTRATION DIRECTIVE</b> '
        f'<span style="color:{MUTED};font-size:0.8rem">window '
        f'{ws:%d %b %H:%M} - {we:%H:%M} UTC</span><br/><br/>'
        f'{plan["headline"]}</div>',
        unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    cols = st.columns(len(plan["shifts"]))
    for col, shift in zip(cols, plan["shifts"]):
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{shift.source} &rarr; '
            f'{shift.destination}</div>'
            f'<div class="kpi-value">{shift.shift_mw:.1f} MW</div>'
            f'<div class="kpi-delta">{shift.shift_pct_of_source:.1f}% of '
            f'source load - CO2 saved {shift.co2_saved_kg:,.0f} kg - '
            f'water saved {shift.water_saved_litres:,.0f} L<br/><br/>'
            f'{shift.rationale}</div></div>',
            unsafe_allow_html=True)

    st.markdown(
        f'<br/><div class="narrative"><b>Strategic operator note:</b> '
        f'{plan["donor_region"]} is the marginal-cost outlier this hour: '
        f'every additional megawatt placed there pays a premium in chiller '
        f'energy, stressed-basin water, and grid carbon. Redistribution is '
        f'a four-hour tactical action, not a permanent migration - the twin '
        f're-scores all regions hourly, and the directive reverses '
        f'automatically as wet-bulb stress subsides.</div>',
        unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    inject_css()
    telemetry = load_telemetry()
    engine, report = build_engine()
    scores = latest_scores()
    plan = optimisation_plan()

    if scores.empty:
        st.error("No regional scores available - check data generation.")
        return

    render_header()
    render_kpi_row(scores, plan)
    render_map(scores)

    tab1, tab2, tab3 = st.tabs(
        ["Regional Drill-Down", "ML Intelligence", "Prescriptive Action"])
    with tab1:
        render_regional(telemetry, scores)
    with tab2:
        render_ml(report)
    with tab3:
        render_prescriptive(plan)

    st.markdown(
        '<div class="footer-insight">"Thermal efficiency degradation is '
        'increasingly governed by wet-bulb volatility and water stress, '
        'not just IT load intensity. The next constraint on AI is '
        'ecological, and it is already measurable."</div>',
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
