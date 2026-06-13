"""
Precision Agriculture Decision Support System
Integrated Environmental Analysis, Yield Prediction, Crop Optimization & Recommendation
"""
import io
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Precision Agriculture DSS",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes glowPulse {
        0% { box-shadow: 0 4px 15px rgba(90,125,58,0.2); transform: translateY(0); }
        50% { box-shadow: 0 8px 25px rgba(90,125,58,0.35); transform: translateY(-2px); }
        100% { box-shadow: 0 4px 15px rgba(90,125,58,0.2); transform: translateY(0); }
    }
    @keyframes borderGlow {
        0% { border-color: rgba(90,125,58,0.3); }
        50% { border-color: rgba(141,174,107,0.8); }
        100% { border-color: rgba(90,125,58,0.3); }
    }
    @keyframes textGlow {
        0% { text-shadow: 0 0 4px rgba(61,90,37,0.2); }
        50% { text-shadow: 0 0 14px rgba(107,155,72,0.5); }
        100% { text-shadow: 0 0 4px rgba(61,90,37,0.2); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(24px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-6px); }
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes breathe {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    @keyframes pulseGlow {
        0%, 100% { filter: drop-shadow(0 0 6px rgba(90,125,58,0.3)); }
        50% { filter: drop-shadow(0 0 16px rgba(90,125,58,0.6)); }
    }
    @keyframes borderPulse {
        0% { border-color: #DAECA2; }
        50% { border-color: #8DAE6B; }
        100% { border-color: #DAECA2; }
    }

    .stApp {
        background: linear-gradient(rgba(245,247,238,0.88), rgba(232,239,224,0.88)),
                    url('https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fthf.bing.com%2Fth%2Fid%2FOIP.GXPQMW5ytUBouLvCNjTwMAHaEP%3Fr%3D0%26cb%3Dthfc1falcon2%26pid%3DApi&f=1&ipt=81086a05309f87d74d982edc64e4f53950e84791e5fb97c1b4b477373c6fa3e6&ipo=images');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .main .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    h1, h2, h3 {
        color: #3D5A25;
        animation: textGlow 3s ease-in-out infinite;
    }
    p, li, span, div { color: #2D3A1F; }

    .stSidebar {
        background: linear-gradient(180deg, #2D3A1F, #3D5A25) !important;
    }
    .stSidebar .sidebar-content { color: #DAECA2; }
    .stSidebar p { color: #DAECA2 !important; }
    div[data-testid="stSidebarNav"] {
        background: linear-gradient(180deg, #2D3A1F, #3D5A25) !important;
    }
    div[data-testid="stSidebarNav"] span { color: #DAECA2 !important; }

    .metric-card {
        background: white; border-radius: 16px; padding: 1.2rem;
        animation: fadeInUp 0.6s ease-out;
        border-left: 5px solid #5A7D3A; margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(90,125,58,0.12);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 10px 30px rgba(90,125,58,0.2);
    }
    .metric-label { color: #6B7F5A; font-size: 0.85rem; margin-bottom: 0.3rem; letter-spacing: 0.5px; }
    .metric-value { color: #3D5A25; font-size: 1.5rem; font-weight: 700; }

    .page-header {
        color: #3D5A25; border-bottom: 3px solid #5A7D3A;
        padding-bottom: 0.5rem; margin-bottom: 1.5rem;
        animation: borderGlow 3s ease-in-out infinite;
    }

    div.stButton > button:first-child {
        background: linear-gradient(135deg, #5A7D3A, #6B9B48);
        color: white; border: none; border-radius: 10px;
        padding: 0.6rem 2rem; font-weight: 600; letter-spacing: 0.3px;
        box-shadow: 0 4px 14px rgba(90,125,58,0.25), 0 2px 6px rgba(0,0,0,0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: glowPulse 3s ease-in-out infinite;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #3D5A25, #5A7D3A);
        transform: translateY(-3px) scale(1.03);
        box-shadow: 0 8px 25px rgba(90,125,58,0.35), 0 4px 10px rgba(0,0,0,0.15);
        color: white;
    }
    div.stButton > button:first-child:active {
        transform: translateY(0) scale(0.98);
        box-shadow: 0 2px 8px rgba(90,125,58,0.2);
    }
    div.stButton > button:first-child p { color: white !important; }

    .stDataFrame {
        border-radius: 10px; overflow: hidden;
        box-shadow: 0 2px 12px rgba(90,125,58,0.08);
    }

    .badge { padding: 0.2rem 0.8rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .badge-green { background-color: #DAECA2; color: #3D5A25; }
    .badge-red { background-color: #FFCDD2; color: #B71C1C; }
    .badge-blue { background-color: #C8E6C9; color: #2E7D32; }

    .rec-card {
        background: linear-gradient(135deg, #3D5A25, #5A7D3A, #6B9B48);
        background-size: 200% 200%;
        animation: gradientShift 6s ease infinite, glowPulse 3s ease-in-out infinite, float 4s ease-in-out infinite;
        border-radius: 20px; padding: 1.8rem 2.5rem; margin-bottom: 1.5rem;
        text-align: center; box-shadow: 0 8px 30px rgba(61,90,37,0.3);
    }

    .step-box {
        background: white; border: 1px solid #DAECA2; border-radius: 10px;
        padding: 1rem; margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(90,125,58,0.06);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideIn 0.4s ease-out;
    }
    .step-box:hover {
        transform: translateX(6px);
        box-shadow: 0 6px 20px rgba(90,125,58,0.15);
        border-color: #8DAE6B;
    }
    .step-label { color: #3D5A25; font-weight: 600; }
    .step-value { color: #4A5F3A; font-family: monospace; font-size: 0.9rem; }

    .calc-box {
        background: rgba(255,255,255,0.92); border-left: 4px solid #5A7D3A;
        border-radius: 6px; padding: 0.8rem 1rem; margin: 0.5rem 0;
        font-family: monospace; font-size: 0.85rem;
        animation: borderGlow 4s ease-in-out infinite;
        box-shadow: 0 2px 8px rgba(90,125,58,0.06);
    }

    div.stTabs [data-baseweb="tab-list"] { gap: 8px; }
    div.stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0; padding: 10px 20px;
        font-weight: 500; color: #5A7D3A;
        transition: all 0.3s ease;
    }
    div.stTabs [data-baseweb="tab"]:hover {
        background: rgba(90,125,58,0.08);
        transform: translateY(-3px);
    }
    div.stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #5A7D3A, #6B9B48) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(90,125,58,0.25);
    }
    div.stTabs [aria-selected="true"]:hover { transform: translateY(-2px); }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(90,125,58,0.2);
        border-radius: 12px; transition: all 0.3s ease; margin-bottom: 0.5rem;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(141,174,107,0.6);
        box-shadow: 0 4px 16px rgba(90,125,58,0.12);
    }
    div.stAlert { border-radius: 10px; border-left: 4px solid #5A7D3A; }
    iframe { border-radius: 10px; }

    /* Sidebar nav buttons */
    div[data-testid="stSidebar"] div.stButton > button:first-child {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(4px);
        box-shadow: none; animation: none;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.12);
    }
    div[data-testid="stSidebar"] div.stButton > button:first-child:hover {
        background: rgba(255,255,255,0.18);
        transform: translateX(4px);
        border-color: #DAECA2;
    }
    div[data-testid="stSidebar"] div.stButton > button[type="primary"] {
        background: rgba(218,236,162,0.2) !important;
        border-color: #DAECA2 !important;
    }

    /* Inputs & selects */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    textarea {
        border-color: #8DAE6B !important;
        border-radius: 8px !important;
    }
    textarea:focus, input:focus { border-color: #5A7D3A !important; }

    /* Radio groups */
    div[role="radiogroup"] {
        background: rgba(255,255,255,0.7);
        padding: 0.5rem; border-radius: 10px;
        border: 1px solid #DAECA2;
    }

    /* Progress bar */
    div.stProgress > div > div > div {
        background: linear-gradient(90deg, #5A7D3A, #8DAE6B, #DAECA2) !important;
        border-radius: 10px !important;
    }
    div.stProgress > div {
        border-radius: 10px !important;
        background: rgba(90,125,58,0.1) !important;
    }

    /* Data editor / table */
    div[data-testid="stDataFrame"] { border-radius: 10px; }

    /* Checkboxes */
    div[role="checkbox"]:checked + div {
        background: #5A7D3A !important;
    }

    /* Info/Warning/Success/Error boxes */
    div[data-testid="stInfoBox"] { border-left-color: #5A7D3A !important; }
    div[data-testid="stSuccessBox"] { border-left-color: #5A7D3A !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def init_state():
    keys = {
        "page": "dashboard",
        "crop_db": None, "selected_crop": None, "crop_info": None,
        "elevation": None, "dem_bounds": None, "dem_crs": None, "dem_transform": None,
        "slope": None, "slope_stats": None,
        "weather_df": None, "eto_df": None, "cwr_df": None, "kc_values": None,
        "planting_points": None, "n_plants": 0, "planting_density": 0.0,
        "field_area_ha": 0.0, "field_polygon": None,
        "field_polygon_utm": None, "field_utm_zone": 36,
        "econ_results": None, "lulc_data": None,
        "report_ready": False, "report_buffer": None,
        "crop_ranking": None, "recommended_crop": None,
        "opt_yield": None, "opt_water": None,
        "opt_wue": None, "opt_iwue": None, "opt_profit": None, "opt_schedule": None,
        "opt_ready": False,
        "yield_prediction_results": None, "yield_prediction_error": None,
        "ahp_results": None, "ahp_error": None,
        "soil_results": None, "soil_error": None,
        "soil_ph": 6.5, "soil_type": "Loam",
    }
    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def metric_card(label, value):
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )

def page_header(title):
    st.markdown(f'<h1 class="page-header">{title}</h1>', unsafe_allow_html=True)

def calc_box(text):
    st.markdown(f'<div class="calc-box">{text}</div>', unsafe_allow_html=True)

PAGES = [
    ("dashboard", "Dashboard"),
    ("land_topography", "Land Area & Topography"),
    ("crop_placement", "Crop Placement Optimization"),
    ("lulc", "LULC Characterization"),
    ("meteorology", "Meteorological Analysis"),
    ("water", "Evapotranspiration & CWR"),
    ("soil_fertility", "Soil & Fertility Analysis"),
    ("yield_prediction", "Yield Prediction"),
    ("crop_suitability", "Crop Suitability Analysis"),
    ("ahp", "AHP Analysis"),
    ("water_allocation", "Water Allocation Optimization"),
    ("economics", "Economic Analysis"),
    ("irrigation", "Irrigation Scheduling"),
    ("recommendation", "Final Recommendation"),
]

with st.sidebar:
    st.markdown(
        "<h2 style='color:white; text-align:center; margin-bottom:0.5rem;'>Precision<br>Agriculture DSS</h2>"
        ""
        "<hr style='border-color:#2E7D32; margin:0.5rem 0;'>",
        unsafe_allow_html=True,
    )

    for page_key, page_label in PAGES:
        if st.sidebar.button(
            page_label,
            key=f"nav_{page_key}",
            use_container_width=True,
            type="primary" if st.session_state.page == page_key else "secondary",
        ):
            st.session_state.page = page_key

    st.sidebar.markdown("<hr style='border-color:#2E7D32; margin-top:1rem;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<p style='color:#A5D6A7; font-size:0.75rem; text-align:center;'>"
        "Developed by Prossy</p>",
        unsafe_allow_html=True,
    )

    # Page change cleanup
    if "_prev_page" not in st.session_state:
        st.session_state._prev_page = st.session_state.page
    elif st.session_state._prev_page != st.session_state.page:
        prev = st.session_state._prev_page
        cur = st.session_state.page
        st.session_state._prev_page = cur
        # Clear heavy arrays when leaving topography page
        if prev == "land_topography" and cur != "land_topography":
            st.session_state.elevation = None
            st.session_state.slope = None
            import gc; gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# LAZY MODULE LOADERS — imported only when the page needs them
# ═══════════════════════════════════════════════════════════════════════════════

def get_crop_analysis():
    from modules.crop_analysis import load_crop_database, get_crop_options, get_selected_crop
    return load_crop_database, get_crop_options, get_selected_crop

def get_terrain():
    from modules.terrain_analysis import load_dem, compute_slope, compute_slope_stats, compute_surface_area
    return load_dem, compute_slope, compute_slope_stats, compute_surface_area

def get_weather():
    from modules.weather_analysis import fetch_weather_data, compute_eto, compute_crop_water_requirement, generate_kc_values, compute_irrigation_days
    return fetch_weather_data, compute_eto, compute_crop_water_requirement, generate_kc_values, compute_irrigation_days

def get_planting():
    from modules.planting_analysis import generate_planting_grid
    return generate_planting_grid

def get_optimization():
    from modules.optimization import rank_crops, estimate_yield, compute_water_allocation, compute_wue, compute_iwue, compute_profit, generate_irrigation_schedule
    return rank_crops, estimate_yield, compute_water_allocation, compute_wue, compute_iwue, compute_profit, generate_irrigation_schedule

def get_yield_prediction():
    from modules.yield_prediction import run_yield_prediction
    return run_yield_prediction

def get_ahp():
    from modules.ahp_analysis import run_ahp
    return run_ahp

def get_soil():
    from modules.soil_fertility import run_soil_analysis, PH_SUITABILITY, SOIL_TYPES, get_crop_fertilizer_recommendations
    return run_soil_analysis, PH_SUITABILITY, SOIL_TYPES, get_crop_fertilizer_recommendations

def get_report():
    from modules.report_generator import generate_pdf_report
    return generate_pdf_report


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.page == "dashboard":
    page_header("Dashboard — Precision Agriculture Decision Support System")

    st.markdown("""
    This system integrates **three analysis modules** into a unified recommendation workflow:
    1. **Environmental & Farm Analysis** — Land, topography, weather, soil, LULC
    2. **Yield Prediction** — Hybrid GRU-LSTM vs XGBoost comparison
    3. **Crop Optimization & Recommendation** — Suitability, AHP, water allocation, economics
    """)

    checks = {
        "Crop Database Loaded": st.session_state.crop_db is not None,
        "Farm Boundary Set": st.session_state.field_polygon is not None,
        "Topography / Slope Computed": st.session_state.slope is not None,
        "Weather Data Fetched": st.session_state.weather_df is not None,
        "ET₀ & CWR Computed": st.session_state.cwr_df is not None,
        "Planting Analysis Done": st.session_state.n_plants > 0,
        "Yield Prediction Run": st.session_state.yield_prediction_results is not None,
        "AHP Analysis Run": st.session_state.ahp_results is not None,
        "Optimization Run": st.session_state.opt_ready,
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### System Status")
        for label, ready in checks.items():
            badge = "badge-green" if ready else "badge-red"
            icon = "✓" if ready else "○"
            st.markdown(f"<span class='badge {badge}'>{icon}</span> {label}", unsafe_allow_html=True)

    with col2:
        st.markdown("### Quick Actions")
        if st.button("Load All Prerequisites", use_container_width=True):
            load_crop_database, _, _ = get_crop_analysis()
            load_dem, compute_slope, _, _ = get_terrain()
            fetch_weather_data, compute_eto, compute_crop_water_requirement, generate_kc_values, _ = get_weather()

            df = load_crop_database()
            if df is not None:
                st.session_state.crop_db = df
            clip_bounds = None
            if st.session_state.field_polygon_utm:
                xs = [p[0] for p in st.session_state.field_polygon_utm]
                ys = [p[1] for p in st.session_state.field_polygon_utm]
                clip_bounds = (min(xs), min(ys), max(xs), max(ys))
            dem_result = load_dem(clip_bounds)
            if dem_result:
                elevation, bounds, crs, transform = dem_result
                st.session_state.elevation = elevation
                st.session_state.dem_bounds = bounds
                st.session_state.dem_crs = crs
                st.session_state.dem_transform = transform
                slope = compute_slope(elevation, transform)
                st.session_state.slope = slope
            wdf = fetch_weather_data()
            if wdf is not None:
                st.session_state.weather_df = wdf
                eto = compute_eto(wdf)
                st.session_state.eto_df = eto
                kc = generate_kc_values(len(eto))
                st.session_state.kc_values = kc
                cwr = compute_crop_water_requirement(eto, kc)
                st.session_state.cwr_df = cwr
            st.success("All prerequisites loaded! Navigate via sidebar.")

        if st.button("⚡ Pre-compute All for Demo", type="primary", use_container_width=True):
            _prog = st.progress(0, text="Setting default field boundary...")
            if not st.session_state.field_polygon:
                import utm as _utm
                _lonlats = [(33.70, 0.37), (33.75, 0.37), (33.75, 0.32), (33.70, 0.32)]
                _pts_utm = []
                for _lo, _la in _lonlats:
                    _e, _n, _zn, _ = _utm.from_latlon(_la, _lo)
                    _pts_utm.append((_e, _n))
                st.session_state.field_polygon = _lonlats
                st.session_state.field_polygon_utm = _pts_utm
                st.session_state.field_utm_zone = _zn
                st.session_state.field_area_ha = __import__('shapely.geometry', fromlist=['Polygon']).Polygon(_pts_utm).area / 10000.0

            _prog.progress(15, text="Loading crop database...")
            _crop, _, _ = get_crop_analysis()
            _dem, _slp, _, _ = get_terrain()
            _wth, _eto, _cwr, _kc, _ = get_weather()
            _df = _crop()
            if _df is not None:
                st.session_state.crop_db = _df

            _prog.progress(30, text="Loading DEM & computing slope...")
            _clip = None
            if st.session_state.field_polygon_utm:
                _xs = [p[0] for p in st.session_state.field_polygon_utm]
                _ys = [p[1] for p in st.session_state.field_polygon_utm]
                _clip = (min(_xs), min(_ys), max(_xs), max(_ys))
            _dem_result = _dem(_clip)
            if _dem_result:
                _elev, _, _, _xform = _dem_result
                st.session_state.elevation = _elev
                st.session_state.slope = _slp(_elev, _xform)

            _prog.progress(50, text="Fetching weather data from NASA POWER...")
            _wdf = _wth()
            if _wdf is not None:
                st.session_state.weather_df = _wdf
                _eto_df = _eto(_wdf)
                st.session_state.eto_df = _eto_df
                st.session_state.kc_values = _kc(len(_eto_df))
                st.session_state.cwr_df = _cwr(_eto_df, st.session_state.kc_values)

            _prog.progress(70, text="Running complete optimization...")
            _opt_rank, _opt_est, _opt_water, _opt_wue, _opt_iwue, _opt_profit, _opt_sched = get_optimization()
            _wdf2 = st.session_state.weather_df
            _cwr2 = st.session_state.cwr_df
            _mean_sl, _, _ = __import__('modules.terrain_analysis', fromlist=['compute_slope_stats']).compute_slope_stats(st.session_state.slope) if st.session_state.slope is not None else (0,0,0)
            _avg_t = ((_wdf2["Tmax"] + _wdf2["Tmin"]) / 2).mean() if _wdf2 is not None else 25
            _rain = _wdf2["Rainfall"].sum() if _wdf2 is not None else 1000
            _avg_kc = _cwr2["Kc"].mean() if _cwr2 is not None and "Kc" in _cwr2.columns else 1.0
            _ranking = _opt_rank(st.session_state.crop_db, _avg_t, _rain, _mean_sl, _avg_kc)
            if _ranking:
                st.session_state.crop_ranking = _ranking
                _idc = "Crop_ID" if "Crop_ID" in st.session_state.crop_db.columns else "Crop_Number"
                _best = _ranking[0]
                _row = st.session_state.crop_db[st.session_state.crop_db[_idc] == _best["crop_id"]].iloc[0]
                st.session_state.recommended_crop = _best
                _sum_cwr = _cwr2["CWR"].sum() if _cwr2 is not None else 500
                _sum_etc = _cwr2["ETc"].sum() if _cwr2 is not None else 400
                _irr_d = __import__('modules.weather_analysis', fromlist=['compute_irrigation_days']).compute_irrigation_days(_cwr2) if _cwr2 is not None else 30
                st.session_state.opt_yield = _opt_est(_row, _best["score"], _sum_cwr, _sum_etc)
                st.session_state.opt_water = _opt_water(_sum_cwr, st.session_state.field_area_ha or 1.0, _irr_d)
                st.session_state.opt_wue = _opt_wue(st.session_state.opt_yield, _sum_etc)
                st.session_state.opt_iwue = _opt_iwue(st.session_state.opt_yield, _sum_cwr)
                st.session_state.opt_profit = _opt_profit(st.session_state.opt_yield, st.session_state.field_area_ha or 1.0, _row)
                st.session_state.opt_schedule = _opt_sched(_cwr2, _irr_d, _row.get("Growth_Stages", ""), _row.get("Avg_Kc", 1.0))
                st.session_state.opt_ready = True

            _prog.progress(85, text="Running yield prediction & AHP...")
            _yp = get_yield_prediction()
            _yr, _ = _yp(_wdf2)
            if _yr:
                st.session_state.yield_prediction_results = _yr
            _ahp = get_ahp()
            _ar, _ = _ahp()
            if _ar:
                st.session_state.ahp_results = _ar

            _prog.progress(100, text="Done!"), _prog.empty()

        checks_opt = [
            st.session_state.crop_db is not None,
            st.session_state.weather_df is not None,
            st.session_state.cwr_df is not None,
            st.session_state.slope is not None,
        ]
        if all(checks_opt):
            if st.button("▶ Run Complete Optimization", type="primary", use_container_width=True):
                try:
                    _pg = st.progress(0, text="Ranking crops...")
                    rank_crops, estimate_yield, compute_water_allocation, compute_wue, compute_iwue, compute_profit, generate_irrigation_schedule = get_optimization()
                    _, _, compute_slope_stats, _ = get_terrain()
                    run_yield_prediction = get_yield_prediction()
                    run_ahp = get_ahp()
                    run_soil_analysis, _, _ = get_soil()

                    wdf = st.session_state.weather_df
                    cwr = st.session_state.cwr_df
                    mean_deg, _, _ = compute_slope_stats(st.session_state.slope)
                    avg_tmean = ((wdf["Tmax"] + wdf["Tmin"]) / 2).mean()
                    total_rain = wdf["Rainfall"].sum()
                    avg_kc_actual = cwr["Kc"].mean() if "Kc" in cwr.columns else 1.0

                    ranking = rank_crops(st.session_state.crop_db, avg_tmean, total_rain, mean_deg, avg_kc_actual)
                    st.session_state.crop_ranking = ranking

                    _pg.progress(25, text="Computing water allocation & economics...")
                    id_col = "Crop_ID" if "Crop_ID" in st.session_state.crop_db.columns else "Crop_Number"
                    if ranking:
                        best = ranking[0]
                        best_row = st.session_state.crop_db[st.session_state.crop_db[id_col] == best["crop_id"]].iloc[0]
                        st.session_state.recommended_crop = best
                        total_cwr_sum = cwr["CWR"].sum()
                        total_etc_sum = cwr["ETc"].sum()
                        _, _, _, _, cid = get_weather()
                        irr_days = cid(cwr)
                        est_yield = estimate_yield(best_row, best["score"], total_cwr_sum, total_etc_sum)
                        st.session_state.opt_yield = est_yield
                        water = compute_water_allocation(total_cwr_sum, st.session_state.field_area_ha or 1.0, irr_days)
                        st.session_state.opt_water = water
                        st.session_state.opt_wue = compute_wue(est_yield, total_etc_sum)
                        st.session_state.opt_iwue = compute_iwue(est_yield, total_cwr_sum)
                        profit = compute_profit(est_yield, st.session_state.field_area_ha or 1.0, best_row)
                        st.session_state.opt_profit = profit
                        growth_stages = best_row.get("Growth_Stages", "")
                        avg_kc = best_row.get("Avg_Kc", 1.0)
                        schedule = generate_irrigation_schedule(cwr, irr_days, growth_stages, avg_kc)
                        st.session_state.opt_schedule = schedule
                        st.session_state.opt_ready = True

                    _pg.progress(50, text="Running yield prediction...")
                    yr, yr_err = run_yield_prediction(wdf)
                    if yr:
                        st.session_state.yield_prediction_results = yr

                    _pg.progress(75, text="Computing AHP analysis...")
                    ahp_res, ahp_err = run_ahp()
                    if ahp_res:
                        st.session_state.ahp_results = ahp_res

                    _pg.progress(90, text="Running soil analysis...")
                    soil_ph = st.session_state.get("soil_ph", 6.5)
                    soil_type = st.session_state.get("soil_type", "Loam")
                    soil_res = run_soil_analysis(soil_ph, soil_type)
                    st.session_state.soil_results = soil_res

                    _pg.progress(100, text="Done!"), _pg.empty()
                    st.success("Optimization complete!")
                except Exception as e:
                    st.error(f"Optimization failed: {e}")

    if st.session_state.get("opt_ready") and st.session_state.recommended_crop:
        rec = st.session_state.recommended_crop
        st.markdown(
            f"""<div class="rec-card">
                <div style='color:#A5D6A7; font-size:0.9rem;'>Recommended Crop</div>
                <div style='color:white; font-size:2.2rem; font-weight:700;'>{rec['name']}</div>
                <div style='color:#C8E6C9; font-size:1.1rem;'>Suitability Score: {rec['score']}/100</div>
            </div>""",
            unsafe_allow_html=True,
        )

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            metric_card("Expected Yield", f"{st.session_state.opt_yield} t/ha" if st.session_state.opt_yield else "N/A")
        with col_b:
            metric_card("Total CWR", f"{st.session_state.opt_water['total_cwr_mm']} mm" if st.session_state.opt_water else "N/A")
        with col_c:
            metric_card("WUE", f"{st.session_state.opt_wue} kg/m³" if st.session_state.opt_wue else "N/A")
        with col_d:
            metric_card("Profit", f"{st.session_state.opt_profit['profit_ugx']:,.0f} UGX" if st.session_state.opt_profit else "N/A")

        if st.session_state.yield_prediction_results:
            yr = st.session_state.yield_prediction_results
            with col_a:
                metric_card("Best Model", yr.get("best_model", "N/A"))
            with col_b:
                metric_card("Best RMSE", f"{yr.get('best_rmse', 'N/A')}" if yr.get('best_rmse') else "N/A")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Land Area & Topography
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "land_topography":
    page_header("Land Area & Topography Analysis")
    st.markdown("Field boundary definition, DEM analysis, and 3D surface area calculation.")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Field Boundary")
        default_coords = "0.37, 33.70\n0.37, 33.75\n0.32, 33.75\n0.32, 33.70"
        coords_text = st.text_area("Polygon vertices (lat, lon per line):", value=default_coords, height=120)
        coord_format = st.radio("Format:", ["Lat/Lon (degrees)", "UTM (meters)"], horizontal=True)

        if st.button("Set Field Boundary", type="primary", use_container_width=True):
            try:
                points_lonlat = []
                points_utm = []
                utm_zone = 36
                for line in coords_text.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.replace(",", " ").split()
                    if len(parts) >= 2:
                        if coord_format == "UTM (meters)":
                            import utm
                            easting, northing = float(parts[0]), float(parts[1])
                            utm_zone = int(float(parts[2])) if len(parts) >= 3 else 36
                            lat, lon = utm.to_latlon(easting, northing, utm_zone, "N")
                            points_lonlat.append((lon, lat))
                            points_utm.append((easting, northing))
                        else:
                            lat, lon = float(parts[0]), float(parts[1])
                            points_lonlat.append((lon, lat))

                if len(points_lonlat) >= 3:
                    import shapely.geometry as geom
                    polygon_lonlat = geom.Polygon(points_lonlat)
                    if not polygon_lonlat.is_valid:
                        st.error("Invalid polygon.")
                    else:
                        st.session_state.field_polygon = points_lonlat
                        # Always compute correct area from UTM (convert lat/lon if needed)
                        if coord_format == "Lat/Lon (degrees)":
                            import utm
                            points_utm = []
                            for lon, lat in points_lonlat:
                                e, n, zn, _ = utm.from_latlon(lat, lon)
                                points_utm.append((e, n))
                                utm_zone = zn
                        polygon_utm = geom.Polygon(points_utm)
                        area_m2 = polygon_utm.area
                        area_ha = area_m2 / 10000.0
                        st.session_state.field_polygon_utm = points_utm
                        st.session_state.field_utm_zone = utm_zone
                        st.session_state.field_area_ha = area_ha
                        st.success(f"Boundary set! Area = {area_ha:.4f} ha ({area_m2:.1f} m²)")
                        calc_box(f"A2D = {area_m2:.2f} m² = {area_ha:.4f} ha")
                else:
                    st.error("Need at least 3 vertices.")
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("### DEM & Slope Analysis")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if st.button("Load DEM", use_container_width=True):
                load_dem, compute_slope, compute_slope_stats, compute_surface_area = get_terrain()
                clip_bounds = None
                if st.session_state.field_polygon_utm:
                    xs = [p[0] for p in st.session_state.field_polygon_utm]
                    ys = [p[1] for p in st.session_state.field_polygon_utm]
                    clip_bounds = (min(xs), min(ys), max(xs), max(ys))
                dem_result = load_dem(clip_bounds)
                if dem_result:
                    elevation, bounds, crs, transform = dem_result
                    st.session_state.elevation = elevation
                    st.session_state.dem_bounds = bounds
                    st.session_state.dem_crs = crs
                    st.session_state.dem_transform = transform
                    slope = compute_slope(elevation, transform)
                    st.session_state.slope = slope
                    mean_deg, max_deg, min_deg = compute_slope_stats(slope)
                    st.session_state.slope_stats = (mean_deg, max_deg, min_deg)
                    calc_box(f"DEM loaded: {elevation.shape[0]}×{elevation.shape[1]} cells, "
                             f"bounds={bounds}")
                    calc_box("Slope = arctan(√(dx² + dy²)), dx,dy = gradient(Elevation)")
                    st.success("DEM loaded & slope computed!")
                else:
                    st.error("Failed to load DEM.")

    with col_right:
        if st.session_state.field_polygon:
            pts = st.session_state.field_polygon
            lons, lats = zip(*pts)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(lons) + [lons[0]], y=list(lats) + [lats[0]],
                mode="lines+markers", fill="toself",
                fillcolor="rgba(46,125,50,0.2)", line=dict(color="#2E7D32", width=3),
                marker=dict(size=8, color="#1B5E20"), name="Field Boundary",
            ))
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis=dict(scaleanchor="y"), title="Field Boundary")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No field boundary set yet.")

    if st.session_state.slope is not None:
        slope = st.session_state.slope
        _, _, compute_slope_stats, compute_surface_area = get_terrain()
        mean_deg, max_deg, min_deg = compute_slope_stats(slope)

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            metric_card("Mean Slope", f"{mean_deg:.2f}°")
        with col_s2:
            metric_card("Max Slope", f"{max_deg:.2f}°")
        with col_s3:
            metric_card("Min Slope", f"{min_deg:.2f}°")

        if st.session_state.field_area_ha > 0 and mean_deg > 0:
            mean_slope_rad = np.radians(mean_deg)
            surf_area = compute_surface_area(st.session_state.field_area_ha, mean_slope_rad)
            area_diff = surf_area - st.session_state.field_area_ha

            calc_box(f"A3D = A2D / cos(θ) = {st.session_state.field_area_ha:.4f} / cos({mean_deg:.2f}°)")
            calc_box(f"cos({mean_deg:.2f}°) = {np.cos(mean_slope_rad):.6f}")
            calc_box(f"A3D = {st.session_state.field_area_ha:.4f} / {np.cos(mean_slope_rad):.6f} = {surf_area:.4f} ha")

            col_s4, col_s5, col_s6 = st.columns(3)
            with col_s4:
                metric_card("2D Area (A2D)", f"{st.session_state.field_area_ha:.4f} ha")
            with col_s5:
                metric_card("3D Surface Area (A3D)", f"{surf_area:.4f} ha")
            with col_s6:
                metric_card("Area Difference", f"{area_diff:.4f} ha")

        dem_tabs = st.tabs(["Slope Map", "Elevation (DEM)", "Slope Distribution"])
        with dem_tabs[0]:
            fig = px.imshow(slope, color_continuous_scale="Viridis", title="Slope (radians)", aspect="auto")
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with dem_tabs[1]:
            if st.session_state.elevation is not None:
                fig = px.imshow(st.session_state.elevation, color_continuous_scale="earth",
                                title="Digital Elevation Model (m)", aspect="auto")
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
        with dem_tabs[2]:
            slope_deg = np.degrees(slope[~np.isnan(slope)])
            fig = px.histogram(x=slope_deg, nbins=50, title="Slope Distribution",
                               labels={"x": "Slope (degrees)", "y": "Count"},
                               color_discrete_sequence=["#2E7D32"])
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click 'Load DEM' to see elevation and slope analysis.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Crop Placement Optimization
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "crop_placement":
    page_header("Crop Placement Optimization")
    st.markdown("Planting grid generation — determines plant positions within the field boundary.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Step 1: Select Crop")
        if st.button("Load Crop Database", use_container_width=True):
            load_crop_database, _, _ = get_crop_analysis()
            df = load_crop_database()
            if df is not None:
                st.session_state.crop_db = df
                st.success(f"Loaded {len(df)} crops!")

        if st.session_state.crop_db is not None:
            _, get_crop_options, get_selected_crop = get_crop_analysis()
            options = get_crop_options(st.session_state.crop_db)
            selected = st.selectbox("Choose a crop:", options, index=None)
            if selected:
                ci = get_selected_crop(st.session_state.crop_db, selected)
                st.session_state.crop_info = ci
                if ci:
                    st.markdown(
                        f"**{ci['name']}** — Row spacing: {ci['row_spacing']}m, Plant spacing: {ci['plant_spacing']}m")

    with col2:
        st.markdown("### Step 2: Generate Grid")
        has_crop = st.session_state.crop_info is not None
        has_field = st.session_state.field_polygon is not None

        if has_crop and has_field:
            ci = st.session_state.crop_info
            st.markdown(f"Field: {st.session_state.field_area_ha:.2f} ha")
            st.markdown(f"Crop: {ci['name']} ({ci['row_spacing']}m × {ci['plant_spacing']}m)")
            calc_box(f"Grid vectors: x = x₀ : {ci['plant_spacing']} : x₁")
            calc_box(f"y = y₀ : {ci['row_spacing']} : y₁")
            calc_box("Points inside polygon via shapely.contains()")

    # Fertilizer recommendations — shown below the columns when a crop is selected
    if st.session_state.crop_info is not None and st.session_state.crop_db is not None:
        ci = st.session_state.crop_info
        row = st.session_state.crop_db[st.session_state.crop_db["Common_Name"] == ci["name"]]
        if not row.empty:
            r = row.iloc[0]
            n_val = r.get("N_kg_ha", 0)
            p_val = r.get("P_kg_ha", 0)
            k_val = r.get("K_kg_ha", 0)
            rec_fert = r.get("Recommended_Fertilizers", "")
            if pd.notna(n_val) and n_val > 0:
                with st.expander(f"Fertilizer Recommendations for {ci['name']}", expanded=True):
                    col_f1, col_f2 = st.columns([2, 1])
                    with col_f1:
                        run_soil_analysis, _, _, get_crop_fert = get_soil()
                        fert_df = get_crop_fert(ci["name"], n_val, p_val, k_val, str(rec_fert))
                        if fert_df is not None and not fert_df.empty:
                            st.dataframe(fert_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No specific fertilizer recommendations for this crop.")
                    with col_f2:
                        st.markdown("**Crop NPK Requirements (kg/ha):**")
                        st.markdown(f"- **Nitrogen (N):** {n_val}")
                        st.markdown(f"- **Phosphorus (P):** {p_val}")
                        st.markdown(f"- **Potassium (K):** {k_val}")
                        if pd.notna(rec_fert) and rec_fert:
                            st.markdown("**Recommended blend:**")
                            st.markdown(f"`{rec_fert}`")

    if has_crop and has_field:
        if st.button("Generate Planting Grid", type="primary", use_container_width=True):
            generate_planting_grid = get_planting()
            grid_poly = st.session_state.field_polygon_utm or st.session_state.field_polygon
            ci = st.session_state.crop_info
            pts, n, density = generate_planting_grid(
                grid_poly, ci["plant_spacing"], ci["row_spacing"],
                st.session_state.field_area_ha,
            )
            st.session_state.planting_points = pts
            st.session_state.n_plants = n
            st.session_state.planting_density = density
            calc_box(f"N_plants = floor((W/sx) + 1) × floor((H/sy) + 1) = {n}")
            calc_box(f"Density = {n} / {st.session_state.field_area_ha:.2f} ha = {density:,.0f} plants/ha")
            st.success(f"Grid generated: {n:,} plants at {density:,.0f} plants/ha")
    else:
        missing = [m for m, c in [("crop", has_crop), ("field boundary", has_field)] if not c]
        st.info(f"Select a crop and set field boundary first. Missing: {', '.join(missing)}")

    if st.session_state.n_plants > 0:
        st.markdown("### Results")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            metric_card("Total Plants", f"{st.session_state.n_plants:,}")
        with col_r2:
            metric_card("Plant Density", f"{st.session_state.planting_density:,.0f} plants/ha")

        pts = st.session_state.planting_points
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode="markers",
            marker=dict(size=4, color="#2E7D32", opacity=0.6), name="Plant Locations",
        ))
        poly = st.session_state.field_polygon_utm or st.session_state.field_polygon
        xs, ys = zip(*poly)
        fig.add_trace(go.Scatter(
            x=list(xs) + [xs[0]], y=list(ys) + [ys[0]],
            mode="lines", line=dict(color="#D32F2F", width=2), name="Field Boundary",
        ))
        fig.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(scaleanchor="y"), title="Planting Grid Layout (MATLAB meshgrid equivalent)")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Intermediate Calculations"):
            st.markdown("**Planting Grid Generation (MATLAB equivalent):**")
            calc_box("[xx, yy] = meshgrid(x₀:sx:x₁, y₀:sy:y₁)")
            calc_box("points = inpolygon(xx, yy, field_x, field_y)")
            calc_box(f"sx = {st.session_state.crop_info['plant_spacing']}m "
                     f"(Plant_Spacing_m), sy = {st.session_state.crop_info['row_spacing']}m "
                     f"(Row_Spacing_m)")
            calc_box(f"N = {st.session_state.n_plants}, Density = {st.session_state.planting_density:,.0f} plants/ha")

        with st.expander("Full Crop Database"):
            if st.session_state.crop_db is not None:
                st.dataframe(st.session_state.crop_db, use_container_width=True, height=400)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LULC Characterization
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "lulc":
    page_header("LULC Characterization")
    st.markdown("Land Use / Land Cover classification from satellite imagery.")

    lulc_path = os.path.join(os.path.dirname(__file__), "data", "Uganda_Sentinel2_LULC2016.tif")
    lulc_url = "https://raw.githubusercontent.com/KevinCodeZ-code/agri-optimizer/main/data/Uganda_Sentinel2_LULC2016.tif"

    if os.path.exists(lulc_path):
        st.success(f"LULC raster found: {lulc_path}")
    else:
        with st.spinner("Downloading LULC data (~1.2MB)..."):
            try:
                import urllib.request
                os.makedirs(os.path.dirname(lulc_path), exist_ok=True)
                urllib.request.urlretrieve(lulc_url, lulc_path)
                st.success("LULC raster downloaded.")
            except Exception:
                st.warning("LULC raster file not available.")

    if st.button("Load LULC Data", type="primary", use_container_width=True):
        try:
            import rasterio
            with rasterio.open(lulc_path) as src:
                data = src.read(1)
                st.session_state.lulc_data = {
                    "array": data, "bounds": src.bounds, "crs": src.crs,
                    "transform": src.transform, "nodata": src.nodata, "profile": src.profile,
                }
            st.success("LULC data loaded!")
        except Exception as e:
            st.error(f"Error loading LULC: {e}")

    if st.session_state.lulc_data is not None:
        lulc = st.session_state.lulc_data["array"]
        nodata = st.session_state.lulc_data.get("nodata", 255)

        unique_vals, counts = np.unique(lulc[lulc != nodata], return_counts=True)

        labels_map = {
            1: "Water", 2: "Forest", 3: "Grassland", 4: "Cropland",
            5: "Built-up", 6: "Bareland", 7: "Wetland",
        }
        class_labels = [labels_map.get(v, f"Class {v}") for v in unique_vals]
        total_pixels = counts.sum()

        calc_box(f"LULC raster: {lulc.shape[0]}×{lulc.shape[1]} pixels, "
                 f"valid pixels: {total_pixels:,}, classes: {len(unique_vals)}")

        col_l1, col_l2 = st.columns(2)

        with col_l1:
            fig = px.imshow(lulc, color_continuous_scale="viridis",
                            title="LULC Classification Map (MATLAB equivalent)", aspect="auto")
            fig.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Cropped LULC (field extent)**")
            if st.session_state.field_polygon:
                try:
                    import rasterio
                    from rasterio.mask import mask
                    with rasterio.open(lulc_path) as src:
                        import shapely.geometry as geom
                        import geopandas as gpd
                        poly = geom.Polygon([(p[0], p[1]) for p in st.session_state.field_polygon])
                        poly_gdf = gpd.GeoDataFrame([{"geometry": poly}], crs="EPSG:4326")
                        poly_utm = poly_gdf.to_crs(src.crs)
                        out_image, out_transform = mask(src, [poly_utm.geometry.iloc[0]],
                                                        crop=True, nodata=0)
                        cropped = out_image[0]

                    fig_crop = px.imshow(cropped, color_continuous_scale="viridis",
                                         title="Cropped LULC (Field Extent)", aspect="auto")
                    fig_crop.update_layout(height=400)
                    st.plotly_chart(fig_crop, use_container_width=True)

                    unique_c, counts_c = np.unique(cropped[cropped > 0], return_counts=True)
                    calc_box(f"Cropped raster: {cropped.shape[0]}×{cropped.shape[1]} pixels")
                    st.markdown("**Cropped Class Distribution:**")
                    for v, c in zip(unique_c, counts_c):
                        st.markdown(f"- {labels_map.get(v, f'Class {v}')}: {c} pixels ({c / counts_c.sum() * 100:.1f}%)")
                except Exception as e:
                    st.caption(f"Cropping not available: {e}")

        with col_l2:
            fig = go.Figure(data=[
                go.Pie(labels=class_labels, values=counts, hole=0.4,
                       marker=dict(colors=px.colors.sequential.Viridis),
                       textinfo="label+percent")
            ])
            fig.update_layout(title="LULC Class Distribution", height=450,
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Class Coverage Table")
            coverage_df = pd.DataFrame({
                "Class": class_labels,
                "Pixels": counts,
                "Coverage (%)": [f"{c / total_pixels * 100:.2f}" for c in counts],
            })
            st.dataframe(coverage_df, use_container_width=True, hide_index=True)
    else:
        st.info("Load the LULC raster to visualize land cover classification.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Meteorological Analysis
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "meteorology":
    page_header("Meteorological Analysis")
    st.markdown("Daily weather variables: Temperature, Rainfall, Solar Radiation, Wind Speed, Humidity.")

    if st.button("Fetch Weather Data", type="primary", use_container_width=True):
        with st.spinner("Fetching from NASA POWER API..."):
            fetch_weather_data, _, _, _, _ = get_weather()
            df = fetch_weather_data()
            st.session_state.weather_df = df
            if df is not None:
                st.success(f"Weather data loaded: {len(df)} days")
                calc_box(f"Data source: NASA POWER API (lat=0.35, lon=33.75)")
                calc_box(f"Period: {df['Date'].min().date()} to {df['Date'].max().date()}")

    if st.session_state.weather_df is not None:
        df = st.session_state.weather_df

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            metric_card("Avg Tmax", f"{df['Tmax'].mean():.1f} °C")
        with col_m2:
            metric_card("Avg Tmin", f"{df['Tmin'].mean():.1f} °C")
        with col_m3:
            metric_card("Total Rainfall", f"{df['Rainfall'].sum():.0f} mm")
        with col_m4:
            metric_card("Avg Rs", f"{df['Rs'].mean():.1f} MJ/m²/d")
        with col_m5:
            metric_card("Avg WS", f"{df['WS'].mean():.1f} m/s")

        with st.expander("Intermediate Calculations (Weather Variables)"):
            calc_box("NASA POWER parameters: T2M_MAX, T2M_MIN, RH2M, ALLSKY_SFC_SW_DWN, WS2M, PRECTOTCORR")
            if "RH" in df.columns:
                calc_box(f"Relative Humidity: mean={df['RH'].mean():.1f}%, range=[{df['RH'].min():.1f}, {df['RH'].max():.1f}]")
            calc_box(f"Diurnal Temperature Range: DTR = Tmax - Tmin")
            calc_box(f"Mean DTR: {(df['Tmax'] - df['Tmin']).mean():.2f} °C")

        tabs = st.tabs(["Temperature", "Rainfall & Radiation", "Wind & Humidity", "Data Table", "MATLAB Weather Plot"])

        with tabs[0]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Tmax"], name="Tmax (MATLAB equivalent)",
                                     line=dict(color="#D32F2F")))
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Tmin"], name="Tmin",
                                     line=dict(color="#1976D2")))
            fig.update_layout(height=400, title="Daily Temperatures (°C) — subplot(2,1,1)",
                              margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=df["Date"], y=df["Rainfall"], name="Rainfall",
                                 marker_color="#43A047", opacity=0.6), secondary_y=False)
            if "Rs" in df.columns:
                fig.add_trace(go.Scatter(x=df["Date"], y=df["Rs"], name="Solar Radiation",
                                         line=dict(color="#FF8F00")), secondary_y=True)
            fig.update_layout(height=400, title="Rainfall & Solar Radiation — subplot(2,1,2)",
                              margin=dict(l=20, r=20, t=40, b=20))
            fig.update_yaxes(title_text="Rainfall (mm)", secondary_y=False)
            fig.update_yaxes(title_text="Rs (MJ/m²/day)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            if "WS" in df.columns:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=df["Date"], y=df["WS"], name="Wind Speed",
                                         line=dict(color="#7B1FA2")), secondary_y=False)
                if "RH" in df.columns:
                    fig.add_trace(go.Scatter(x=df["Date"], y=df["RH"], name="Relative Humidity",
                                             line=dict(color="#F57C00")), secondary_y=True)
                fig.update_layout(height=350, title="Wind Speed & Humidity",
                                  margin=dict(l=20, r=20, t=40, b=20))
                fig.update_yaxes(title_text="WS (m/s)", secondary_y=False)
                fig.update_yaxes(title_text="RH (%)", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)

        with tabs[3]:
            st.dataframe(df, use_container_width=True, height=400)

        with tabs[4]:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=("Temperature (°C)", "Rainfall (mm)", "Solar Radiation (MJ/m²/d)"))
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Tmax"], name="Tmax", line=dict(color="red")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Tmin"], name="Tmin", line=dict(color="blue")), row=1, col=1)
            fig.add_trace(go.Bar(x=df["Date"], y=df["Rainfall"], name="Rainfall",
                                 marker_color="green", opacity=0.5), row=2, col=1)
            if "Rs" in df.columns:
                fig.add_trace(go.Scatter(x=df["Date"], y=df["Rs"], name="Rs",
                                         line=dict(color="orange")), row=3, col=1)
            fig.update_layout(height=600, title="MATLAB-style subplot(t,3,1) Weather Variables",
                              margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Fetch weather data to begin meteorological analysis.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Evapotranspiration & CWR
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "water":
    page_header("Evapotranspiration & Crop Water Requirement")
    st.markdown("Reference evapotranspiration (ET₀), actual crop ET (ETc), effective rainfall, and CWR.")

    col_w1, col_w2 = st.columns([1, 1])

    with col_w1:
        st.markdown("### Compute ET₀ — Hargreaves-Samani Method")
        calc_box("ET₀ = 0.0023 × (Tmean + 17.8) × √(Tmax - Tmin) × Rs × 0.408")
        calc_box("Tmean = (Tmax + Tmin) / 2")

        if st.session_state.weather_df is None:
            st.warning("Fetch weather data first (Meteorological Analysis page).")
        elif st.button("Step 1: Compute ET₀", type="primary", use_container_width=True):
            _, compute_eto, _, _, _ = get_weather()
            eto_df = compute_eto(st.session_state.weather_df)
            st.session_state.eto_df = eto_df
            calc_box(f"ET₀ computed for {len(eto_df)} days")
            calc_box(f"Mean ET₀ = {eto_df['ET0'].mean():.3f} mm/day")
            calc_box(f"Total ET₀ = {eto_df['ET0'].sum():.1f} mm")
            st.success("ET₀ computed!")

        if st.session_state.eto_df is not None:
            if st.button("Step 2: Compute CWR (ETc − Peff)", type="primary", use_container_width=True):
                _, _, compute_crop_water_requirement, generate_kc_values, _ = get_weather()
                n = len(st.session_state.eto_df)
                kc = generate_kc_values(n)
                st.session_state.kc_values = kc
                cwr_df = compute_crop_water_requirement(st.session_state.eto_df, kc)
                st.session_state.cwr_df = cwr_df
                calc_box("ETc = ET₀ × Kc")
                calc_box("Peff = Rainfall × 0.75")
                calc_box("CWR = ETc − Peff (clipped to ≥ 0)")
                calc_box(f"Total ETc = {cwr_df['ETc'].sum():.1f} mm")
                calc_box(f"Total Peff = {cwr_df['Peff'].sum():.1f} mm")
                calc_box(f"Total CWR = {cwr_df['CWR'].sum():.1f} mm")
                st.success("CWR computed!")

    with col_w2:
        if st.session_state.cwr_df is not None:
            cwr = st.session_state.cwr_df
            _, _, _, _, compute_irrigation_days = get_weather()
            irr_days = compute_irrigation_days(cwr)
            metric_card("Avg ET₀", f"{cwr['ET0'].mean():.3f} mm/day")
            metric_card("Avg ETc", f"{cwr['ETc'].mean():.3f} mm/day")
            metric_card("Total ET₀", f"{cwr['ET0'].sum():.1f} mm")
            metric_card("Total ETc", f"{cwr['ETc'].sum():.1f} mm")
            metric_card("Total Peff", f"{cwr['Peff'].sum():.1f} mm")
            metric_card("Total CWR", f"{cwr['CWR'].sum():.1f} mm")
            metric_card("Days Requiring Irrigation", f"{irr_days}")
        elif st.session_state.eto_df is not None:
            eto = st.session_state.eto_df
            metric_card("Avg ET₀", f"{eto['ET0'].mean():.3f} mm/day")
            metric_card("Total ET₀", f"{eto['ET0'].sum():.1f} mm")

    if st.session_state.cwr_df is not None:
        cwr = st.session_state.cwr_df
        tabs = st.tabs(["ET₀ & ETc", "Water Balance (MATLAB fig)", "Kc Curve", "Intermediate Table"])

        with tabs[0]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cwr["Date"], y=cwr["ET0"], name="ET₀ (Reference)",
                                     line=dict(color="#1976D2", width=2)))
            fig.add_trace(go.Scatter(x=cwr["Date"], y=cwr["ETc"], name="ETc (Crop)",
                                     line=dict(color="#D32F2F", width=2)))
            fig.update_layout(height=400, title="Reference & Crop Evapotranspiration",
                              margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=cwr["Date"], y=cwr["Rainfall"], name="Rainfall",
                                 marker_color="#43A047", opacity=0.5))
            fig.add_trace(go.Bar(x=cwr["Date"], y=cwr["ETc"], name="ETc",
                                 marker_color="#1976D2", opacity=0.5))
            fig.add_trace(go.Bar(x=cwr["Date"], y=cwr["CWR"], name="CWR (Deficit)",
                                 marker_color="#D32F2F", opacity=0.7))
            fig.update_layout(height=400, title="Water Balance — MATLAB bar(t,3) equivalent",
                              barmode="overlay", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cwr["Date"], y=cwr["Kc"], name="Kc",
                                     line=dict(color="#2E7D32", width=3)))
            fig.update_layout(height=300, title="Crop Coefficient (Kc) Curve",
                              margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            calc_box("Kc values: Initial stage (0.3) → Mid-season (1.0) → Late season (1.0)")

        with tabs[3]:
            st.dataframe(cwr, use_container_width=True, height=400)

    elif st.session_state.eto_df is not None:
        eto = st.session_state.eto_df
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eto["Date"], y=eto["ET0"], name="ET₀",
                                 line=dict(color="#1976D2")))
        fig.update_layout(height=350, title="Reference Evapotranspiration (ET₀)",
                          margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run ET₀ and CWR calculations to see the water balance analysis.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Soil & Fertility Analysis
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "soil_fertility":
    page_header("Soil & Fertility Analysis")
    st.markdown("Soil type, pH suitability, and nutrient management recommendations.")

    if st.session_state.crop_info:
        ci = st.session_state.crop_info
        st.markdown(f"**Selected crop:** {ci['name']} — Fertilizer recommendations will be tailored to this crop.")

    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        st.markdown("### Soil Parameters")
        ph_val = st.slider("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        soil_type = st.selectbox(
            "Soil Type",
            ["Loam", "Sandy Loam", "Clay Loam", "Silt Loam", "Sandy Clay"],
            index=0,
        )

        if st.button("Run Soil Analysis", type="primary", use_container_width=True):
            run_soil_analysis, _, _, _ = get_soil()
            st.session_state.soil_ph = ph_val
            st.session_state.soil_type = soil_type
            ci = st.session_state.crop_info
            if ci and st.session_state.crop_db is not None:
                row = st.session_state.crop_db[
                    st.session_state.crop_db["Common_Name"] == ci["name"]
                ]
                if not row.empty:
                    r = row.iloc[0]
                    result = run_soil_analysis(
                        ph_val, soil_type,
                        crop_name=ci["name"],
                        n_kg_ha=r.get("N_kg_ha"),
                        p_kg_ha=r.get("P_kg_ha"),
                        k_kg_ha=r.get("K_kg_ha"),
                        recommended_fertilizers=r.get("Recommended_Fertilizers"),
                    )
                else:
                    result = run_soil_analysis(ph_val, soil_type)
            else:
                result = run_soil_analysis(ph_val, soil_type)
            st.session_state.soil_results = result
            st.success("Soil analysis complete!")

    if st.session_state.soil_results is not None:
        sr = st.session_state.soil_results

        with col_s2:
            st.markdown("### Soil Classification")
            metric_card("Soil pH", f"{sr['ph_value']}")
            metric_card("pH Suitability", sr['ph_suitability'])
            metric_card("Soil Type", sr['soil_type'])
            metric_card("Soil Suitability Score", f"{sr['soil_score']}/100")

        col_s3, col_s4 = st.columns(2)

        with col_s3:
            with st.expander("pH Suitability Table (MATLAB equivalent)"):
                st.markdown("**pH Classification:**")
                _, PH_SUITABILITY, _, _ = get_soil()
                st.dataframe(pd.DataFrame(
                    [(f"{lo}-{hi}", desc) for (lo, hi), desc in PH_SUITABILITY.items()],
                    columns=["pH Range", "Classification"]
                ), use_container_width=True, hide_index=True)

            with st.expander("Soil Type Details"):
                if sr.get("soil_info"):
                    info = sr["soil_info"]
                    st.markdown(f"- **pH Range**: {info.get('ph_range', 'N/A')}")
                    st.markdown(f"- **Organic Matter**: {info.get('organic_matter', 'N/A')}")
                    st.markdown(f"- **Drainage**: {info.get('drainage', 'N/A')}")
                    st.markdown(f"- **Suitable Crops**: {', '.join(info.get('suitable_crops', ['N/A']))}")

        with col_s4:
            with st.expander("Nutrient Deficiency & Recommendations"):
                ndf = sr.get("nutrient_recommendations")
                if ndf is not None and not ndf.empty:
                    st.dataframe(ndf, use_container_width=True, hide_index=True)

            with st.expander("Soil Suitability Calculation"):
                calc_box(f"ph = {sr['ph_value']}")
                calc_box("Score = 100 if 5.5 ≤ pH ≤ 7.5")
                calc_box("Score = 60 if 5.0 ≤ pH < 5.5 or 7.5 < pH ≤ 8.0")
                calc_box("Score = 30 if 4.5 ≤ pH < 5.0 or 8.0 < pH ≤ 8.5")
                calc_box("Score = 10 otherwise")
                calc_box(f"Final score: {sr['soil_score']}/100")

        fert_df = sr.get("crop_fertilizer_recommendations")
        if fert_df is not None and not fert_df.empty:
            st.markdown("### Crop-Specific Fertilizer Recommendations")
            crop_name = sr.get("crop_name", "Selected crop")
            st.markdown(f"Recommended fertilizers for **{crop_name}** based on NPK requirements:")
            col_f1, col_f2 = st.columns([2, 1])
            with col_f1:
                st.dataframe(fert_df, use_container_width=True, hide_index=True)
            with col_f2:
                total_n = fert_df["N_kg_ha"].sum()
                total_p = fert_df["P_kg_ha"].sum()
                total_k = fert_df["K_kg_ha"].sum()
                st.markdown("**Total NPK applied (kg/ha):**")
                st.markdown(f"- **N:** {total_n:.1f}")
                st.markdown(f"- **P:** {total_p:.1f}")
                st.markdown(f"- **K:** {total_k:.1f}")
                st.markdown("")
                st.markdown("**Fertilizer blend derived from:**")
                for _, row in fert_df.iterrows():
                    st.markdown(f"- {row['Fertilizer']} @ {row['Rate_kg_ha']} kg/ha")

        st.markdown("### Available Soil Types")
        _, _, SOIL_TYPES, _ = get_soil()
        soil_df = pd.DataFrame([
            {"Soil Type": k, "Organic Matter": v["organic_matter"],
             "Drainage": v["drainage"], "pH Range": f"{v['ph_range'][0]}-{v['ph_range'][1]}"}
            for k, v in SOIL_TYPES.items()
        ])
        st.dataframe(soil_df, use_container_width=True, hide_index=True)

        fig = go.Figure(data=go.Scatterpolar(
            r=[sr['soil_score'], sr['soil_score'] if sr['soil_score'] > 50 else 0, 100 - abs(7 - sr['ph_value']) * 10],
            theta=["Soil pH", "Organic Matter", "Drainage"],
            fill="toself", marker_color="#2E7D32",
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                          title="Soil Suitability Profile", height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Configure soil parameters and run the analysis.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Yield Prediction (GRU-LSTM vs XGBoost)
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "yield_prediction":
    page_header("Yield Prediction — Hybrid GRU-LSTM vs XGBoost")
    st.markdown("""
    **Module 2:** Predictive modeling of crop yield from climatic variables.
    - **Step 1:** Feature Engineering (DTR, Sin_Month, Cos_Month)
    - **Step 2:** Chronological 80/10/10 Partition
    - **Step 3:** Min-Max Normalization
    - **Step 4a:** Hybrid GRU-LSTM Network
    - **Step 4b:** XGBoost Regressor
    - **Step 5:** RMSE/MAE Comparison
    """)

    if st.session_state.weather_df is None:
        st.warning("Weather data is required. Fetch it on the Meteorological Analysis page first.")
    else:
        if st.button("Run Yield Prediction Pipeline", type="primary", use_container_width=True):
            with st.spinner("Running yield prediction (GRU-LSTM & XGBoost)..."):
                run_yield_prediction = get_yield_prediction()
                result, error = run_yield_prediction(st.session_state.weather_df)
                if error:
                    st.session_state.yield_prediction_error = error
                    st.error(error)
                else:
                    st.session_state.yield_prediction_results = result
                    st.success("Yield prediction complete!")

    if st.session_state.yield_prediction_results:
        yr = st.session_state.yield_prediction_results

        with st.expander("Step 1: Feature Engineering", expanded=True):
            fe = yr["feature_engineering"]
            st.markdown("**Engineered Features:**")
            calc_box("DTR = Tmax - Tmin (Diurnal Temperature Range)")
            calc_box("Sin_Month = sin(2π × Month / 12)")
            calc_box("Cos_Month = cos(2π × Month / 12)")

            if not fe["engineered_data"].empty:
                eng_cols = fe["feature_cols"] + ["Yield"]
                eng_cols = [c for c in eng_cols if c in fe["engineered_data"].columns]
                st.dataframe(fe["engineered_data"][eng_cols].head(10), use_container_width=True, hide_index=True)

            eng_df = fe["engineered_data"]
            corr_cols = [c for c in fe["feature_cols"] + ["Yield"] if c in eng_df.columns]
            if len(corr_cols) >= 2:
                corr = eng_df[corr_cols].corr()
                fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                                title="Feature Correlation Matrix", aspect="auto")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("Step 2: Chronological Partitioning (80/10/10)", expanded=True):
            p = yr["partitioning"]
            st.markdown(f"**Split:** Train={p['sizes']['train']}, Val={p['sizes']['val']}, Test={p['sizes']['test']}")
            calc_box(f"Train: 80% = {p['sizes']['train']} samples")
            calc_box(f"Validation: 10% = {p['sizes']['val']} samples")
            calc_box(f"Test: 10% = {p['sizes']['test']} samples")

            fig = go.Figure()
            for name, color, subset in [("Train", "#2E7D32", p["train"]),
                                         ("Validation", "#FF8F00", p["val"]),
                                         ("Test", "#D32F2F", p["test"])]:
                if "Date" in subset.columns and "Yield" in subset.columns:
                    fig.add_trace(go.Scatter(x=subset["Date"], y=subset["Yield"],
                                             mode="markers", name=name,
                                             marker=dict(color=color, size=3, opacity=0.6)))
            fig.update_layout(height=350, title="Yield Chronological Partition",
                              margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Step 3: Min-Max Normalization"):
            norm = yr["normalization"]
            st.markdown("**Normalization Parameters (from training set):**")
            norm_df = pd.DataFrame({
                "Feature": list(norm["feature_mins"].keys()),
                "Min": list(norm["feature_mins"].values()),
                "Max": list(norm["feature_maxs"].values()),
            })
            st.dataframe(norm_df, use_container_width=True, hide_index=True)
            calc_box("X_norm = (X - X_min) / (X_max - X_min)")

        with st.expander("Step 4a: Hybrid GRU-LSTM Network", expanded=True):
            gru = yr.get("gru_lstm")
            if gru:
                st.markdown("**Architecture:**")
                calc_box("Input → GRU(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1)")
                calc_box(f"Optimizer: Adam, Loss: MSE, Epochs: 50, Batch: 16")

                if "history" in gru:
                    h = gru["history"]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=h["epochs"], y=h["train_loss"],
                                             name="Train Loss", line=dict(color="#2E7D32")))
                    fig.add_trace(go.Scatter(x=h["epochs"], y=h["val_loss"],
                                             name="Val Loss", line=dict(color="#D32F2F")))
                    fig.update_layout(height=350, title="GRU-LSTM Training History",
                                      xaxis_title="Epoch", yaxis_title="MSE Loss")
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Predictions:**")
                preds = gru["predictions"]
                fig = go.Figure()
                for split_name, split_data in [("Train", preds["train"]),
                                                ("Validation", preds["val"]),
                                                ("Test", preds["test"])]:
                    if split_data["dates"]:
                        fig.add_trace(go.Scatter(x=split_data["dates"], y=split_data["actual"],
                                                 mode="markers", name=f"{split_name} Actual",
                                                 marker=dict(size=3, opacity=0.5)))
                        fig.add_trace(go.Scatter(x=split_data["dates"], y=split_data["predicted"],
                                                 name=f"{split_name} Predicted",
                                                 line=dict(width=2)))
                fig.update_layout(height=400, title="GRU-LSTM: Actual vs Predicted Yield",
                                  margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

                if preds["test"]["actual"] and preds["test"]["predicted"]:
                    fig = go.Figure()
                    actual = preds["test"]["actual"]
                    predicted = preds["test"]["predicted"]
                    fig.add_trace(go.Scatter(x=actual, y=predicted, mode="markers",
                                             marker=dict(color="#2E7D32", size=8, opacity=0.7),
                                             name="Test Samples"))
                    min_val = min(min(actual), min(predicted))
                    max_val = max(max(actual), max(predicted))
                    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                             mode="lines", name="Perfect Fit",
                                             line=dict(color="red", dash="dash")))
                    fig.update_layout(height=350, title="GRU-LSTM: Test Set — Actual vs Predicted",
                                      xaxis_title="Actual Yield", yaxis_title="Predicted Yield")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("GRU-LSTM not available. TensorFlow/Keras may not be installed.")

        with st.expander("Step 4b: XGBoost Regressor", expanded=True):
            xgb = yr.get("xgboost")
            if xgb:
                st.markdown("**Hyperparameters:**")
                calc_box("n_estimators=200, max_depth=6, learning_rate=0.05")
                calc_box("subsample=0.8, colsample_bytree=0.8")

                if "feature_importance" in xgb and xgb["feature_importance"]:
                    st.markdown("**Feature Importance:**")
                    cols = yr["feature_engineering"]["feature_cols"]
                    fi_df = pd.DataFrame({
                        "Feature": cols,
                        "Importance": xgb["feature_importance"],
                    }).sort_values("Importance", ascending=False)

                    fig = go.Figure(go.Bar(
                        x=fi_df["Importance"], y=fi_df["Feature"],
                        orientation="h", marker_color="#2E7D32",
                        text=[f"{v:.3f}" for v in fi_df["Importance"]],
                        textposition="outside",
                    ))
                    fig.update_layout(height=300, title="XGBoost Feature Importance",
                                      margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Predictions:**")
                preds = xgb["predictions"]
                fig = go.Figure()
                for split_name, split_data in [("Train", preds["train"]),
                                                ("Validation", preds["val"]),
                                                ("Test", preds["test"])]:
                    if split_data["dates"]:
                        fig.add_trace(go.Scatter(x=split_data["dates"], y=split_data["actual"],
                                                 mode="markers", name=f"{split_name} Actual",
                                                 marker=dict(size=3, opacity=0.5)))
                        fig.add_trace(go.Scatter(x=split_data["dates"], y=split_data["predicted"],
                                                 name=f"{split_name} Predicted",
                                                 line=dict(width=2)))
                fig.update_layout(height=400, title="XGBoost: Actual vs Predicted Yield",
                                  margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

                if preds["test"]["actual"] and preds["test"]["predicted"]:
                    fig = go.Figure()
                    actual = preds["test"]["actual"]
                    predicted = preds["test"]["predicted"]
                    fig.add_trace(go.Scatter(x=actual, y=predicted, mode="markers",
                                             marker=dict(color="#FF8F00", size=8, opacity=0.7),
                                             name="Test Samples"))
                    min_val = min(min(actual), min(predicted))
                    max_val = max(max(actual), max(predicted))
                    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                             mode="lines", name="Perfect Fit",
                                             line=dict(color="red", dash="dash")))
                    fig.update_layout(height=350, title="XGBoost: Test Set — Actual vs Predicted",
                                      xaxis_title="Actual Yield", yaxis_title="Predicted Yield")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("XGBoost not available. Install xgboost package.")

        st.markdown("### Step 5: Model Comparison — RMSE & MAE")
        gru_metrics = yr["gru_lstm"]["metrics"] if yr.get("gru_lstm") else {}
        xgb_metrics = yr["xgboost"]["metrics"] if yr.get("xgboost") else {}

        if gru_metrics or xgb_metrics:
            comparison = []
            for model_name, metrics in [("GRU-LSTM", gru_metrics), ("XGBoost", xgb_metrics)]:
                if metrics:
                    comparison.append({
                        "Model": model_name,
                        "RMSE (Train)": metrics.get("rmse_train", "N/A"),
                        "RMSE (Val)": metrics.get("rmse_val", "N/A"),
                        "RMSE (Test)": metrics.get("rmse_test", "N/A"),
                        "MAE (Train)": metrics.get("mae_train", "N/A"),
                        "MAE (Val)": metrics.get("mae_val", "N/A"),
                        "MAE (Test)": metrics.get("mae_test", "N/A"),
                    })
            if comparison:
                comp_df = pd.DataFrame(comparison)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

                fig = go.Figure()
                for model_name, metrics in [("GRU-LSTM", gru_metrics), ("XGBoost", xgb_metrics)]:
                    if metrics:
                        fig.add_trace(go.Bar(
                            name=model_name,
                            x=["RMSE Train", "RMSE Val", "RMSE Test", "MAE Train", "MAE Val", "MAE Test"],
                            y=[metrics.get("rmse_train", 0), metrics.get("rmse_val", 0),
                               metrics.get("rmse_test", 0), metrics.get("mae_train", 0),
                               metrics.get("mae_val", 0), metrics.get("mae_test", 0)],
                        ))
                fig.update_layout(title="GRU-LSTM vs XGBoost: RMSE/MAE Comparison (MATLAB bar equivalent)",
                                  barmode="group", height=400, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

                st.success(f"**Best Model:** {yr.get('best_model', 'N/A')} (Test RMSE: {yr.get('best_rmse', 'N/A')})")
    else:
        st.info("Run the yield prediction pipeline to see results.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Crop Suitability Analysis
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "crop_suitability":
    page_header("Crop Suitability Analysis")
    st.markdown("Environmental suitability scoring for all crops in the database.")

    missing = [m for m, c in [
        ("Crop Database", st.session_state.crop_db is not None),
        ("Weather Data", st.session_state.weather_df is not None),
        ("CWR Data", st.session_state.cwr_df is not None),
        ("Slope Data", st.session_state.slope is not None),
    ] if not c]

    if missing:
        st.warning(f"Missing prerequisites: {', '.join(missing)}")
    else:
        if st.button("Run Suitability Analysis", type="primary", use_container_width=True):
            with st.spinner("Computing crop suitability scores..."):
                rank_crops, _, _, _, _, _, _ = get_optimization()
                _, _, compute_slope_stats, _ = get_terrain()

                wdf = st.session_state.weather_df
                cwr = st.session_state.cwr_df
                mean_deg, _, _ = compute_slope_stats(st.session_state.slope)
                avg_tmean = ((wdf["Tmax"] + wdf["Tmin"]) / 2).mean()
                total_rain = wdf["Rainfall"].sum()
                avg_kc_actual = cwr["Kc"].mean() if "Kc" in cwr.columns else 1.0

                ranking = rank_crops(st.session_state.crop_db, avg_tmean, total_rain, mean_deg, avg_kc_actual)
                st.session_state.crop_ranking = ranking

                calc_box(f"Mean Temperature: {avg_tmean:.2f} °C")
                calc_box(f"Total Rainfall: {total_rain:.1f} mm")
                calc_box(f"Mean Slope: {mean_deg:.2f}°")
                calc_box(f"Avg Kc: {avg_kc_actual:.3f}")
                calc_box(f"Weights: Temp={0.35}, Rain={0.35}, Slope={0.15}, Kc={0.15}")
                st.success("Suitability analysis complete!")

        if st.session_state.crop_ranking:
            ranking = st.session_state.crop_ranking

            col_c1, col_c2 = st.columns([1, 1])

            with col_c1:
                st.markdown("### Crop Ranking")
                rank_df = pd.DataFrame(ranking)
                rank_df = rank_df.rename(columns={
                    "crop_id": "ID", "name": "Crop", "score": "Score",
                    "temperature": "Temp", "rainfall": "Rain",
                    "slope": "Slope", "kc_match": "Kc",
                })
                rank_df = rank_df[["ID", "Crop", "Score", "Temp", "Rain", "Slope", "Kc"]]
                rank_df.index = range(1, len(rank_df) + 1)
                st.dataframe(rank_df, use_container_width=True, height=500)

            with col_c2:
                top_n = min(15, len(ranking))
                top = ranking[:top_n]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[c["score"] for c in top],
                    y=[c["name"] for c in top],
                    orientation="h", marker=dict(color="#2E7D32"),
                    text=[f"{c['score']}" for c in top],
                    textposition="outside",
                ))
                fig.update_layout(height=500, title="Top Crops by Suitability Score (MATLAB barh equivalent)",
                                  xaxis_title="Suitability Score (0-100)",
                                  margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Top 5 — Factor Breakdown")
            top5 = ranking[:5]
            fig = go.Figure()
            for crop in top5:
                fig.add_trace(go.Scatterpolar(
                    r=[crop["temperature"], crop["rainfall"], crop["slope"], crop["kc_match"], crop["temperature"]],
                    theta=["Temperature", "Rainfall", "Slope", "Kc Match", "Temperature"],
                    fill="toself", name=crop["name"],
                ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                              title="Suitability Factor Comparison (Radar)", height=400)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Suitability Calculation Details"):
                calc_box("Temperature Score: Triangular membership (Tmin, Topt, Tmax)")
                calc_box("Rainfall Score: Piecewise function (min, opt)")
                calc_box("Slope Score: Linear decreasing with slope")
                calc_box("Kc Match: ratio = min(Kc_crop, Kc_env) / max(Kc_crop, Kc_env)")
                calc_box("Overall = 0.35×Temp + 0.35×Rain + 0.15×Slope + 0.15×Kc")
        else:
            st.info("Run the suitability analysis to see crop rankings.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AHP Analysis
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "ahp":
    page_header("AHP Analysis — Analytic Hierarchy Process")
    st.markdown("Multi-criteria decision making for crop selection using pairwise comparisons.")

    if st.button("Run AHP Analysis", type="primary", use_container_width=True):
        run_ahp = get_ahp()
        result, error = run_ahp()
        if error:
            st.session_state.ahp_error = error
            st.error(error)
        else:
            st.session_state.ahp_results = result
            st.success("AHP analysis complete!")

    if st.session_state.ahp_results:
        ar = st.session_state.ahp_results

        col_a1, col_a2 = st.columns([1, 1])

        with col_a1:
            st.markdown("### Pairwise Comparison Matrix")
            criteria = ar["criteria"]
            matrix = ar["pairwise_matrix"]
            n = len(criteria)

            matrix_df = pd.DataFrame(
                [[round(matrix[i, j], 3) for j in range(n)] for i in range(n)],
                index=criteria, columns=criteria,
            )
            st.dataframe(matrix_df, use_container_width=True)

            calc_box("Saaty Scale: 1=Equal, 3=Moderate, 5=Strong, 7=Very Strong, 9=Extreme")
            calc_box("Reciprocal: a_ji = 1 / a_ij")

        with col_a2:
            st.markdown("### Weights & Ranking")
            ranking = ar["ranking"]
            st.dataframe(ranking, use_container_width=True)

            fig = go.Figure(go.Bar(
                x=ranking["Weight"], y=ranking["Criterion"],
                orientation="h", marker_color="#2E7D32",
                text=ranking["Percentage"], textposition="outside",
            ))
            fig.update_layout(height=350, title="AHP Criteria Weights",
                              margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Consistency Analysis")
        cons = ar["consistency"]
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            metric_card("λ_max", cons["lambda_max"])
        with col_c2:
            metric_card("CI", cons["ci"])
        with col_c3:
            metric_card("RI", cons["ri"])
        with col_c4:
            metric_card("CR", cons["cr"])

        cr_val = cons["cr"]
        if cons.get("consistent"):
            st.success(f"Consistency Ratio CR = {cr_val} < 0.1 → Matrix is consistent ✓")
        else:
            st.warning(f"Consistency Ratio CR = {cr_val} ≥ 0.1 → Matrix may need revision")

        calc_box("CI = (λ_max - n) / (n - 1)")
        calc_box(f"CR = CI / RI = {cons['ci']} / {cons['ri']} = {cr_val}")
        calc_box("CR < 0.1 → Acceptable consistency")

        st.markdown("### Combined AHP-Suitability Score")
        if st.session_state.crop_ranking:
            ahp_weights = ar["weights"]
            weights_dict = dict(zip(ar["criteria"], ahp_weights))

            combined = []
            for crop in st.session_state.crop_ranking[:10]:
                combined_score = (
                    weights_dict.get("Yield Potential", 0) * crop["temperature"] +
                    weights_dict.get("Water Requirement", 0) * (100 - crop["rainfall"]) +
                    weights_dict.get("Market Price", 0) * crop["kc_match"] +
                    weights_dict.get("Rainfall Suitability", 0) * crop["rainfall"] +
                    weights_dict.get("Temperature Suitability", 0) * crop["temperature"] +
                    weights_dict.get("Soil pH Suitability", 0) * 80
                )
                combined.append({
                    "Crop": crop["name"],
                    "Suitability Score": crop["score"],
                    "AHP-Weighted Score": round(combined_score / 100, 1),
                })

            comb_df = pd.DataFrame(combined)
            st.dataframe(comb_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run AHP analysis to compute weights and consistency.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Water Allocation Optimization
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "water_allocation":
    page_header("Water Allocation Optimization")
    st.markdown("Water distribution planning based on CWR, field area, and irrigation requirements.")

    if st.session_state.cwr_df is None:
        st.warning("CWR data is required. Compute it on the ET₀ & CWR page first.")
    else:
        cwr = st.session_state.cwr_df
        total_cwr = cwr["CWR"].sum()
        _, _, _, _, compute_irrigation_days = get_weather()
        irr_days = compute_irrigation_days(cwr)

        calc_box(f"Total CWR = {total_cwr:.1f} mm")
        calc_box(f"Irrigation days = days where CWR > 0 = {irr_days}")

        if st.button("Optimize Water Allocation", type="primary", use_container_width=True):
            area = st.session_state.field_area_ha if st.session_state.field_area_ha > 0 else 1.0
            _, _, compute_water_allocation, _, _, _, _ = get_optimization()
            water = compute_water_allocation(total_cwr, area, irr_days)
            st.session_state.opt_water = water
            calc_box(f"Total Volume = (CWR / 1000) × Area × 10000")
            calc_box(f"= ({total_cwr:.1f} / 1000) × {area:.2f} × 10000 = {water['total_volume_m3']:,.0f} m³")
            calc_box(f"Per Event = {water['total_volume_m3']:,.0f} / {irr_days} = {water['per_event_m3']:,.0f} m³")
            st.success("Water allocation optimized!")

        if st.session_state.opt_water:
            wt = st.session_state.opt_water
            col_w1, col_w2, col_w3, col_w4 = st.columns(4)
            with col_w1:
                metric_card("Total CWR", f"{wt['total_cwr_mm']} mm")
            with col_w2:
                metric_card("Field Area", f"{st.session_state.field_area_ha:.2f} ha" if st.session_state.field_area_ha else "N/A")
            with col_w3:
                metric_card("Total Volume", f"{wt['total_volume_m3']:,.0f} m³")
            with col_w4:
                metric_card("Per Irrigation Event", f"{wt['per_event_m3']:,.0f} m³")

            if "Peff" in cwr.columns:
                total_peff = cwr["Peff"].sum()
                fig = go.Figure(data=[go.Pie(
                    labels=["Effective Rainfall (Peff)", "Irrigation Required (CWR)"],
                    values=[total_peff, total_cwr],
                    hole=0.4, marker_colors=["#43A047", "#1976D2"],
                    textinfo="label+value+percent",
                )])
                fig.update_layout(title="Water Source Breakdown (Peff + CWR = ETc)", height=400)
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Water Balance — Intermediate Calculations"):
                water_df = pd.DataFrame({
                    "Metric": ["Total CWR (mm)", "Field Area (ha)", "Total Volume (m³)",
                               "Irrigation Events", "Per Event Volume (m³)"],
                    "Value": [f"{wt['total_cwr_mm']}", f"{st.session_state.field_area_ha:.2f}",
                              f"{wt['total_volume_m3']:,.0f}", f"{irr_days}",
                              f"{wt['per_event_m3']:,.0f}"],
                })
                st.dataframe(water_df, use_container_width=True, hide_index=True)

        if st.session_state.opt_yield and st.session_state.opt_water:
            col_w5, col_w6 = st.columns(2)
            with col_w5:
                _, _, _, _, _, compute_wue, _ = get_optimization()
                wue_val = st.session_state.opt_wue or compute_wue(st.session_state.opt_yield,
                                                                  st.session_state.cwr_df["ETc"].sum())
                metric_card("WUE", f"{wue_val} kg/m³")
                calc_box(f"WUE = Yield(kg/ha) / ETc(m³/ha)")
                calc_box(f"= {st.session_state.opt_yield * 1000:.0f} / {cwr['ETc'].sum() * 10:.0f} = {wue_val} kg/m³")
            with col_w6:
                _, _, _, _, _, _, compute_iwue = get_optimization()
                iwue_val = st.session_state.opt_iwue or compute_iwue(st.session_state.opt_yield, total_cwr)
                metric_card("IWUE", f"{iwue_val} kg/m³")
                calc_box(f"IWUE = Yield(kg/ha) / Irrigation(mm×10)")
                calc_box(f"= {st.session_state.opt_yield * 1000:.0f} / {total_cwr * 10:.0f} = {iwue_val} kg/m³")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Economic Analysis
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "economics":
    page_header("Economic Analysis")
    st.markdown("Revenue, costs, and profit projections in UGX.")

    if st.session_state.opt_profit:
        pf = st.session_state.opt_profit
        metric_card("Recommended Crop Yield", f"{pf.get('expected_yield_t_ha', 'N/A')} t/ha")

    col_e1, col_e2 = st.columns([1, 1])

    with col_e1:
        st.markdown("### Input Parameters")
        area_ha_val = st.number_input(
            "Farm Area (ha)",
            value=float(st.session_state.field_area_ha) if st.session_state.field_area_ha > 0 else 1.0,
            min_value=0.1, step=0.5, format="%.2f",
        )
        yield_val = st.number_input("Expected Yield (t/ha)", value=7.8, min_value=0.1, step=0.5)
        price_val = st.number_input("Price (UGX/kg)", value=450, min_value=1, step=50)
        cost_val = st.number_input("Production Cost (UGX/ha)", value=1850000, min_value=0, step=100000)

    with col_e2:
        st.markdown("### Calculation")
        calc_box(f"Total Yield = {yield_val} t/ha × {area_ha_val} ha = {yield_val * area_ha_val:.1f} tonnes")
        calc_box(f"Total Yield (kg) = {yield_val * area_ha_val:.1f} × 1000 = {yield_val * area_ha_val * 1000:.0f} kg")
        calc_box(f"Revenue = {yield_val * area_ha_val * 1000:.0f} kg × {price_val} UGX/kg")

        total_yield_kg = yield_val * area_ha_val * 1000
        revenue = total_yield_kg * price_val
        total_cost = cost_val * area_ha_val
        profit = revenue - total_cost
        margin = (profit / revenue * 100) if revenue > 0 else 0

        calc_box(f"Revenue = {revenue:,.0f} UGX")
        calc_box(f"Cost = {cost_val:,} UGX/ha × {area_ha_val} ha = {total_cost:,.0f} UGX")
        calc_box(f"Profit = Revenue - Cost = {profit:,.0f} UGX")
        calc_box(f"Margin = {profit:,.0f} / {revenue:,.0f} = {margin:.1f}%")

    col_e3, col_e4, col_e5 = st.columns(3)
    with col_e3:
        metric_card("Revenue", f"{revenue:,.0f} UGX")
    with col_e4:
        metric_card("Total Cost", f"{total_cost:,.0f} UGX")
    with col_e5:
        metric_card("Net Profit", f"{profit:,.0f} UGX")

    fig = go.Figure(data=[go.Pie(
        labels=["Revenue", "Cost"],
        values=[revenue, total_cost],
        hole=0.4, marker_colors=["#2E7D32", "#D32F2F"],
        textinfo="label+value+percent+percent",
    )])
    fig.update_layout(title="Revenue vs Cost Breakdown", height=400)
    st.plotly_chart(fig, use_container_width=True)

    if st.session_state.opt_profit:
        st.markdown("### Optimization Comparison")
        opt = st.session_state.opt_profit
        comp_df = pd.DataFrame([
            ["Expected Yield (t/ha)", f"{opt.get('expected_yield_t_ha', 'N/A')}"],
            ["Price (UGX/kg)", f"{opt.get('price_ugx_per_kg', 'N/A')}"],
            ["Revenue (UGX)", f"{opt.get('revenue_ugx', 'N/A'):,.0f}"],
            ["Total Cost (UGX)", f"{opt.get('total_cost_ugx', 'N/A'):,.0f}"],
            ["Profit (UGX)", f"{opt.get('profit_ugx', 'N/A'):,.0f}"],
            ["Margin", f"{opt.get('margin_pct', 'N/A')}%"],
        ], columns=["Parameter", "Value (Optimized)"])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Irrigation Scheduling
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "irrigation":
    page_header("Irrigation Scheduling")
    st.markdown("Detailed irrigation schedule by growth stage.")

    if st.session_state.cwr_df is None:
        st.warning("CWR data is required. Compute it first on the ET₀ page.")
    else:
        cwr = st.session_state.cwr_df
        _, _, _, _, compute_irrigation_days = get_weather()
        irr_days = compute_irrigation_days(cwr)

        st.markdown("### Generate Schedule")

        if st.session_state.recommended_crop:
            crop_name = st.session_state.recommended_crop["name"]
            st.info(f"Using recommended crop: **{crop_name}**")
            id_col = "Crop_ID" if "Crop_ID" in st.session_state.crop_db.columns else "Crop_Number"
            best_row = st.session_state.crop_db[
                st.session_state.crop_db[id_col] == st.session_state.recommended_crop["crop_id"]
            ].iloc[0]
            growth_stages = best_row.get("Growth_Stages", "Initial|Development|Mid-season|Late-season")
            avg_kc = best_row.get("Avg_Kc", 1.0)
        else:
            growth_stages = "Initial|Development|Mid-season|Late-season"
            avg_kc = 1.0
            st.info("No crop selected. Using default growth stages.")

        if st.button("Generate Irrigation Schedule", type="primary", use_container_width=True):
            _, _, _, _, _, _, generate_irrigation_schedule = get_optimization()
            schedule = generate_irrigation_schedule(cwr, irr_days, growth_stages, avg_kc)
            st.session_state.opt_schedule = schedule
            calc_box(f"Growth stages: {growth_stages}")
            calc_box(f"Irrigation events: {len(schedule)}")
            calc_box(f"Each event: CWR / events per stage")
            st.success(f"Schedule generated: {len(schedule)} events")

        if st.session_state.opt_schedule:
            schedule = st.session_state.opt_schedule
            sched_df = pd.DataFrame(schedule)
            sched_df["date"] = sched_df["date"].dt.strftime("%Y-%m-%d") if "date" in sched_df.columns else ""
            st.dataframe(
                sched_df.rename(columns={
                    "stage": "Growth Stage", "date": "Date",
                    "day": "Day", "water_mm": "Water (mm)",
                }),
                use_container_width=True, hide_index=True,
            )

            total_irr = sum(s["water_mm"] for s in schedule)
            col_i1, col_i2 = st.columns(2)
            with col_i1:
                metric_card("Total Irrigation", f"{total_irr:.0f} mm")
            with col_i2:
                metric_card("Number of Events", f"{len(schedule)}")

            fig = go.Figure()
            stages_order = list(dict.fromkeys(s["stage"] for s in schedule))
            colors = ["#2E7D32", "#43A047", "#66BB6A", "#81C784", "#A5D6A7"]
            for i, stage in enumerate(stages_order):
                stage_events = [s for s in schedule if s["stage"] == stage]
                fig.add_trace(go.Bar(
                    x=[s["date"] for s in stage_events],
                    y=[s["water_mm"] for s in stage_events],
                    name=stage, marker_color=colors[i % len(colors)],
                ))
            fig.update_layout(
                height=400, title="Irrigation Schedule — MATLAB bar equivalent",
                barmode="stack", margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Date", yaxis_title="Water (mm)",
            )
            st.plotly_chart(fig, use_container_width=True)

            cum_water = np.cumsum([s["water_mm"] for s in schedule])
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=[s["date"] for s in schedule],
                y=cum_water, mode="lines+markers",
                line=dict(color="#1976D2", width=2), name="Cumulative Irrigation",
                fill="tozeroy",
            ))
            fig2.update_layout(height=300, title="Cumulative Irrigation Water Applied",
                               xaxis_title="Date", yaxis_title="Cumulative Water (mm)")
            st.plotly_chart(fig2, use_container_width=True)

            with st.expander("Irrigation Calculation Details"):
                calc_box(f"Total CWR: {cwr['CWR'].sum():.1f} mm over {irr_days} days")
                calc_box(f"Growth stages: {growth_stages}")
                calc_box("Each stage: distribute CWR / n_events, event interval = stage_duration / n_events")
        else:
            st.info("Generate the irrigation schedule to see the detailed plan.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Final Recommendation
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "recommendation":
    page_header("Final Recommendation — Consolidated Output")
    st.markdown("This page consolidates outputs from all three modules into a single comprehensive recommendation.")

    if not st.session_state.opt_ready:
        st.warning("Run optimization first using the Dashboard to see consolidated recommendations.")
    else:
        rec = st.session_state.recommended_crop

        st.markdown(
            f"""<div class="rec-card">
                <div style='color:#A5D6A7; font-size:0.9rem;'>Recommended Crop</div>
                <div style='color:white; font-size:2.2rem; font-weight:700;'>{rec['name']}</div>
                <div style='color:#C8E6C9; font-size:1.1rem;'>Suitability Score: {rec['score']}/100</div>
            </div>""",
            unsafe_allow_html=True,
        )

        tabs = st.tabs(["Summary", "Environmental", "Yield Prediction", "Economic", "Water", "Report"])

        with tabs[0]:
            st.markdown("### Consolidated Decision Support Summary")
            items = []
            if st.session_state.field_polygon:
                items.append(("Field Area", f"{st.session_state.field_area_ha:.2f} ha"))
            if st.session_state.slope_stats:
                items.append(("Mean Slope", f"{st.session_state.slope_stats[0]:.2f}°"))
            if st.session_state.weather_df is not None:
                wdf = st.session_state.weather_df
                items.append(("Avg Temperature", f"{(wdf['Tmax'] + wdf['Tmin']).mean() / 2:.1f} °C"))
                items.append(("Total Rainfall", f"{wdf['Rainfall'].sum():.0f} mm"))
            if st.session_state.opt_yield:
                items.append(("Expected Yield", f"{st.session_state.opt_yield} t/ha"))
            if st.session_state.yield_prediction_results:
                yr = st.session_state.yield_prediction_results
                items.append(("Best Yield Model", yr.get("best_model", "N/A")))
            if st.session_state.opt_profit:
                items.append(("Expected Profit", f"{st.session_state.opt_profit.get('profit_ugx', 0):,.0f} UGX"))
            if st.session_state.opt_water:
                items.append(("Irrigation Required", f"{st.session_state.opt_water['total_cwr_mm']} mm"))
            if st.session_state.ahp_results:
                items.append(("AHP Consistency", f"CR={st.session_state.ahp_results['consistency']['cr']}"))
            if st.session_state.soil_results:
                items.append(("Soil pH", f"{st.session_state.soil_results['ph_value']}"))

            for label, value in items:
                st.markdown(f"- **{label}:** {value}")

        with tabs[1]:
            st.markdown("### Environmental Factors")
            col_r1, col_r2, col_r3 = st.columns(3)
            if st.session_state.weather_df is not None:
                wdf = st.session_state.weather_df
                with col_r1:
                    metric_card("Avg Tmax", f"{wdf['Tmax'].mean():.1f} °C")
                    metric_card("Avg Tmin", f"{wdf['Tmin'].mean():.1f} °C")
                with col_r2:
                    metric_card("Total Rainfall", f"{wdf['Rainfall'].sum():.0f} mm")
                    metric_card("Avg Solar Rad.", f"{wdf['Rs'].mean():.1f} MJ/m²/d")
                with col_r3:
                    if st.session_state.slope_stats:
                        metric_card("Mean Slope", f"{st.session_state.slope_stats[0]:.2f}°")
                    if st.session_state.cwr_df is not None:
                        metric_card("Total ETc", f"{st.session_state.cwr_df['ETc'].sum():.1f} mm")
            else:
                st.info("Environmental data not loaded.")

            if st.session_state.ahp_results:
                st.markdown("### AHP Criteria Weights")
                ar = st.session_state.ahp_results
                fig = go.Figure(go.Bar(
                    x=ar["ranking"]["Weight"], y=ar["ranking"]["Criterion"],
                    orientation="h", marker_color="#2E7D32",
                    text=ar["ranking"]["Percentage"], textposition="outside",
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            if st.session_state.soil_results:
                st.markdown("### Soil Analysis")
                sr = st.session_state.soil_results
                st.markdown(f"- **Soil Type:** {sr['soil_type']}")
                st.markdown(f"- **pH:** {sr['ph_value']} — {sr['ph_suitability']}")
                st.markdown(f"- **Suitability Score:** {sr['soil_score']}/100")

        with tabs[2]:
            st.markdown("### Yield Prediction Summary")
            yr = st.session_state.yield_prediction_results
            if yr:
                col_y1, col_y2 = st.columns(2)
                with col_y1:
                    if yr.get("gru_lstm"):
                        metrics = yr["gru_lstm"]["metrics"]
                        st.markdown("**GRU-LSTM:** RMSE Test: " + str(metrics.get('rmse_test', 'N/A')))
                with col_y2:
                    if yr.get("xgboost"):
                        metrics = yr["xgboost"]["metrics"]
                        st.markdown("**XGBoost:** RMSE Test: " + str(metrics.get('rmse_test', 'N/A')))
                st.success(f"**Best Model:** {yr.get('best_model', 'N/A')} with Test RMSE = {yr.get('best_rmse', 'N/A')}")
                st.markdown(f"**Expected Yield (Optimization):** {st.session_state.opt_yield} t/ha")
            else:
                st.info("Run yield prediction to see results.")

        with tabs[3]:
            st.markdown("### Economic Analysis")
            if st.session_state.opt_profit:
                pf = st.session_state.opt_profit
                col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                with col_e1:
                    metric_card("Expected Yield", f"{pf.get('expected_yield_t_ha', 'N/A')} t/ha")
                with col_e2:
                    metric_card("Revenue", f"{pf.get('revenue_ugx', 0):,.0f} UGX")
                with col_e3:
                    metric_card("Cost", f"{pf.get('total_cost_ugx', 0):,.0f} UGX")
                with col_e4:
                    metric_card("Profit", f"{pf.get('profit_ugx', 0):,.0f} UGX")

                fig = go.Figure(data=[go.Pie(
                    labels=["Revenue", "Cost", "Profit"],
                    values=[pf.get('revenue_ugx', 0), pf.get('total_cost_ugx', 0), pf.get('profit_ugx', 0)],
                    hole=0.4, marker_colors=["#2E7D32", "#D32F2F", "#FF8F00"],
                    textinfo="label+value+percent",
                )])
                fig.update_layout(title="Economic Breakdown", height=350)
                st.plotly_chart(fig, use_container_width=True)

        with tabs[4]:
            st.markdown("### Water Management")
            if st.session_state.opt_water:
                wt = st.session_state.opt_water
                col_w1, col_w2, col_w3 = st.columns(3)
                with col_w1:
                    metric_card("Total CWR", f"{wt['total_cwr_mm']} mm")
                with col_w2:
                    metric_card("Total Volume", f"{wt['total_volume_m3']:,.0f} m³")
                with col_w3:
                    metric_card("Per Irrigation", f"{wt['per_event_m3']:,.0f} m³")

            if st.session_state.opt_wue:
                col_w4, col_w5 = st.columns(2)
                with col_w4:
                    metric_card("WUE", f"{st.session_state.opt_wue} kg/m³")
                with col_w5:
                    metric_card("IWUE", f"{st.session_state.opt_iwue} kg/m³")

            if st.session_state.opt_schedule:
                schedule = st.session_state.opt_schedule
                sched_df = pd.DataFrame(schedule)
                sched_df["date"] = sched_df["date"].dt.strftime("%Y-%m-%d") if "date" in sched_df.columns else ""
                st.dataframe(
                    sched_df.rename(columns={
                        "stage": "Stage", "date": "Date",
                        "day": "Day", "water_mm": "Water (mm)",
                    }),
                    use_container_width=True, hide_index=True,
                )

        with tabs[5]:
            st.markdown("### Generate PDF Report")
            report_checks = {
                "Crop Selection": st.session_state.crop_info is not None or st.session_state.recommended_crop is not None,
                "Farm Boundary": st.session_state.field_polygon is not None,
                "Topography": st.session_state.slope is not None,
                "Weather Data": st.session_state.weather_df is not None,
                "CWR Analysis": st.session_state.cwr_df is not None,
                "Planting Analysis": st.session_state.n_plants > 0,
                "Yield Prediction": st.session_state.yield_prediction_results is not None,
                "AHP Analysis": st.session_state.ahp_results is not None,
                "Optimization": st.session_state.opt_ready,
            }
            all_report_ready = all(report_checks.values())
            for label, ready in report_checks.items():
                st.markdown(f"{'✓' if ready else '○'} {label}")

            if all_report_ready and st.button("Generate PDF Report", type="primary", use_container_width=True):
                with st.spinner("Generating comprehensive PDF report..."):
                    from PIL import Image as PILImage
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    chart_images = []
                    cwr = st.session_state.cwr_df

                    f1, a1 = plt.subplots(figsize=(8, 4))
                    a1.plot(cwr["Date"], cwr["ET0"], label="ET₀", color="#1976D2")
                    a1.plot(cwr["Date"], cwr["ETc"], label="ETc", color="#D32F2F")
                    a1.set_title("Reference & Crop ET")
                    a1.legend()
                    a1.tick_params(axis="x", rotation=45)
                    f1.tight_layout()
                    chart_images.append(("ET₀ & ETc", f1))

                    f2, a2 = plt.subplots(figsize=(8, 4))
                    a2.bar(cwr["Date"], cwr["Rainfall"], label="Rainfall", color="#43A047", alpha=0.5)
                    a2.bar(cwr["Date"], cwr["CWR"], label="CWR", color="#D32F2F", alpha=0.7)
                    a2.set_title("Rainfall vs CWR")
                    a2.legend()
                    a2.tick_params(axis="x", rotation=45)
                    f2.tight_layout()
                    chart_images.append(("CWR vs Rainfall", f2))

                    if st.session_state.planting_points is not None and len(st.session_state.planting_points) > 0:
                        pts = st.session_state.planting_points
                        f3, a3 = plt.subplots(figsize=(8, 4))
                        a3.scatter(pts[:, 0], pts[:, 1], s=5, c="#2E7D32", alpha=0.5)
                        poly = st.session_state.field_polygon_utm or st.session_state.field_polygon
                        xs, ys = zip(*poly)
                        a3.plot(list(xs) + [xs[0]], list(ys) + [ys[0]], "r-", lw=2)
                        a3.set_title("Planting Grid")
                        a3.set_aspect("equal")
                        f3.tight_layout()
                        chart_images.append(("Planting Grid", f3))

                    pil_images = []
                    for title, fig in chart_images:
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                        buf.seek(0)
                        pil_images.append((title, PILImage.open(buf)))
                        plt.close(fig)

                    _, _, compute_slope_stats, compute_surface_area = get_terrain()
                    _, _, _, _, compute_irrigation_days = get_weather()
                    generate_pdf_report = get_report()
                    report_data = {
                        "crop_name": rec["name"] if rec else "N/A",
                        "area_ha": st.session_state.field_area_ha,
                        "n_plants": st.session_state.n_plants,
                        "planting_density": st.session_state.planting_density,
                        "mean_slope_deg": compute_slope_stats(st.session_state.slope)[0] if st.session_state.slope is not None else 0,
                        "surface_area_ha": compute_surface_area(
                            st.session_state.field_area_ha,
                            np.radians(compute_slope_stats(st.session_state.slope)[0])
                        ) if st.session_state.field_area_ha > 0 and st.session_state.slope is not None else 0,
                        "avg_eto": cwr["ET0"].mean(),
                        "avg_etc": cwr["ETc"].mean(),
                        "total_irrigation_mm": cwr["CWR"].sum(),
                        "irrigation_days": compute_irrigation_days(cwr),
                        "revenue": st.session_state.opt_profit.get("revenue_ugx", 0) if st.session_state.opt_profit else 0,
                        "cost": st.session_state.opt_profit.get("total_cost_ugx", 0) if st.session_state.opt_profit else 0,
                        "profit": st.session_state.opt_profit.get("profit_ugx", 0) if st.session_state.opt_profit else 0,
                        "profit_margin_pct": st.session_state.opt_profit.get("margin_pct", 0) if st.session_state.opt_profit else 0,
                        "charts": pil_images,
                        "lat": 0.35, "lon": 33.75,
                        "yield_prediction_best_model": st.session_state.yield_prediction_results.get("best_model", "N/A") if st.session_state.yield_prediction_results else "N/A",
                        "best_rmse": st.session_state.yield_prediction_results.get("best_rmse", "N/A") if st.session_state.yield_prediction_results else "N/A",
                        "expected_yield": st.session_state.opt_yield,
                        "wue": st.session_state.opt_wue,
                        "iwue": st.session_state.opt_iwue,
                        "weather_summary": {
                            "avg_tmax": st.session_state.weather_df["Tmax"].mean(),
                            "avg_tmin": st.session_state.weather_df["Tmin"].mean(),
                            "total_rainfall": st.session_state.weather_df["Rainfall"].sum(),
                            "avg_rs": st.session_state.weather_df["Rs"].mean(),
                        },
                    }
                    pdf_buffer = generate_pdf_report(report_data)
                    st.session_state.report_buffer = pdf_buffer
                    st.session_state.report_ready = True
                st.success("Report generated!")

            if st.session_state.report_ready and st.session_state.report_buffer is not None:
                st.download_button(
                    label="Download PDF Report",
                    data=st.session_state.report_buffer,
                    file_name=f"precision_ag_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#999; font-size:0.85rem;'>"
        "Precision Agriculture Decision Support System v2.0</p>",
        unsafe_allow_html=True,
    )
