"""PARKNEROO multimodal monitoring platform.
Connects Voice, Handwriting, Finger-Tapping, Gait, and Tremor assessments
with Personal Baseline and Longitudinal Progress tracking.
Run: streamlit run app.py
"""
import io
import json
import os
import sqlite3
import tempfile
from datetime import datetime

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

import database
import finger_tap_analysis
import gait_analysis
import tremor_analysis
import vision_features
from features import FEATURES, extract_features

st.set_page_config(
    page_title="PARKNEROO | Digital Biomarker Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme State & Sidebar Switcher
if "theme" not in st.session_state:
    st.session_state["theme"] = "🌙 Dark Mode"

st.sidebar.title("🧠 PARKNEROO")
st.sidebar.caption("Multimodal Digital Biomarker Platform")

theme_choice = st.sidebar.radio(
    "🎨 Display Theme",
    ["🌙 Dark Mode", "☀️ Light Mode"],
    index=0 if st.session_state.get("theme", "🌙 Dark Mode") == "🌙 Dark Mode" else 1,
    horizontal=True,
    key="app_theme_switcher"
)
is_dark = "Dark" in theme_choice
st.session_state["theme"] = theme_choice
st.sidebar.markdown("---")

# Plotly Dynamic Theme Colors
plotly_template = "plotly_dark" if is_dark else "plotly_white"
chart_font_color = "#cbd5e1" if is_dark else "#1e293b"
chart_grid_color = "rgba(255, 255, 255, 0.08)" if is_dark else "#e2e8f0"
chart_polar_bg = "rgba(15, 23, 42, 0.6)" if is_dark else "rgba(248, 250, 252, 0.85)"
chart_legend_bg = "rgba(15, 23, 42, 0.7)" if is_dark else "rgba(255, 255, 255, 0.9)"
chart_legend_border = "rgba(255, 255, 255, 0.1)" if is_dark else "rgba(0, 0, 0, 0.1)"
gauge_num_color = "#f8fafc" if is_dark else "#0f172a"
gauge_bg = "rgba(255, 255, 255, 0.05)" if is_dark else "rgba(0, 0, 0, 0.04)"

# Dynamic CSS Injection based on Selected Theme
if is_dark:
    theme_specific_css = """
    .stApp {
        background: radial-gradient(circle at 18% 10%, rgba(56, 189, 248, 0.08) 0%, transparent 45%),
                    radial-gradient(circle at 82% 14%, rgba(129, 140, 248, 0.09) 0%, transparent 45%),
                    radial-gradient(circle at 50% 85%, rgba(16, 185, 129, 0.05) 0%, transparent 50%),
                    #070b14 !important;
        color: #f1f5f9;
    }
    header[data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090e1a 0%, #060911 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 50%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 25px rgba(56, 189, 248, 0.08);
    }
    .hero-title {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 45%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle { color: #94a3b8; }
    .hero-stat-pill {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.09);
        color: #cbd5e1;
    }
    .tab-hero-title {
        background: linear-gradient(135deg, #f8fafc 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        color: #f8fafc;
    }
    .tab-hero-sub { color: #94a3b8; }
    .ribbon-pill {
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
    }
    .ribbon-pill-user {
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38bdf8;
    }
    .ribbon-pill-sessions {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #cbd5e1;
    }
    .ribbon-pill-calib {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
    }
    .status-guidance-text { color: #94a3b8; }
    .status-banner-stable {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.35) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.12), 0 2px 8px rgba(0, 0, 0, 0.4);
    }
    .status-banner-flagged {
        background: linear-gradient(135deg, rgba(136, 19, 55, 0.35) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(244, 63, 94, 0.55);
        box-shadow: 0 8px 32px rgba(244, 63, 94, 0.16), 0 2px 8px rgba(0, 0, 0, 0.4);
    }
    .status-banner-building {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.35) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(245, 158, 11, 0.45);
        box-shadow: 0 8px 32px rgba(245, 158, 11, 0.12), 0 2px 8px rgba(0, 0, 0, 0.4);
    }
    .status-icon-box {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .status-title { color: #ffffff !important; }
    .status-desc { color: #cbd5e1; }
    .launcher-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(255, 255, 255, 0.09);
    }
    .launcher-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
    }
    .launcher-icon { background: rgba(255, 255, 255, 0.05); }
    .launcher-name { color: #f1f5f9; }
    .launcher-sub { color: #94a3b8; }
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
    }
    .kpi-title { color: #f1f5f9; }
    .kpi-value { color: #ffffff; }
    .kpi-base { color: #94a3b8; }
    div[data-testid="stMetric"], .stMetric {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9)) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3) !important;
    }
    div[data-testid="stMetricLabel"] p, div[data-testid="stMetricLabel"] label, div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    div[data-testid="stMetricValue"] div, div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; }
    .stTabs [data-baseweb="tab"]:hover { color: #f8fafc; background: rgba(255, 255, 255, 0.05); }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(129, 140, 248, 0.2)) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.5) !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.2);
    }
    .patient-id-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(56, 189, 248, 0.25);
    }
    .patient-id-name { color: #f8fafc; }
    .patient-id-label { color: #94a3b8; }
    .patient-id-sessions { color: #38bdf8; }
    .chart-panel-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.09);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    }
    .chart-panel-title { color: #f1f5f9; }
    .chart-panel-desc { color: #94a3b8; }
    .disclaimer-card {
        background: rgba(30, 41, 59, 0.65) !important;
        border-left: 4px solid #38bdf8 !important;
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #cbd5e1 !important;
    }
    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 2px dashed rgba(56, 189, 248, 0.3) !important;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    [data-baseweb="select"] > div {
        background: rgba(15, 23, 42, 0.8) !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
        color: #f8fafc !important;
    }
    [data-baseweb="input"] {
        background: rgba(15, 23, 42, 0.8) !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
        color: #f8fafc !important;
    }
    """
else:
    theme_specific_css = """
    .stApp {
        background: radial-gradient(circle at 18% 10%, rgba(2, 132, 199, 0.05) 0%, transparent 45%),
                    radial-gradient(circle at 82% 14%, rgba(99, 102, 241, 0.05) 0%, transparent 45%),
                    #f8fafc !important;
        color: #0f172a !important;
    }
    header[data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    h1, h2, h3, h4 {
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }
    p, span, label { color: #334155; }
    .hero-banner {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%) !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05) !important;
    }
    .hero-title {
        background: linear-gradient(135deg, #0284c7 0%, #4f46e5 50%, #9333ea 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    .hero-subtitle { color: #475569 !important; }
    .hero-stat-pill {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #334155 !important;
    }
    .tab-hero-title {
        background: linear-gradient(135deg, #0f172a 0%, #334155 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        color: #0f172a !important;
    }
    .tab-hero-sub { color: #475569 !important; }
    .ribbon-pill {
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
    }
    .ribbon-pill-user {
        background: #e0f2fe !important;
        border: 1px solid #7dd3fc !important;
        color: #0369a1 !important;
    }
    .ribbon-pill-sessions {
        background: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        color: #334155 !important;
    }
    .ribbon-pill-calib {
        background: #dcfce7 !important;
        border: 1px solid #86efac !important;
        color: #15803d !important;
    }
    .status-guidance-text { color: #64748b !important; }
    .status-banner-stable {
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%) !important;
        border: 1px solid #86efac !important;
        box-shadow: 0 4px 16px rgba(22, 101, 52, 0.08) !important;
    }
    .status-banner-stable .status-title { color: #14532d !important; }
    .status-banner-stable .status-desc { color: #166534 !important; }
    .status-banner-flagged {
        background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%) !important;
        border: 1px solid #fca5a5 !important;
        box-shadow: 0 4px 16px rgba(185, 28, 28, 0.08) !important;
    }
    .status-banner-flagged .status-title { color: #7f1d1d !important; }
    .status-banner-flagged .status-desc { color: #991b1b !important; }
    .status-banner-building {
        background: linear-gradient(135deg, #fffbeb 0%, #ffffff 100%) !important;
        border: 1px solid #fde68a !important;
        box-shadow: 0 4px 16px rgba(180, 83, 9, 0.08) !important;
    }
    .status-banner-building .status-title { color: #78350f !important; }
    .status-banner-building .status-desc { color: #92400e !important; }
    .status-icon-box {
        background: #f1f5f9 !important;
        border: 1px solid #e2e8f0 !important;
    }
    .status-title { color: #0f172a !important; }
    .status-desc { color: #475569 !important; }
    .flagged-chip {
        background: #fee2e2 !important;
        border: 1px solid #fca5a5 !important;
        color: #991b1b !important;
    }
    .launcher-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    }
    .launcher-card:hover {
        border-color: #0284c7 !important;
        box-shadow: 0 6px 16px rgba(2, 132, 199, 0.12) !important;
    }
    .launcher-icon { background: #f1f5f9 !important; }
    .launcher-name { color: #0f172a !important; }
    .launcher-sub { color: #64748b !important; }
    .kpi-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.05) !important;
    }
    .kpi-title { color: #1e293b !important; }
    .kpi-value { color: #0f172a !important; }
    .kpi-base { color: #64748b !important; }
    div[data-testid="stMetric"], .stMetric {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    }
    div[data-testid="stMetricLabel"] p, div[data-testid="stMetricLabel"] label, div[data-testid="stMetricLabel"] {
        color: #64748b !important;
    }
    div[data-testid="stMetricValue"] div, div[data-testid="stMetricValue"] {
        color: #0f172a !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #f1f5f9 !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    }
    .stTabs [data-baseweb="tab"] { color: #64748b !important; }
    .stTabs [data-baseweb="tab"]:hover { color: #0f172a !important; background: rgba(0, 0, 0, 0.04) !important; }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #0284c7 !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
    }
    .patient-id-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    }
    .patient-id-name { color: #0f172a !important; }
    .patient-id-label { color: #64748b !important; }
    .patient-id-sessions { color: #0284c7 !important; }
    .chart-panel-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 3px 14px rgba(0, 0, 0, 0.05) !important;
    }
    .chart-panel-title { color: #0f172a !important; }
    .chart-panel-desc { color: #64748b !important; }
    .disclaimer-card {
        background: #f8fafc !important;
        border-left: 4px solid #0284c7 !important;
        border: 1px solid #e2e8f0 !important;
        color: #334155 !important;
    }
    [data-testid="stFileUploader"] {
        background: #f8fafc !important;
        border: 2px dashed #93c5fd !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #0284c7 !important;
        background: #f0f9ff !important;
    }
    [data-testid="stDataFrame"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
    }
    [data-baseweb="select"] > div {
        background: #ffffff !important;
        border-color: #cbd5e1 !important;
        color: #0f172a !important;
    }
    [data-baseweb="input"] {
        background: #ffffff !important;
        border-color: #cbd5e1 !important;
        color: #0f172a !important;
    }
    """

# Custom CSS for modern, high-aesthetic clinical/research dashboard
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    {theme_specific_css}

    /* Pulsing status dots */
    .pulsing-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulseGreen 2s infinite;
    }}
    .pulsing-dot-green {{
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulseGreen 2s infinite;
    }}
    .pulsing-dot-red {{
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #f43f5e;
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.7);
        animation: pulseRed 2s infinite;
    }}
    .pulsing-dot-amber {{
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #f59e0b;
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7);
        animation: pulseAmber 2s infinite;
    }}
    @keyframes pulseGreen {{
        0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
        70% {{ transform: scale(1.15); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }}
        100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
    }}
    @keyframes pulseRed {{
        0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.7); }}
        70% {{ transform: scale(1.15); box-shadow: 0 0 0 8px rgba(244, 63, 94, 0); }}
        100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(244, 63, 94, 0); }}
    }}
    @keyframes pulseAmber {{
        0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }}
        70% {{ transform: scale(1.15); box-shadow: 0 0 0 8px rgba(245, 158, 11, 0); }}
        100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }}
    }}

    /* Hero Banner Header */
    .hero-banner {{
        position: relative;
        overflow: hidden;
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 20px;
    }}
    .hero-glow {{
        position: absolute;
        top: -60px;
        right: -60px;
        width: 280px;
        height: 280px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.22) 0%, rgba(129, 140, 248, 0.1) 50%, transparent 70%);
        pointer-events: none;
        border-radius: 50%;
    }}
    .hero-badge {{
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin-bottom: 10px;
    }}
    .hero-title {{
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0 0 6px 0;
        line-height: 1.15;
    }}
    .hero-subtitle {{
        font-size: 0.96rem;
        max-width: 880px;
        margin: 0 0 14px 0;
        line-height: 1.5;
    }}
    .hero-stats-row {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }}
    .hero-stat-pill {{
        border-radius: 8px;
        padding: 5px 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }}

    /* Custom Status Banners */
    .status-banner {{
        display: flex;
        gap: 18px;
        align-items: flex-start;
        border-radius: 18px;
        padding: 20px 24px;
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(14px);
    }}
    .status-icon-box {{
        font-size: 2.2rem;
        width: 54px;
        height: 54px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }}
    .status-content {{
        flex: 1;
    }}
    .status-top-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
        flex-wrap: wrap;
        gap: 8px;
    }}
    .status-badge {{
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}
    .status-badge-stable {{
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }}
    .status-badge-flagged {{
        background: rgba(244, 63, 94, 0.2);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.4);
    }}
    .status-badge-building {{
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }}
    .status-meta {{
        font-size: 0.78rem;
        color: #94a3b8;
    }}
    .status-title {{
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        line-height: 1.3;
    }}
    .status-desc {{
        font-size: 0.88rem;
        margin: 0;
        line-height: 1.5;
    }}
    .flagged-chip-row {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 10px;
    }}
    .flagged-chip {{
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }}

    /* Quick Assessment Launcher Grid */
    .launcher-grid {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        margin-bottom: 22px;
    }}
    .launcher-card {{
        border-radius: 14px;
        padding: 12px 14px;
        display: flex;
        align-items: center;
        gap: 10px;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }}
    .launcher-card:hover {{
        transform: translateY(-2px);
    }}
    .launcher-icon {{
        font-size: 1.4rem;
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }}
    .launcher-info {{
        flex: 1;
        min-width: 0;
    }}
    .launcher-name {{
        font-size: 0.82rem;
        font-weight: 700;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .launcher-sub {{
        font-size: 0.72rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    /* Custom KPI Cards */
    .kpi-card {{
        position: relative;
        overflow: hidden;
        border-top: 3px solid #38bdf8 !important;
        border-radius: 16px;
        padding: 16px 14px;
        backdrop-filter: blur(12px);
        margin-bottom: 12px;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s ease, border-color 0.25s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-3px);
    }}
    .kpi-card-voice {{ border-top-color: #38bdf8 !important; }}
    .kpi-card-voice:hover {{ border-color: rgba(56, 189, 248, 0.6) !important; box-shadow: 0 8px 25px rgba(56, 189, 248, 0.22) !important; }}
    .kpi-card-spiral {{ border-top-color: #f59e0b !important; }}
    .kpi-card-spiral:hover {{ border-color: rgba(245, 158, 11, 0.6) !important; box-shadow: 0 8px 25px rgba(245, 158, 11, 0.22) !important; }}
    .kpi-card-tap {{ border-top-color: #10b981 !important; }}
    .kpi-card-tap:hover {{ border-color: rgba(16, 185, 129, 0.6) !important; box-shadow: 0 8px 25px rgba(16, 185, 129, 0.22) !important; }}
    .kpi-card-gait {{ border-top-color: #a855f7 !important; }}
    .kpi-card-gait:hover {{ border-color: rgba(168, 85, 247, 0.6) !important; box-shadow: 0 8px 25px rgba(168, 85, 247, 0.22) !important; }}
    .kpi-card-tremor {{ border-top-color: #f43f5e !important; }}
    .kpi-card-tremor:hover {{ border-color: rgba(244, 63, 94, 0.6) !important; box-shadow: 0 8px 25px rgba(244, 63, 94, 0.22) !important; }}

    .kpi-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }}
    .kpi-title {{
        font-weight: 700;
        font-size: 0.88rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .kpi-badge {{
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 9999px;
        letter-spacing: 0.03em;
    }}
    .kpi-badge-stable {{
        background: rgba(34, 197, 94, 0.16);
        color: #16a34a;
        border: 1px solid rgba(74, 222, 128, 0.35);
    }}
    .kpi-badge-flagged {{
        background: rgba(239, 68, 68, 0.18);
        color: #dc2626;
        border: 1px solid rgba(248, 113, 113, 0.45);
    }}
    .kpi-badge-building {{
        background: rgba(245, 158, 11, 0.18);
        color: #d97706;
        border: 1px solid rgba(251, 191, 36, 0.4);
    }}
    .kpi-badge-empty {{
        background: rgba(148, 163, 184, 0.15);
        color: #64748b;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }}
    .kpi-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 800;
        margin: 4px 0 8px 0;
        letter-spacing: -0.02em;
    }}
    .kpi-sub {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.78rem;
        padding-top: 6px;
        border-top: 1px solid rgba(150, 150, 150, 0.12);
    }}
    .delta-stable {{
        color: #16a34a;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }}
    .delta-flagged {{
        color: #dc2626;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }}
    .delta-empty {{
        color: #64748b;
        font-style: italic;
    }}

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        padding: 6px 8px;
        border-radius: 14px;
        backdrop-filter: blur(14px);
        margin-bottom: 22px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.92rem;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: transparent !important;
    }}

    /* Patient profile card in sidebar */
    .patient-id-card {{
        display: flex;
        align-items: center;
        gap: 12px;
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 16px;
    }}
    .patient-id-avatar {{
        font-size: 1.8rem;
        background: rgba(56, 189, 248, 0.15);
        border-radius: 10px;
        width: 44px;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .patient-id-label {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }}
    .patient-id-name {{
        font-size: 1.1rem;
        font-weight: 700;
    }}
    .patient-id-sessions {{
        font-size: 0.75rem;
        font-weight: 500;
    }}

    /* Chart Frame Container */
    .chart-panel-card {{
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 16px;
    }}
    .chart-panel-title {{
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0 0 4px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .chart-panel-desc {{
        font-size: 0.8rem;
        margin: 0 0 12px 0;
    }}

    /* Clinical Disclaimer Banner */
    .disclaimer-card {{
        padding: 12px 18px !important;
        border-radius: 10px !important;
        margin-top: 10px;
        margin-bottom: 15px;
        font-size: 0.88rem !important;
    }}
    .section-desc {{
        font-size: 0.95rem;
        margin-bottom: 15px;
    }}

    /* Primary Button Styling */
    div.stButton > button:first-child {{
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 9px 18px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
        transition: all 0.2s ease !important;
    }}
    div.stButton > button:first-child:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5) !important;
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_voice_model():
    try:
        return joblib.load("outputs/model.joblib")
    except Exception as e:
        return None

bundle = load_voice_model()
con = database.get_connection()

# Sidebar: Patient Profile & Cross-Session Configuration
st.sidebar.subheader("Patient Profile")
all_users = database.get_all_users(con)
if "demo_user" not in all_users:
    all_users = ["demo_user"] + all_users

user_selection_type = st.sidebar.radio("Profile Select", ["Existing Profile", "New Profile"], horizontal=True)
if user_selection_type == "Existing Profile":
    user = st.sidebar.selectbox("User ID", all_users, index=0)
else:
    user = st.sidebar.text_input("New User ID", "patient_001")

# Quick session count
total_sessions = 0
for tbl in ["voice_sessions", "spiral_sessions", "finger_tap_sessions", "gait_sessions", "tremor_sessions"]:
    try:
        total_sessions += con.execute(f"SELECT COUNT(*) FROM {tbl} WHERE user=?", (user,)).fetchone()[0]
    except Exception:
        pass

st.sidebar.markdown(f"""
<div class="patient-id-card">
    <div class="patient-id-avatar">👤</div>
    <div>
        <div class="patient-id-label">Active Subject Profile</div>
        <div class="patient-id-name">{user}</div>
        <div class="patient-id-sessions">📊 {total_sessions} total sessions logged</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.subheader("Evaluation Tools")
if st.sidebar.button("📥 Load Sample Baseline Data", help="Populates 4 test sessions across all 5 modalities for demo/testing"):
    database.seed_sample_patient_data(user, con)
    st.sidebar.success(f"Sample data loaded for '{user}'!")
    st.rerun()

if st.sidebar.button("🗑️ Reset Profile History", help="Clears historical records for selected User ID"):
    for tbl in ["voice_sessions", "spiral_sessions", "finger_tap_sessions", "gait_sessions", "tremor_sessions"]:
        con.execute(f"DELETE FROM {tbl} WHERE user=?", (user,))
    con.commit()
    st.sidebar.warning(f"History reset for '{user}'")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class='disclaimer-card'>
    <strong>⚠️ Clinical Disclaimer:</strong><br>
    PARKNEROO is an AI-assisted monitoring and research platform.
    <strong>Never interpret a change by itself as proof of Parkinson’s disease.</strong>
    Changes may reflect normal biological variability, fatigue, medication timing, or sensor noise.
</div>
""", unsafe_allow_html=True)

# Main Hero Banner Header
st.markdown("""
<div class="hero-banner">
    <div class="hero-glow"></div>
    <div class="hero-content">
        <div class="hero-badge">
            <span class="pulsing-dot"></span>
            <span>AI MULTIMODAL NEUROLOGICAL ENGINE &bull; ACTIVE</span>
        </div>
        <h1 class="hero-title">PARKNEROO</h1>
        <p class="hero-subtitle">
            Non-invasive longitudinal digital biomarker tracking across 
            <strong>Voice Phonation, Handwriting Kinematics, Finger-Tapping, Gait Dynamics, and Tremor Oscillations</strong>.
        </p>
        <div class="hero-stats-row">
            <div class="hero-stat-pill">🔒 SQLite Encrypted Local Vault</div>
            <div class="hero-stat-pill">🧬 5 Multimodal Biomarkers</div>
            <div class="hero-stat-pill">🎯 Baseline Shift Z-Score Detection</div>
            <div class="hero-stat-pill">⚕️ Research & Monitoring Support</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs definition
tab_dash, tab_voice, tab_handwriting, tab_finger, tab_gait, tab_tremor = st.tabs([
    "📊 Personal Baseline & Progress",
    "🎤 Voice Analysis",
    "✍️ Handwriting Kinematics",
    "👆 Finger-Tapping Test",
    "🚶 Gait & Walking Analysis",
    "〰️ Tremor & Movement Signal"
])

# ==============================================================================
# TAB 1: PERSONAL BASELINE & PROGRESS DASHBOARD (THE CENTRAL HUB)
# ==============================================================================
with tab_dash:
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:10px;">
        <div>
            <h2 class="tab-hero-title" style="margin:0 0 4px 0; font-size:1.8rem; font-weight:800;">
                📊 Personal Baseline & Progress Dashboard
            </h2>
            <p class="tab-hero-sub" style="margin:0; font-size:0.92rem;">
                The central neurological hub connecting Voice, Handwriting, Finger-Tapping, Gait, and Tremor assessments.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load history across all 5 modalities
    df_voice = database.get_modality_history(user, "voice", con)
    df_spiral = database.get_modality_history(user, "spiral", con)
    df_tap = database.get_modality_history(user, "finger_tap", con)
    df_gait = database.get_modality_history(user, "gait", con)
    df_tremor = database.get_modality_history(user, "tremor", con)
    
    # Analyze baseline shifts for each modality
    res_voice = database.analyze_baseline_shift(df_voice, "prob", higher_is_worse=True)
    res_spiral = database.analyze_baseline_shift(df_spiral, "tremor", higher_is_worse=True)
    res_tap = database.analyze_baseline_shift(df_tap, "rhythm_cv", higher_is_worse=True)
    res_gait = database.analyze_baseline_shift(df_gait, "left_right_asymmetry", higher_is_worse=True)
    res_tremor = database.analyze_baseline_shift(df_tremor, "tremor_power", higher_is_worse=True)
    
    all_res = [res_voice, res_spiral, res_tap, res_gait, res_tremor]
    established_baselines = sum(1 for r in all_res if r["session_count"] >= 3)
    active_domains = sum(1 for r in all_res if r["session_count"] >= 1)
    
    # Flagging review check
    flagged_modalities = []
    if res_voice["is_flagged"]:
        flagged_modalities.append(("Voice Phonation", res_voice["description"]))
    if res_spiral["is_flagged"]:
        flagged_modalities.append(("Handwriting Spiral", res_spiral["description"]))
    if res_tap["is_flagged"]:
        flagged_modalities.append(("Finger-Tapping Rhythm", res_tap["description"]))
    if res_gait["is_flagged"]:
        flagged_modalities.append(("Gait Asymmetry", res_gait["description"]))
    if res_tremor["is_flagged"]:
        flagged_modalities.append(("Tremor Signal Power", res_tremor["description"]))
        
    # Patient Status Quick Ribbon
    col_rib1, col_rib2 = st.columns([3, 1.2])
    with col_rib1:
        st.markdown(f"""
        <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:14px;">
            <span class="ribbon-pill ribbon-pill-user">
                👤 Subject ID: {user}
            </span>
            <span class="ribbon-pill ribbon-pill-sessions">
                📊 Sessions: {total_sessions} Total Logged
            </span>
            <span class="ribbon-pill ribbon-pill-calib">
                🎯 Calibration: {established_baselines}/5 Baselines Established
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col_rib2:
        if total_sessions < 4:
            if st.button("⚡ Populate 4-Session Demo Data", key="tab1_seed_btn", use_container_width=True, help="Instantly seeds 4 progressive sessions across Voice, Spiral, Tap, Gait, and Tremor"):
                database.seed_sample_patient_data(user, con)
                st.rerun()

    # High-Aesthetic Longitudinal Status Banner
    if flagged_modalities:
        chips_html = "".join([f"<span class='flagged-chip'>⚠️ {mod}: {desc}</span>" for mod, desc in flagged_modalities])
        st.markdown(f"""
        <div class="status-banner status-banner-flagged">
            <div class="status-icon-box" style="background:rgba(244,63,94,0.15); border-color:rgba(244,63,94,0.35);">⚠️</div>
            <div class="status-content">
                <div class="status-top-row">
                    <span class="status-badge status-badge-flagged"><span class="pulsing-dot-red"></span> STATISTICAL SHIFT DETECTED &bull; REVIEW ADVISED</span>
                    <span class="status-meta">{len(flagged_modalities)} Domain(s) Exhibiting Sustained Divergence</span>
                </div>
                <h3 class="status-title">Persistent Biomarker Divergence from Personal Baseline</h3>
                <p class="status-desc">PARKNEROO detected sustained statistical shifts (|z| &ge; 1.75 across sliding multi-session window) relative to your calibrated reference baseline. Changes may reflect kinematic variation, motor fatigue, medication response, or biological fluctuation:</p>
                <div class="flagged-chip-row">
                    {chips_html}
                </div>
                <p class="status-guidance-text" style="font-size:0.78rem; margin-top:8px;">💡 <em>Clinical Guidance: Export the clinical dossier below to review trends with your neurologist or care specialist.</em></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif established_baselines >= 1:
        st.markdown(f"""
        <div class="status-banner status-banner-stable">
            <div class="status-icon-box" style="background:rgba(16,185,129,0.15); border-color:rgba(16,185,129,0.35);">🛡️</div>
            <div class="status-content">
                <div class="status-top-row">
                    <span class="status-badge status-badge-stable"><span class="pulsing-dot-green"></span> BASELINE CONGRUENT &bull; STABLE</span>
                    <span class="status-meta">{established_baselines} Domain Baselines Calibrated</span>
                </div>
                <h3 class="status-title">All Monitored Biomarkers Consistent with Reference Baseline</h3>
                <p class="status-desc">Recent measurements for subject <strong>{user}</strong> remain within normal statistical tolerance (|z| &lt; 1.75). Longitudinal variance across voice phonation, drawing kinematics, tapping rhythm, gait patterns, and tremor power reflects stability.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-banner status-banner-building">
            <div class="status-icon-box" style="background:rgba(245,158,11,0.15); border-color:rgba(245,158,11,0.35);">⏳</div>
            <div class="status-content">
                <div class="status-top-row">
                    <span class="status-badge status-badge-building"><span class="pulsing-dot-amber"></span> ESTABLISHING PERSONAL BASELINE</span>
                    <span class="status-meta">{total_sessions} Session(s) Recorded</span>
                </div>
                <h3 class="status-title">Initial Baseline Calibration In Progress</h3>
                <p class="status-desc">PARKNEROO establishes a personalized mathematical baseline from your first 3 sessions in each modality to filter out diurnal and day-to-day variance. Complete assessments or load sample data to view longitudinal shift analysis.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Multimodal Assessment Quick Launcher Grid
    st.markdown("""
    <div class="launcher-grid">
        <div class="launcher-card">
            <div class="launcher-icon">🎤</div>
            <div class="launcher-info">
                <div class="launcher-name">Voice Phonation</div>
                <div class="launcher-sub">""" + f"{res_voice['session_count']} sessions &bull; {res_voice['status']}" + """</div>
            </div>
        </div>
        <div class="launcher-card">
            <div class="launcher-icon">✍️</div>
            <div class="launcher-info">
                <div class="launcher-name">Spiral Kinematics</div>
                <div class="launcher-sub">""" + f"{res_spiral['session_count']} sessions &bull; {res_spiral['status']}" + """</div>
            </div>
        </div>
        <div class="launcher-card">
            <div class="launcher-icon">👆</div>
            <div class="launcher-info">
                <div class="launcher-name">Finger-Tapping</div>
                <div class="launcher-sub">""" + f"{res_tap['session_count']} sessions &bull; {res_tap['status']}" + """</div>
            </div>
        </div>
        <div class="launcher-card">
            <div class="launcher-icon">🚶</div>
            <div class="launcher-info">
                <div class="launcher-name">Gait & Walking</div>
                <div class="launcher-sub">""" + f"{res_gait['session_count']} sessions &bull; {res_gait['status']}" + """</div>
            </div>
        </div>
        <div class="launcher-card">
            <div class="launcher-icon">〰️</div>
            <div class="launcher-info">
                <div class="launcher-name">Tremor Oscillations</div>
                <div class="launcher-sub">""" + f"{res_tremor['session_count']} sessions &bull; {res_tremor['status']}" + """</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Modality KPI Summary Cards
    st.markdown("### 🔍 Modality Summary Cards")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    def render_kpi(col, title, icon, res, unit="", help_text="", card_type="voice"):
        with col:
            status = res["status"]
            if status == "Stable":
                badge_html = "<span class='kpi-badge kpi-badge-stable'>● Stable</span>"
                bar_color = "#4ade80"
                bar_width = "100%"
            elif status == "Review Recommended":
                badge_html = "<span class='kpi-badge kpi-badge-flagged'>▲ Review Needed</span>"
                bar_color = "#f87171"
                bar_width = "100%"
            elif status == "Baseline Building":
                count = res['session_count']
                pct = min(100, int((count / 3.0) * 100))
                badge_html = f"<span class='kpi-badge kpi-badge-building'>⏳ {count}/3 Base</span>"
                bar_color = "#fbbf24"
                bar_width = f"{pct}%"
            else:
                badge_html = "<span class='kpi-badge kpi-badge-empty'>No Sessions</span>"
                bar_color = "#334155"
                bar_width = "0%"
                
            if res["baseline_mean"] is not None and res["recent_mean"] is not None:
                curr_str = f"{res['recent_mean']:.2f}{unit}"
                base_str = f"Base: {res['baseline_mean']:.2f}{unit}"
                z_val = res.get('z_score', 0.0)
                delta_val = res.get('delta', 0.0)
                sign = "+" if delta_val >= 0 else ""
                arrow = "▲ " if delta_val > 0.01 else ("▼ " if delta_val < -0.01 else "● ")
                delta_class = "delta-flagged" if res["is_flagged"] else "delta-stable"
                delta_html = f"<span class='{delta_class}'>{arrow}{sign}{delta_val:.2f}{unit} (z={z_val:+.1f})</span>"
            else:
                curr_str = "--"
                base_str = "No baseline yet"
                delta_html = "<span class='delta-empty'>Awaiting test</span>"
                
            card_html = f"""
            <div class='kpi-card kpi-card-{card_type}' title='{help_text}'>
                <div class='kpi-header'>
                    <span class='kpi-title'>{icon} {title}</span>
                    {badge_html}
                </div>
                <div class='kpi-value'>{curr_str}</div>
                <div class='kpi-sub'>
                    <span class='kpi-base'>{base_str}</span>
                    <span class='kpi-delta'>{delta_html}</span>
                </div>
                <div style='background: rgba(255,255,255,0.06); height: 3px; border-radius: 2px; margin-top: 10px; overflow: hidden;'>
                    <div style='background: {bar_color}; width: {bar_width}; height: 100%; border-radius: 2px; transition: width 0.4s ease;'></div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
                
    render_kpi(c1, "Voice Prob", "🎤", res_voice, help_text="Probability from acoustic voice model", card_type="voice")
    render_kpi(c2, "Spiral Tremor", "✍️", res_spiral, help_text="Kinematic tremor index from spiral drawing", card_type="spiral")
    render_kpi(c3, "Tap Rhythm CV", "👆", res_tap, "%", help_text="Variability of inter-tap intervals", card_type="tap")
    render_kpi(c4, "Gait Asymmetry", "🚶", res_gait, "%", help_text="Left-right step timing asymmetry", card_type="gait")
    render_kpi(c5, "Tremor Power", "〰️", res_tremor, help_text="Movement power in 3.5-7.5 Hz Parkinsonian band", card_type="tremor")
    
    st.markdown("---")
    
    # Visual Analytics: Stability Score + Radar + Longitudinal Trends + Z-Score Profile
    col_chart1, col_chart2 = st.columns([1.1, 1.4])
    
    with col_chart1:
        st.markdown("""
        <div class="chart-panel-card">
            <div class="chart-panel-title">🛡️ Multimodal Baseline & Stability Index</div>
            <div class="chart-panel-desc">Composite index and normalized personal baseline vs recent measurements across all 5 domains.</div>
        """, unsafe_allow_html=True)
        
        # Calculate Composite Biomarker Stability Index (0 - 100)
        if established_baselines >= 1:
            base_score = 98 - (len(flagged_modalities) * 15) - ((5 - established_baselines) * 3)
            stability_score = max(35, min(98, base_score))
            if len(flagged_modalities) > 0:
                gauge_color = "#f43f5e"
                gauge_status = f"REVIEW ADVISED ({len(flagged_modalities)} SHIFTS)"
            else:
                gauge_color = "#10b981"
                gauge_status = "STABLE & CONGRUENT"
        else:
            stability_score = max(20, min(85, int((total_sessions / 12.0) * 85)))
            gauge_color = "#f59e0b"
            gauge_status = f"CALIBRATING ({total_sessions}/15 SESSIONS)"
            
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=stability_score,
            number={"suffix": " / 100", "font": {"size": 28, "color": gauge_num_color, "family": "JetBrains Mono"}},
            title={"text": f"<b>BIOMARKER STABILITY INDEX</b><br><span style='font-size:0.78em; color:{gauge_color}; font-weight:700;'>● {gauge_status}</span>", "font": {"size": 13, "color": chart_font_color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": chart_font_color, "tickwidth": 1, "ticklen": 4, "tickfont": {"color": chart_font_color}},
                "bar": {"color": gauge_color, "thickness": 0.22},
                "bgcolor": gauge_bg,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 65], "color": "rgba(244, 63, 94, 0.15)"},
                    {"range": [65, 85], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [85, 100], "color": "rgba(16, 185, 129, 0.15)"}
                ],
                "threshold": {"line": {"color": "#38bdf8", "width": 3}, "thickness": 0.8, "value": stability_score}
            }
        ))
        fig_gauge.update_layout(
            template=plotly_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=chart_font_color), height=190, margin=dict(l=20, r=20, t=35, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Normalize metrics to 0-100 scale for radar comparison
        def norm_val(val, min_v, max_v):
            if val is None:
                return 0
            return max(0.0, min(100.0, ((val - min_v) / (max_v - min_v + 1e-9)) * 100.0))
            
        categories = ["Voice Phonation", "Spiral Tremor", "Tap Arrhythmia", "Gait Asymmetry", "Tremor Power"]
        base_vals = [
            norm_val(res_voice["baseline_mean"], 0.0, 1.0),
            norm_val(res_spiral["baseline_mean"], 1.0, 1.5),
            norm_val(res_tap["baseline_mean"], 5.0, 35.0),
            norm_val(res_gait["baseline_mean"], 2.0, 30.0),
            norm_val(res_tremor["baseline_mean"], 0.01, 0.30)
        ]
        recent_vals = [
            norm_val(res_voice["recent_mean"], 0.0, 1.0),
            norm_val(res_spiral["recent_mean"], 1.0, 1.5),
            norm_val(res_tap["recent_mean"], 5.0, 35.0),
            norm_val(res_gait["recent_mean"], 2.0, 30.0),
            norm_val(res_tremor["recent_mean"], 0.01, 0.30)
        ]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=base_vals + [base_vals[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Personal Baseline',
            line=dict(color='#38bdf8', width=2),
            fillcolor='rgba(56, 189, 248, 0.22)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=recent_vals + [recent_vals[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Recent Sessions',
            line=dict(color='#f43f5e' if flagged_modalities else '#4ade80', width=2),
            fillcolor='rgba(244, 63, 94, 0.25)' if flagged_modalities else 'rgba(74, 222, 128, 0.2)'
        ))
        fig_radar.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=chart_font_color, size=11),
            polar=dict(
                bgcolor=chart_polar_bg,
                radialaxis=dict(visible=True, range=[0, 100], gridcolor=chart_grid_color, color=chart_font_color, tickfont=dict(size=9)),
                angularaxis=dict(gridcolor=chart_grid_color, color=chart_font_color, tickfont=dict(size=10))
            ),
            showlegend=True,
            legend=dict(bgcolor=chart_legend_bg, bordercolor=chart_legend_border, orientation="h", y=-0.15, x=0.1),
            margin=dict(l=35, r=35, t=15, b=35),
            height=290
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_chart2:
        st.markdown("""
        <div class="chart-panel-card">
            <div class="chart-panel-title">📈 Longitudinal Movement Progressions</div>
            <div class="chart-panel-desc">Spline trend progression across Finger-Tapping, Gait Asymmetry, and Spiral Tremor.</div>
        """, unsafe_allow_html=True)
        
        # Combine movement time points into clean multi-metric timeline
        trend_frames = []
        if not df_tap.empty:
            sub_t = df_tap[["ts", "rhythm_cv"]].copy()
            sub_t["Metric"] = "Tap Rhythm CV (%)"
            sub_t.rename(columns={"rhythm_cv": "Value"}, inplace=True)
            trend_frames.append(sub_t)
        if not df_gait.empty:
            sub_g = df_gait[["ts", "left_right_asymmetry"]].copy()
            sub_g["Metric"] = "Gait Asymmetry (%)"
            sub_g.rename(columns={"left_right_asymmetry": "Value"}, inplace=True)
            trend_frames.append(sub_g)
        if not df_spiral.empty:
            sub_s = df_spiral[["ts", "tremor"]].copy()
            sub_s["Metric"] = "Spiral Tremor Index"
            sub_s.rename(columns={"tremor": "Value"}, inplace=True)
            trend_frames.append(sub_s)
            
        if trend_frames:
            df_trends = pd.concat(trend_frames, ignore_index=True)
            fig_trends = px.line(
                df_trends, x="ts", y="Value", color="Metric",
                markers=True,
                title="",
                color_discrete_sequence=["#38bdf8", "#fbbf24", "#a855f7"]
            )
            fig_trends.update_traces(line_shape="spline", line=dict(width=2.5), marker=dict(size=7))
            fig_trends.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=chart_font_color),
                xaxis=dict(gridcolor=chart_grid_color, title="Session Timestamp"),
                yaxis=dict(gridcolor=chart_grid_color, title="Metric Value"),
                legend=dict(bgcolor=chart_legend_bg, bordercolor=chart_legend_border, orientation="h", y=1.12),
                margin=dict(l=20, r=20, t=10, b=20),
                height=240
            )
            st.plotly_chart(fig_trends, use_container_width=True)
        else:
            st.info("No movement sessions recorded yet. Click 'Populate 4-Session Demo Data' or run an assessment.")
            
        # Modality Shift Z-Score Profile Bar Chart
        st.markdown("<div style='margin-top:14px;'><strong style='font-size:0.95rem;'>🎯 Modality Shift Profile (Z-Score Relative to Baseline)</strong></div>", unsafe_allow_html=True)
        z_labels = ["Voice Risk", "Spiral Tremor", "Tap Arrhythmia", "Gait Asymmetry", "Tremor Power"]
        z_scores = [
            res_voice.get("z_score", 0.0),
            res_spiral.get("z_score", 0.0),
            res_tap.get("z_score", 0.0),
            res_gait.get("z_score", 0.0),
            res_tremor.get("z_score", 0.0)
        ]
        z_colors = ["#f43f5e" if abs(z) >= 1.75 else ("#fbbf24" if abs(z) >= 1.0 else "#10b981") for z in z_scores]
        
        df_z = pd.DataFrame({"Modality": z_labels, "Z-Score": z_scores, "Color": z_colors})
        fig_z = go.Figure()
        fig_z.add_trace(go.Bar(
            y=df_z["Modality"],
            x=df_z["Z-Score"],
            orientation='h',
            marker=dict(color=z_colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
            text=[f"{z:+.2f}" for z in z_scores],
            textposition="auto",
            textfont=dict(color="#ffffff", family="JetBrains Mono")
        ))
        # Alert threshold reference lines at z = +/- 1.75
        fig_z.add_vline(x=1.75, line_dash="dash", line_color="#f43f5e", line_width=1.5, annotation_text="+1.75 Review", annotation_position="top right", annotation_font_color="#f87171")
        fig_z.add_vline(x=-1.75, line_dash="dash", line_color="#f43f5e", line_width=1.5)
        fig_z.add_vrect(x0=-1.75, x1=1.75, fillcolor="rgba(16, 185, 129, 0.08)", line_width=0, layer="below")
        
        fig_z.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=chart_font_color),
            xaxis=dict(gridcolor=chart_grid_color, title="Standard Deviations Shift (Z-Score)", range=[-3.0, 3.0]),
            yaxis=dict(gridcolor=chart_grid_color, autorange="reversed"),
            margin=dict(l=20, r=20, t=25, b=20),
            height=200
        )
        st.plotly_chart(fig_z, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
            
    st.markdown("---")
    
    # Detailed Longitudinal Modality Breakdown
    st.subheader("📑 Modality Longitudinal History Tables")
    hist_tab1, hist_tab2, hist_tab3, hist_tab4, hist_tab5 = st.tabs([
        "🎤 Voice Sessions", "✍️ Spiral Sessions", "👆 Finger-Tap Sessions", "🚶 Gait Sessions", "〰️ Tremor Sessions"
    ])
    with hist_tab1:
        if not df_voice.empty:
            st.dataframe(df_voice[["ts", "prob"]].rename(columns={"ts": "Timestamp", "prob": "Parkinson's Score"}), use_container_width=True)
        else:
            st.write("No voice records.")
    with hist_tab2:
        if not df_spiral.empty:
            st.dataframe(df_spiral[["ts", "tremor", "variance"]].rename(columns={"ts": "Timestamp", "tremor": "Tremor Index", "variance": "Pressure Variance"}), use_container_width=True)
        else:
            st.write("No spiral records.")
    with hist_tab3:
        if not df_tap.empty:
            st.dataframe(df_tap[["ts", "tap_count", "mean_iti", "rhythm_cv", "amp_decrement", "tap_speed"]].rename(columns={
                "ts": "Timestamp", "tap_count": "Taps", "mean_iti": "Mean ITI (s)", "rhythm_cv": "Rhythm CV (%)", "amp_decrement": "Decrement (%)", "tap_speed": "Speed (Hz)"
            }), use_container_width=True)
        else:
            st.write("No finger-tapping records.")
    with hist_tab4:
        if not df_gait.empty:
            st.dataframe(df_gait[["ts", "cadence", "speed_proxy", "left_right_asymmetry", "arm_swing_ratio"]].rename(columns={
                "ts": "Timestamp", "cadence": "Cadence (SPM)", "speed_proxy": "Speed (m/s)", "left_right_asymmetry": "Step Asym (%)", "arm_swing_ratio": "Arm Swing Ratio"
            }), use_container_width=True)
        else:
            st.write("No gait records.")
    with hist_tab5:
        if not df_tremor.empty:
            st.dataframe(df_tremor[["ts", "peak_freq", "tremor_power", "rms_amplitude", "variability", "source_type"]].rename(columns={
                "ts": "Timestamp", "peak_freq": "Peak Freq (Hz)", "tremor_power": "PD Band Power", "rms_amplitude": "RMS Amp", "variability": "Variability", "source_type": "Source"
            }), use_container_width=True)
        else:
            st.write("No tremor records.")

    st.markdown("---")
    st.subheader("📄 Clinical Consultation Summary & Export")
    with st.expander("🩺 View Patient Longitudinal Dossier (Printable / Exportable Summary)", expanded=False):
        dossier_col1, dossier_col2 = st.columns([2, 1])
        with dossier_col1:
            def fmt_val(v, unit="", prec=2):
                return f"{v:.{prec}f}{unit}" if v is not None else "--"

            st.markdown(f"""
            #### Patient Monitoring Dossier &bull; `{user}`
            * **Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            * **Total Assessment Sessions:** {total_sessions}
            * **Longitudinal Review Status:** {"⚠️ Persistent Shift Flagged" if flagged_modalities else "✅ Consistent with Baseline"}
            
            | Modality Domain | Personal Baseline | Recent Measurement | Z-Score Shift | Status |
            | :--- | :--- | :--- | :--- | :--- |
            | **🎤 Voice Phonation** | `{fmt_val(res_voice['baseline_mean'])}` | `{fmt_val(res_voice['recent_mean'])}` | `{res_voice['z_score']:+.2f}` | **{res_voice['status']}** |
            | **✍️ Handwriting Spiral** | `{fmt_val(res_spiral['baseline_mean'])}` | `{fmt_val(res_spiral['recent_mean'])}` | `{res_spiral['z_score']:+.2f}` | **{res_spiral['status']}** |
            | **👆 Finger-Tapping Rhythm** | `{fmt_val(res_tap['baseline_mean'], '%', 1)}` | `{fmt_val(res_tap['recent_mean'], '%', 1)}` | `{res_tap['z_score']:+.2f}` | **{res_tap['status']}** |
            | **🚶 Gait Step Asymmetry** | `{fmt_val(res_gait['baseline_mean'], '%', 1)}` | `{fmt_val(res_gait['recent_mean'], '%', 1)}` | `{res_gait['z_score']:+.2f}` | **{res_gait['status']}** |
            | **〰️ Tremor Band Power** | `{fmt_val(res_tremor['baseline_mean'], '', 4)}` | `{fmt_val(res_tremor['recent_mean'], '', 4)}` | `{res_tremor['z_score']:+.2f}` | **{res_tremor['status']}** |
            """)
        with dossier_col2:
            st.markdown("#### Patient Data Export")
            export_payload = {
                "patient_id": user,
                "report_timestamp": datetime.now().isoformat(),
                "total_sessions": total_sessions,
                "summary": {
                    "voice": res_voice,
                    "spiral": res_spiral,
                    "finger_tapping": res_tap,
                    "gait": res_gait,
                    "tremor": res_tremor
                }
            }
            json_str = json.dumps(export_payload, indent=2, default=str)
            st.download_button(
                label="📥 Download Clinical JSON Export",
                data=json_str,
                file_name=f"parkneroo_{user}_report.json",
                mime="application/json",
                use_container_width=True
            )
            st.caption("🔒 Stored in local SQLite database (`sessions.db`). This file can be shared directly with your neurologist or care team.")

# ==============================================================================
# TAB 2: VOICE PATTERN ANALYSIS
# ==============================================================================
with tab_voice:
    st.header("🎤 Voice Pattern Analysis")
    st.markdown("""
    <p class='section-desc'>
    Analyzes acoustic perturbations in sustained phonation ('aaah' sound for 3-5s).
    Extracts fundamental frequency (F0), jitter, shimmer, and Harmonics-to-Noise Ratio (HNR).
    </p>
    """, unsafe_allow_html=True)
    
    if bundle is None:
        st.error("Model bundle 'outputs/model.joblib' not found. Run `python train.py` to train the voice model.")
    else:
        model, background = bundle["model"], bundle["background"]
        predict_fn = lambda d: model.predict_proba(pd.DataFrame(d, columns=FEATURES))[:, 1]
        
        up_voice = st.file_uploader("Upload sustained vowel audio (.wav)", type=["wav"], key="voice_uploader")
        if up_voice and st.button("Analyse & Save Voice Session", key="voice_analyze_btn", use_container_width=True):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(up_voice.read())
                temp_wav_path = f.name
            try:
                with st.spinner("Extracting Praat acoustic biomarkers and generating SHAP explanations..."):
                    v_feats = extract_features(temp_wav_path)
                    x_v = pd.DataFrame([v_feats])[FEATURES]
                    prob_v = float(predict_fn(x_v)[0])
                    
                    # Save to database
                    database.save_voice_session(user, prob_v, v_feats, con=con)
                    
                st.success("Voice pattern analysis complete and session stored!")
                
                vm1, vm2 = st.columns([1, 2])
                with vm1:
                    st.metric("Parkinson's-Associated Probability", f"{prob_v:.2f}")
                    st.caption("Derived from group cross-validated XGBoost/LogReg voice classifier.")
                with vm2:
                    st.write("**Top Contributing Acoustic Features (SHAP Waterfall)**")
                    expl = shap.Explainer(predict_fn, background)(x_v)
                    fig_shap, ax_shap = plt.subplots(figsize=(6, 3))
                    fig_shap.patch.set_facecolor('#0f172a')
                    ax_shap.set_facecolor('#1e293b')
                    shap.plots.waterfall(expl[0], max_display=6, show=False)
                    st.pyplot(fig_shap)
                    plt.close()
            except Exception as e:
                st.error(f"Voice analysis failed: {e}")
            finally:
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)
                    
        st.markdown("### 📈 Voice Longitudinal Monitoring")
        hist_v = database.get_modality_history(user, "voice", con)
        if hist_v.empty:
            st.info("No voice sessions recorded yet for this user.")
        else:
            fig_v = px.line(hist_v, x="ts", y="prob", markers=True, title="Voice Risk Probability Trend", color_discrete_sequence=["#38bdf8"])
            fig_v.add_hline(y=0.5, line_dash="dash", line_color="#f43f5e", annotation_text="Classification Threshold")
            fig_v.update_layout(
                template=plotly_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=chart_font_color), xaxis=dict(gridcolor=chart_grid_color, title="Timestamp"),
                yaxis=dict(gridcolor=chart_grid_color, title="Probability (0-1)", range=[0, 1])
            )
            st.plotly_chart(fig_v, use_container_width=True)

# ==============================================================================
# TAB 3: HANDWRITING & MOVEMENT (SPIRAL)
# ==============================================================================
with tab_handwriting:
    st.header("✍️ Handwriting Kinematics (Spiral Drawing)")
    st.markdown("""
    <p class='section-desc'>
    Extracts kinematic drawing biomarkers from spiral drawings using computer vision.
    Measures Tremor Index (arc length divergence), Thickness Variance (pressure hesitation proxy),
    and Drawing Extent.
    </p>
    """, unsafe_allow_html=True)
    
    col_up_img, col_prev_img = st.columns(2)
    up_spiral = col_up_img.file_uploader("Upload spiral drawing image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"], key="spiral_uploader")
    
    if up_spiral is not None:
        col_up_img.image(up_spiral, caption="Original Drawing Upload", use_container_width=True)
        if st.button("Extract & Save Kinematic Features", key="spiral_analyze_btn", use_container_width=True):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(up_spiral.read())
                temp_img_path = f.name
            try:
                with st.spinner("Running OpenCV contour extraction & kinematic biomarker profiling..."):
                    sp_feats, processed_img = vision_features.analyze_spiral(temp_img_path)
                    database.save_spiral_session(user, sp_feats["tremor_index"], sp_feats["thickness_variance"], sp_feats, con=con)
                    
                with col_prev_img:
                    st.image(processed_img, channels="BGR", caption="Processed Vision Overlay", use_container_width=True)
                    
                st.success("Handwriting biomarkers extracted and saved!")
                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("Tremor Index", f"{sp_feats['tremor_index']:.3f}", help="Ratio of raw perimeter to smoothed path.")
                sm2.metric("Thickness Variance", f"{sp_feats['thickness_variance']:.2f}", help="Proxy for micro-pauses and pressure hesitation.")
                sm3.metric("Drawing Length", f"{sp_feats['drawing_length']:.1f} px")
            except Exception as e:
                st.error(f"Handwriting analysis failed: {e}")
            finally:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                    
    st.markdown("### 📈 Handwriting Longitudinal Monitoring")
    hist_sp = database.get_modality_history(user, "spiral", con)
    if hist_sp.empty:
        st.info("No handwriting sessions recorded yet.")
    else:
        fig_sp = px.line(hist_sp, x="ts", y=["tremor", "variance"], markers=True, title="Handwriting Tremor & Pressure Variance Over Time",
                         color_discrete_sequence=["#38bdf8", "#fbbf24"])
        fig_sp.update_layout(
            template=plotly_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=chart_font_color), xaxis=dict(gridcolor=chart_grid_color, title="Timestamp"),
            yaxis=dict(gridcolor=chart_grid_color, title="Biomarker Value"),
            legend=dict(bgcolor=chart_legend_bg, bordercolor=chart_legend_border)
        )
        st.plotly_chart(fig_sp, use_container_width=True)

# ==============================================================================
# TAB 4: FINGER-TAPPING TEST
# ==============================================================================
with tab_finger:
    st.header("👆 Finger-Tapping Test")
    st.markdown("""
    <p class='section-desc'>
    <strong>Symptom-related changes:</strong> Slowness of movement (bradykinesia), reduced movement amplitude (hypometria / fatigue), and difficulty maintaining a regular rhythm (arrhythmokinesis).<br>
    <strong>User does:</strong> Repeatedly taps thumb and index finger in front of the camera with the widest possible excursion.<br>
    <strong>Your code measures:</strong> Number of taps, Time between taps (ITI), Tap rhythm variability (CV %), and Change in finger movement amplitude.
    </p>
    """, unsafe_allow_html=True)
    
    mode_ft = st.radio("Input Source", ["Upload Video (.mp4, .mov, .avi)", "Guided Movement Test Preset"], horizontal=True, key="ft_input_mode")
    
    feats_ft = None
    ts_ft = None
    keyframe_ft = None
    
    if mode_ft == "Upload Video (.mp4, .mov, .avi)":
        up_video = st.file_uploader("Upload video of finger-tapping", type=["mp4", "mov", "avi"], key="ft_video_uploader")
        if up_video and st.button("Analyse Finger Tapping Video", key="ft_analyze_vid_btn", use_container_width=True):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(up_video.read())
                temp_vid_path = f.name
            try:
                with st.spinner("Processing video frames with MediaPipe Hands & OpenCV..."):
                    feats_ft, ts_ft, keyframe_ft = finger_tap_analysis.analyze_finger_tapping_video(temp_vid_path)
                    database.save_finger_tap_session(
                        user, feats_ft["tap_count"], feats_ft["mean_iti"], feats_ft["rhythm_cv"],
                        feats_ft["amp_decrement"], feats_ft["tap_speed"], feats_ft, con=con
                    )
                st.success("Finger-tapping analysis complete and saved to profile!")
            except Exception as e:
                st.error(f"Finger-tapping analysis failed: {e}")
            finally:
                if os.path.exists(temp_vid_path):
                    os.remove(temp_vid_path)
    else:
        st.markdown("**Standardized Guided Movement Simulation:**")
        preset_choice = st.selectbox("Select Condition Preset", ["Parkinsonian Pattern (Bradykinesia, Arrhythmokinesis, Amplitude Decay)", "Healthy Baseline Pattern (Regular Rhythm, Preserved Excursion)"])
        cond_key = "parkinsonian" if "Parkinsonian" in preset_choice else "healthy"
        if st.button("Run Guided Assessment", key="ft_demo_btn", use_container_width=True):
            with st.spinner("Simulating physiological finger kinematics and tracking..."):
                feats_ft, ts_ft, keyframe_ft = finger_tap_analysis.simulate_finger_tapping_kinematics(cond_key)
                database.save_finger_tap_session(
                    user, feats_ft["tap_count"], feats_ft["mean_iti"], feats_ft["rhythm_cv"],
                    feats_ft["amp_decrement"], feats_ft["tap_speed"], feats_ft, con=con
                )
            st.success("Guided finger-tapping assessment completed and saved to profile!")
            
    if feats_ft is not None:
        # Display Key Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Number of Taps", feats_ft["tap_count"], help="Total valid tap closures completed.")
        m2.metric("Mean Time Between Taps", f"{feats_ft['mean_iti']:.3f} s", help="Inter-Tap Interval (ITI).")
        m3.metric("Tap Rhythm Variability", f"{feats_ft['rhythm_cv']:.1f}%", help="Coefficient of Variation (CV) of ITI.")
        m4.metric("Amplitude Decrement", f"{feats_ft['amp_decrement']:+.1f}%", help="Reduction in excursion from early to late taps (fatigue/hypometria).")
        m5.metric("Tapping Speed", f"{feats_ft['tap_speed']:.1f} Hz", help="Taps per second.")
        
        # Plotly Kinematic Aperture Chart
        col_plot1, col_plot2 = st.columns([2, 1])
        with col_plot1:
            st.write("**Finger Aperture Waveform Over Time**")
            df_wave = pd.DataFrame({
                "Time (s)": ts_ft["timestamps"],
                "Normalized Aperture": ts_ft["smoothed_dist"]
            })
            fig_ap = px.line(df_wave, x="Time (s)", y="Normalized Aperture", title="Thumb-Index Separation Distance",
                             color_discrete_sequence=["#38bdf8"])
            
            # Highlight tap closure points
            if ts_ft["tap_indices"]:
                t_taps = [ts_ft["timestamps"][idx] for idx in ts_ft["tap_indices"] if idx < len(ts_ft["timestamps"])]
                d_taps = [ts_ft["smoothed_dist"][idx] for idx in ts_ft["tap_indices"] if idx < len(ts_ft["smoothed_dist"])]
                fig_ap.add_trace(go.Scatter(
                    x=t_taps, y=d_taps, mode="markers",
                    name="Tap Contact", marker=dict(color="#f43f5e", size=8, symbol="triangle-down")
                ))
            fig_ap.update_layout(
                template=plotly_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=chart_font_color), xaxis=dict(gridcolor=chart_grid_color, title="Time (s)"),
                yaxis=dict(gridcolor=chart_grid_color, title="Aperture (Normalized)"),
                legend=dict(bgcolor=chart_legend_bg, bordercolor=chart_legend_border)
            )
            st.plotly_chart(fig_ap, use_container_width=True)
            
        with col_plot2:
            st.write("**Landmark Overlay Snapshot**")
            if keyframe_ft is not None:
                st.image(keyframe_ft, channels="BGR", use_container_width=True)

# ==============================================================================
# TAB 5: GAIT & WALKING ANALYSIS
# ==============================================================================
with tab_gait:
    st.header("🚶 Gait & Walking Analysis")
    st.markdown("""
    <p class='section-desc'>
    <strong>Symptom-related changes:</strong> Shorter steps (hypokinesia), slower walking, irregular timing, and altered/reduced arm swing.<br>
    <strong>User does:</strong> Records a short walking video in a suitable space.<br>
    <strong>Your code measures:</strong> Approximate step timing, Walking speed, Left–right movement differences, and Arm-swing and body-motion patterns.<br>
    <em>Video-based estimates depend on camera position, viewing angle, and recording conditions.</em>
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='disclaimer-card'>
        <strong>ℹ️ Recording Conditions Notice:</strong>
        Video-based estimates depend on camera position, subject distance, and lighting.
        All results are labeled as <strong>estimated movement features</strong>.
    </div>
    """, unsafe_allow_html=True)
    
    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        mode_gait = st.radio("Input Source", ["Upload Walking Video (.mp4, .mov, .avi)", "Guided Gait Assessment Preset"], horizontal=True, key="gait_input_mode")
    with col_g2:
        calib_height = st.number_input("User Height (m) for Calibration", min_value=1.20, max_value=2.20, value=1.70, step=0.05)
        
    feats_gt = None
    ts_gt = None
    keyframe_gt = None
    
    if mode_gait == "Upload Walking Video (.mp4, .mov, .avi)":
        up_gait = st.file_uploader("Upload walking video", type=["mp4", "mov", "avi"], key="gait_video_uploader")
        if up_gait and st.button("Analyse Gait Video", key="gait_analyze_btn", use_container_width=True):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(up_gait.read())
                temp_gait_path = f.name
            try:
                with st.spinner("Extracting body pose landmarks with MediaPipe Pose..."):
                    feats_gt, ts_gt, keyframe_gt = gait_analysis.analyze_gait_video(temp_gait_path, user_height_m=calib_height)
                    database.save_gait_session(
                        user, feats_gt["cadence_spm"], feats_gt["walking_speed_mps"],
                        feats_gt["left_right_asymmetry_pct"], feats_gt["arm_swing_ratio"], feats_gt, con=con
                    )
                st.success("Gait analysis complete and saved to profile!")
            except Exception as e:
                st.error(f"Gait analysis failed: {e}")
            finally:
                if os.path.exists(temp_gait_path):
                    os.remove(temp_gait_path)
    else:
        st.markdown("**Standardized Guided Movement Simulation:**")
        preset_gait = st.selectbox("Select Gait Pattern", [
            "Parkinsonian Pattern (Slower Cadence, Reduced Shuffled Steps, Asymmetric Arm Swing)",
            "Healthy Balanced Gait (Bilateral Symmetric Arm Swing, 112 SPM)"
        ])
        cond_gait = "parkinsonian" if "Parkinsonian" in preset_gait else "healthy"
        if st.button("Run Guided Gait Assessment", key="gait_demo_btn", use_container_width=True):
            with st.spinner("Simulating walking kinematic cycles..."):
                feats_gt, ts_gt, keyframe_gt = gait_analysis.simulate_gait_kinematics(cond_gait, user_height_m=calib_height)
                database.save_gait_session(
                    user, feats_gt["cadence_spm"], feats_gt["walking_speed_mps"],
                    feats_gt["left_right_asymmetry_pct"], feats_gt["arm_swing_ratio"], feats_gt, con=con
                )
            st.success("Guided gait assessment complete and saved to profile!")
            
    if feats_gt is not None:
        gm1, gm2, gm3, gm4 = st.columns(4)
        gm1.metric("Approx. Cadence", f"{feats_gt['cadence_spm']:.1f} SPM", help="Steps per minute.")
        gm2.metric("Walking Speed (Calibrated)", f"{feats_gt['walking_speed_mps']:.2f} m/s", help=f"Estimated based on {calib_height}m height.")
        gm3.metric("Left-Right Step Asymmetry", f"{feats_gt['left_right_asymmetry_pct']:.1f}%", help="Percentage difference between left and right step timing.")
        gm4.metric("Arm Swing Ratio (Min/Max)", f"{feats_gt['arm_swing_ratio']:.2f}", help="Bilateral arm swing ratio. Values < 0.70 indicate significant unilateral arm reduction.")
        
        col_gp1, col_gp2 = st.columns([2, 1])
        with col_gp1:
            st.write("**Bilateral Arm Swing Excursion Patterns**")
            df_arms = pd.DataFrame({
                "Time (s)": ts_gt["timestamps"],
                "Left Arm Excursion (m)": ts_gt["left_wrist_excursion"],
                "Right Arm Excursion (m)": ts_gt["right_wrist_excursion"]
            })
            fig_arms = px.line(df_arms, x="Time (s)", y=["Left Arm Excursion (m)", "Right Arm Excursion (m)"],
                               title="Left vs Right Arm Swing Waveform",
                               color_discrete_map={"Left Arm Excursion (m)": "#38bdf8", "Right Arm Excursion (m)": "#f43f5e"})
            fig_arms.update_layout(
                template=plotly_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=chart_font_color), xaxis=dict(gridcolor=chart_grid_color, title="Time (s)"),
                yaxis=dict(gridcolor=chart_grid_color, title="Excursion (m)"),
                legend=dict(bgcolor=chart_legend_bg, bordercolor=chart_legend_border)
            )
            st.plotly_chart(fig_arms, use_container_width=True)
        with col_gp2:
            st.write("**Pose Skeleton Snapshot**")
            if keyframe_gt is not None:
                st.image(keyframe_gt, channels="BGR", use_container_width=True)

# ==============================================================================
# TAB 6: TREMOR & MOVEMENT SIGNAL ANALYSIS
# ==============================================================================
with tab_tremor:
    st.header("〰️ Tremor & Movement Signal Analysis")
    st.markdown("""
    <p class='section-desc'>
    <strong>Symptom-related changes:</strong> Rhythmic shaking or other involuntary movement patterns (characteristic resting tremor: 4–6 Hz, postural tremor: 5–8 Hz).<br>
    <strong>User does:</strong> Performs a standardized hand-position task or completes a guided movement test.<br>
    <strong>Your code measures:</strong> Signal amplitude (RMS, PtP), Frequency components (FFT, Welch PSD, PD Band Power), Movement variability, and Changes across repeated sessions.<br>
    <em>A phone sensor or video does not capture every tremor reliably; label results as estimated movement features.</em>
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='disclaimer-card'>
        <strong>⚠️ Sensor Disclaimer:</strong>
        A phone sensor or video does not capture every tremor reliably; results are labeled as <strong>estimated movement features</strong>.
    </div>
    """, unsafe_allow_html=True)
    
    tremor_mode = st.radio("Signal Input Source", ["Guided Movement Test Preset", "Upload Accelerometer CSV (Phone Sensor)", "Upload Hand Video"], horizontal=True, key="tremor_source_mode")
    
    feats_tr = None
    psd_tr = None
    fig_mat_tr = None
    
    if tremor_mode == "Guided Movement Test Preset":
        preset_tr = st.selectbox("Select Movement Task Preset", [
            "Resting Tremor Pattern (4.8 Hz, Parkinsonian resting tremor band)",
            "Postural Tremor Pattern (6.2 Hz, Action/Postural tremor)",
            "Physiological Baseline (9.5 Hz, Normal benign physiological jitter)"
        ])
        key_tr = "resting_tremor" if "Resting" in preset_tr else ("postural_tremor" if "Postural" in preset_tr else "physiological")
        if st.button("Run Guided Tremor Test", key="run_tr_preset_btn", use_container_width=True):
            t_sim, sig_sim, df_sim = tremor_analysis.generate_synthetic_tremor_data(key_tr)
            feats_tr, psd_tr, fig_mat_tr = tremor_analysis.analyze_tremor_signal(t_sim, sig_sim, source_label="guided_sensor_task")
            database.save_tremor_session(
                user, feats_tr["dominant_freq_hz"], feats_tr["pd_band_power"],
                feats_tr["rms_amplitude"], feats_tr["spectral_entropy"], "guided_preset", feats_tr, con=con
            )
            st.success("Guided tremor test complete and saved to profile!")
            
    elif tremor_mode == "Upload Accelerometer CSV (Phone Sensor)":
        up_csv = st.file_uploader("Upload accelerometer CSV (time, ax, ay, az)", type=["csv"], key="tremor_csv_uploader")
        if up_csv and st.button("Analyse Accelerometer Signal", key="run_tr_csv_btn", use_container_width=True):
            try:
                t_csv, sig_csv = tremor_analysis.parse_accelerometer_csv(up_csv)
                feats_tr, psd_tr, fig_mat_tr = tremor_analysis.analyze_tremor_signal(t_csv, sig_csv, source_label="phone_accelerometer")
                database.save_tremor_session(
                    user, feats_tr["dominant_freq_hz"], feats_tr["pd_band_power"],
                    feats_tr["rms_amplitude"], feats_tr["spectral_entropy"], "phone_csv", feats_tr, con=con
                )
                st.success("Accelerometer signal processed and saved to profile!")
            except Exception as e:
                st.error(f"Accelerometer processing failed: {e}")
                
    else:
        up_tr_vid = st.file_uploader("Upload hand tremor video (.mp4, .mov)", type=["mp4", "mov"], key="tremor_video_uploader")
        if up_tr_vid and st.button("Analyse Video Tremor", key="run_tr_vid_btn", use_container_width=True):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(up_tr_vid.read())
                temp_tr_vid_path = f.name
            try:
                with st.spinner("Extracting sub-pixel finger oscillations with MediaPipe Hands..."):
                    feats_tr, psd_tr, fig_mat_tr, _ = tremor_analysis.analyze_tremor_video(temp_tr_vid_path)
                    database.save_tremor_session(
                        user, feats_tr["dominant_freq_hz"], feats_tr["pd_band_power"],
                        feats_tr["rms_amplitude"], feats_tr["spectral_entropy"], "video_landmarks", feats_tr, con=con
                    )
                st.success("Video tremor oscillations analyzed and saved!")
            except Exception as e:
                st.error(f"Video tremor analysis failed: {e}")
            finally:
                if os.path.exists(temp_tr_vid_path):
                    os.remove(temp_tr_vid_path)
                    
    if feats_tr is not None:
        tm1, tm2, tm3, tm4 = st.columns(4)
        tm1.metric("Dominant Frequency", f"{feats_tr['dominant_freq_hz']:.2f} Hz", help="Peak frequency in the 1.5 - 14.0 Hz spectrum.")
        tm2.metric("Signal RMS Amplitude", f"{feats_tr['rms_amplitude']:.4f}", help="Root Mean Square magnitude of tremor oscillation.")
        tm3.metric("PD Band Power Ratio", f"{feats_tr['pd_power_ratio_pct']:.1f}%", help="Energy percentage concentrated in the 3.5 - 7.5 Hz Parkinsonian band.")
        tm4.metric("Movement Regularity Index", f"{feats_tr['regularity_index']:.2f}", help="Rhythmicity of tremor oscillation (higher means more regular/rhythmic).")
        
        col_tp1, col_tp2 = st.columns(2)
        with col_tp1:
            st.write("**Time-Domain Filtered Tremor Waveform**")
            df_sig = pd.DataFrame({
                "Time (s)": psd_tr["time"],
                "Filtered Signal": psd_tr["filtered_signal"]
            })
            fig_sig = px.line(df_sig, x="Time (s)", y="Filtered Signal", title="Bandpass Filtered Movement (1.5-15 Hz)",
                              color_discrete_sequence=["#38bdf8"])
            fig_sig.update_layout(
                template=plotly_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=chart_font_color), xaxis=dict(gridcolor=chart_grid_color, title="Time (s)"),
                yaxis=dict(gridcolor=chart_grid_color, title="Amplitude")
            )
            st.plotly_chart(fig_sig, use_container_width=True)
            
        with col_tp2:
            st.write("**Power Spectral Density (PSD - Welch Method)**")
            df_psd = pd.DataFrame({
                "Frequency (Hz)": psd_tr["frequencies"],
                "Power Density": psd_tr["psd"]
            })
            # Filter to 0-16 Hz
            df_psd = df_psd[df_psd["Frequency (Hz)"] <= 16.0]
            fig_psd = px.line(df_psd, x="Frequency (Hz)", y="Power Density", title="Frequency Distribution & Tremor Power",
                              color_discrete_sequence=["#38bdf8"])
            fig_psd.add_vrect(x0=3.5, x1=7.5, fillcolor="#f43f5e", opacity=0.2, line_width=0, annotation_text="PD Band (3.5-7.5 Hz)")
            fig_psd.add_vrect(x0=8.0, x1=12.0, fillcolor="#60a5fa", opacity=0.15, line_width=0, annotation_text="Physiological Band")
            fig_psd.update_layout(
                template=plotly_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=chart_font_color), xaxis=dict(gridcolor=chart_grid_color, title="Frequency (Hz)"),
                yaxis=dict(gridcolor=chart_grid_color, title="Power Density")
            )
            st.plotly_chart(fig_psd, use_container_width=True)
            
        with st.expander("Clinical Report Diagnostic Plot (Matplotlib)"):
            if fig_mat_tr is not None:
                st.pyplot(fig_mat_tr)

st.markdown("---")
st.markdown("<center><small>PARKNEROO Digital Biomarker Research Platform &bull; Intended for Longitudinal Trend Monitoring &bull; Not a Diagnostic Tool</small></center>", unsafe_allow_html=True)