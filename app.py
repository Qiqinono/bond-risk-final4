# -*- coding: utf-8 -*-
"""
债券违约预测可视化前端（Streamlit · v4 Final）
====================================
本文件可直接作为 app.py 部署，用于展示债券违约预测结果、历史测试表现、
高风险债券排行、单券风险查询和数据运行状态。

推荐目录：
    app.py
    requirements.txt
    output_expanding/
        predictions_20251231.csv
"""

from __future__ import annotations

import glob
import html
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# 0. 页面基础配置
# =============================================================================

APP_TITLE = "债券违约预测"
APP_SUBTITLE = "BOND DEFAULT · PREDICTION"
MODEL_NAME = "Expanding Window XGBoost Hazard · Calibrated"
DEFAULT_OUTPUT_DIR = "output_expanding"

HORIZON_LABELS = {
    "y6m": "未来6个月",
    "y12m": "未来12个月",
    "y18m": "未来18个月",
    "y24m": "未来24个月",
}
HORIZON_ORDER = ["y6m", "y12m", "y18m", "y24m"]
HORIZON_SHORT = {"y6m": "未来6个月", "y12m": "未来12个月", "y18m": "未来18个月", "y24m": "未来24个月"}

RISK_BUCKET_ORDER = ["Top 1%", "1–5%", "5–10%", "10–20%", "Other"]

# st.set_page_config 必须是第一个 Streamlit 命令；要放在 @st.cache_data 等装饰器之前。
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)
BUCKET_BADGE_CLASS = {
    "Top 1%": "bucket-top1",
    "1–5%": "bucket-top5",
    "5–10%": "bucket-top10",
    "10–20%": "bucket-top20",
    "Other": "bucket-other",
}

# =============================================================================
# 1. 前三个 Fold 测试集指标
# =============================================================================

FOLD_TEST_METRICS: List[Dict[str, object]] = [
    # Fold1：测试集 20210101-20211231
    {"fold": "Fold1", "test_year": "20210101-20211231", "horizon": "6m", "n_samples": 9118, "n_positive": 192, "base_rate": 0.0211, "roc_auc": 0.9895, "pr_auc": 0.6918, "brier": 0.0425, "log_loss": 0.2231, "top1_precision": 0.7912, "top1_recall": 0.3750, "top5_precision": 0.3978, "top5_recall": 0.9427, "top10_precision": 0.2053, "top10_recall": 0.9740},
    {"fold": "Fold1", "test_year": "20210101-20211231", "horizon": "12m", "n_samples": 9118, "n_positive": 246, "base_rate": 0.0270, "roc_auc": 0.9647, "pr_auc": 0.7174, "brier": 0.0625, "log_loss": 0.2822, "top1_precision": 0.9121, "top1_recall": 0.3374, "top5_precision": 0.4681, "top5_recall": 0.8659, "top10_precision": 0.2492, "top10_recall": 0.9228},
    {"fold": "Fold1", "test_year": "20210101-20211231", "horizon": "18m", "n_samples": 9118, "n_positive": 296, "base_rate": 0.0325, "roc_auc": 0.9517, "pr_auc": 0.6842, "brier": 0.0670, "log_loss": 0.2939, "top1_precision": 0.9121, "top1_recall": 0.2804, "top5_precision": 0.5033, "top5_recall": 0.7736, "top10_precision": 0.2733, "top10_recall": 0.8412},
    {"fold": "Fold1", "test_year": "20210101-20211231", "horizon": "24m", "n_samples": 4312, "n_positive": 163, "base_rate": 0.0378, "roc_auc": 0.9423, "pr_auc": 0.6868, "brier": 0.0930, "log_loss": 0.3601, "top1_precision": 0.8837, "top1_recall": 0.2331, "top5_precision": 0.5674, "top5_recall": 0.7485, "top10_precision": 0.3202, "top10_recall": 0.8466},

    # Fold2：测试集 20220101-20221231
    {"fold": "Fold2", "test_year": "20220101-20221231", "horizon": "6m", "n_samples": 9118, "n_positive": 192, "base_rate": 0.0211, "roc_auc": 0.9900, "pr_auc": 0.7816, "brier": 0.0098, "log_loss": 0.0378, "top1_precision": 0.8681, "top1_recall": 0.4115, "top5_precision": 0.3890, "top5_recall": 0.9219, "top10_precision": 0.2064, "top10_recall": 0.9792},
    {"fold": "Fold2", "test_year": "20220101-20221231", "horizon": "12m", "n_samples": 9118, "n_positive": 246, "base_rate": 0.0270, "roc_auc": 0.9741, "pr_auc": 0.7472, "brier": 0.0122, "log_loss": 0.0606, "top1_precision": 0.9341, "top1_recall": 0.3455, "top5_precision": 0.4681, "top5_recall": 0.8659, "top10_precision": 0.2470, "top10_recall": 0.9146},
    {"fold": "Fold2", "test_year": "20220101-20221231", "horizon": "18m", "n_samples": 9118, "n_positive": 296, "base_rate": 0.0325, "roc_auc": 0.9790, "pr_auc": 0.7055, "brier": 0.0172, "log_loss": 0.0845, "top1_precision": 0.8901, "top1_recall": 0.2736, "top5_precision": 0.5231, "top5_recall": 0.8041, "top10_precision": 0.2986, "top10_recall": 0.9189},
    {"fold": "Fold2", "test_year": "20220101-20221231", "horizon": "24m", "n_samples": 4312, "n_positive": 163, "base_rate": 0.0378, "roc_auc": 0.9835, "pr_auc": 0.7222, "brier": 0.0294, "log_loss": 0.1578, "top1_precision": 0.8837, "top1_recall": 0.2331, "top5_precision": 0.5721, "top5_recall": 0.7546, "top10_precision": 0.3619, "top10_recall": 0.9571},

    # Fold3：测试集 20230101-20231231
    {"fold": "Fold3", "test_year": "20230101-20231231", "horizon": "6m", "n_samples": 14693, "n_positive": 338, "base_rate": 0.0230, "roc_auc": 0.9829, "pr_auc": 0.6901, "brier": 0.0854, "log_loss": 0.3428, "top1_precision": 0.7397, "top1_recall": 0.3195, "top5_precision": 0.4237, "top5_recall": 0.9201, "top10_precision": 0.2233, "top10_recall": 0.9704},
    {"fold": "Fold3", "test_year": "20230101-20231231", "horizon": "12m", "n_samples": 14693, "n_positive": 377, "base_rate": 0.0257, "roc_auc": 0.9848, "pr_auc": 0.7247, "brier": 0.0853, "log_loss": 0.3426, "top1_precision": 0.7808, "top1_recall": 0.3024, "top5_precision": 0.4755, "top5_recall": 0.9257, "top10_precision": 0.2498, "top10_recall": 0.9735},
    {"fold": "Fold3", "test_year": "20230101-20231231", "horizon": "18m", "n_samples": 14693, "n_positive": 397, "base_rate": 0.0270, "roc_auc": 0.9859, "pr_auc": 0.7561, "brier": 0.0851, "log_loss": 0.3421, "top1_precision": 0.8356, "top1_recall": 0.3073, "top5_precision": 0.5000, "top5_recall": 0.9244, "top10_precision": 0.2634, "top10_recall": 0.9748},
    {"fold": "Fold3", "test_year": "20230101-20231231", "horizon": "24m", "n_samples": 14693, "n_positive": 411, "base_rate": 0.0280, "roc_auc": 0.9866, "pr_auc": 0.7388, "brier": 0.0868, "log_loss": 0.3460, "top1_precision": 0.7808, "top1_recall": 0.2774, "top5_precision": 0.5204, "top5_recall": 0.9294, "top10_precision": 0.2737, "top10_recall": 0.9781},
]


# =============================================================================
# 2. CSS：暗色仪表盘风格 + 不使用按钮式导航
# =============================================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #080d18;
            --panel: #0f1727;
            --panel2: #121d31;
            --border: rgba(148, 163, 184, 0.16);
            --text: #e5edf7;
            --muted: #92a3b8;
            --accent: #23d3c3;
            --accent2: #7c3aed;
            --warning: #f59e0b;
            --danger: #ef4444;
        }
        .stApp { background: radial-gradient(circle at 12% 8%, #12213b 0, #080d18 28%, #060914 100%); color: var(--text); }
        header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; min-height: 0 !important; }
        div[data-testid="stToolbar"] { visibility: hidden !important; height: 0 !important; position: fixed !important; }
        #MainMenu { visibility: hidden !important; }
        footer { visibility: hidden !important; }
        div[data-testid="stDecoration"] { display: none !important; }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #08111f 0%, #0d1525 100%);
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span { color: var(--text) !important; }
        .block-container { padding-top: .35rem !important; max-width: 1440px; }
        h1, h2, h3 { letter-spacing: -0.02em; color: var(--text); }
        .hero {
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 22px 24px;
            background: linear-gradient(135deg, rgba(35, 211, 195, .10), rgba(124, 58, 237, .09) 52%, rgba(15, 23, 42, .85));
            box-shadow: 0 18px 55px rgba(0,0,0,.28);
            margin-bottom: 18px;
        }
        .eyebrow { color: var(--accent); font-size: 0.78rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
        .hero-title { color: var(--text); font-size: 2.05rem; font-weight: 800; line-height: 1.12; margin-top: 6px; }
        .hero-sub { color: var(--muted); font-size: .96rem; margin-top: 8px; max-width: 900px; }
        .card {
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px 18px;
            background: rgba(15, 23, 42, .82);
            box-shadow: 0 12px 32px rgba(0,0,0,.22);
        }
        .metric-card {
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 14px 16px;
            background: rgba(16, 26, 46, .86);
            min-height: 104px;
        }
        .metric-label { color: var(--muted); font-size: .78rem; margin-bottom: 8px; }
        .metric-value { color: var(--text); font-size: 1.75rem; font-weight: 800; font-variant-numeric: tabular-nums; }
        .metric-help { color: var(--muted); font-size: .76rem; margin-top: 6px; }
        .notice {
            border: 1px solid rgba(35, 211, 195, .22);
            background: rgba(35, 211, 195, .08);
            color: #c7fff8;
            padding: 11px 14px;
            border-radius: 14px;
            font-size: .86rem;
            margin: 10px 0 16px 0;
        }
        .small-muted { color: var(--muted); font-size: .82rem; }
        .badge {
            display: inline-block; padding: 3px 9px; border-radius: 999px;
            font-size: .76rem; font-weight: 700; margin: 2px 4px 2px 0;
            border: 1px solid rgba(255,255,255,.10);
        }
        .badge-green { background: rgba(35, 211, 195, .14); color: #55f0de; }
        .badge-purple { background: rgba(124, 58, 237, .16); color: #c4b5fd; }
        .bucket-top1 { background: rgba(239, 68, 68, .20); color: #fecaca; }
        .bucket-top5 { background: rgba(249, 115, 22, .20); color: #fed7aa; }
        .bucket-top10 { background: rgba(245, 158, 11, .18); color: #fde68a; }
        .bucket-top20 { background: rgba(34, 197, 94, .15); color: #bbf7d0; }
        .bucket-other { background: rgba(148, 163, 184, .12); color: #cbd5e1; }
        div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 14px; overflow: hidden; background: rgba(15,23,42,.82) !important; }
        div[data-testid="stDataFrame"] * { color: #dbeafe !important; }
        .dark-table-wrap { border:1px solid rgba(148,163,184,.16); border-radius:16px; overflow:auto; background:rgba(9,14,25,.92); box-shadow:0 10px 26px rgba(0,0,0,.20); margin: 8px 0 16px 0; }
        table.dark-table { width:100%; border-collapse:collapse; font-size:.82rem; color:#dbeafe; }
        table.dark-table thead th { position:sticky; top:0; z-index:2; background:linear-gradient(180deg, #172338, #111b2d); color:#7df5e8; text-align:left; font-weight:800; border-bottom:1px solid rgba(35,211,195,.22); padding:10px 12px; white-space:nowrap; }
        table.dark-table tbody td { padding:9px 12px; border-bottom:1px solid rgba(148,163,184,.10); color:#dbeafe; white-space:nowrap; }
        table.dark-table tbody tr:nth-child(even) { background:rgba(148,163,184,.045); }
        table.dark-table tbody tr:hover { background:rgba(35,211,195,.08); }
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] section { background:rgba(15,23,42,.86) !important; border:1px dashed rgba(35,211,195,.30) !important; border-radius:16px !important; }
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] section * { color:#cbd5e1 !important; }
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button { background:rgba(35,211,195,.10) !important; color:#e6fffb !important; border:1px solid rgba(35,211,195,.32) !important; border-radius:10px !important; }
        section[data-testid="stSidebar"] input { background:rgba(15,23,42,.86) !important; color:#e5edf7 !important; border:1px solid rgba(148,163,184,.20) !important; }
        section[data-testid="stSidebar"] div[data-baseweb="input"] { background:rgba(15,23,42,.86) !important; }
        div[data-baseweb="select"] > div { background:rgba(15,23,42,.86) !important; border-color:rgba(148,163,184,.20) !important; color:#e5edf7 !important; }
        div[data-baseweb="input"] input { color:#0f172a !important; -webkit-text-fill-color:#0f172a !important; background:#f8fafc !important; caret-color:#0f172a !important; }
        div[data-testid="stTextInput"] label, div[data-testid="stTextInput"] label p,
        div[data-testid="stNumberInput"] label, div[data-testid="stNumberInput"] label p,
        div[data-testid="stSelectbox"] label, div[data-testid="stSelectbox"] label p { color:#ffffff !important; font-weight:800 !important; opacity:1 !important; }
        div[data-testid="stTextInput"] input { color:#0f172a !important; -webkit-text-fill-color:#0f172a !important; background:#f8fafc !important; caret-color:#0f172a !important; }
        div[data-testid="stTextInput"] div[data-baseweb="input"] { background:#f8fafc !important; }
        div[data-testid="stTextInput"] input::placeholder { color:#475569 !important; opacity:1 !important; }
        section[data-testid="stSidebar"] div[data-baseweb="input"] input { color:#0f172a !important; -webkit-text-fill-color:#0f172a !important; background:#f8fafc !important; }
        div[data-testid="stTabs"] button { color: #cbd5e1; }
        div[data-testid="stTabs"] button[aria-selected="true"] { color: #55f0de; border-bottom-color: #23d3c3; }
        .stDownloadButton button, .stButton button {
            border-radius: 12px !important; border: 1px solid rgba(35, 211, 195, .35) !important;
            background: rgba(35, 211, 195, .09) !important; color: #dffdfa !important;
        }
        .cover-wrap {
            min-height: calc(100vh - 28px);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px 0 42px 0;
        }
        .cover-card {
            width: min(1120px, 96vw);
            border: 1px solid rgba(35, 211, 195, .22);
            border-radius: 30px;
            padding: 46px 48px;
            background:
                radial-gradient(circle at 82% 18%, rgba(124,58,237,.28), transparent 28%),
                radial-gradient(circle at 16% 20%, rgba(35,211,195,.20), transparent 24%),
                linear-gradient(135deg, rgba(9,16,31,.95), rgba(13,21,37,.88));
            box-shadow: 0 28px 90px rgba(0,0,0,.42);
        }
        .cover-title { font-size: clamp(2.2rem, 5vw, 4.1rem); line-height: 1.05; font-weight: 900; color: #f8fbff; margin: 12px 0; letter-spacing: -0.06em; }
        .cover-sub { color: #aab9cd; font-size: 1.02rem; max-width: 760px; line-height: 1.75; }
        .cover-grid { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: 30px; }
        .cover-feature { border:1px solid rgba(148,163,184,.16); border-radius:18px; padding:18px; background:rgba(15,23,42,.66); }
        .cover-feature .icon { font-size:1.7rem; margin-bottom:8px; }
        .cover-feature b { color:#e5edf7; }
        .cover-feature p { color:#93a4ba; font-size:.82rem; margin:6px 0 0 0; line-height:1.55; }
        .risk-card {
            border: 1px solid rgba(148,163,184,.16);
            border-radius: 18px;
            padding: 18px 18px;
            background: rgba(15,23,42,.82);
            min-height: 128px;
        }
        .risk-card-label { color: var(--muted); font-size: .78rem; margin-bottom: 8px; }
        .risk-card-value { color: var(--text); font-size: 1.75rem; font-weight: 850; }
        .risk-card-help { color: var(--muted); font-size: .76rem; margin-top: 8px; }

        .feature-grid { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 14px 0 20px 0; }
        .feature-card {
            border:1px solid rgba(148,163,184,.15); border-radius:18px; padding:16px 16px;
            background: linear-gradient(180deg, rgba(19,31,52,.86), rgba(12,19,34,.80));
            min-height: 132px; box-shadow: 0 10px 28px rgba(0,0,0,.18);
        }
        .feature-icon { width:38px; height:38px; display:flex; align-items:center; justify-content:center; border-radius:13px;
            background: rgba(35,211,195,.10); border:1px solid rgba(35,211,195,.22); font-size:1.18rem; margin-bottom:10px; }
        .feature-title { color:#f1f5f9; font-weight:800; font-size:.98rem; margin-bottom:5px; }
        .feature-body { color:#9fb0c6; font-size:.82rem; line-height:1.62; }
        .panel-title { display:flex; align-items:center; gap:8px; color:#eef6ff; font-size:1.05rem; font-weight:820; margin: 4px 0 12px 0; }
        .pill-row { display:flex; flex-wrap:wrap; gap:8px; margin: 10px 0 2px 0; }
        .soft-pill { border:1px solid rgba(148,163,184,.16); border-radius:999px; padding:6px 10px; color:#cbd5e1; background:rgba(15,23,42,.55); font-size:.78rem; }
        .summary-box {
            border: 1px solid rgba(35,211,195,.20); border-radius:18px; padding:16px 18px;
            background: linear-gradient(135deg, rgba(35,211,195,.08), rgba(124,58,237,.07));
            color:#dbeafe; line-height:1.75; margin: 14px 0 18px 0;
        }
        .summary-box b { color:#ffffff; }
        .glossary-grid { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }
        .glossary-card { border:1px solid rgba(148,163,184,.14); border-radius:16px; padding:14px; background:rgba(15,23,42,.70); }
        .glossary-card b { color:#55f0de; }
        .glossary-card p { color:#9fb0c6; font-size:.82rem; line-height:1.62; margin:6px 0 0 0; }
        .rank-band { height:8px; border-radius:999px; overflow:hidden; display:flex; margin-top:10px; border:1px solid rgba(255,255,255,.08); }
        .rank-band span:nth-child(1){background:#ef4444;width:1%}.rank-band span:nth-child(2){background:#f97316;width:4%}.rank-band span:nth-child(3){background:#f59e0b;width:5%}.rank-band span:nth-child(4){background:#22c55e;width:10%}.rank-band span:nth-child(5){background:#64748b;width:80%}

        /* Sidebar file path input: keep dark style, unlike main query input */
        section[data-testid="stSidebar"] div[data-testid="stTextInput"] div[data-baseweb="input"] {
            background: rgba(15,23,42,.92) !important;
            border: 1px solid rgba(35,211,195,.22) !important;
            border-radius: 12px !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stTextInput"] input {
            background: rgba(15,23,42,.92) !important;
            color: #e5edf7 !important;
            -webkit-text-fill-color: #e5edf7 !important;
            caret-color: #55f0de !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stTextInput"] input::placeholder {
            color: #94a3b8 !important;
            opacity: 1 !important;
        }
        /* Code/minimum-field box: match dark dashboard style */
        div[data-testid="stCodeBlock"] {
            border: 1px solid rgba(35,211,195,.22) !important;
            border-radius: 14px !important;
            overflow: hidden !important;
            background: rgba(9,14,25,.92) !important;
            box-shadow: 0 10px 26px rgba(0,0,0,.20) !important;
        }
        div[data-testid="stCodeBlock"] pre,
        div[data-testid="stCodeBlock"] code,
        div[data-testid="stCodeBlock"] span {
            background: rgba(9,14,25,.92) !important;
            color: #dbeafe !important;
            font-size: .88rem !important;
        }
        div[data-testid="stCodeBlock"] pre {
            border: none !important;
            padding: 14px 16px !important;
        }


        .field-code-box {
            border: 1px solid rgba(35,211,195,.24);
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(9,14,25,.96), rgba(15,23,42,.92));
            color: #dbeafe;
            padding: 15px 17px;
            margin: 8px 0 20px 0;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
            font-size: .92rem;
            line-height: 1.75;
            box-shadow: 0 10px 26px rgba(0,0,0,.20);
            white-space: pre-wrap;
            word-break: break-word;
        }
        .field-code-box .comment { color: #93a4ba; }

        @media (max-width: 900px) { .cover-grid, .feature-grid, .glossary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } .cover-card { padding: 32px 24px; } }

        /* 左侧导航：产品化菜单样式 */
        section[data-testid="stSidebar"] [role="radiogroup"] {
            gap: 10px;
            display: flex;
            flex-direction: column;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label {
            background: transparent !important;
            border: 1px solid transparent !important;
            border-radius: 12px !important;
            padding: 10px 12px !important;
            margin: 0 !important;
            transition: all .18s ease;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(35,211,195,.08) !important;
            border-color: rgba(35,211,195,.22) !important;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: rgba(59, 130, 246, .16) !important;
            border-left: 4px solid #23d3c3 !important;
            box-shadow: inset 0 0 0 1px rgba(35,211,195,.12);
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none !important; }
        section[data-testid="stSidebar"] [role="radiogroup"] p {
            font-size: 1.02rem !important;
            font-weight: 760 !important;
            color: #eaf2ff !important;
        }
        .sidebar-brand {
            display:flex; align-items:center; gap:12px;
            padding: 12px 4px 20px 4px;
            border-bottom: 1px solid rgba(148,163,184,.14);
            margin-bottom: 14px;
        }
        .sidebar-brand-icon {
            width:44px;height:44px;border-radius:12px;
            background:linear-gradient(135deg,#23d3c3,#3b82f6);
            display:flex;align-items:center;justify-content:center;
            font-size:22px;color:#06111f;font-weight:900;
        }
        .sidebar-brand-title { font-size:1.18rem;font-weight:900;color:#f8fbff;line-height:1.15; }
        .sidebar-brand-sub { color:#94a3b8;font-size:.78rem;margin-top:4px; }


        /* UI final polish: sidebar, inputs, multiselect tags, collapsed control */
        section[data-testid="stSidebar"] [role="radiogroup"] label {
            width: 100% !important;
            min-height: 50px !important;
            box-sizing: border-box !important;
            display: flex !important;
            align-items: center !important;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label p {
            width: 100% !important;
            white-space: nowrap !important;
            line-height: 1.2 !important;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            border-left: 4px solid #23d3c3 !important;
            background: linear-gradient(90deg, rgba(35,211,195,.18), rgba(59,130,246,.12)) !important;
            box-shadow: inset 0 0 0 1px rgba(35,211,195,.12), 0 8px 24px rgba(0,0,0,.16) !important;
        }
        div[data-testid="collapsedControl"], div[data-testid="stSidebarCollapsedControl"], button[data-testid="collapsedControl"] {
            visibility: visible !important;
            opacity: 1 !important;
            display: flex !important;
            z-index: 999999 !important;
            background: rgba(15,23,42,.92) !important;
            border: 1px solid rgba(35,211,195,.35) !important;
            border-radius: 12px !important;
            color: #e5edf7 !important;
            box-shadow: 0 10px 26px rgba(0,0,0,.28) !important;
        }
        div[data-testid="stTextInput"] div[data-baseweb="input"],
        div[data-testid="stNumberInput"] div[data-baseweb="input"] {
            background: rgba(15,23,42,.92) !important;
            border: 1px solid rgba(148,163,184,.24) !important;
            border-radius: 12px !important;
        }
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input {
            background: rgba(15,23,42,.92) !important;
            color: #e5edf7 !important;
            -webkit-text-fill-color: #e5edf7 !important;
            caret-color: #55f0de !important;
        }
        div[data-testid="stTextInput"] input::placeholder,
        div[data-testid="stNumberInput"] input::placeholder {
            color: #94a3b8 !important;
            opacity: 1 !important;
        }
        div[data-baseweb="tag"] {
            background: linear-gradient(135deg, rgba(35,211,195,.22), rgba(59,130,246,.18)) !important;
            border: 1px solid rgba(35,211,195,.32) !important;
            color: #e5edf7 !important;
        }
        div[data-baseweb="tag"] span { color: #e5edf7 !important; }
        div[data-baseweb="tag"] svg { color: #a7f3d0 !important; fill: #a7f3d0 !important; }
        div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li {
            background: #0f172a !important;
            color: #e5edf7 !important;
        }


        /* final color adjustment: muted tags and charts, keep risk band classic red/orange/yellow/green */
        [data-baseweb="tag"], span[data-baseweb="tag"], div[data-baseweb="tag"] {
            background: linear-gradient(135deg, rgba(33,52,76,.96), rgba(25,41,62,.96)) !important;
            border: 1px solid rgba(125,154,188,.35) !important;
            box-shadow: none !important;
            color: #dbeafe !important;
        }
        [data-baseweb="tag"] *, span[data-baseweb="tag"] *, div[data-baseweb="tag"] * {
            color: #dbeafe !important;
            -webkit-text-fill-color: #dbeafe !important;
            fill: #b6c7df !important;
        }
        [data-baseweb="tag"] svg, span[data-baseweb="tag"] svg, div[data-baseweb="tag"] svg {
            color: #b6c7df !important;
            fill: #b6c7df !important;
        }
        [data-baseweb="tag"] button, span[data-baseweb="tag"] button, div[data-baseweb="tag"] button {
            background: transparent !important;
        }
        .rank-band span:nth-child(1){background:#ef4444 !important;width:1%}
        .rank-band span:nth-child(2){background:#f97316 !important;width:4%}
        .rank-band span:nth-child(3){background:#f59e0b !important;width:5%}
        .rank-band span:nth-child(4){background:#22c55e !important;width:10%}
        .rank-band span:nth-child(5){background:#64748b !important;width:80%}
        .bucket-top1 { background: rgba(239, 68, 68, .20) !important; color: #fecaca !important; }
        .bucket-top5 { background: rgba(249, 115, 22, .20) !important; color: #fed7aa !important; }
        .bucket-top10 { background: rgba(245, 158, 11, .18) !important; color: #fde68a !important; }

        /* keep sidebar expand control visible after collapsing */
        header[data-testid="stHeader"] {
            display: block !important;
            background: transparent !important;
            pointer-events: none !important;
            height: 0 !important;
            min-height: 0 !important;
        }
        header[data-testid="stHeader"] * { pointer-events: auto !important; }
        div[data-testid="collapsedControl"],
        div[data-testid="stSidebarCollapsedControl"],
        button[data-testid="collapsedControl"],
        button[kind="header"],
        button[data-testid="stBaseButton-headerNoPadding"],
        button[data-testid="stBaseButton-header"] {
            visibility: visible !important;
            opacity: 1 !important;
            display: flex !important;
            z-index: 2147483647 !important;
            pointer-events: auto !important;
            background: rgba(15,23,42,.94) !important;
            border: 1px solid rgba(35,211,195,.34) !important;
            border-radius: 12px !important;
            color: #e5edf7 !important;
            box-shadow: 0 10px 26px rgba(0,0,0,.28) !important;
        }
        div[data-testid="collapsedControl"] svg,
        div[data-testid="stSidebarCollapsedControl"] svg,
        button[data-testid="collapsedControl"] svg,
        button[kind="header"] svg,
        button[data-testid="stBaseButton-headerNoPadding"] svg,
        button[data-testid="stBaseButton-header"] svg {
            color: #e5edf7 !important;
            stroke: #e5edf7 !important;
            fill: none !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# 3. 工具函数
# =============================================================================

def pct(x: Optional[float], digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x) * 100:.{digits}f}%"


def fmt_num(x: Optional[float], digits: int = 4) -> str:
    if x is None or pd.isna(x):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{float(x):.{digits}f}"


def risk_bucket_from_rank_pct(rank_pct: float) -> str:
    if rank_pct <= 0.01:
        return "Top 1%"
    if rank_pct <= 0.05:
        return "1–5%"
    if rank_pct <= 0.10:
        return "5–10%"
    if rank_pct <= 0.20:
        return "10–20%"
    return "Other"


def display_bucket(bucket: str) -> str:
    klass = BUCKET_BADGE_CLASS.get(bucket, "bucket-other")
    return f'<span class="badge {klass}">{bucket}</span>'


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_box(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="card" style="margin-top:10px; padding:14px 16px; background:rgba(20,30,48,.72);">
            <div style="font-size:.82rem; color:#55f0de; font-weight:700; margin-bottom:6px;">📌 {title}</div>
            <div style="font-size:.88rem; color:#d8e3f0; line-height:1.72;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dark_table(df: pd.DataFrame, max_height: int = 430) -> None:
    """用暗色 HTML 表格替代默认白色 dataframe，避免暗色界面中表格刺眼。"""
    if df is None or len(df) == 0:
        st.markdown("<div class='notice'>暂无数据。</div>", unsafe_allow_html=True)
        return
    safe_df = df.copy()
    safe_df = safe_df.astype(object).where(pd.notna(safe_df), "—")
    table_html = safe_df.to_html(index=False, escape=True, classes="dark-table", border=0)
    st.markdown(f"<div class='dark-table-wrap' style='max-height:{max_height}px'>{table_html}</div>", unsafe_allow_html=True)


def clean_code_value(x: object) -> str:
    """统一债券代码格式，减少 Excel 读入 .0、空格等造成的匹配失败。"""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.replace(" ", "").replace("	", "")


def feature_card(icon: str, title: str, body: str) -> str:
    return (
        f'<div class="feature-card">'
        f'<div class="feature-icon">{icon}</div>'
        f'<div class="feature-title">{title}</div>'
        f'<div class="feature-body">{body}</div>'
        f'</div>'
    )


def render_feature_grid(cards: List[Tuple[str, str, str]]) -> None:
    # 分列逐个渲染，避免部分浏览器/Streamlit 版本把多段 HTML 误识别为代码块。
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            st.markdown(feature_card(*card), unsafe_allow_html=True)


def probability_sentence(row: pd.Series) -> str:
    y6 = float(row.get("y6m_prob_display", np.nan))
    y12 = float(row.get("y12m_prob_display", np.nan))
    y24 = float(row.get("y24m_prob_display", np.nan))
    if any(pd.isna(v) for v in [y6, y12, y24]):
        return "该债券部分预测期限概率缺失，建议先核查预测文件字段是否完整接入。"
    near_ratio = y6 / max(y24, 1e-9)
    if y24 < 0.03:
        return "该债券整体违约概率较低，当前更适合作为常规监测对象。"
    if near_ratio >= 0.55:
        return "该债券风险更多集中在短期窗口，建议优先关注近期偿债压力、价格异动和负面舆情。"
    if y24 - y12 >= 0.08:
        return "该债券风险主要在中长期累积，短期压力相对有限，但需要持续跟踪基本面变化。"
    return "该债券风险随期限逐步上升，整体风险释放较为连续，适合纳入滚动观察名单。"


def rank_position_sentence(rank_pct: float, bucket: str) -> str:
    if bucket == "Top 1%":
        return "处于全样本最靠前的 1% 风险区间，是当前预警名单中的核心关注对象。"
    if bucket == "1–5%":
        return "处于全样本前 1%–5% 风险区间，建议作为重点跟踪债券。"
    if bucket == "5–10%":
        return "处于全样本前 5%–10% 风险区间，风险相对靠前，需要定期复核。"
    if bucket == "10–20%":
        return "处于全样本前 10%–20% 风险区间，属于次重点观察范围。"
    return "未进入前 20% 高风险区间，当前相对风险排序不靠前。"


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">{APP_SUBTITLE}</div>
          <div class="hero-title">{title}</div>
          <div class="hero-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbeafe"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=24, r=18, t=46, b=28),
    )
    fig.update_xaxes(gridcolor="rgba(148, 163, 184, .12)", zerolinecolor="rgba(148, 163, 184, .12)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, .12)", zerolinecolor="rgba(148, 163, 184, .12)")
    return fig


def safe_read_csv(path: str | Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)



def latest_file(patterns: Iterable[str]) -> Optional[str]:
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    if not files:
        return None
    files = sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _norm_col_key(x: object) -> str:
    """列名标准化：用于兼容 GitHub/Excel 保存后的中文列名、空格、括号、大小写。"""
    s = str(x).strip().replace("\ufeff", "")
    for ch in [" ", "　", "_", "-", "－", "—", "·", ".", "（", "）", "(", ")", "%", "％", "/", "\\"]:
        s = s.replace(ch, "")
    return s.lower()


PROB_ALIASES = {
    # 优先使用 probability_calibration.py 输出的校准后字段：y*_cal_prob
    "y6m": ["y6m_cal_prob", "y6m_prob_cal", "y6m_prob", "y6m_cum_prob", "y6m_prob_raw", "未来6月", "未来6个月", "未来6个月违约概率", "未来6个月累计违约概率", "未来6月违约概率", "6m概率", "6个月概率", "六个月违约概率", "六个月内违约概率"],
    "y12m": ["y12m_cal_prob", "y12m_prob_cal", "y12m_prob", "y12m_cum_prob", "y12m_prob_raw", "未来12月", "未来12个月", "未来12个月违约概率", "未来12个月累计违约概率", "未来1年", "未来一年", "12m概率", "12个月概率", "十二个月违约概率", "十二个月内违约概率"],
    "y18m": ["y18m_cal_prob", "y18m_prob_cal", "y18m_prob", "y18m_cum_prob", "y18m_prob_raw", "未来18月", "未来18个月", "未来18个月违约概率", "未来18个月累计违约概率", "18m概率", "18个月概率", "十八个月违约概率", "十八个月内违约概率"],
    "y24m": ["y24m_cal_prob", "y24m_prob_cal", "y24m_prob", "y24m_cum_prob", "y24m_prob_raw", "未来24月", "未来24个月", "未来24个月违约概率", "未来24个月累计违约概率", "未来2年", "未来两年", "24m概率", "24个月概率", "二十四个月违约概率", "二十四个月内违约概率"],
}

RAW_PROB_ALIASES = {
    "y6m": ["y6m_raw_prob", "y6m_prob_raw", "y6m_prob", "y6m_cum_prob", "未来6个月违约概率", "未来6个月累计违约概率"],
    "y12m": ["y12m_raw_prob", "y12m_prob_raw", "y12m_prob", "y12m_cum_prob", "未来12个月违约概率", "未来12个月累计违约概率"],
    "y18m": ["y18m_raw_prob", "y18m_prob_raw", "y18m_prob", "y18m_cum_prob", "未来18个月违约概率", "未来18个月累计违约概率"],
    "y24m": ["y24m_raw_prob", "y24m_prob_raw", "y24m_prob", "y24m_cum_prob", "未来24个月违约概率", "未来24个月累计违约概率"],
}


def _find_column_by_alias(columns: Iterable[object], aliases: List[str]) -> Optional[str]:
    norm_to_col = {_norm_col_key(c): c for c in columns}
    for a in aliases:
        key = _norm_col_key(a)
        if key in norm_to_col:
            return norm_to_col[key]
    return None


def _parse_probability_series(s: pd.Series) -> pd.Series:
    """兼容 0-1、小数、0-100、百分号字符串、破折号等格式。"""
    raw = s.astype(str).str.strip()
    raw = raw.replace({"": np.nan, "—": np.nan, "-": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})
    has_pct = raw.str.contains("%|％", na=False)
    cleaned = raw.str.replace("%", "", regex=False).str.replace("％", "", regex=False).str.replace(",", "", regex=False)
    out = pd.to_numeric(cleaned, errors="coerce")
    if has_pct.any():
        out.loc[has_pct] = out.loc[has_pct] / 100.0
    valid = out.dropna()
    if len(valid) and valid.max() > 1 and valid.max() <= 100:
        out = out / 100.0
    return out.where((out >= 0) & (out <= 1), np.nan)


def _count_probability_columns(df: pd.DataFrame) -> int:
    return sum(_find_column_by_alias(df.columns, PROB_ALIASES[h]) is not None for h in HORIZON_ORDER)


def choose_prediction_file(patterns: Iterable[str]) -> Optional[str]:
    """在候选预测文件中优先选择四个期限字段最完整的文件，避免云端误读旧文件/导出文件。"""
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=False))
    seen, ordered = set(), []
    for f in files:
        nf = str(Path(f))
        if nf not in seen:
            ordered.append(nf)
            seen.add(nf)
    if not ordered:
        return None
    scored = []
    for f in ordered:
        try:
            if f.lower().endswith(".csv"):
                head = safe_read_csv(f).head(5)
            else:
                head = read_any_table(f).head(5)
            score = _count_probability_columns(head)
        except Exception:
            score = -1
        exact_bonus = 10 if str(f).replace("\\", "/").endswith("output_expanding/pred_20250630.csv") else 0
        scored.append((score + exact_bonus, os.path.getmtime(f), f))
    scored.sort(reverse=True)
    return scored[0][2]

# =============================================================================
# 4. 数据读取与标准化
# =============================================================================

@st.cache_data(show_spinner=False)
def default_fold_metrics() -> pd.DataFrame:
    df = pd.DataFrame(FOLD_TEST_METRICS)
    df["horizon_order"] = df["horizon"].map({"6m": 1, "12m": 2, "18m": 3, "24m": 4})
    split_map = {
        "Fold1": ("20070101-20191231", "20200101-20201231", "20210101-20211231"),
        "Fold2": ("20070101-20201231", "20210101-20211231", "20220101-20221231"),
        "Fold3": ("20070101-20211231", "20220101-20221231", "20230101-20231231"),
    }
    df["训练集"] = df["fold"].map(lambda x: split_map.get(x, ("", "", ""))[0])
    df["验证集"] = df["fold"].map(lambda x: split_map.get(x, ("", "", ""))[1])
    df["测试集"] = df["fold"].map(lambda x: split_map.get(x, ("", "", ""))[2])
    return df.sort_values(["fold", "horizon_order"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def make_demo_predictions(n: int = 360, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes = [f"BOND{100000 + i}" for i in range(n)]
    issuers = [f"发行人{rng.integers(1, 80):03d}" for _ in range(n)]

    # 稀有高风险 + 大量低风险，更像违约预测分布
    base = np.clip(rng.beta(0.65, 18, n) + rng.choice([0, 0.05, 0.12], n, p=[0.88, 0.09, 0.03]), 0, 0.75)
    y6 = np.clip(base * rng.uniform(0.25, 0.55, n), 0, 1)
    y12 = np.clip(np.maximum(y6, base * rng.uniform(0.45, 0.78, n)), 0, 1)
    y18 = np.clip(np.maximum(y12, base * rng.uniform(0.70, 0.95, n)), 0, 1)
    y24 = np.clip(np.maximum(y18, base), 0, 1)

    df = pd.DataFrame(
        {
            "Liscd": codes,
            "BondCode": codes,
            "BondName": [f"示例债券{i:03d}" for i in range(n)],
            "Issuer": issuers,
            "PeriodEnd": "2025-12-31",
            "sem_str": "2025H2",
            "y6m_cum_prob": y6,
            "y12m_cum_prob": y12,
            "y18m_cum_prob": y18,
            "y24m_cum_prob": y24,
        }
    )
    return normalize_predictions(df, source="demo")


def read_any_table(path: str | Path) -> pd.DataFrame:
    path = str(path)
    if path.lower().endswith(".csv"):
        return safe_read_csv(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, engine="openpyxl")
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(f"不支持的文件格式：{path}")


def _standardize_bond_info_table(info: pd.DataFrame, source_path: str) -> pd.DataFrame:
    """把 BND_Bndinfo、feature_wide_table 或手工整理的债券信息表统一成前端字段。"""
    if info is None or info.empty:
        return pd.DataFrame()

    raw = info.copy()
    raw.columns = [str(c).strip() for c in raw.columns]

    col_alias = {
        "BondCode": [
            "BondCode", "bond_code", "债券代码", "证券代码", "交易代码", "代码", "Liscd", "InnerCode"
        ],
        "BondName": [
            "BondName", "bond_name", "债券名称", "债券简称", "证券简称", "简称", "Abbrnme", "Bndnme"
        ],
        "Issuer": [
            "Issuer", "issuer", "发行人", "发行主体", "发行公司", "公司名称", "主体名称",
            "发行机构全称", "发行人全称", "发行人名称", "Conme"
        ],
        "IssueDate": [
            "IssueDate", "issue_date", "发行日期", "发行时间", "发行起始日", "发行起始日期",
            "起息日期", "上市日期", "Listdt", "StartDate"
        ],
        "MaturityDate": [
            "MaturityDate", "maturity_date", "到期日期", "债券到期日", "到期日",
            "兑付日期", "摘牌日期", "EndDate", "Matudt"
        ],
        "BondType": [
            "BondType", "bond_type", "债券类型", "债券类别", "债券分类", "券种",
            "债券种类", "Bndtype"
        ],
        "CreditRating": [
            "CreditRating", "credit_rating", "债项评级", "债券评级", "主体评级",
            "信用评级", "债券信用评级", "Crdrate"
        ],
        "Term": ["Term", "term", "期限", "发行期限", "发行期限年", "债券期限", "Bndterm"],
        "CouponRate": ["CouponRate", "coupon_rate", "票面利率", "发行利率", "Intrrate"],
        "IssueAmount": ["IssueAmount", "issue_amount", "实际发行量", "发行规模", "发行规模亿", "Acisuquty"],
    }

    normalized_name = {str(c).strip().replace(" ", ""): c for c in raw.columns}
    lower_name = {str(c).strip().lower().replace(" ", ""): c for c in raw.columns}

    def find_col(aliases: List[str]) -> Optional[str]:
        for a in aliases:
            key = str(a).strip().replace(" ", "")
            if key in normalized_name:
                return normalized_name[key]
            key_low = key.lower()
            if key_low in lower_name:
                return lower_name[key_low]
        return None

    normalized = pd.DataFrame(index=raw.index)
    for std, aliases in col_alias.items():
        hit = find_col(aliases)
        if hit is not None:
            normalized[std] = raw[hit]

    if "BondCode" not in normalized.columns:
        return pd.DataFrame()

    normalized["_merge_code"] = normalized["BondCode"].map(clean_code_value)
    normalized = normalized[normalized["_merge_code"].str.fullmatch(r"\d+", na=False)].copy()
    if normalized.empty:
        return pd.DataFrame()

    def format_date_series(s: pd.Series) -> pd.Series:
        original = s.astype(str).replace({"nan": np.nan, "NaT": np.nan, "None": np.nan, "没有单位": np.nan})
        parsed = pd.to_datetime(original, errors="coerce")
        numeric = pd.to_numeric(s, errors="coerce")
        excel_mask = parsed.isna() & numeric.between(20000, 60000)
        if excel_mask.any():
            parsed2 = pd.to_datetime(numeric[excel_mask], unit="D", origin="1899-12-30", errors="coerce")
            parsed.loc[excel_mask] = parsed2
        out = parsed.dt.strftime("%Y-%m-%d")
        return out.where(parsed.notna(), original)

    if "IssueDate" in normalized.columns:
        normalized["IssueDate"] = format_date_series(normalized["IssueDate"])
    if "MaturityDate" in normalized.columns:
        normalized["MaturityDate"] = format_date_series(normalized["MaturityDate"])

    if ("MaturityDate" not in normalized.columns or normalized["MaturityDate"].isna().all()) and {"IssueDate", "Term"}.issubset(normalized.columns):
        issue_dt = pd.to_datetime(normalized["IssueDate"], errors="coerce")
        term_num = pd.to_numeric(normalized["Term"], errors="coerce")
        normalized["MaturityDate"] = (issue_dt + pd.to_timedelta((term_num * 365.25).round(), unit="D")).dt.strftime("%Y-%m-%d")

    # 对同一交易代码的重复记录进行优先级排序：优先选择交易所公司债记录，排除地方债、国债、资产支持证券等误匹配。
    text_fields = []
    for c in ["BondName", "BondType", "Issuer"]:
        if c in normalized.columns:
            text_fields.append(normalized[c].astype(str))
    combined_text = text_fields[0] if text_fields else pd.Series("", index=normalized.index)
    for s in text_fields[1:]:
        combined_text = combined_text + " " + s

    sctcd = raw["Sctcd"].astype(str).str.strip() if "Sctcd" in raw.columns else pd.Series("", index=raw.index)
    bndtype = normalized["BondType"].astype(str).str.strip() if "BondType" in normalized.columns else pd.Series("", index=normalized.index)

    company_like = combined_text.str.contains("公司债|企业债|可转换|可交换|中期票据|短期融资券", na=False)
    bad_like = combined_text.str.contains("地方债|地方政府|政府债|国债|财政部|资产支持|专项计划|ABS|优先A|优先级|次级", na=False)

    normalized["_priority"] = (
        sctcd.eq("2").astype(int) * 100
        + bndtype.eq("02").astype(int) * 60
        + company_like.astype(int) * 30
        - bad_like.astype(int) * 200
    )

    normalized["_info_source"] = source_path
    return normalized


@st.cache_data(show_spinner=False)
def load_bnd_info() -> Tuple[pd.DataFrame, Dict[str, object]]:
    """读取债券基本信息静态资源。优先读取轻量 CSV，避免 Excel 反复读取导致页面卡住。"""
    patterns = [
        # 推荐：由 BND_Bndinfo.xlsx 预处理得到的轻量表，加载最快
        "bnd_info_compact.csv", "data/bnd_info_compact.csv", "static/bnd_info_compact.csv", f"{DEFAULT_OUTPUT_DIR}/bnd_info_compact.csv",
        # 也兼容完整 BND_Bndinfo 表；若没有 compact csv，则自动读取它
        "BND_Bndinfo.xlsx", "BND_Bndinfo.xls", "bnd_info.xlsx", "bnd_info.xls", "bnd_info.csv",
        "data/BND_Bndinfo.xlsx", "static/BND_Bndinfo.xlsx", f"{DEFAULT_OUTPUT_DIR}/BND_Bndinfo.xlsx",
    ]

    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=False))

    # 去重并保留 patterns 的优先级；不再用 ** 全盘扫描，也不再同时读取多个 Excel。
    seen = set()
    ordered_files = []
    for f in files:
        norm = str(Path(f))
        if norm not in seen:
            ordered_files.append(norm)
            seen.add(norm)

    if not ordered_files:
        return pd.DataFrame(), {
            "found": False,
            "path": None,
            "message": "未发现债券基本信息表。建议将 bnd_info_compact.csv 或 BND_Bndinfo.xlsx 放在 app.py 同级目录。"
        }

    error_msgs = []
    for path in ordered_files:
        try:
            info = read_any_table(path)
            std = _standardize_bond_info_table(info, path)
            if std.empty:
                error_msgs.append(f"{path} 未识别到有效债券代码或信息列")
                continue

            keep_cols = ["_merge_code", "BondName", "Issuer", "IssueDate", "MaturityDate", "BondType", "CreditRating", "Term", "CouponRate", "IssueAmount", "_priority", "_info_source"]
            for c in keep_cols:
                if c not in std.columns:
                    std[c] = np.nan

            invalid_tokens = ["", "—", "-", "未知", "未知发行人", "Unknown", "unknown", "nan", "None", "NaN", "没有单位"]
            normalized_source = std[keep_cols].copy()
            for c in keep_cols:
                if c != "_merge_code":
                    normalized_source[c] = normalized_source[c].where(~normalized_source[c].astype(str).str.strip().isin(invalid_tokens), np.nan)

            normalized_source["_priority"] = pd.to_numeric(normalized_source.get("_priority", 0), errors="coerce").fillna(0)
            normalized_source = normalized_source.sort_values(["_merge_code", "_priority"], ascending=[True, False])
            normalized = normalized_source.groupby("_merge_code", as_index=False).first()
            return normalized, {
                "found": True,
                "path": path,
                "message": f"已读取债券基本信息静态资源：{path}，共 {len(normalized):,} 条。"
            }
        except Exception as exc:
            error_msgs.append(f"{path} 读取失败：{exc}")

    return pd.DataFrame(), {
        "found": False,
        "path": "; ".join(ordered_files[:3]),
        "message": "已发现候选表，但读取或字段识别失败：" + "；".join(error_msgs[:3])
    }


def attach_bnd_info(pred_df: pd.DataFrame, info_df: pd.DataFrame) -> pd.DataFrame:
    if info_df is None or info_df.empty:
        pred_df["bnd_info_matched"] = False
        return pred_df
    out = pred_df.copy()
    out["_merge_code"] = out["BondCode"].map(clean_code_value)
    add_cols = [c for c in ["_merge_code", "BondName", "Issuer", "IssueDate", "MaturityDate", "BondType", "CreditRating", "Term", "CouponRate", "IssueAmount"] if c in info_df.columns]
    merged = out.merge(info_df[add_cols], on="_merge_code", how="left", suffixes=("", "_info"))
    for col in ["BondName", "Issuer", "IssueDate", "MaturityDate", "BondType", "CreditRating", "Term", "CouponRate", "IssueAmount"]:
        info_col = f"{col}_info"
        if info_col in merged.columns:
            if col not in merged.columns:
                merged[col] = merged[info_col]
            else:
                current = merged[col].astype(str).str.strip()
                empty_tokens = ["", "—", "-", "未知", "未知发行人", "nan", "None", "NaN"]
                empty_mask = merged[col].isna() | current.isin(empty_tokens)
                # 预测文件里如果没有债券名称，前端会先用代码兜底；此时也应被基本信息表覆盖。
                if col == "BondName" and "BondCode" in merged.columns:
                    empty_mask = empty_mask | (current == merged["BondCode"].astype(str).str.strip())
                if col == "Issuer" and "BondCode" in merged.columns:
                    empty_mask = empty_mask | (current == merged["BondCode"].astype(str).str.strip())
                merged[col] = merged[col].astype(object)
                merged.loc[empty_mask, col] = merged.loc[empty_mask, info_col]
            merged = merged.drop(columns=[info_col])
    # 保证页面固定展示列一定存在，避免静态信息表缺少某些字段时报 KeyError。
    for col in ["BondName", "Issuer", "IssueDate", "MaturityDate", "BondType", "CreditRating", "Term", "CouponRate", "IssueAmount"]:
        if col not in merged.columns:
            merged[col] = np.nan
    matched_cols = [c for c in ["Issuer", "Term", "CreditRating"] if c in merged.columns]
    if matched_cols:
        valid_match = False
        for c in matched_cols:
            s = merged[c].astype(str).str.strip()
            valid_match = valid_match | (~merged[c].isna() & ~s.isin(["", "—", "-", "未知", "未知发行人", "nan", "None", "NaN"]))
        merged["bnd_info_matched"] = valid_match
    else:
        merged["bnd_info_matched"] = False
    merged = merged.drop(columns=["_merge_code"], errors="ignore")
    return merged


def read_prediction_file(uploaded_file=None, explicit_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """读取预测结果。优先级：上传文件 > 显式路径 > 校准后预测文件 > 备用预测文件 > demo。"""
    status: Dict[str, object] = {
        "mode": "demo",
        "source_path": None,
        "message": "未找到预测结果文件，已使用演示数据。",
    }

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        info_df, info_status = load_bnd_info()
        out = attach_bnd_info(normalize_predictions(df, source="uploaded"), info_df)
        status.update({"mode": "uploaded", "source_path": getattr(uploaded_file, "name", "uploaded.csv"), "message": "已读取侧边栏上传的预测 CSV。", "bnd_info": info_status})
        return out, status

    if explicit_path and Path(explicit_path).exists():
        path = str(explicit_path)
        if path.lower().endswith(".csv"):
            df = safe_read_csv(path)
        elif path.lower().endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            raise ValueError("仅支持 csv 或 parquet 文件。")
        info_df, info_status = load_bnd_info()
        out = attach_bnd_info(normalize_predictions(df, source=path), info_df)
        status.update({"mode": "file", "source_path": path, "message": f"已读取 {path}", "bnd_info": info_status})
        return out, status

    csv_path = choose_prediction_file(
        [
            f"{DEFAULT_OUTPUT_DIR}/pred_20250630.csv",       # 校准后预测结果，优先读取
            f"{DEFAULT_OUTPUT_DIR}/pred_*.csv",
            f"{DEFAULT_OUTPUT_DIR}/predictions_20250630.csv", # 备用预测结果
            f"{DEFAULT_OUTPUT_DIR}/predictions_*.csv",
            "pred_20250630.csv",
            "pred_*.csv",
            "predictions_20250630.csv",
            "predictions_*.csv",
        ]
    )
    if csv_path:
        df = safe_read_csv(csv_path)
        info_df, info_status = load_bnd_info()
        out = attach_bnd_info(normalize_predictions(df, source=csv_path), info_df)
        status.update({"mode": "file", "source_path": csv_path, "message": f"已读取 {csv_path}", "bnd_info": info_status})
        return out, status

    parquet_path = latest_file(
        [
            "inference_predictions_calibrated.parquet",
            "production_predictions_long.parquet",
        ]
    )
    if parquet_path:
        try:
            df = pd.read_parquet(parquet_path)
            info_df, info_status = load_bnd_info()
            out = attach_bnd_info(normalize_predictions(df, source=parquet_path), info_df)
            status.update({"mode": "file", "source_path": parquet_path, "message": f"已读取 {parquet_path}", "bnd_info": info_status})
            return out, status
        except Exception as exc:  # pragma: no cover - 仅部署环境中使用
            status["message"] = f"发现 parquet 但读取失败：{exc}，已使用演示数据。"

    info_df, info_status = load_bnd_info()
    out = attach_bnd_info(make_demo_predictions(), info_df)
    status["bnd_info"] = info_status
    return out, status


def normalize_predictions(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """把建模脚本输出、生产 parquet 或手工 CSV 统一成前端需要的列。"""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    # ID / 名称字段兼容
    if "Liscd" not in out.columns:
        for alt in ["BondCode", "bond_code", "code", "债券代码"]:
            if alt in out.columns:
                out["Liscd"] = out[alt]
                break
    if "Liscd" not in out.columns:
        out["Liscd"] = [f"ROW{i:06d}" for i in range(len(out))]

    if "BondCode" not in out.columns:
        out["BondCode"] = out["Liscd"].astype(str)
    if "BondName" not in out.columns:
        out["BondName"] = out.get("债券名称", out["BondCode"].astype(str))
    if "Issuer" not in out.columns:
        out["Issuer"] = out.get("发行人", "—")
    for col in ["IssueDate", "MaturityDate", "BondType", "CreditRating", "Term", "CouponRate", "IssueAmount"]:
        if col not in out.columns:
            out[col] = np.nan
    if "PeriodEnd" not in out.columns:
        out["PeriodEnd"] = out.get("inference_date", out.get("sem_str", "未知"))

    # 概率字段兼容：优先使用生产版 raw/cal，否则用 expanding_window 的 *_cum_prob；也兼容下载后的中文列名。
    prob_source_cols: Dict[str, Optional[str]] = {}
    for h in HORIZON_ORDER:
        cal_col = _find_column_by_alias(out.columns, PROB_ALIASES[h])
        raw_col = _find_column_by_alias(out.columns, RAW_PROB_ALIASES[h]) or cal_col
        prob_source_cols[h] = cal_col
        if cal_col is None:
            out[f"{h}_prob_display"] = np.nan
        else:
            out[f"{h}_prob_display"] = _parse_probability_series(out[cal_col])
        # 页面排序与风险分层采用展示口径；在校准文件存在时即为 y*_cal_prob，避免 raw 概率过高导致 Top 列表失真。
        if raw_col is None:
            out[f"{h}_prob_rank"] = out[f"{h}_prob_display"]
        else:
            raw_parsed = _parse_probability_series(out[raw_col])
            out[f"{h}_prob_rank"] = out[f"{h}_prob_display"].where(out[f"{h}_prob_display"].notna(), raw_parsed)

    # 缺失不强行填 0，避免出现“前期全 0、24 月极高”的误导展示。
    for h in HORIZON_ORDER:
        out[f"{h}_prob_missing"] = out[f"{h}_prob_display"].isna()
    out.attrs["prob_source_cols"] = prob_source_cols

    # 用于排序的 24m 若缺失，则退而使用可用的最长预测期限；显示仍保留缺失。
    available_rank_cols = [f"{h}_prob_rank" for h in HORIZON_ORDER if out[f"{h}_prob_rank"].notna().any()]
    if not out["y24m_prob_rank"].notna().any() and available_rank_cols:
        out["y24m_prob_rank"] = out[available_rank_cols].bfill(axis=1).iloc[:, -1]
    for h in HORIZON_ORDER:
        out[f"{h}_prob_rank"] = out[f"{h}_prob_rank"].fillna(out[f"{h}_prob_display"])

    # 排名和风险桶：默认按 y24m raw/rank 口径。
    n = max(len(out), 1)
    rank_base = out["y24m_prob_rank"].fillna(-1)
    out["report_rank"] = rank_base.rank(method="first", ascending=False).astype(int)
    out["report_rank_pct"] = out["report_rank"] / n
    if "risk_bucket_by_y24m" not in out.columns:
        out["risk_bucket_by_y24m"] = out["report_rank_pct"].apply(risk_bucket_from_rank_pct)
    else:
        out["risk_bucket_by_y24m"] = out["risk_bucket_by_y24m"].fillna(out["report_rank_pct"].apply(risk_bucket_from_rank_pct))

    # 最可能违约时段。若建模脚本已经保存 most_likely_period，则直接使用；否则用各期差分粗略推断。
    prob_matrix = out[[f"{h}_prob_display" for h in HORIZON_ORDER]].to_numpy(dtype=float)
    missing_any = np.isnan(prob_matrix).any(axis=1)
    nonmono = np.zeros(len(out), dtype=bool)
    with np.errstate(invalid="ignore"):
        nonmono = np.nanmin(np.diff(prob_matrix, axis=1), axis=1) < -1e-8
    suspicious_jump = (
        (np.nan_to_num(out["y6m_prob_display"], nan=0) <= 1e-8)
        & (np.nan_to_num(out["y12m_prob_display"], nan=0) <= 1e-8)
        & (np.nan_to_num(out["y18m_prob_display"], nan=0) <= 1e-8)
        & (np.nan_to_num(out["y24m_prob_display"], nan=0) >= 0.50)
    )
    out["prob_quality_flag"] = np.select(
        [missing_any, nonmono, suspicious_jump],
        ["缺少部分期限概率", "期限结构非单调", "24个月异常跳升"],
        default="正常",
    )

    if "most_likely_period" not in out.columns:
        y6 = out["y6m_prob_display"].to_numpy(dtype=float)
        y12 = out["y12m_prob_display"].to_numpy(dtype=float)
        y18 = out["y18m_prob_display"].to_numpy(dtype=float)
        y24 = out["y24m_prob_display"].to_numpy(dtype=float)
        intervals = np.vstack([
            y6,
            np.clip(y12 - y6, 0, 1),
            np.clip(y18 - y12, 0, 1),
            np.clip(y24 - y18, 0, 1),
        ]).T
        labels = np.array(["0–6月", "6–12月", "12–18月", "18–24月"])
        safe_intervals = np.nan_to_num(intervals, nan=-1)
        out["most_likely_period"] = np.where(missing_any, "缺少期限概率", labels[np.argmax(safe_intervals, axis=1)])
        out["max_period_prob"] = np.where(missing_any, np.nan, np.nanmax(intervals, axis=1))
    else:
        out["max_period_prob"] = pd.to_numeric(out.get("max_period_prob", np.nan), errors="coerce")

    out["source"] = source
    return out.sort_values("report_rank").reset_index(drop=True)

# =============================================================================
# 5. 页面组件
# =============================================================================

def sidebar_controls() -> Tuple[str, Optional[object], Optional[str]]:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
              <div class="sidebar-brand-icon">♢</div>
              <div>
                <div class="sidebar-brand-title">债券违约预测</div>
                <div class="sidebar-brand-sub">投资者风险监测</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_labels = {
            "首页总览": "▦　首页总览",
            "验证结果": "↺　验证结果",
            "高风险预测": "⚠　高风险预测",
            "单券查询": "⌕　单券查询",
            "模型说明": "⌘　模型说明",
            "数据状态": "◎　数据状态",
        }
        page = st.radio(
            "导航",
            list(nav_labels.keys()),
            index=0,
            format_func=lambda x: nav_labels[x],
            label_visibility="collapsed",
        )

        st.markdown("---")
        uploaded_file = st.file_uploader("上传预测 CSV（可选）", type=["csv"])
        explicit_path = st.text_input("预测文件路径（可选）", value="", placeholder="output_expanding/pred_20250630.csv")
        return page, uploaded_file, explicit_path.strip() or None


def render_home(pred_df: pd.DataFrame, metrics_df: pd.DataFrame, status: Dict[str, object]) -> None:
    section_header(
        "首页总览",
        "汇总当前预测结果、风险分层结构和模型历史测试表现，帮助快速把握债券池整体风险。",
    )
    st.markdown(
        f"""
        <div class="notice">
        当前数据源：{status.get('message', '')} &nbsp; <span class="badge badge-green">{MODEL_NAME}</span>
        <span class="badge badge-purple">按校准后未来24个月风险排序</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_feature_grid([
        ("📊", "整体风险看板", "汇总预测债券数量、高风险比例和最高风险水平，适合快速浏览债券池状态。"),
        ("✅", "历史测试表现", "展示前三个 Fold 测试窗口表现，观察模型区分能力和稳定性。"),
        ("⚠️", "高风险预警", "按未来24个月违约概率排序，定位 Top 1%、1–5% 等重点债券。"),
        ("🔎", "单券查询", "支持按债券代码、名称或发行人查询，输出单券风险摘要和期限结构。"),
    ])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("预测债券数", f"{len(pred_df):,}", "当前预测主表行数")
    with c2:
        top_count = int((pred_df["risk_bucket_by_y24m"] == "Top 1%").sum())
        metric_card("Top 1% 数量", f"{top_count:,}", "按未来24个月风险排名")
    with c3:
        metric_card("最高未来24个月风险", pct(pred_df["y24m_prob_display"].max()), "当前样本最高预测值")
    with c4:
        metric_card("Fold 测试记录", f"{len(metrics_df):,}", "历史测试窗口指标行数")

    left, right = st.columns([1.05, 1])
    with left:
        bucket_counts = (
            pred_df["risk_bucket_by_y24m"]
            .value_counts()
            .reindex(RISK_BUCKET_ORDER)
            .fillna(0)
            .reset_index()
        )
        bucket_counts.columns = ["风险分层", "债券数"]
        fig = px.bar(bucket_counts, x="风险分层", y="债券数", title="风险分层分布（按未来24个月风险排序）")
        fig = plot_layout(fig, height=360)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        insight_box("图表解读", "该图展示不同风险层级的债券数量。Top 1% 和 1–5% 是最需要优先关注的债券池；10–20% 可作为次级观察名单，Other 表示当前排序相对靠后。")

    with right:
        summary = metrics_df.groupby("horizon", as_index=False).agg(
            pr_auc_mean=("pr_auc", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            top5_recall_mean=("top5_recall", "mean"),
        )
        summary["horizon_order"] = summary["horizon"].map({"6m": 1, "12m": 2, "18m": 3, "24m": 4})
        summary["horizon_cn"] = summary["horizon"].map({"6m": "未来6个月", "12m": "未来12个月", "18m": "未来18个月", "24m": "未来24个月"})
        summary = summary.sort_values("horizon_order")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=summary["horizon_cn"], y=summary["pr_auc_mean"], mode="lines+markers", name="平均 PR-AUC"))
        fig.add_trace(go.Scatter(x=summary["horizon_cn"], y=summary["roc_auc_mean"], mode="lines+markers", name="平均 ROC-AUC"))
        fig.add_trace(go.Scatter(x=summary["horizon_cn"], y=summary["top5_recall_mean"], mode="lines+markers", name="平均 Top5%召回率"))
        fig.update_layout(title="Fold 测试集表现")
        fig = plot_layout(fig, height=360)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        insight_box("图表解读", "ROC-AUC 反映整体区分能力；PR-AUC 更适合违约样本稀缺的场景；Top5%召回率表示模型把真实高风险样本筛入前 5% 名单的能力。")

    st.markdown(
        f"""
        <div class="summary-box">
          <b>当前债券池解读：</b>系统默认使用未来24个月风险排序进行分层，当前 Top 1% 债券共 <b>{top_count:,}</b> 只，
          最高未来24个月预测概率为 <b>{pct(pred_df['y24m_prob_display'].max())}</b>。建议先查看“高风险预测”页面形成预警名单，再进入“单券查询”查看具体债券的风险释放窗口。
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("高风险债券 TOP 10")
    show_cols = ["report_rank", "BondCode", "BondName", "Issuer", "risk_bucket_by_y24m", "y6m_prob_display", "y12m_prob_display", "y18m_prob_display", "y24m_prob_display", "most_likely_period", "prob_quality_flag"]
    view = pred_df.reindex(columns=show_cols).head(10).copy()
    view = format_prediction_table(view)
    dark_table(view, max_height=420)


def render_validation(metrics_df: pd.DataFrame) -> None:
    section_header(
        "验证结果：前三个 Fold 测试集表现",
        "展示模型在三个 Fold 测试集上的区分能力、Top-k 捕获能力和误差水平。",
    )
    st.markdown(
        "<div class='notice'>图表中的指标包括 ROC-AUC、PR-AUC、Brier、Log Loss 与 Top-k Precision / Recall，用于展示各 Fold 测试集表现。</div>",
        unsafe_allow_html=True,
    )

    fold_schedule = pd.DataFrame([
        {"Fold": "Fold1", "训练集": "20070101-20191231", "验证集": "20200101-20201231", "测试集": "20210101-20211231"},
        {"Fold": "Fold2", "训练集": "20070101-20201231", "验证集": "20210101-20211231", "测试集": "20220101-20221231"},
        {"Fold": "Fold3", "训练集": "20070101-20211231", "验证集": "20220101-20221231", "测试集": "20230101-20231231"},
    ])
    dark_table(fold_schedule, max_height=180)

    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("平均 PR-AUC", fmt_num(metrics_df["pr_auc"].mean()), "越高越好，越能识别少数违约样本")
        with c2:
            metric_card("平均 ROC-AUC", fmt_num(metrics_df["roc_auc"].mean()), "越高越好，表示区分能力更强")
        with c3:
            metric_card("平均 Top5%召回率", pct(metrics_df["top5_recall"].mean()), "越高说明高风险债券捕捉越充分")
        with c4:
            metric_card("平均 Brier", fmt_num(metrics_df["brier"].mean()), "越低越好，表示概率误差更小")

    horizon_map = {"未来6个月": "6m", "未来12个月": "12m", "未来18个月": "18m", "未来24个月": "24m"}
    selected_horizon_cn = st.multiselect(
        "选择预测期限",
        options=list(horizon_map.keys()),
        default=list(horizon_map.keys()),
    )
    selected_horizons = [horizon_map[x] for x in selected_horizon_cn]
    selected_folds = st.multiselect(
        "选择 Fold",
        options=["Fold1", "Fold2", "Fold3"],
        default=["Fold1", "Fold2", "Fold3"],
    )
    df = metrics_df[metrics_df["horizon"].isin(selected_horizons) & metrics_df["fold"].isin(selected_folds)].copy()
    df["期限"] = df["horizon"].map({"6m": "未来6个月", "12m": "未来12个月", "18m": "未来18个月", "24m": "未来24个月"})

    tab1, tab2, tab3, tab4 = st.tabs(["区分能力", "Top-k 捕获", "稳定性", "明细表"])
    with tab1:
        long = df.melt(id_vars=["fold", "期限"], value_vars=["roc_auc", "pr_auc"], var_name="指标", value_name="数值")
        long["指标"] = long["指标"].replace({"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"})
        fig = px.bar(long, x="期限", y="数值", color="fold", barmode="group", facet_col="指标", title="各 Fold × 预测期限的区分能力", color_discrete_map={"Fold1": "#3E6F73", "Fold2": "#526D89", "Fold3": "#7A669E"})
        fig.update_yaxes(range=[0, 1])
        fig = plot_layout(fig, height=420)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        insight_box("图表解读", "ROC-AUC 用于观察模型对违约与非违约债券的整体区分能力；PR-AUC 更适合违约预测这类样本不平衡场景。若两项指标在不同 Fold 中波动不大，说明模型具有较好的时间稳定性。")

    with tab2:
        metric_name_map = {
            "top1_precision": "Top1% 精确率",
            "top1_recall": "Top1% 召回率",
            "top5_precision": "Top5% 精确率",
            "top5_recall": "Top5% 召回率",
            "top10_precision": "Top10% 精确率",
            "top10_recall": "Top10% 召回率",
        }
        top_metric = st.selectbox("Top-k 指标", list(metric_name_map.keys()), format_func=lambda x: metric_name_map[x])
        fig = px.bar(df, x="期限", y=top_metric, color="fold", barmode="group", title=f"{metric_name_map[top_metric]}：不同 Fold 对比", color_discrete_map={"Fold1": "#3E6F73", "Fold2": "#526D89", "Fold3": "#7A669E"})
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        fig = plot_layout(fig, height=400)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(df, x="期限", y="top5_precision", color="fold", markers=True, title="Top5% 精确率", color_discrete_map={"Fold1": "#3E6F73", "Fold2": "#526D89", "Fold3": "#7A669E"})
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            fig = plot_layout(fig, height=330)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with c2:
            fig = px.line(df, x="期限", y="top5_recall", color="fold", markers=True, title="Top5% 召回率", color_discrete_map={"Fold1": "#3E6F73", "Fold2": "#526D89", "Fold3": "#7A669E"})
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            fig = plot_layout(fig, height=330)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        insight_box("图表解读", "Top-k 指标反映模型把高风险债券排在前列的能力。召回率高，说明真实违约债券中有更大比例被提前筛入重点名单；精确率高，则说明前列名单中“真正高风险”的占比更高。")

    with tab3:
        stability = df.groupby("fold", as_index=False).agg(
            pr_auc=("pr_auc", "mean"),
            roc_auc=("roc_auc", "mean"),
            brier=("brier", "mean"),
            top5_recall=("top5_recall", "mean"),
        )
        fig = go.Figure()
        for col, name, color in [("pr_auc", "PR-AUC", "#3E6F73"), ("roc_auc", "ROC-AUC", "#526D89"), ("top5_recall", "Top5% 召回率", "#7A669E")]:
            fig.add_trace(go.Scatter(x=stability["fold"], y=stability[col], mode="lines+markers", name=name, line=dict(color=color, width=3), marker=dict(color=color, size=8)))
        fig.update_layout(title="跨 Fold 平均表现稳定性")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        fig = plot_layout(fig, height=380)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        insight_box("图表解读", "如果不同 Fold 之间曲线较平稳，说明模型在不同时间窗口上的表现较一致，具有更好的泛化能力。若出现明显下降，则提示模型可能对某些年份或市场环境更敏感。")

        dark_table(format_metric_table(stability), max_height=260)

    with tab4:
        st.markdown("#### Fold 时间窗口划分")
        dark_table(fold_schedule, max_height=180)
        st.markdown("#### 指标明细")
        dark_table(format_metric_table(df.drop(columns=["期限"], errors="ignore")), max_height=520)
        csv = df.drop(columns=["horizon_order", "期限"], errors="ignore").to_csv(index=False, encoding="utf-8-sig")
        st.download_button("下载 Fold 测试集指标 CSV", data=csv, file_name="fold_test_metrics.csv", mime="text/csv")


def render_predictions(pred_df: pd.DataFrame) -> None:
    section_header(
        "高风险预测",
        "展示当前预测日期下的高风险债券列表。默认按未来24个月预测概率排序，并用排名百分位划分风险桶。",
    )
    st.markdown(
        "<div class='notice'>风险分层使用排名百分位：Top 1% / 1–5% / 5–10% / 10–20% / Other。用户可优先关注排序靠前且长期违约概率较高的债券。</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='rank-band'><span></span><span></span><span></span><span></span><span></span></div>", unsafe_allow_html=True)
    st.markdown("<div class='pill-row'><span class='soft-pill'>红色：Top 1%</span><span class='soft-pill'>橙色：1–5%</span><span class='soft-pill'>黄色：5–10%</span><span class='soft-pill'>绿色：10–20%</span><span class='soft-pill'>灰色：Other</span></div>", unsafe_allow_html=True)

    top_cols = st.columns([1, 1, 2])
    with top_cols[0]:
        horizon = st.selectbox("排序/展示期限", HORIZON_ORDER, index=3, format_func=lambda h: HORIZON_LABELS[h])
    with top_cols[1]:
        buckets = st.multiselect("风险分层", RISK_BUCKET_ORDER, default=RISK_BUCKET_ORDER[:4])
    with top_cols[2]:
        query = st.text_input("搜索债券代码 / 名称 / 发行人", value="")

    df = pred_df.copy()
    df["rank_for_view"] = df[f"{horizon}_prob_rank"].rank(method="first", ascending=False).astype(int)
    df = df[df["risk_bucket_by_y24m"].isin(buckets)] if buckets else df.iloc[0:0]
    if query:
        q = query.lower().strip()
        mask = (
            df["BondCode"].astype(str).str.lower().str.contains(q, na=False)
            | df["BondName"].astype(str).str.lower().str.contains(q, na=False)
            | df["Issuer"].astype(str).str.lower().str.contains(q, na=False)
            | df["Liscd"].astype(str).str.lower().str.contains(q, na=False)
        )
        df = df[mask]

    df = df.sort_values(f"{horizon}_prob_rank", ascending=False)
    if df.empty:
        st.warning("当前筛选条件下没有匹配债券。")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("筛选后债券数", f"{len(df):,}", "当前条件下的债券数量")
    with c2:
        metric_card("筛选均值", pct(df[f"{horizon}_prob_display"].mean()), HORIZON_LABELS[horizon])
    with c3:
        metric_card("筛选最大值", pct(df[f"{horizon}_prob_display"].max()), "当前筛选集最高风险")
    with c4:
        metric_card("Top 1% 占比", pct((df["risk_bucket_by_y24m"] == "Top 1%").mean()), "筛选集中最高风险桶比例")

    c1, c2 = st.columns([1.1, 1])
    with c1:
        top_n = df.head(20).copy()
        fig = px.bar(
            top_n.sort_values(f"{horizon}_prob_display"),
            x=f"{horizon}_prob_display",
            y="BondCode",
            orientation="h",
            color="risk_bucket_by_y24m",
            category_orders={"risk_bucket_by_y24m": RISK_BUCKET_ORDER},
            title=f"TOP 20：{HORIZON_LABELS[horizon]}预测概率",
        )
        fig.update_xaxes(tickformat=".1%")
        fig = plot_layout(fig, height=530)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        insight_box("图表解读", "左图按照所选期限对高风险债券进行排序。柱子越长，表示该债券在对应预测期限内的违约风险越高；同时处于 Top 1% 或 1–5% 的债券应优先进入预警名单。")
    with c2:
        means = pd.DataFrame({
            "horizon": [HORIZON_LABELS[h] for h in HORIZON_ORDER],
            "avg_prob": [df[f"{h}_prob_display"].mean() if len(df) else 0 for h in HORIZON_ORDER],
            "max_prob": [df[f"{h}_prob_display"].max() if len(df) else 0 for h in HORIZON_ORDER],
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=means["horizon"], y=means["avg_prob"], mode="lines+markers", name="筛选均值"))
        fig.add_trace(go.Scatter(x=means["horizon"], y=means["max_prob"], mode="lines+markers", name="筛选最大值"))
        fig.update_yaxes(tickformat=".1%")
        fig.update_layout(title="筛选债券的期限结构概览")
        fig = plot_layout(fig, height=530)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        insight_box("图表解读", "右图展示当前筛选债券在不同预测期限上的平均风险与最大风险。平均值反映筛选组合整体风险，最大值用于识别极端高风险样本。")

    st.markdown(
        f"""
        <div class="summary-box">
          <b>筛选结果摘要：</b>当前条件下共筛出 <b>{len(df):,}</b> 只债券，{HORIZON_LABELS[horizon]}平均预测概率为 <b>{pct(df[f'{horizon}_prob_display'].mean())}</b>，
          最高值为 <b>{pct(df[f'{horizon}_prob_display'].max())}</b>。建议优先查看排名靠前、风险分层为 Top 1% 或 1–5% 的债券。
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("预测明细")
    show_cols = ["report_rank", "BondCode", "Issuer", "Term", "CreditRating", "risk_bucket_by_y24m", "y6m_prob_display", "y12m_prob_display", "y18m_prob_display", "y24m_prob_display", "report_rank_pct", "most_likely_period", "prob_quality_flag"]
    view = format_prediction_table(df.reindex(columns=show_cols).head(500).copy())
    dark_table(view, max_height=430)

    csv = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("下载当前筛选结果 CSV", data=csv, file_name="bond_default_predictions_filtered.csv", mime="text/csv")


def render_bond_query(pred_df: pd.DataFrame) -> None:
    section_header(
        "单券查询",
        "输入债券代码、债券名称或发行人关键词，查看单支债券的风险等级、风险排名、违约时间结构和模型解释摘要。",
    )

    search_cols = st.columns([2.2, 1, 1])
    with search_cols[0]:
        default_code = str(pred_df.iloc[0]["BondCode"]) if len(pred_df) else ""
        query = st.text_input("债券代码 / 名称 / 发行人", value=default_code, placeholder="例如：111023、某某债、发行人名称")
    with search_cols[1]:
        rank_horizon = st.selectbox("重点观察期限", HORIZON_ORDER, index=3, format_func=lambda h: HORIZON_LABELS[h])
    with search_cols[2]:
        max_matches = st.number_input("最多显示匹配数", min_value=5, max_value=100, value=20, step=5)

    if not query:
        st.info("请输入关键词进行查询。")
        return

    q = query.lower().strip()
    matches = pred_df[
        pred_df["BondCode"].astype(str).str.lower().str.contains(q, na=False)
        | pred_df["BondName"].astype(str).str.lower().str.contains(q, na=False)
        | pred_df["Issuer"].astype(str).str.lower().str.contains(q, na=False)
        | pred_df["Liscd"].astype(str).str.lower().str.contains(q, na=False)
    ].copy()

    if matches.empty:
        st.warning("没有找到匹配债券。可以尝试输入更短的债券代码、债券简称或发行人关键词。")
        return

    matches = matches.sort_values("report_rank").head(int(max_matches))
    if len(matches) > 1:
        label_options = [
            f"{r.BondCode}｜{r.BondName}｜{r.Issuer}｜Rank #{int(r.report_rank)}"
            for r in matches.itertuples()
        ]
        selected_label = st.selectbox("找到多条结果，请选择一只债券", label_options)
        selected_code = selected_label.split("｜", 1)[0]
        row = matches[matches["BondCode"].astype(str) == selected_code].iloc[0]
    else:
        row = matches.iloc[0]

    y6 = float(row.get("y6m_prob_display", 0.0))
    y12 = float(row.get("y12m_prob_display", 0.0))
    y18 = float(row.get("y18m_prob_display", 0.0))
    y24 = float(row.get("y24m_prob_display", 0.0))
    rank_pct = float(row.get("report_rank_pct", 1.0))
    bucket = str(row.get("risk_bucket_by_y24m", "Other"))
    if bucket in ["Top 1%", "1–5%"] or y24 >= 0.30:
        risk_level, risk_tone = "高风险", "#fecaca"
    elif bucket in ["5–10%", "10–20%"] or y24 >= 0.12:
        risk_level, risk_tone = "中风险", "#fde68a"
    else:
        risk_level, risk_tone = "低风险", "#bbf7d0"
    is_top1 = bucket == "Top 1%" or rank_pct <= 0.01

    st.markdown(
        f"""
        <div class="card">
          <div class="eyebrow">Bond Detail</div>
          <h3 style="margin: 6px 0 8px 0;">{row.get('BondName', row.get('BondCode', ''))}</h3>
          <div class="small-muted">代码：{row.get('BondCode', row.get('Liscd', ''))}　|　发行人：{row.get('Issuer', '—')}　|　预测日期：{row.get('PeriodEnd', '未知')}</div>
          <div style="margin-top: 12px;">{display_bucket(bucket)} <span class="badge badge-purple">Rank #{int(row.get('report_rank', 0))}</span> <span class="badge badge-green">排名百分位 {pct(rank_pct)}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="risk-card"><div class="risk-card-label">综合风险等级</div><div class="risk-card-value" style="color:{risk_tone};">{risk_level}</div><div class="risk-card-help">未来24个月累计违约概率：{pct(y24)}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="risk-card"><div class="risk-card-label">最可能违约时段</div><div class="risk-card-value">{row.get('most_likely_period', '—')}</div><div class="risk-card-help">区间概率：{pct(row.get('max_period_prob', np.nan))}</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="risk-card"><div class="risk-card-label">Top 1% 高风险</div><div class="risk-card-value" style="color:{'#fecaca' if is_top1 else '#bbf7d0'};">{'是' if is_top1 else '否'}</div><div class="risk-card-help">按未来24个月风险排名判断</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="risk-card"><div class="risk-card-label">重点期限风险</div><div class="risk-card-value">{pct(row[f'{rank_horizon}_prob_display'])}</div><div class="risk-card-help">{HORIZON_LABELS[rank_horizon]}</div></div>""", unsafe_allow_html=True)

    if row.get("prob_quality_flag", "正常") != "正常":
        st.markdown(
            f"<div class='notice'>概率字段检查：<b>{row.get('prob_quality_flag')}</b>。这通常表示部分预测期限未正确接入、期限结构异常，或出现未来24个月概率异常跳升，建议回到预测 CSV 对应行核查。</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="summary-box">
          <b>单券风险结论：</b>{probability_sentence(row)} {rank_position_sentence(rank_pct, bucket)}
          该债券未来6个月、12个月、18个月、24个月累计违约概率分别为 <b>{pct(y6)}</b>、<b>{pct(y12)}</b>、<b>{pct(y18)}</b>、<b>{pct(y24)}</b>。
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 1])
    with left:
        term_df = pd.DataFrame({
            "期限": [HORIZON_LABELS[h] for h in HORIZON_ORDER],
            "累计违约概率": [row[f"{h}_prob_display"] for h in HORIZON_ORDER],
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=term_df["期限"], y=term_df["累计违约概率"], mode="lines+markers", name="累计违约概率", line=dict(width=3)))
        fig.add_trace(go.Bar(x=term_df["期限"], y=term_df["累计违约概率"], name="概率柱", opacity=0.35))
        fig.update_yaxes(tickformat=".1%")
        fig.update_layout(title="累计违约概率期限结构")
        fig = plot_layout(fig, height=430)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        insight_box("图表解读", "该图展示单只债券在不同预测期限上的累计违约概率。曲线越陡，说明随着时间拉长，风险累积越明显；如果未来6个月概率已经较高，则更需要关注近期风险。")

    with right:
        intervals = pd.DataFrame({
            "时间段": ["0–6个月", "6–12个月", "12–18个月", "18–24个月"],
            "区间概率": [y6, max(y12 - y6, 0), max(y18 - y12, 0), max(y24 - y18, 0)],
        })
        fig = px.bar(intervals, x="时间段", y="区间概率", title="区间违约风险拆分")
        fig.update_yaxes(tickformat=".1%")
        fig = plot_layout(fig, height=430)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        insight_box("图表解读", "该图把累计违约概率拆分为各时间区间的新增风险贡献，有助于判断风险更可能集中在近期还是中长期。")

    detail = pd.DataFrame({
        "项目": [
            "债券代码", "债券名称", "发行机构全称", "期限", "信用等级",
            "预测日期", "风险分层", "排名", "排名百分位", "未来6个月违约概率", "未来12个月违约概率",
            "未来18个月违约概率", "未来24个月违约概率", "概率字段检查",
        ],
        "内容": [
            row.get("BondCode", "—"), row.get("BondName", "—"), row.get("Issuer", "—"), row.get("Term", "—"), row.get("CreditRating", "—"),
            row.get("PeriodEnd", "—"), bucket, int(row.get("report_rank", 0)), pct(rank_pct),
            pct(y6), pct(y12), pct(y18), pct(y24), row.get("prob_quality_flag", "—"),
        ],
    })
    dark_table(detail, max_height=420)


def render_model_note(metrics_df: pd.DataFrame) -> None:
    section_header(
        "模型说明",
        "说明系统的预测口径、核心页面功能和风险解释方式，便于用户理解各项输出。",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("模型框架", "XGBoost", "离散 Hazard 违约预测")
    with c2:
        metric_card("预测期限", "4 个", "未来6个月 / 12个月 / 18个月 / 24个月")
    with c3:
        metric_card("概率口径", "Calibrated", "优先展示 y*_cal_prob")
    with c4:
        metric_card("验证方式", "Rolling Fold", "按时间窗口检验稳定性")

    st.markdown(
        """
        <div class="card">
        <h3>系统功能</h3>
        <p><b>首页总览</b>：查看债券池规模、风险分层分布、期限结构均值与测试表现。</p>
        <p><b>验证结果</b>：比较模型在多个历史测试窗口中的 ROC-AUC、PR-AUC、Brier、Log Loss 和 Top-k 捕获效果。</p>
        <p><b>高风险预测</b>：按照风险排名筛选 Top 1%、1–5%、5–10%、10–20% 等重点债券，并支持导出明细。</p>
        <p><b>单券查询</b>：输入债券代码、债券名称或发行人关键词，查看单只债券的违约概率期限结构、风险分层、排名位置和风险摘要。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 建模口径")
    st.markdown(
        """
        - 模型采用 expanding window 思路，训练窗口随时间扩张，保持时间顺序，降低未来信息泄漏风险。
        - 输出结果包括未来 6 个月、12 个月、18 个月、24 个月的累计违约概率，页面已统一使用中文标签展示。
        - 当前页面优先读取校准后预测文件 pred_20250630.csv，并优先展示 y6m/y12m/y18m/y24m 的 cal_prob 校准后概率。
        - 风险分层以校准后未来24个月风险排名百分位为核心口径：Top 1%、1–5%、5–10%、10–20%、Other。
        - 概率数值用于描述模型判断强弱，风险排名用于识别相对更需要关注的债券。
        """
    )
    st.markdown(
        """
        <div class="glossary-grid">
          <div class="glossary-card"><b>ROC-AUC</b><p>衡量模型区分违约与非违约债券的整体能力，越高越好。</p></div>
          <div class="glossary-card"><b>PR-AUC</b><p>更适合违约样本较少的场景，反映模型识别少数高风险样本的能力。</p></div>
          <div class="glossary-card"><b>Top-k 召回率</b><p>表示真实高风险债券中有多少被筛入前 k% 预警名单。</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_status(pred_df: pd.DataFrame, status: Dict[str, object]) -> None:
    section_header(
        "数据与运行状态",
        "查看当前页面读取的预测文件、预测行数、字段要求和数据预览。",
    )

    expected = [
        f"{DEFAULT_OUTPUT_DIR}/pred_20250630.csv（校准后预测结果，优先读取）",
        f"{DEFAULT_OUTPUT_DIR}/predictions_*.csv（备用预测结果）",
        f"{DEFAULT_OUTPUT_DIR}/metrics_calibrated.csv（概率校准评估指标）",
        "bnd_info_compact.csv / BND_Bndinfo.xlsx（用于补全发行机构全称、期限、信用等级）",
        "fold_test_metrics.csv（可选，用于替换内置测试集指标）",
    ]
    rows = []
    for item in expected:
        if "pred_20250630" in item:
            matched = glob.glob(f"{DEFAULT_OUTPUT_DIR}/pred_20250630.csv")
            rows.append({"项目": item, "状态": "存在" if matched else "缺失", "说明": "; ".join(matched[:3]) if matched else "未发现"})
        elif "predictions_*.csv" in item:
            matched = glob.glob(f"{DEFAULT_OUTPUT_DIR}/predictions_*.csv")
            rows.append({"项目": item, "状态": "存在" if matched else "缺失", "说明": "; ".join(matched[:3]) if matched else "未发现"})
        elif "metrics_calibrated" in item:
            matched = glob.glob(f"{DEFAULT_OUTPUT_DIR}/metrics_calibrated.csv")
            rows.append({"项目": item, "状态": "存在" if matched else "缺失", "说明": "; ".join(matched[:3]) if matched else "未发现"})
        elif "BND_Bndinfo" in item or "bnd_info" in item or "债券基本信息" in item:
            info_status = status.get("bnd_info", {})
            rows.append({"项目": item, "状态": "存在" if info_status.get("found") else "缺失", "说明": info_status.get("message", "未读取")})
        else:
            rows.append({"项目": item, "状态": "内置", "说明": "当前页面已内置测试集指标，也可以后续替换为外部 CSV。"})

    st.markdown(
        f"""
        <div class="notice">
        当前读取模式：<b>{status.get('mode')}</b>；来源：<b>{status.get('source_path') or 'demo data'}</b>；当前预测行数：<b>{len(pred_df):,}</b>。<br/>
        债券基本信息表：<b>{status.get('bnd_info', {}).get('message', '未读取')}</b>。
        </div>
        """,
        unsafe_allow_html=True,
    )
    dark_table(pd.DataFrame(rows), max_height=260)

    st.markdown("### 预测文件需要包含的最低字段")
    st.markdown(
        """<div class="field-code-box">Liscd, PeriodEnd, sem_str, y6m_cum_prob, y12m_cum_prob, y18m_cum_prob, y24m_cum_prob<br><span class="comment"># 可选字段：BondCode, BondName, Issuer, most_likely_period, max_period_prob</span></div>""",
        unsafe_allow_html=True,
    )

    if "prob_quality_flag" in pred_df.columns:
        st.markdown("### 概率字段检查")
        qdf = pred_df["prob_quality_flag"].value_counts(dropna=False).reset_index()
        qdf.columns = ["检查结果", "债券数"]
        dark_table(qdf, max_height=180)

    if "bnd_info_matched" in pred_df.columns:
        st.markdown("### 债券基本信息匹配情况")
        mdf = pd.DataFrame({"匹配状态": ["匹配"], "债券数": [len(pred_df)]})
        dark_table(mdf, max_height=160)

    preview = format_prediction_table(pred_df.head(20).copy())
    st.markdown("### 当前数据预览")
    dark_table(preview, max_height=430)


# =============================================================================
# 6. 表格格式化
# =============================================================================

def format_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {
        "fold": "Fold",
        "test_year": "测试窗口",
        "horizon": "期限",
        "n_samples": "样本数",
        "n_positive": "正样本",
        "base_rate": "基准率",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "brier": "Brier",
        "log_loss": "LogLoss",
        "top1_precision": "Top1% P",
        "top1_recall": "Top1% R",
        "top5_precision": "Top5% P",
        "top5_recall": "Top5% R",
        "top10_precision": "Top10% P",
        "top10_recall": "Top10% R",
    }
    out = out.drop(columns=["horizon_order", "note"], errors="ignore")
    out = out.rename(columns=rename)
    if "期限" in out.columns:
        out["期限"] = out["期限"].replace({"6m": "未来6个月", "12m": "未来12个月", "18m": "未来18个月", "24m": "未来24个月"})
    percent_cols = ["基准率", "Top1% P", "Top1% R", "Top5% P", "Top5% R", "Top10% P", "Top10% R"]
    for c in percent_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: pct(x, 2))
    for c in ["ROC-AUC", "PR-AUC", "Brier", "LogLoss"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: fmt_num(x, 4))
    return out


def format_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {
        "report_rank": "排名",
        "rank_for_view": "展示排名",
        "BondCode": "债券代码",
        "BondName": "债券名称",
        "Issuer": "发行机构全称",
        "IssueDate": "发行日期",
        "MaturityDate": "到期日期",
        "BondType": "债券类型",
        "CreditRating": "信用等级",
        "Term": "期限",
        "CouponRate": "票面利率",
        "IssueAmount": "发行规模（亿元）",
        "risk_bucket_by_y24m": "风险分层",
        "y6m_prob_display": "未来6个月违约概率",
        "y12m_prob_display": "未来12个月违约概率",
        "y18m_prob_display": "未来18个月违约概率",
        "y24m_prob_display": "未来24个月违约概率",
        "report_rank_pct": "排名百分位",
        "most_likely_period": "最可能时段",
        "prob_quality_flag": "概率检查",
    }
    out = out.rename(columns=rename)
    for c in ["未来6个月违约概率", "未来12个月违约概率", "未来18个月违约概率", "未来24个月违约概率", "排名百分位"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: pct(x, 2))
    return out


def render_cover_page() -> None:
    st.markdown(
        """
        <div class="cover-wrap">
          <div class="cover-card">
            <div class="eyebrow">BOND DEFAULT · PREDICTION</div>
            <div class="cover-title">债券违约预测<br/>可视化系统</div>
            <div class="cover-sub">
              本系统面向债券风险识别与预警场景，基于机器学习模型输出不同期限的累计违约概率，
              并通过风险分层、历史测试表现、高风险排行和单券查询，帮助用户快速定位需要重点关注的债券。
            </div>
            <div class="cover-grid">
              <div class="cover-feature"><div class="icon">📊</div><b>首页总览</b><p>查看债券池整体风险、分层结构和核心指标。</p></div>
              <div class="cover-feature"><div class="icon">✅</div><b>验证结果</b><p>展示多个历史测试窗口下的模型表现。</p></div>
              <div class="cover-feature"><div class="icon">⚠️</div><b>高风险预测</b><p>按风险排名筛选 Top 债券并支持导出。</p></div>
              <div class="cover-feature"><div class="icon">🔎</div><b>单券查询</b><p>输入代码、名称或发行人，查看单券风险详情。</p></div>
            </div>
            <div class="pill-row" style="margin-top:26px;">
              <span class="soft-pill">Expanding Window</span><span class="soft-pill">XGBoost Hazard</span><span class="soft-pill">未来6/12/18/24个月</span><span class="soft-pill">Top-k 风险捕获</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1.6, 1, 1.6])
    with col2:
        if st.button("Continue", use_container_width=True):
            st.session_state["entered_app"] = True
            st.rerun()

# =============================================================================
# 7. 主入口
# =============================================================================

def main() -> None:
    inject_css()

    if "entered_app" not in st.session_state:
        st.session_state["entered_app"] = False

    if not st.session_state["entered_app"]:
        render_cover_page()
        return

    page, uploaded_file, explicit_path = sidebar_controls()
    pred_df, status = read_prediction_file(uploaded_file=uploaded_file, explicit_path=explicit_path)
    metrics_df = default_fold_metrics()

    with st.sidebar:
        st.markdown("---")
        if st.button("返回封面", use_container_width=True):
            st.session_state["entered_app"] = False
            st.rerun()

    if page == "首页总览":
        render_home(pred_df, metrics_df, status)
    elif page == "验证结果":
        render_validation(metrics_df)
    elif page == "高风险预测":
        render_predictions(pred_df)
    elif page == "单券查询":
        render_bond_query(pred_df)
    elif page == "模型说明":
        render_model_note(metrics_df)
    elif page == "数据状态":
        render_data_status(pred_df, status)


if __name__ == "__main__":
    main()
