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
        header[data-testid="stHeader"] { display: none !important; }
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
        .bucket-top1 { background: rgba(34, 211, 238, .18); color: #a5f3fc; }
        .bucket-top5 { background: rgba(56, 189, 248, .18); color: #bae6fd; }
        .bucket-top10 { background: rgba(167, 139, 250, .18); color: #ddd6fe; }
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
        .rank-band span:nth-child(1){background:#ef4444;width:1%}.rank-band span:nth-child(2){background:#f97316;width:4%}.rank-band span:nth-child(3){background:#eab308;width:5%}.rank-band span:nth-child(4){background:#22c55e;width:10%}.rank-band span:nth-child(5){background:#64748b;width:80%}

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


        /* FINAL UI PATCH 2026-06-03: keep original sidebar visible, remove collapse option, unify menus, tags and search boxes */
        /* 1) 禁止用户收起侧边栏：隐藏 Streamlit 原生收起/展开控制，不影响侧边栏本体 */
        div[data-testid="collapsedControl"],
        button[data-testid="collapsedControl"],
        div[data-testid="stSidebarCollapsedControl"],
        button[data-testid="stSidebarCollapsedControl"],
        section[data-testid="stSidebar"] button[data-testid="stSidebarCollapseButton"],
        section[data-testid="stSidebar"] div[data-testid="stSidebarCollapseButton"],
        section[data-testid="stSidebar"] button[title*="sidebar"],
        section[data-testid="stSidebar"] button[aria-label*="sidebar"],
        section[data-testid="stSidebar"] button[title*="Sidebar"],
        section[data-testid="stSidebar"] button[aria-label*="Sidebar"],
        section[data-testid="stSidebar"] button[title*="侧边栏"],
        section[data-testid="stSidebar"] button[aria-label*="侧边栏"] {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            width: 0 !important;
            height: 0 !important;
            min-width: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            pointer-events: none !important;
        }

        /* 4) 侧边栏菜单高亮背景长度统一 */
        section[data-testid="stSidebar"] div[data-testid="stRadio"],
        section[data-testid="stSidebar"] div[data-testid="stRadio"] > div,
        section[data-testid="stSidebar"] [role="radiogroup"] {
            width: 100% !important;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label {
            width: 100% !important;
            min-height: 50px !important;
            padding: 11px 14px !important;
            margin: 0 0 6px 0 !important;
            border-radius: 13px !important;
            box-sizing: border-box !important;
            display: flex !important;
            align-items: center !important;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            width: 100% !important;
            background: linear-gradient(90deg, rgba(35,211,195,.18), rgba(59,130,246,.13)) !important;
            border-left: 4px solid #23d3c3 !important;
            border-top: 1px solid rgba(35,211,195,.14) !important;
            border-right: 1px solid rgba(35,211,195,.10) !important;
            border-bottom: 1px solid rgba(35,211,195,.10) !important;
            box-shadow: inset 0 0 0 1px rgba(35,211,195,.10), 0 8px 24px rgba(0,0,0,.16) !important;
        }

        /* 3) 多选标签颜色：去掉不协调的亮红色，改为暗色科技蓝灰 */
        div[data-testid="stMultiSelect"] div[data-baseweb="tag"],
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"],
        div[data-baseweb="select"] div[data-baseweb="tag"],
        div[data-baseweb="select"] span[data-baseweb="tag"],
        div[data-baseweb="tag"] {
            background: linear-gradient(135deg, rgba(30,58,95,.96), rgba(30,64,91,.88)) !important;
            background-color: #1e3a5f !important;
            border: 1px solid rgba(125,211,252,.36) !important;
            color: #e0f2fe !important;
            box-shadow: none !important;
        }
        div[data-testid="stMultiSelect"] div[data-baseweb="tag"] *,
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"] *,
        div[data-baseweb="tag"] * {
            color: #e0f2fe !important;
            -webkit-text-fill-color: #e0f2fe !important;
        }
        div[data-baseweb="tag"] svg,
        div[data-testid="stMultiSelect"] svg {
            color: #93c5fd !important;
            fill: #93c5fd !important;
        }

        /* 5) 搜索框/输入框与下拉框统一为暗色，输入文字为白色 */
        div[data-testid="stTextInput"] div[data-baseweb="input"],
        div[data-testid="stNumberInput"] div[data-baseweb="input"] {
            background: rgba(15,23,42,.92) !important;
            border: 1px solid rgba(148,163,184,.24) !important;
            border-radius: 12px !important;
            box-shadow: none !important;
        }
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input {
            background: rgba(15,23,42,.92) !important;
            color: #f8fafc !important;
            -webkit-text-fill-color: #f8fafc !important;
            caret-color: #55f0de !important;
        }
        div[data-testid="stTextInput"] input::placeholder,
        div[data-testid="stNumberInput"] input::placeholder {
            color: #94a3b8 !important;
            -webkit-text-fill-color: #94a3b8 !important;
            opacity: 1 !important;
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

            invalid_tokens = ["", "—", "-", "未知", "未知发行人", "Unknown", "unknown", "nan"
