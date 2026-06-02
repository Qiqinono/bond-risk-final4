#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
======================================================================
债券违约预测 —— 概率校准脚本
======================================================================
【功能说明】
    1. 读取超参数重新训练模型
    2. 执行单 Fold 的累计概率校准
    3. 报告完整的评估指标（ROC-AUC、PR-AUC、Top K、Brier、Log Loss）
    4. 输出校准曲线和分桶偏差

【时间切分】
    训练集：2007-01-01 ~ 2022-06-30
    验证集：2022-12-31 ~ 2023-06-30
    预测集：2025-06-30 的快照数据

【输出指标】
    ROC-AUC、PR-AUC、Top 1%/5%/10% Precision/Recall
    Brier Score、Log Loss、校准曲线、分桶偏差
======================================================================
"""

import os
import sys
import json
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import IsotonicRegression, CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)

warnings.filterwarnings("ignore")

# ======================================================================
# 配置参数
# ======================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, r"feature_wide_table（更新债券违约数据）(1).xlsx")
DICT_PATH = os.path.join(SCRIPT_DIR, "feature_dictionary.xlsx")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "./output_expanding")
PARAMS_PATH = os.path.join(OUTPUT_DIR, "best_params.json")

# 时间切分
TRAIN_START, TRAIN_END = "2007-01-01", "2022-06-30"
VAL_START,   VAL_END   = "2022-12-31", "2023-06-30"
PREDICT_DATE = "2025-06-30"

# 列名配置
LABELS = ["y6m", "y12m", "y18m", "y24m"]
VALID_LABELS = ["valid_6m", "valid_12m", "valid_18m", "valid_24m"]
HAZARD_E_COLS = ["hazard_e1", "hazard_e2", "hazard_e3", "hazard_e4"]
INTERVAL_NAMES = ["interval_0_6m", "interval_6_12m", "interval_12_18m", "interval_18_24m"]
HORIZONS = ["6m", "12m", "18m", "24m"]

ID_COLS = ["Liscd", "PeriodEnd", "sem_str", "first_event_date"]
NON_FEAT_COLS = set(ID_COLS + LABELS + VALID_LABELS + HAZARD_E_COLS)

# 模型配置
SEED = 42
MAX_BOOST_ROUNDS = 1000
EARLY_STOPPING_RND = 100
TOP_FRACS = [0.01, 0.05, 0.10]
CALIB_N_BINS = 10

# ======================================================================
# 工具函数
# ======================================================================
def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(y_true) == 0 or y_true.min() == y_true.max():
        return None
    return float(roc_auc_score(y_true, y_score))

def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(y_true) == 0 or y_true.min() == y_true.max():
        return None
    return float(average_precision_score(y_true, y_score))

def safe_brier(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(y_true) == 0 or y_true.min() == y_true.max():
        return None
    return float(brier_score_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7)))

def safe_logloss(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(y_true) == 0 or y_true.min() == y_true.max():
        return None
    return float(log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7)))

def top_k_precision_recall(
    y_true: np.ndarray, y_score: np.ndarray, frac: float
) -> Tuple[Optional[float], Optional[float]]:
    n = len(y_true)
    if n == 0:
        return None, None
    k = max(1, int(np.floor(n * frac)))
    total_pos = int(y_true.sum())
    if total_pos == 0:
        return 0.0, 0.0
    top_idx = np.argpartition(-y_score, k - 1)[:k]
    hit = int(y_true[top_idx].sum())
    precision = hit / k
    recall = hit / total_pos
    return precision, recall

def compute_bucket_bias(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10):
    """计算分桶偏差"""
    if len(y_true) == 0 or y_true.min() == y_true.max():
        return None
    
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy='quantile')
    bucket_stats = []
    
    for i, (pt, pp) in enumerate(zip(prob_true, prob_pred)):
        mask = (y_score >= np.percentile(y_score, i * (100/n_bins))) & \
               (y_score <= np.percentile(y_score, (i+1) * (100/n_bins)))
        n_samples = int(mask.sum())
        n_positive = int(y_true[mask].sum())
        
        bucket_stats.append({
            'bucket': i + 1,
            'prob_pred_mean': pp,
            'prob_true_mean': pt,
            'bias': pp - pt,
            'n_samples': n_samples,
            'n_positive': n_positive,
            'actual_rate': n_positive / n_samples if n_samples > 0 else 0.0
        })
    
    return pd.DataFrame(bucket_stats)

# ======================================================================
# Hazard → Cumulative 转换
# ======================================================================
def hazard_to_cumulative(hazard_pred: np.ndarray) -> np.ndarray:
    """将Hazard概率转换为累计违约概率"""
    h = np.clip(hazard_pred, 0.0, 1.0).astype(np.float64)
    h1, h2, h3, h4 = h[:, 0], h[:, 1], h[:, 2], h[:, 3]
    
    # 生存函数
    s1 = 1.0 - h1
    s2 = s1 * (1.0 - h2)
    s3 = s2 * (1.0 - h3)
    s4 = s3 * (1.0 - h4)
    
    # 累计概率
    cumulative = np.stack([
        h1,                              # p6 = h1
        1.0 - s1 * (1.0 - h2),           # p12 = 1 - (1-h1)(1-h2)
        1.0 - s2 * (1.0 - h3),           # p18 = 1 - (1-h1)(1-h2)(1-h3)
        1.0 - s3 * (1.0 - h4),           # p24 = 1 - (1-h1)(1-h2)(1-h3)(1-h4)
    ], axis=1)
    
    return np.clip(cumulative, 0.0, 1.0).astype(np.float32)

def cumulative_to_interval(cumulative: np.ndarray) -> np.ndarray:
    """从累计概率反推区间概率"""
    interval = np.zeros_like(cumulative)
    interval[:, 0] = cumulative[:, 0]  # 0-6m
    for j in range(1, 4):
        interval[:, j] = cumulative[:, j] - cumulative[:, j-1]
    return interval

# ======================================================================
# 数据读取与特征分类
# ======================================================================
def load_data_and_dict(data_path: str, dict_path: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    print(">> 读取建模总表 ...")
    df = pd.read_excel(data_path, engine="openpyxl")
    df["PeriodEnd"] = pd.to_datetime(df["PeriodEnd"], errors="coerce")
    df = df.dropna(subset=["Liscd", "PeriodEnd"]).copy()
    df = df.sort_values(["Liscd", "PeriodEnd"], kind="mergesort").reset_index(drop=True)
    print(f"   总行数：{len(df):,}，总列数：{len(df.columns)}")

    print(">> 读取特征字典 ...")
    feat_dict = pd.read_excel(dict_path, engine="openpyxl")
    feat_dict.columns = feat_dict.columns.str.strip()

    keep_groups = {"trade", "finance", "macro", "static"}
    dict_feats = feat_dict[
        feat_dict["Group"].astype(str).apply(
            lambda g: any(g.startswith(k) for k in keep_groups)
        )
    ]["Feature"].astype(str).tolist()

    dict_feats = [c for c in dict_feats if c in df.columns]

    num_feats, cat_feats = [], []
    for col in dict_feats:
        if col in NON_FEAT_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            num_feats.append(col)
        else:
            cat_feats.append(col)

    print(f"   数值特征：{len(num_feats)} 个，类别特征：{len(cat_feats)} 个")
    return df, num_feats, cat_feats

# ======================================================================
# 时间切分
# ======================================================================
def time_split(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask = (df["PeriodEnd"] >= TRAIN_START) & (df["PeriodEnd"] <= TRAIN_END)
    val_mask   = (df["PeriodEnd"] >= VAL_START)   & (df["PeriodEnd"] <= VAL_END)
    pred_mask  = df["PeriodEnd"] == pd.to_datetime(PREDICT_DATE)

    print(f"\n【时间切分】")
    print(f"   训练集 ({TRAIN_START} ~ {TRAIN_END}): {train_mask.sum():,} 样本")
    print(f"   验证集 ({VAL_START} ~ {VAL_END}): {val_mask.sum():,} 样本")
    print(f"   预测集 ({PREDICT_DATE}): {pred_mask.sum():,} 样本")

    return (
        np.where(train_mask)[0],
        np.where(val_mask)[0],
        np.where(pred_mask)[0],
    )

# ======================================================================
# 预处理类
# ======================================================================
class Preprocessor:
    def __init__(self, num_feats: List[str], cat_feats: List[str], winsor_mult: float = 6.0):
        self.num_feats = num_feats
        self.cat_feats = cat_feats
        self.winsor_mult = winsor_mult
        self.medians = {}
        self.winsor_bounds = {}
        self.cat_maps = {}

    def fit(self, df: pd.DataFrame):
        for col in self.num_feats:
            if col in df.columns:
                self.medians[col] = df[col].median()
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                self.winsor_bounds[col] = (q1 - self.winsor_mult * iqr, q3 + self.winsor_mult * iqr)

        for col in self.cat_feats:
            if col in df.columns:
                freq = df[col].value_counts(dropna=False).sort_values(ascending=False)
                self.cat_maps[col] = {v: i for i, v in enumerate(freq.index)}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.num_feats:
            if col in out.columns and col in self.medians:
                out[col] = out[col].fillna(self.medians[col])
                lower, upper = self.winsor_bounds[col]
                out[col] = out[col].clip(lower, upper)

        for col in self.cat_feats:
            if col in out.columns and col in self.cat_maps:
                out[col] = out[col].map(self.cat_maps[col]).fillna(-1).astype(int)

        return out[self.num_feats + self.cat_feats]

# ======================================================================
# 模型训练
# ======================================================================
def train_xgb_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params: Dict
) -> List[xgb.Booster]:
    """使用最优参数训练4个XGBoost模型"""
    models = []
    
    for j, col in enumerate(HAZARD_E_COLS):
        print(f"\n   训练 {col} ...")
        
        train_mask = y_train[:, j] >= 0
        val_mask   = y_val[:, j] >= 0
        
        dtrain = xgb.DMatrix(X_train[train_mask], label=y_train[train_mask, j])
        dval   = xgb.DMatrix(X_val[val_mask],   label=y_val[val_mask, j])
        
        pos_weight = (train_mask.sum() - y_train[train_mask, j].sum()) / max(y_train[train_mask, j].sum(), 1)
        
        params = best_params[col].copy()
        params.update({
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "scale_pos_weight": float(pos_weight),
            "random_state": SEED,
            "verbosity": 0,
            "nthread": -1,
        })
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=MAX_BOOST_ROUNDS,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=EARLY_STOPPING_RND,
            verbose_eval=False,
        )
        
        val_pred = model.predict(dval)
        val_score = float(average_precision_score(y_val[val_mask, j], val_pred))
        print(f"   OK {col} 验证集 PR-AUC: {val_score:.4f}")
        
        models.append(model)
    
    return models

# ======================================================================
# 概率校准
# ======================================================================
def fit_calibrators(
    y_true: np.ndarray,
    valid: np.ndarray,
    raw_probs: np.ndarray,
    min_samples: int = 10,
    min_positive: int = 5,
    y_train: np.ndarray = None,
    valid_train: np.ndarray = None,
    raw_probs_train: np.ndarray = None
) -> List:
    """为每个horizon拟合校准器（混合策略）"""
    calibrators = []
    
    for j in range(4):
        # 首先尝试使用验证集
        mask = (valid[:, j] > 0.5) & np.isfinite(y_true[:, j]) & np.isfinite(raw_probs[:, j])
        yt = y_true[mask, j]
        pp = raw_probs[mask, j]
        source = "val"
        n_positive = int(yt.sum())
        
        print(f"   检查 Horizon {HORIZONS[j]}: 验证集有效样本数={len(yt)}, 正样本数={n_positive}")
        
        # 如果验证集样本不足，尝试使用训练集
        if len(yt) < min_samples or n_positive < min_positive:
            if y_train is not None and valid_train is not None and raw_probs_train is not None:
                mask_train = (valid_train[:, j] > 0.5) & np.isfinite(y_train[:, j]) & np.isfinite(raw_probs_train[:, j])
                yt = y_train[mask_train, j]
                pp = raw_probs_train[mask_train, j]
                source = "train"
                n_positive = int(yt.sum())
                print(f"   WARNING 验证集样本不足，切换到训练集: 训练集有效样本数={len(yt)}, 正样本数={n_positive}")
        
        # 最终检查
        if len(yt) < min_samples or n_positive < min_positive:
            print(f"   WARNING Horizon {HORIZONS[j]}: 样本不足或正样本太少，使用恒等映射")
            calibrators.append(lambda x: x)
            continue
        
        # 使用 Platt Scaling (逻辑回归校准) - 更平滑，产生更有区分度的概率
        pp_2d = pp.reshape(-1, 1)
        # 添加小量正则化防止过拟合
        cal = LogisticRegression(solver='lbfgs', C=0.5, random_state=SEED)
        cal.fit(pp_2d, yt)
        cal_type = "Platt"
        
        calibrators.append((cal, cal_type))
        print(f"   OK Horizon {HORIZONS[j]}: {cal_type}校准器拟合完成 (使用{source}集, {len(yt)} 样本, {n_positive} 正样本)")
    
    return calibrators

def apply_calibrators(probs: np.ndarray, calibrators: List) -> np.ndarray:
    """应用校准器"""
    calibrated = np.zeros_like(probs)
    for j in range(4):
        cal_item = calibrators[j]
        
        # 检查是否是元组 (calibrator, type)
        if isinstance(cal_item, tuple):
            cal, cal_type = cal_item
            if cal_type == "Platt":
                # Platt Scaling 使用 predict_proba
                pp_2d = probs[:, j].reshape(-1, 1)
                calibrated[:, j] = cal.predict_proba(pp_2d)[:, 1]
            else:
                # Isotonic Regression 使用 predict
                calibrated[:, j] = cal.predict(probs[:, j])
        elif hasattr(cal_item, 'predict'):
            # IsotonicRegression或其他使用predict方法的校准器
            calibrated[:, j] = cal_item.predict(probs[:, j])
        else:
            # 恒等映射或其他可调用对象
            calibrated[:, j] = cal_item(probs[:, j])
    
    return calibrated

def enforce_monotonicity(probs: np.ndarray) -> np.ndarray:
    """强制跨期限单调性 p6 <= p12 <= p18 <= p24，同时保持平滑过渡"""
    n_samples = probs.shape[0]
    result = np.zeros_like(probs)
    
    for i in range(n_samples):
        row = probs[i, :].copy()
        # 前向传播保证单调性
        for j in range(1, 4):
            row[j] = max(row[j], row[j-1])
        
        # 后向调整，确保相邻期限之间不会有过大的跳跃
        for j in range(2, -1, -1):
            # 限制增长幅度，避免突然跳跃
            max_increase = (1.0 - row[j]) / (4 - j)
            row[j] = min(row[j], row[j+1] - 0.001)  # 确保严格递增
            
        result[i, :] = row
    
    return np.clip(result, 0.0, 1.0)

# ======================================================================
# 指标计算
# ======================================================================
def compute_all_metrics(
    y_true: np.ndarray,
    valid: np.ndarray,
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
    split_name: str
) -> pd.DataFrame:
    """计算所有指标（原始和校准后）"""
    rows = []
    
    for j, hor in enumerate(HORIZONS):
        mask = (valid[:, j] > 0.5) & np.isfinite(y_true[:, j])
        yt = y_true[mask, j].astype(np.float64)
        rp = raw_probs[mask, j].astype(np.float64)
        cp = cal_probs[mask, j].astype(np.float64)
        
        if len(yt) == 0 or yt.min() == yt.max():
            continue
        
        # 计算原始概率指标
        row_raw = {
            "split": split_name,
            "horizon": hor,
            "calibrated": "raw",
            "n_samples": len(yt),
            "n_positive": int(yt.sum()),
            "base_rate": round(float(yt.mean()), 6),
            "roc_auc": round(safe_roc_auc(yt, rp), 6),
            "pr_auc": round(safe_pr_auc(yt, rp), 6),
            "brier": round(safe_brier(yt, rp), 6),
            "log_loss": round(safe_logloss(yt, rp), 6),
        }
        
        for frac in TOP_FRACS:
            prec, rec = top_k_precision_recall(yt, rp, frac)
            row_raw[f"top{int(frac*100)}_prec"] = round(prec, 6)
            row_raw[f"top{int(frac*100)}_rec"] = round(rec, 6)
        
        rows.append(row_raw)
        
        # 计算校准后概率指标
        row_cal = {
            "split": split_name,
            "horizon": hor,
            "calibrated": "calibrated",
            "n_samples": len(yt),
            "n_positive": int(yt.sum()),
            "base_rate": round(float(yt.mean()), 6),
            "roc_auc": round(safe_roc_auc(yt, cp), 6),
            "pr_auc": round(safe_pr_auc(yt, cp), 6),
            "brier": round(safe_brier(yt, cp), 6),
            "log_loss": round(safe_logloss(yt, cp), 6),
        }
        
        for frac in TOP_FRACS:
            prec, rec = top_k_precision_recall(yt, cp, frac)
            row_cal[f"top{int(frac*100)}_prec"] = round(prec, 6)
            row_cal[f"top{int(frac*100)}_rec"] = round(rec, 6)
        
        rows.append(row_cal)
    
    return pd.DataFrame(rows)

# ======================================================================
# 绘制校准曲线
# ======================================================================
def plot_calibration_curve(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
    horizon: str,
    output_path: str
):
    """绘制校准曲线"""
    plt.figure(figsize=(8, 6))
    
    # 原始概率
    prob_true_raw, prob_pred_raw = calibration_curve(y_true, raw_probs, n_bins=CALIB_N_BINS, strategy='quantile')
    plt.plot(prob_pred_raw, prob_true_raw, 's-', label=f'Raw (Brier={safe_brier(y_true, raw_probs):.4f})')
    
    # 校准后概率
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, cal_probs, n_bins=CALIB_N_BINS, strategy='quantile')
    plt.plot(prob_pred_cal, prob_true_cal, 'o-', label=f'Calibrated (Brier={safe_brier(y_true, cal_probs):.4f})')
    
    # 完美校准线
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Probability')
    plt.title(f'Calibration Curve - {horizon}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# ======================================================================
# 主流程
# ======================================================================
def main():
    np.random.seed(SEED)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 读取数据
    print("=" * 60)
    print("Step 1: 读取数据与特征分析")
    print("=" * 60)
    df, num_feats, cat_feats = load_data_and_dict(DATA_PATH, DICT_PATH)
    
    # 2. 时间切分
    print("\n" + "=" * 60)
    print("Step 2: 时间切分")
    print("=" * 60)
    train_idx, val_idx, pred_idx = time_split(df)
    
    df_train = df.iloc[train_idx].copy()
    df_val   = df.iloc[val_idx].copy()
    df_pred  = df.iloc[pred_idx].copy()
    
    # 3. 预处理
    print("\n" + "=" * 60)
    print("Step 3: 数据预处理")
    print("=" * 60)
    preproc = Preprocessor(num_feats, cat_feats)
    preproc.fit(df_train)
    
    X_train = preproc.transform(df_train).values
    X_val   = preproc.transform(df_val).values
    X_pred  = preproc.transform(df_pred).values
    
    # 获取标签
    def get_labels(df_part: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y     = df_part[LABELS].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
        valid = df_part[VALID_LABELS].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
        hazard_e = df_part[HAZARD_E_COLS].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
        return y, valid, hazard_e
    
    y_tr, v_tr, he_tr = get_labels(df_train)
    y_va, v_va, he_va = get_labels(df_val)
    
    # 4. 加载超参数并训练模型
    print("\n" + "=" * 60)
    print("Step 4: 加载超参数并训练模型")
    print("=" * 60)
    with open(PARAMS_PATH, 'r', encoding='utf-8') as f:
        best_params = json.load(f)
    print(f"   已加载最优参数：{list(best_params.keys())}")
    
    models = train_xgb_models(X_train, he_tr, X_val, he_va, best_params)
    
    # 5. 预测
    print("\n" + "=" * 60)
    print("Step 5: 预测")
    print("=" * 60)
    
    def predict(X: np.ndarray) -> np.ndarray:
        hazard_pred = np.zeros((X.shape[0], 4), dtype=np.float32)
        for j, model in enumerate(models):
            hazard_pred[:, j] = model.predict(xgb.DMatrix(X))
        return hazard_pred
    
    print("   验证集 ...")
    hazard_val = predict(X_val)
    raw_cum_val = hazard_to_cumulative(hazard_val)
    
    print("   训练集 ...")
    hazard_train = predict(X_train)
    raw_cum_train = hazard_to_cumulative(hazard_train)
    
    print(f"   预测集 ({PREDICT_DATE}) ...")
    hazard_pred = predict(X_pred)
    raw_cum_pred = hazard_to_cumulative(hazard_pred)
    
    # 6. 拟合校准器（直接在Hazard概率上校准）
    print("\n" + "=" * 60)
    print("Step 6: 拟合 Platt Scaling 校准器")
    print("=" * 60)
    
    # 先检查验证集各个期限的样本情况
    print("\n   【验证集样本统计】")
    for j, hor in enumerate(HORIZONS):
        mask = (v_va[:, j] > 0.5) & np.isfinite(y_va[:, j])
        n_samples = int(mask.sum())
        n_positive = int(y_va[mask, j].sum())
        print(f"   Horizon {hor}: 有效样本={n_samples}, 正样本={n_positive}")
    
    # 直接在 Hazard 概率上拟合校准器（不是累计概率）
    calibrators = fit_calibrators(y_va, v_va, hazard_val, 
                                  y_train=y_tr, valid_train=v_tr, raw_probs_train=hazard_train)
    
    # 7. 应用校准器并转换为累计概率
    print("\n" + "=" * 60)
    print("Step 7: 应用校准器并转换为累计概率")
    print("=" * 60)
    
    # 验证集：先校准 Hazard，再转换为累计概率
    cal_hazard_val = apply_calibrators(hazard_val, calibrators)
    cal_cum_val_raw = hazard_to_cumulative(cal_hazard_val)
    cal_cum_val = enforce_monotonicity(cal_cum_val_raw)
    
    # 训练集：先校准 Hazard，再转换为累计概率
    cal_hazard_train = apply_calibrators(hazard_train, calibrators)
    cal_cum_train_raw = hazard_to_cumulative(cal_hazard_train)
    cal_cum_train = enforce_monotonicity(cal_cum_train_raw)
    
    # 预测集：先校准 Hazard，再转换为累计概率
    cal_hazard_pred = apply_calibrators(hazard_pred, calibrators)
    cal_cum_pred_raw = hazard_to_cumulative(cal_hazard_pred)
    cal_cum_pred = enforce_monotonicity(cal_cum_pred_raw)
    
    # 8. 反推区间概率
    print("\n" + "=" * 60)
    print("Step 8: 反推区间概率")
    print("=" * 60)
    
    # 原始区间概率
    raw_interval_val = cumulative_to_interval(raw_cum_val)
    raw_interval_train = cumulative_to_interval(raw_cum_train)
    raw_interval_pred = cumulative_to_interval(raw_cum_pred)
    
    # 校准后区间概率
    cal_interval_val = cumulative_to_interval(cal_cum_val)
    cal_interval_train = cumulative_to_interval(cal_cum_train)
    cal_interval_pred = cumulative_to_interval(cal_cum_pred)
    
    # 9. 计算指标
    print("\n" + "=" * 60)
    print("Step 9: 计算指标")
    print("=" * 60)
    
    train_metrics = compute_all_metrics(y_tr, v_tr, raw_cum_train, cal_cum_train, "train")
    val_metrics = compute_all_metrics(y_va, v_va, raw_cum_val, cal_cum_val, "val")
    
    all_metrics = pd.concat([train_metrics, val_metrics], ignore_index=True)
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_calibrated.csv")
    all_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"   指标表已保存：{metrics_path}")
    
    # 打印报告
    print("\n" + "=" * 80)
    print("概率校准评估报告")
    print("=" * 80)
    
    for split in ["train", "val"]:
        print(f"\n【{split.upper()} SET】")
        split_df = all_metrics[all_metrics["split"] == split]
        
        for hor in HORIZONS:
            raw_rows = split_df[(split_df["horizon"] == hor) & (split_df["calibrated"] == "raw")]
            cal_rows = split_df[(split_df["horizon"] == hor) & (split_df["calibrated"] == "calibrated")]
            
            if len(raw_rows) == 0 or len(cal_rows) == 0:
                print(f"\nHorizon: {hor}")
                print(f"  无有效样本")
                continue
            
            raw_row = raw_rows.iloc[0]
            cal_row = cal_rows.iloc[0]
            
            print(f"\nHorizon: {hor}")
            print(f"  样本数: {raw_row['n_samples']:,} | 正样本: {raw_row['n_positive']:,} | 基准率: {raw_row['base_rate']:.4f}")
            print(f"\n  [RAW]")
            print(f"    ROC-AUC: {raw_row['roc_auc']:.4f} | PR-AUC: {raw_row['pr_auc']:.4f}")
            print(f"    Brier Score: {raw_row['brier']:.4f} | Log Loss: {raw_row['log_loss']:.4f}")
            print(f"    Top1%: Precision={raw_row['top1_prec']:.4f}, Recall={raw_row['top1_rec']:.4f}")
            print(f"    Top5%: Precision={raw_row['top5_prec']:.4f}, Recall={raw_row['top5_rec']:.4f}")
            print(f"    Top10%: Precision={raw_row['top10_prec']:.4f}, Recall={raw_row['top10_rec']:.4f}")
            
            print(f"\n  [CALIBRATED]")
            print(f"    ROC-AUC: {cal_row['roc_auc']:.4f} | PR-AUC: {cal_row['pr_auc']:.4f}")
            print(f"    Brier Score: {cal_row['brier']:.4f} | Log Loss: {cal_row['log_loss']:.4f}")
            print(f"    Top1%: Precision={cal_row['top1_prec']:.4f}, Recall={cal_row['top1_rec']:.4f}")
            print(f"    Top5%: Precision={cal_row['top5_prec']:.4f}, Recall={cal_row['top5_rec']:.4f}")
            print(f"    Top10%: Precision={cal_row['top10_prec']:.4f}, Recall={cal_row['top10_rec']:.4f}")
    
    # 10. 生成校准曲线和分桶偏差
    print("\n" + "=" * 60)
    print("Step 10: 生成校准曲线和分桶偏差")
    print("=" * 60)
    
    for j, hor in enumerate(HORIZONS):
        mask = (v_va[:, j] > 0.5) & np.isfinite(y_va[:, j])
        yt = y_va[mask, j]
        rp = raw_cum_val[mask, j]
        cp = cal_cum_val[mask, j]
        
        if len(yt) > 0 and yt.min() != yt.max():
            # 绘制校准曲线
            curve_path = os.path.join(OUTPUT_DIR, f"calibration_{hor}.png")
            plot_calibration_curve(yt, rp, cp, hor, curve_path)
            print(f"   OK 校准曲线已保存: calibration_{hor}.png")
            
            # 计算分桶偏差
            bucket_df_raw = compute_bucket_bias(yt, rp)
            bucket_df_cal = compute_bucket_bias(yt, cp)
            
            if bucket_df_raw is not None:
                bucket_df_raw["calibrated"] = "raw"
                bucket_df_cal["calibrated"] = "calibrated"
                bucket_df = pd.concat([bucket_df_raw, bucket_df_cal], ignore_index=True)
                bucket_path = os.path.join(OUTPUT_DIR, f"bucket_bias_{hor}.csv")
                bucket_df.to_csv(bucket_path, index=False, encoding="utf-8-sig")
                print(f"   OK 分桶偏差已保存: bucket_bias_{hor}.csv")
    
    # 11. 保存预测结果
    print("\n" + "=" * 60)
    print("Step 11: 保存预测结果")
    print("=" * 60)
    
    # 训练集预测
    out_train = df_train[["Liscd", "PeriodEnd"]].copy().reset_index(drop=True)
    out_train["split"] = "train"
    for j, lab in enumerate(LABELS):
        out_train[f"{lab}_true"] = y_tr[:, j]
        out_train[f"{lab}_valid"] = v_tr[:, j]
        out_train[f"{lab}_raw_prob"] = raw_cum_train[:, j]
        out_train[f"{lab}_cal_prob"] = cal_cum_train[:, j]
    for j, name in enumerate(INTERVAL_NAMES):
        out_train[f"{name}_raw"] = raw_interval_train[:, j]
        out_train[f"{name}_cal"] = cal_interval_train[:, j]
    out_train.to_csv(os.path.join(OUTPUT_DIR, "pred_train.csv"), index=False, encoding="utf-8-sig")
    
    # 验证集预测
    out_val = df_val[["Liscd", "PeriodEnd"]].copy().reset_index(drop=True)
    out_val["split"] = "val"
    for j, lab in enumerate(LABELS):
        out_val[f"{lab}_true"] = y_va[:, j]
        out_val[f"{lab}_valid"] = v_va[:, j]
        out_val[f"{lab}_raw_prob"] = raw_cum_val[:, j]
        out_val[f"{lab}_cal_prob"] = cal_cum_val[:, j]
    for j, name in enumerate(INTERVAL_NAMES):
        out_val[f"{name}_raw"] = raw_interval_val[:, j]
        out_val[f"{name}_cal"] = cal_interval_val[:, j]
    out_val.to_csv(os.path.join(OUTPUT_DIR, "pred_val.csv"), index=False, encoding="utf-8-sig")
    
    # 预测集预测
    out_pred = df_pred[["Liscd", "PeriodEnd", "sem_str"]].copy().reset_index(drop=True)
    out_pred["split"] = f"predict_{PREDICT_DATE}"
    for j, lab in enumerate(LABELS):
        out_pred[f"{lab}_raw_prob"] = raw_cum_pred[:, j]
        out_pred[f"{lab}_cal_prob"] = cal_cum_pred[:, j]
    for j, name in enumerate(INTERVAL_NAMES):
        out_pred[f"{name}_raw"] = raw_interval_pred[:, j]
        out_pred[f"{name}_cal"] = cal_interval_pred[:, j]
    
    # 添加无违约概率
    out_pred["no_default_raw"] = 1.0 - raw_cum_pred[:, 3]
    out_pred["no_default_cal"] = 1.0 - cal_cum_pred[:, 3]
    
    pred_path = os.path.join(OUTPUT_DIR, f"pred_{PREDICT_DATE.replace('-', '')}.csv")
    out_pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
    
    print("\nAll Done! 输出文件：")
    print(f"   - {metrics_path}")
    print(f"   - pred_train.csv")
    print(f"   - pred_val.csv")
    print(f"   - {pred_path}")
    print(f"   - calibration_*.png")
    print(f"   - bucket_bias_*.csv")
    
    # 12. 交互式查询
    print("\n" + "=" * 60)
    print("债券违约概率查询系统")
    print("=" * 60)
    
    while True:
        code = input("\n请输入债券代码（输入 'q' 退出，输入 'top' 查看违约概率最高的债券）：").strip()
        if code.lower() == 'q':
            break
        elif code.lower() == 'top':
            out_pred_sorted = out_pred.sort_values('y24m_cal_prob', ascending=False).head(10)
            print(f"\n【{PREDICT_DATE} 违约概率最高的前10支债券】")
            print("-" * 80)
            print(f"{'债券代码':<12} {'6m(raw)':<10} {'6m(cal)':<10} {'12m(raw)':<10} {'12m(cal)':<10} {'24m(raw)':<10} {'24m(cal)':<10}")
            print("-" * 80)
            for _, row in out_pred_sorted.iterrows():
                print(f"{row['Liscd']:<12} {row['y6m_raw_prob']:.4f}      {row['y6m_cal_prob']:.4f}      {row['y12m_raw_prob']:.4f}      {row['y12m_cal_prob']:.4f}      {row['y24m_raw_prob']:.4f}      {row['y24m_cal_prob']:.4f}")
            continue
        
        mask = out_pred["Liscd"].astype(str) == str(code)
        if not mask.any():
            print(f"未找到债券代码: {code}")
            continue
        
        bond_data = out_pred[mask].iloc[0]
        print(f"\n债券代码: {bond_data['Liscd']}")
        print(f"数据日期: {bond_data['sem_str']}")
        
        print("\n【原始概率 (Raw)】")
        print(f"  ├─ 未来6个月: {bond_data['y6m_raw_prob']:.4f} ({bond_data['y6m_raw_prob']*100:.2f}%)")
        print(f"  ├─ 未来12个月: {bond_data['y12m_raw_prob']:.4f} ({bond_data['y12m_raw_prob']*100:.2f}%)")
        print(f"  ├─ 未来18个月: {bond_data['y18m_raw_prob']:.4f} ({bond_data['y18m_raw_prob']*100:.2f}%)")
        print(f"  └─ 未来24个月: {bond_data['y24m_raw_prob']:.4f} ({bond_data['y24m_raw_prob']*100:.2f}%)")
        
        print("\n【校准后概率 (Calibrated)】")
        print(f"  ├─ 未来6个月: {bond_data['y6m_cal_prob']:.4f} ({bond_data['y6m_cal_prob']*100:.2f}%)")
        print(f"  ├─ 未来12个月: {bond_data['y12m_cal_prob']:.4f} ({bond_data['y12m_cal_prob']*100:.2f}%)")
        print(f"  ├─ 未来18个月: {bond_data['y18m_cal_prob']:.4f} ({bond_data['y18m_cal_prob']*100:.2f}%)")
        print(f"  └─ 未来24个月: {bond_data['y24m_cal_prob']:.4f} ({bond_data['y24m_cal_prob']*100:.2f}%)")
        
        print("\n【区间违约概率 (Calibrated)】")
        print(f"  ├─ 0-6个月: {bond_data['interval_0_6m_cal']:.4f}")
        print(f"  ├─ 6-12个月: {bond_data['interval_6_12m_cal']:.4f}")
        print(f"  ├─ 12-18个月: {bond_data['interval_12_18m_cal']:.4f}")
        print(f"  ├─ 18-24个月: {bond_data['interval_18_24m_cal']:.4f}")
        print(f"  └─ 24个月内不违约: {bond_data['no_default_cal']:.4f}")

if __name__ == "__main__":
    main()