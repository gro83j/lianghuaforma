#!/usr/bin/env python3
"""
Mode4 TP2 深度优化（Round 019，TP2-only）

Hard rules:
- Only optimize mode4 (long/short). Do NOT touch mode1/mode2 artifacts or code.
- Selection uses pre-OS (2015-2022) only; OS (2023-2025) is for final reporting,
  except a feasibility constraint OS_epd>0.
- Purge/Embargo fixed: 40/40 bars (M5).
- Initial capital fixed: 200 USD; report maxDD_usd and maxDD%.

Outputs (must be written):
BASE_DIR/20260112/019.txt
BASE_DIR/20260112/019_artifacts/*

Run with:
python "experiments/20260112_019_Mode4_TP2_ConfidenceMax.py"
"""

from __future__ import annotations

import dataclasses
import hashlib
import heapq
import json
import math
import os
import pickle
import subprocess
import shutil
import sys
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# sklearn/lightgbm are available in conda env trend_py311

# Reduce noisy sklearn feature-name warnings (does not affect correctness).
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but .* was fitted with feature names",
    category=UserWarning,
)


# =============================
# Config
# =============================


@dataclass(frozen=True)
class Paths:
    out_dir: Path
    artifacts_dir: Path
    report_path: Path
    desktop_script_copy: Path


@dataclass(frozen=True)
class TimeConfig:
    start_utc: str = "2010-01-01"
    end_utc: str = "2100-01-01"
    backtest_start_utc: str = "2015-01-01"
    backtest_end_utc: str = "2025-12-26 23:59:59"
    preos_start_utc: str = "2015-01-01"
    preos_end_utc: str = "2022-12-31 23:59:59"
    os_start_utc: str = "2023-01-01"


@dataclass(frozen=True)
class MarketConfig:
    contract_size: float = 100.0
    roundtrip_cost_price: float = 0.200
    slippage_buffer_price: float = 0.050  # used in lot_math_audit and conservative sizing
    initial_capital_usd: float = 200.0
    lot_step: float = 0.01
    min_lot: float = 0.01
    max_lot: float = 0.06


@dataclass(frozen=True)
class Mode4SignalConfig:
    sl_cross_lookback_bars: int = 240
    near_window_bars: int = 6
    warmup_min_bars: int = 300
    entry_delay: int = 0
    confirm_window: int = 0
    fast_abs_ratio: float = 1.0
    zero_eps_mult: float = 0.0


@dataclass(frozen=True)
class SignalSearchConfig:
    entry_delay_grid: Tuple[int, ...] = (0, 1, 2)
    confirm_window_grid: Tuple[int, ...] = (0, 2, 4, 6)
    fast_abs_ratio_grid: Tuple[float, ...] = (0.6, 0.7, 0.8, 0.9, 1.0)
    zero_eps_grid: Tuple[float, ...] = (0.0, 1e-6, 1e-5)


@dataclass(frozen=True)
class CVConfig:
    purge_bars: int = 40
    embargo_bars: int = 40
    calib_cv_splits: int = 2
    seed: int = 10


@dataclass(frozen=True)
class ThresholdConfig:
    score_lookback_days: int = 60
    min_score_history: int = 80
    # q means threshold = running_quantile(score_hist, q), accept if score>=thr
    # allow higher pass rate to recover epd; selection still pre-OS only
    q_grid: Tuple[float, ...] = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
    # Gate-1 (tail) uses q_tail on p_tail history: accept if p_tail <= thr_tail.
    # Larger q_tail => higher take_rate (less aggressive).
    q_tail_grid: Tuple[float, ...] = (0.60, 0.70, 0.80, 0.90, 0.95, 0.98)
    gate1_take_rate_min: float = 0.60
    tp1_target_p: float = 0.90
    tp1_min_posterior: float = 0.70


@dataclass(frozen=True)
class RiskConfig:
    max_risk_usd_per_trade_grid: Tuple[float, ...]
    daily_stop_loss_usd_grid: Tuple[float, ...] = (10.0, 12.0, 15.0)
    max_parallel_same_dir_grid: Tuple[int, ...] = (1,)
    tickets_per_signal_grid: Tuple[int, ...] = (1,)
    cooldown_bars_grid: Tuple[int, ...] = (0,)
    dd_trigger_usd_grid: Tuple[float, ...] = (45.0, 60.0, 75.0, 90.0)
    dd_stop_cooldown_bars_grid: Tuple[int, ...] = (480, 960, 1440)
    risk_scale_min_grid: Tuple[float, ...] = (0.10, 0.15, 0.20)
    # rolling DD governor (降风险+冷却恢复; 禁止长期 dd_stop 跳单)
    dd_rolling_window_days: int = 180
    dd_trigger_usd: float = 60.0
    dd_trigger_usd_year: float = 60.0
    dd_trigger_usd_quarter: float = 60.0
    dd_stop_cooldown_bars: int = 960  # default mid-point for search
    dd_recover_ratio: float = 0.85  # re-arm dd stop after partial recovery
    risk_scale_min: float = 0.15
    equity_floor_usd: float = 140.0  # conservative backstop


@dataclass(frozen=True)
class ExitSearchConfig:
    # Economic prune (theoretical lower bound): require TP1_R >= k * cost_R
    # where cost_R = (roundtrip_cost_price+slippage_buffer_price)/sl_dist.
    tp1_over_cost_k_grid: Tuple[float, ...] = (1.5, 2.0, 3.0, 4.0)
    entry_grid: Tuple[str, ...] = ("event",)
    H_grid: Tuple[int, ...] = (144, 240)
    # TP2-only horizon candidates (post-TP1)
    H2_grid: Tuple[int, ...] = (240, 360, 480, 720)
    tp1_atr_mult_grid: Tuple[float, ...] = (1.0,)
    sl_atr_mult_grid: Tuple[float, ...] = (1.0,)
    tp1_close_frac_grid: Tuple[float, ...] = (0.5,)
    tp2_mult_grid: Tuple[float, ...] = (2.0,)
    tp1_q_grid: Tuple[float, ...] = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25)
    sl_q_grid: Tuple[float, ...] = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
    tp2_q_grid: Tuple[float, ...] = (0.40, 0.45, 0.50)
    # legacy single-quantile defaults (kept for backward compatibility in older helpers)
    tp1_q: float = 0.15
    sl_q: float = 0.80
    tp2_q: float = 0.45
    min_tp1_r: float = 0.1
    max_tp1_r: float = 3.0
    min_sl_r: float = 0.6
    max_sl_r: float = 3.5
    tp2_n1_grid: Tuple[int, ...] = (3, 5, 8)
    tp2_n2_grid: Tuple[int, ...] = (10, 13)


@dataclass(frozen=True)
class ModelConfig:
    # LightGBM grid (to avoid "No further splits" under low-variance features)
    lgbm_base_params: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "min_gain_to_split": 0.0,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "reg_lambda": 2.0,
            "random_state": 10,
            "n_jobs": -1,
            "verbose": -1,
        }
    )
    # Runtime guard: keep grid intentionally small; selection is still WF/OOF on pre-OS.
    min_data_in_leaf_grid: Tuple[int, ...] = (20, 40)
    num_leaves_grid: Tuple[int, ...] = (31,)
    max_depth_grid: Tuple[int, ...] = (3, 4)
    feature_fraction_grid: Tuple[float, ...] = (0.9,)
    min_gain_to_split_grid: Tuple[float, ...] = (0.0,)
    calib_methods: Tuple[str, ...] = ("sigmoid", "isotonic")
    min_train_events: int = 600


# =============================
# Utils
# =============================


def to_utc_ts(x: str) -> pd.Timestamp:
    return pd.Timestamp(x, tz="UTC")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy2_into_dir(src: Path, dst_dir: Path) -> Path:
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        v = float(x)
        if not np.isfinite(v):
            return "NA"
        return f"{v:.{nd}f}"
    except Exception:
        return str(x)


def locate_xauusd_m5(root: Path) -> Path:
    candidates: List[Path] = []
    preferred = root / "data_xauusd" / "xauusd_M5.csv"
    if preferred.exists():
        return preferred
    patterns = [
        "**/xauusd_M5.csv",
        "**/XAUUSD*_M5*.csv",
        "**/*XAUUSD*M5*.csv",
    ]
    for pat in patterns:
        candidates.extend([p for p in root.glob(pat) if p.is_file()])
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(f"未在 {root} 下定位到 XAUUSD M5 CSV")
    # pick largest (most rows)
    return max(candidates, key=lambda p: p.stat().st_size)


# =============================
# Indicators (causal)
# =============================


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.astype(float).ewm(span=int(span), adjust=False).mean()


def wilder_ema(series: pd.Series, period: int) -> pd.Series:
    alpha = 1.0 / float(period)
    return series.astype(float).ewm(alpha=alpha, adjust=False).mean()


def compute_atr14(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return wilder_ema(tr, 14)


def compute_rsi14(df: pd.DataFrame) -> pd.Series:
    close = df["close"].astype(float)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = wilder_ema(up, 14)
    roll_down = wilder_ema(down, 14)
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_adx14(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)

    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, 14)
    plus_di = 100.0 * wilder_ema(plus_dm, 14) / atr.replace(0.0, np.nan)
    minus_di = 100.0 * wilder_ema(minus_dm, 14) / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = wilder_ema(dx.fillna(0.0), 14)
    return adx.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)


def compute_kdj(df: pd.DataFrame, n: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    high_n = df["high"].astype(float).rolling(int(n), min_periods=int(n)).max()
    low_n = df["low"].astype(float).rolling(int(n), min_periods=int(n)).min()
    close = df["close"].astype(float)
    denom = (high_n - low_n).replace(0.0, np.nan)
    rsv = (close - low_n) / denom * 100.0
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3.0 * k - 2.0 * d
    return k.fillna(50.0), d.fillna(50.0), j.fillna(50.0)


def compute_boll_width(df: pd.DataFrame, n: int = 20, n_std: float = 2.0) -> pd.Series:
    close = df["close"].astype(float)
    ma = close.rolling(int(n), min_periods=int(n)).mean()
    std = close.rolling(int(n), min_periods=int(n)).std(ddof=0)
    upper = ma + float(n_std) * std
    lower = ma - float(n_std) * std
    width = (upper - lower) / ma.replace(0.0, np.nan)
    return width.fillna(0.0)


def compute_macd(series: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series]:
    m = ema(series, fast) - ema(series, slow)
    s = ema(m, signal)
    return m, s


def compute_rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.astype(float).diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1.0 / float(n), adjust=False, min_periods=int(n)).mean()
    roll_down = down.ewm(alpha=1.0 / float(n), adjust=False, min_periods=int(n)).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    sma = tp.rolling(int(n), min_periods=int(n)).mean()
    mad = (tp - sma).abs().rolling(int(n), min_periods=int(n)).mean()
    denom = (0.015 * mad).replace(0.0, np.nan)
    cci = (tp - sma) / denom
    return cci.fillna(0.0)


def compute_roc(series: pd.Series, n: int) -> pd.Series:
    s = series.astype(float)
    roc = s / s.shift(int(n)) - 1.0
    roc = roc.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return roc


def safe_div(a: np.ndarray, b: np.ndarray, default: float = 0.0) -> np.ndarray:
    out = np.full_like(a, default, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    out[mask] = a[mask] / b[mask]
    return out


def compute_crosses(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    up = (a > b) & (np.roll(a, 1) <= np.roll(b, 1))
    dn = (a < b) & (np.roll(a, 1) >= np.roll(b, 1))
    up[0] = False
    dn[0] = False
    return up, dn


def last_true_index(mask: np.ndarray) -> np.ndarray:
    n = int(len(mask))
    out = np.full(n, -1, dtype=int)
    last = -1
    for i in range(n):
        if bool(mask[i]):
            last = i
        out[i] = last
    return out


def compute_macd_segment_context(macd12: np.ndarray, *, zero_eps: float = 1e-12) -> Dict[str, np.ndarray]:
    n = int(len(macd12))
    prev_opp_abs = np.full(n, np.nan, dtype=float)
    prev_opp_extreme_i = np.full(n, -1, dtype=int)
    seg_len = np.full(n, np.nan, dtype=float)
    seg_slope = np.full(n, np.nan, dtype=float)
    seg_start_i = np.full(n, -1, dtype=int)
    seg_peak_abs = np.full(n, np.nan, dtype=float)

    last_completed_peak_abs: Dict[int, float] = {1: float("nan"), -1: float("nan")}
    last_completed_peak_i: Dict[int, int] = {1: -1, -1: -1}

    cur_sign: Optional[int] = None
    cur_peak_abs = float("nan")
    cur_peak_i = -1
    cur_prev_opp_abs = float("nan")
    cur_prev_opp_i = -1
    cur_start_i = -1
    cur_start_v = float("nan")

    for i in range(n):
        v = float(macd12[i])
        if not np.isfinite(v) or abs(v) <= float(zero_eps):
            prev_opp_abs[i] = float(cur_prev_opp_abs) if np.isfinite(cur_prev_opp_abs) else float("nan")
            prev_opp_extreme_i[i] = int(cur_prev_opp_i)
            seg_start_i[i] = int(cur_start_i)
            continue
        sign = 1 if v > 0.0 else -1
        abs_v = abs(v)

        if cur_sign is None:
            cur_sign = int(sign)
            cur_start_i = int(i)
            cur_start_v = float(v)
            cur_prev_opp_abs = float(last_completed_peak_abs.get(int(-sign), float("nan")))
            cur_prev_opp_i = int(last_completed_peak_i.get(int(-sign), -1))
            cur_peak_abs = float(abs_v)
            cur_peak_i = int(i)
        elif int(sign) != int(cur_sign):
            last_completed_peak_abs[int(cur_sign)] = float(cur_peak_abs)
            last_completed_peak_i[int(cur_sign)] = int(cur_peak_i)
            cur_sign = int(sign)
            cur_start_i = int(i)
            cur_start_v = float(v)
            cur_prev_opp_abs = float(last_completed_peak_abs.get(int(-sign), float("nan")))
            cur_prev_opp_i = int(last_completed_peak_i.get(int(-sign), -1))
            cur_peak_abs = float(abs_v)
            cur_peak_i = int(i)
        else:
            if (not np.isfinite(cur_peak_abs)) or (abs_v > float(cur_peak_abs)):
                cur_peak_abs = float(abs_v)
                cur_peak_i = int(i)

        prev_opp_abs[i] = float(cur_prev_opp_abs) if np.isfinite(cur_prev_opp_abs) else float("nan")
        prev_opp_extreme_i[i] = int(cur_prev_opp_i)
        seg_start_i[i] = int(cur_start_i)
        seg_peak_abs[i] = float(cur_peak_abs) if np.isfinite(cur_peak_abs) else float("nan")
        if int(cur_start_i) >= 0 and np.isfinite(cur_start_v):
            L = int(i - int(cur_start_i) + 1)
            seg_len[i] = float(L)
            denom = float(max(1, L - 1))
            seg_slope[i] = float((v - float(cur_start_v)) / denom)

    return {
        "prev_opp_abs": prev_opp_abs,
        "prev_opp_extreme_i": prev_opp_extreme_i,
        "seg_len": seg_len,
        "seg_slope": seg_slope,
        "seg_start_i": seg_start_i,
        "seg_peak_abs": seg_peak_abs,
    }


def segment_peak_running(macd: np.ndarray, *, zero_eps: float = 1e-12) -> np.ndarray:
    m = np.asarray(macd, dtype=float)
    sign = np.sign(m)
    sign[np.abs(m) <= float(zero_eps)] = 0.0
    prev = np.roll(sign, 1)
    prev[0] = sign[0]
    change = sign != prev
    change[0] = True
    group = np.cumsum(change)
    abs_m = np.abs(m)
    return pd.Series(abs_m).groupby(group, sort=False).cummax().to_numpy(dtype=float)


def compute_macd5_area_since_last_cross(macd: pd.Series, sig: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    m = macd.astype(float).to_numpy()
    s = sig.astype(float).to_numpy()
    hist = m - s
    up, dn = compute_crosses(m, s)
    cross = up | dn
    n = int(len(hist))
    area = np.full(n, 0.0, dtype=float)
    bars_since = np.full(n, np.nan, dtype=float)
    last_cross = -1
    acc = 0.0
    for i in range(n):
        if bool(cross[i]):
            last_cross = i
            acc = 0.0
            area[i] = 0.0
            bars_since[i] = 0.0
            continue
        if last_cross < 0:
            area[i] = 0.0
            bars_since[i] = np.nan
            continue
        v = float(hist[i]) if np.isfinite(hist[i]) else 0.0
        acc += v
        area[i] = float(acc)
        bars_since[i] = float(i - last_cross)
    return area, bars_since


def compute_fractals_confirmed(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    n = int(len(df))
    fh = np.full(n, np.nan, dtype=float)
    fl = np.full(n, np.nan, dtype=float)
    for i in range(2, n - 2):
        h = float(high[i])
        l = float(low[i])
        if np.isfinite(h):
            if h > float(high[i - 1]) and h > float(high[i - 2]) and h > float(high[i + 1]) and h > float(high[i + 2]):
                fh[i + 2] = h  # confirmed at i+2 (causal)
        if np.isfinite(l):
            if l < float(low[i - 1]) and l < float(low[i - 2]) and l < float(low[i + 1]) and l < float(low[i + 2]):
                fl[i + 2] = l
    return fh, fl


# =============================
# Feature context (causal)
# =============================


HOUR_OH_COLS: Tuple[str, ...] = tuple([f"hour_{h:02d}" for h in range(24)])

MACD_SLOW_GRID: Tuple[Tuple[int, int, int], ...] = (
    (8, 20, 8),
    (8, 20, 9),
    (8, 20, 10),
    (8, 26, 8),
    (8, 26, 9),
    (8, 26, 10),
    (8, 34, 8),
    (8, 34, 9),
    (8, 34, 10),
    (10, 20, 8),
    (10, 20, 9),
    (10, 20, 10),
    (10, 26, 8),
    (10, 26, 9),
    (10, 26, 10),
    (10, 34, 8),
    (10, 34, 9),
    (10, 34, 10),
    (12, 20, 8),
    (12, 20, 9),
    (12, 20, 10),
    (12, 26, 8),
    (12, 26, 9),
    (12, 26, 10),
    (12, 34, 8),
    (12, 34, 9),
    (12, 34, 10),
)

MACD_SLOW_FEATURES: Tuple[str, ...] = tuple(
    f"macd_slow_{fast}_{slow}_{signal}_{suffix}"
    for (fast, slow, signal) in MACD_SLOW_GRID
    for suffix in ("hist", "cross_age", "hist_slope_atr", "seg_peak_run")
)

FEATURE_COLS: Tuple[str, ...] = (
    # mode4 signal structure
    "macd_fast_abs_to_prev_opp_peak",
    "seg_len",
    "seg_slope_atr",
    "seg_peak_atr",
    "hist_slope_atr",
    "macd12_hist",
    "macd12_hist_sign",
    "macd12_hist_z",
    "macd12_hist_burst",
    "macd12_cross_age",
    "macd12_hist_slope_atr",
    "macd12_seg_peak_run",
    "macd5_hist",
    "macd5_cross_age",
    "macd5_hist_slope_atr",
    "macd5_seg_peak_run",
    "cross_to_entry_bars",
    # price action
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "roc_4",
    "roc_8",
    "roc_12",
    "roc_20",
    "range_pos_20",
    "dist_high_20",
    "dist_low_20",
    "bar_range",
    "bar_range_to_tr",
    "wick_ratio",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "mom_1",
    "mom_1_diff",
    "mom_1_diff2",
    "consec_up",
    "consec_down",
    "bearish_engulfing",
    "bullish_engulfing",
    "hammer",
    "pivot_dist_high",
    "pivot_dist_low",
    "breakout_flag",
    "fractal_peak_count_20",
    "fractal_trough_count_20",
    # trend / vol
    "ema5_slope_atr",
    "ema10_slope_atr",
    "ema20_slope_atr",
    "ema50_slope_atr",
    "ema100_slope_atr",
    "ema5_10_cross",
    "ema10_20_cross",
    "ema20_50_cross",
    "ema5_10_cross_age",
    "ema10_20_cross_age",
    "ma5_10_cross",
    "ma10_20_cross",
    "ma20_50_cross",
    "ma5_10_cross_age",
    "ma10_20_cross_age",
    "ema20_slope_lag1",
    "ema20_slope_lag3",
    "ema20_slope_lag5",
    "ema_div_20_100",
    "ema_div_20_200",
    "price_vs_ema20",
    "price_vs_ema50",
    "price_vs_ema100",
    "price_vs_ma10",
    "price_vs_ma20",
    "price_vs_ma50",
    "adx14",
    "plus_di",
    "minus_di",
    "rsi7",
    "rsi14",
    "rsi21",
    "stoch_k",
    "stoch_d",
    "stoch_j",
    "cci20",
    "atr14",
    "atr_rel",
    "atr_rel_252",
    "true_range",
    "rolling_vol_20",
    "rolling_returns_std",
    "hh_count_10",
    "ll_count_10",
) + MACD_SLOW_FEATURES + HOUR_OH_COLS

MODEL_FEATURE_COLS: Tuple[str, ...] = FEATURE_COLS + (
    "cost_r",
    "tp1_dist_ratio",
    "tp1_over_cost",
    "sl_over_cost",
    "cost_to_sl_dist",
    "tp1_r_over_cost",
    "sl_r_over_cost",
)

PATH_POST_WINDOWS: Tuple[int, ...] = (1, 2, 3, 4, 5)
PATH_FEATURE_COLS: Tuple[str, ...] = (
    "path_pre10_max_up_r",
    "path_pre10_max_down_r",
    "path_pre10_atr_mean",
    "path_pre10_atr_rel_mean",
    "path_pre10_ret_sum",
    "path_pre10_ret_std",
    "path_pre10_consec_up_max",
    "path_pre10_consec_down_max",
    "path_pre10_fib_pos",
) + tuple(
    f"path_post{n}_{suffix}"
    for n in PATH_POST_WINDOWS
    for suffix in (
        "max_up_r",
        "max_down_r",
        "atr_mean",
        "atr_rel_mean",
        "macd_hist_slope_mean",
        "ema5_slope_mean",
        "ema10_slope_mean",
        "consec_up_max",
        "consec_down_max",
        "fib_pos",
    )
)

GATE_FEATURE_COLS: Tuple[str, ...] = tuple(dict.fromkeys(list(MODEL_FEATURE_COLS) + list(PATH_FEATURE_COLS)))

POST_TP1_FEATURES: Tuple[str, ...] = (
    "rsi7",
    "rsi14",
    "rsi21",
    "adx14",
    "plus_di",
    "minus_di",
    "macd12_hist",
    "macd5_hist",
    "rolling_vol_20",
    "rolling_returns_std",
    "price_vs_ema20",
    "price_vs_ema50",
    "price_vs_ema100",
    "ema5_slope_atr",
    "ema10_slope_atr",
    "ema20_slope_atr",
    "seg_peak_atr",
    "ema_div_20_100",
    "ema_div_20_200",
    "stoch_k",
    "stoch_d",
    "stoch_j",
)

# TP2-only feature set (TP1 hit time t_tp1, strictly causal)
TP2_MACD_VARIANTS: Tuple[Tuple[str, str, int, int, int], ...] = (
    ("c12_26_9", "close", 12, 26, 9),
    ("hl2_5_13_5", "hl2", 5, 13, 5),
    ("c24_52_18", "close", 24, 52, 18),
    ("c36_78_27", "close", 36, 78, 27),
)

TP2_MACD_FEATURES: Tuple[str, ...] = tuple(
    f"tp2_macd_{name}_{suffix}"
    for (name, _src, _fast, _slow, _sig) in TP2_MACD_VARIANTS
    for suffix in ("hist", "slope_atr", "slope2_atr")
)

TP2_BASE_FEATURES: Tuple[str, ...] = (
    "atr14",
    "atr_rel",
    "atr_rel_252",
    "bar_range",
    "bar_range_to_tr",
    "wick_ratio",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "consec_up",
    "consec_down",
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "roc_4",
    "roc_8",
    "roc_12",
    "roc_20",
    "range_pos_20",
    "dist_high_20",
    "dist_low_20",
    "ema5_slope_atr",
    "ema10_slope_atr",
    "ema20_slope_atr",
    "ema5_10_cross",
    "ema10_20_cross",
    "ema20_50_cross",
    "ema5_10_cross_age",
    "ema10_20_cross_age",
    "adx14",
    "plus_di",
    "minus_di",
    "rsi7",
    "rsi14",
    "rsi21",
    "stoch_k",
    "stoch_d",
    "stoch_j",
    "cci20",
    "rolling_vol_20",
    "rolling_returns_std",
    "hh_count_10",
    "ll_count_10",
    "fractal_peak_count_20",
    "fractal_trough_count_20",
    "price_vs_ema20",
    "price_vs_ema50",
    "price_vs_ema100",
    "ema_div_20_100",
    "ema_div_20_200",
    "macd12_hist",
    "macd12_hist_slope_atr",
    "macd5_hist",
    "macd5_hist_slope_atr",
)

TP2_FEATURE_COLS: Tuple[str, ...] = (
    TP2_BASE_FEATURES
    + TP2_MACD_FEATURES
    + (
        "tp1_r_dyn",
        "sl_r_dyn",
        "tp1_dist_ratio",
        "hour_sin",
        "hour_cos",
        "vol_regime",
        "trend_regime",
        "internal_regime",
        "session_regime",
    )
)


# =============================
# TP2 post-TP1 序列数据集（019）
# =============================


TP2_SEQ_H2_GRID: Tuple[int, ...] = (240, 360, 480)
TP2_SEQ_H2_MAX: int = int(max(TP2_SEQ_H2_GRID))
TP2_TARGET_EXTRA_R_CLAMP: Tuple[float, float] = (0.20, 1.20)  # in R units vs sl_dist
TP2_TRAIL_NO_TP2: float = 0.60  # trail_stop_px = 0.60 * ATR_ref
TP2_DISABLE_TRAIL_MULT: float = 1e6  # disable trailing while attempting TP2 (BE-only stop)


def _last_seen_value_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(int(x.size), np.nan, dtype=float)
    last = float("nan")
    for i in range(int(x.size)):
        v = float(x[int(i)])
        if np.isfinite(v):
            last = v
        out[int(i)] = last
    return out


def _bars_since_last_finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(int(x.size), np.nan, dtype=float)
    last = -1
    for i in range(int(x.size)):
        v = float(x[int(i)])
        if np.isfinite(v):
            last = int(i)
        out[int(i)] = float(int(i) - int(last)) if int(last) >= 0 else float("nan")
    return out


def _hour_sin_cos(ts: pd.Timestamp) -> Tuple[float, float]:
    h = int(ts.hour) if isinstance(ts, pd.Timestamp) and ts is not pd.NaT else 0
    ang = 2.0 * math.pi * (float(h) / 24.0)
    return float(math.sin(ang)), float(math.cos(ang))


def build_tp2_sequence_dataset_preos(
    *,
    df_prices: pd.DataFrame,
    ctx: Dict[str, np.ndarray],
    regimes: Dict[str, np.ndarray],
    tp2_macd: Dict[str, np.ndarray],
    trades_csv_path: Path,
    pre_start: pd.Timestamp,
    pre_end: pd.Timestamp,
    H2_max: int,
    out_path: Path,
) -> Dict[str, Any]:
    if not trades_csv_path.exists():
        return {"ok": False, "reason": f"missing_trades_csv: {str(trades_csv_path)}"}
    if H2_max <= 0:
        return {"ok": False, "reason": "invalid_H2_max"}

    tr = pd.read_csv(trades_csv_path)
    if tr.empty:
        return {"ok": False, "reason": "empty_trades_csv"}

    tp1_hit = tr.get("tp1_hit")
    tp1_hit_i = tr.get("tp1_hit_i")
    if tp1_hit is None or tp1_hit_i is None:
        return {"ok": False, "reason": "missing_tp1_hit_fields"}

    tr["tp1_hit"] = tp1_hit.astype(bool)
    tr["tp1_hit_i"] = pd.to_numeric(tp1_hit_i, errors="coerce").fillna(-1).astype(int)

    if "tp1_time" in tr.columns:
        tr["tp1_time"] = pd.to_datetime(tr["tp1_time"], utc=True, errors="coerce")
        t1_ts = tr["tp1_time"]
    else:
        idx = tr["tp1_hit_i"].to_numpy(dtype=int)
        t1_ts = pd.to_datetime(df_prices.index.to_series().reset_index(drop=True).reindex(idx).to_numpy(), utc=True, errors="coerce")
        tr["tp1_time"] = t1_ts

    tr = tr[tr["tp1_hit"] & (tr["tp1_hit_i"] >= 0)].copy()
    tr = tr[(tr["tp1_time"] >= pre_start) & (tr["tp1_time"] <= pre_end)].copy()
    if tr.empty:
        return {"ok": False, "reason": "no_preos_tp1_hits"}

    n_bars = int(len(df_prices))
    close = df_prices["close"].to_numpy(dtype=float)
    high = df_prices["high"].to_numpy(dtype=float)
    low = df_prices["low"].to_numpy(dtype=float)

    fh, fl = compute_fractals_confirmed(df_prices)
    last_fh = _last_seen_value_nan(fh)
    last_fl = _last_seen_value_nan(fl)
    fh_dist = _bars_since_last_finite(fh)
    fl_dist = _bars_since_last_finite(fl)
    atr14 = np.asarray(ctx.get("atr14", np.full(n_bars, np.nan)), dtype=float)

    # columns present in 018 backtest trades
    def _col(name: str, default: Any) -> np.ndarray:
        if name not in tr.columns:
            return np.full(int(len(tr)), default)
        return pd.to_numeric(tr[name], errors="coerce").to_numpy()

    trade_id = pd.to_numeric(tr.get("trade_id"), errors="coerce").fillna(-1).astype(int).to_numpy(dtype=int)
    signal_i = pd.to_numeric(tr.get("signal_i"), errors="coerce").fillna(-1).astype(int).to_numpy(dtype=int)
    direction = pd.to_numeric(tr.get("direction"), errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
    entry_price = _col("entry_price", float("nan")).astype(float)
    sl_dist = _col("sl_dist", float("nan")).astype(float)
    tp1_dist = _col("tp1_dist", float("nan")).astype(float)
    # 018 trades uses tp1_r (R units vs sl_dist) instead of tp1_dist (px); recover if needed.
    if "tp1_r" in tr.columns:
        tp1_r = _col("tp1_r", float("nan")).astype(float)
        m = ~np.isfinite(tp1_dist) & np.isfinite(tp1_r) & np.isfinite(sl_dist)
        if np.any(m):
            tp1_dist[m] = tp1_r[m] * sl_dist[m]
    cost_r = _col("cost_r", float("nan")).astype(float)
    exit_i = pd.to_numeric(tr.get("exit_i"), errors="coerce").fillna(-1).astype(int).to_numpy(dtype=int)

    # feature pick list (kept compact; strictly causal per-bar)
    feat_cols = [
        "atr14",
        "atr_rel",
        "atr_rel_252",
        "ema5_slope_atr",
        "ema10_slope_atr",
        "ema20_slope_atr",
        "adx14",
        "plus_di",
        "minus_di",
        "rsi7",
        "rsi14",
        "rsi21",
        "consec_up",
        "consec_down",
        "dist_high_20",
        "dist_low_20",
    ]
    macd_cols = [k for k in tp2_macd.keys() if str(k).startswith("tp2_macd_")]
    # only keep hist/slope features (exclude any unexpected arrays)
    macd_cols = [c for c in macd_cols if any(c.endswith(suf) for suf in ("hist", "slope_atr", "slope2_atr"))]

    rows: List[Dict[str, Any]] = []
    idx_ts = df_prices.index
    for j in range(int(len(tr))):
        t1 = int(tr["tp1_hit_i"].iloc[int(j)])
        if not (0 <= t1 < n_bars):
            continue
        ex = int(exit_i[int(j)])
        end_i = int(min(n_bars - 1, t1 + int(H2_max)))
        if 0 <= ex < n_bars:
            end_i = int(min(end_i, ex))
        if end_i < t1:
            continue

        dir_i = int(direction[int(j)])
        ent = float(entry_price[int(j)])
        tp1d = float(tp1_dist[int(j)])
        sld = float(sl_dist[int(j)])
        if not (np.isfinite(ent) and np.isfinite(tp1d) and np.isfinite(sld) and sld > 1e-12 and int(dir_i) in (-1, 1)):
            continue
        # Stop at BE as soon as it becomes reachable after TP1 (strictly causal; no bars after BE).
        if int(t1) + 1 <= int(end_i):
            cr = float(cost_r[int(j)])
            if np.isfinite(cr):
                be_px = float(ent + float(dir_i) * float(cr) * float(sld))
                if int(dir_i) > 0:
                    be_mask = low[int(t1) + 1 : int(end_i) + 1] <= float(be_px)
                else:
                    be_mask = high[int(t1) + 1 : int(end_i) + 1] >= float(be_px)
                if bool(np.any(be_mask)):
                    be_i = int(int(t1) + 1 + int(np.argmax(be_mask)))
                    end_i = int(min(end_i, be_i))
        if end_i < t1:
            continue
        tp1_price = float(ent + float(dir_i) * float(tp1d))
        best = float(tp1_price)
        for i in range(int(t1), int(end_i) + 1):
            ts = idx_ts[int(i)]
            h_sin, h_cos = _hour_sin_cos(pd.Timestamp(ts))
            if int(dir_i) > 0:
                best = max(best, float(high[int(i)]))
                mfe_r = float(max(0.0, (float(best) - float(tp1_price)) / float(sld)))
            else:
                best = min(best, float(low[int(i)]))
                mfe_r = float(max(0.0, (float(tp1_price) - float(best)) / float(sld)))

            rr_r = float(((float(close[int(i)]) - float(tp1_price)) * float(dir_i)) / float(sld))
            peak_atr = float(atr14[int(i)]) if int(i) < atr14.size else float("nan")
            fp = float(last_fh[int(i)]) if int(i) < last_fh.size else float("nan")
            flv = float(last_fl[int(i)]) if int(i) < last_fl.size else float("nan")
            if np.isfinite(fp) and np.isfinite(peak_atr) and abs(float(peak_atr)) > 1e-12:
                pv_peak = float((float(close[int(i)]) - float(fp)) / float(peak_atr))
            else:
                pv_peak = float("nan")
            if np.isfinite(flv) and np.isfinite(peak_atr) and abs(float(peak_atr)) > 1e-12:
                pv_trough = float((float(close[int(i)]) - float(flv)) / float(peak_atr))
            else:
                pv_trough = float("nan")

            row = {
                "trade_id": int(trade_id[int(j)]) if int(trade_id[int(j)]) >= 0 else int(j),
                "signal_i": int(signal_i[int(j)]),
                "side": str(tr.get("side").iloc[int(j)]) if "side" in tr.columns else ("long" if int(dir_i) > 0 else "short"),
                "direction": int(dir_i),
                "t1_i": int(t1),
                "bar_i": int(i),
                "bar_offset": int(i - int(t1)),
                "bar_time": str(ts),
                "entry_price": float(ent),
                "tp1_price": float(tp1_price),
                "tp1_dist": float(tp1d),
                "sl_dist": float(sld),
                "time_since_tp1": int(i - int(t1)),
                "relative_return_from_t1_r": float(rr_r),
                "max_favorable_excursion_from_t1_r": float(mfe_r),
                "hour_sin": float(h_sin),
                "hour_cos": float(h_cos),
                "session_regime": int(0 if int(pd.Timestamp(ts).hour) <= 7 else (1 if int(pd.Timestamp(ts).hour) <= 15 else 2)),
                "vol_regime": int(regimes.get("vol_regime", np.full(n_bars, -1))[int(i)]),
                "trend_regime": int(regimes.get("trend_regime", np.full(n_bars, -1))[int(i)]),
                "internal_regime": int(regimes.get("internal_regime", np.full(n_bars, -1))[int(i)]),
                "fractal_peak_dist": float(fh_dist[int(i)]) if int(i) < fh_dist.size else float("nan"),
                "fractal_trough_dist": float(fl_dist[int(i)]) if int(i) < fl_dist.size else float("nan"),
                "price_vs_recent_pivot_peak_atr": float(pv_peak),
                "price_vs_recent_pivot_trough_atr": float(pv_trough),
            }

            for c in feat_cols:
                arr = np.asarray(ctx.get(str(c), np.array([], dtype=float)), dtype=float)
                row[str(c)] = float(arr[int(i)]) if int(i) < int(arr.size) and np.isfinite(float(arr[int(i)])) else float("nan")
            for c in macd_cols:
                arr = np.asarray(tp2_macd.get(str(c), np.array([], dtype=float)), dtype=float)
                row[str(c)] = float(arr[int(i)]) if int(i) < int(arr.size) and np.isfinite(float(arr[int(i)])) else float("nan")

            rows.append(row)

    if not rows:
        return {"ok": False, "reason": "no_rows_built"}

    df_out = pd.DataFrame(rows)
    ensure_dir(out_path.parent)
    df_out.to_parquet(out_path, index=False)
    return {"ok": True, "rows": int(len(df_out)), "cols": int(len(df_out.columns)), "path": str(out_path)}


# =============================
# TP2 (019) Deep Optimization helpers
# =============================


def tp2_deep_optimize_019(
    *,
    paths: "Paths",
    time_cfg: "TimeConfig",
    cv_cfg: "CVConfig",
    mdl_cfg: "ModelConfig",
    mkt: "MarketConfig",
    risk: "RiskConfig",
    esc: "ExitSearchConfig",
    df_prices: pd.DataFrame,
    ctx: Dict[str, np.ndarray],
    regimes: Dict[str, np.ndarray],
    tp2_macd: Dict[str, np.ndarray],
    ds_stage1: pd.DataFrame,
    gate_pass: np.ndarray,
    ctx_r0: "FastSimCtx",
    ctx_r1: "FastSimCtx",
    ctx_full: "FastSimCtx",
    risk_fixed: Dict[str, Any],
    risk_trial: "RiskConfig",
) -> Dict[str, Any]:
    """
    Round-019 TP2-only deep optimization:
    - Train TP2 prob/quantile models on preOS TP1-hit sequence features
    - Grid search TP2 policy (successive halving) using preOS only
    - Return selected TP2 policy + scored events (ds_final) for downstream simulate_trading.
    """
    pre0 = to_utc_ts(time_cfg.preos_start_utc)
    pre1 = to_utc_ts(time_cfg.preos_end_utc)
    os0 = to_utc_ts(time_cfg.os_start_utc)

    tp2_seq_path = paths.artifacts_dir / "expanded_tp2_features.parquet"
    if not tp2_seq_path.exists():
        raise FileNotFoundError(f"missing expanded_tp2_features.parquet: {str(tp2_seq_path)}")

    df_seq = pd.read_parquet(tp2_seq_path)
    if df_seq.empty:
        raise RuntimeError("expanded_tp2_features.parquet is empty")

    df_seq["bar_offset"] = pd.to_numeric(df_seq.get("bar_offset"), errors="coerce").fillna(-1).astype(int)
    df_t1 = df_seq[df_seq["bar_offset"] == 0].copy()
    if df_t1.empty:
        raise RuntimeError("expanded_tp2_features.parquet 缺少 bar_offset==0 的 t1 rows")

    df_t1["t1_time"] = pd.to_datetime(df_t1.get("bar_time"), utc=True, errors="coerce")
    df_t1 = df_t1[pd.notna(df_t1["t1_time"])].copy()
    df_t1 = df_t1.sort_values(["t1_i", "trade_id"], kind="mergesort").drop_duplicates(subset=["trade_id"], keep="last")
    df_t1_pre = df_t1[(df_t1["t1_time"] >= pre0) & (df_t1["t1_time"] <= pre1)].copy()
    if df_t1_pre.empty:
        raise RuntimeError("TP2 训练集为空：expanded_tp2_features.parquet 未覆盖 preOS TP1-hit t1 rows")

    # Model features (strictly causal at/after TP1).
    base_cols = [
        "direction",
        "hour_sin",
        "hour_cos",
        "session_regime",
        "vol_regime",
        "trend_regime",
        "internal_regime",
        "fractal_peak_dist",
        "fractal_trough_dist",
        "price_vs_recent_pivot_peak_atr",
        "price_vs_recent_pivot_trough_atr",
        "atr14",
        "atr_rel",
        "atr_rel_252",
        "ema5_slope_atr",
        "ema10_slope_atr",
        "ema20_slope_atr",
        "adx14",
        "plus_di",
        "minus_di",
        "rsi7",
        "rsi14",
        "rsi21",
        "consec_up",
        "consec_down",
        "dist_high_20",
        "dist_low_20",
    ]
    macd_cols = [c for c in df_t1_pre.columns if str(c).startswith("tp2_macd_")]
    tp2_feature_cols = [c for c in (base_cols + macd_cols) if c in df_t1_pre.columns]
    if not tp2_feature_cols:
        raise RuntimeError("TP2 feature cols empty after schema check")

    X_train = df_t1_pre.loc[:, tp2_feature_cols].to_numpy(dtype=float)
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X_train = np.where(np.isfinite(X_train), X_train, med)

    t1_i = pd.to_numeric(df_t1_pre.get("t1_i"), errors="coerce").fillna(-1).astype(int).to_numpy()
    entry_px = pd.to_numeric(df_t1_pre.get("entry_price"), errors="coerce").to_numpy(dtype=float)
    direction = pd.to_numeric(df_t1_pre.get("direction"), errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
    tp1_dist = pd.to_numeric(df_t1_pre.get("tp1_dist"), errors="coerce").to_numpy(dtype=float)
    sl_dist = pd.to_numeric(df_t1_pre.get("sl_dist"), errors="coerce").to_numpy(dtype=float)

    high = df_prices["high"].to_numpy(dtype=float)
    low = df_prices["low"].to_numpy(dtype=float)
    close = df_prices["close"].to_numpy(dtype=float)
    cost_total_px = float(mkt.roundtrip_cost_price) + float(mkt.slippage_buffer_price)

    # =============================
    # 1) Labels from expanded sequence (preOS only)
    # =============================
    df_seq["trade_id"] = pd.to_numeric(df_seq.get("trade_id"), errors="coerce").fillna(-1).astype(int)
    df_seq["bar_offset"] = pd.to_numeric(df_seq.get("bar_offset"), errors="coerce").fillna(-1).astype(int)
    df_seq["max_favorable_excursion_from_t1_r"] = (
        pd.to_numeric(df_seq.get("max_favorable_excursion_from_t1_r"), errors="coerce").fillna(0.0).astype(float)
    )
    df_t1_pre["trade_id"] = pd.to_numeric(df_t1_pre.get("trade_id"), errors="coerce").fillna(-1).astype(int)

    extra_mfe_by_h2: Dict[int, np.ndarray] = {}
    for H2 in TP2_SEQ_H2_GRID:
        g = df_seq[df_seq["bar_offset"] <= int(H2)].groupby("trade_id", sort=False)["max_favorable_excursion_from_t1_r"].max()
        extra = df_t1_pre["trade_id"].map(g).fillna(0.0).to_numpy(dtype=float)
        extra_mfe_by_h2[int(H2)] = np.clip(extra, 0.0, 50.0)

    # =============================
    # 2) TP2 prob model (y: extra_MFE_R>=0.20) + quantile regressors
    # =============================
    from sklearn.metrics import brier_score_loss, roc_auc_score

    gap = int(cv_cfg.purge_bars + cv_cfg.embargo_bars)
    n_splits = max(3, int(cv_cfg.calib_cv_splits))

    prob_oof: Dict[int, np.ndarray] = {}
    prob_meta: Dict[int, Dict[str, Any]] = {}
    prob_model: Dict[int, Any] = {}
    quant_oof: Dict[Tuple[int, float], np.ndarray] = {}
    quant_meta: Dict[Tuple[int, float], Dict[str, Any]] = {}
    quant_model: Dict[Tuple[int, float], Any] = {}

    # grid: include 0.60 for spec, but selection uses 0.40..0.55
    q_grid_all = (0.40, 0.45, 0.50, 0.55, 0.60)

    for H2 in TP2_SEQ_H2_GRID:
        H2i = int(H2)
        extra = np.asarray(extra_mfe_by_h2[H2i], dtype=float)
        y = (extra >= float(TP2_TARGET_EXTRA_R_CLAMP[0])).astype(int)
        exit_i_tp2 = np.clip(np.asarray(t1_i, dtype=int) + int(H2i), 0, max(0, int(len(df_prices)) - 1))

        p_raw_oof, oof_meta = oof_predict_purged_lgbm_custom(
            cv_cfg,
            mdl_cfg,
            X=X_train,
            y=y,
            entry_i=np.asarray(t1_i, dtype=int),
            exit_i=exit_i_tp2.astype(int),
        )
        cal = calibrate_platt_isotonic(p_oof=p_raw_oof, y=y)
        if bool(cal.get("ok", False)):
            cal_best = dict(cal)
            cal_best["best"] = {"method": "sigmoid"}
            p_oof = apply_calibration(p_raw_oof, cal=cal_best)
        else:
            cal_best = {"ok": False, "best": {"method": "none"}, "models": {}}
            p_oof = np.asarray(p_raw_oof, dtype=float)

        auc = float("nan")
        if int(np.unique(y).size) >= 2:
            try:
                auc = float(roc_auc_score(y, p_oof))
            except Exception:
                auc = float("nan")
        brier = float("nan")
        try:
            brier = float(brier_score_loss(y, p_oof))
        except Exception:
            brier = float("nan")
        ece, ece_table = expected_calibration_error(y, p_oof, n_bins=10)

        prob_oof[H2i] = np.asarray(p_oof, dtype=float)
        prob_meta[H2i] = {
            "method": "sigmoid",
            "gap": int(gap),
            "folds": int(oof_meta.get("folds", n_splits)),
            "base_rate": float(np.mean(y)) if y.size else float("nan"),
            "auc_oof": float(auc),
            "brier_oof": float(brier),
            "ece_oof": float(ece),
            "ece_table": ece_table,
            "oof_meta": oof_meta,
        }

        base_model = fit_lgbm_classifier_full(mdl_cfg, X=X_train, y=y)
        prob_model[H2i] = {"base_model": base_model, "calibration": cal_best, "method": "sigmoid"}

        for q in q_grid_all:
            p_q_oof, q_meta = oof_predict_purged_lgbm_quantile(
                cv_cfg,
                mdl_cfg,
                X=X_train,
                y=extra.astype(float),
                entry_i=np.asarray(t1_i, dtype=int),
                exit_i=exit_i_tp2.astype(int),
                alpha=float(q),
            )
            quant_oof[(H2i, float(q))] = np.asarray(p_q_oof, dtype=float)
            quant_meta[(H2i, float(q))] = q_meta
            quant_model[(H2i, float(q))] = fit_lgbm_quantile_full(mdl_cfg, X=X_train, y=extra.astype(float), alpha=float(q))

    # =============================
    # 3) Successive halving (preOS only) on TP2 grid
    # =============================
    t1_time = pd.to_datetime(df_t1_pre["t1_time"], utc=True, errors="coerce")
    stage0_end = to_utc_ts("2017-12-31 23:59:59")
    stage1_end = to_utc_ts("2020-12-31 23:59:59")
    m0 = (t1_time >= pre0) & (t1_time <= stage0_end)
    m1 = (t1_time >= pre0) & (t1_time <= stage1_end)
    m2 = (t1_time >= pre0) & (t1_time <= pre1)

    adx14_src = pd.to_numeric(df_t1_pre.get("adx14"), errors="coerce").to_numpy(dtype=float)
    atr_rel_src = pd.to_numeric(df_t1_pre.get("atr_rel"), errors="coerce").to_numpy(dtype=float)
    slope20_src = pd.to_numeric(df_t1_pre.get("ema20_slope_atr"), errors="coerce").to_numpy(dtype=float)
    rsi14_src = pd.to_numeric(df_t1_pre.get("rsi14"), errors="coerce").to_numpy(dtype=float)
    slope_abs = np.abs(slope20_src)
    adx_hi = float(np.nanquantile(adx14_src, 0.75)) if np.any(np.isfinite(adx14_src)) else 25.0
    adx_lo = float(np.nanquantile(adx14_src, 0.25)) if np.any(np.isfinite(adx14_src)) else 20.0
    slope_hi = float(np.nanquantile(slope_abs, 0.75)) if np.any(np.isfinite(slope_abs)) else 0.5
    slope_lo = float(np.nanquantile(slope_abs, 0.25)) if np.any(np.isfinite(slope_abs)) else 0.2
    atr_hi = float(np.nanquantile(atr_rel_src, 0.75)) if np.any(np.isfinite(atr_rel_src)) else 1.0
    strong_trend = (adx14_src >= float(adx_hi)) & (slope_abs >= float(slope_hi))
    momentum_strong = np.abs(rsi14_src - 50.0) >= 10.0
    sideway = (adx14_src <= float(adx_lo)) | (slope_abs <= float(slope_lo))
    vol_high = atr_rel_src >= float(atr_hi)

    def _extra_target(
        q_pred: np.ndarray,
        *,
        regime_w: str,
        scale_base: float,
        strong_trend_mask: Optional[np.ndarray] = None,
        momentum_strong_mask: Optional[np.ndarray] = None,
        sideway_mask: Optional[np.ndarray] = None,
        vol_high_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        q_pred = np.asarray(q_pred, dtype=float)
        scale = np.full(int(q_pred.size), float(scale_base), dtype=float)
        if str(regime_w) == "trend_only":
            st = strong_trend if strong_trend_mask is None else np.asarray(strong_trend_mask, dtype=bool)
            ms = momentum_strong if momentum_strong_mask is None else np.asarray(momentum_strong_mask, dtype=bool)
            scale *= np.where(st & ms, 1.1, 1.0)
        elif str(regime_w) == "vol_only":
            sw = sideway if sideway_mask is None else np.asarray(sideway_mask, dtype=bool)
            vh = vol_high if vol_high_mask is None else np.asarray(vol_high_mask, dtype=bool)
            scale *= np.where(sw | vh, 0.9, 1.0)
        out = q_pred * scale
        out = np.clip(out, float(TP2_TARGET_EXTRA_R_CLAMP[0]), float(TP2_TARGET_EXTRA_R_CLAMP[1]))
        out[~np.isfinite(out)] = float(TP2_TARGET_EXTRA_R_CLAMP[0])
        return out.astype(float)

    def _eval(mask: np.ndarray, *, H2: int, q: float, thr_prob: float, regime_w: str, scale_base: float) -> Dict[str, Any]:
        mask = np.asarray(mask, dtype=bool)
        p = np.asarray(prob_oof[int(H2)], dtype=float)[mask]
        q_pred = np.asarray(quant_oof[(int(H2), float(q))], dtype=float)[mask]
        extra = np.asarray(extra_mfe_by_h2[int(H2)], dtype=float)[mask]
        n_tp1 = int(p.size)
        if n_tp1 <= 0:
            return {"n_tp1": 0, "n_att": 0, "k": 0, "attempt_rate": float("nan"), "cond_hit": float("nan"), "posterior": 0.0, "score": -1e18}
        att = p >= float(thr_prob)
        n_att = int(np.sum(att))
        extra_tgt = _extra_target(
            q_pred,
            regime_w=str(regime_w),
            scale_base=float(scale_base),
            strong_trend_mask=strong_trend[mask],
            momentum_strong_mask=momentum_strong[mask],
            sideway_mask=sideway[mask],
            vol_high_mask=vol_high[mask],
        )
        hit = extra >= extra_tgt
        k = int(np.sum(hit & att)) if n_att > 0 else 0
        posterior = float(beta_posterior_prob_ge(k, n_att, 0.60)) if n_att > 0 else 0.0
        cond = float(k / max(1, n_att)) if n_att > 0 else float("nan")
        att_rate = float(n_att / max(1, n_tp1))
        extra_mean = float(np.nanmean(extra_tgt[att])) if n_att > 0 else float("nan")
        score = float(att_rate) * float(cond if np.isfinite(cond) else 0.0) * float(extra_mean if np.isfinite(extra_mean) else 0.0)
        return {
            "n_tp1": int(n_tp1),
            "n_att": int(n_att),
            "k": int(k),
            "attempt_rate": float(att_rate),
            "cond_hit": float(cond),
            "posterior": float(posterior),
            "extra_r_mean": float(extra_mean),
            "score": float(score),
        }

    thr_grid = (0.50, 0.55, 0.60)
    q_sel_grid = (0.40, 0.45, 0.50, 0.55)
    regime_grid = ("none", "trend_only", "vol_only")
    scale_grid = (0.8, 0.9, 1.0, 1.1, 1.2)

    key_cols = ["H2", "thresh_prob", "q_target", "regime_weighting", "scale_base"]
    rows0: List[Dict[str, Any]] = []
    for H2 in TP2_SEQ_H2_GRID:
        for thr in thr_grid:
            for q in q_sel_grid:
                for rw in regime_grid:
                    for sc in scale_grid:
                        r0 = _eval(m0.to_numpy(), H2=int(H2), q=float(q), thr_prob=float(thr), regime_w=str(rw), scale_base=float(sc))
                        rows0.append(
                            {
                                "H2": int(H2),
                                "thresh_prob": float(thr),
                                "q_target": float(q),
                                "regime_weighting": str(rw),
                                "scale_base": float(sc),
                                **{f"stage0_{k}": v for k, v in r0.items()},
                            }
                        )
    df_all = pd.DataFrame(rows0)
    df_all = df_all.sort_values("stage0_score", ascending=False).reset_index(drop=True)

    keep0_n = max(60, int(len(df_all) * 0.2))
    df_keep0 = df_all[(df_all["stage0_posterior"] >= 0.80) & (df_all["stage0_n_att"] >= 200)].head(int(keep0_n)).copy()
    if df_keep0.empty:
        df_keep0 = df_all.head(int(keep0_n)).copy()

    rows1: List[Dict[str, Any]] = []
    for r in df_keep0.itertuples(index=False):
        r1 = _eval(
            m1.to_numpy(),
            H2=int(getattr(r, "H2")),
            q=float(getattr(r, "q_target")),
            thr_prob=float(getattr(r, "thresh_prob")),
            regime_w=str(getattr(r, "regime_weighting")),
            scale_base=float(getattr(r, "scale_base")),
        )
        rows1.append({**{k: getattr(r, k) for k in key_cols}, **{f"stage1_{k}": v for k, v in r1.items()}})
    df_stage1 = pd.DataFrame(rows1)
    df_all = df_all.merge(df_stage1, on=key_cols, how="left")

    df_keep1 = df_stage1.sort_values("stage1_score", ascending=False).reset_index(drop=True)
    keep1_n = max(20, int(len(df_keep1) * 0.3))
    df_keep1_ok = df_keep1[(df_keep1["stage1_posterior"] >= 0.80) & (df_keep1["stage1_n_att"] >= 250)].copy()
    df_keep1 = (df_keep1_ok if not df_keep1_ok.empty else df_keep1).head(int(keep1_n)).copy()

    rows2: List[Dict[str, Any]] = []
    for r in df_keep1.itertuples(index=False):
        r2 = _eval(
            m2.to_numpy(),
            H2=int(getattr(r, "H2")),
            q=float(getattr(r, "q_target")),
            thr_prob=float(getattr(r, "thresh_prob")),
            regime_w=str(getattr(r, "regime_weighting")),
            scale_base=float(getattr(r, "scale_base")),
        )
        rows2.append({**{k: getattr(r, k) for k in key_cols}, **{f"stage2_{k}": v for k, v in r2.items()}})
    df_stage2 = pd.DataFrame(rows2)
    df_all = df_all.merge(df_stage2, on=key_cols, how="left")
    df_keep2 = df_stage2.sort_values("stage2_score", ascending=False).reset_index(drop=True)
    if df_keep2.empty:
        raise RuntimeError("TP2 successive halving produced no candidates")

    best = df_keep2.iloc[0].to_dict()
    best_h2 = int(best["H2"])
    best_thr = float(best["thresh_prob"])
    best_q = float(best["q_target"])
    best_rw = str(best["regime_weighting"])
    best_sc = float(best["scale_base"])
    best_post = float(best.get("stage2_posterior", 0.0))

    ensure_dir(paths.artifacts_dir)
    df_all.to_csv(paths.artifacts_dir / "tp2_candidates.csv", index=False)

    # =============================
    # 4) Posterior buckets report (preOS)
    # =============================
    vol_pre = pd.to_numeric(df_t1_pre.get("atr_rel"), errors="coerce").to_numpy(dtype=float)
    trend_pre = pd.to_numeric(df_t1_pre.get("ema20_slope_atr"), errors="coerce").to_numpy(dtype=float)
    vol_q_lo = float(np.nanquantile(vol_pre, float(TP2_VOL_Q_LO))) if np.any(np.isfinite(vol_pre)) else 1.0
    vol_q_hi = float(np.nanquantile(vol_pre, float(TP2_VOL_Q_HI))) if np.any(np.isfinite(vol_pre)) else 1.0
    trend_q_lo = float(np.nanquantile(trend_pre, float(TP2_TREND_Q_LO))) if np.any(np.isfinite(trend_pre)) else 0.0
    trend_q_hi = float(np.nanquantile(trend_pre, float(TP2_TREND_Q_HI))) if np.any(np.isfinite(trend_pre)) else 0.0
    buckets = build_tp2_buckets(df_t1_pre, vol_q_lo=vol_q_lo, vol_q_hi=vol_q_hi, trend_q_lo=trend_q_lo, trend_q_hi=trend_q_hi)

    p_sel = np.asarray(prob_oof[best_h2], dtype=float)
    q_sel = np.asarray(quant_oof[(best_h2, float(best_q))], dtype=float)
    extra_sel = np.asarray(extra_mfe_by_h2[best_h2], dtype=float)
    extra_tgt_sel = _extra_target(q_sel, regime_w=best_rw, scale_base=best_sc)
    att_sel = (p_sel >= float(best_thr)) & (float(best_post) >= 0.80)
    hit_sel = extra_sel >= extra_tgt_sel
    df_bs = pd.DataFrame({"bucket": buckets.astype(str), "attempt": att_sel.astype(int), "hit": (hit_sel & att_sel).astype(int)})
    bucket_rows: List[Dict[str, Any]] = []
    for b, g in df_bs.groupby("bucket", sort=False):
        n = int(len(g))
        n_att = int(g["attempt"].sum())
        k = int(g["hit"].sum())
        cond = float(k / max(1, n_att)) if n_att > 0 else float("nan")
        post = float(beta_posterior_prob_ge(k, n_att, 0.60)) if n_att > 0 else 0.0
        bucket_rows.append({"bucket": str(b), "n_tp1": int(n), "n_att": int(n_att), "k_tp2": int(k), "cond_hit": float(cond), "posterior": float(post)})
    pd.DataFrame(bucket_rows).sort_values(["posterior", "cond_hit"], ascending=False).to_csv(paths.artifacts_dir / "tp2_bucket_stats.csv", index=False)

    # =============================
    # 5) Leakage audit (TP2)
    # =============================
    tp2_leak = leakage_audit_tp2_features(seed=int(cv_cfg.seed), df_full=df_prices, tp1_indices=df_t1_pre["t1_i"].astype(int).to_numpy())
    if not bool(tp2_leak.get("ok", False)) or int(tp2_leak.get("failures_n", 1)) != 0:
        raise RuntimeError(f"tp2 leakage audit failed: failures_n={int(tp2_leak.get('failures_n', 999))}")

    # =============================
    # 6) Deploy selected TP2 policy to full ds_stage1
    # =============================
    ds_final = ds_stage1.copy()
    ds_final["_entry_ts"] = pd.to_datetime(ds_final.get("_entry_ts"), utc=True, errors="coerce")
    if "event_id" not in ds_final.columns:
        ds_final = ds_final.reset_index(drop=True)
        ds_final["event_id"] = np.arange(int(len(ds_final)))

    eid = pd.to_numeric(ds_final.get("event_id"), errors="coerce").fillna(-1).astype(int).to_numpy()
    gate_pass_final = np.zeros(int(len(ds_final)), dtype=bool)
    ok_eid = (eid >= 0) & (eid < int(len(gate_pass)))
    gate_pass_final[ok_eid] = np.asarray(gate_pass, dtype=bool)[eid[ok_eid]]
    ds_final["gate_pass"] = gate_pass_final

    tp1_hit_mask = ds_final.get("tp1_hit").astype(bool).to_numpy() if "tp1_hit" in ds_final.columns else np.zeros(int(len(ds_final)), dtype=bool)
    tp1_i_all = pd.to_numeric(ds_final.get("tp1_hit_i"), errors="coerce").fillna(-1).astype(int).to_numpy(dtype=int)
    tp1_pos = np.where(tp1_hit_mask & gate_pass_final & (tp1_i_all >= 0))[0]

    # feature frame at TP1 for TP1-hit traded events
    feat_tp2 = build_tp2_feature_frame(
        df=df_prices,
        ctx=ctx,
        regimes=regimes,
        tp1_idx=tp1_i_all[tp1_pos],
        tp1_r_dyn=pd.to_numeric(ds_final.get("tp1_r_dyn"), errors="coerce").to_numpy(dtype=float)[tp1_pos],
        sl_r_dyn=pd.to_numeric(ds_final.get("sl_r_dyn"), errors="coerce").to_numpy(dtype=float)[tp1_pos],
        tp1_dist_ratio=pd.to_numeric(ds_final.get("tp1_dist_ratio"), errors="coerce").to_numpy(dtype=float)[tp1_pos],
        macd_extra=tp2_macd,
    )
    feat_tp2["direction"] = pd.to_numeric(ds_final.get("direction"), errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)[tp1_pos]

    fh, fl = compute_fractals_confirmed(df_prices)
    last_fh = _last_seen_value_nan(fh)
    last_fl = _last_seen_value_nan(fl)
    fh_dist = _bars_since_last_finite(fh)
    fl_dist = _bars_since_last_finite(fl)
    atr14_all = np.asarray(ctx.get("atr14", np.full(int(len(df_prices)), np.nan)), dtype=float)
    close_all = df_prices["close"].to_numpy(dtype=float)
    fp = tp1_i_all[tp1_pos]
    feat_tp2["fractal_peak_dist"] = fh_dist[fp].astype(float)
    feat_tp2["fractal_trough_dist"] = fl_dist[fp].astype(float)
    feat_tp2["price_vs_recent_pivot_peak_atr"] = np.where(
        np.isfinite(last_fh[fp]) & np.isfinite(atr14_all[fp]) & (atr14_all[fp] > 1e-12),
        (close_all[fp] - last_fh[fp]) / atr14_all[fp],
        np.nan,
    ).astype(float)
    feat_tp2["price_vs_recent_pivot_trough_atr"] = np.where(
        np.isfinite(last_fl[fp]) & np.isfinite(atr14_all[fp]) & (atr14_all[fp] > 1e-12),
        (close_all[fp] - last_fl[fp]) / atr14_all[fp],
        np.nan,
    ).astype(float)

    for c in tp2_feature_cols:
        if c not in feat_tp2.columns:
            feat_tp2[c] = np.nan
    X_tp2 = feat_tp2.loc[:, tp2_feature_cols].to_numpy(dtype=float)
    X_tp2 = np.where(np.isfinite(X_tp2), X_tp2, med)

    base_prob = prob_model[best_h2]["base_model"]
    cal_prob = prob_model[best_h2]["calibration"]
    p_raw = np.asarray(base_prob.predict_proba(X_tp2)[:, 1], dtype=float)
    p_pred = apply_calibration(p_raw, cal=cal_prob)

    q_mdl = quant_model[(best_h2, float(best_q))]
    q_pred = np.asarray(q_mdl.predict(X_tp2), dtype=float)
    adx_d = pd.to_numeric(feat_tp2.get("adx14"), errors="coerce").to_numpy(dtype=float)
    atr_rel_d = pd.to_numeric(feat_tp2.get("atr_rel"), errors="coerce").to_numpy(dtype=float)
    slope20_d = pd.to_numeric(feat_tp2.get("ema20_slope_atr"), errors="coerce").to_numpy(dtype=float)
    rsi14_d = pd.to_numeric(feat_tp2.get("rsi14"), errors="coerce").to_numpy(dtype=float)
    slope_abs_d = np.abs(slope20_d)
    strong_trend_d = (adx_d >= float(adx_hi)) & (slope_abs_d >= float(slope_hi))
    momentum_strong_d = np.abs(rsi14_d - 50.0) >= 10.0
    sideway_d = (adx_d <= float(adx_lo)) | (slope_abs_d <= float(slope_lo))
    vol_high_d = atr_rel_d >= float(atr_hi)
    extra_tgt = _extra_target(
        q_pred,
        regime_w=best_rw,
        scale_base=best_sc,
        strong_trend_mask=strong_trend_d,
        momentum_strong_mask=momentum_strong_d,
        sideway_mask=sideway_d,
        vol_high_mask=vol_high_d,
    )

    posterior_gate_ok = bool(float(best_post) >= 0.80)
    tp2_attempt = (p_pred >= float(best_thr)) & posterior_gate_ok

    ds_final["tp2_attempt"] = False
    ds_final["tp2_prob"] = np.nan
    ds_final["tp2_extra_r_target"] = np.nan
    ds_final["tp2_h2_policy"] = int(best_h2)

    entry_px_all = pd.to_numeric(ds_final.get("entry_price"), errors="coerce").to_numpy(dtype=float)
    dir_all = pd.to_numeric(ds_final.get("direction"), errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
    atr_ref_all = pd.to_numeric(ds_final.get("atr_ref"), errors="coerce").to_numpy(dtype=float)
    tp1_dist_all = pd.to_numeric(ds_final.get("tp1_dist"), errors="coerce").to_numpy(dtype=float)
    sl_dist_all = pd.to_numeric(ds_final.get("sl_dist"), errors="coerce").to_numpy(dtype=float)
    tp1_r_all = pd.to_numeric(ds_final.get("tp1_r"), errors="coerce").to_numpy(dtype=float)
    cost_r_all = pd.to_numeric(ds_final.get("cost_r"), errors="coerce").to_numpy(dtype=float)
    tp1_close_frac_all = pd.to_numeric(ds_final.get("tp1_close_frac"), errors="coerce").fillna(float(esc.tp1_close_frac_grid[0])).to_numpy(dtype=float)

    for c in ("runner_r", "runner_exit_i", "runner_exit_type", "runner_exit_price", "tail_r", "tail_exit_i", "exit_i", "exit_type", "tp2_hit", "net_r", "runner_cash_r", "tp1_cash_r"):
        if c not in ds_final.columns:
            ds_final[c] = np.nan

    for j, ridx in enumerate(tp1_pos):
        ds_final.at[int(ridx), "tp2_prob"] = float(p_pred[int(j)]) if np.isfinite(float(p_pred[int(j)])) else float("nan")
        ds_final.at[int(ridx), "tp2_extra_r_target"] = float(extra_tgt[int(j)]) if np.isfinite(float(extra_tgt[int(j)])) else float("nan")

        tp1_i = int(tp1_i_all[int(ridx)])
        entry_px = float(entry_px_all[int(ridx)])
        direction_i = int(dir_all[int(ridx)])
        atr_ref = float(atr_ref_all[int(ridx)])
        tp1_dist_px = float(tp1_dist_all[int(ridx)])
        sl_dist_px = float(sl_dist_all[int(ridx)])
        tp1_r_act = float(tp1_r_all[int(ridx)])
        cost_r = float(cost_r_all[int(ridx)])
        tp1_close_frac = float(tp1_close_frac_all[int(ridx)])
        if not (np.isfinite(entry_px) and np.isfinite(atr_ref) and atr_ref > 1e-12 and int(direction_i) in (-1, 1) and np.isfinite(tp1_dist_px) and tp1_dist_px > 1e-12 and np.isfinite(sl_dist_px) and sl_dist_px > 1e-12):
            continue
        end_i = int(min(int(len(df_prices)) - 1, int(tp1_i) + int(best_h2)))
        be_price = float(entry_px + float(direction_i) * float(cost_total_px))

        do_tp2 = bool(tp2_attempt[int(j)])
        ds_final.at[int(ridx), "tp2_attempt"] = bool(do_tp2)
        if do_tp2:
            tp2_price = float(entry_px + float(direction_i) * float(tp1_dist_px + float(extra_tgt[int(j)]) * float(sl_dist_px)))
            trail_stop_px = float(TP2_DISABLE_TRAIL_MULT) * float(atr_ref)
        else:
            tp2_price = float(entry_px + float(direction_i) * float(tp1_dist_px + 9999.0 * float(sl_dist_px)))
            trail_stop_px = float(TP2_TRAIL_NO_TP2) * float(atr_ref)

        runner = runner_after_tp1_dynamic(
            high=high,
            low=low,
            close=close,
            direction=int(direction_i),
            tp1_hit_i=int(tp1_i),
            entry_price=float(entry_px),
            be_price=float(be_price),
            tp2_price=float(tp2_price),
            trail_stop_px=float(trail_stop_px),
            end_i=int(end_i),
            schedule=None,
        )
        tp2_hit = bool(runner.get("tp2_hit", False)) if do_tp2 else False
        runner_exit_type = str(runner.get("runner_exit_type", "NA"))
        runner_exit_i = int(runner.get("runner_exit_i", tp1_i))
        runner_exit_px = float(runner.get("runner_exit_price", be_price))
        if not np.isfinite(runner_exit_px):
            runner_exit_px = float(be_price)

        if tp2_hit:
            runner_r = float(tp1_r_act + float(extra_tgt[int(j)]))
        else:
            rr = float(((runner_exit_px - entry_px) * float(direction_i)) / float(sl_dist_px))
            runner_r = float(np.clip(rr, float(cost_r), float(tp1_r_act + float(extra_tgt[int(j)]) if do_tp2 else tp1_r_act + float(TP2_TARGET_EXTRA_R_CLAMP[1]))))

        net_r = float(tp1_close_frac * tp1_r_act + (1.0 - tp1_close_frac) * runner_r - cost_r)
        ds_final.at[int(ridx), "tp2_hit"] = bool(tp2_hit)
        ds_final.at[int(ridx), "runner_exit_type"] = str(runner_exit_type)
        ds_final.at[int(ridx), "runner_exit_i"] = int(runner_exit_i)
        ds_final.at[int(ridx), "runner_exit_price"] = float(runner_exit_px)
        ds_final.at[int(ridx), "runner_r"] = float(runner_r)
        ds_final.at[int(ridx), "tail_r"] = float(runner_r)
        ds_final.at[int(ridx), "tail_exit_i"] = int(runner_exit_i)
        ds_final.at[int(ridx), "exit_i"] = int(max(tp1_i, runner_exit_i))
        ds_final.at[int(ridx), "exit_type"] = "TP2" if tp2_hit else str(runner_exit_type)
        ds_final.at[int(ridx), "tp1_cash_r"] = float(tp1_close_frac * tp1_r_act)
        ds_final.at[int(ridx), "runner_cash_r"] = float((1.0 - tp1_close_frac) * runner_r - cost_r)
        ds_final.at[int(ridx), "net_r"] = float(net_r)

    # =============================
    # 7) Summary metrics + outputs for downstream pipeline
    # =============================
    # Ensure required scoring columns exist (TP2-only path should not depend on legacy blocks).
    if "signal_i" not in ds_final.columns:
        ds_final["signal_i"] = -1
    if "side" not in ds_final.columns:
        dtmp = pd.to_numeric(ds_final.get("direction"), errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
        ds_final["side"] = np.where(dtmp > 0, "long", "short")
    if "p_score" not in ds_final.columns:
        ds_final["p_score"] = 1.0
    if "p_tail" not in ds_final.columns:
        ds_final["p_tail"] = 0.0
    if "vol_regime" not in ds_final.columns:
        ds_final["vol_regime"] = -1
    if "trend_regime" not in ds_final.columns:
        ds_final["trend_regime"] = -1
    if "mae_r" not in ds_final.columns:
        ds_final["mae_r"] = np.nan
    if "adx14" not in ds_final.columns:
        ds_final["adx14"] = np.nan

    arr = build_scored_event_arrays(ds_final, mkt=mkt)
    pass_indices = np.where(gate_pass_final)[0]
    lot_max = lot_max_for_risk_cap(mkt, sl_dist_risk=arr.sl_dist_risk, risk_cap_usd=float(risk_fixed["risk_cap_usd"]))
    pre_m, os_m, all_m, meta = simulate_trading_fast_metrics(
        ctx_full,
        mkt,
        risk_trial,
        arr=arr,
        pass_indices=pass_indices,
        lot_max_by_ticket=[lot_max],
        daily_stop_loss_usd=float(risk_fixed["daily_stop_loss_usd"]),
        max_parallel_same_dir=1,
        tickets_per_signal=1,
        tp1_close_frac=float(esc.tp1_close_frac_grid[0]),
        cooldown_bars=0,
        with_breakdowns=False,
    )
    pre_dd = float(meta.get("max_dd_usd_preos", float("nan")))

    ref_cfg_path = paths.out_dir / "016_artifacts" / "selected_config.json"
    ref_cfg = json.loads(ref_cfg_path.read_text(encoding="utf-8")) if ref_cfg_path.exists() else {}

    tp2_selected = {
        "name": "tp2_deep_019",
        "tp2_kind": "deep",
        "tp2_thr": float(best_thr),
        "tp2_mult_low": float("nan"),
        "tp2_mult_high": float("nan"),
        "tp2_attempt_rate_pre": float(best.get("stage2_attempt_rate", float("nan"))),
        "pre_tp2_cond_hit": float(best.get("stage2_cond_hit", float("nan"))),
        "posterior_tp2": float(best_post),
        "n_tp1": int(best.get("stage2_n_tp1", 0)),
        "k_tp2": int(best.get("stage2_k", 0)),
        "n_att": int(best.get("stage2_n_att", 0)),
        "extra_r_mean": float(best.get("stage2_extra_r_mean", float("nan"))),
        "H2": int(best_h2),
        "q_target": float(best_q),
        "regime_weighting": str(best_rw),
        "scale_base": float(best_sc),
        "pre_epd": float(pre_m.get("epd", float("nan"))),
        "pre_pf": float(pre_m.get("pf", float("nan"))),
        "pre_ev_r": float(pre_m.get("ev_r", float("nan"))),
        "pre_hit_tp1": float(pre_m.get("hit_tp1", float("nan"))),
        "pre_hit_tp2": float(pre_m.get("hit_tp2", float("nan"))),
        "pre_maxdd_usd": float(pre_dd),
        "os_epd": float(os_m.get("epd", float("nan"))),
        "all_epd": float(all_m.get("epd", float("nan"))),
        "all_ev_r": float(all_m.get("ev_r", float("nan"))),
        "constraints_ok": bool(np.isfinite(pre_dd) and float(pre_dd) <= 100.0 and float(pre_m.get("epd", 0.0)) >= 0.8 and float(pre_m.get("hit_tp1", 0.0)) >= 0.70 and float(pre_m.get("ev_r", -1.0)) >= 0.0),
    }

    best_row = {
        "pre_epd": float(pre_m.get("epd", float("nan"))),
        "pre_pf": float(pre_m.get("pf", float("nan"))),
        "pre_ev_r": float(pre_m.get("ev_r", float("nan"))),
        "pre_hit_tp1": float(pre_m.get("hit_tp1", float("nan"))),
        "pre_hit_tp2": float(pre_m.get("hit_tp2", float("nan"))),
        "pre_maxdd_usd": float(pre_dd),
        "os_epd": float(os_m.get("epd", float("nan"))),
        "all_epd": float(all_m.get("epd", float("nan"))),
        "all_pf": float(all_m.get("pf", float("nan"))),
        "all_ev_r": float(all_m.get("ev_r", float("nan"))),
        "posterior_tp1": float(ref_cfg.get("posterior_tp1", float("nan"))),
        "posterior_tp2": float(best_post),
        "posterior_sl": float(ref_cfg.get("posterior_sl", float("nan"))),
        "entry_delay": int(ref_cfg.get("entry_delay", 0)),
        "confirm_window": int(ref_cfg.get("confirm_window", 0)),
        "fast_abs_ratio": float(ref_cfg.get("fast_abs_ratio", 1.0)),
        "zero_eps_mult": float(ref_cfg.get("zero_eps_mult", 0.0)),
        "H1": int(ref_cfg.get("H1", 0)),
        "H2": int(best_h2),
        "tp1_q": float(ref_cfg.get("tp1_q", float("nan"))),
        "sl_q": float(ref_cfg.get("sl_q", float("nan"))),
        "tp2_q": float(best_q),
        "k_cost": float(ref_cfg.get("k_cost", float("nan"))),
        "risk_cap_usd": float(ref_cfg.get("risk_cap_usd", risk_fixed.get("risk_cap_usd", 0.0))),
        "daily_stop_loss_usd": float(ref_cfg.get("daily_stop_loss_usd", risk_fixed.get("daily_stop_loss_usd", 0.0))),
        "dd_trigger_usd": float(ref_cfg.get("dd_trigger_usd", risk_fixed.get("dd_trigger_usd", 0.0))),
        "dd_stop_cooldown_bars": int(ref_cfg.get("dd_stop_cooldown_bars", risk_fixed.get("dd_stop_cooldown_bars", 0))),
        "risk_scale_min": float(ref_cfg.get("risk_scale_min", risk_fixed.get("risk_scale_min", 0.05))),
        "tp2_n1": int(ref_cfg.get("tp2_n1", 0)),
        "tp2_n2": int(ref_cfg.get("tp2_n2", 0)),
        "state_thr": float(ref_cfg.get("state_thr", float("nan"))),
        "tp2_thresh_prob": float(best_thr),
        "tp2_regime_weighting": str(best_rw),
        "tp2_scale_base": float(best_sc),
        "constraints_ok": bool(tp2_selected.get("constraints_ok")),
        "score": float(tp2_selected.get("pre_ev_r", float("nan"))),
    }

    tp2_calibration = {"selected_H2": int(best_h2), **(prob_meta.get(best_h2) or {})}
    tp2_model_meta = {
        "prob_oof": {str(k): v for k, v in prob_meta.items()},
        "quant_oof": {f"H2={k[0]}_q={k[1]}": v for k, v in quant_meta.items()},
        "selected": {"H2": int(best_h2), "q_target": float(best_q)},
        "regime_thresholds": {"adx_hi": float(adx_hi), "adx_lo": float(adx_lo), "slope_hi": float(slope_hi), "slope_lo": float(slope_lo), "atr_hi": float(atr_hi)},
    }

    tp2_model_obj = {
        "version": "20260112_019",
        "feature_cols": list(tp2_feature_cols),
        "imputer_med": [float(x) for x in np.asarray(med, dtype=float)],
        "H2": int(best_h2),
        "q_target": float(best_q),
        "thresh_prob": float(best_thr),
        "regime_weighting": str(best_rw),
        "scale_base": float(best_sc),
        "posterior_p_ge_0.60": float(best_post),
        "prob_model": prob_model[best_h2],
        "quant_model": quant_model[(best_h2, float(best_q))],
    }

    write_json(
        paths.artifacts_dir / "tp2_policy.json",
        {
            "name": "tp2_deep_019",
            "H2": int(best_h2),
            "thresh_prob": float(best_thr),
            "q_target": float(best_q),
            "regime_weighting": str(best_rw),
            "scale_base": float(best_sc),
            "trail_no_tp2_mult": float(TP2_TRAIL_NO_TP2),
            "trail_tp2_mult": float(TP2_DISABLE_TRAIL_MULT),
            "posterior_P(p>=0.60)": float(best_post),
            "posterior_gate": {"p0": 0.60, "require_P_ge": 0.80},
        },
    )

    return {
        "tp2_selected": tp2_selected,
        "tp2_model_obj": tp2_model_obj,
        "tp2_model_meta": tp2_model_meta,
        "tp2_calibration": tp2_calibration,
        "tp2_leak": tp2_leak,
        "tp2_regime_report": {},
        "tp2_selection_status": "ok" if float(best_post) >= 0.80 else "infeasible",
        "best_row": best_row,
        "best_ds_final": ds_final,
        "best_ds_base": ds_stage1.copy(),
        "best_meta": {"tp2_model_meta": tp2_model_meta, "tp2_calibration": tp2_calibration, "tp2_selected": tp2_selected},
        "best_sig_cfg": None,
        "candidates_rows": [tp2_selected],
    }


# =============================
# TP2 (018) Confidence-Max policy
# =============================


TP2_P0: float = 0.60
TP2_POSTERIOR_TARGET: float = 0.95
TP2_MIN_BUCKET_N: int = 300
TP2_EXTRA_R_GRID: Tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20)
TP2_EXTRA_R_CLIP: Tuple[float, float] = (0.10, 1.20)
# Make mid buckets large enough to satisfy n>=300 in practice (pre-OS only).
TP2_VOL_Q_LO: float = 0.10
TP2_VOL_Q_HI: float = 0.90
TP2_TREND_Q_LO: float = 0.10
TP2_TREND_Q_HI: float = 0.90
# Disable trailing while attempting TP2 (keeps BE-only stop; no negative risk).
TP2_DISABLE_TRAIL_MULT: float = 1e6


def session_name(session_regime: int) -> str:
    if int(session_regime) == 0:
        return "asia"
    if int(session_regime) == 1:
        return "eu"
    if int(session_regime) == 2:
        return "us"
    return "na"


def _vol3(x: float, *, q_lo: float, q_hi: float) -> str:
    if not np.isfinite(float(x)):
        return "mid"
    if float(x) <= float(q_lo):
        return "low"
    if float(x) >= float(q_hi):
        return "high"
    return "mid"


def _trend3(x: float, *, q_lo: float, q_hi: float) -> str:
    if not np.isfinite(float(x)):
        return "flat"
    if float(x) <= float(q_lo):
        return "down"
    if float(x) >= float(q_hi):
        return "up"
    return "flat"


def build_tp2_buckets(
    df_tp1: pd.DataFrame,
    *,
    vol_q_lo: float,
    vol_q_hi: float,
    trend_q_lo: float,
    trend_q_hi: float,
) -> pd.Series:
    direction = pd.to_numeric(df_tp1.get("direction"), errors="coerce").fillna(0).astype(int)
    side = np.where(direction.to_numpy(dtype=int) > 0, "L", "S")
    vol_src = pd.to_numeric(df_tp1.get("atr_rel"), errors="coerce").to_numpy(dtype=float)
    trend_src = pd.to_numeric(df_tp1.get("ema20_slope_atr"), errors="coerce").to_numpy(dtype=float)
    sess_src = pd.to_numeric(df_tp1.get("session_regime"), errors="coerce").fillna(-1).astype(int).to_numpy(dtype=int)

    vol_cat = np.array([_vol3(v, q_lo=vol_q_lo, q_hi=vol_q_hi) for v in vol_src], dtype=object)
    trend_cat = np.array([_trend3(v, q_lo=trend_q_lo, q_hi=trend_q_hi) for v in trend_src], dtype=object)
    sess_cat = np.array([session_name(int(s)) for s in sess_src], dtype=object)
    keys = np.array([f"{s}|{v}|{t}|{ss}" for s, v, t, ss in zip(side, vol_cat, trend_cat, sess_cat)], dtype=object)
    return pd.Series(keys, index=df_tp1.index, name="bucket")


def tp2_extra_mfe_r(
    *,
    high: np.ndarray,
    low: np.ndarray,
    direction: int,
    tp1_hit_i: int,
    entry_price: float,
    tp1_dist: float,
    sl_dist: float,
    H2: int,
    cost_total_px: float,
) -> float:
    """
    extra_MFE_R = (max_favorable_price_after_t1 - price_at_t1) / sl_dist
    computed on (t1+1 .. min(t1+H2, BE_hit-1)) so that reach(extra_R) is equivalent to extra_MFE_R>=extra_R
    under conservative "BE first if same bar" assumption.
    """
    if not (np.isfinite(sl_dist) and float(sl_dist) > 1e-12 and np.isfinite(entry_price) and np.isfinite(tp1_dist)):
        return float("nan")
    start = int(tp1_hit_i + 1)
    end = int(min(len(high) - 1, int(tp1_hit_i) + int(H2)))
    if start > end:
        return 0.0
    tp1_price = float(entry_price + float(direction) * float(tp1_dist))
    be_price = float(entry_price + float(direction) * float(cost_total_px))
    h = high[start : end + 1]
    l = low[start : end + 1]
    if h.size == 0 or l.size == 0:
        return 0.0
    if int(direction) > 0:
        be_mask = l <= float(be_price)
    else:
        be_mask = h >= float(be_price)
    be_idx = int(np.argmax(be_mask)) if bool(np.any(be_mask)) else -1
    # Exclude the BE bar itself (conservative).
    eff_end = int(start + be_idx - 1) if be_idx > 0 else (start - 1 if be_idx == 0 else end)
    if eff_end < start:
        return 0.0
    if int(direction) > 0:
        fav = float(np.nanmax(high[start : eff_end + 1])) - float(tp1_price)
    else:
        fav = float(tp1_price) - float(np.nanmin(low[start : eff_end + 1]))
    if not np.isfinite(fav) or fav <= 0:
        return 0.0
    return float(np.clip(fav / float(sl_dist), 0.0, 50.0))


def purged_year_splits(
    *,
    years: np.ndarray,
    entry_i: np.ndarray,
    exit_i: np.ndarray,
    gap: int,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    years = np.asarray(years, dtype=int)
    entry_i = np.asarray(entry_i, dtype=int)
    exit_i = np.asarray(exit_i, dtype=int)
    uniq = [int(x) for x in sorted(np.unique(years)) if int(x) > 0]
    for y in uniq:
        test_idx = np.where(years == int(y))[0]
        if test_idx.size < 200:
            continue
        test_entry_min = int(np.min(entry_i[test_idx]))
        test_exit_max = int(np.max(exit_i[test_idx]))
        # Remove any train samples whose (entry,exit) window overlaps [test_entry_min-gap, test_exit_max+gap]
        lo = int(test_entry_min - int(gap))
        hi = int(test_exit_max + int(gap))
        train_idx = np.where(years != int(y))[0]
        overlap = (exit_i[train_idx] >= lo) & (entry_i[train_idx] <= hi)
        train_idx = train_idx[~overlap]
        if train_idx.size < 600:
            continue
        yield train_idx, test_idx


def precompute_indicators(df: pd.DataFrame, *, zero_eps: float = 1e-12) -> Dict[str, Any]:
    atr14 = compute_atr14(df)
    adx14, plus_di, minus_di = compute_adx14(df)
    rsi7 = compute_rsi(df["close"], 7)
    rsi14 = compute_rsi(df["close"], 14)
    rsi21 = compute_rsi(df["close"], 21)
    k9, d9, j9 = compute_kdj(df, 9)
    cci20 = compute_cci(df, 20)

    ema5 = ema(df["close"], 5)
    ema10 = ema(df["close"], 10)
    ema20 = ema(df["close"], 20)
    ema50 = ema(df["close"], 50)
    ema100 = ema(df["close"], 100)
    ema200 = ema(df["close"], 200)

    ma5 = df["close"].astype(float).rolling(5, min_periods=5).mean()
    ma10 = df["close"].astype(float).rolling(10, min_periods=10).mean()
    ma20 = df["close"].astype(float).rolling(20, min_periods=20).mean()
    ma50 = df["close"].astype(float).rolling(50, min_periods=50).mean()

    mid = ((df["high"] + df["low"]) / 2.0).astype(float)
    macd12, macd12_sig = compute_macd(df["close"], 12, 26, 9)
    macd5, macd5_sig = compute_macd(mid, 5, 13, 5)
    seg_ctx = compute_macd_segment_context(macd12.to_numpy(dtype=float), zero_eps=float(zero_eps))
    macd5_area, macd5_bars_since = compute_macd5_area_since_last_cross(macd5, macd5_sig)

    macd_slow: Dict[str, pd.Series] = {}
    for fast, slow, signal in MACD_SLOW_GRID:
        m, s = compute_macd(df["close"], int(fast), int(slow), int(signal))
        key = f"macd_slow_{int(fast)}_{int(slow)}_{int(signal)}"
        macd_slow[key] = m
        macd_slow[f"{key}_sig"] = s

    return {
        "atr14": atr14,
        "adx14": adx14,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "rsi7": rsi7,
        "rsi14": rsi14,
        "rsi21": rsi21,
        "stoch_k": k9,
        "stoch_d": d9,
        "stoch_j": j9,
        "cci20": cci20,
        "ema5": ema5,
        "ema10": ema10,
        "ema20": ema20,
        "ema50": ema50,
        "ema100": ema100,
        "ema200": ema200,
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "ma50": ma50,
        "macd12": macd12,
        "macd12_sig": macd12_sig,
        "macd5": macd5,
        "macd5_sig": macd5_sig,
        "zero_eps": float(zero_eps),
        "macd12_prev_opp_abs": seg_ctx["prev_opp_abs"],
        "macd12_prev_opp_extreme_i": seg_ctx["prev_opp_extreme_i"],
        "macd12_seg_len": seg_ctx["seg_len"],
        "macd12_seg_slope": seg_ctx["seg_slope"],
        "macd12_seg_peak_abs": seg_ctx["seg_peak_abs"],
        "macd5_area": macd5_area,
        "macd5_bars_since": macd5_bars_since,
        **macd_slow,
    }


def compute_feature_context(df: pd.DataFrame, ind: Dict[str, Any]) -> Dict[str, np.ndarray]:
    n = int(len(df))
    if n == 0:
        return {}
    open_ = df["open"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()

    atr14 = ind["atr14"].astype(float).to_numpy()
    atr_sma200 = pd.Series(atr14, index=df.index).rolling(window=200, min_periods=200).mean().to_numpy(dtype=float)
    atr_rel = safe_div(atr14, atr_sma200, 1.0)
    atr_sma252 = pd.Series(atr14, index=df.index).rolling(window=252, min_periods=252).mean().to_numpy(dtype=float)
    atr_rel_252 = safe_div(atr14, atr_sma252, 1.0)

    adx14 = ind["adx14"].astype(float).to_numpy()
    plus_di = ind["plus_di"].astype(float).to_numpy()
    minus_di = ind["minus_di"].astype(float).to_numpy()
    rsi7 = ind["rsi7"].astype(float).to_numpy()
    rsi14 = ind["rsi14"].astype(float).to_numpy()
    rsi21 = ind["rsi21"].astype(float).to_numpy()
    stoch_k = ind["stoch_k"].astype(float).to_numpy()
    stoch_d = ind["stoch_d"].astype(float).to_numpy()
    stoch_j = ind["stoch_j"].astype(float).to_numpy()
    cci20 = ind["cci20"].astype(float).to_numpy()

    ema5 = ind["ema5"].astype(float).to_numpy()
    ema10 = ind["ema10"].astype(float).to_numpy()
    ema20 = ind["ema20"].astype(float).to_numpy()
    ema50 = ind["ema50"].astype(float).to_numpy()
    ema100 = ind["ema100"].astype(float).to_numpy()
    ema200 = ind["ema200"].astype(float).to_numpy()
    ma5 = ind["ma5"].astype(float).to_numpy()
    ma10 = ind["ma10"].astype(float).to_numpy()
    ma20 = ind["ma20"].astype(float).to_numpy()
    ma50 = ind["ma50"].astype(float).to_numpy()
    ema5_slope_atr = safe_div(np.diff(ema5, prepend=np.nan), atr14, 0.0)
    ema10_slope_atr = safe_div(np.diff(ema10, prepend=np.nan), atr14, 0.0)
    ema20_slope_atr = safe_div(np.diff(ema20, prepend=np.nan), atr14, 0.0)
    ema50_slope_atr = safe_div(np.diff(ema50, prepend=np.nan), atr14, 0.0)
    ema100_slope_atr = safe_div(np.diff(ema100, prepend=np.nan), atr14, 0.0)
    ema20_slope_lag1 = ema20_slope_atr
    ema20_slope_lag3 = safe_div(ema20 - np.roll(ema20, 3), atr14 * 3.0, 0.0)
    ema20_slope_lag5 = safe_div(ema20 - np.roll(ema20, 5), atr14 * 5.0, 0.0)
    ema_div_20_100 = safe_div(ema20 - ema100, atr14, 0.0)
    ema_div_20_200 = safe_div(ema20 - ema200, atr14, 0.0)
    price_vs_ema20 = safe_div(close - ema20, ema20, 0.0)
    price_vs_ema50 = safe_div(close - ema50, ema50, 0.0)
    price_vs_ema100 = safe_div(close - ema100, ema100, 0.0)
    price_vs_ma10 = safe_div(close - ma10, ma10, 0.0)
    price_vs_ma20 = safe_div(close - ma20, ma20, 0.0)
    price_vs_ma50 = safe_div(close - ma50, ma50, 0.0)

    zero_eps = float(ind.get("zero_eps", 1e-12))
    macd12 = ind["macd12"].astype(float).to_numpy()
    macd12_sig = ind["macd12_sig"].astype(float).to_numpy()
    macd12_hist = macd12 - macd12_sig
    macd12_hist_sign = np.sign(macd12_hist)
    macd_hist_std = (
        pd.Series(macd12_hist, index=df.index)
        .rolling(window=20, min_periods=20)
        .std(ddof=0)
        .to_numpy(dtype=float)
    )
    macd12_hist_z = safe_div(macd12_hist, macd_hist_std, 0.0)
    macd12_hist_burst = np.abs(macd12_hist_z)
    prev_opp_abs = np.asarray(ind["macd12_prev_opp_abs"], dtype=float)
    macd_fast_abs_to_prev_opp_peak = safe_div(np.abs(macd12), prev_opp_abs, 0.0)

    seg_len = np.asarray(ind.get("macd12_seg_len", np.full(n, np.nan, dtype=float)), dtype=float)
    seg_slope = np.asarray(ind.get("macd12_seg_slope", np.full(n, np.nan, dtype=float)), dtype=float)
    seg_peak_abs = np.asarray(ind.get("macd12_seg_peak_abs", np.full(n, np.nan, dtype=float)), dtype=float)
    seg_slope_atr = safe_div(seg_slope, atr14, 0.0)
    seg_peak_atr = safe_div(seg_peak_abs, atr14, 0.0)
    hist_slope_atr = safe_div(np.diff(macd12_hist, prepend=np.nan), atr14, 0.0)
    macd12_hist_slope_atr = hist_slope_atr
    macd12_seg_peak_run = seg_peak_abs

    # cross distance proxy (bars since last MACD5 cross)
    cross_to_entry_bars = np.asarray(ind["macd5_bars_since"], dtype=float)
    macd5 = ind["macd5"].astype(float).to_numpy()
    macd5_sig = ind["macd5_sig"].astype(float).to_numpy()
    macd5_hist = macd5 - macd5_sig
    macd5_hist_slope_atr = safe_div(np.diff(macd5_hist, prepend=np.nan), atr14, 0.0)
    macd5_seg_peak_run = segment_peak_running(macd5, zero_eps=zero_eps)
    macd12_up, macd12_dn = compute_crosses(macd12, macd12_sig)
    macd5_up, macd5_dn = compute_crosses(macd5, macd5_sig)
    macd12_cross = macd12_up | macd12_dn
    macd5_cross = macd5_up | macd5_dn
    last_cross12 = last_true_index(macd12_cross)
    last_cross5 = last_true_index(macd5_cross)
    macd12_cross_age = np.where(last_cross12 >= 0, np.arange(n) - last_cross12, np.nan).astype(float)
    macd5_cross_age = np.where(last_cross5 >= 0, np.arange(n) - last_cross5, np.nan).astype(float)

    macd_slow_features: Dict[str, np.ndarray] = {}
    for fast, slow, signal in MACD_SLOW_GRID:
        key = f"macd_slow_{int(fast)}_{int(slow)}_{int(signal)}"
        macd_s = ind.get(key)
        sig_s = ind.get(f"{key}_sig")
        if macd_s is None or sig_s is None:
            continue
        m = np.asarray(macd_s, dtype=float)
        s = np.asarray(sig_s, dtype=float)
        hist = m - s
        up_s, dn_s = compute_crosses(m, s)
        cross_s = up_s | dn_s
        last_cross_s = last_true_index(cross_s)
        cross_age_s = np.where(last_cross_s >= 0, np.arange(n) - last_cross_s, np.nan).astype(float)
        hist_slope_s = safe_div(np.diff(hist, prepend=np.nan), atr14, 0.0)
        seg_peak_run_s = segment_peak_running(m, zero_eps=zero_eps)
        macd_slow_features[f"{key}_hist"] = hist
        macd_slow_features[f"{key}_cross_age"] = cross_age_s
        macd_slow_features[f"{key}_hist_slope_atr"] = hist_slope_s
        macd_slow_features[f"{key}_seg_peak_run"] = seg_peak_run_s

    ema5_gt_ema10 = ema5 > ema10
    ema10_gt_ema20 = ema10 > ema20
    ema20_gt_ema50 = ema20 > ema50
    ema5_10_cross = ema5_gt_ema10.astype(float)
    ema10_20_cross = ema10_gt_ema20.astype(float)
    ema20_50_cross = ema20_gt_ema50.astype(float)
    ema5_10_event = ema5_gt_ema10 != np.roll(ema5_gt_ema10, 1)
    ema10_20_event = ema10_gt_ema20 != np.roll(ema10_gt_ema20, 1)
    ema5_10_event[0] = False
    ema10_20_event[0] = False
    last_ema5_10 = last_true_index(ema5_10_event)
    last_ema10_20 = last_true_index(ema10_20_event)
    ema5_10_cross_age = np.where(last_ema5_10 >= 0, np.arange(n) - last_ema5_10, np.nan).astype(float)
    ema10_20_cross_age = np.where(last_ema10_20 >= 0, np.arange(n) - last_ema10_20, np.nan).astype(float)

    ma5_gt_ma10 = ma5 > ma10
    ma10_gt_ma20 = ma10 > ma20
    ma20_gt_ma50 = ma20 > ma50
    ma5_10_cross = ma5_gt_ma10.astype(float)
    ma10_20_cross = ma10_gt_ma20.astype(float)
    ma20_50_cross = ma20_gt_ma50.astype(float)
    ma5_10_event = ma5_gt_ma10 != np.roll(ma5_gt_ma10, 1)
    ma10_20_event = ma10_gt_ma20 != np.roll(ma10_gt_ma20, 1)
    ma5_10_event[0] = False
    ma10_20_event[0] = False
    last_ma5_10 = last_true_index(ma5_10_event)
    last_ma10_20 = last_true_index(ma10_20_event)
    ma5_10_cross_age = np.where(last_ma5_10 >= 0, np.arange(n) - last_ma5_10, np.nan).astype(float)
    ma10_20_cross_age = np.where(last_ma10_20 >= 0, np.arange(n) - last_ma10_20, np.nan).astype(float)

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    true_range = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

    def _ret(k: int) -> np.ndarray:
        r = safe_div(close, np.roll(close, int(k)), 1.0) - 1.0
        r[: int(k)] = 0.0
        return r

    ret_1 = _ret(1)
    ret_3 = _ret(3)
    ret_6 = _ret(6)
    ret_12 = _ret(12)
    close_s = pd.Series(close, index=df.index)
    roc_4 = compute_roc(close_s, 4).to_numpy(dtype=float)
    roc_8 = compute_roc(close_s, 8).to_numpy(dtype=float)
    roc_12 = compute_roc(close_s, 12).to_numpy(dtype=float)
    roc_20 = compute_roc(close_s, 20).to_numpy(dtype=float)

    rolling_vol_20 = (
        pd.Series(ret_1, index=df.index)
        .rolling(window=20, min_periods=20)
        .std(ddof=0)
        .to_numpy(dtype=float)
    )
    rolling_returns_std = (
        pd.Series(ret_1, index=df.index)
        .rolling(window=20, min_periods=20)
        .std(ddof=0)
        .to_numpy(dtype=float)
    )

    rp_n = 20
    roll_high = pd.Series(high, index=df.index).rolling(window=rp_n, min_periods=rp_n).max().to_numpy(dtype=float)
    roll_low = pd.Series(low, index=df.index).rolling(window=rp_n, min_periods=rp_n).min().to_numpy(dtype=float)
    range_pos_20 = safe_div(close - roll_low, roll_high - roll_low, 0.5)
    range_pos_20 = np.clip(range_pos_20, 0.0, 1.0)
    dist_high_20 = safe_div(roll_high - close, roll_high - roll_low, 0.0)
    dist_low_20 = safe_div(close - roll_low, roll_high - roll_low, 0.0)

    prev_roll_high = pd.Series(high, index=df.index).rolling(window=rp_n, min_periods=rp_n).max().shift(1).to_numpy(dtype=float)
    prev_roll_low = pd.Series(low, index=df.index).rolling(window=rp_n, min_periods=rp_n).min().shift(1).to_numpy(dtype=float)
    pivot_dist_high = safe_div(prev_roll_high - close, atr14, 0.0)
    pivot_dist_low = safe_div(close - prev_roll_low, atr14, 0.0)
    breakout_up = close > (prev_roll_high + 0.5 * atr14)
    breakout_dn = close < (prev_roll_low - 0.5 * atr14)
    breakout_flag = np.where(breakout_up, 1.0, np.where(breakout_dn, -1.0, 0.0))

    # causal fractal-like counts (swing highs/lows within rolling window)
    win_fr = 5
    swing_high = (high >= pd.Series(high, index=df.index).rolling(window=win_fr, min_periods=win_fr).max()).astype(int)
    swing_low = (low <= pd.Series(low, index=df.index).rolling(window=win_fr, min_periods=win_fr).min()).astype(int)
    fractal_peak_count_20 = pd.Series(swing_high, index=df.index).rolling(window=20, min_periods=20).sum().to_numpy(dtype=float)
    fractal_trough_count_20 = pd.Series(swing_low, index=df.index).rolling(window=20, min_periods=20).sum().to_numpy(dtype=float)

    bar_range = np.maximum(0.0, high - low)
    candle_range = np.maximum(1e-12, bar_range)
    candle_body = np.abs(close - open_)
    body_ratio = safe_div(candle_body, candle_range, 0.0)
    wick_ratio = np.clip(1.0 - body_ratio, 0.0, 1.0)
    bar_range_to_tr = safe_div(bar_range, true_range, 0.0)

    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    upper_wick_ratio = safe_div(upper_wick, candle_body, 0.0)
    lower_wick_ratio = safe_div(lower_wick, candle_body, 0.0)

    mom_1 = close - prev_close
    mom_1_diff = mom_1 - np.roll(mom_1, 1)
    mom_1_diff2 = mom_1_diff - np.roll(mom_1_diff, 1)
    mom_1_diff[:1] = 0.0
    mom_1_diff2[:2] = 0.0

    consec_up = np.zeros(int(n), dtype=float)
    consec_down = np.zeros(int(n), dtype=float)
    up_move = close > prev_close
    down_move = close < prev_close
    up_cnt = 0
    dn_cnt = 0
    for i in range(int(n)):
        if bool(up_move[int(i)]):
            up_cnt += 1
        else:
            up_cnt = 0
        if bool(down_move[int(i)]):
            dn_cnt += 1
        else:
            dn_cnt = 0
        consec_up[int(i)] = float(up_cnt)
        consec_down[int(i)] = float(dn_cnt)

    prev_open = np.roll(open_, 1)
    prev_close2 = np.roll(close, 1)
    prev_open[0] = open_[0]
    prev_close2[0] = close[0]
    bullish_engulfing = (prev_close2 < prev_open) & (close > open_) & (close >= prev_open) & (open_ <= prev_close2)
    bearish_engulfing = (prev_close2 > prev_open) & (close < open_) & (open_ >= prev_close2) & (close <= prev_open)
    hammer = (lower_wick_ratio >= 2.0) & (upper_wick_ratio <= 0.5)
    bullish_engulfing[:1] = False
    bearish_engulfing[:1] = False
    hammer[:1] = False

    hh = (high > np.roll(high, 1)).astype(int)
    ll = (low < np.roll(low, 1)).astype(int)
    hh[:1] = 0
    ll[:1] = 0
    hh_count_10 = pd.Series(hh, index=df.index).rolling(window=10, min_periods=10).sum().to_numpy(dtype=float)
    ll_count_10 = pd.Series(ll, index=df.index).rolling(window=10, min_periods=10).sum().to_numpy(dtype=float)

    return {
        "macd_fast_abs_to_prev_opp_peak": macd_fast_abs_to_prev_opp_peak,
        "seg_len": seg_len,
        "seg_slope_atr": seg_slope_atr,
        "seg_peak_atr": seg_peak_atr,
        "hist_slope_atr": hist_slope_atr,
        "macd12_hist": macd12_hist,
        "macd12_hist_sign": macd12_hist_sign,
        "macd12_hist_z": macd12_hist_z,
        "macd12_hist_burst": macd12_hist_burst,
        "macd12_cross_age": macd12_cross_age,
        "macd12_hist_slope_atr": macd12_hist_slope_atr,
        "macd12_seg_peak_run": macd12_seg_peak_run,
        "macd5_hist": macd5_hist,
        "macd5_cross_age": macd5_cross_age,
        "macd5_hist_slope_atr": macd5_hist_slope_atr,
        "macd5_seg_peak_run": macd5_seg_peak_run,
        "cross_to_entry_bars": cross_to_entry_bars,
        "ret_1": ret_1,
        "ret_3": ret_3,
        "ret_6": ret_6,
        "ret_12": ret_12,
        "roc_4": roc_4,
        "roc_8": roc_8,
        "roc_12": roc_12,
        "roc_20": roc_20,
        "true_range": true_range,
        "rolling_vol_20": rolling_vol_20,
        "rolling_returns_std": rolling_returns_std,
        "range_pos_20": range_pos_20,
        "dist_high_20": dist_high_20,
        "dist_low_20": dist_low_20,
        "bar_range": bar_range,
        "bar_range_to_tr": bar_range_to_tr,
        "wick_ratio": wick_ratio,
        "body_ratio": body_ratio,
        "upper_wick_ratio": upper_wick_ratio,
        "lower_wick_ratio": lower_wick_ratio,
        "mom_1": mom_1,
        "mom_1_diff": mom_1_diff,
        "mom_1_diff2": mom_1_diff2,
        "consec_up": consec_up,
        "consec_down": consec_down,
        "bearish_engulfing": bearish_engulfing.astype(float),
        "bullish_engulfing": bullish_engulfing.astype(float),
        "hammer": hammer.astype(float),
        "ema5_slope_atr": ema5_slope_atr,
        "ema10_slope_atr": ema10_slope_atr,
        "ema20_slope_atr": ema20_slope_atr,
        "ema50_slope_atr": ema50_slope_atr,
        "ema100_slope_atr": ema100_slope_atr,
        "ema5_10_cross": ema5_10_cross,
        "ema10_20_cross": ema10_20_cross,
        "ema20_50_cross": ema20_50_cross,
        "ema5_10_cross_age": ema5_10_cross_age,
        "ema10_20_cross_age": ema10_20_cross_age,
        "ma5_10_cross": ma5_10_cross,
        "ma10_20_cross": ma10_20_cross,
        "ma20_50_cross": ma20_50_cross,
        "ma5_10_cross_age": ma5_10_cross_age,
        "ma10_20_cross_age": ma10_20_cross_age,
        "ema20_slope_lag1": ema20_slope_lag1,
        "ema20_slope_lag3": ema20_slope_lag3,
        "ema20_slope_lag5": ema20_slope_lag5,
        "ema_div_20_100": ema_div_20_100,
        "ema_div_20_200": ema_div_20_200,
        "price_vs_ema20": price_vs_ema20,
        "price_vs_ema50": price_vs_ema50,
        "price_vs_ema100": price_vs_ema100,
        "price_vs_ma10": price_vs_ma10,
        "price_vs_ma20": price_vs_ma20,
        "price_vs_ma50": price_vs_ma50,
        "adx14": adx14,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "rsi7": rsi7,
        "rsi14": rsi14,
        "rsi21": rsi21,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "stoch_j": stoch_j,
        "cci20": cci20,
        "atr14": atr14,
        "atr_rel": atr_rel,
        "atr_rel_252": atr_rel_252,
        "pivot_dist_high": pivot_dist_high,
        "pivot_dist_low": pivot_dist_low,
        "breakout_flag": breakout_flag,
        "fractal_peak_count_20": fractal_peak_count_20,
        "fractal_trough_count_20": fractal_trough_count_20,
        "hh_count_10": hh_count_10,
        "ll_count_10": ll_count_10,
        **macd_slow_features,
    }


# =============================
# Leakage audit (truncation)
# =============================


def leakage_audit_by_truncation(
    *,
    seed: int,
    df_full: pd.DataFrame,
    feature_cols: Sequence[str],
    sample_n: int = 10,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    n = int(len(df_full))
    if n < 5000:
        return {"ok": False, "reason": "df_too_small", "n": n, "failures": 1}
    # avoid warmup area and ensure truncation possible
    sample_idx = rng.choice(np.arange(1500, n - 500), size=min(sample_n, n - 2500), replace=False)
    sample_idx = sorted([int(x) for x in sample_idx])

    ind_full = precompute_indicators(df_full)
    ctx_full = compute_feature_context(df_full, ind_full)
    regimes_full = build_regimes_2x2(df_full, ctx_full, window_bars=252)

    failures: List[Dict[str, Any]] = []
    for idx in sample_idx:
        df_trunc = df_full.iloc[: idx + 1].copy()
        ind_trunc = precompute_indicators(df_trunc)
        ctx_trunc = compute_feature_context(df_trunc, ind_trunc)
        regimes_trunc = build_regimes_2x2(df_trunc, ctx_trunc, window_bars=252)
        for c in feature_cols:
            if str(c).startswith("hour_") and len(str(c)) == 7:
                try:
                    h = int(str(c).split("_")[1])
                    a = 1.0 if int(df_full.index[int(idx)].hour) == h else 0.0
                    b = 1.0 if int(df_trunc.index[int(idx)].hour) == h else 0.0
                except Exception:
                    a = float("nan")
                    b = float("nan")
            else:
                a = float(ctx_full[c][idx]) if idx < len(ctx_full.get(c, [])) else float("nan")
                b = float(ctx_trunc[c][idx]) if idx < len(ctx_trunc.get(c, [])) else float("nan")
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            if abs(a - b) > 1e-9:
                failures.append({"idx": int(idx), "feature": str(c), "full": float(a), "trunc": float(b)})
        # regimes are part of gating; must be causal too
        for reg_name in ("vol_regime", "trend_regime"):
            a = int(regimes_full.get(reg_name, np.array([], dtype=int))[idx]) if idx < len(regimes_full.get(reg_name, [])) else -999
            b = int(regimes_trunc.get(reg_name, np.array([], dtype=int))[idx]) if idx < len(regimes_trunc.get(reg_name, [])) else -999
            if a != b:
                failures.append({"idx": int(idx), "feature": str(reg_name), "full": int(a), "trunc": int(b)})
    return {
        "ok": len(failures) == 0,
        "sample_n": int(len(sample_idx)),
        "sample_indices": sample_idx,
        "features_tested": list(feature_cols),
        "regimes_tested": ["vol_regime", "trend_regime"],
        "failures": failures[:50],
        "failures_n": int(len(failures)),
    }


def leakage_audit_tp2_features(
    *,
    seed: int,
    df_full: pd.DataFrame,
    tp1_indices: np.ndarray,
    sample_n: int = 8,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    idx = np.asarray(tp1_indices, dtype=int)
    ok_idx = idx[(idx >= 1500) & (idx < int(len(df_full)) - 10)]
    if ok_idx.size == 0:
        return {"ok": False, "reason": "no_valid_tp1_indices", "failures_n": 1, "failures": []}
    sample = rng.choice(ok_idx, size=min(int(sample_n), int(ok_idx.size)), replace=False)
    sample = sorted([int(x) for x in sample])

    ind_full = precompute_indicators(df_full)
    ctx_full = compute_feature_context(df_full, ind_full)
    regimes_full = build_regimes_2x2(df_full, ctx_full, window_bars=252)
    macd_full = compute_tp2_macd_features(df_full, ctx=ctx_full)

    audit_cols = [c for c in TP2_FEATURE_COLS if c not in ("tp1_r_dyn", "sl_r_dyn", "tp1_dist_ratio")]

    failures: List[Dict[str, Any]] = []
    for i in sample:
        df_trunc = df_full.iloc[: int(i) + 1].copy()
        ind_trunc = precompute_indicators(df_trunc)
        ctx_trunc = compute_feature_context(df_trunc, ind_trunc)
        regimes_trunc = build_regimes_2x2(df_trunc, ctx_trunc, window_bars=252)
        macd_trunc = compute_tp2_macd_features(df_trunc, ctx=ctx_trunc)

        feats_full = build_tp2_feature_frame(
            df=df_full,
            ctx=ctx_full,
            regimes=regimes_full,
            tp1_idx=np.array([int(i)]),
            tp1_r_dyn=np.array([0.0]),
            sl_r_dyn=np.array([0.0]),
            tp1_dist_ratio=np.array([0.0]),
            macd_extra=macd_full,
        )
        feats_trunc = build_tp2_feature_frame(
            df=df_trunc,
            ctx=ctx_trunc,
            regimes=regimes_trunc,
            tp1_idx=np.array([int(i)]),
            tp1_r_dyn=np.array([0.0]),
            sl_r_dyn=np.array([0.0]),
            tp1_dist_ratio=np.array([0.0]),
            macd_extra=macd_trunc,
        )
        for c in audit_cols:
            if c not in feats_full.columns or c not in feats_trunc.columns:
                continue
            a = float(feats_full.at[0, c]) if np.isfinite(float(feats_full.at[0, c])) else float("nan")
            b = float(feats_trunc.at[0, c]) if np.isfinite(float(feats_trunc.at[0, c])) else float("nan")
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            if abs(a - b) > 1e-9:
                failures.append({"idx": int(i), "feature": str(c), "full": float(a), "trunc": float(b)})
    return {
        "ok": len(failures) == 0,
        "sample_n": int(len(sample)),
        "sample_indices": sample,
        "features_tested": audit_cols,
        "failures": failures[:50],
        "failures_n": int(len(failures)),
    }


# =============================
# Regimes (2x2, causal)
# =============================


def rolling_quantile_shifted(x: pd.Series, window: int, q: float) -> pd.Series:
    return x.rolling(window=window, min_periods=window).quantile(q).shift(1)


def build_regimes_2x2(df: pd.DataFrame, ctx: Dict[str, np.ndarray], window_bars: int = 252) -> Dict[str, np.ndarray]:
    atr_rel = pd.Series(ctx.get("atr_rel_252", ctx["atr_rel"]), index=df.index).astype(float)
    adx14 = pd.Series(ctx["adx14"], index=df.index).astype(float)
    ema_div = pd.Series(ctx.get("ema_div_20_100", np.zeros(len(df))), index=df.index).astype(float)

    q_vol = rolling_quantile_shifted(atr_rel, int(window_bars), 0.5).to_numpy(dtype=float)
    vol = atr_rel.to_numpy(dtype=float)
    vol_reg = np.full(int(len(df)), -1, dtype=int)
    ok = np.isfinite(vol) & np.isfinite(q_vol)
    vol_reg[ok & (vol <= q_vol)] = 0
    vol_reg[ok & (vol > q_vol)] = 1

    tr = adx14.to_numpy(dtype=float)
    tr_reg = np.full(int(len(df)), -1, dtype=int)
    tr_reg[np.isfinite(tr) & (tr < 25.0)] = 0
    tr_reg[np.isfinite(tr) & (tr >= 25.0)] = 1

    internal = ema_div.to_numpy(dtype=float)
    internal_reg = np.full(int(len(df)), -1, dtype=int)
    internal_reg[np.isfinite(internal) & (internal < 0.0)] = 0
    internal_reg[np.isfinite(internal) & (internal >= 0.0)] = 1
    return {"vol_regime": vol_reg, "trend_regime": tr_reg, "internal_regime": internal_reg}


# =============================
# Mode4 event generation (fixed)
# =============================


def generate_mode4_events(sig_cfg: Mode4SignalConfig, *, df: pd.DataFrame, ind: Dict[str, Any]) -> pd.DataFrame:
    open_ = df["open"].to_numpy(dtype=float)
    high_ = df["high"].to_numpy(dtype=float)
    low_ = df["low"].to_numpy(dtype=float)
    close_ = df["close"].to_numpy(dtype=float)
    n = int(len(df))

    macd12 = ind["macd12"].astype(float).to_numpy()
    macd12_sig = ind["macd12_sig"].astype(float).to_numpy()
    macd5 = ind["macd5"].astype(float).to_numpy()
    macd5_sig = ind["macd5_sig"].astype(float).to_numpy()
    atr14 = ind["atr14"].astype(float).to_numpy()
    rsi14 = ind["rsi14"].astype(float).to_numpy()
    macd12_hist = macd12 - macd12_sig

    up12, dn12 = compute_crosses(macd12, macd12_sig)
    up5, dn5 = compute_crosses(macd5, macd5_sig)
    dn12_above0 = dn12 & (macd12 > 0)
    up12_below0 = up12 & (macd12 < 0)

    last_dn5 = last_true_index(dn5)
    last_up5 = last_true_index(up5)

    prev_opp_abs = np.asarray(ind["macd12_prev_opp_abs"], dtype=float)
    prev_opp_extreme_i = np.asarray(ind["macd12_prev_opp_extreme_i"], dtype=int)

    warmup = int(max(26, 13, 9, 5) * 5)
    warmup = max(int(sig_cfg.warmup_min_bars), warmup)
    warmup = min(warmup, max(200, n - 2))

    def _window_slice(end_i: int) -> slice:
        w = int(sig_cfg.near_window_bars)
        start = int(max(0, end_i - w + 1))
        return slice(start, int(end_i) + 1)

    def _safe_min(arr: np.ndarray, s: slice) -> float:
        if s.start is None or s.stop is None or s.stop <= s.start:
            return float("nan")
        return float(np.nanmin(arr[s]))

    def _safe_max(arr: np.ndarray, s: slice) -> float:
        if s.start is None or s.stop is None or s.stop <= s.start:
            return float("nan")
        return float(np.nanmax(arr[s]))

    r_thresh = float(sig_cfg.fast_abs_ratio)
    confirm_w = int(max(0, int(sig_cfg.confirm_window)))
    entry_delay = int(max(0, int(sig_cfg.entry_delay)))
    confirm_long = (rsi14 >= 50.0) & (macd12_hist > 0.0)
    confirm_short = (rsi14 <= 50.0) & (macd12_hist < 0.0)

    def _first_confirm(mask: np.ndarray, start_i: int, window: int) -> int:
        if int(start_i) < 0 or int(start_i) >= int(n):
            return -1
        end_i = int(min(n - 2, start_i + max(0, int(window))))
        for j in range(int(start_i), int(end_i) + 1):
            if bool(mask[int(j)]):
                return int(j)
        return -1

    rows: List[Dict[str, Any]] = []
    for i in range(1, n - 1):
        if i < warmup:
            continue

        # short: zero-above dead cross, abs fast smaller than prev opposite segment abs-peak
        if bool(dn12_above0[i]):
            prev_abs_peak = float(prev_opp_abs[i])
            prev_ext_i = int(prev_opp_extreme_i[i])
            current_abs = abs(float(macd12[i])) if np.isfinite(macd12[i]) else float("nan")
            if np.isfinite(prev_abs_peak) and np.isfinite(current_abs) and current_abs <= float(r_thresh) * prev_abs_peak and prev_ext_i >= 0:
                j5 = int(last_dn5[i])
                if j5 < 0 or (i - j5) > int(sig_cfg.sl_cross_lookback_bars):
                    j5 = -1
                if j5 > prev_ext_i and prev_ext_i >= 0:
                    low_between = float(np.nanmin(low_[int(prev_ext_i) : int(j5) + 1])) if int(j5) >= int(prev_ext_i) else float("nan")
                    ref_high = _safe_max(high_, _window_slice(j5))
                    rng = float(ref_high - low_between) if (np.isfinite(ref_high) and np.isfinite(low_between)) else float("nan")
                    if np.isfinite(rng) and rng > 0:
                        ci = _first_confirm(confirm_short, int(i), int(confirm_w))
                        if int(ci) >= 0:
                            entry_i = int(ci + 1 + int(entry_delay))
                            if 0 <= int(entry_i) < int(n):
                                rows.append(
                                    {
                                        "side": "short",
                                        "direction": -1,
                                        "signal_i": int(i),
                                        "confirm_i": int(ci),
                                        "entry_i": int(entry_i),
                                    }
                                )

        # long: zero-below golden cross, abs fast larger than prev opposite segment abs-peak
        if bool(up12_below0[i]):
            prev_abs_peak = float(prev_opp_abs[i])
            prev_ext_i = int(prev_opp_extreme_i[i])
            current_abs = abs(float(macd12[i])) if np.isfinite(macd12[i]) else float("nan")
            if np.isfinite(prev_abs_peak) and np.isfinite(current_abs) and current_abs >= float(r_thresh) * prev_abs_peak and prev_ext_i >= 0:
                j5 = int(last_up5[i])
                if j5 < 0 or (i - j5) > int(sig_cfg.sl_cross_lookback_bars):
                    j5 = -1
                if j5 > prev_ext_i and prev_ext_i >= 0:
                    high_between = float(np.nanmax(high_[int(prev_ext_i) : int(j5) + 1])) if int(j5) >= int(prev_ext_i) else float("nan")
                    ref_low = _safe_min(low_, _window_slice(j5))
                    rng = float(high_between - ref_low) if (np.isfinite(high_between) and np.isfinite(ref_low)) else float("nan")
                    if np.isfinite(rng) and rng > 0:
                        ci = _first_confirm(confirm_long, int(i), int(confirm_w))
                        if int(ci) >= 0:
                            entry_i = int(ci + 1 + int(entry_delay))
                            if 0 <= int(entry_i) < int(n):
                                rows.append(
                                    {
                                        "side": "long",
                                        "direction": 1,
                                        "signal_i": int(i),
                                        "confirm_i": int(ci),
                                        "entry_i": int(entry_i),
                                    }
                                )

    ev = pd.DataFrame(rows)
    if ev.empty:
        return ev

    sig_i = ev["signal_i"].astype(int).to_numpy()
    confirm_i = ev["confirm_i"].astype(int).to_numpy()
    entry_i = ev["entry_i"].astype(int).to_numpy()
    ev["signal_time"] = df.index[sig_i].astype(str)
    ev["_signal_ts"] = pd.to_datetime(df.index[sig_i], utc=True, errors="coerce")
    ev["confirm_time"] = df.index[confirm_i].astype(str)
    ev["_confirm_ts"] = pd.to_datetime(df.index[confirm_i], utc=True, errors="coerce")
    ev["entry_time"] = df.index[entry_i].astype(str)
    ev["_entry_ts"] = pd.to_datetime(df.index[entry_i], utc=True, errors="coerce")
    ev["entry_price"] = open_[entry_i].astype(float)
    ev["atr_ref"] = atr14[sig_i].astype(float)  # causal: ATR on signal bar
    ev["entry_delay"] = int(sig_cfg.entry_delay)
    ev["confirm_window"] = int(sig_cfg.confirm_window)
    ev["fast_abs_ratio"] = float(sig_cfg.fast_abs_ratio)
    ev["zero_eps_mult"] = float(sig_cfg.zero_eps_mult)
    return ev.reset_index(drop=True)


def attach_event_features(ev: pd.DataFrame, *, df: pd.DataFrame, ctx: Dict[str, np.ndarray]) -> pd.DataFrame:
    if ev.empty:
        return ev.copy()
    out = ev.copy()
    entry_i = out["entry_i"].astype(int).to_numpy()

    for c in FEATURE_COLS:
        if c in HOUR_OH_COLS:
            continue
        if c not in ctx:
            continue
        out[c] = np.asarray(ctx[c], dtype=float)[entry_i]

    # UTC hour one-hot (event-level)
    entry_ts = pd.to_datetime(out["_entry_ts"], utc=True, errors="coerce")
    hours = entry_ts.dt.hour.fillna(-1).astype(int)
    for h in range(24):
        out[f"hour_{h:02d}"] = (hours == int(h)).astype(int)
    session_regime = np.full(int(len(out)), -1, dtype=int)
    session_regime[(hours >= 0) & (hours <= 7)] = 0
    session_regime[(hours >= 8) & (hours <= 15)] = 1
    session_regime[(hours >= 16) & (hours <= 23)] = 2
    out["session_regime"] = session_regime

    return out


def compute_path_features(
    df: pd.DataFrame,
    *,
    ctx: Dict[str, np.ndarray],
    ev: pd.DataFrame,
    pre_window: int = 10,
    post_windows: Sequence[int] = PATH_POST_WINDOWS,
    fib_window: int = 20,
) -> pd.DataFrame:
    if ev.empty:
        return pd.DataFrame(index=ev.index, columns=list(PATH_FEATURE_COLS))
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr14 = np.asarray(ctx.get("atr14", np.full(len(df), np.nan)), dtype=float)
    atr_rel = np.asarray(ctx.get("atr_rel", np.full(len(df), np.nan)), dtype=float)
    macd_hist_slope = np.asarray(ctx.get("macd12_hist_slope_atr", ctx.get("hist_slope_atr", np.full(len(df), np.nan))), dtype=float)
    ema5_slope = np.asarray(ctx.get("ema5_slope_atr", np.full(len(df), np.nan)), dtype=float)
    ema10_slope = np.asarray(ctx.get("ema10_slope_atr", np.full(len(df), np.nan)), dtype=float)
    consec_up = np.asarray(ctx.get("consec_up", np.full(len(df), np.nan)), dtype=float)
    consec_down = np.asarray(ctx.get("consec_down", np.full(len(df), np.nan)), dtype=float)

    entry_i = pd.to_numeric(ev.get("entry_i"), errors="coerce").fillna(-1).astype(int).to_numpy()
    entry_px = pd.to_numeric(ev.get("entry_price"), errors="coerce").to_numpy(dtype=float)
    atr_ref = pd.to_numeric(ev.get("atr_ref"), errors="coerce").to_numpy(dtype=float)
    n_bars = int(len(df))

    out_rows: List[Dict[str, Any]] = []
    for idx in range(int(len(ev))):
        ei = int(entry_i[idx])
        if not (0 <= ei < n_bars):
            out_rows.append({c: float("nan") for c in PATH_FEATURE_COLS})
            continue
        ref_px = float(entry_px[idx]) if np.isfinite(entry_px[idx]) else float(close[ei])
        sl_ref = float(atr_ref[idx]) if np.isfinite(atr_ref[idx]) and atr_ref[idx] > 1e-12 else float(atr14[ei])
        if not np.isfinite(sl_ref) or sl_ref <= 1e-12:
            sl_ref = float(np.nanmedian(atr14)) if np.isfinite(np.nanmedian(atr14)) else 1.0

        pre_start = max(0, int(ei) - int(pre_window))
        pre_end = int(ei)
        if pre_end <= pre_start:
            pre_max_up = float("nan")
            pre_max_dn = float("nan")
            pre_atr_mean = float("nan")
            pre_atr_rel_mean = float("nan")
            pre_ret_sum = float("nan")
            pre_ret_std = float("nan")
            pre_consec_up_max = float("nan")
            pre_consec_down_max = float("nan")
            fib_low = float("nan")
            fib_high = float("nan")
        else:
            pre_high = high[pre_start:pre_end]
            pre_low = low[pre_start:pre_end]
            pre_max_up = float((np.nanmax(pre_high) - ref_px) / sl_ref) if pre_high.size else float("nan")
            pre_max_dn = float((ref_px - np.nanmin(pre_low)) / sl_ref) if pre_low.size else float("nan")
            pre_atr_mean = float(np.nanmean(atr14[pre_start:pre_end]) / sl_ref) if pre_high.size else float("nan")
            pre_atr_rel_mean = float(np.nanmean(atr_rel[pre_start:pre_end])) if pre_high.size else float("nan")
            pre_close = close[pre_start:pre_end]
            if pre_close.size > 1:
                pre_ret = np.diff(pre_close) / pre_close[:-1]
                pre_ret_sum = float(np.nanmean(pre_ret))
                pre_ret_std = float(np.nanstd(pre_ret))
            else:
                pre_ret_sum = float("nan")
                pre_ret_std = float("nan")
            pre_consec_up_max = float(np.nanmax(consec_up[pre_start:pre_end])) if pre_high.size else float("nan")
            pre_consec_down_max = float(np.nanmax(consec_down[pre_start:pre_end])) if pre_high.size else float("nan")
            fib_start = max(0, int(ei) - int(fib_window))
            fib_high = float(np.nanmax(high[fib_start:pre_end])) if pre_end > fib_start else float("nan")
            fib_low = float(np.nanmin(low[fib_start:pre_end])) if pre_end > fib_start else float("nan")

        fib_range = float(fib_high - fib_low) if np.isfinite(fib_high) and np.isfinite(fib_low) else float("nan")
        pre_fib_pos = float("nan")
        if np.isfinite(fib_range) and fib_range > 1e-12:
            pre_fib_pos = float(np.clip((ref_px - fib_low) / fib_range, 0.0, 1.0))

        row: Dict[str, Any] = {
            "path_pre10_max_up_r": pre_max_up,
            "path_pre10_max_down_r": pre_max_dn,
            "path_pre10_atr_mean": pre_atr_mean,
            "path_pre10_atr_rel_mean": pre_atr_rel_mean,
            "path_pre10_ret_sum": pre_ret_sum,
            "path_pre10_ret_std": pre_ret_std,
            "path_pre10_consec_up_max": pre_consec_up_max,
            "path_pre10_consec_down_max": pre_consec_down_max,
            "path_pre10_fib_pos": pre_fib_pos,
        }

        for n in post_windows:
            post_start = int(ei)
            post_end = int(min(n_bars - 1, int(ei) + int(n) - 1))
            if post_end < post_start:
                row[f"path_post{n}_max_up_r"] = float("nan")
                row[f"path_post{n}_max_down_r"] = float("nan")
                row[f"path_post{n}_atr_mean"] = float("nan")
                row[f"path_post{n}_atr_rel_mean"] = float("nan")
                row[f"path_post{n}_macd_hist_slope_mean"] = float("nan")
                row[f"path_post{n}_ema5_slope_mean"] = float("nan")
                row[f"path_post{n}_ema10_slope_mean"] = float("nan")
                row[f"path_post{n}_consec_up_max"] = float("nan")
                row[f"path_post{n}_consec_down_max"] = float("nan")
                row[f"path_post{n}_fib_pos"] = float("nan")
                continue
            post_high = high[post_start : post_end + 1]
            post_low = low[post_start : post_end + 1]
            row[f"path_post{n}_max_up_r"] = float((np.nanmax(post_high) - ref_px) / sl_ref) if post_high.size else float("nan")
            row[f"path_post{n}_max_down_r"] = float((ref_px - np.nanmin(post_low)) / sl_ref) if post_low.size else float("nan")
            row[f"path_post{n}_atr_mean"] = float(np.nanmean(atr14[post_start : post_end + 1]) / sl_ref) if post_high.size else float("nan")
            row[f"path_post{n}_atr_rel_mean"] = float(np.nanmean(atr_rel[post_start : post_end + 1])) if post_high.size else float("nan")
            row[f"path_post{n}_macd_hist_slope_mean"] = float(np.nanmean(macd_hist_slope[post_start : post_end + 1])) if post_high.size else float("nan")
            row[f"path_post{n}_ema5_slope_mean"] = float(np.nanmean(ema5_slope[post_start : post_end + 1])) if post_high.size else float("nan")
            row[f"path_post{n}_ema10_slope_mean"] = float(np.nanmean(ema10_slope[post_start : post_end + 1])) if post_high.size else float("nan")
            row[f"path_post{n}_consec_up_max"] = float(np.nanmax(consec_up[post_start : post_end + 1])) if post_high.size else float("nan")
            row[f"path_post{n}_consec_down_max"] = float(np.nanmax(consec_down[post_start : post_end + 1])) if post_high.size else float("nan")
            if np.isfinite(fib_range) and fib_range > 1e-12:
                row[f"path_post{n}_fib_pos"] = float(np.clip((float(close[post_end]) - fib_low) / fib_range, 0.0, 1.0))
            else:
                row[f"path_post{n}_fib_pos"] = float("nan")
        out_rows.append(row)

    return pd.DataFrame(out_rows, index=ev.index)


def _tp2_take(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    idx = np.asarray(idx, dtype=int)
    out = np.full(int(idx.size), np.nan, dtype=float)
    ok = (idx >= 0) & (idx < int(arr.size))
    if np.any(ok):
        out[ok] = arr[idx[ok]]
    return out


def compute_tp2_macd_features(df: pd.DataFrame, *, ctx: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    atr14 = np.asarray(ctx.get("atr14", np.full(len(df), np.nan)), dtype=float)
    hl2 = ((df["high"] + df["low"]) / 2.0).astype(float)
    out: Dict[str, np.ndarray] = {}
    for name, src, fast, slow, sig in TP2_MACD_VARIANTS:
        series = df["close"] if str(src) == "close" else hl2
        macd, macd_sig = compute_macd(series, int(fast), int(slow), int(sig))
        hist = (macd - macd_sig).astype(float).to_numpy()
        slope = np.diff(hist, prepend=np.nan)
        slope2 = np.diff(slope, prepend=np.nan)
        out[f"tp2_macd_{name}_hist"] = hist
        out[f"tp2_macd_{name}_slope_atr"] = safe_div(slope, atr14, 0.0)
        out[f"tp2_macd_{name}_slope2_atr"] = safe_div(slope2, atr14, 0.0)
    return out


def build_tp2_feature_frame(
    *,
    df: pd.DataFrame,
    ctx: Dict[str, np.ndarray],
    regimes: Dict[str, np.ndarray],
    tp1_idx: np.ndarray,
    tp1_r_dyn: np.ndarray,
    sl_r_dyn: np.ndarray,
    tp1_dist_ratio: np.ndarray,
    macd_extra: Dict[str, np.ndarray],
) -> pd.DataFrame:
    idx = np.asarray(tp1_idx, dtype=int)
    out: Dict[str, np.ndarray] = {}
    for c in TP2_BASE_FEATURES:
        if c in ctx:
            out[c] = _tp2_take(np.asarray(ctx[c], dtype=float), idx)
    for k, v in (macd_extra or {}).items():
        out[str(k)] = _tp2_take(np.asarray(v, dtype=float), idx)
    out["tp1_r_dyn"] = np.asarray(tp1_r_dyn, dtype=float)
    out["sl_r_dyn"] = np.asarray(sl_r_dyn, dtype=float)
    out["tp1_dist_ratio"] = np.asarray(tp1_dist_ratio, dtype=float)

    idx_clip = np.clip(idx, 0, max(0, int(len(df)) - 1))
    tp1_ts = pd.to_datetime(df.index[idx_clip], utc=True, errors="coerce")
    tp1_ts = tp1_ts.where((idx >= 0) & (idx < int(len(df))), pd.NaT)
    hours = tp1_ts.hour.fillna(-1).astype(int).to_numpy()
    out["hour_sin"] = np.sin(2.0 * np.pi * (hours / 24.0))
    out["hour_cos"] = np.cos(2.0 * np.pi * (hours / 24.0))

    out["vol_regime"] = _tp2_take(np.asarray(regimes.get("vol_regime", np.full(len(df), -1)), dtype=float), idx)
    out["trend_regime"] = _tp2_take(np.asarray(regimes.get("trend_regime", np.full(len(df), -1)), dtype=float), idx)
    out["internal_regime"] = _tp2_take(np.asarray(regimes.get("internal_regime", np.full(len(df), -1)), dtype=float), idx)
    session_regime = np.full(int(len(hours)), -1, dtype=int)
    session_regime[(hours >= 0) & (hours <= 7)] = 0
    session_regime[(hours >= 8) & (hours <= 15)] = 1
    session_regime[(hours >= 16) & (hours <= 23)] = 2
    out["session_regime"] = session_regime.astype(float)

    feat = pd.DataFrame(out)
    return feat


def expected_calibration_error(y: np.ndarray, p: np.ndarray, *, n_bins: int = 10) -> Tuple[float, List[Dict[str, Any]]]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    if int(np.sum(ok)) == 0:
        return float("nan"), []
    y = y[ok]
    p = p[ok]
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    rows: List[Dict[str, Any]] = []
    for i in range(int(n_bins)):
        lo = float(bins[i])
        hi = float(bins[i + 1])
        m = (p >= lo) & (p < hi) if i < int(n_bins) - 1 else (p >= lo) & (p <= hi)
        n = int(np.sum(m))
        if n == 0:
            rows.append({"bin": int(i), "lo": lo, "hi": hi, "n": 0, "p_mean": float("nan"), "y_mean": float("nan"), "gap": float("nan")})
            continue
        p_mean = float(np.mean(p[m]))
        y_mean = float(np.mean(y[m]))
        gap = abs(p_mean - y_mean)
        ece += float(n) / max(1.0, float(len(y))) * float(gap)
        rows.append({"bin": int(i), "lo": lo, "hi": hi, "n": int(n), "p_mean": float(p_mean), "y_mean": float(y_mean), "gap": float(gap)})
    return float(ece), rows


def build_tp2_regime_report(
    df: pd.DataFrame,
    *,
    pre_start: pd.Timestamp,
    pre_end: pd.Timestamp,
    os_start: pd.Timestamp,
) -> Dict[str, Any]:
    if df.empty:
        return {"pre": {"by_year": [], "by_regime": []}, "os": {"by_year": [], "by_regime": []}, "all": {"by_year": [], "by_regime": []}}

    def _tp2_stats(d: pd.DataFrame) -> Dict[str, Any]:
        tp1_hit = d["tp1_hit"].astype(bool).to_numpy()
        tp2_hit = d["tp2_hit"].astype(bool).to_numpy()
        n = int(len(d))
        n_tp1 = int(np.sum(tp1_hit))
        k_tp2 = int(np.sum(tp2_hit & tp1_hit))
        tp2_cond = float(k_tp2 / max(1, n_tp1)) if n_tp1 > 0 else float("nan")
        post = float(beta_posterior_prob_ge(k_tp2, n_tp1, 0.60)) if n_tp1 > 0 else float("nan")
        return {
            "n": int(n),
            "n_tp1": int(n_tp1),
            "k_tp2": int(k_tp2),
            "tp2_cond_hit": float(tp2_cond),
            "posterior_p_ge_0.60": float(post),
            "hit_tp1": float(np.mean(tp1_hit)) if n else float("nan"),
            "hit_tp2": float(np.mean(tp2_hit)) if n else float("nan"),
        }

    def _by_year(d: pd.DataFrame) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if d.empty:
            return rows
        d = d.copy()
        d["_entry_ts"] = pd.to_datetime(d["_entry_ts"], utc=True, errors="coerce")
        d["year"] = d["_entry_ts"].dt.year
        for year, g in d.groupby("year"):
            r = _tp2_stats(g)
            r["year"] = int(year)
            rows.append(r)
        return rows

    def _by_regime(d: pd.DataFrame) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if d.empty:
            return rows
        for key, g in d.groupby(["vol_regime", "trend_regime", "session_regime"], dropna=False):
            r = _tp2_stats(g)
            r["vol_regime"] = int(key[0]) if key[0] == key[0] else -1
            r["trend_regime"] = int(key[1]) if key[1] == key[1] else -1
            r["session_regime"] = int(key[2]) if key[2] == key[2] else -1
            rows.append(r)
        return rows

    df = df.copy()
    df["_entry_ts"] = pd.to_datetime(df["_entry_ts"], utc=True, errors="coerce")
    pre = df[(df["_entry_ts"] >= pre_start) & (df["_entry_ts"] <= pre_end)].copy()
    os = df[df["_entry_ts"] >= os_start].copy()

    return {
        "pre": {"by_year": _by_year(pre), "by_regime": _by_regime(pre)},
        "os": {"by_year": _by_year(os), "by_regime": _by_regime(os)},
        "all": {"by_year": _by_year(df), "by_regime": _by_regime(df)},
    }


def macd_zero_cross_fib_stats(
    df: pd.DataFrame,
    *,
    macd: np.ndarray,
    sig: np.ndarray,
    start_i: int,
    end_i: int,
    fib_window: int = 20,
    lookahead: int = 48,
    fib_ratio: float = 0.618,
) -> Dict[str, Any]:
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    macd = np.asarray(macd, dtype=float)
    sig = np.asarray(sig, dtype=float)
    up, dn = compute_crosses(macd, sig)
    bull_mask = up & (macd < 0.0)
    bear_mask = dn & (macd > 0.0)
    n = int(len(close))

    def _count(mask: np.ndarray, *, direction: int) -> Tuple[int, int]:
        idxs = np.where(mask)[0]
        idxs = idxs[(idxs >= int(start_i)) & (idxs <= int(end_i))]
        total = 0
        hit = 0
        for i in idxs:
            i = int(i)
            pre_start = max(0, i - int(fib_window))
            if pre_start >= i:
                continue
            pre_high = float(np.nanmax(high[pre_start:i]))
            pre_low = float(np.nanmin(low[pre_start:i]))
            rng = float(pre_high - pre_low)
            if not np.isfinite(rng) or rng <= 1e-12:
                continue
            thr = float(rng * float(fib_ratio))
            end = int(min(n - 1, i + int(lookahead)))
            if end <= i:
                continue
            total += 1
            if int(direction) > 0:
                moved = float(np.nanmax(high[i : end + 1]) - float(close[i]))
            else:
                moved = float(float(close[i]) - np.nanmin(low[i : end + 1]))
            if np.isfinite(moved) and moved >= float(thr):
                hit += 1
        return total, hit

    bull_n, bull_hit = _count(bull_mask, direction=1)
    bear_n, bear_hit = _count(bear_mask, direction=-1)
    return {
        "bull_total": int(bull_n),
        "bull_hit": int(bull_hit),
        "bull_p": float(bull_hit / max(1, bull_n)),
        "bear_total": int(bear_n),
        "bear_hit": int(bear_hit),
        "bear_p": float(bear_hit / max(1, bear_n)),
        "fib_ratio": float(fib_ratio),
        "lookahead": int(lookahead),
        "fib_window": int(fib_window),
    }


# =============================
# Exit / runner (TP1 partial + BE(cost) + TP2)
# =============================


@dataclass(frozen=True)
class TPSLH:
    H: int
    tp1_atr_mult: float
    sl_atr_mult: float


@dataclass(frozen=True)
class ExitConfig:
    entry: str  # "signal_close" | "next_open"
    tpslh: TPSLH
    tp1_close_frac: float  # must be < 1.0 in this round
    tp2_mult: float


def triple_barrier_tp1sl(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    direction: int,
    entry_i: int,
    scan_start_i: int,
    tp1_price: float,
    sl_price: float,
    H: int,
) -> Dict[str, Any]:
    end_i = int(min(len(close) - 1, entry_i + int(H)))
    h = high[scan_start_i : end_i + 1]
    l = low[scan_start_i : end_i + 1]
    if h.size == 0:
        return {"tp1_hit": False, "tp1_hit_i": -1, "exit_i_base": entry_i, "exit_type_base": "TIME", "exit_price_time": float(close[entry_i])}

    if int(direction) > 0:
        tp1_mask = h >= float(tp1_price)
        sl_mask = l <= float(sl_price)
    else:
        tp1_mask = l <= float(tp1_price)
        sl_mask = h >= float(sl_price)

    tp1_idx = int(np.argmax(tp1_mask)) if bool(np.any(tp1_mask)) else -1
    sl_idx = int(np.argmax(sl_mask)) if bool(np.any(sl_mask)) else -1

    if tp1_idx >= 0 and (sl_idx < 0 or tp1_idx < sl_idx):
        tp1_hit_i = int(scan_start_i + tp1_idx)
        return {
            "tp1_hit": True,
            "tp1_hit_i": int(tp1_hit_i),
            "exit_i_base": int(tp1_hit_i),
            "exit_type_base": "TP1",
            "exit_price_time": float(close[end_i]),
        }
    if sl_idx >= 0:
        sl_hit_i = int(scan_start_i + sl_idx)
        return {
            "tp1_hit": False,
            "tp1_hit_i": -1,
            "exit_i_base": int(sl_hit_i),
            "exit_type_base": "SL",
            "exit_price_time": float(close[end_i]),
        }
    return {
        "tp1_hit": False,
        "tp1_hit_i": -1,
        "exit_i_base": int(end_i),
        "exit_type_base": "TIME",
        "exit_price_time": float(close[end_i]),
    }


def runner_after_tp1(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    direction: int,
    tp1_hit_i: int,
    entry_price: float,
    be_price: float,
    tp2_price: float,
    end_i: int,
) -> Dict[str, Any]:
    start = int(tp1_hit_i + 1)
    if start > int(end_i):
        return {"tp2_hit": False, "runner_exit_type": "BE", "runner_exit_i": int(tp1_hit_i), "runner_exit_price": float(be_price)}
    h = high[start : int(end_i) + 1]
    l = low[start : int(end_i) + 1]
    c_end = float(close[int(end_i)])

    if int(direction) > 0:
        tp2_mask = h >= float(tp2_price)
        be_mask = l <= float(be_price)
    else:
        tp2_mask = l <= float(tp2_price)
        be_mask = h >= float(be_price)

    tp2_idx = int(np.argmax(tp2_mask)) if bool(np.any(tp2_mask)) else -1
    be_idx = int(np.argmax(be_mask)) if bool(np.any(be_mask)) else -1

    # Conservative: if same bar, BE first
    if be_idx >= 0 and (tp2_idx < 0 or be_idx <= tp2_idx):
        return {"tp2_hit": False, "runner_exit_type": "BE", "runner_exit_i": int(start + be_idx), "runner_exit_price": float(be_price)}
    if tp2_idx >= 0:
        return {"tp2_hit": True, "runner_exit_type": "TP2", "runner_exit_i": int(start + tp2_idx), "runner_exit_price": float(tp2_price)}
    return {"tp2_hit": False, "runner_exit_type": "TIME", "runner_exit_i": int(end_i), "runner_exit_price": float(c_end)}


def runner_after_tp1_dynamic(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    direction: int,
    tp1_hit_i: int,
    entry_price: float,
    be_price: float,
    tp2_price: float,
    trail_stop_px: float,
    end_i: int,
    schedule: Optional[Sequence[Tuple[int, float, float]]] = None,
) -> Dict[str, Any]:
    start = int(tp1_hit_i + 1)
    if start > int(end_i):
        return {"tp2_hit": False, "runner_exit_type": "BE", "runner_exit_i": int(tp1_hit_i), "runner_exit_price": float(be_price)}
    best = float(entry_price)
    stop_price = float(be_price)
    sched: List[Tuple[int, float, float]] = []
    if schedule:
        for off, tp2_p, trail_p in schedule:
            if int(off) < 0 or not np.isfinite(float(tp2_p)) or not np.isfinite(float(trail_p)):
                continue
            sched.append((int(off), float(tp2_p), float(trail_p)))
        sched.sort(key=lambda x: int(x[0]))
    cur_tp2 = float(tp2_price)
    cur_trail_px = float(trail_stop_px)
    next_k = 0
    if sched:
        cur_tp2 = float(sched[0][1])
        cur_trail_px = float(sched[0][2])
        next_k = 1
    for i in range(int(start), int(end_i) + 1):
        if sched:
            offset = int(i - int(tp1_hit_i))
            while int(next_k) < int(len(sched)) and int(offset) >= int(sched[int(next_k)][0]):
                cur_tp2 = float(sched[int(next_k)][1])
                cur_trail_px = float(sched[int(next_k)][2])
                next_k += 1
        if int(direction) > 0:
            best = max(best, float(high[int(i)]))
            trail_stop = float(best - float(cur_trail_px))
            stop_price = max(float(be_price), float(trail_stop))
            if float(low[int(i)]) <= float(stop_price):
                return {"tp2_hit": False, "runner_exit_type": "TRAIL", "runner_exit_i": int(i), "runner_exit_price": float(stop_price)}
            if float(high[int(i)]) >= float(cur_tp2):
                return {"tp2_hit": True, "runner_exit_type": "TP2", "runner_exit_i": int(i), "runner_exit_price": float(cur_tp2)}
        else:
            best = min(best, float(low[int(i)]))
            trail_stop = float(best + float(cur_trail_px))
            stop_price = min(float(be_price), float(trail_stop))
            if float(high[int(i)]) >= float(stop_price):
                return {"tp2_hit": False, "runner_exit_type": "TRAIL", "runner_exit_i": int(i), "runner_exit_price": float(stop_price)}
            if float(low[int(i)]) <= float(cur_tp2):
                return {"tp2_hit": True, "runner_exit_type": "TP2", "runner_exit_i": int(i), "runner_exit_price": float(cur_tp2)}
    return {"tp2_hit": False, "runner_exit_type": "TIME", "runner_exit_i": int(end_i), "runner_exit_price": float(close[int(end_i)])}


def additional_mfe_after_tp1(
    *,
    high: np.ndarray,
    low: np.ndarray,
    direction: int,
    entry_price: float,
    sl_dist_base: float,
    tp1_hit_i: int,
    H2: int,
) -> float:
    start = int(tp1_hit_i + 1)
    end = int(min(len(high) - 1, int(tp1_hit_i) + int(H2)))
    if start > end or not np.isfinite(sl_dist_base) or float(sl_dist_base) <= 1e-12:
        return float("nan")
    if int(direction) > 0:
        path = (high[start : end + 1] - float(entry_price)) / float(sl_dist_base)
    else:
        path = (float(entry_price) - low[start : end + 1]) / float(sl_dist_base)
    return float(np.nanmax(path)) if path.size else float("nan")


def tail_after_tp1_be_or_time(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    direction: int,
    tp1_hit_i: int,
    be_price: float,
    end_i: int,
) -> Dict[str, Any]:
    start = int(tp1_hit_i + 1)
    if start > int(end_i):
        return {"tail_exit_type": "BE", "tail_exit_i": int(tp1_hit_i), "tail_exit_price": float(be_price)}
    h = high[start : int(end_i) + 1]
    l = low[start : int(end_i) + 1]
    c_end = float(close[int(end_i)])
    if int(direction) > 0:
        be_mask = l <= float(be_price)
    else:
        be_mask = h >= float(be_price)
    be_idx = int(np.argmax(be_mask)) if bool(np.any(be_mask)) else -1
    if be_idx >= 0:
        return {"tail_exit_type": "BE", "tail_exit_i": int(start + be_idx), "tail_exit_price": float(be_price)}
    return {"tail_exit_type": "TIME", "tail_exit_i": int(end_i), "tail_exit_price": float(c_end)}


def compute_event_outcomes(
    mkt: MarketConfig,
    *,
    df: pd.DataFrame,
    ev: pd.DataFrame,
    ex: ExitConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes TP1/SL/H base outcome, then runner after TP1:
    - TP1 partial close at tp1_close_frac
    - Remaining part SL moved to BE(price includes cost): be_price = entry + direction*(roundtrip_cost+slippage)
    - Remaining runs to TP2 or BE or TIME (within same H)
    - net_r computed in R units, cost subtracted once (roundtrip).
    """
    if ev.empty:
        return ev.copy(), {"econ_pruned": 0, "events_in": 0}

    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n_bars = int(len(df))

    t = ex.tpslh
    cost_total_px = float(mkt.roundtrip_cost_price) + float(mkt.slippage_buffer_price)
    rows: List[Dict[str, Any]] = []

    for _, r in ev.iterrows():
        si = int(r["signal_i"])
        if not (0 <= si < n_bars):
            continue
        atr_ref = float(r["atr_ref"])
        if not (np.isfinite(atr_ref) and atr_ref > 1e-12):
            continue
        direction = int(r["direction"])

        if "entry_i" in r and "entry_price" in r:
            ei = int(r["entry_i"])
            entry_px = float(r["entry_price"])
            scan_start_i = int(ei)
        elif str(ex.entry) == "next_open":
            ei = int(r["entry_i_next_open"])
            entry_px = float(r["entry_price_next_open"])
            scan_start_i = int(ei)
        else:
            ei = int(r["entry_i_signal_close"])
            entry_px = float(r["entry_price_signal_close"])
            scan_start_i = int(ei + 1)

        if not (0 <= ei < n_bars and 0 <= scan_start_i < n_bars and np.isfinite(entry_px)):
            continue

        tp1_dist = float(atr_ref * float(t.tp1_atr_mult))
        sl_dist = float(atr_ref * float(t.sl_atr_mult))
        if not (tp1_dist > 1e-12 and sl_dist > 1e-12):
            continue

        tp1_price = float(entry_px + direction * tp1_dist)
        sl_price = float(entry_px - direction * sl_dist)
        tp1_r = float(tp1_dist / sl_dist)
        cost_r = float(cost_total_px / sl_dist) if cost_total_px > 0 else 0.0
        tp1_dist_ratio = float(tp1_dist / max(1e-12, cost_total_px)) if cost_total_px > 0 else float("inf")

        base = triple_barrier_tp1sl(
            high=high,
            low=low,
            close=close,
            direction=direction,
            entry_i=ei,
            scan_start_i=scan_start_i,
            tp1_price=tp1_price,
            sl_price=sl_price,
            H=int(t.H),
        )
        tp1_hit = bool(base["tp1_hit"])
        exit_i_base = int(base["exit_i_base"])
        exit_type_base = str(base["exit_type_base"])
        tp1_hit_i = int(base["tp1_hit_i"])
        end_i = int(min(n_bars - 1, ei + int(t.H)))

        fib_price = float(entry_px + direction * float(tp1_dist) * 0.9)
        fib_hit = False
        if not tp1_hit and np.isfinite(fib_price):
            h1 = high[int(scan_start_i) : int(end_i) + 1]
            l1 = low[int(scan_start_i) : int(end_i) + 1]
            if h1.size > 0:
                if int(direction) > 0:
                    fib_mask = h1 >= float(fib_price)
                    sl_mask = l1 <= float(sl_price)
                else:
                    fib_mask = l1 <= float(fib_price)
                    sl_mask = h1 >= float(sl_price)
                fib_idx = int(np.argmax(fib_mask)) if bool(np.any(fib_mask)) else -1
                sl_idx = int(np.argmax(sl_mask)) if bool(np.any(sl_mask)) else -1
                if fib_idx >= 0 and (sl_idx < 0 or fib_idx < sl_idx):
                    fib_hit = True
        y_success = int(bool(tp1_hit) or bool(fib_hit))

        tp2_hit = False
        runner_exit_type = "NA"
        runner_exit_i = -1
        runner_exit_price = float("nan")
        runner_r = float("nan")
        tail_exit_type = "NA"
        tail_exit_i = -1
        tail_exit_price = float("nan")
        tail_r = float("nan")
        mae_r = float("nan")
        mfe_r = float("nan")

        if tp1_hit:
            tp2_dist = float(ex.tp2_mult * tp1_dist)
            tp2_price = float(entry_px + direction * tp2_dist)
            # BE includes cost
            be_price = float(entry_px + direction * float(cost_total_px))
            runner = runner_after_tp1(
                high=high,
                low=low,
                close=close,
                direction=direction,
                tp1_hit_i=tp1_hit_i,
                entry_price=entry_px,
                be_price=be_price,
                tp2_price=tp2_price,
                end_i=end_i,
            )
            tp2_hit = bool(runner["tp2_hit"])
            runner_exit_type = str(runner["runner_exit_type"])
            runner_exit_i = int(runner["runner_exit_i"])
            runner_exit_price = float(runner["runner_exit_price"])
            final_exit_i = int(max(exit_i_base, runner_exit_i))
            final_exit_type = runner_exit_type

            if runner_exit_type == "TP2":
                runner_r = float(ex.tp2_mult * tp1_r)
            elif runner_exit_type == "BE":
                runner_r = float(cost_total_px / sl_dist)  # BE-cost => ~cost_r in R
            else:
                rr = float(((runner_exit_price - entry_px) * direction) / sl_dist)
                # After TP1, runner cannot go negative in this design
                runner_r = float(np.clip(rr, float(cost_total_px / sl_dist), float(ex.tp2_mult * tp1_r)))
            net_r = float(ex.tp1_close_frac * tp1_r + (1.0 - ex.tp1_close_frac) * runner_r - cost_r)

            tail = tail_after_tp1_be_or_time(high=high, low=low, close=close, direction=direction, tp1_hit_i=tp1_hit_i, be_price=be_price, end_i=end_i)
            tail_exit_type = str(tail["tail_exit_type"])
            tail_exit_i = int(tail["tail_exit_i"])
            tail_exit_price = float(tail["tail_exit_price"])
            if tail_exit_type == "BE":
                tail_r = float(cost_total_px / sl_dist)
            else:
                rr_t = float(((tail_exit_price - entry_px) * direction) / sl_dist)
                tail_r = float(np.clip(rr_t, float(cost_total_px / sl_dist), 50.0))
        else:
            final_exit_i = exit_i_base
            final_exit_type = exit_type_base
            if exit_type_base == "SL":
                net_r = float(-1.0 - cost_r)
            else:
                exit_px = float(base["exit_price_time"])
                r_mk = float(((exit_px - entry_px) * direction) / sl_dist)
                r_mk = float(np.clip(r_mk, -1.0, tp1_r))
                net_r = float(r_mk - cost_r)
            runner_r = float("nan")

        # MAE/MFE: excursion before planned exit horizon (uses high/low, causal within window)
        try:
            start_mae = int(scan_start_i)
            end_mae = int(end_i)
            if 0 <= start_mae <= end_mae < n_bars:
                if int(direction) > 0:
                    path = (low[start_mae : end_mae + 1] - float(entry_px)) / float(sl_dist)
                else:
                    path = (float(entry_px) - high[start_mae : end_mae + 1]) / float(sl_dist)
                mae_r = float(np.nanmin(path)) if path.size else float("nan")
                if int(direction) > 0:
                    path_f = (high[start_mae : end_mae + 1] - float(entry_px)) / float(sl_dist)
                else:
                    path_f = (float(entry_px) - low[start_mae : end_mae + 1]) / float(sl_dist)
                mfe_r = float(np.nanmax(path_f)) if path_f.size else float("nan")
        except Exception:
            mae_r = float("nan")
            mfe_r = float("nan")

        tp1_cash_r = float(ex.tp1_close_frac * tp1_r) if tp1_hit else 0.0
        runner_cash_r = float((1.0 - ex.tp1_close_frac) * float(runner_r) - cost_r) if tp1_hit else 0.0

        row_out: Dict[str, Any] = {
                "side": str(r["side"]),
                "direction": int(direction),
                "signal_i": int(si),
                "signal_time": str(r["signal_time"]),
                "_signal_ts": pd.to_datetime(r["_signal_ts"], utc=True, errors="coerce"),
                "entry_i": int(ei),
                "entry_time": str(df.index[int(ei)]),
                "_entry_ts": pd.to_datetime(df.index[int(ei)], utc=True),
                "entry_price": float(entry_px),
                "entry_scheme": str(ex.entry),
                "atr_ref": float(atr_ref),
                "H": int(t.H),
                "tp1_atr_mult": float(t.tp1_atr_mult),
                "sl_atr_mult": float(t.sl_atr_mult),
                "tp1_close_frac": float(ex.tp1_close_frac),
                "tp2_mult": float(ex.tp2_mult),
                "tp1_dist": float(tp1_dist),
                "sl_dist": float(sl_dist),
                "tp1_r": float(tp1_r),
                "cost_r": float(cost_r),
                "cost_to_sl_dist": float(cost_r),
                "tp1_over_cost": float(tp1_dist / max(1e-12, cost_total_px)) if cost_total_px > 0 else float("inf"),
                "sl_over_cost": float(sl_dist / max(1e-12, cost_total_px)) if cost_total_px > 0 else float("inf"),
                "tp1_r_over_cost": float(tp1_r / max(1e-12, cost_r)) if np.isfinite(cost_r) else float("nan"),
                "sl_r_over_cost": float(1.0 / max(1e-12, cost_r)) if np.isfinite(cost_r) else float("nan"),
                "tp1_dist_ratio": float(tp1_dist_ratio),
                "tp1_hit": bool(tp1_hit),
                "tp1_fib10_hit": bool(fib_hit),
                "y_success": int(y_success),
                "tp2_hit": bool(tp2_hit),
                "tp1_hit_i": int(tp1_hit_i),
                "exit_i_base": int(exit_i_base),
                "exit_type_base": str(exit_type_base),
                "exit_i": int(final_exit_i),
                "exit_time": str(df.index[int(final_exit_i)]),
                "exit_type": str(final_exit_type),
                "net_r": float(net_r),
                "mae_r": float(mae_r),
                "mfe_r": float(mfe_r),
                "tp1_cash_r": float(tp1_cash_r),
                "runner_cash_r": float(runner_cash_r),
                "runner_r": float(runner_r) if np.isfinite(runner_r) else float("nan"),
                "runner_exit_i": int(runner_exit_i) if int(runner_exit_i) >= 0 else -1,
                "runner_exit_type": str(runner_exit_type) if str(runner_exit_type) != "NA" else "NA",
                "tail_r": float(tail_r) if np.isfinite(tail_r) else float("nan"),
                "tail_exit_i": int(tail_exit_i) if int(tail_exit_i) >= 0 else -1,
                "tail_exit_type": str(tail_exit_type) if str(tail_exit_type) != "NA" else "NA",
        }

        # carry precomputed features/regimes from event row (strictly causal at signal time)
        for c in FEATURE_COLS:
            if c in r:
                try:
                    row_out[c] = float(r[c])
                except Exception:
                    pass
        for c in PATH_FEATURE_COLS:
            if c in r:
                try:
                    row_out[c] = float(r[c])
                except Exception:
                    pass
        for c in ("vol_regime", "trend_regime", "internal_regime", "session_regime"):
            if c in r:
                try:
                    row_out[c] = int(r[c])
                except Exception:
                    pass

        rows.append(row_out)

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {"econ_pruned": 0, "events_in": int(len(ev))}
    out = out.sort_values("_entry_ts", kind="mergesort").reset_index(drop=True)
    return out, {"econ_pruned": 0, "events_in": int(len(ev))}


def compute_event_outcomes_dynamic(
    mkt: MarketConfig,
    *,
    df: pd.DataFrame,
    ev: pd.DataFrame,
    base_sl_atr_mult: float,
) -> pd.DataFrame:
    if ev.empty:
        return ev.copy()

    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n_bars = int(len(df))
    cost_total_px = float(mkt.roundtrip_cost_price) + float(mkt.slippage_buffer_price)

    rows: List[Dict[str, Any]] = []
    for _, r in ev.iterrows():
        if "entry_i" not in r or "entry_price" not in r:
            continue
        ei = int(r["entry_i"])
        if not (0 <= int(ei) < int(n_bars)):
            continue
        entry_px = float(r["entry_price"])
        direction = int(r.get("direction", 0))
        atr_ref = float(r.get("atr_ref", float("nan")))
        if not (np.isfinite(entry_px) and np.isfinite(atr_ref) and atr_ref > 1e-12 and int(direction) in (-1, 1)):
            continue

        H1 = int(r.get("H1", r.get("H", 0)))
        H2 = int(r.get("H2", max(0, int(H1))))
        tp1_r_dyn = float(r.get("tp1_r_dyn", float("nan")))
        sl_r_dyn = float(r.get("sl_r_dyn", float("nan")))
        tp2_r_dyn = float(r.get("tp2_r_dyn", float("nan")))
        tp1_close_frac = float(r.get("tp1_close_frac", 0.5))
        trail_mult = float(r.get("trail_mult", 0.4))
        tp2_n1 = int(r.get("tp2_n1", -1))
        tp2_n2 = int(r.get("tp2_n2", -1))
        tp2_r_n0 = float(r.get("tp2_r_n0", tp2_r_dyn))
        tp2_r_n1 = float(r.get("tp2_r_n1", tp2_r_dyn))
        tp2_r_n2 = float(r.get("tp2_r_n2", tp2_r_dyn))
        trail_mult_n0 = float(r.get("trail_mult_n0", trail_mult))
        trail_mult_n1 = float(r.get("trail_mult_n1", trail_mult))
        trail_mult_n2 = float(r.get("trail_mult_n2", trail_mult))

        if not (np.isfinite(tp1_r_dyn) and np.isfinite(sl_r_dyn) and np.isfinite(tp2_r_dyn)):
            continue
        if float(sl_r_dyn) <= 1e-12:
            continue
        sl_dist_base = float(atr_ref * float(base_sl_atr_mult))
        if not (np.isfinite(sl_dist_base) and sl_dist_base > 1e-12):
            continue
        sl_dist = float(sl_dist_base * float(sl_r_dyn))
        tp1_dist = float(sl_dist_base * float(tp1_r_dyn))
        tp2_dist = float(sl_dist_base * float(tp2_r_dyn))
        if not (sl_dist > 1e-12 and tp1_dist > 1e-12):
            continue

        tp1_price = float(entry_px + float(direction) * tp1_dist)
        sl_price = float(entry_px - float(direction) * sl_dist)
        tp2_price = float(entry_px + float(direction) * tp2_dist)
        tp1_r_actual = float(tp1_dist / sl_dist)
        tp2_r_actual = float(tp2_dist / sl_dist)
        cost_r = float(cost_total_px / sl_dist) if cost_total_px > 0 else 0.0
        tp1_dist_ratio = float(tp1_dist / max(1e-12, cost_total_px)) if cost_total_px > 0 else float("inf")

        scan_start_i = int(ei)
        end_i = int(min(n_bars - 1, int(ei) + int(H1)))
        base = triple_barrier_tp1sl(
            high=high,
            low=low,
            close=close,
            direction=direction,
            entry_i=ei,
            scan_start_i=scan_start_i,
            tp1_price=tp1_price,
            sl_price=sl_price,
            H=int(H1),
        )
        tp1_hit = bool(base["tp1_hit"])
        tp1_hit_i = int(base["tp1_hit_i"])
        exit_i_base = int(base["exit_i_base"])
        exit_type_base = str(base["exit_type_base"])

        tp2_hit = False
        runner_exit_type = "NA"
        runner_exit_i = -1
        runner_exit_price = float("nan")
        runner_r = float("nan")
        tail_r = float("nan")

        additional_mfe_r = float("nan")

        if tp1_hit and int(tp1_hit_i) >= 0:
            end_i2 = int(min(n_bars - 1, int(tp1_hit_i) + int(H2)))
            be_price = float(entry_px + float(direction) * float(cost_total_px))
            trail_stop_px = float(max(0.0, float(trail_mult) * float(atr_ref)))
            schedule: List[Tuple[int, float, float]] = []
            if np.isfinite(tp2_r_n0) and np.isfinite(trail_mult_n0):
                schedule.append((0, float(entry_px + float(direction) * float(sl_dist_base) * float(tp2_r_n0)), float(max(0.0, float(trail_mult_n0) * float(atr_ref)))))
            if int(tp2_n1) > 0 and np.isfinite(tp2_r_n1) and np.isfinite(trail_mult_n1):
                schedule.append(
                    (
                        int(tp2_n1),
                        float(entry_px + float(direction) * float(sl_dist_base) * float(tp2_r_n1)),
                        float(max(0.0, float(trail_mult_n1) * float(atr_ref))),
                    )
                )
            if int(tp2_n2) > 0 and np.isfinite(tp2_r_n2) and np.isfinite(trail_mult_n2):
                schedule.append(
                    (
                        int(tp2_n2),
                        float(entry_px + float(direction) * float(sl_dist_base) * float(tp2_r_n2)),
                        float(max(0.0, float(trail_mult_n2) * float(atr_ref))),
                    )
                )
            runner = runner_after_tp1_dynamic(
                high=high,
                low=low,
                close=close,
                direction=direction,
                tp1_hit_i=int(tp1_hit_i),
                entry_price=float(entry_px),
                be_price=float(be_price),
                tp2_price=float(tp2_price),
                trail_stop_px=float(trail_stop_px),
                end_i=int(end_i2),
                schedule=schedule if schedule else None,
            )
            tp2_hit = bool(runner["tp2_hit"])
            runner_exit_type = str(runner["runner_exit_type"])
            runner_exit_i = int(runner["runner_exit_i"])
            runner_exit_price = float(runner["runner_exit_price"])
            if runner_exit_type == "TP2":
                runner_r = float(tp2_r_actual)
            else:
                rr = float(((runner_exit_price - entry_px) * float(direction)) / float(sl_dist))
                runner_r = float(np.clip(rr, float(cost_r), float(tp2_r_actual)))
            net_r = float(tp1_close_frac * tp1_r_actual + (1.0 - tp1_close_frac) * runner_r - cost_r)
            additional_mfe_r = additional_mfe_after_tp1(
                high=high,
                low=low,
                direction=direction,
                entry_price=entry_px,
                sl_dist_base=sl_dist_base,
                tp1_hit_i=int(tp1_hit_i),
                H2=int(H2),
            )
            tail_r = float(runner_r)
            final_exit_i = int(max(exit_i_base, runner_exit_i))
            final_exit_type = runner_exit_type
        else:
            final_exit_i = exit_i_base
            final_exit_type = exit_type_base
            if exit_type_base == "SL":
                net_r = float(-1.0 - cost_r)
            else:
                exit_px = float(base["exit_price_time"])
                rr = float(((exit_px - entry_px) * float(direction)) / float(sl_dist))
                rr = float(np.clip(rr, -1.0, float(tp1_r_actual)))
                net_r = float(rr - cost_r)

        row = dict(r)
        row.update(
            {
                "entry_i": int(ei),
                "entry_price": float(entry_px),
                "tp1_hit": bool(tp1_hit),
                "tp2_hit": bool(tp2_hit),
                "tp1_hit_i": int(tp1_hit_i),
                "exit_i": int(final_exit_i),
                "exit_type": str(final_exit_type),
                "tp1_r": float(tp1_r_actual),
                "tp2_r": float(tp2_r_actual),
                "sl_r": float(sl_r_dyn),
                "sl_dist": float(sl_dist),
                "tp1_dist": float(tp1_dist),
                "tp2_dist": float(tp2_dist),
                "cost_r": float(cost_r),
                "cost_to_sl_dist": float(cost_r),
                "tp1_over_cost": float(tp1_dist / max(1e-12, cost_total_px)) if cost_total_px > 0 else float("inf"),
                "sl_over_cost": float(sl_dist / max(1e-12, cost_total_px)) if cost_total_px > 0 else float("inf"),
                "tp1_r_over_cost": float(tp1_r_actual / max(1e-12, cost_r)) if np.isfinite(cost_r) else float("nan"),
                "sl_r_over_cost": float(sl_r_dyn / max(1e-12, cost_r)) if np.isfinite(cost_r) else float("nan"),
                "tp1_dist_ratio": float(tp1_dist_ratio),
                "net_r": float(net_r),
                "runner_exit_type": str(runner_exit_type),
                "runner_exit_i": int(runner_exit_i) if int(runner_exit_i) >= 0 else -1,
                "runner_exit_price": float(runner_exit_price) if np.isfinite(runner_exit_price) else float("nan"),
                "additional_mfe_r": float(additional_mfe_r),
                "tail_r": float(tail_r),
            }
        )
        if bool(tp1_hit):
            row["tp1_cash_r"] = float(tp1_close_frac * tp1_r_actual)
            row["runner_cash_r"] = float((1.0 - tp1_close_frac) * float(runner_r) - float(cost_r))
        else:
            row["tp1_cash_r"] = 0.0
            row["runner_cash_r"] = 0.0
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("_entry_ts", kind="mergesort").reset_index(drop=True)
    return out


# =============================
# Purged CV + model fitting
# =============================


class PurgedTimeSeriesSplit:
    def __init__(self, *, n_splits: int, entry_i: np.ndarray, exit_i: np.ndarray, gap: int) -> None:
        self.n_splits = int(n_splits)
        self.entry_i = np.asarray(entry_i, dtype=int)
        self.exit_i = np.asarray(exit_i, dtype=int)
        self.gap = int(gap)

    def get_n_splits(self, X=None, y=None, groups=None):  # noqa: ANN001, ANN201
        return int(self.n_splits)

    def split(self, X, y=None, groups=None):  # noqa: ANN001
        n = int(len(X))
        if n <= 200 or self.n_splits < 2:
            idx = np.arange(n)
            yield idx, idx
            return
        order = np.arange(n)
        test_size = max(200, n // (self.n_splits + 1))
        for k in range(self.n_splits):
            test_start = int((k + 1) * test_size)
            test_end = int(min(n, test_start + test_size))
            if test_start >= n or test_end <= test_start:
                break
            test_idx = order[test_start:test_end]
            test_entry_start = int(self.entry_i[test_start])
            train_idx = order[:test_start]
            cut = int(test_entry_start - self.gap)
            train_idx = train_idx[self.exit_i[train_idx] < cut]
            if train_idx.size < 300:
                continue
            yield train_idx, test_idx


def predict_proba_1(model: Any, X: np.ndarray) -> np.ndarray:
    try:
        p = model.predict_proba(X)[:, 1]
        return np.asarray(p, dtype=float)
    except Exception:
        return np.full(int(X.shape[0]), 0.5, dtype=float)


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    if int(np.sum(ok)) == 0:
        return float("nan")
    return float(np.mean((y[ok] - p[ok]) ** 2))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    if int(np.sum(ok)) < 200 or int(np.unique(y[ok]).size) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y[ok], p[ok]))
    except Exception:
        return float("nan")


def fit_calibrated_lgbm(
    cv_cfg: CVConfig,
    mdl_cfg: ModelConfig,
    *,
    X: np.ndarray,
    y: np.ndarray,
    entry_i: np.ndarray,
    exit_i: np.ndarray,
    calib_method: str,
    num_leaves: Optional[int] = None,
    max_depth: Optional[int] = None,
    min_data_in_leaf: Optional[int] = None,
) -> Any:
    import lightgbm as lgb
    from sklearn.calibration import CalibratedClassifierCV

    y = np.asarray(y, dtype=int)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    spw = float(neg / max(1, pos))
    nl = int(num_leaves) if num_leaves is not None else int(mdl_cfg.num_leaves_grid[1])
    md = int(max_depth) if max_depth is not None else int(mdl_cfg.max_depth_grid[1])
    mdl = int(min_data_in_leaf) if min_data_in_leaf is not None else int(mdl_cfg.min_data_in_leaf_grid[1])
    base = lgb.LGBMClassifier(**{**mdl_cfg.lgbm_base_params, "num_leaves": nl, "max_depth": md, "min_data_in_leaf": mdl, "scale_pos_weight": spw})
    gap = int(cv_cfg.purge_bars + cv_cfg.embargo_bars)
    cv = PurgedTimeSeriesSplit(n_splits=int(cv_cfg.calib_cv_splits), entry_i=entry_i, exit_i=exit_i, gap=gap)
    cal = CalibratedClassifierCV(base, method=str(calib_method), cv=cv, ensemble=False)
    cal.fit(X, y)
    return cal


def fit_logreg_baseline(
    cv_cfg: CVConfig,
    *,
    X: np.ndarray,
    y: np.ndarray,
    entry_i: np.ndarray,
    exit_i: np.ndarray,
    calib_method: str,
) -> Any:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    base = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=int(cv_cfg.seed),
    )
    gap = int(cv_cfg.purge_bars + cv_cfg.embargo_bars)
    cv = PurgedTimeSeriesSplit(n_splits=int(cv_cfg.calib_cv_splits), entry_i=entry_i, exit_i=exit_i, gap=gap)
    cal = CalibratedClassifierCV(base, method=str(calib_method), cv=cv, ensemble=False)
    cal.fit(X, y)
    return cal


def oof_calibrated_predictions(
    cv_cfg: CVConfig,
    *,
    X: np.ndarray,
    y: np.ndarray,
    entry_i: np.ndarray,
    exit_i: np.ndarray,
    calib_method: str,
    base_estimator: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from sklearn.calibration import CalibratedClassifierCV

    y = np.asarray(y, dtype=int)
    gap = int(cv_cfg.purge_bars + cv_cfg.embargo_bars)
    splitter = PurgedTimeSeriesSplit(n_splits=max(3, int(cv_cfg.calib_cv_splits)), entry_i=entry_i, exit_i=exit_i, gap=gap)
    p = np.full(int(len(y)), np.nan, dtype=float)
    folds = 0
    for tr_idx, te_idx in splitter.split(X, y):
        folds += 1
        if tr_idx.size < 200 or int(np.unique(y[tr_idx]).size) < 2:
            p[te_idx] = float(np.mean(y[tr_idx])) if tr_idx.size else float(np.mean(y))
            continue
        cal = CalibratedClassifierCV(base_estimator, method=str(calib_method), cv=3, ensemble=False)
        cal.fit(X[tr_idx], y[tr_idx])
        p[te_idx] = cal.predict_proba(X[te_idx])[:, 1].astype(float)
    base_p = float(np.mean(y)) if int(len(y)) else 0.5
    p[~np.isfinite(p)] = float(base_p)
    return p, {"method": str(calib_method), "folds": int(folds), "n": int(len(y)), "gap": int(gap)}


def fit_lgbm_uncalibrated(
    mdl_cfg: ModelConfig,
    *,
    X: np.ndarray,
    y: np.ndarray,
    num_leaves: Optional[int] = None,
    max_depth: Optional[int] = None,
    min_data_in_leaf: Optional[int] = None,
    feature_fraction: Optional[float] = None,
    min_gain_to_split: Optional[float] = None,
) -> Any:
    import lightgbm as lgb

    y = np.asarray(y, dtype=int)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    spw = float(neg / max(1, pos))
    nl = int(num_leaves) if num_leaves is not None else int(mdl_cfg.num_leaves_grid[1])
    md = int(max_depth) if max_depth is not None else int(mdl_cfg.max_depth_grid[1])
    mdl = int(min_data_in_leaf) if min_data_in_leaf is not None else int(mdl_cfg.min_data_in_leaf_grid[1])
    ff = float(feature_fraction) if feature_fraction is not None else float(mdl_cfg.lgbm_base_params.get("feature_fraction", 0.9))
    mgs = float(min_gain_to_split) if min_gain_to_split is not None else float(mdl_cfg.lgbm_base_params.get("min_gain_to_split", 0.0))
    return lgb.LGBMClassifier(
        **{
            **mdl_cfg.lgbm_base_params,
            "num_leaves": int(nl),
            "max_depth": int(md),
            "min_data_in_leaf": int(mdl),
            "feature_fraction": float(ff),
            "min_gain_to_split": float(mgs),
            "scale_pos_weight": float(spw),
        }
    ).fit(X, y)


def year_bounds(year: int, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    ys = pd.Timestamp(f"{year}-01-01", tz="UTC")
    ye = pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC")
    return max(start_ts, ys), min(end_ts, ye)


def oof_predict_purged(
    cv_cfg: CVConfig,
    mdl_cfg: ModelConfig,
    *,
    ds_side_pre: pd.DataFrame,
    target: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Strictly pre-OS OOF scores via purged splits (no calibration).
    Uses LightGBM parameter grid to avoid "No further splits" under low-variance features.
    """
    ds2 = ds_side_pre.sort_values("_entry_ts", kind="mergesort").reset_index(drop=True)

    def _prune_features(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[List[str], Dict[str, Any]]:
        # NOTE: pre-OS only; safe (no future leakage). KISS thresholds.
        x = df.loc[:, list(cols)]
        miss = x.isna().mean().to_dict()
        var = x.var(ddof=0, numeric_only=True).to_dict()
        max_missing = 0.25
        var_min = 1e-12
        removed_missing = sorted([c for c in cols if float(miss.get(c, 1.0)) > float(max_missing) + 1e-12])
        removed_lowvar = sorted([c for c in cols if not np.isfinite(float(var.get(c, float("nan")))) or float(var.get(c, 0.0)) < float(var_min)])
        removed = sorted(set(removed_missing) | set(removed_lowvar))
        kept = [c for c in cols if c not in removed]
        if len(kept) < max(8, int(len(cols) * 0.25)):
            # do not over-prune (prevents degenerate no-split)
            kept = list(cols)
            removed = []
            removed_missing = []
            removed_lowvar = []
            note = "fallback_keep_all"
        else:
            note = "ok"
        return kept, {
            "note": str(note),
            "missing_rate_max": float(max_missing),
            "var_min": float(var_min),
            "removed_high_missing": removed_missing,
            "removed_low_variance": removed_lowvar,
            "removed_all": removed,
            "kept_n": int(len(kept)),
            "total_n": int(len(cols)),
        }

    feat_cols, prune_meta = _prune_features(ds2, cols=list(FEATURE_COLS))
    X = ds2[list(feat_cols)].to_numpy(dtype=float)
    if str(target) == "win":
        y = (pd.to_numeric(ds2["net_r"], errors="coerce").to_numpy(dtype=float) > 0.0).astype(int)
    elif str(target) == "tail":
        y = (pd.to_numeric(ds2.get("mae_r"), errors="coerce").to_numpy(dtype=float) <= -1.0).astype(int)
    else:
        raise ValueError(f"unknown target={target!r}")
    entry_i = ds2["entry_i"].astype(int).to_numpy()
    exit_i = ds2["exit_i"].astype(int).to_numpy()
    gap = int(cv_cfg.purge_bars + cv_cfg.embargo_bars)
    base_p = float(np.mean(y)) if y.size else 0.5
    if int(np.unique(y).size) < 2 or int(len(ds2)) < int(mdl_cfg.min_train_events):
        p0 = np.full(int(len(ds2)), float(base_p), dtype=float)
        return p0, {
            "ok": False,
            "reason": "insufficient_preos_or_single_class",
            "target": str(target),
            "gap": int(gap),
            "base_p": float(base_p),
            "n": int(len(ds2)),
            "features_used": list(feat_cols),
            "feature_prune": prune_meta,
        }

    grid: List[Tuple[int, int, int, float, float]] = []
    for mdl in mdl_cfg.min_data_in_leaf_grid:
        for nl in mdl_cfg.num_leaves_grid:
            for md in mdl_cfg.max_depth_grid:
                for ff in mdl_cfg.feature_fraction_grid:
                    for mgs in mdl_cfg.min_gain_to_split_grid:
                        grid.append((int(mdl), int(nl), int(md), float(ff), float(mgs)))

    best = {
        "auc": -float("inf"),
        "brier": float("inf"),
        "min_data_in_leaf": None,
        "num_leaves": None,
        "max_depth": None,
        "feature_fraction": None,
        "min_gain_to_split": None,
    }
    best_p = np.full(int(len(ds2)), float(base_p), dtype=float)
    grid_rows: List[Dict[str, Any]] = []

    for (min_leaf, num_leaves, max_depth, feature_fraction, min_gain_to_split) in grid:
        splitter = PurgedTimeSeriesSplit(n_splits=max(3, int(cv_cfg.calib_cv_splits)), entry_i=entry_i, exit_i=exit_i, gap=gap)
        p = np.full(int(len(ds2)), np.nan, dtype=float)
        folds = 0
        for tr_idx, te_idx in splitter.split(X, y):
            folds += 1
            if tr_idx.size < int(mdl_cfg.min_train_events) or int(np.unique(y[tr_idx]).size) < 2:
                p[te_idx] = float(np.mean(y[tr_idx])) if tr_idx.size else float(base_p)
                continue
            try:
                m = fit_lgbm_uncalibrated(
                    mdl_cfg,
                    X=X[tr_idx],
                    y=y[tr_idx],
                    num_leaves=int(num_leaves),
                    max_depth=int(max_depth),
                    min_data_in_leaf=int(min_leaf),
                    feature_fraction=float(feature_fraction),
                    min_gain_to_split=float(min_gain_to_split),
                )
                p[te_idx] = predict_proba_1(m, X[te_idx])
            except Exception:
                p[te_idx] = float(np.mean(y[tr_idx]))

        p[~np.isfinite(p)] = float(base_p)
        auc = float(roc_auc(y, p))
        brier = float(brier_score(y, p))
        grid_rows.append(
            {
                "min_data_in_leaf": int(min_leaf),
                "num_leaves": int(num_leaves),
                "max_depth": int(max_depth),
                "feature_fraction": float(feature_fraction),
                "min_gain_to_split": float(min_gain_to_split),
                "auc": float(auc),
                "brier": float(brier),
                "folds": int(folds),
            }
        )

        better = False
        if np.isfinite(auc) and (not np.isfinite(best["auc"]) or auc > float(best["auc"]) + 1e-12):
            better = True
        elif np.isfinite(auc) and np.isfinite(float(best["auc"])) and abs(auc - float(best["auc"])) <= 1e-12:
            # tie-break by lower brier
            if np.isfinite(brier) and (not np.isfinite(best["brier"]) or brier < float(best["brier"]) - 1e-12):
                better = True
        if better:
            best = {
                "auc": float(auc),
                "brier": float(brier),
                "min_data_in_leaf": int(min_leaf),
                "num_leaves": int(num_leaves),
                "max_depth": int(max_depth),
                "feature_fraction": float(feature_fraction),
                "min_gain_to_split": float(min_gain_to_split),
            }
            best_p = np.asarray(p, dtype=float)

    # train split stats on full pre-OS for the selected hyperparams (debug "No further splits")
    split_stats: Dict[str, Any] = {"ok": False}
    if best.get("min_data_in_leaf") is not None:
        try:
            m_full = fit_lgbm_uncalibrated(
                mdl_cfg,
                X=X,
                y=y,
                num_leaves=int(best["num_leaves"]),
                max_depth=int(best["max_depth"]),
                min_data_in_leaf=int(best["min_data_in_leaf"]),
                feature_fraction=float(best.get("feature_fraction") or 1.0),
                min_gain_to_split=float(best.get("min_gain_to_split") or 0.0),
            )
            booster = getattr(m_full, "booster_", None)
            if booster is not None:
                imp_split = booster.feature_importance(importance_type="split").tolist()
                imp_gain = booster.feature_importance(importance_type="gain").tolist()
                total_splits = int(np.sum(np.asarray(imp_split, dtype=float)))
                used_features = int(np.sum(np.asarray(imp_split, dtype=float) > 0))
                top_idx = np.argsort(np.asarray(imp_gain, dtype=float))[::-1][:10].tolist()
                split_stats = {
                    "ok": True,
                    "total_splits": int(total_splits),
                    "used_features": int(used_features),
                    "top_gain_features": [{"feature": str(feat_cols[int(i)]), "gain": float(imp_gain[int(i)]), "splits": int(imp_split[int(i)])} for i in top_idx if 0 <= int(i) < int(len(feat_cols))],
                }
        except Exception as e:
            split_stats = {"ok": False, "reason": "full_fit_failed", "err": str(e)}

    meta = {
        "ok": True,
        "target": str(target),
        "gap": int(gap),
        "base_p": float(base_p),
        "n": int(len(ds2)),
        "features_used": list(feat_cols),
        "feature_prune": prune_meta,
        "best": best,
        "grid": grid_rows,
        "split_stats": split_stats,
    }
    return best_p, meta


def oof_predict_purged_logreg(
    cv_cfg: CVConfig,
    *,
    ds_side_pre: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression

    ds2 = ds_side_pre.sort_values("_entry_ts", kind="mergesort").reset_index(drop=True)
    X = ds2[list(FEATURE_COLS)].to_numpy(dtype=float)
    y = (pd.to_numeric(ds2["net_r"], errors="coerce").to_numpy(dtype=float) > 0.0).astype(int)
    entry_i = ds2["entry_i"].astype(int).to_numpy()
    exit_i = ds2["exit_i"].astype(int).to_numpy()
    gap = int(cv_cfg.purge_bars + cv_cfg.embargo_bars)
    splitter = PurgedTimeSeriesSplit(n_splits=max(3, int(cv_cfg.calib_cv_splits)), entry_i=entry_i, exit_i=exit_i, gap=gap)
    p = np.full(int(len(ds2)), np.nan, dtype=float)
    folds = 0
    for tr_idx, te_idx in splitter.split(X, y):
        folds += 1
        if tr_idx.size < 400 or int(np.unique(y[tr_idx]).size) < 2:
            p[te_idx] = float(np.mean(y[tr_idx])) if tr_idx.size else 0.5
            continue
        try:
            lr = LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=int(cv_cfg.seed),
            ).fit(X[tr_idx], y[tr_idx])
            p[te_idx] = lr.predict_proba(X[te_idx])[:, 1].astype(float)
        except Exception:
            p[te_idx] = float(np.mean(y[tr_idx]))
    base_p = float(np.mean(y)) if y.size else 0.5
    p[~np.isfinite(p)] = base_p
    return p, {"folds": int(folds), "gap": int(gap), "base_p": float(base_p), "n": int(len(ds2))}


def calibrate_platt_isotonic(
    *,
    p_oof: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    """
    Fit calibrators on pre-OS OOF predictions (uses only pre-OS info).
    Returns best mapping by Brier score.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=int)
    p = np.asarray(p_oof, dtype=float)
    ok = np.isfinite(p)
    p = np.clip(p[ok], 1e-6, 1 - 1e-6)
    y = y[ok]
    if p.size < 500 or int(np.unique(y).size) < 2:
        return {"ok": False, "reason": "insufficient", "best": {"method": "none"}}

    # Platt scaling on logit(p)
    logit_p = np.log(p / (1.0 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=2000).fit(logit_p, y)
    p_platt = lr.predict_proba(logit_p)[:, 1].astype(float)
    b_platt = float(brier_score(y, p_platt))

    # Isotonic on p
    iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    p_iso = iso.predict(p).astype(float)
    b_iso = float(brier_score(y, p_iso))

    if np.isfinite(b_iso) and (not np.isfinite(b_platt) or b_iso <= b_platt):
        best = {"method": "isotonic", "brier": b_iso}
    else:
        best = {"method": "sigmoid", "brier": b_platt}
    return {
        "ok": True,
        "platt": {"brier": b_platt},
        "isotonic": {"brier": b_iso},
        "best": best,
        "models": {"platt_lr": lr, "isotonic": iso},
    }


def apply_calibration(p_raw: np.ndarray, *, cal: Dict[str, Any]) -> np.ndarray:
    p = np.asarray(p_raw, dtype=float)
    if not cal or not bool(cal.get("ok")):
        return p
    best = (cal.get("best") or {}).get("method")
    if best == "isotonic":
        iso = (cal.get("models") or {}).get("isotonic")
        if iso is not None:
            return np.asarray(iso.predict(p), dtype=float)
    if best == "sigmoid":
        lr = (cal.get("models") or {}).get("platt_lr")
        if lr is not None:
            p2 = np.clip(p, 1e-6, 1 - 1e-6)
            logit_p = np.log(p2 / (1.0 - p2)).reshape(-1, 1)
            return np.asarray(lr.predict_proba(logit_p)[:, 1], dtype=float)
    return p


def calibration_curve_table(p: np.ndarray, y: np.ndarray, *, bins: int = 10) -> List[Dict[str, Any]]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    ok = np.isfinite(p)
    p = p[ok]
    y = y[ok]
    if p.size == 0:
        return []
    edges = np.linspace(0.0, 1.0, int(max(2, bins)) + 1)
    rows: List[Dict[str, Any]] = []
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == len(edges) - 2:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if int(np.sum(m)) == 0:
            rows.append({"bin": f"{lo:.2f}-{hi:.2f}", "n": 0, "p_mean": float("nan"), "y_rate": float("nan")})
        else:
            rows.append({"bin": f"{lo:.2f}-{hi:.2f}", "n": int(np.sum(m)), "p_mean": float(np.mean(p[m])), "y_rate": float(np.mean(y[m]))})
    return rows

def choose_big_loss_threshold(
    *,
    p_big_loss: np.ndarray,
    y_big_loss: np.ndarray,
    min_take_rate: float,
    target_reduction: float = 0.5,
    q_grid: Optional[Sequence[float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    p = np.asarray(p_big_loss, dtype=float)
    y = np.asarray(y_big_loss, dtype=int)
    base_rate = float(np.mean(y)) if y.size else 0.0
    best_thr = 1.0
    best_meta: Dict[str, Any] = {"base_rate": float(base_rate)}
    p_ok = p[np.isfinite(p)]
    q_vals = list(q_grid) if q_grid else list(np.linspace(0.05, 0.95, 19))
    for q in q_vals:
        if p_ok.size == 0:
            continue
        thr = float(np.quantile(p_ok, float(q)))
        mask = p <= float(thr)
        take_rate = float(np.mean(mask)) if mask.size else 0.0
        if take_rate < float(min_take_rate):
            continue
        rate_after = float(np.mean(y[mask])) if np.any(mask) else float("nan")
        reduction = float((base_rate - rate_after) / max(1e-12, base_rate)) if np.isfinite(rate_after) else 0.0
        ok = reduction >= float(target_reduction)
        if ok:
            best_thr = float(thr)
            best_meta = {
                "base_rate": float(base_rate),
                "rate_after": float(rate_after),
                "reduction": float(reduction),
                "take_rate": float(take_rate),
                "q": float(q),
            }
            break
    return float(best_thr), best_meta


def oof_predict_purged_lgbm_custom(
    cv_cfg: CVConfig,
    mdl_cfg: ModelConfig,
    *,
    X: np.ndarray,
    y: np.ndarray,
    entry_i: np.ndarray,
    exit_i: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    import lightgbm as lgb

    y = np.asarray(y, dtype=int)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    spw = float(neg / max(1, pos))
    gap = int(cv_cfg.purge_bars + cv_cfg.embargo_bars)
    splitter = PurgedTimeSeriesSplit(n_splits=max(3, int(cv_cfg.calib_cv_splits)), entry_i=entry_i, exit_i=exit_i, gap=gap)
    p = np.full(int(len(y)), np.nan, dtype=float)
    folds = 0
    for tr_idx, te_idx in splitter.split(X, y):
        folds += 1
        if tr_idx.size < 400 or int(np.unique(y[tr_idx]).size) < 2:
            p[te_idx] = float(np.mean(y[tr_idx])) if tr_idx.size else 0.5
            continue
        base = lgb.LGBMClassifier(
            **{
                **mdl_cfg.lgbm_base_params,
                "num_leaves": int(mdl_cfg.num_leaves_grid[0]),
                "max_depth": int(mdl_cfg.max_depth_grid[0]),
                "min_data_in_leaf": int(mdl_cfg.min_data_in_leaf_grid[0]),
                "scale_pos_weight": float(spw),
            }
        )
        try:
            base.fit(X[tr_idx], y[tr_idx])
            p[te_idx] = base.predict_proba(X[te_idx])[:, 1].astype(float)
        except Exception:
            p[te_idx] = float(np.mean(y[tr_idx]))
    base_p = float(np.mean(y)) if y.size else 0.5
    p[~np.isfinite(p)] = base_p
    return p, {"folds": int(folds), "gap": int(gap), "base_p": float(base_p), "n": int(len(y))}


def oof_predict_purged_sklearn(
    cv_cfg: CVConfig,
    *,
    X: np.ndarray,
    y: np.ndarray,
    entry_i: np.ndarray,
    exit_i: np.ndarray,
    estimator: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=int)
    gap = int(cv_cfg.purge_bars + cv_cfg.embargo_bars)
    splitter = PurgedTimeSeriesSplit(n_splits=max(3, int(cv_cfg.calib_cv_splits)), entry_i=entry_i, exit_i=exit_i, gap=gap)
    p = np.full(int(len(y)), np.nan, dtype=float)
    folds = 0
    for tr_idx, te_idx in splitter.split(X, y):
        folds += 1
        if tr_idx.size < 300 or int(np.unique(y[tr_idx]).size) < 2:
            p[te_idx] = float(np.mean(y[tr_idx])) if tr_idx.size else 0.5
            continue
        try:
            if str(estimator) == "logreg":
                mdl = LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=int(cv_cfg.seed),
                ).fit(X[tr_idx], y[tr_idx])
            elif str(estimator) == "rf":
                mdl = RandomForestClassifier(
                    n_estimators=80,
                    max_depth=6,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=int(cv_cfg.seed),
                    n_jobs=1,
                ).fit(X[tr_idx], y[tr_idx])
            else:
                raise ValueError(f"unknown estimator={estimator!r}")
            p[te_idx] = mdl.predict_proba(X[te_idx])[:, 1].astype(float)
        except Exception:
            p[te_idx] = float(np.mean(y[tr_idx]))
    base_p = float(np.mean(y)) if y.size else 0.5
    p[~np.isfinite(p)] = base_p
    return p, {"folds": int(folds), "gap": int(gap), "base_p": float(base_p), "n": int(len(y))}


def _maybe_import_xgb() -> Any:
    try:
        import xgboost as xgb  # type: ignore

        return xgb
    except Exception:
        return None


def oof_predict_purged_xgb(
    cv_cfg: CVConfig,
    *,
    X: np.ndarray,
    y: np.ndarray,
    entry_i: np.ndarray,
    exit_i: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    xgb = _maybe_import_xgb()
    y = np.asarray(y, dtype=int)
    base_p = float(np.mean(y)) if y.size else 0.5
    if xgb is None:
        return np.full(int(len(y)), float(base_p), dtype=float), {"ok": False, "reason": "xgboost_not_available", "base_p": float(base_p)}
    gap = int(cv_cfg.purge_bars + cv_cfg.embargo_bars)
    splitter = PurgedTimeSeriesSplit(n_splits=max(3, int(cv_cfg.calib_cv_splits)), entry_i=entry_i, exit_i=exit_i, gap=gap)
    p = np.full(int(len(y)), np.nan, dtype=float)
    folds = 0
    for tr_idx, te_idx in splitter.split(X, y):
        folds += 1
        if tr_idx.size < 300 or int(np.unique(y[tr_idx]).size) < 2:
            p[te_idx] = float(np.mean(y[tr_idx])) if tr_idx.size else float(base_p)
            continue
        try:
            mdl = xgb.XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=int(cv_cfg.seed),
                n_jobs=1,
            ).fit(X[tr_idx], y[tr_idx])
            p[te_idx] = mdl.predict_proba(X[te_idx])[:, 1].astype(float)
        except Exception:
            p[te_idx] = float(np.mean(y[tr_idx]))
    p[~np.isfinite(p)] = float(base_p)
    return p, {"ok": True, "folds": int(folds), "gap": int(gap), "base_p": float(base_p), "n": int(len(y))}


def fit_logreg_full(
    cv_cfg: CVConfig,
    *,
    X: np.ndarray,
    y: np.ndarray,
) -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=int(cv_cfg.seed),
    ).fit(X, np.asarray(y, dtype=int))


def fit_rf_full(
    cv_cfg: CVConfig,
    *,
    X: np.ndarray,
    y: np.ndarray,
) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=int(cv_cfg.seed),
        n_jobs=1,
    ).fit(X, np.asarray(y, dtype=int))


def fit_xgb_full(
    cv_cfg: CVConfig,
    *,
    X: np.ndarray,
    y: np.ndarray,
) -> Any:
    xgb = _maybe_import_xgb()
    if xgb is None:
        return None
    return xgb.XGBClassifier(
        n_estimators=160,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=int(cv_cfg.seed),
        n_jobs=1,
    ).fit(X, np.asarray(y, dtype=int))


def regime_key_from_row(r: pd.Series) -> Tuple[int, int, int]:
    try:
        v = int(r.get("vol_regime", -1))
        t = int(r.get("trend_regime", -1))
        i = int(r.get("internal_regime", -1))
        return int(v), int(t), int(i)
    except Exception:
        return -1, -1, -1


def build_regime_weights(
    *,
    ds_pre: pd.DataFrame,
    y: np.ndarray,
    p_map: Dict[str, np.ndarray],
    min_n: int = 200,
) -> Dict[Tuple[int, int, int], Dict[str, float]]:
    weights: Dict[Tuple[int, int, int], Dict[str, float]] = {}
    if ds_pre.empty:
        return weights
    for key, sub in ds_pre.groupby(["vol_regime", "trend_regime", "internal_regime"]):
        if int(len(sub)) < int(min_n):
            continue
        idx = sub.index.to_numpy(dtype=int)
        yk = np.asarray(y, dtype=int)[idx]
        if int(np.unique(yk).size) < 2:
            continue
        for name, p in p_map.items():
            p_sub = np.asarray(p, dtype=float)[idx]
            b = float(brier_score(yk, p_sub))
            if not np.isfinite(b):
                continue
            weights.setdefault(tuple(int(x) for x in key), {})[str(name)] = float(1.0 / max(1e-6, b))
    # normalize
    for key, w in list(weights.items()):
        s = float(np.sum(list(w.values())))
        if s <= 0:
            weights.pop(key, None)
        else:
            weights[key] = {k: float(v / s) for k, v in w.items()}
    return weights


def combine_ensemble_probs(
    ds_all: pd.DataFrame,
    *,
    p_map: Dict[str, np.ndarray],
    weights_regime: Dict[Tuple[int, int, int], Dict[str, float]],
    weights_fallback: Dict[str, float],
) -> np.ndarray:
    n = int(len(ds_all))
    p_out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        r = ds_all.iloc[int(i)]
        key = regime_key_from_row(r)
        w = weights_regime.get(tuple(key), weights_fallback)
        if not w:
            p_out[i] = 0.5
            continue
        s = 0.0
        v = 0.0
        for name, p in p_map.items():
            if name not in w:
                continue
            pi = float(p[int(i)]) if int(i) < len(p) else float("nan")
            if not np.isfinite(pi):
                continue
            wv = float(w.get(name, 0.0))
            v += float(wv) * float(pi)
            s += float(wv)
        p_out[i] = float(v / s) if s > 0 else 0.5
    p_out[~np.isfinite(p_out)] = 0.5
    return p_out

def fit_lgbm_classifier_full(
    mdl_cfg: ModelConfig,
    *,
    X: np.ndarray,
    y: np.ndarray,
) -> Any:
    import lightgbm as lgb

    y = np.asarray(y, dtype=int)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    spw = float(neg / max(1, pos))
    return lgb.LGBMClassifier(
        **{
            **mdl_cfg.lgbm_base_params,
            "num_leaves": int(mdl_cfg.num_leaves_grid[0]),
            "max_depth": int(mdl_cfg.max_depth_grid[0]),
            "min_data_in_leaf": int(mdl_cfg.min_data_in_leaf_grid[0]),
            "scale_pos_weight": float(spw),
        }
    ).fit(X, y)


def oof_predict_purged_lgbm_quantile(
    cv_cfg: CVConfig,
    mdl_cfg: ModelConfig,
    *,
    X: np.ndarray,
    y: np.ndarray,
    entry_i: np.ndarray,
    exit_i: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    import lightgbm as lgb

    y = np.asarray(y, dtype=float)
    gap = int(cv_cfg.purge_bars + cv_cfg.embargo_bars)
    splitter = PurgedTimeSeriesSplit(n_splits=max(3, int(cv_cfg.calib_cv_splits)), entry_i=entry_i, exit_i=exit_i, gap=gap)
    p = np.full(int(len(y)), np.nan, dtype=float)
    folds = 0
    for tr_idx, te_idx in splitter.split(X, y):
        folds += 1
        if tr_idx.size < 400:
            p[te_idx] = float(np.nanmedian(y[tr_idx])) if tr_idx.size else float(np.nanmedian(y))
            continue
        model = lgb.LGBMRegressor(
            **{
                **mdl_cfg.lgbm_base_params,
                "objective": "quantile",
                "alpha": float(alpha),
                "num_leaves": int(mdl_cfg.num_leaves_grid[0]),
                "max_depth": int(mdl_cfg.max_depth_grid[0]),
                "min_data_in_leaf": int(mdl_cfg.min_data_in_leaf_grid[0]),
            }
        )
        try:
            model.fit(X[tr_idx], y[tr_idx])
            p[te_idx] = model.predict(X[te_idx]).astype(float)
        except Exception:
            p[te_idx] = float(np.nanmedian(y[tr_idx])) if tr_idx.size else float(np.nanmedian(y))
    base_q = float(np.nanmedian(y)) if y.size else float("nan")
    p[~np.isfinite(p)] = base_q
    return p, {"folds": int(folds), "gap": int(gap), "base_q": float(base_q), "n": int(len(y))}


def fit_lgbm_quantile_full(
    mdl_cfg: ModelConfig,
    *,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> Any:
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        **{
            **mdl_cfg.lgbm_base_params,
            "objective": "quantile",
            "alpha": float(alpha),
            "num_leaves": int(mdl_cfg.num_leaves_grid[0]),
            "max_depth": int(mdl_cfg.max_depth_grid[0]),
            "min_data_in_leaf": int(mdl_cfg.min_data_in_leaf_grid[0]),
        }
    ).fit(X, np.asarray(y, dtype=float))


def beta_posterior_prob_ge(k: int, n: int, threshold: float) -> float:
    try:
        from scipy.stats import beta as beta_dist

        return float(1.0 - beta_dist.cdf(float(threshold), 1 + int(k), 1 + int(n - k)))
    except Exception:
        return float("nan")


def beta_posterior_prob_le(k: int, n: int, threshold: float) -> float:
    try:
        from scipy.stats import beta as beta_dist

        return float(beta_dist.cdf(float(threshold), 1 + int(k), 1 + int(n - k)))
    except Exception:
        return float("nan")


def build_post_tp1_feature_matrix(
    ctx: Dict[str, np.ndarray],
    *,
    tp1_idx: np.ndarray,
    feature_cols: Sequence[str],
    mean_window: int = 3,
) -> np.ndarray:
    n = int(len(tp1_idx))
    X = np.full((n, int(len(feature_cols)) * 2), np.nan, dtype=float)
    for i in range(n):
        idx = int(tp1_idx[i])
        if idx < 0:
            continue
        for j, c in enumerate(feature_cols):
            arr = np.asarray(ctx.get(str(c), np.array([])), dtype=float)
            if idx >= len(arr):
                continue
            X[i, j * 2] = float(arr[idx])
            end = min(len(arr), int(idx + max(1, int(mean_window))))
            seg = arr[idx:end]
            X[i, j * 2 + 1] = float(np.nanmean(seg)) if seg.size else float("nan")
    return X


def stable_feature_importance_by_year(
    mdl_cfg: ModelConfig,
    *,
    ds_pre: pd.DataFrame,
    y: np.ndarray,
    feature_cols: Sequence[str],
    min_year_n: int = 200,
) -> List[Dict[str, Any]]:
    import lightgbm as lgb

    years = ds_pre["_entry_ts"].dt.year.astype(int).to_numpy()
    feats = list(feature_cols)
    imp_by_year: Dict[int, np.ndarray] = {}
    for yr in sorted(np.unique(years)):
        mask = years == int(yr)
        if int(np.sum(mask)) < int(min_year_n):
            continue
        X = ds_pre.loc[mask, feats].to_numpy(dtype=float)
        y2 = np.asarray(y, dtype=int)[mask]
        pos = int(np.sum(y2 == 1))
        neg = int(np.sum(y2 == 0))
        spw = float(neg / max(1, pos))
        model = lgb.LGBMClassifier(
            **{
                **mdl_cfg.lgbm_base_params,
                "num_leaves": int(mdl_cfg.num_leaves_grid[0]),
                "max_depth": int(mdl_cfg.max_depth_grid[0]),
                "min_data_in_leaf": int(mdl_cfg.min_data_in_leaf_grid[0]),
                "scale_pos_weight": float(spw),
            }
        )
        try:
            model.fit(X, y2)
            booster = getattr(model, "booster_", None)
            if booster is None:
                continue
            imp_gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
            imp_by_year[int(yr)] = imp_gain
        except Exception:
            continue
    if not imp_by_year:
        return []
    gains = np.vstack([imp_by_year[yr] for yr in sorted(imp_by_year.keys())])
    mean_gain = np.ravel(np.nanmean(gains, axis=0))
    std_gain = np.ravel(np.nanstd(gains, axis=0))
    stability = mean_gain / (std_gain + 1e-9)
    order = np.lexsort((-mean_gain, -stability))[::-1]
    out: List[Dict[str, Any]] = []
    for idx in order[:10]:
        out.append({"feature": str(feats[int(idx)]), "mean_gain": float(mean_gain[int(idx)]), "std_gain": float(std_gain[int(idx)]), "stability": float(stability[int(idx)])})
    return out


def summarize_big_loss_rules(
    ds_pre: pd.DataFrame,
    *,
    y: np.ndarray,
    side: str,
) -> List[str]:
    feats = ["rsi14", "macd12_hist", "rolling_vol_20", "price_vs_ema20", "adx14"]
    rows: List[Tuple[float, str, float, float]] = []
    y = np.asarray(y, dtype=int)
    for f in feats:
        if f not in ds_pre:
            continue
        a = pd.to_numeric(ds_pre.loc[y == 1, f], errors="coerce")
        b = pd.to_numeric(ds_pre.loc[y == 0, f], errors="coerce")
        if a.empty or b.empty:
            continue
        diff = float(a.median() - b.median())
        rows.append((abs(diff), f, float(a.quantile(0.35)), float(a.quantile(0.65))))
    rows = sorted(rows, key=lambda x: (-x[0], x[1]))[:3]
    rules: List[str] = []
    for _, f, q35, q65 in rows:
        if float(q35) < float(q65):
            rules.append(f"{side}: {f} <= {q35:.4f} 或 {f} >= {q65:.4f} 时更易出现大亏")
        else:
            rules.append(f"{side}: {f} 在极端分位区间更易出现大亏")
    return rules


def score_side_preos_os(
    time_cfg: TimeConfig,
    cv_cfg: CVConfig,
    mdl_cfg: ModelConfig,
    *,
    df_prices: pd.DataFrame,
    ds_all_side: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns ds_side with:
    - p_score: win probability proxy (net_r>0)
    - p_tail: tail probability proxy (mae_r<=-1)
    pre-OS uses purged OOF; OS uses model fit on full pre-OS.
    """
    ds2 = ds_all_side.sort_values("_entry_ts", kind="mergesort").reset_index(drop=True)
    pre0 = to_utc_ts(time_cfg.preos_start_utc)
    pre1 = to_utc_ts(time_cfg.preos_end_utc)
    os0 = to_utc_ts(time_cfg.os_start_utc)
    pre_mask = (ds2["_entry_ts"] >= pre0) & (ds2["_entry_ts"] <= pre1)
    os_mask = ds2["_entry_ts"] >= os0
    ds_pre = ds2.loc[pre_mask].copy()
    ds_os = ds2.loc[os_mask].copy()
    meta: Dict[str, Any] = {"pre_n": int(len(ds_pre)), "os_n": int(len(ds_os))}

    p_win_all = np.full(int(len(ds2)), np.nan, dtype=float)
    p_tail_all = np.full(int(len(ds2)), np.nan, dtype=float)
    if not ds_pre.empty:
        p_pre_win, meta_pre_win = oof_predict_purged(cv_cfg, mdl_cfg, ds_side_pre=ds_pre, target="win")
        p_pre_tail, meta_pre_tail = oof_predict_purged(cv_cfg, mdl_cfg, ds_side_pre=ds_pre, target="tail")
        p_win_all[np.where(pre_mask.to_numpy())[0]] = p_pre_win
        p_tail_all[np.where(pre_mask.to_numpy())[0]] = p_pre_tail
        meta["pre_oof_win"] = meta_pre_win
        meta["pre_oof_tail"] = meta_pre_tail

        # OS scores
        feat_win = list((meta_pre_win or {}).get("features_used") or list(FEATURE_COLS))
        feat_tail = list((meta_pre_tail or {}).get("features_used") or list(FEATURE_COLS))
        X_pre_win = ds_pre[feat_win].to_numpy(dtype=float)
        X_pre_tail = ds_pre[feat_tail].to_numpy(dtype=float)
        y_win_pre = (pd.to_numeric(ds_pre["net_r"], errors="coerce").to_numpy(dtype=float) > 0.0).astype(int)
        y_tail_pre = (pd.to_numeric(ds_pre.get("mae_r"), errors="coerce").to_numpy(dtype=float) <= -1.0).astype(int)
        best_win = (meta_pre_win.get("best") or {}) if isinstance(meta_pre_win, dict) else {}
        best_tail = (meta_pre_tail.get("best") or {}) if isinstance(meta_pre_tail, dict) else {}

        ok_win = int(np.unique(y_win_pre).size) >= 2 and int(len(ds_pre)) >= int(mdl_cfg.min_train_events) and best_win.get("min_data_in_leaf") is not None
        ok_tail = int(np.unique(y_tail_pre).size) >= 2 and int(len(ds_pre)) >= int(mdl_cfg.min_train_events) and best_tail.get("min_data_in_leaf") is not None

        if ok_win or ok_tail:
            try:
                m_full_win = (
                    fit_lgbm_uncalibrated(
                        mdl_cfg,
                        X=X_pre_win,
                        y=y_win_pre,
                        num_leaves=int(best_win["num_leaves"]),
                        max_depth=int(best_win["max_depth"]),
                        min_data_in_leaf=int(best_win["min_data_in_leaf"]),
                        feature_fraction=float(best_win.get("feature_fraction") or 1.0),
                        min_gain_to_split=float(best_win.get("min_gain_to_split") or 0.0),
                    )
                    if ok_win
                    else None
                )
                m_full_tail = (
                    fit_lgbm_uncalibrated(
                        mdl_cfg,
                        X=X_pre_tail,
                        y=y_tail_pre,
                        num_leaves=int(best_tail["num_leaves"]),
                        max_depth=int(best_tail["max_depth"]),
                        min_data_in_leaf=int(best_tail["min_data_in_leaf"]),
                        feature_fraction=float(best_tail.get("feature_fraction") or 1.0),
                        min_gain_to_split=float(best_tail.get("min_gain_to_split") or 0.0),
                    )
                    if ok_tail
                    else None
                )
                if not ds_os.empty:
                    X_os_win = ds_os[feat_win].to_numpy(dtype=float)
                    X_os_tail = ds_os[feat_tail].to_numpy(dtype=float)
                    if m_full_win is not None:
                        p_win_all[np.where(os_mask.to_numpy())[0]] = predict_proba_1(m_full_win, X_os_win)
                    if m_full_tail is not None:
                        p_tail_all[np.where(os_mask.to_numpy())[0]] = predict_proba_1(m_full_tail, X_os_tail)
                meta["full_pre_model"] = {"ok": True, "win": bool(ok_win), "tail": bool(ok_tail)}
            except Exception:
                meta["full_pre_model"] = {"ok": False, "reason": "fit_failed"}
        else:
            meta["full_pre_model"] = {"ok": False, "reason": "insufficient_preos"}

    # fill remaining NaNs with 0.5
    p_win_all[~np.isfinite(p_win_all)] = 0.5
    p_tail_all[~np.isfinite(p_tail_all)] = 0.5
    ds2["p_score"] = p_win_all
    ds2["p_tail"] = p_tail_all
    return ds2, meta


# =============================
# Trading simulation with audits
# =============================


class RollingMax:
    def __init__(self, *, window: timedelta) -> None:
        self.window = window
        self.q: Deque[Tuple[pd.Timestamp, float]] = deque()
        self.maxq: Deque[Tuple[pd.Timestamp, float]] = deque()

    def prune(self, now: pd.Timestamp) -> None:
        cutoff = now - self.window
        while self.q and self.q[0][0] < cutoff:
            t, v = self.q.popleft()
            if self.maxq and self.maxq[0][0] == t and abs(self.maxq[0][1] - v) < 1e-12:
                self.maxq.popleft()

    def add(self, ts: pd.Timestamp, v: float) -> None:
        self.q.append((ts, v))
        while self.maxq and self.maxq[-1][1] <= v + 1e-12:
            self.maxq.pop()
        self.maxq.append((ts, v))

    def max(self) -> float:
        return float(self.maxq[0][1]) if self.maxq else float("nan")


def dd_risk_scale(risk: RiskConfig, dd_usd: float) -> float:
    if float(dd_usd) >= float(risk.dd_trigger_usd) - 1e-12:
        return float(risk.risk_scale_min)
    return 1.0


def lot_for_risk(
    mkt: MarketConfig,
    *,
    sl_dist_risk: float,
    risk_usd_cap: float,
) -> float:
    if not (np.isfinite(sl_dist_risk) and sl_dist_risk > 1e-12 and np.isfinite(risk_usd_cap) and risk_usd_cap > 0):
        return float("nan")
    lot = float(risk_usd_cap / (float(mkt.contract_size) * float(sl_dist_risk)))
    lot = float(min(float(mkt.max_lot), lot))
    step = float(getattr(mkt, "lot_step", 0.01))
    if np.isfinite(step) and step > 1e-12:
        lot = float(math.floor(lot / step + 1e-12) * step)
        # keep stable decimal representation for audits
        lot = float(round(lot, 4))
    if lot < float(mkt.min_lot) - 1e-12:
        return float("nan")
    return float(lot)


def quantile_threshold(hist: Sequence[float], q: float) -> float:
    s = np.asarray(hist, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return -float("inf")
    qq = float(np.clip(float(q), 0.0, 0.999))
    return float(np.quantile(s, qq))


@dataclass(frozen=True)
class FilterConfig:
    q: float
    lookback_days: int
    min_hist: int
    # regime-aware thresholding
    min_regime_hist: int = 60
    q_tail: float = 0.80


@dataclass(frozen=True)
class StrategyConfig:
    exit: ExitConfig
    filt: FilterConfig
    risk_cap_usd: float
    daily_stop_loss_usd: float
    max_parallel_same_dir: int
    tickets_per_signal: int
    cooldown_bars: int


def simulate_trading(
    time_cfg: TimeConfig,
    mkt: MarketConfig,
    risk: RiskConfig,
    *,
    df_prices: pd.DataFrame,
    scored_events: pd.DataFrame,
    strat: StrategyConfig,
    store_thresholds: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if scored_events.empty:
        return scored_events.copy(), {"ok": False, "reason": "no_events"}

    dfp = df_prices
    bt_start = to_utc_ts(time_cfg.backtest_start_utc)
    bt_end = min(to_utc_ts(time_cfg.backtest_end_utc), pd.to_datetime(dfp.index.max(), utc=True))
    pre_end = to_utc_ts(time_cfg.preos_end_utc)
    os_start = to_utc_ts(time_cfg.os_start_utc)

    # NOTE: caller should pre-sort by ["entry_i","_entry_ts"] to avoid repeated sort cost during grid search.
    ev = scored_events
    close = dfp["close"].to_numpy(dtype=float)

    # rolling score history per (side, vol_regime, trend_regime) with fallback to side-only
    lookback = timedelta(days=int(strat.filt.lookback_days))
    hist_side_win: Dict[str, Deque[Tuple[pd.Timestamp, float]]] = {"long": deque(), "short": deque()}
    hist_regime_win: Dict[Tuple[str, int, int], Deque[Tuple[pd.Timestamp, float]]] = defaultdict(deque)
    hist_side_tail: Dict[str, Deque[Tuple[pd.Timestamp, float]]] = {"long": deque(), "short": deque()}
    hist_regime_tail: Dict[Tuple[str, int, int], Deque[Tuple[pd.Timestamp, float]]] = defaultdict(deque)
    thresholds_daily: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)

    equity = float(mkt.initial_capital_usd)
    peak = float(equity)
    max_dd_usd = 0.0
    equity_points: List[Tuple[pd.Timestamp, float]] = [(bt_start, equity)]

    # Segment drawdowns for strict pre-OS selection (do NOT use OS info for pre-OS constraints)
    pre_peak = float(equity)
    pre_max_dd_usd = 0.0
    os_inited = False
    os_start_equity = float("nan")
    os_peak = float("nan")
    os_max_dd_usd = float("nan")

    roll_peak = RollingMax(window=timedelta(days=int(risk.dd_rolling_window_days)))
    roll_peak.add(bt_start, equity)
    cur_year = int(bt_start.year)
    cur_quarter = int((int(bt_start.month) - 1) // 3 + 1)
    year_peak = float(equity)
    year_max_dd = 0.0
    quarter_peak = float(equity)
    quarter_max_dd = 0.0

    # strict stop-out (terminal)
    run_status = "OK"
    stop_out_ts: Optional[pd.Timestamp] = None
    stop_out_bar_i: Optional[int] = None
    stop_trading = False

    # DD stop-open + cooldown + min-risk recovery (non-terminal)
    dd_trigger_count = 0
    dd_trigger_count_roll = 0
    dd_trigger_count_year = 0
    dd_trigger_count_quarter = 0
    dd_stop_armed = True
    stop_open_until_entry_i = -10**9
    risk_scale_changes: List[Dict[str, Any]] = [{"ts": str(bt_start), "risk_scale": 1.0, "dd_now": 0.0}]

    daily_pnl: Dict[pd.Timestamp, float] = {}
    cooldown_until_entry_i = -10**9

    open_trades: Dict[int, Dict[str, Any]] = {}
    open_direction: Optional[int] = None
    cashflows: List[Tuple[int, int, int, str]] = []
    trade_id = 0
    seq = 0

    # execution audit counters
    audit = defaultdict(int)
    audit_regime_trades = defaultdict(int)

    def _seg(ts: pd.Timestamp) -> str:
        if ts <= pre_end:
            return "preOS"
        if ts >= os_start:
            return "OS"
        return "GAP"

    def _prune_hist(h: Deque[Tuple[pd.Timestamp, float]], now: pd.Timestamp) -> None:
        while h and (now - h[0][0]) > lookback:
            h.popleft()

    def _get_threshold_from_hist(
        *,
        ts: pd.Timestamp,
        side: str,
        vr: int,
        tr: int,
        hist_side: Dict[str, Deque[Tuple[pd.Timestamp, float]]],
        hist_regime: Dict[Tuple[str, int, int], Deque[Tuple[pd.Timestamp, float]]],
        q: float,
        default_if_insufficient: float,
    ) -> float:
        h_key = hist_regime[(str(side), int(vr), int(tr))]
        _prune_hist(h_key, ts)
        if len(h_key) >= int(strat.filt.min_regime_hist):
            return float(quantile_threshold([v for _, v in h_key], q=float(q)))
        h_s = hist_side[str(side)]
        _prune_hist(h_s, ts)
        if len(h_s) < int(strat.filt.min_hist):
            return float(default_if_insufficient)
        return float(quantile_threshold([v for _, v in h_s], q=float(q)))

    def _get_threshold_win(ts: pd.Timestamp, side: str, vr: int, tr: int) -> float:
        return _get_threshold_from_hist(
            ts=ts,
            side=side,
            vr=vr,
            tr=tr,
            hist_side=hist_side_win,
            hist_regime=hist_regime_win,
            q=float(strat.filt.q),
            default_if_insufficient=-float("inf"),
        )

    def _get_threshold_tail(ts: pd.Timestamp, side: str, vr: int, tr: int) -> float:
        # accept if p_tail <= thr_tail; with insufficient history, do not filter (thr=+inf)
        return _get_threshold_from_hist(
            ts=ts,
            side=side,
            vr=vr,
            tr=tr,
            hist_side=hist_side_tail,
            hist_regime=hist_regime_tail,
            q=float(strat.filt.q_tail),
            default_if_insufficient=float("inf"),
        )

    def _update_equity(ts: pd.Timestamp, pnl_usd: float) -> None:
        nonlocal equity, peak, max_dd_usd, pre_peak, pre_max_dd_usd, os_inited, os_start_equity, os_peak, os_max_dd_usd
        nonlocal cur_year, cur_quarter, year_peak, year_max_dd, quarter_peak, quarter_max_dd
        # initialize OS baseline at first realized cashflow >= os_start
        if (not os_inited) and ts >= os_start:
            os_inited = True
            os_start_equity = float(equity)  # equity BEFORE applying pnl at this timestamp
            os_peak = float(equity)
            os_max_dd_usd = 0.0
        equity = float(equity + float(pnl_usd))
        equity_points.append((ts, float(equity)))
        peak = max(peak, equity)
        max_dd_usd = max(max_dd_usd, peak - equity)
        year_now = int(ts.year)
        quarter_now = int((int(ts.month) - 1) // 3 + 1)
        if int(year_now) != int(cur_year):
            cur_year = int(year_now)
            year_peak = float(equity)
            year_max_dd = 0.0
            cur_quarter = int(quarter_now)
            quarter_peak = float(equity)
            quarter_max_dd = 0.0
        elif int(quarter_now) != int(cur_quarter):
            cur_quarter = int(quarter_now)
            quarter_peak = float(equity)
            quarter_max_dd = 0.0
        year_peak = max(float(year_peak), float(equity))
        year_max_dd = max(float(year_max_dd), float(year_peak) - float(equity))
        quarter_peak = max(float(quarter_peak), float(equity))
        quarter_max_dd = max(float(quarter_max_dd), float(quarter_peak) - float(equity))
        if ts <= pre_end:
            pre_peak = max(pre_peak, equity)
            pre_max_dd_usd = max(pre_max_dd_usd, pre_peak - equity)
        if os_inited and ts >= os_start:
            os_peak = max(float(os_peak), equity) if np.isfinite(float(os_peak)) else float(equity)
            os_max_dd_usd = max(float(os_max_dd_usd), float(os_peak) - equity)
        roll_peak.prune(ts)
        roll_peak.add(ts, float(equity))

    def _push_cashflow(bar_i: int, tid: int, kind: str) -> None:
        nonlocal seq
        seq += 1
        cashflows.append((int(bar_i), int(seq), int(tid), str(kind)))
        cashflows.sort()

    def _pnl_usd_for_cash_r(trade: Dict[str, Any], cash_r: float) -> float:
        sl_dist_px = float(trade.get("sl_dist", float("nan")))
        lot = float(trade.get("lot", float("nan")))
        if np.isfinite(sl_dist_px) and sl_dist_px > 1e-12 and np.isfinite(lot) and lot > 0:
            pnl_scale = float(lot * float(mkt.contract_size) * sl_dist_px)
        else:
            pnl_scale = float(trade.get("risk_usd", 0.0))
        return float(float(cash_r) * pnl_scale)

    def _stop_out_close_trade(ts: pd.Timestamp, bar_i: int, trd: Dict[str, Any], px: float) -> float:
        direction = int(trd.get("direction", 0))
        sl_dist = float(trd.get("sl_dist", float("nan")))
        entry_px = float(trd.get("entry_price", float("nan")))
        cost_r = float(trd.get("cost_r", 0.0))
        tp1_r = float(trd.get("tp1_r", 1.0))
        frac = float(trd.get("tp1_close_frac", 0.5))
        tp2_mult = float(trd.get("tp2_mult", 2.0))

        if not (np.isfinite(px) and np.isfinite(sl_dist) and sl_dist > 1e-12 and np.isfinite(entry_px) and int(direction) in (-1, 1)):
            rr = -1.0
        else:
            rr = float(((px - entry_px) * float(direction)) / sl_dist)

        if bool(trd.get("_tp1_done", False)):
            # after TP1, runner is protected to BE(cost), so rr cannot go below cost_r
            rr2 = float(np.clip(rr, float(cost_r), float(tp2_mult * tp1_r))) if np.isfinite(rr) else float(cost_r)
            cash_r_exit = float((1.0 - frac) * rr2 - cost_r)
            trd["runner_cash_r"] = float(cash_r_exit)
            tp1_real = float(trd.get("_tp1_cash_r_realized", trd.get("tp1_cash_r", 0.0)))
            trd["net_r"] = float(tp1_real + cash_r_exit)
        else:
            rr2 = float(np.clip(rr, -1.0, float(tp1_r))) if np.isfinite(rr) else -1.0
            cash_r_exit = float(rr2 - cost_r)
            trd["net_r"] = float(cash_r_exit)

        trd["_exit_done"] = True
        trd["_exit_cash_r_realized"] = float(cash_r_exit)
        trd["exit_i"] = int(bar_i)
        trd["exit_time"] = ts
        trd["exit_type"] = "STOP_OUT"
        return float(cash_r_exit)

    def _trigger_stop_out(ts: pd.Timestamp, bar_i: int) -> None:
        nonlocal run_status, stop_out_ts, stop_out_bar_i, stop_trading, open_direction
        if bool(stop_trading):
            return
        run_status = "STOP_OUT"
        stop_out_ts = ts
        stop_out_bar_i = int(bar_i)
        stop_trading = True
        audit[f"stop_out_{_seg(ts)}"] += 1

        px = float(close[int(bar_i)]) if 0 <= int(bar_i) < int(close.size) else float("nan")
        to_close = list(open_trades.values())
        open_trades.clear()
        open_direction = None
        cashflows.clear()
        for trd in to_close:
            if bool(trd.get("_exit_done", False)):
                continue
            cash_r_exit = _stop_out_close_trade(ts, int(bar_i), trd, float(px))
            pnl_usd2 = _pnl_usd_for_cash_r(trd, cash_r_exit)
            day2 = ts.normalize()
            daily_pnl[day2] = float(daily_pnl.get(day2, 0.0) + pnl_usd2)
            _update_equity(ts, pnl_usd2)

    def _apply_cashflow(ts: pd.Timestamp, bar_i: int, trade: Dict[str, Any], kind: str) -> None:
        nonlocal daily_pnl, open_direction, run_status, stop_trading
        if kind == "TP1":
            if bool(trade.get("_tp1_done", False)):
                return
            trade["_tp1_done"] = True
            trade["tp1_reached"] = True
            trade["tp1_time"] = ts
            cash_r = float(trade["tp1_cash_r"])  # no cost here; cost booked at final exit
            trade["_tp1_cash_r_realized"] = float(cash_r)
        else:
            if bool(trade.get("_exit_done", False)):
                return
            if bool(trade.get("_tp1_done", False)):
                cash_r = float(trade["runner_cash_r"])
            else:
                cash_r = float(trade["net_r"])
            trade["_exit_done"] = True
            trade["_exit_cash_r_realized"] = float(cash_r)
            trade["exit_i"] = int(bar_i)
            trade["exit_time"] = ts
            trade["exit_type"] = str(trade.get("exit_type_planned") or "EXIT")
            etp = str(trade.get("exit_type_planned") or "")
            if etp == "TP1":
                trade["tp1_reached"] = True
                trade["tp1_time"] = ts
            if etp == "TP2":
                trade["tp2_reached"] = True

        # cash_r is in R units vs sl_dist (net_r already includes roundtrip cost in R).
        # Convert to USD using sl_dist (NOT sl_dist_risk), otherwise costs get double-counted and DD/stop logic explodes.
        sl_dist_px = float(trade.get("sl_dist", float("nan")))
        lot = float(trade.get("lot", float("nan")))
        if np.isfinite(sl_dist_px) and sl_dist_px > 1e-12 and np.isfinite(lot) and lot > 0:
            pnl_scale = float(lot * float(mkt.contract_size) * sl_dist_px)
        else:
            pnl_scale = float(trade.get("risk_usd", 0.0))
        pnl_usd = float(cash_r * pnl_scale)
        day = ts.normalize()
        daily_pnl[day] = float(daily_pnl.get(day, 0.0) + pnl_usd)
        _update_equity(ts, pnl_usd)
        if (not bool(stop_trading)) and equity <= float(risk.equity_floor_usd) + 1e-12:
            _trigger_stop_out(ts, int(bar_i))

    def _apply_cashflows_before(bar_i: int) -> None:
        nonlocal open_direction
        while cashflows and int(cashflows[0][0]) <= int(bar_i):
            cf_i, _, tid, kind = cashflows.pop(0)
            trd = open_trades.get(int(tid))
            if trd is None:
                continue
            if not (0 <= int(cf_i) < int(len(dfp))):
                continue
            ts = pd.to_datetime(dfp.index[int(cf_i)], utc=True)
            _apply_cashflow(ts, int(cf_i), trd, str(kind))
            if bool(trd.get("_exit_done", False)):
                open_trades.pop(int(tid), None)
        if not open_trades:
            open_direction = None

    trades_out: List[Dict[str, Any]] = []

    for _, r in ev.iterrows():
        if bool(stop_trading):
            break
        entry_ts = pd.to_datetime(r["_entry_ts"], utc=True, errors="coerce")
        if not (pd.notna(entry_ts) and entry_ts.tzinfo is not None):
            continue
        if entry_ts < bt_start or entry_ts > bt_end:
            continue

        audit[f"events_seen_{_seg(entry_ts)}"] += 1

        ei = int(r["entry_i"])
        _apply_cashflows_before(ei)
        if bool(stop_trading):
            break

        gate_pass = bool(r.get("gate_pass", True))
        if not bool(gate_pass):
            audit[f"skipped_gate_{_seg(entry_ts)}"] += 1
            continue

        # score & adaptive thresholds (optional; defaults pass-through)
        score = float(r.get("p_score", 1.0))
        tail_score = float(r.get("p_tail", 0.0))
        side = str(r["side"])
        vr = int(r.get("vol_regime", -1))
        tr = int(r.get("trend_regime", -1))
        key = f"{side}_v{vr}_t{tr}"
        thr_win = _get_threshold_win(entry_ts, side=side, vr=vr, tr=tr)
        thr_tail = _get_threshold_tail(entry_ts, side=side, vr=vr, tr=tr)

        # store thresholds (daily snapshot per key)
        if store_thresholds:
            dkey = entry_ts.normalize().strftime("%Y-%m-%d")
            if key not in thresholds_daily[dkey]:
                thresholds_daily[dkey][key] = {"win": float(thr_win), "tail": float(thr_tail)}

        # update histories (event-level, always)
        if np.isfinite(score):
            hist_side_win[side].append((entry_ts, float(score)))
            hist_regime_win[(side, int(vr), int(tr))].append((entry_ts, float(score)))
        if np.isfinite(tail_score):
            hist_side_tail[side].append((entry_ts, float(tail_score)))
            hist_regime_tail[(side, int(vr), int(tr))].append((entry_ts, float(tail_score)))

        if np.isfinite(tail_score) and tail_score > float(thr_tail) + 1e-12:
            audit[f"skipped_tail_{_seg(entry_ts)}"] += 1
            continue
        if not (np.isfinite(score) and score >= float(thr_win) - 1e-12):
            audit[f"skipped_threshold_{_seg(entry_ts)}"] += 1
            continue

        # daily stop loss (risk mgmt should not affect threshold histories)
        day = entry_ts.normalize()
        if float(daily_pnl.get(day, 0.0)) <= -float(strat.daily_stop_loss_usd) + 1e-12:
            audit[f"skipped_daily_stop_{_seg(entry_ts)}"] += 1
            continue

        # execution cooldown (bars) at signal-level
        if int(strat.cooldown_bars) > 0 and int(ei) < int(cooldown_until_entry_i):
            audit[f"skipped_cooldown_{_seg(entry_ts)}"] += 1
            continue

        # one-direction-at-a-time; allow same-direction parallel
        direction = int(r["direction"])
        if open_direction is not None and direction != int(open_direction):
            audit[f"skipped_open_direction_conflict_{_seg(entry_ts)}"] += 1
            continue
        if int(len(open_trades)) >= int(strat.max_parallel_same_dir):
            audit[f"skipped_max_parallel_{_seg(entry_ts)}"] += 1
            continue

        # DD governor: rolling/year/quarter DD -> stop_open cooldown + min-risk recovery (never permanent)
        roll_peak.prune(entry_ts)
        rp = float(roll_peak.max()) if np.isfinite(float(roll_peak.max())) else float(equity)
        dd_now = float(max(rp, float(equity)) - float(equity))
        dd_total = float(max(0.0, float(peak) - float(equity)))
        dd_total = float(max(0.0, float(peak) - float(equity)))
        dd_total = float(max(0.0, float(peak) - float(equity)))
        if not np.isfinite(dd_now):
            dd_now = 0.0
        dd_total = float(max(0.0, float(peak) - float(equity)))
        dd_year = float(max(0.0, float(year_peak) - float(equity)))
        dd_quarter = float(max(0.0, float(quarter_peak) - float(equity)))
        if int(ei) >= int(stop_open_until_entry_i):
            dd_recover = (
                float(dd_now) <= float(risk.dd_trigger_usd) * float(risk.dd_recover_ratio) + 1e-12
                and float(dd_year) <= float(risk.dd_trigger_usd_year) * float(risk.dd_recover_ratio) + 1e-12
                and float(dd_quarter) <= float(risk.dd_trigger_usd_quarter) * float(risk.dd_recover_ratio) + 1e-12
            )
            if bool(dd_recover):
                dd_stop_armed = True
        dd_exceeded = (
            float(dd_now) > float(risk.dd_trigger_usd) + 1e-12
            or float(dd_total) > float(risk.dd_trigger_usd) + 1e-12
            or float(dd_year) > float(risk.dd_trigger_usd_year) + 1e-12
            or float(dd_quarter) > float(risk.dd_trigger_usd_quarter) + 1e-12
        )
        if bool(dd_stop_armed) and bool(dd_exceeded):
            dd_trigger_count += 1
            if float(dd_now) > float(risk.dd_trigger_usd) + 1e-12:
                dd_trigger_count_roll += 1
            if float(dd_year) > float(risk.dd_trigger_usd_year) + 1e-12:
                dd_trigger_count_year += 1
            if float(dd_quarter) > float(risk.dd_trigger_usd_quarter) + 1e-12:
                dd_trigger_count_quarter += 1
            dd_stop_armed = False
            stop_open_until_entry_i = max(int(stop_open_until_entry_i), int(ei) + int(risk.dd_stop_cooldown_bars))
            risk_scale_changes.append({"ts": str(entry_ts), "risk_scale": float(risk.risk_scale_min), "dd_now": float(dd_now), "event": "cooldown"})
        if int(ei) < int(stop_open_until_entry_i):
            audit[f"skipped_dd_stop_{_seg(entry_ts)}"] += 1
            continue
        rs = float(risk.risk_scale_min) if bool(dd_exceeded) else 1.0

        sl_dist = float(r["sl_dist"])
        sl_dist_risk = float(sl_dist + float(mkt.roundtrip_cost_price) + float(mkt.slippage_buffer_price))
        step = float(getattr(mkt, "lot_step", 0.01))

        tp1_hit = bool(r.get("tp1_hit", False))
        tp2_hit = bool(r.get("tp2_hit", False))
        tp1_close_frac = float(r.get("tp1_close_frac", strat.exit.tp1_close_frac))
        tickets_plan = int(max(1, int(strat.tickets_per_signal)))
        shares = ticket_risk_shares(tickets_per_signal=int(tickets_plan), tp1_close_frac=float(tp1_close_frac))
        opened_any = False

        for j, share in enumerate(shares if int(tickets_plan) > 1 else (1.0,)):
            if int(len(open_trades)) >= int(strat.max_parallel_same_dir):
                break
            if int(tickets_plan) > 1 and not (np.isfinite(float(share)) and float(share) > 1e-12):
                continue

            risk_cap_ticket = float(strat.risk_cap_usd) * float(share) if int(tickets_plan) > 1 else float(strat.risk_cap_usd)
            lot_max = lot_for_risk(mkt, sl_dist_risk=sl_dist_risk, risk_usd_cap=float(risk_cap_ticket))
            if not np.isfinite(lot_max):
                audit[f"skipped_min_lot_{_seg(entry_ts)}"] += 1
                audit[f"skipped_over_risk_cap_{_seg(entry_ts)}"] += 1
                continue

            lot_scaled = float(lot_max * float(rs))
            if np.isfinite(step) and step > 1e-12:
                lot_scaled = float(math.floor(lot_scaled / step + 1e-12) * step)
                lot_scaled = float(round(lot_scaled, 4))
            lot = float(max(float(mkt.min_lot), min(float(lot_max), lot_scaled)))
            risk_usd = float(lot * float(mkt.contract_size) * float(sl_dist_risk))

            ticket_kind = "LEGACY" if int(tickets_plan) <= 1 else ticket_kind_for_index(int(tickets_plan), int(j))
            net_r_ticket = float(r.get("net_r", np.nan))
            exit_i_planned = int(r.get("exit_i", -1))
            exit_type_planned = str(r.get("exit_type", ""))
            tp2_hit_ticket = False

            if int(tickets_plan) > 1:
                if tp1_hit:
                    cost_r = float(r.get("cost_r", 0.0))
                    if str(ticket_kind) == "TP1":
                        net_r_ticket = float(float(r.get("tp1_r", np.nan)) - cost_r)
                        exit_i_planned = int(r.get("tp1_hit_i", -1))
                        exit_type_planned = "TP1"
                    elif str(ticket_kind) == "TP2":
                        net_r_ticket = float(float(r.get("runner_r", np.nan)) - cost_r)
                        exit_i_planned = int(r.get("runner_exit_i", -1))
                        exit_type_planned = str(r.get("runner_exit_type", "TP2"))
                        tp2_hit_ticket = bool(tp2_hit)
                    else:  # TAIL
                        net_r_ticket = float(float(r.get("tail_r", np.nan)) - cost_r)
                        exit_i_planned = int(r.get("tail_exit_i", -1))
                        exit_type_planned = f"TAIL_{str(r.get('tail_exit_type', 'TIME'))}"
                else:
                    # no TP1 => all tickets share base exit
                    net_r_ticket = float(r.get("net_r", np.nan))
                    exit_i_planned = int(r.get("exit_i", -1))
                    exit_type_planned = str(r.get("exit_type", ""))

            if not (np.isfinite(net_r_ticket) and int(exit_i_planned) >= 0):
                continue

            trade_id += 1
            trd = {
                "trade_id": int(trade_id),
                "ticket_kind": str(ticket_kind),
                "ticket_share": float(share) if int(tickets_plan) > 1 else 1.0,
                "tickets_per_signal": int(tickets_plan),
                "side": side,
                "direction": int(direction),
                "vol_regime": int(vr),
                "trend_regime": int(tr),
                "signal_i": int(r["signal_i"]),
                "signal_time": str(r["signal_time"]),
                "entry_i": int(ei),
                "entry_time": entry_ts,
                "entry_price": float(r["entry_price"]),
                "entry_scheme": str(r.get("entry_scheme", "event")),
                "tp1_hit": bool(tp1_hit),
                "tp2_hit": bool(tp2_hit_ticket) if int(tickets_plan) > 1 else bool(tp2_hit),
                "tp1_hit_i": int(r.get("tp1_hit_i", -1)),
                "exit_i_planned": int(exit_i_planned),
                "exit_time_planned": str(dfp.index[int(exit_i_planned)]) if 0 <= int(exit_i_planned) < int(len(dfp)) else "",
                "exit_type_planned": str(exit_type_planned),
                "H": int(r.get("H", 0)),
                "tp1_atr_mult": float(r.get("tp1_atr_mult", np.nan)),
                "sl_atr_mult": float(r.get("sl_atr_mult", np.nan)),
                "tp1_close_frac": float(tp1_close_frac),
                "tp2_mult": float(r.get("tp2_mult", np.nan)),
                "score": float(score),
                "threshold": float(thr_win),
                "tail_score": float(tail_score),
                "tail_threshold": float(thr_tail),
                "risk_scale": float(rs),
                "sl_dist": float(sl_dist),
                "tp1_r": float(r.get("tp1_r", np.nan)),
                "cost_r": float(r.get("cost_r", np.nan)),
                "net_r": float(net_r_ticket),
                "tp1_cash_r": float(r.get("tp1_cash_r", 0.0)) if int(tickets_plan) <= 1 else 0.0,
                "runner_cash_r": float(r.get("runner_cash_r", 0.0)) if int(tickets_plan) <= 1 else 0.0,
                "lot": float(lot),
                "risk_usd": float(risk_usd),
                "pnl_usd": 0.0,
                "tp1_reached": False,
                "tp2_reached": False,
                "_tp1_done": False,
                "_exit_done": False,
            }
            open_trades[int(trade_id)] = trd

            if open_direction is None:
                open_direction = int(direction)

            audit[f"trades_opened_{_seg(entry_ts)}"] += 1
            audit_regime_trades[key] += 1

            if int(tickets_plan) <= 1:
                if bool(trd["tp1_hit"]):
                    _push_cashflow(int(trd["tp1_hit_i"]), int(trade_id), "TP1")
                _push_cashflow(int(trd["exit_i_planned"]), int(trade_id), "EXIT")
            else:
                _push_cashflow(int(trd["exit_i_planned"]), int(trade_id), "EXIT")

            trades_out.append(trd)
            opened_any = True

        if opened_any:
            audit[f"signals_opened_{_seg(entry_ts)}"] += 1
            cooldown_until_entry_i = int(ei) + int(max(0, int(strat.cooldown_bars)))

    # flush remaining
    _apply_cashflows_before(int(len(dfp) - 1))

    stop_open_until_ts = None
    if int(stop_open_until_entry_i) > 0 and int(stop_open_until_entry_i) < int(len(dfp)):
        stop_open_until_ts = str(dfp.index[int(stop_open_until_entry_i)])
    stop_open_active_end = bool(int(stop_open_until_entry_i) > int(len(dfp) - 1))

    out = pd.DataFrame(trades_out)
    if out.empty:
        meta = {
            "ok": True,
            "audit": dict(audit),
            "run_status": str(run_status),
            "stop_out_ts": str(stop_out_ts) if stop_out_ts is not None else None,
            "stop_out_bar_i": int(stop_out_bar_i) if stop_out_bar_i is not None else None,
            "dd_trigger_count": int(dd_trigger_count),
            "dd_trigger_count_roll": int(dd_trigger_count_roll),
            "dd_trigger_count_year": int(dd_trigger_count_year),
            "dd_trigger_count_quarter": int(dd_trigger_count_quarter),
            "dd_stop_skip": 0,
            "risk_scale_min": float(risk.risk_scale_min),
            "risk_scale_changes": list(risk_scale_changes),
            "dd_rolling_window_days": int(risk.dd_rolling_window_days),
            "dd_stop_cooldown_bars": int(risk.dd_stop_cooldown_bars),
            "dd_trigger_usd": float(risk.dd_trigger_usd),
            "dd_trigger_usd_year": float(risk.dd_trigger_usd_year),
            "dd_trigger_usd_quarter": float(risk.dd_trigger_usd_quarter),
            "stop_open_until_ts": stop_open_until_ts,
            "stop_open_active_end": bool(stop_open_active_end),
            "thresholds_daily": thresholds_daily if store_thresholds else None,
            "equity_end": float(equity),
            "max_dd_usd": float(max_dd_usd),
            "max_dd_usd_year": float(year_max_dd),
            "max_dd_usd_quarter": float(quarter_max_dd),
            "max_dd_usd_preos": float(pre_max_dd_usd),
            "max_dd_usd_os": float(os_max_dd_usd) if np.isfinite(os_max_dd_usd) else float("nan"),
            "os_start_equity": float(os_start_equity) if np.isfinite(os_start_equity) else float("nan"),
        }
        return out, meta

    # Reporting PnL: net_r is in R units vs sl_dist.
    out["pnl_usd"] = out["net_r"].astype(float) * out["sl_dist"].astype(float) * float(mkt.contract_size) * out["lot"].astype(float)

    meta = {
        "ok": True,
        "audit": dict(audit),
        "audit_regime_trades": dict(audit_regime_trades),
        "run_status": str(run_status),
        "stop_out_ts": str(stop_out_ts) if stop_out_ts is not None else None,
        "stop_out_bar_i": int(stop_out_bar_i) if stop_out_bar_i is not None else None,
        "dd_trigger_count": int(dd_trigger_count),
        "dd_trigger_count_roll": int(dd_trigger_count_roll),
        "dd_trigger_count_year": int(dd_trigger_count_year),
        "dd_trigger_count_quarter": int(dd_trigger_count_quarter),
        "dd_stop_skip": 0,
        "risk_scale_min": float(risk.risk_scale_min),
        "risk_scale_changes": list(risk_scale_changes),
        "dd_rolling_window_days": int(risk.dd_rolling_window_days),
        "dd_stop_cooldown_bars": int(risk.dd_stop_cooldown_bars),
        "dd_trigger_usd": float(risk.dd_trigger_usd),
        "dd_trigger_usd_year": float(risk.dd_trigger_usd_year),
        "dd_trigger_usd_quarter": float(risk.dd_trigger_usd_quarter),
        "stop_open_until_ts": stop_open_until_ts,
        "stop_open_active_end": bool(stop_open_active_end),
        "thresholds_daily": thresholds_daily if store_thresholds else None,
        "equity_end": float(equity),
        "max_dd_usd": float(max_dd_usd),
        "max_dd_usd_year": float(year_max_dd),
        "max_dd_usd_quarter": float(quarter_max_dd),
        "max_dd_pct": float(max_dd_usd / float(mkt.initial_capital_usd) * 100.0) if mkt.initial_capital_usd > 0 else float("nan"),
        "max_dd_usd_preos": float(pre_max_dd_usd),
        "max_dd_pct_preos": float(pre_max_dd_usd / float(mkt.initial_capital_usd) * 100.0) if mkt.initial_capital_usd > 0 else float("nan"),
        "os_start_equity": float(os_start_equity) if np.isfinite(os_start_equity) else float("nan"),
        "max_dd_usd_os": float(os_max_dd_usd) if np.isfinite(os_max_dd_usd) else float("nan"),
        "max_dd_pct_os": float(os_max_dd_usd / float(os_start_equity) * 100.0)
        if (np.isfinite(os_max_dd_usd) and np.isfinite(os_start_equity) and os_start_equity > 0)
        else float("nan"),
    }
    return out, meta


# =============================
# Fast grid simulation (metrics-only)
# =============================


_NS_PER_DAY = 86_400_000_000_000


def _dt_series_to_ns_utc(s: pd.Series) -> np.ndarray:
    """
    Convert a datetime-like Series (tz-aware preferred) to int64 ns (UTC).
    """
    try:
        return s.view("int64").to_numpy(dtype=np.int64, copy=False)
    except Exception:
        s2 = pd.to_datetime(s, utc=True, errors="coerce")
        return s2.view("int64").to_numpy(dtype=np.int64, copy=False)


def _dt_index_to_ns_utc(idx: pd.DatetimeIndex) -> np.ndarray:
    try:
        return idx.view("int64").astype(np.int64, copy=False)
    except Exception:
        idx2 = pd.to_datetime(idx, utc=True, errors="coerce")
        return idx2.view("int64").astype(np.int64, copy=False)


@dataclass(frozen=True)
class ScoredEventArrays:
    signal_i: np.ndarray
    entry_i: np.ndarray
    entry_ts_ns: np.ndarray
    side: np.ndarray
    direction: np.ndarray
    vol_regime: np.ndarray
    trend_regime: np.ndarray
    p_score: np.ndarray
    p_tail: np.ndarray
    sl_dist: np.ndarray
    sl_dist_risk: np.ndarray
    tp1_r: np.ndarray
    cost_r: np.ndarray
    tp1_hit: np.ndarray
    tp2_hit: np.ndarray
    tp1_hit_i: np.ndarray
    exit_i: np.ndarray
    runner_r: np.ndarray
    runner_exit_i: np.ndarray
    tail_r: np.ndarray
    tail_exit_i: np.ndarray
    tp1_cash_r: np.ndarray
    runner_cash_r: np.ndarray
    net_r: np.ndarray
    mae_r: np.ndarray
    adx14: np.ndarray


@dataclass(frozen=True)
class FastSimCtx:
    idx_ts_ns: np.ndarray
    year_by_bar: np.ndarray
    bt_start_ns: int
    bt_end_ns: int
    pre_start_ns: int
    pre_end_ns: int
    os_start_ns: int
    days_pre: float
    days_os: float
    days_all: float


def build_scored_event_arrays(scored: pd.DataFrame, *, mkt: MarketConfig) -> ScoredEventArrays:
    ev = scored.sort_values(["entry_i", "_entry_ts"], kind="mergesort").reset_index(drop=True)
    signal_i = pd.to_numeric(ev.get("signal_i"), errors="coerce").fillna(-1).astype(int).to_numpy(dtype=np.int32)
    entry_i = pd.to_numeric(ev["entry_i"], errors="coerce").fillna(-1).astype(int).to_numpy(dtype=np.int32)
    entry_ts_ns = _dt_series_to_ns_utc(pd.to_datetime(ev["_entry_ts"], utc=True, errors="coerce"))
    side = ev["side"].astype(str).to_numpy(dtype=object)
    direction = pd.to_numeric(ev["direction"], errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int8)
    vol_regime = pd.to_numeric(ev.get("vol_regime"), errors="coerce").fillna(-1).astype(int).to_numpy(dtype=np.int8)
    trend_regime = pd.to_numeric(ev.get("trend_regime"), errors="coerce").fillna(-1).astype(int).to_numpy(dtype=np.int8)
    p_score = pd.to_numeric(ev.get("p_score"), errors="coerce").to_numpy(dtype=float)
    p_tail = pd.to_numeric(ev.get("p_tail"), errors="coerce").to_numpy(dtype=float)
    p_score[~np.isfinite(p_score)] = 1.0
    p_tail[~np.isfinite(p_tail)] = 0.0
    sl_dist = pd.to_numeric(ev.get("sl_dist"), errors="coerce").to_numpy(dtype=float)
    cost_total_px = float(mkt.roundtrip_cost_price) + float(mkt.slippage_buffer_price)
    sl_dist_risk = sl_dist + float(cost_total_px)
    tp1_r = pd.to_numeric(ev.get("tp1_r"), errors="coerce").to_numpy(dtype=float)
    cost_r = pd.to_numeric(ev.get("cost_r"), errors="coerce").to_numpy(dtype=float)
    tp1_hit = ev.get("tp1_hit").astype(int).to_numpy(dtype=np.int8) > 0
    tp2_hit = ev.get("tp2_hit").astype(int).to_numpy(dtype=np.int8) > 0
    tp1_hit_i = pd.to_numeric(ev.get("tp1_hit_i"), errors="coerce").fillna(-1).astype(int).to_numpy(dtype=np.int32)
    exit_i = pd.to_numeric(ev.get("exit_i"), errors="coerce").fillna(-1).astype(int).to_numpy(dtype=np.int32)
    runner_r = pd.to_numeric(ev.get("runner_r"), errors="coerce").to_numpy(dtype=float)
    runner_exit_i = pd.to_numeric(ev.get("runner_exit_i"), errors="coerce").fillna(-1).astype(int).to_numpy(dtype=np.int32)
    tail_r = pd.to_numeric(ev.get("tail_r"), errors="coerce").to_numpy(dtype=float)
    tail_exit_i = pd.to_numeric(ev.get("tail_exit_i"), errors="coerce").fillna(-1).astype(int).to_numpy(dtype=np.int32)
    tp1_cash_r = pd.to_numeric(ev.get("tp1_cash_r"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    runner_cash_r = pd.to_numeric(ev.get("runner_cash_r"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    net_r = pd.to_numeric(ev.get("net_r"), errors="coerce").to_numpy(dtype=float)
    mae_r = pd.to_numeric(ev.get("mae_r"), errors="coerce").to_numpy(dtype=float)
    adx14 = pd.to_numeric(ev.get("adx14"), errors="coerce").to_numpy(dtype=float)
    return ScoredEventArrays(
        signal_i=signal_i,
        entry_i=entry_i,
        entry_ts_ns=entry_ts_ns,
        side=side,
        direction=direction,
        vol_regime=vol_regime,
        trend_regime=trend_regime,
        p_score=p_score,
        p_tail=p_tail,
        sl_dist=sl_dist,
        sl_dist_risk=sl_dist_risk,
        tp1_r=tp1_r,
        cost_r=cost_r,
        tp1_hit=tp1_hit,
        tp2_hit=tp2_hit,
        tp1_hit_i=tp1_hit_i,
        exit_i=exit_i,
        runner_r=runner_r,
        runner_exit_i=runner_exit_i,
        tail_r=tail_r,
        tail_exit_i=tail_exit_i,
        tp1_cash_r=tp1_cash_r,
        runner_cash_r=runner_cash_r,
        net_r=net_r,
        mae_r=mae_r,
        adx14=adx14,
    )


class RollingMaxNS:
    def __init__(self, *, window_ns: int) -> None:
        self.window_ns = int(window_ns)
        self.q: Deque[Tuple[int, float]] = deque()
        self.maxq: Deque[Tuple[int, float]] = deque()

    def prune(self, now_ns: int) -> None:
        cutoff = int(now_ns) - int(self.window_ns)
        while self.q and int(self.q[0][0]) < cutoff:
            t, v = self.q.popleft()
            if self.maxq and int(self.maxq[0][0]) == int(t) and abs(float(self.maxq[0][1]) - float(v)) < 1e-12:
                self.maxq.popleft()

    def add(self, ts_ns: int, v: float) -> None:
        t = int(ts_ns)
        vv = float(v)
        self.q.append((t, vv))
        while self.maxq and float(self.maxq[-1][1]) <= vv + 1e-12:
            self.maxq.pop()
        self.maxq.append((t, vv))

    def max(self) -> float:
        return float(self.maxq[0][1]) if self.maxq else float("nan")


def lot_max_for_risk_cap(mkt: MarketConfig, *, sl_dist_risk: np.ndarray, risk_cap_usd: float) -> np.ndarray:
    sdr = np.asarray(sl_dist_risk, dtype=float)
    ok = np.isfinite(sdr) & (sdr > 1e-12) & np.isfinite(float(risk_cap_usd)) & (float(risk_cap_usd) > 0)
    lot = np.full(int(sdr.size), np.nan, dtype=float)
    if int(np.sum(ok)) == 0:
        return lot
    denom = float(mkt.contract_size) * sdr[ok]
    lot0 = float(risk_cap_usd) / denom
    lot0 = np.minimum(float(mkt.max_lot), lot0)
    step = float(getattr(mkt, "lot_step", 0.01))
    if np.isfinite(step) and step > 1e-12:
        lot0 = np.floor(lot0 / step + 1e-12) * step
        lot0 = np.round(lot0, 4)
    lot[ok] = lot0.astype(float)
    return lot


def compute_filter_masks(arr: ScoredEventArrays, *, filt: FilterConfig) -> Dict[str, np.ndarray]:
    """
    Walk-forward, regime-aware masks:
    - win_ok: score >= thr_win
    - tail_ok: p_tail <= thr_tail (or NaN)
    - pass: win_ok & tail_ok

    Important: histories update on all events (independent of risk mgmt), matching simulate_trading.
    """
    n = int(len(arr.entry_i))
    win_ok = np.zeros(n, dtype=bool)
    tail_ok = np.ones(n, dtype=bool)
    lookback_ns = int(filt.lookback_days) * int(_NS_PER_DAY)

    hist_side_win: Dict[str, Deque[Tuple[int, float]]] = {"long": deque(), "short": deque()}
    hist_regime_win: Dict[Tuple[str, int, int], Deque[Tuple[int, float]]] = defaultdict(deque)
    hist_side_tail: Dict[str, Deque[Tuple[int, float]]] = {"long": deque(), "short": deque()}
    hist_regime_tail: Dict[Tuple[str, int, int], Deque[Tuple[int, float]]] = defaultdict(deque)

    def _prune(dq: Deque[Tuple[int, float]], cutoff_ns: int) -> None:
        while dq and int(dq[0][0]) < int(cutoff_ns):
            dq.popleft()

    def _thr(
        *,
        ts_ns: int,
        side: str,
        vr: int,
        tr: int,
        q: float,
        default_if_insufficient: float,
        hist_side: Dict[str, Deque[Tuple[int, float]]],
        hist_regime: Dict[Tuple[str, int, int], Deque[Tuple[int, float]]],
    ) -> float:
        cutoff = int(ts_ns) - int(lookback_ns)
        hs = hist_side.get(side)
        if hs is None:
            hs = deque()
            hist_side[side] = hs
        _prune(hs, cutoff)
        hk = hist_regime[(str(side), int(vr), int(tr))]
        _prune(hk, cutoff)
        if len(hk) >= int(filt.min_regime_hist):
            return float(quantile_threshold([v for _, v in hk], q=float(q)))
        if len(hs) < int(filt.min_hist):
            return float(default_if_insufficient)
        return float(quantile_threshold([v for _, v in hs], q=float(q)))

    for i in range(n):
        ts_ns = int(arr.entry_ts_ns[i])
        side = str(arr.side[i])
        vr = int(arr.vol_regime[i])
        tr = int(arr.trend_regime[i])
        score = float(arr.p_score[i]) if np.isfinite(arr.p_score[i]) else float("nan")
        tail = float(arr.p_tail[i]) if np.isfinite(arr.p_tail[i]) else float("nan")

        thr_win = _thr(
            ts_ns=ts_ns,
            side=side,
            vr=vr,
            tr=tr,
            q=float(filt.q),
            default_if_insufficient=-float("inf"),
            hist_side=hist_side_win,
            hist_regime=hist_regime_win,
        )
        thr_tail = _thr(
            ts_ns=ts_ns,
            side=side,
            vr=vr,
            tr=tr,
            q=float(filt.q_tail),
            default_if_insufficient=float("inf"),
            hist_side=hist_side_tail,
            hist_regime=hist_regime_tail,
        )

        # update histories after threshold snapshot
        if np.isfinite(score):
            hist_side_win[side].append((ts_ns, float(score)))
            hist_regime_win[(side, int(vr), int(tr))].append((ts_ns, float(score)))
        if np.isfinite(tail):
            hist_side_tail[side].append((ts_ns, float(tail)))
            hist_regime_tail[(side, int(vr), int(tr))].append((ts_ns, float(tail)))

        okw = bool(np.isfinite(score) and score >= float(thr_win) - 1e-12)
        okt = not (np.isfinite(tail) and tail > float(thr_tail) + 1e-12)
        win_ok[i] = bool(okw)
        tail_ok[i] = bool(okt)

    return {"win_ok": win_ok, "tail_ok": tail_ok, "pass": (win_ok & tail_ok)}


def compute_filter_pass_mask(arr: ScoredEventArrays, *, filt: FilterConfig) -> np.ndarray:
    return compute_filter_masks(arr, filt=filt)["pass"]


def _metrics_from_acc(
    *,
    signals_n: int,
    tickets_n: int,
    tp1_signals: int,
    tp2_signals: int,
    pos_sum: float,
    neg_sum: float,
    net_sum: float,
    days: float,
) -> Dict[str, float]:
    if int(signals_n) <= 0 or int(tickets_n) <= 0:
        return {"epd": 0.0, "tpd": 0.0, "hit_tp1": float("nan"), "hit_tp2": float("nan"), "pf": float("nan"), "ev_r": float("nan")}
    epd = float(float(signals_n) / max(1.0, float(days)))
    tpd = float(float(tickets_n) / max(1.0, float(days)))
    hit1 = float(int(tp1_signals) / max(1, int(signals_n)))
    hit2 = float(int(tp2_signals) / max(1, int(signals_n)))
    pf = float(np.sum(pos_sum) / max(1e-12, abs(float(neg_sum)))) if (pos_sum > 0 and neg_sum < 0) else float("nan")
    ev = float(net_sum / max(1, int(tickets_n))) if np.isfinite(net_sum) else float("nan")
    return {"epd": epd, "tpd": tpd, "hit_tp1": hit1, "hit_tp2": hit2, "pf": float(pf), "ev_r": ev}


def ticket_risk_shares(*, tickets_per_signal: int, tp1_close_frac: float) -> Tuple[float, ...]:
    n = int(max(1, tickets_per_signal))
    frac = float(tp1_close_frac)
    if not np.isfinite(frac):
        frac = 0.5
    frac = float(np.clip(frac, 0.0, 1.0))
    if n <= 1:
        return (1.0,)
    if n == 2:
        return (frac, float(max(0.0, 1.0 - frac)))
    rem = float(max(0.0, 1.0 - frac))
    # fixed split for tail vs TP2 to keep KISS (tail is smaller).
    return (frac, rem * 0.6, rem * 0.4)


def ticket_kind_for_index(n: int, j: int) -> str:
    if int(n) <= 1:
        return "LEGACY"
    if int(j) <= 0:
        return "TP1"
    if int(j) == 1:
        return "TP2"
    return "TAIL"


def ticket_net_r_and_exit_i(arr: ScoredEventArrays, *, idx: int, kind: str) -> Tuple[float, int]:
    i = int(idx)
    tp1_hit = bool(arr.tp1_hit[i])
    if not tp1_hit:
        return float(arr.net_r[i]), int(arr.exit_i[i])

    cost_r = float(arr.cost_r[i]) if np.isfinite(arr.cost_r[i]) else 0.0
    if str(kind) == "TP1":
        tp1_r = float(arr.tp1_r[i]) if np.isfinite(arr.tp1_r[i]) else float("nan")
        exit_i = int(arr.tp1_hit_i[i]) if int(arr.tp1_hit_i[i]) >= 0 else int(arr.exit_i[i])
        return float(tp1_r - cost_r), int(exit_i)
    if str(kind) == "TP2":
        rr = float(arr.runner_r[i]) if np.isfinite(arr.runner_r[i]) else float(cost_r)
        exit_i = int(arr.runner_exit_i[i]) if int(arr.runner_exit_i[i]) >= 0 else int(arr.exit_i[i])
        return float(rr - cost_r), int(exit_i)
    # TAIL
    rr = float(arr.tail_r[i]) if np.isfinite(arr.tail_r[i]) else (float(arr.runner_r[i]) if np.isfinite(arr.runner_r[i]) else float(cost_r))
    exit_i = int(arr.tail_exit_i[i]) if int(arr.tail_exit_i[i]) >= 0 else int(arr.exit_i[i])
    return float(rr - cost_r), int(exit_i)


def simulate_trading_fast_metrics(
    ctx: FastSimCtx,
    mkt: MarketConfig,
    risk: RiskConfig,
    *,
    arr: ScoredEventArrays,
    pass_indices: np.ndarray,
    lot_max_by_ticket: Sequence[np.ndarray],
    daily_stop_loss_usd: float,
    max_parallel_same_dir: int,
    tickets_per_signal: int,
    tp1_close_frac: float,
    cooldown_bars: int,
    with_breakdowns: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """
    Fast event-driven simulator for grid search: computes segment metrics + DD meta, without building per-trade rows.
    Assumes filter pass mask is precomputed and independent of risk mgmt (simulate_trading updates histories on all events).
    """
    if pass_indices.size == 0:
        empty_m = {"epd": 0.0, "tpd": 0.0, "hit_tp1": float("nan"), "hit_tp2": float("nan"), "pf": float("nan"), "ev_r": float("nan")}
        meta = {
            "ok": True,
            "run_status": "OK",
            "stop_out_ts": None,
            "stop_out_bar_i": None,
            "dd_trigger_count": 0,
            "dd_trigger_count_roll": 0,
            "dd_trigger_count_year": 0,
            "dd_trigger_count_quarter": 0,
            "dd_stop_skip": 0,
            "risk_scale_min": float(risk.risk_scale_min),
            "dd_rolling_window_days": int(risk.dd_rolling_window_days),
            "dd_stop_cooldown_bars": int(risk.dd_stop_cooldown_bars),
            "dd_trigger_usd": float(risk.dd_trigger_usd),
            "dd_trigger_usd_year": float(risk.dd_trigger_usd_year),
            "dd_trigger_usd_quarter": float(risk.dd_trigger_usd_quarter),
            "stop_open_until_ts": None,
            "stop_open_active_end": False,
            "max_dd_usd": 0.0,
            "max_dd_pct": 0.0,
            "max_dd_usd_year": 0.0,
            "max_dd_usd_quarter": 0.0,
            "max_dd_usd_preos": 0.0,
            "max_dd_pct_preos": 0.0,
            "os_start_equity": float("nan"),
            "max_dd_usd_os": float("nan"),
            "max_dd_pct_os": float("nan"),
        }
        if bool(with_breakdowns):
            meta["preos_year_table"] = []
            meta["preos_regime_adx3_table"] = []
            meta["ev_r_preos_p25"] = float("nan")
        return empty_m, empty_m, empty_m, meta

    equity = float(mkt.initial_capital_usd)
    peak = float(equity)
    max_dd_usd = 0.0
    pre_peak = float(equity)
    pre_max_dd_usd = 0.0

    os_inited = False
    os_start_equity = float("nan")
    os_peak = float("nan")
    os_max_dd_usd = float("nan")

    run_status = "OK"
    stop_out_ts_ns: Optional[int] = None
    stop_out_bar_i: Optional[int] = None
    stop_trading = False

    # DD governor (rolling peak of realized equity) + stop_open cooldown
    roll_peak = RollingMaxNS(window_ns=int(risk.dd_rolling_window_days) * int(_NS_PER_DAY))
    dd_trigger_count = 0
    dd_trigger_count_roll = 0
    dd_trigger_count_year = 0
    dd_trigger_count_quarter = 0
    dd_stop_armed = True
    stop_open_until_entry_i = -10**9
    cur_year = int(pd.Timestamp(int(ctx.bt_start_ns), unit="ns", tz="UTC").year)
    cur_quarter = int((int(pd.Timestamp(int(ctx.bt_start_ns), unit="ns", tz="UTC").month) - 1) // 3 + 1)
    year_peak = float(equity)
    year_max_dd = 0.0
    quarter_peak = float(equity)
    quarter_max_dd = 0.0

    # position constraints
    open_direction: Optional[int] = None
    open_count = 0

    # cashflows heap: (bar_i, seq, cash_usd, is_exit)
    cashflows: List[Tuple[int, int, float, bool]] = []
    seq = 0

    daily_pnl: Dict[int, float] = {}

    # Pre-OS stability breakdowns (year + ADX3 regime) to reduce multiple-testing survivor bias.
    # NOTE: computed on executed trades only (post filter + execution constraints), and uses entry-time buckets.
    year_acc: Dict[int, Dict[str, Any]] = {}
    reg_acc: Dict[int, Dict[str, Any]] = {}
    pre0_ts = pd.Timestamp(int(ctx.pre_start_ns), unit="ns", tz="UTC") if bool(with_breakdowns) else None
    pre1_ts = pd.Timestamp(int(ctx.pre_end_ns), unit="ns", tz="UTC") if bool(with_breakdowns) else None

    def _acc_init() -> Dict[str, Any]:
        return {"signals": 0, "tickets": 0, "tp1_signals": 0, "tp2_signals": 0, "pos_sum": 0.0, "neg_sum": 0.0, "net_sum": 0.0}

    def _get_acc(d: Dict[int, Dict[str, Any]], key: int) -> Dict[str, Any]:
        a = d.get(int(key))
        if a is None:
            a = _acc_init()
            d[int(key)] = a
        return a

    def _acc_ticket(a: Dict[str, Any], net_r: float) -> None:
        a["tickets"] = int(a.get("tickets", 0)) + 1
        a["net_sum"] = float(a.get("net_sum", 0.0) + float(net_r))
        if float(net_r) > 0:
            a["pos_sum"] = float(a.get("pos_sum", 0.0) + float(net_r))
        elif float(net_r) < 0:
            a["neg_sum"] = float(a.get("neg_sum", 0.0) + float(net_r))

    def _acc_signal(a: Dict[str, Any], tp1_h: int, tp2_h: int) -> None:
        a["signals"] = int(a.get("signals", 0)) + 1
        a["tp1_signals"] = int(a.get("tp1_signals", 0)) + int(tp1_h)
        a["tp2_signals"] = int(a.get("tp2_signals", 0)) + int(tp2_h)

    # metrics accumulators (pre / os / all by signal entry_time)
    pre_signals = 0
    pre_tickets = 0
    pre_tp1_signals = 0
    pre_tp2_signals = 0
    pre_pos = 0.0
    pre_neg = 0.0
    pre_sum = 0.0

    os_signals = 0
    os_tickets = 0
    os_tp1_signals = 0
    os_tp2_signals = 0
    os_pos = 0.0
    os_neg = 0.0
    os_sum = 0.0

    all_signals = 0
    all_tickets = 0
    all_tp1_signals = 0
    all_tp2_signals = 0
    all_pos = 0.0
    all_neg = 0.0
    all_sum = 0.0

    step = float(getattr(mkt, "lot_step", 0.01))
    contract = float(mkt.contract_size)
    next_allowed_entry_i = -10**9

    def _apply_cashflows_upto(bar_i: int) -> None:
        nonlocal equity, peak, max_dd_usd, pre_peak, pre_max_dd_usd, os_inited, os_start_equity, os_peak, os_max_dd_usd
        nonlocal run_status, stop_out_ts_ns, stop_out_bar_i, stop_trading, open_count, open_direction
        nonlocal cur_year, cur_quarter, year_peak, year_max_dd, quarter_peak, quarter_max_dd
        while cashflows and int(cashflows[0][0]) <= int(bar_i):
            cf_bar_i, _, cash_usd, is_exit = heapq.heappop(cashflows)
            if not (0 <= int(cf_bar_i) < int(ctx.idx_ts_ns.size)):
                continue
            ts_ns = int(ctx.idx_ts_ns[int(cf_bar_i)])
            day_id = int(ts_ns // int(_NS_PER_DAY))
            daily_pnl[day_id] = float(daily_pnl.get(day_id, 0.0) + float(cash_usd))

            if (not os_inited) and ts_ns >= int(ctx.os_start_ns):
                os_inited = True
                os_start_equity = float(equity)
                os_peak = float(equity)
                os_max_dd_usd = 0.0

            equity = float(equity + float(cash_usd))
            peak = max(float(peak), float(equity))
            max_dd_usd = max(float(max_dd_usd), float(peak) - float(equity))
            ts_pd = pd.Timestamp(int(ts_ns), tz="UTC")
            year_now = int(ts_pd.year)
            quarter_now = int((int(ts_pd.month) - 1) // 3 + 1)
            if int(year_now) != int(cur_year):
                cur_year = int(year_now)
                year_peak = float(equity)
                year_max_dd = 0.0
                cur_quarter = int(quarter_now)
                quarter_peak = float(equity)
                quarter_max_dd = 0.0
            elif int(quarter_now) != int(cur_quarter):
                cur_quarter = int(quarter_now)
                quarter_peak = float(equity)
                quarter_max_dd = 0.0
            year_peak = max(float(year_peak), float(equity))
            year_max_dd = max(float(year_max_dd), float(year_peak) - float(equity))
            quarter_peak = max(float(quarter_peak), float(equity))
            quarter_max_dd = max(float(quarter_max_dd), float(quarter_peak) - float(equity))
            if ts_ns <= int(ctx.pre_end_ns):
                pre_peak = max(float(pre_peak), float(equity))
                pre_max_dd_usd = max(float(pre_max_dd_usd), float(pre_peak) - float(equity))
            if os_inited and ts_ns >= int(ctx.os_start_ns):
                os_peak = max(float(os_peak), float(equity)) if np.isfinite(float(os_peak)) else float(equity)
                os_max_dd_usd = max(float(os_max_dd_usd), float(os_peak) - float(equity))

            roll_peak.prune(ts_ns)
            roll_peak.add(ts_ns, float(equity))

            if bool(is_exit):
                open_count = int(max(0, int(open_count) - 1))
                if int(open_count) == 0:
                    open_direction = None

            if (not bool(stop_trading)) and float(equity) <= float(risk.equity_floor_usd) + 1e-12:
                stop_trading = True
                run_status = "STOP_OUT"
                stop_out_ts_ns = int(ts_ns)
                stop_out_bar_i = int(cf_bar_i)

            # stop_out disabled in fast sim (use stop_open cooldown instead)

    for idx in pass_indices.astype(int):
        if not (0 <= int(idx) < int(arr.entry_i.size)):
            continue
        entry_i = int(arr.entry_i[int(idx)])
        entry_ts_ns = int(arr.entry_ts_ns[int(idx)])
        if entry_ts_ns < int(ctx.bt_start_ns) or entry_ts_ns > int(ctx.bt_end_ns):
            continue

        _apply_cashflows_upto(int(entry_i))
        if bool(stop_trading):
            continue

        # daily stop loss
        day_id = int(entry_ts_ns // int(_NS_PER_DAY))
        if float(daily_pnl.get(day_id, 0.0)) <= -float(daily_stop_loss_usd) + 1e-12:
            continue

        # execution cooldown (bars) at signal-level
        if int(entry_i) < int(next_allowed_entry_i):
            continue

        # direction constraints (single-direction only)
        direction = int(arr.direction[int(idx)])
        if open_direction is not None and int(direction) != int(open_direction):
            continue

        # DD governor: rolling_dd -> stop_open cooldown + min-risk recovery
        roll_peak.prune(entry_ts_ns)
        rp = float(roll_peak.max())
        if not np.isfinite(rp):
            rp = float(equity)
        dd_now = float(max(rp, float(equity)) - float(equity))
        dd_total = float(max(0.0, float(peak) - float(equity)))
        dd_year = float(max(0.0, float(year_peak) - float(equity)))
        dd_quarter = float(max(0.0, float(quarter_peak) - float(equity)))
        if int(entry_i) >= int(stop_open_until_entry_i):
            dd_recover = (
                float(dd_now) <= float(risk.dd_trigger_usd) * float(risk.dd_recover_ratio) + 1e-12
                and float(dd_year) <= float(risk.dd_trigger_usd_year) * float(risk.dd_recover_ratio) + 1e-12
                and float(dd_quarter) <= float(risk.dd_trigger_usd_quarter) * float(risk.dd_recover_ratio) + 1e-12
            )
            if bool(dd_recover):
                dd_stop_armed = True
        dd_exceeded = (
            float(dd_now) > float(risk.dd_trigger_usd) + 1e-12
            or float(dd_total) > float(risk.dd_trigger_usd) + 1e-12
            or float(dd_year) > float(risk.dd_trigger_usd_year) + 1e-12
            or float(dd_quarter) > float(risk.dd_trigger_usd_quarter) + 1e-12
        )
        if bool(dd_stop_armed) and bool(dd_exceeded):
            dd_trigger_count += 1
            if float(dd_now) > float(risk.dd_trigger_usd) + 1e-12:
                dd_trigger_count_roll += 1
            if float(dd_year) > float(risk.dd_trigger_usd_year) + 1e-12:
                dd_trigger_count_year += 1
            if float(dd_quarter) > float(risk.dd_trigger_usd_quarter) + 1e-12:
                dd_trigger_count_quarter += 1
            dd_stop_armed = False
            stop_open_until_entry_i = max(int(stop_open_until_entry_i), int(entry_i) + int(risk.dd_stop_cooldown_bars))
        if int(entry_i) < int(stop_open_until_entry_i):
            continue
        rs = float(risk.risk_scale_min) if bool(dd_exceeded) else 1.0

        sl_dist = float(arr.sl_dist[int(idx)])
        if not (np.isfinite(sl_dist) and sl_dist > 1e-12):
            continue

        is_pre = bool(int(ctx.pre_start_ns) <= entry_ts_ns <= int(ctx.pre_end_ns))
        is_os = bool(entry_ts_ns >= int(ctx.os_start_ns))

        year_key = None
        reg_key = None
        if bool(with_breakdowns) and bool(is_pre):
            if 0 <= int(entry_i) < int(getattr(ctx.year_by_bar, "size", 0)):
                try:
                    year_key = int(ctx.year_by_bar[int(entry_i)])
                except Exception:
                    year_key = None
            adx_v = float(arr.adx14[int(idx)]) if np.isfinite(float(arr.adx14[int(idx)])) else float("nan")
            if np.isfinite(adx_v):
                if adx_v < 20.0:
                    reg_key = 0
                elif adx_v < 30.0:
                    reg_key = 1
                else:
                    reg_key = 2

        opened_tickets = 0

        n_tickets_plan = int(max(1, tickets_per_signal))
        if int(n_tickets_plan) <= 1:
            if int(open_count) >= int(max_parallel_same_dir):
                continue
            if not lot_max_by_ticket:
                continue
            lm_total = float(lot_max_by_ticket[0][int(idx)]) if np.isfinite(float(lot_max_by_ticket[0][int(idx)])) else float("nan")
            if not np.isfinite(lm_total) or lm_total < float(mkt.min_lot) - 1e-12:
                continue

            lot_scaled = float(lm_total * float(rs))
            if np.isfinite(step) and step > 1e-12:
                lot_scaled = float(math.floor(lot_scaled / step + 1e-12) * step)
                lot_scaled = float(round(lot_scaled, 4))
            lot = float(max(float(mkt.min_lot), min(float(lm_total), float(lot_scaled))))
            if not (np.isfinite(lot) and lot > 0):
                continue
            pnl_scale = float(lot * contract * sl_dist)

            # legacy cashflows: TP1 partial + runner
            if bool(arr.tp1_hit[int(idx)]):
                tp1_i = int(arr.tp1_hit_i[int(idx)])
                if tp1_i >= 0:
                    seq += 1
                    cash_usd_tp1 = float(arr.tp1_cash_r[int(idx)]) * pnl_scale
                    heapq.heappush(cashflows, (tp1_i, int(seq), float(cash_usd_tp1), False))
            seq += 1
            ex_i = int(arr.exit_i[int(idx)])
            if ex_i < 0:
                continue
            cash_r_exit = float(arr.runner_cash_r[int(idx)]) if bool(arr.tp1_hit[int(idx)]) else float(arr.net_r[int(idx)])
            cash_usd_exit = float(cash_r_exit) * pnl_scale
            heapq.heappush(cashflows, (ex_i, int(seq), float(cash_usd_exit), True))

            if open_direction is None:
                open_direction = int(direction)
            open_count += 1
            opened_tickets = 1

            net_r = float(arr.net_r[int(idx)])
            if np.isfinite(net_r):
                all_tickets += 1
                all_sum += net_r
                if net_r > 0:
                    all_pos += net_r
                elif net_r < 0:
                    all_neg += net_r
                if is_pre:
                    pre_tickets += 1
                    pre_sum += net_r
                    if net_r > 0:
                        pre_pos += net_r
                    elif net_r < 0:
                        pre_neg += net_r
                if is_os:
                    os_tickets += 1
                    os_sum += net_r
                    if net_r > 0:
                        os_pos += net_r
                    elif net_r < 0:
                        os_neg += net_r
                if bool(with_breakdowns) and bool(is_pre) and year_key is not None:
                    _acc_ticket(_get_acc(year_acc, int(year_key)), float(net_r))
                if bool(with_breakdowns) and bool(is_pre) and reg_key is not None:
                    _acc_ticket(_get_acc(reg_acc, int(reg_key)), float(net_r))
        else:
            shares = ticket_risk_shares(tickets_per_signal=int(n_tickets_plan), tp1_close_frac=float(tp1_close_frac))
            for j, share in enumerate(shares):
                if int(open_count) >= int(max_parallel_same_dir):
                    break
                if int(j) >= int(len(lot_max_by_ticket)):
                    break
                if not (np.isfinite(float(share)) and float(share) > 1e-12):
                    continue

                lm_ticket = float(lot_max_by_ticket[int(j)][int(idx)]) if np.isfinite(float(lot_max_by_ticket[int(j)][int(idx)])) else float("nan")
                if np.isfinite(step) and step > 1e-12:
                    lm_ticket = float(math.floor(lm_ticket / step + 1e-12) * step)
                    lm_ticket = float(round(lm_ticket, 4))
                if not np.isfinite(lm_ticket) or lm_ticket < float(mkt.min_lot) - 1e-12:
                    continue

                lot_scaled = float(lm_ticket * float(rs))
                if np.isfinite(step) and step > 1e-12:
                    lot_scaled = float(math.floor(lot_scaled / step + 1e-12) * step)
                    lot_scaled = float(round(lot_scaled, 4))
                lot = float(max(float(mkt.min_lot), min(float(lm_ticket), float(lot_scaled))))
                if not (np.isfinite(lot) and lot > 0):
                    continue
                pnl_scale = float(lot * contract * sl_dist)

                kind = ticket_kind_for_index(int(n_tickets_plan), int(j))
                net_r_t, exit_i_t = ticket_net_r_and_exit_i(arr, idx=int(idx), kind=str(kind))
                if not (np.isfinite(net_r_t) and int(exit_i_t) >= 0):
                    continue

                seq += 1
                cash_usd_exit = float(net_r_t) * pnl_scale
                heapq.heappush(cashflows, (int(exit_i_t), int(seq), float(cash_usd_exit), True))

                if open_direction is None:
                    open_direction = int(direction)
                open_count += 1
                opened_tickets += 1

                all_tickets += 1
                all_sum += float(net_r_t)
                if net_r_t > 0:
                    all_pos += float(net_r_t)
                elif net_r_t < 0:
                    all_neg += float(net_r_t)
                if is_pre:
                    pre_tickets += 1
                    pre_sum += float(net_r_t)
                    if net_r_t > 0:
                        pre_pos += float(net_r_t)
                    elif net_r_t < 0:
                        pre_neg += float(net_r_t)
                if is_os:
                    os_tickets += 1
                    os_sum += float(net_r_t)
                    if net_r_t > 0:
                        os_pos += float(net_r_t)
                    elif net_r_t < 0:
                        os_neg += float(net_r_t)
                if bool(with_breakdowns) and bool(is_pre) and year_key is not None:
                    _acc_ticket(_get_acc(year_acc, int(year_key)), float(net_r_t))
                if bool(with_breakdowns) and bool(is_pre) and reg_key is not None:
                    _acc_ticket(_get_acc(reg_acc, int(reg_key)), float(net_r_t))

        if int(opened_tickets) <= 0:
            continue

        tp1_h = int(bool(arr.tp1_hit[int(idx)]))
        tp2_h = int(bool(arr.tp2_hit[int(idx)]))

        all_signals += 1
        all_tp1_signals += tp1_h
        all_tp2_signals += tp2_h
        if is_pre:
            pre_signals += 1
            pre_tp1_signals += tp1_h
            pre_tp2_signals += tp2_h
        if is_os:
            os_signals += 1
            os_tp1_signals += tp1_h
            os_tp2_signals += tp2_h

        if bool(with_breakdowns) and bool(is_pre) and year_key is not None:
            _acc_signal(_get_acc(year_acc, int(year_key)), int(tp1_h), int(tp2_h))
        if bool(with_breakdowns) and bool(is_pre) and reg_key is not None:
            _acc_signal(_get_acc(reg_acc, int(reg_key)), int(tp1_h), int(tp2_h))

        next_allowed_entry_i = int(entry_i) + int(max(0, int(cooldown_bars)))

    # flush remaining cashflows to end of data window
    if not stop_trading:
        _apply_cashflows_upto(int(ctx.idx_ts_ns.size - 1))

    pre_m = _metrics_from_acc(
        signals_n=int(pre_signals),
        tickets_n=int(pre_tickets),
        tp1_signals=int(pre_tp1_signals),
        tp2_signals=int(pre_tp2_signals),
        pos_sum=float(pre_pos),
        neg_sum=float(pre_neg),
        net_sum=float(pre_sum),
        days=float(ctx.days_pre),
    )
    os_m = _metrics_from_acc(
        signals_n=int(os_signals),
        tickets_n=int(os_tickets),
        tp1_signals=int(os_tp1_signals),
        tp2_signals=int(os_tp2_signals),
        pos_sum=float(os_pos),
        neg_sum=float(os_neg),
        net_sum=float(os_sum),
        days=float(ctx.days_os),
    )
    all_m = _metrics_from_acc(
        signals_n=int(all_signals),
        tickets_n=int(all_tickets),
        tp1_signals=int(all_tp1_signals),
        tp2_signals=int(all_tp2_signals),
        pos_sum=float(all_pos),
        neg_sum=float(all_neg),
        net_sum=float(all_sum),
        days=float(ctx.days_all),
    )

    stop_out_ts = str(pd.Timestamp(int(stop_out_ts_ns), tz="UTC")) if stop_out_ts_ns is not None else None
    stop_open_until_ts = None
    if int(stop_open_until_entry_i) > 0 and int(stop_open_until_entry_i) < int(ctx.idx_ts_ns.size):
        stop_open_until_ts = str(pd.Timestamp(int(ctx.idx_ts_ns[int(stop_open_until_entry_i)]), tz="UTC"))
    stop_open_active_end = bool(int(stop_open_until_entry_i) > int(ctx.idx_ts_ns.size - 1))

    meta = {
        "ok": True,
        "run_status": str(run_status),
        "stop_out_ts": stop_out_ts,
        "stop_out_bar_i": int(stop_out_bar_i) if stop_out_bar_i is not None else None,
        "dd_trigger_count": int(dd_trigger_count),
        "dd_trigger_count_roll": int(dd_trigger_count_roll),
        "dd_trigger_count_year": int(dd_trigger_count_year),
        "dd_trigger_count_quarter": int(dd_trigger_count_quarter),
        "dd_stop_skip": 0,
        "risk_scale_min": float(risk.risk_scale_min),
        "dd_rolling_window_days": int(risk.dd_rolling_window_days),
        "dd_stop_cooldown_bars": int(risk.dd_stop_cooldown_bars),
        "dd_trigger_usd": float(risk.dd_trigger_usd),
        "dd_trigger_usd_year": float(risk.dd_trigger_usd_year),
        "dd_trigger_usd_quarter": float(risk.dd_trigger_usd_quarter),
        "stop_open_until_ts": stop_open_until_ts,
        "stop_open_active_end": bool(stop_open_active_end),
        "equity_end": float(equity),
        "max_dd_usd": float(max_dd_usd),
        "max_dd_usd_year": float(year_max_dd),
        "max_dd_usd_quarter": float(quarter_max_dd),
        "max_dd_pct": float(max_dd_usd / float(mkt.initial_capital_usd) * 100.0) if mkt.initial_capital_usd > 0 else float("nan"),
        "max_dd_usd_preos": float(pre_max_dd_usd),
        "max_dd_pct_preos": float(pre_max_dd_usd / float(mkt.initial_capital_usd) * 100.0) if mkt.initial_capital_usd > 0 else float("nan"),
        "os_start_equity": float(os_start_equity) if np.isfinite(os_start_equity) else float("nan"),
        "max_dd_usd_os": float(os_max_dd_usd) if np.isfinite(os_max_dd_usd) else float("nan"),
        "max_dd_pct_os": float(os_max_dd_usd / float(os_start_equity) * 100.0)
        if (np.isfinite(os_max_dd_usd) and np.isfinite(os_start_equity) and os_start_equity > 0)
        else float("nan"),
    }

    if bool(with_breakdowns):
        year_table: List[Dict[str, Any]] = []
        if pre0_ts is not None and pre1_ts is not None:
            for yy in sorted(year_acc.keys()):
                a = year_acc.get(int(yy)) or {}
                ys = pd.Timestamp(f"{int(yy)}-01-01", tz="UTC")
                ye = pd.Timestamp(f"{int(yy)}-12-31 23:59:59", tz="UTC")
                start = max(pre0_ts, ys)
                end = min(pre1_ts, ye)
                days_y = max(1.0, float((end - start).total_seconds() / 86400.0))
                m = _metrics_from_acc(
                    signals_n=int(a.get("signals", 0)),
                    tickets_n=int(a.get("tickets", 0)),
                    tp1_signals=int(a.get("tp1_signals", 0)),
                    tp2_signals=int(a.get("tp2_signals", 0)),
                    pos_sum=float(a.get("pos_sum", 0.0)),
                    neg_sum=float(a.get("neg_sum", 0.0)),
                    net_sum=float(a.get("net_sum", 0.0)),
                    days=float(days_y),
                )
                year_table.append({"year": int(yy), "signals": int(a.get("signals", 0)), "tickets": int(a.get("tickets", 0)), **m})

        reg_table: List[Dict[str, Any]] = []
        reg_labels = {0: "ADX<20", 1: "20<=ADX<30", 2: "ADX>=30"}
        for rk in sorted(reg_acc.keys()):
            a = reg_acc.get(int(rk)) or {}
            m = _metrics_from_acc(
                signals_n=int(a.get("signals", 0)),
                tickets_n=int(a.get("tickets", 0)),
                tp1_signals=int(a.get("tp1_signals", 0)),
                tp2_signals=int(a.get("tp2_signals", 0)),
                pos_sum=float(a.get("pos_sum", 0.0)),
                neg_sum=float(a.get("neg_sum", 0.0)),
                net_sum=float(a.get("net_sum", 0.0)),
                days=float(ctx.days_pre),
            )
            reg_table.append({"adx_bin": int(rk), "adx_label": str(reg_labels.get(int(rk), "NA")), "signals": int(a.get("signals", 0)), "tickets": int(a.get("tickets", 0)), **m})

        evs = [float(r.get("ev_r")) for r in year_table if np.isfinite(float(r.get("ev_r", float("nan")))) and int(r.get("tickets", 0)) > 0]
        ev_p25 = float(np.quantile(np.asarray(evs, dtype=float), 0.25)) if evs else float("nan")
        meta["preos_year_table"] = year_table
        meta["preos_regime_adx3_table"] = reg_table
        meta["ev_r_preos_p25"] = float(ev_p25)

    return pre_m, os_m, all_m, meta


# =============================
# Metrics
# =============================


def metrics_from_trades(
    time_cfg: TimeConfig,
    mkt: MarketConfig,
    *,
    trades: pd.DataFrame,
    start_utc: Optional[str] = None,
    end_utc: Optional[str] = None,
) -> Dict[str, float]:
    if trades.empty:
        return {
            "epd": 0.0,
            "tpd": 0.0,
            "hit_tp1": float("nan"),
            "hit_tp2": float("nan"),
            "pf": float("nan"),
            "ev_r": float("nan"),
            "maxDD_usd": float("nan"),
            "maxDD_pct": float("nan"),
        }
    t = trades.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True, errors="coerce")
    t = t[pd.notna(t["entry_time"])]
    s = to_utc_ts(start_utc) if start_utc else to_utc_ts(time_cfg.backtest_start_utc)
    e = to_utc_ts(end_utc) if end_utc else to_utc_ts(time_cfg.backtest_end_utc)
    t = t[(t["entry_time"] >= s) & (t["entry_time"] <= e)]
    days = float((e - s).total_seconds() / 86400.0)
    days = max(1.0, days)
    tickets_n = int(len(t))
    if "signal_i" in t.columns:
        signals_n = int(pd.to_numeric(t["signal_i"], errors="coerce").dropna().astype(int).nunique())
    else:
        signals_n = int(tickets_n)
    epd = float(float(signals_n) / days)
    tpd = float(float(tickets_n) / days)

    if "signal_i" in t.columns:
        grp = t.groupby("signal_i", dropna=True)
        tp1_by = grp["tp1_hit"].max() if "tp1_hit" in t.columns else grp["tp1_reached"].max()
        tp2_by = grp["tp2_hit"].max() if "tp2_hit" in t.columns else grp["tp2_reached"].max()
        hit_tp1 = float(np.mean(pd.to_numeric(tp1_by, errors="coerce").fillna(0).astype(int)))
        hit_tp2 = float(np.mean(pd.to_numeric(tp2_by, errors="coerce").fillna(0).astype(int)))
    else:
        col_tp1 = "tp1_hit" if "tp1_hit" in t.columns else "tp1_reached"
        col_tp2 = "tp2_hit" if "tp2_hit" in t.columns else "tp2_reached"
        hit_tp1 = float(np.mean(pd.to_numeric(t[col_tp1], errors="coerce").fillna(0).astype(int)))
        hit_tp2 = float(np.mean(pd.to_numeric(t[col_tp2], errors="coerce").fillna(0).astype(int)))
    net_r = pd.to_numeric(t["net_r"], errors="coerce").to_numpy(dtype=float)
    pos = net_r[np.isfinite(net_r) & (net_r > 0)]
    neg = net_r[np.isfinite(net_r) & (net_r < 0)]
    pf = float(np.sum(pos) / max(1e-12, abs(np.sum(neg)))) if pos.size and neg.size else float("nan")
    ev_r = float(np.nanmean(net_r)) if net_r.size else float("nan")

    # maxDD is provided by simulator meta; keep placeholder here
    return {"epd": epd, "tpd": tpd, "hit_tp1": hit_tp1, "hit_tp2": hit_tp2, "pf": pf, "ev_r": ev_r}


def slice_segment(trades: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    t = trades.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True, errors="coerce")
    s = to_utc_ts(start)
    e = to_utc_ts(end)
    return t[(t["entry_time"] >= s) & (t["entry_time"] <= e)].copy()


# =============================
# Lot math audit + risk grid auto-derive
# =============================


def lot_math_audit(
    time_cfg: TimeConfig,
    mkt: MarketConfig,
    esc: ExitSearchConfig,
    *,
    events: pd.DataFrame,
    sl_atr_mult_for_audit: float,
    min_lot_values: Sequence[float],
) -> Dict[str, Any]:
    pre0 = to_utc_ts(time_cfg.preos_start_utc)
    pre1 = to_utc_ts(time_cfg.preos_end_utc)
    ev = events.copy()
    ev["_signal_ts"] = pd.to_datetime(ev["_signal_ts"], utc=True, errors="coerce")
    ev = ev[pd.notna(ev["_signal_ts"])]
    ev = ev[(ev["_signal_ts"] >= pre0) & (ev["_signal_ts"] <= pre1)]
    atr = pd.to_numeric(ev["atr_ref"], errors="coerce").to_numpy(dtype=float)
    atr = atr[np.isfinite(atr) & (atr > 1e-12)]
    sl_dist = atr * float(sl_atr_mult_for_audit)
    sl_dist = sl_dist[np.isfinite(sl_dist) & (sl_dist > 1e-12)]

    out: Dict[str, Any] = {
        "sl_atr_mult": float(sl_atr_mult_for_audit),
        "preos_events": int(len(ev)),
        "atr_ref_p50": float(np.quantile(atr, 0.50)) if atr.size else float("nan"),
        "atr_ref_p80": float(np.quantile(atr, 0.80)) if atr.size else float("nan"),
        "atr_ref_p95": float(np.quantile(atr, 0.95)) if atr.size else float("nan"),
        "sl_dist_p50": float(np.quantile(sl_dist, 0.50)) if sl_dist.size else float("nan"),
        "sl_dist_p80": float(np.quantile(sl_dist, 0.80)) if sl_dist.size else float("nan"),
        "sl_dist_p95": float(np.quantile(sl_dist, 0.95)) if sl_dist.size else float("nan"),
        "min_lot_branches": {},
    }
    for ml in min_lot_values:
        risk_usd = (sl_dist + float(mkt.roundtrip_cost_price) + float(mkt.slippage_buffer_price)) * float(mkt.contract_size) * float(ml)
        out["min_lot_branches"][str(ml)] = {
            "risk_usd_p50": float(np.quantile(risk_usd, 0.50)) if risk_usd.size else float("nan"),
            "risk_usd_p80": float(np.quantile(risk_usd, 0.80)) if risk_usd.size else float("nan"),
            "risk_usd_p95": float(np.quantile(risk_usd, 0.95)) if risk_usd.size else float("nan"),
        }
    return out


def derive_risk_grid_from_audit(audit: Dict[str, Any], *, min_lot: float) -> Tuple[float, ...]:
    branch = audit.get("min_lot_branches", {}).get(str(min_lot), {})
    p80 = float(branch.get("risk_usd_p80", float("nan")))
    if not np.isfinite(p80) or p80 <= 0:
        return (4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
    # ensure p80 is covered; add headroom for dd governor scaling (<1).
    # Note: risk_cap is a ceiling; in this script we trade at min_lot, so raising risk_cap does not increase risk,
    # it only avoids structural "min_lot over risk_cap" skips (take_rate collapse).
    base = float(math.ceil(p80 * 2.0) / 2.0)  # 0.5 USD rounding
    min_scale = 0.20  # dd_recovery_risk_scale (worst case) target coverage
    need = float(math.ceil((p80 / max(1e-6, float(min_scale))) * 2.0) / 2.0)
    grid = sorted({base, base * 1.25, base * 1.5, base * 2.0, base * 3.0, base * 5.0, need})
    grid = [float(round(x, 2)) for x in grid if 0.5 <= x <= 60.0]
    return tuple(grid[:10]) if len(grid) > 10 else tuple(grid)


# =============================
# Candidate search
# =============================


def evaluate_candidate(
    time_cfg: TimeConfig,
    mkt: MarketConfig,
    risk: RiskConfig,
    cv_cfg: CVConfig,
    mdl_cfg: ModelConfig,
    thr_cfg: ThresholdConfig,
    *,
    df_prices: pd.DataFrame,
    base_events_feat: pd.DataFrame,
    regimes: Dict[str, np.ndarray],
    exit_cfg: ExitConfig,
    strat_params: Tuple[float, float, int, int, float, int],
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    strat_params:
      (risk_cap_usd, daily_stop_loss_usd, dd_cooldown_days, max_parallel_tickets, q, seed)
    Returns: (row for candidates.csv, trades_final (optional), meta_final (optional))
    """
    (risk_cap_usd, daily_stop_loss_usd, dd_cooldown_days, max_parallel_tickets, q, seed) = strat_params
    # compute outcomes
    ds, out_meta = compute_event_outcomes(mkt, ExitSearchConfig(), df=df_prices, ev=base_events_feat, ex=exit_cfg)
    if ds.empty:
        row = {
            "fail_reason": "fail_no_events_after_econ_prune",
            "risk_cap_usd": risk_cap_usd,
            "daily_stop_loss_usd": daily_stop_loss_usd,
            "dd_cooldown_days": dd_cooldown_days,
            "max_parallel_tickets": max_parallel_tickets,
            "q": q,
            **dataclasses.asdict(exit_cfg.tpslh),
            "entry": exit_cfg.entry,
            "tp1_close_frac": exit_cfg.tp1_close_frac,
            "tp2_mult": exit_cfg.tp2_mult,
            "econ_pruned": out_meta.get("econ_pruned", 0),
        }
        return row, None, None

    # attach regimes at signal time
    sig_i = ds["signal_i"].astype(int).to_numpy()
    ds["vol_regime"] = np.array([int(regimes["vol_regime"][i]) for i in sig_i], dtype=int)
    ds["trend_regime"] = np.array([int(regimes["trend_regime"][i]) for i in sig_i], dtype=int)

    # score models for both sides (fast path): LGBM uncalibrated + purged OOF (pre-OS) + preOS-trained scores (OS)
    pre_end = to_utc_ts(time_cfg.preos_end_utc)
    filter_report: Dict[str, Any] = {"exit_cfg": dataclasses.asdict(exit_cfg), "sides": {}}
    scored_parts: List[pd.DataFrame] = []
    for side in ("long", "short"):
        side_ds = ds[ds["side"] == side].copy()
        if side_ds.empty:
            continue
        scored, meta_s = score_side_preos_os(time_cfg, cv_cfg, mdl_cfg, df_prices=df_prices, ds_all_side=side_ds)
        pre_mask = scored["_entry_ts"] <= pre_end
        y = (pd.to_numeric(scored.loc[pre_mask, "net_r"], errors="coerce").to_numpy(dtype=float) > 0.0).astype(int)
        p = pd.to_numeric(scored.loc[pre_mask, "p_score"], errors="coerce").to_numpy(dtype=float)
        filter_report["sides"][side] = {
            "model_kind": "lgbm_uncalibrated",
            "calib_method": "na",
            "brier_preos": float(brier_score(y, p)),
            "auc_preos": float(roc_auc(y, p)),
            "meta": meta_s,
        }
        scored_parts.append(scored)

    if not scored_parts:
        row = {
            "fail_reason": "fail_no_side_scored",
            "risk_cap_usd": risk_cap_usd,
            "daily_stop_loss_usd": daily_stop_loss_usd,
            "dd_cooldown_days": dd_cooldown_days,
            "max_parallel_tickets": max_parallel_tickets,
            "q": q,
            **dataclasses.asdict(exit_cfg.tpslh),
            "entry": exit_cfg.entry,
            "tp1_close_frac": exit_cfg.tp1_close_frac,
            "tp2_mult": exit_cfg.tp2_mult,
            "econ_pruned": out_meta.get("econ_pruned", 0),
        }
        return row, None, None

    scored_all = pd.concat(scored_parts, axis=0, ignore_index=True)
    # ensure all required columns exist (fill missing for other side)
    scored_all = scored_all.sort_values("_entry_ts", kind="mergesort").reset_index(drop=True)

    # simulate
    strat = StrategyConfig(
        exit=exit_cfg,
        filt=FilterConfig(q=float(q), lookback_days=int(thr_cfg.score_lookback_days), min_hist=int(thr_cfg.min_score_history)),
        risk_cap_usd=float(risk_cap_usd),
        daily_stop_loss_usd=float(daily_stop_loss_usd),
        dd_cooldown_days=int(dd_cooldown_days),
        max_parallel_tickets=int(max_parallel_tickets),
    )
    trades, meta = simulate_trading(time_cfg, mkt, risk, df_prices=df_prices, scored_events=scored_all, strat=strat, store_thresholds=False)

    # segment metrics
    pre_trades = slice_segment(trades, start=time_cfg.preos_start_utc, end=time_cfg.preos_end_utc)
    os_trades = slice_segment(trades, start=time_cfg.os_start_utc, end=time_cfg.backtest_end_utc)
    all_trades = slice_segment(trades, start=time_cfg.backtest_start_utc, end=time_cfg.backtest_end_utc)

    pre_m = metrics_from_trades(time_cfg, mkt, trades=pre_trades, start_utc=time_cfg.preos_start_utc, end_utc=time_cfg.preos_end_utc)
    os_m = metrics_from_trades(time_cfg, mkt, trades=os_trades, start_utc=time_cfg.os_start_utc, end_utc=time_cfg.backtest_end_utc)
    all_m = metrics_from_trades(time_cfg, mkt, trades=all_trades, start_utc=time_cfg.backtest_start_utc, end_utc=time_cfg.backtest_end_utc)

    # DD: use strict pre-OS drawdown for selection; OS/All only for reporting.
    pre_max_dd_usd = float(meta.get("max_dd_usd_preos", float("nan")))
    pre_max_dd_pct = float(meta.get("max_dd_pct_preos", float("nan")))
    os_max_dd_usd = float(meta.get("max_dd_usd_os", float("nan")))
    os_max_dd_pct = float(meta.get("max_dd_pct_os", float("nan")))
    all_max_dd_usd = float(meta.get("max_dd_usd", float("nan")))
    all_max_dd_pct = float(meta.get("max_dd_pct", float("nan")))

    # hard constraints (pre-OS only; OS only for epd>0 feasibility)
    fail_reason = None
    if not (np.isfinite(os_m["epd"]) and os_m["epd"] > 0):
        fail_reason = "fail_os_no_trades"
    elif not (np.isfinite(pre_m["epd"]) and pre_m["epd"] >= 0.8):
        fail_reason = "fail_epd"
    elif not (np.isfinite(pre_m["hit_tp1"]) and pre_m["hit_tp1"] >= 0.70):
        fail_reason = "fail_hit"
    elif not (np.isfinite(pre_m["ev_r"]) and pre_m["ev_r"] >= 0.0):
        fail_reason = "fail_ev"
    elif not (np.isfinite(pre_max_dd_usd) and pre_max_dd_usd <= 45.0):
        fail_reason = "fail_dd"

    row = {
        **dataclasses.asdict(exit_cfg.tpslh),
        "entry": exit_cfg.entry,
        "tp1_close_frac": exit_cfg.tp1_close_frac,
        "tp2_mult": exit_cfg.tp2_mult,
        "risk_cap_usd": float(risk_cap_usd),
        "daily_stop_loss_usd": float(daily_stop_loss_usd),
        "dd_cooldown_days": int(dd_cooldown_days),
        "max_parallel_tickets": int(max_parallel_tickets),
        "q": float(q),
        "econ_pruned": int(out_meta.get("econ_pruned", 0)),
        "pre_epd": float(pre_m["epd"]),
        "pre_tpd": float(pre_m["tpd"]),
        "pre_hit_tp1": float(pre_m["hit_tp1"]),
        "pre_hit_tp2": float(pre_m["hit_tp2"]),
        "pre_pf": float(pre_m["pf"]),
        "pre_ev_r": float(pre_m["ev_r"]),
        "os_epd": float(os_m["epd"]),
        "all_epd": float(all_m["epd"]),
        "all_pf": float(all_m["pf"]),
        # keep legacy keys as pre-OS for selection
        "maxDD_usd": float(pre_max_dd_usd),
        "maxDD_pct": float(pre_max_dd_pct),
        "os_maxDD_usd": float(os_max_dd_usd),
        "os_maxDD_pct": float(os_max_dd_pct),
        "all_maxDD_usd": float(all_max_dd_usd),
        "all_maxDD_pct": float(all_max_dd_pct),
        "dd_stop_triggers": int(meta.get("dd_stop_triggers", 0)),
        "equity_floor_hits": int(meta.get("equity_floor_hits", 0)),
        "fail_reason": str(fail_reason) if fail_reason else "",
        "filter_brier_long": float(filter_report["sides"].get("long", {}).get("brier_preos", float("nan"))),
        "filter_auc_long": float(filter_report["sides"].get("long", {}).get("auc_preos", float("nan"))),
        "filter_brier_short": float(filter_report["sides"].get("short", {}).get("brier_preos", float("nan"))),
        "filter_auc_short": float(filter_report["sides"].get("short", {}).get("auc_preos", float("nan"))),
        "filter_model_long": str(filter_report["sides"].get("long", {}).get("model_kind", "NA")),
        "filter_calib_long": str(filter_report["sides"].get("long", {}).get("calib_method", "NA")),
        "filter_model_short": str(filter_report["sides"].get("short", {}).get("model_kind", "NA")),
        "filter_calib_short": str(filter_report["sides"].get("short", {}).get("calib_method", "NA")),
    }
    return row, trades, {"meta": meta, "filter_report": filter_report}


# =============================
# Main
# =============================


def main() -> int:
    # Paths
    home = Path.home()
    out_dir = home / "Desktop" / "20260112"
    artifacts_dir = out_dir / "012_artifacts"
    report_path = out_dir / "012.txt"
    desktop_script_copy = artifacts_dir / "20260112_012_Mode4_Optimize.py"
    paths = Paths(out_dir=out_dir, artifacts_dir=artifacts_dir, report_path=report_path, desktop_script_copy=desktop_script_copy)
    ensure_dir(paths.out_dir)
    ensure_dir(paths.artifacts_dir)

    # configs
    time_cfg = TimeConfig()
    mkt = MarketConfig()
    sig_cfg = Mode4SignalConfig()
    cv_cfg = CVConfig()
    thr_cfg = ThresholdConfig()
    esc = ExitSearchConfig()
    mdl_cfg = ModelConfig()

    # Data (auto-locate; Win/WSL compatible)
    data_path: Optional[Path] = None
    for data_root in (base / "42swam", base / "trend_project", base):
        try:
            if data_root.exists():
                data_path = locate_xauusd_m5(data_root)
                break
        except Exception:
            continue
    if data_path is None:
        data_path = locate_xauusd_m5(base)
    df0 = pd.read_csv(data_path)
    if "datetime" not in df0.columns:
        raise ValueError("数据缺少 datetime 列")
    df0["datetime"] = pd.to_datetime(df0["datetime"], utc=True, errors="coerce")
    df0 = df0.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates(subset=["datetime"])
    df0 = df0.set_index("datetime")
    df0 = df0.rename(columns={c: c.lower() for c in df0.columns})
    need_cols = {"open", "high", "low", "close"}
    if not need_cols.issubset(set(df0.columns)):
        raise ValueError(f"数据缺少列: {sorted(list(need_cols - set(df0.columns)))}")
    df0 = df0.loc[(df0.index >= to_utc_ts(time_cfg.start_utc)) & (df0.index <= to_utc_ts(time_cfg.end_utc))].copy()

    # Precompute indicators/features
    ind = precompute_indicators(df0)
    ctx = compute_feature_context(df0, ind)
    regimes = build_regimes_2x2(df0, ctx, window_bars=252)

    # Events (mode4 fixed)
    ev_raw = generate_mode4_events(sig_cfg, df=df0, ind=ind)
    if ev_raw.empty:
        raise RuntimeError("mode4 raw events 为空（请检查数据或信号逻辑）")
    ev_feat = attach_event_features(ev_raw, df=df0, ctx=ctx)
    path_feat = compute_path_features(df0, ctx=ctx, ev=ev_feat)
    for c in PATH_FEATURE_COLS:
        if c in path_feat.columns:
            ev_feat[c] = path_feat[c].to_numpy(dtype=float)
    # attach regimes at signal for audits
    sig_i = ev_feat["signal_i"].astype(int).to_numpy()
    ev_feat["vol_regime"] = np.array([int(regimes["vol_regime"][i]) for i in sig_i], dtype=int)
    ev_feat["trend_regime"] = np.array([int(regimes["trend_regime"][i]) for i in sig_i], dtype=int)

    # =============================
    # Step A: feasibility pre-scan
    # =============================
    # 009 baseline used sl_atr_mult=1.00; keep an explicit branch for feasibility comparison.
    lot_audit_sl1 = lot_math_audit(
        time_cfg,
        mkt,
        esc,
        events=ev_feat,
        sl_atr_mult_for_audit=1.00,
        min_lot_values=(0.01, 0.02),
    )
    # For auto risk-cap grid, use the worst-case SL (max grid) to avoid structural skips across exit search.
    lot_audit_slmax = lot_math_audit(
        time_cfg,
        mkt,
        esc,
        events=ev_feat,
        sl_atr_mult_for_audit=float(max(esc.sl_atr_mult_grid)),
        min_lot_values=(0.01, 0.02),
    )
    risk_grid = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
    risk_cfg = RiskConfig(max_risk_usd_per_trade_grid=risk_grid)

    write_json(
        paths.artifacts_dir / "lot_math_audit.json",
        {
            "version": "20260112_012",
            "audits": {
                "sl_atr_mult_1.00": lot_audit_sl1,
                f"sl_atr_mult_{float(max(esc.sl_atr_mult_grid)):.2f}": lot_audit_slmax,
            },
            "fixed_max_risk_usd_per_trade_grid": list(risk_grid),
            "scenarios": {
                "S1": {"min_lot": 0.01, "lot_step": 0.01},
                "S2": {"min_lot": 0.02, "lot_step": 0.01},
            },
        },
    )

    # Leakage audit (features + regimes inputs)
    leak = leakage_audit_by_truncation(seed=cv_cfg.seed, df_full=df0, feature_cols=list(FEATURE_COLS), sample_n=10)
    leak_ok = bool(leak.get("ok", False)) and int(leak.get("failures_n", 0)) == 0
    write_json(paths.artifacts_dir / "leakage_audit.json", {"ok": bool(leak_ok), "failures": int(leak.get("failures_n", 999)), "detail": leak})
    if not leak_ok:
        raise RuntimeError(f"leakage_audit 失败：failures_n={int(leak.get('failures_n', 999))}（必须为0）")

    # Manifest
    script_path = Path(__file__).resolve()
    manifest = {
        "generated_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "paths": {
            "data_path": str(data_path),
            "script_path": str(script_path),
            "repo_root": str(repo_root),
            "out_dir": str(paths.out_dir),
            "artifacts_dir": str(paths.artifacts_dir),
        },
        "data_file": {
            "sha256": sha256_file(data_path),
            "bytes": int(data_path.stat().st_size),
            "mtime_utc": pd.Timestamp(data_path.stat().st_mtime, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
            "rows": int(len(df0)),
            "start": str(df0.index.min()),
            "end": str(df0.index.max()),
        },
        "script_file": {"sha256": sha256_file(script_path)},
        "params": {
            "purge_bars": int(cv_cfg.purge_bars),
            "embargo_bars": int(cv_cfg.embargo_bars),
            "initial_capital_usd": float(mkt.initial_capital_usd),
            "roundtrip_cost_price": float(mkt.roundtrip_cost_price),
            "slippage_buffer_price": float(mkt.slippage_buffer_price),
            "max_risk_usd_per_trade_grid": list(risk_grid),
            "daily_stop_loss_usd_grid": list(risk_cfg.daily_stop_loss_usd_grid),
            "tp1_over_cost_k_grid": list(esc.tp1_over_cost_k_grid),
            "exit_grid": {
                "entry": list(esc.entry_grid),
                "H": list(esc.H_grid),
                "tp1_atr_mult": list(esc.tp1_atr_mult_grid),
                "sl_atr_mult": list(esc.sl_atr_mult_grid),
                "tp1_close_frac": list(esc.tp1_close_frac_grid),
                "tp2_mult": list(esc.tp2_mult_grid),
            },
            "scenarios": {"S1": {"min_lot": 0.01, "lot_step": 0.01}, "S2": {"min_lot": 0.02, "lot_step": 0.01}},
        },
        "env": {"python": sys.version.split()[0]},
    }
    write_json(paths.artifacts_dir / "manifest.json", manifest)

    # =============================
    # Step B/C/D: search (pre-OS selection + OS_epd feasibility)
    # =============================
    candidates_rows: List[Dict[str, Any]] = []

    scenarios: List[Tuple[str, MarketConfig]] = [
        ("S1", dataclasses.replace(mkt, min_lot=0.01, lot_step=0.01)),
        ("S2", dataclasses.replace(mkt, min_lot=0.02, lot_step=0.01)),
    ]
    econ_k_grid = tuple(float(k) for k in esc.tp1_over_cost_k_grid)
    econ_k_min = float(min(econ_k_grid)) if econ_k_grid else 1.2

    # successive halving:
    # - rung0(2015-2017): only evaluate Exit structure (no filter training), coarse ranking
    # - rung1(pre-OS): evaluate strategy grid under two execution scenarios
    exit_candidates: List[ExitConfig] = []
    for entry in esc.entry_grid:
        for H in esc.H_grid:
            for tp1m in esc.tp1_atr_mult_grid:
                for slm in esc.sl_atr_mult_grid:
                    for frac in esc.tp1_close_frac_grid:
                        for tp2m in esc.tp2_mult_grid:
                            exit_candidates.append(
                                ExitConfig(
                                    entry=str(entry),
                                    tpslh=TPSLH(H=int(H), tp1_atr_mult=float(tp1m), sl_atr_mult=float(slm)),
                                    tp1_close_frac=float(frac),
                                    tp2_mult=float(tp2m),
                                )
                            )

    # Bound trials: if too many, sample deterministically by seed
    max_exit_trials = 60
    if len(exit_candidates) > max_exit_trials:
        rng = np.random.default_rng(int(cv_cfg.seed))
        idx = rng.choice(np.arange(len(exit_candidates)), size=max_exit_trials, replace=False)
        exit_candidates = [exit_candidates[int(i)] for i in sorted(idx)]

    r0_start = to_utc_ts("2015-01-01")
    r0_end = to_utc_ts("2017-12-31 23:59:59")
    ev_r0 = ev_feat[(ev_feat["_signal_ts"] >= r0_start) & (ev_feat["_signal_ts"] <= r0_end)].copy()
    # runtime guard: sample rung0 events (only for coarse exit ranking)
    max_r0_events = 3000
    if int(len(ev_r0)) > int(max_r0_events):
        ev_r0 = ev_r0.sample(n=int(max_r0_events), random_state=int(cv_cfg.seed)).sort_values("_signal_ts", kind="mergesort").reset_index(drop=True)

    # Select a small but diverse exit set for the expensive grid stage.
    # Group by (tp1_atr_mult, H) to ensure the expanded search space is actually explored.
    best_by_group: Dict[Tuple[float, int], Tuple[Tuple[float, float, float, float], ExitConfig]] = {}
    for ex in exit_candidates:
        ds0, _meta0 = compute_event_outcomes(mkt, df=df0, ev=ev_r0, ex=ex)
        if ds0.empty:
            continue
        tp1_r0 = pd.to_numeric(ds0.get("tp1_r"), errors="coerce").to_numpy(dtype=float)
        cost_r0 = pd.to_numeric(ds0.get("cost_r"), errors="coerce").to_numpy(dtype=float)
        mask0 = np.isfinite(tp1_r0) & np.isfinite(cost_r0) & (tp1_r0 >= float(econ_k_min) * cost_r0 - 1e-12)
        ds0 = ds0.loc[mask0].copy()
        if ds0.empty:
            continue
        net_r0 = pd.to_numeric(ds0["net_r"], errors="coerce").to_numpy(dtype=float)
        hit0 = float(np.mean(ds0["tp1_hit"].astype(int))) if len(ds0) else 0.0
        evr0 = float(np.nanmean(net_r0)) if net_r0.size else -9.0
        pos = net_r0[np.isfinite(net_r0) & (net_r0 > 0)]
        neg = net_r0[np.isfinite(net_r0) & (net_r0 < 0)]
        pf0 = float(np.sum(pos) / max(1e-12, abs(np.sum(neg)))) if pos.size and neg.size else -9.0
        days0 = max(1.0, (r0_end - r0_start).total_seconds() / 86400.0)
        raw_epd0 = float(len(ds0) / days0)
        key = (float(ex.tpslh.tp1_atr_mult), int(ex.tpslh.H))
        score = (float(raw_epd0), float(hit0), float(evr0), float(pf0))
        cur = best_by_group.get(key)
        if cur is None or score > cur[0]:
            best_by_group[key] = (score, ex)

    top_exit: List[ExitConfig] = []
    for k in sorted(best_by_group.keys(), key=lambda t: (float(t[0]), int(t[1]))):
        top_exit.append(best_by_group[k][1])
    if not top_exit:
        top_exit = exit_candidates[: min(9, len(exit_candidates))]
    # Cap exit variants to control compute (still diversified by sorted grid order).
    max_top_exit = 4
    if int(len(top_exit)) > int(max_top_exit):
        sel = np.linspace(0, int(len(top_exit)) - 1, num=int(max_top_exit), dtype=int)
        top_exit = [top_exit[int(i)] for i in sel]

    # execution-level grids (TicketPolicy + parallel + cooldown)
    # NOTE: keep mp small to control compute; tickets/day is addressed via tickets_per_signal.
    mp_grid = tuple(int(x) for x in risk_cfg.max_parallel_same_dir_grid)
    tps_grid = tuple(int(x) for x in risk_cfg.tickets_per_signal_grid)
    cooldown_grid = tuple(int(x) for x in risk_cfg.cooldown_bars_grid)
    q_grid_eval = (0.30, 0.40, 0.50)
    q_tail_grid_eval = (0.70, 0.80, 0.90)
    dsl_grid_eval = (4.0, 6.0, 8.0, 10.0, 12.0)
    strat_grid: List[Tuple[float, float, int, float]] = []
    for risk_cap in risk_cfg.max_risk_usd_per_trade_grid:
        for dsl in risk_cfg.daily_stop_loss_usd_grid:
            for mp in mp_grid:
                for q in thr_cfg.q_grid:
                    strat_grid.append((float(risk_cap), float(dsl), int(mp), float(q)))

    pre0 = to_utc_ts(time_cfg.preos_start_utc)
    pre1 = to_utc_ts(time_cfg.preos_end_utc)
    os0 = to_utc_ts(time_cfg.os_start_utc)

    # fast simulator context (shared across grid search)
    bt0 = to_utc_ts(time_cfg.backtest_start_utc)
    bt1 = min(to_utc_ts(time_cfg.backtest_end_utc), pd.to_datetime(df0.index.max(), utc=True))
    ctx_fast = FastSimCtx(
        idx_ts_ns=_dt_index_to_ns_utc(df0.index),
        year_by_bar=df0.index.year.to_numpy(dtype=np.int16),
        bt_start_ns=int(bt0.value),
        bt_end_ns=int(bt1.value),
        pre_start_ns=int(pre0.value),
        pre_end_ns=int(pre1.value),
        os_start_ns=int(os0.value),
        days_pre=max(1.0, float((pre1 - pre0).total_seconds() / 86400.0)),
        days_os=max(1.0, float((bt1 - os0).total_seconds() / 86400.0)),
        days_all=max(1.0, float((bt1 - bt0).total_seconds() / 86400.0)),
    )

    # Sample strategy grid (risk/execution) to keep compute bounded while still covering the expanded space.
    strat_all: List[Tuple[str, MarketConfig, float, float, int, int, int]] = []
    for scen_name, mkt_s in scenarios:
        for risk_cap in risk_cfg.max_risk_usd_per_trade_grid:
            for dsl in dsl_grid_eval:
                for mp in mp_grid:
                    for tps in tps_grid:
                        if int(mp) < int(tps):
                            continue
                        for cd in cooldown_grid:
                            strat_all.append((str(scen_name), mkt_s, float(risk_cap), float(dsl), int(mp), int(tps), int(cd)))
    max_strat_trials = 60
    if len(strat_all) > int(max_strat_trials):
        rng = np.random.default_rng(int(cv_cfg.seed))
        sel = rng.choice(np.arange(len(strat_all)), size=int(max_strat_trials), replace=False)
        strat_all = [strat_all[int(i)] for i in sorted(sel)]

    # Keep only top-K candidates for output + breakdown reporting.
    max_candidates_keep = 300
    keep_heap: List[Tuple[Tuple[float, ...], str, Dict[str, Any]]] = []
    kept_breakdowns: Dict[str, Any] = {}

    fail_counts: Dict[str, int] = defaultdict(int)
    global_bounds = {
        "pre_tpd_max": 0.0,
        "pre_hit_tp1_max": -float("inf"),
        "pre_pf_max": -float("inf"),
        "pre_ev_r_max": -float("inf"),
        "pre_maxDD_usd_min": float("inf"),
    }

    def _f(x: Any, default: float) -> float:
        try:
            v = float(x)
            return v if np.isfinite(v) else float(default)
        except Exception:
            return float(default)

    def _cand_id(row: Dict[str, Any]) -> str:
        keys = (
            "scenario",
            "min_lot",
            "tp1_over_cost_k",
            "entry",
            "H",
            "tp1_atr_mult",
            "sl_atr_mult",
            "tp1_close_frac",
            "tp2_mult",
            "risk_cap_usd",
            "daily_stop_loss_usd",
            "max_parallel_same_dir",
            "tickets_per_signal",
            "cooldown_bars",
            "q",
            "q_tail",
        )
        payload = {k: row.get(k) for k in keys}
        s = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

    def _score_tuple(row: Dict[str, Any]) -> Tuple[float, ...]:
        pre_hit = _f(row.get("pre_hit_tp1"), -float("inf"))
        pre_pf = _f(row.get("pre_pf"), -float("inf"))
        pre_ev = _f(row.get("pre_ev_r"), -float("inf"))
        pre_dd = _f(row.get("maxDD_usd"), float("inf"))
        pre_tpd = _f(row.get("pre_tpd"), 0.0)
        os_epd = _f(row.get("os_epd"), 0.0)
        pre_stop = int(_f(row.get("pre_stop_out"), 0.0) > 0.5)

        os_ok = int(os_epd > 0.0)
        dd_ok = int(pre_dd <= 60.0)
        pf_ok = int(pre_pf >= 1.05)
        ev_ok = int(pre_ev >= 0.0)
        so_ok = int(pre_stop == 0)
        # NOTE: stability constraint (ev_r_preos_p25>=0) is applied at final selection after
        # computing breakdowns for kept candidates; do not bake it into the search-stage heap score.
        phase1 = int(os_ok and dd_ok and pf_ok and ev_ok and so_ok)
        phase2 = int(phase1 and (pre_tpd >= 1.5))
        strict = int(phase2 and (pre_hit >= 0.73))
        return (float(strict), float(phase2), float(phase1), float(pre_hit), float(pre_pf), float(pre_ev), -float(pre_dd), float(pre_tpd))

    for ex in top_exit:
        ds, _out_meta = compute_event_outcomes(mkt, df=df0, ev=ev_feat, ex=ex)
        if ds.empty:
            continue

        # score once per exit (per side) on full (unpruned) dataset
        scored_parts: List[pd.DataFrame] = []
        filter_metrics: Dict[str, Dict[str, float]] = {}
        for side in ("long", "short"):
            side_ds = ds[ds["side"] == side].copy()
            if side_ds.empty:
                continue
            scored_side, _meta_s = score_side_preos_os(time_cfg, cv_cfg, mdl_cfg, df_prices=df0, ds_all_side=side_ds)
            pre_mask = (scored_side["_entry_ts"] >= pre0) & (scored_side["_entry_ts"] <= pre1)
            y_win = (pd.to_numeric(scored_side.loc[pre_mask, "net_r"], errors="coerce").to_numpy(dtype=float) > 0.0).astype(int)
            p_win = pd.to_numeric(scored_side.loc[pre_mask, "p_score"], errors="coerce").to_numpy(dtype=float)
            filter_metrics[side] = {"brier": float(brier_score(y_win, p_win)), "auc": float(roc_auc(y_win, p_win))}
            scored_parts.append(scored_side)
        if not scored_parts:
            continue

        scored_all = pd.concat(scored_parts, axis=0, ignore_index=True)
        scored_all = scored_all.sort_values(["entry_i", "_entry_ts"], kind="mergesort").reset_index(drop=True)

        for econ_k in econ_k_grid:
            tp1_r_v = pd.to_numeric(scored_all.get("tp1_r"), errors="coerce").to_numpy(dtype=float)
            cost_r_v = pd.to_numeric(scored_all.get("cost_r"), errors="coerce").to_numpy(dtype=float)
            mask_k = np.isfinite(tp1_r_v) & np.isfinite(cost_r_v) & (tp1_r_v >= float(econ_k) * cost_r_v - 1e-12)
            scored_k = scored_all.loc[mask_k].copy()
            econ_pruned = int(len(scored_all) - len(scored_k))
            if scored_k.empty:
                fail_counts["fail_no_events_after_econ_prune"] += int(len(scenarios))
                continue

            scored_k = scored_k.sort_values(["entry_i", "_entry_ts"], kind="mergesort").reset_index(drop=True)
            # keep backtest range only (simulate_trading ignores events outside bt window; keep filter histories consistent)
            scored_k["_entry_ts"] = pd.to_datetime(scored_k["_entry_ts"], utc=True, errors="coerce")
            scored_k = scored_k[(scored_k["_entry_ts"] >= bt0) & (scored_k["_entry_ts"] <= bt1)].copy()
            if scored_k.empty:
                fail_counts["fail_no_events_in_backtest_window"] += int(len(scenarios))
                continue

            # fast arrays + precomputed ticket-wise lot_max per (risk_cap, tickets_per_signal)
            arr = build_scored_event_arrays(scored_k, mkt=mkt)
            lotmax_map: Dict[Tuple[float, int], List[np.ndarray]] = {}
            for risk_cap in risk_cfg.max_risk_usd_per_trade_grid:
                for tps in tps_grid:
                    shares = ticket_risk_shares(tickets_per_signal=int(tps), tp1_close_frac=float(ex.tp1_close_frac))
                    lotmax_map[(float(risk_cap), int(tps))] = [
                        lot_max_for_risk_cap(mkt, sl_dist_risk=arr.sl_dist_risk, risk_cap_usd=float(risk_cap) * float(share)) for share in shares
                    ]

            pre_mask_arr = (arr.entry_ts_ns >= int(ctx_fast.pre_start_ns)) & (arr.entry_ts_ns <= int(ctx_fast.pre_end_ns))
            for q in q_grid_eval:
                for q_tail in q_tail_grid_eval:
                    filt = FilterConfig(
                        q=float(q),
                        q_tail=float(q_tail),
                        lookback_days=int(thr_cfg.score_lookback_days),
                        min_hist=int(thr_cfg.min_score_history),
                    )
                    masks = compute_filter_masks(arr, filt=filt)
                    pass_mask = masks["pass"]
                    pass_idx = np.where(pass_mask)[0].astype(int)
                    take_rate_tail_pre = float(np.mean(masks["tail_ok"][pre_mask_arr])) if int(np.sum(pre_mask_arr)) > 0 else 0.0

                    for scen_name, mkt_s, risk_cap, dsl, mp, tps, cd in strat_all:
                        lot_max_by_ticket = lotmax_map[(float(risk_cap), int(tps))]
                        pre_m, os_m, all_m, meta = simulate_trading_fast_metrics(
                            ctx_fast,
                            mkt_s,
                            risk_cfg,
                            arr=arr,
                            pass_indices=pass_idx,
                            lot_max_by_ticket=lot_max_by_ticket,
                            daily_stop_loss_usd=float(dsl),
                            max_parallel_same_dir=int(mp),
                            tickets_per_signal=int(tps),
                            tp1_close_frac=float(ex.tp1_close_frac),
                            cooldown_bars=int(cd),
                        )

                        pre_max_dd_usd = float(meta.get("max_dd_usd_preos", float("nan")))
                        pre_max_dd_pct = float(meta.get("max_dd_pct_preos", float("nan")))
                        os_max_dd_usd = float(meta.get("max_dd_usd_os", float("nan")))
                        os_max_dd_pct = float(meta.get("max_dd_pct_os", float("nan")))
                        all_max_dd_usd = float(meta.get("max_dd_usd", float("nan")))
                        all_max_dd_pct = float(meta.get("max_dd_pct", float("nan")))

                        stop_out_ts = meta.get("stop_out_ts")
                        pre_stop_out = False
                        if str(meta.get("run_status")) == "STOP_OUT" and stop_out_ts:
                            ts_so = pd.to_datetime(stop_out_ts, utc=True, errors="coerce")
                            if pd.notna(ts_so) and ts_so <= pre1:
                                pre_stop_out = True

                        fail_reason = ""
                        if not (np.isfinite(os_m["epd"]) and os_m["epd"] > 0):
                            fail_reason = "fail_os_no_trades"
                        elif pre_stop_out:
                            fail_reason = "fail_dd"
                        elif float(take_rate_tail_pre) < float(thr_cfg.gate1_take_rate_min) - 1e-12:
                            fail_reason = "fail_gate1_take_rate"
                        elif not (np.isfinite(pre_m["tpd"]) and pre_m["tpd"] >= 1.5):
                            fail_reason = "fail_tpd"
                        elif not (np.isfinite(pre_m["hit_tp1"]) and pre_m["hit_tp1"] >= 0.73):
                            fail_reason = "fail_hit"
                        elif not (np.isfinite(pre_max_dd_usd) and pre_max_dd_usd <= 60.0):
                            fail_reason = "fail_dd"
                        elif not (np.isfinite(pre_m["pf"]) and pre_m["pf"] >= 1.05):
                            fail_reason = "fail_pf"
                        elif not (np.isfinite(pre_m["ev_r"]) and pre_m["ev_r"] >= 0.0):
                            fail_reason = "fail_ev"

                        row = {
                            "scenario": str(scen_name),
                            "min_lot": float(mkt_s.min_lot),
                            "tp1_over_cost_k": float(econ_k),
                            **dataclasses.asdict(ex.tpslh),
                            "entry": ex.entry,
                            "tp1_close_frac": float(ex.tp1_close_frac),
                            "tp2_mult": float(ex.tp2_mult),
                            "risk_cap_usd": float(risk_cap),
                            "daily_stop_loss_usd": float(dsl),
                            "max_parallel_same_dir": int(mp),
                            "tickets_per_signal": int(tps),
                            "cooldown_bars": int(cd),
                            "q": float(q),
                            "q_tail": float(q_tail),
                            "econ_pruned": int(econ_pruned),
                            "pre_epd": float(pre_m["epd"]),
                            "pre_tpd": float(pre_m["tpd"]),
                            "os_epd": float(os_m["epd"]),
                            "os_tpd": float(os_m["tpd"]),
                            "all_epd": float(all_m["epd"]),
                            "all_tpd": float(all_m["tpd"]),
                            "tpd_preos": float(pre_m["tpd"]),
                            "tpd_os": float(os_m["tpd"]),
                            "tpd_all": float(all_m["tpd"]),
                            "pre_hit_tp1": float(pre_m["hit_tp1"]),
                            "pre_hit_tp2": float(pre_m["hit_tp2"]),
                            "pre_pf": float(pre_m["pf"]),
                            "pre_ev_r": float(pre_m["ev_r"]),
                            "all_pf": float(all_m["pf"]),
                            "maxDD_usd": float(pre_max_dd_usd),
                            "maxDD_pct": float(pre_max_dd_pct),
                            "os_maxDD_usd": float(os_max_dd_usd),
                            "os_maxDD_pct": float(os_max_dd_pct),
                            "all_maxDD_usd": float(all_max_dd_usd),
                            "all_maxDD_pct": float(all_max_dd_pct),
                            "run_status": str(meta.get("run_status", "")),
                            "stop_out_ts": stop_out_ts,
                            "pre_stop_out": int(pre_stop_out),
                            "dd_trigger_count": int(meta.get("dd_trigger_count", 0)),
                            "dd_stop_skip": int(meta.get("dd_stop_skip", 0)),
                            "fail_reason": str(fail_reason),
                            "gate1_take_rate_tail_preos": float(take_rate_tail_pre),
                            "filter_brier_long": float(filter_metrics.get("long", {}).get("brier", float("nan"))),
                            "filter_auc_long": float(filter_metrics.get("long", {}).get("auc", float("nan"))),
                            "filter_brier_short": float(filter_metrics.get("short", {}).get("brier", float("nan"))),
                            "filter_auc_short": float(filter_metrics.get("short", {}).get("auc", float("nan"))),
                        }

                        fr = str(fail_reason) if str(fail_reason) else "pass"
                        fail_counts[fr] += 1
                        global_bounds["pre_tpd_max"] = max(float(global_bounds["pre_tpd_max"]), _f(row.get("pre_tpd"), 0.0))
                        global_bounds["pre_hit_tp1_max"] = max(float(global_bounds["pre_hit_tp1_max"]), _f(row.get("pre_hit_tp1"), -float("inf")))
                        global_bounds["pre_pf_max"] = max(float(global_bounds["pre_pf_max"]), _f(row.get("pre_pf"), -float("inf")))
                        global_bounds["pre_ev_r_max"] = max(float(global_bounds["pre_ev_r_max"]), _f(row.get("pre_ev_r"), -float("inf")))
                        global_bounds["pre_maxDD_usd_min"] = min(float(global_bounds["pre_maxDD_usd_min"]), _f(row.get("maxDD_usd"), float("inf")))

                        cand_id = _cand_id(row)
                        row["candidate_id"] = str(cand_id)
                        score = _score_tuple(row)

                        keep = int(len(keep_heap)) < int(max_candidates_keep) or (keep_heap and score > keep_heap[0][0])
                        if keep:
                            # compute stability breakdowns for kept candidates only
                            lot_max_by_ticket = lotmax_map[(float(risk_cap), int(tps))]
                            _pre_bd, _os_bd, _all_bd, meta_bd = simulate_trading_fast_metrics(
                                ctx_fast,
                                mkt_s,
                                risk_cfg,
                                arr=arr,
                                pass_indices=pass_idx,
                                lot_max_by_ticket=lot_max_by_ticket,
                                daily_stop_loss_usd=float(dsl),
                                max_parallel_same_dir=int(mp),
                                tickets_per_signal=int(tps),
                                tp1_close_frac=float(ex.tp1_close_frac),
                                cooldown_bars=int(cd),
                                with_breakdowns=True,
                            )
                            row["ev_r_preos_p25"] = float(meta_bd.get("ev_r_preos_p25", float("nan")))
                            kept_breakdowns[str(cand_id)] = {
                                "preos_year_table": meta_bd.get("preos_year_table") or [],
                                "preos_regime_adx3_table": meta_bd.get("preos_regime_adx3_table") or [],
                            }

                            if int(len(keep_heap)) < int(max_candidates_keep):
                                heapq.heappush(keep_heap, (score, str(cand_id), row))
                            else:
                                worst_score, worst_id, _ = keep_heap[0]
                                if score > worst_score:
                                    _, old_id, _ = heapq.heapreplace(keep_heap, (score, str(cand_id), row))
                                    kept_breakdowns.pop(str(old_id), None)

    # Materialize kept candidates (top-K) for selection/reporting.
    if keep_heap:
        keep_sorted = sorted(keep_heap, key=lambda t: t[0], reverse=True)
        candidates_rows = [it[2] for it in keep_sorted]
    else:
        candidates_rows = []

    # Two-phase objective:
    # Phase-1: find feasible region (PF/EV/DD + OS_epd>0).
    # Phase-2: within Phase-1, require tpd>=1.5 and maximize hit@TP1.
    candidates_df_tmp = pd.DataFrame(candidates_rows) if candidates_rows else pd.DataFrame()
    selected_row: Optional[Dict[str, Any]] = None
    best_effort_row: Optional[Dict[str, Any]] = None
    if not candidates_df_tmp.empty:
        dfc = candidates_df_tmp.copy()
        for c in ("os_epd", "pre_pf", "pre_ev_r", "maxDD_usd", "pre_tpd", "pre_hit_tp1", "pre_stop_out", "ev_r_preos_p25", "gate1_take_rate_tail_preos"):
            if c in dfc.columns:
                dfc[c] = pd.to_numeric(dfc[c], errors="coerce")

        os_ok = pd.to_numeric(dfc.get("os_epd"), errors="coerce").fillna(0.0) > 0.0
        dd_ok = pd.to_numeric(dfc.get("maxDD_usd"), errors="coerce").fillna(float("inf")) <= 60.0
        pf_ok = pd.to_numeric(dfc.get("pre_pf"), errors="coerce").fillna(-float("inf")) >= 1.05
        ev_ok = pd.to_numeric(dfc.get("pre_ev_r"), errors="coerce").fillna(-float("inf")) >= 0.0
        so_ok = pd.to_numeric(dfc.get("pre_stop_out"), errors="coerce").fillna(0).astype(int) == 0
        stab_ok = pd.to_numeric(dfc.get("ev_r_preos_p25"), errors="coerce").fillna(-float("inf")) >= 0.0
        gate1_ok = pd.to_numeric(dfc.get("gate1_take_rate_tail_preos"), errors="coerce").fillna(0.0) >= float(thr_cfg.gate1_take_rate_min)
        phase1_df = dfc[os_ok & dd_ok & pf_ok & ev_ok & so_ok & stab_ok & gate1_ok].copy()
        phase2_df = phase1_df[pd.to_numeric(phase1_df.get("pre_tpd"), errors="coerce").fillna(0.0) >= 1.5].copy()
        strict_df = phase2_df[pd.to_numeric(phase2_df.get("pre_hit_tp1"), errors="coerce").fillna(-float("inf")) >= 0.73].copy()

        if not strict_df.empty:
            strict_df = strict_df.sort_values(
                ["pre_hit_tp1", "pre_pf", "pre_ev_r", "maxDD_usd", "pre_tpd"],
                ascending=[False, False, False, True, False],
                kind="mergesort",
            )
            selected_row = strict_df.iloc[0].to_dict()
            best_effort_row = selected_row
        elif not phase2_df.empty:
            phase2_df = phase2_df.sort_values(
                ["pre_hit_tp1", "pre_pf", "pre_ev_r", "maxDD_usd", "pre_tpd"],
                ascending=[False, False, False, True, False],
                kind="mergesort",
            )
            best_effort_row = phase2_df.iloc[0].to_dict()
        elif not phase1_df.empty:
            phase1_df = phase1_df.sort_values(
                ["pre_tpd", "pre_hit_tp1", "pre_pf", "pre_ev_r", "maxDD_usd"],
                ascending=[False, False, False, False, True],
                kind="mergesort",
            )
            best_effort_row = phase1_df.iloc[0].to_dict()
        else:
            # fallback: keep OS_epd>0 if possible, otherwise pick the least-bad by PF/EV.
            be_df = dfc[os_ok].copy()
            if be_df.empty:
                be_df = dfc.copy()
            if not be_df.empty:
                be_df = be_df.sort_values(
                    ["pre_pf", "pre_ev_r", "pre_hit_tp1", "maxDD_usd", "pre_tpd"],
                    ascending=[False, False, False, True, False],
                    kind="mergesort",
                )
                best_effort_row = be_df.iloc[0].to_dict()

    strict_pass = bool(selected_row is not None)

    candidates_df = pd.DataFrame(candidates_rows)
    candidates_df.to_csv(paths.artifacts_dir / "candidates.csv", index=False)

    # Regime/year report for all kept candidates (pre-OS only; hard requirement).
    regime_report = {
        "version": "20260112_012",
        "regime_definition": {"name": "ADX14_3bins", "bins": [{"id": 0, "label": "ADX<20"}, {"id": 1, "label": "20<=ADX<30"}, {"id": 2, "label": "ADX>=30"}]},
        "candidates_n": int(len(candidates_rows)),
        "fail_counts": dict(fail_counts),
        "global_bounds_preos": {k: (float(v) if np.isfinite(float(v)) else None) for k, v in dict(global_bounds).items()},
        "candidates": kept_breakdowns,
    }
    write_json(paths.artifacts_dir / "regime_report.json", regime_report)

    # =============================
    # Finalize outputs
    # =============================
    selected_config: Dict[str, Any] = {}
    thresholds_payload: Dict[str, Any] = {}
    filter_report_out: Dict[str, Any] = {}
    execution_audit: Dict[str, Any] = {}

    final_row = selected_row if selected_row is not None else best_effort_row

    if final_row is not None:
        # re-run final with thresholds storage + execution audit
        mkt_s = dataclasses.replace(mkt, min_lot=float(final_row.get("min_lot") or mkt.min_lot), lot_step=0.01)
        risk_s = dataclasses.replace(risk_cfg, equity_floor_usd=float(mkt_s.initial_capital_usd) - 45.0)
        econ_k = float(final_row.get("tp1_over_cost_k") or float(min(econ_k_grid) if econ_k_grid else 1.2))

        ex = ExitConfig(
            entry=str(final_row["entry"]),
            tpslh=TPSLH(H=int(final_row["H"]), tp1_atr_mult=float(final_row["tp1_atr_mult"]), sl_atr_mult=float(final_row["sl_atr_mult"])),
            tp1_close_frac=float(final_row["tp1_close_frac"]),
            tp2_mult=float(final_row["tp2_mult"]),
        )
        strat = StrategyConfig(
            exit=ex,
            filt=FilterConfig(
                q=float(final_row["q"]),
                q_tail=float(final_row.get("q_tail") or 0.80),
                lookback_days=int(thr_cfg.score_lookback_days),
                min_hist=int(thr_cfg.min_score_history),
            ),
            risk_cap_usd=float(final_row["risk_cap_usd"]),
            daily_stop_loss_usd=float(final_row["daily_stop_loss_usd"]),
            max_parallel_same_dir=int(final_row.get("max_parallel_same_dir") or final_row.get("max_parallel_tickets") or 3),
            tickets_per_signal=int(final_row.get("tickets_per_signal") or 1),
            cooldown_bars=int(final_row.get("cooldown_bars") or 0),
        )

        # recompute ds & scores using selected exit config
        ds, out_meta = compute_event_outcomes(mkt_s, df=df0, ev=ev_feat, ex=ex)
        sig_i2 = ds["signal_i"].astype(int).to_numpy()
        ds["vol_regime"] = np.array([int(regimes["vol_regime"][i]) for i in sig_i2], dtype=int)
        ds["trend_regime"] = np.array([int(regimes["trend_regime"][i]) for i in sig_i2], dtype=int)

        pre_end = to_utc_ts(time_cfg.preos_end_utc)
        scored_parts: List[pd.DataFrame] = []
        filter_report_out = {"exit_cfg": dataclasses.asdict(ex), "sides": {}}
        for side in ("long", "short"):
            side_ds = ds[ds["side"] == side].copy()
            if side_ds.empty:
                continue
            # trading scores (pre-OS OOF + OS via full-pre model)
            scored_side, meta_side = score_side_preos_os(time_cfg, cv_cfg, mdl_cfg, df_prices=df0, ds_all_side=side_ds)
            pre0 = to_utc_ts(time_cfg.preos_start_utc)
            pre_mask = (scored_side["_entry_ts"] >= pre0) & (scored_side["_entry_ts"] <= pre_end)
            ds_pre = scored_side.loc[pre_mask].copy()
            ds_pre = ds_pre.sort_values("_entry_ts", kind="mergesort").reset_index(drop=True)

            y_win = (pd.to_numeric(ds_pre["net_r"], errors="coerce").to_numpy(dtype=float) > 0.0).astype(int)
            p_win = pd.to_numeric(ds_pre.get("p_score"), errors="coerce").to_numpy(dtype=float)
            y_tail = (pd.to_numeric(ds_pre.get("mae_r"), errors="coerce").to_numpy(dtype=float) <= -1.0).astype(int)
            p_tail = pd.to_numeric(ds_pre.get("p_tail"), errors="coerce").to_numpy(dtype=float)

            win_oof = {"brier": float(brier_score(y_win, p_win)), "auc": float(roc_auc(y_win, p_win)), "meta": meta_side.get("pre_oof_win", {})}
            tail_oof = {"brier": float(brier_score(y_tail, p_tail)), "auc": float(roc_auc(y_tail, p_tail)), "meta": meta_side.get("pre_oof_tail", {})}

            # regime stability (OOF; pre-OS only)
            reg_table = []
            if "vol_regime" in ds_pre.columns and "trend_regime" in ds_pre.columns:
                for (vr, tr), g in ds_pre.groupby(["vol_regime", "trend_regime"], dropna=False):
                    yw = (pd.to_numeric(g["net_r"], errors="coerce").to_numpy(dtype=float) > 0.0).astype(int)
                    pw = pd.to_numeric(g.get("p_score"), errors="coerce").to_numpy(dtype=float)
                    yt = (pd.to_numeric(g.get("mae_r"), errors="coerce").to_numpy(dtype=float) <= -1.0).astype(int)
                    pt = pd.to_numeric(g.get("p_tail"), errors="coerce").to_numpy(dtype=float)
                    reg_table.append(
                        {
                            "vol_regime": int(vr),
                            "trend_regime": int(tr),
                            "n": int(len(g)),
                            "win_brier": float(brier_score(yw, pw)),
                            "win_auc": float(roc_auc(yw, pw)),
                            "tail_brier": float(brier_score(yt, pt)),
                            "tail_auc": float(roc_auc(yt, pt)),
                            "hit_tp1": float(np.mean(g["tp1_hit"].astype(int))) if len(g) else float("nan"),
                            "ev_r": float(np.nanmean(pd.to_numeric(g["net_r"], errors="coerce").to_numpy(dtype=float))) if len(g) else float("nan"),
                        }
                    )

            win_meta = meta_side.get("pre_oof_win") or {}
            tail_meta = meta_side.get("pre_oof_tail") or {}
            filter_report_out["sides"][side] = {
                "win_oof": win_oof,
                "tail_oof": tail_oof,
                "regime_oof_table": reg_table,
                "feature_prune_win": win_meta.get("feature_prune") or {},
                "feature_prune_tail": tail_meta.get("feature_prune") or {},
                "feature_importance_win_top": (win_meta.get("split_stats") or {}).get("top_gain_features") or [],
                "feature_importance_tail_top": (tail_meta.get("split_stats") or {}).get("top_gain_features") or [],
                "scoring_meta": meta_side,
            }
            scored_parts.append(scored_side)
        scored_all = pd.concat(scored_parts, axis=0, ignore_index=True).sort_values(["entry_i", "_entry_ts"], kind="mergesort").reset_index(drop=True)

        # econ prune (tp1_over_cost_min, same as search-stage): TP1_R >= k * cost_R
        tp1_r_all = pd.to_numeric(scored_all.get("tp1_r"), errors="coerce").to_numpy(dtype=float)
        cost_r_all = pd.to_numeric(scored_all.get("cost_r"), errors="coerce").to_numpy(dtype=float)
        econ_mask = np.isfinite(tp1_r_all) & np.isfinite(cost_r_all) & (tp1_r_all >= float(econ_k) * cost_r_all - 1e-12)
        econ_pruned_final = int(len(scored_all) - int(np.sum(econ_mask)))
        scored_all = scored_all.loc[econ_mask].copy()
        scored_all = scored_all.sort_values(["entry_i", "_entry_ts"], kind="mergesort").reset_index(drop=True)

        # Gate diagnostics (pre-OS only; post econ-prune, pre execution/risk mgmt)
        pre0_ns = int(to_utc_ts(time_cfg.preos_start_utc).value)
        pre1_ns = int(pre_end.value)
        for side in ("long", "short"):
            if side not in (filter_report_out.get("sides") or {}):
                continue
            df_side = scored_all[scored_all["side"] == side].copy()
            if df_side.empty:
                continue
            arr_side = build_scored_event_arrays(df_side, mkt=mkt_s)
            masks = compute_filter_masks(arr_side, filt=strat.filt)
            pre_mask_side = (arr_side.entry_ts_ns >= int(pre0_ns)) & (arr_side.entry_ts_ns <= int(pre1_ns))
            y_tail = (np.asarray(arr_side.mae_r, dtype=float) <= -1.0) & np.isfinite(np.asarray(arr_side.mae_r, dtype=float))

            def _rate(mask: np.ndarray) -> float:
                m = np.asarray(mask, dtype=bool)
                if int(np.sum(m)) <= 0:
                    return float("nan")
                return float(np.mean(y_tail[m]))

            def _take(mask: np.ndarray) -> float:
                m = np.asarray(mask, dtype=bool)
                if int(np.sum(pre_mask_side)) <= 0:
                    return 0.0
                return float(np.mean(m[pre_mask_side]))

            pre_mask_side_b = np.asarray(pre_mask_side, dtype=bool)
            tail_ok = np.asarray(masks.get("tail_ok"), dtype=bool)
            win_ok = np.asarray(masks.get("win_ok"), dtype=bool)
            pass_ok = np.asarray(masks.get("pass"), dtype=bool)

            base_tail_rate = _rate(pre_mask_side_b)
            tail_rate_after_gate1 = _rate(pre_mask_side_b & tail_ok)
            tail_rate_after_gate2 = _rate(pre_mask_side_b & pass_ok)

            filter_report_out["sides"][side]["gating_preos"] = {
                "gate1_take_rate_min": float(thr_cfg.gate1_take_rate_min),
                "take_rate_tail": float(_take(tail_ok)),
                "take_rate_win": float(_take(win_ok)),
                "take_rate_pass": float(_take(pass_ok)),
                "big_loss_rate_base": float(base_tail_rate),
                "big_loss_rate_after_gate1": float(tail_rate_after_gate1),
                "big_loss_rate_after_gate2": float(tail_rate_after_gate2),
            }

        trades_final, meta_final = simulate_trading(time_cfg, mkt_s, risk_s, df_prices=df0, scored_events=scored_all, strat=strat, store_thresholds=True)
        trades_final.to_csv(paths.artifacts_dir / "backtest_mode4_trades.csv", index=False)

        pre_tr = slice_segment(trades_final, start=time_cfg.preos_start_utc, end=time_cfg.preos_end_utc)
        os_tr = slice_segment(trades_final, start=time_cfg.os_start_utc, end=time_cfg.backtest_end_utc)

        def _signals_n(tdf: pd.DataFrame) -> int:
            if tdf.empty:
                return 0
            if "signal_i" in tdf.columns:
                return int(pd.to_numeric(tdf["signal_i"], errors="coerce").dropna().astype(int).nunique())
            return int(len(tdf))

        pre_signals_taken = _signals_n(pre_tr)
        os_signals_taken = _signals_n(os_tr)

        thresholds_payload = {
            "version": "20260112_012",
            "q_win": float(strat.filt.q),
            "q_tail": float(strat.filt.q_tail),
            "lookback_days": int(strat.filt.lookback_days),
            "min_hist": int(strat.filt.min_hist),
            "min_regime_hist": int(strat.filt.min_regime_hist),
            "thresholds_daily": meta_final.get("thresholds_daily") or {},
        }
        write_json(paths.artifacts_dir / "thresholds_walkforward.json", thresholds_payload)

        # execution audit
        audit_c = meta_final.get("audit") or {}
        # econ-prune attribution (on raw events; uses signal-time ATR)
        _raw = ev_feat.copy()
        _raw["_signal_ts"] = pd.to_datetime(_raw["_signal_ts"], utc=True, errors="coerce")
        _raw = _raw[pd.notna(_raw["_signal_ts"])]
        pre_mask_raw = (_raw["_signal_ts"] >= to_utc_ts(time_cfg.preos_start_utc)) & (_raw["_signal_ts"] <= to_utc_ts(time_cfg.preos_end_utc))
        os_mask_raw = (_raw["_signal_ts"] >= to_utc_ts(time_cfg.os_start_utc)) & (_raw["_signal_ts"] <= to_utc_ts(time_cfg.backtest_end_utc))
        atr_ref_raw = pd.to_numeric(_raw["atr_ref"], errors="coerce").to_numpy(dtype=float)
        cost_total_px = float(mkt_s.roundtrip_cost_price) + float(mkt_s.slippage_buffer_price)
        sl_dist_raw = atr_ref_raw * float(ex.tpslh.sl_atr_mult)
        cost_r_raw = np.where(np.isfinite(sl_dist_raw) & (sl_dist_raw > 1e-12), float(cost_total_px) / sl_dist_raw, np.nan)
        tp1_r_raw = float(ex.tpslh.tp1_atr_mult) / max(1e-12, float(ex.tpslh.sl_atr_mult))
        # pruned if TP1_R < k * cost_R
        econ_prune_mask = np.isfinite(cost_r_raw) & (float(tp1_r_raw) < float(econ_k) * cost_r_raw - 1e-12)

        raw_pre = int(len(ev_feat[(ev_feat["_signal_ts"] >= to_utc_ts(time_cfg.preos_start_utc)) & (ev_feat["_signal_ts"] <= to_utc_ts(time_cfg.preos_end_utc))]))
        raw_os = int(len(ev_feat[(ev_feat["_signal_ts"] >= to_utc_ts(time_cfg.os_start_utc)) & (ev_feat["_signal_ts"] <= to_utc_ts(time_cfg.backtest_end_utc))]))
        days_pre = max(1.0, (to_utc_ts(time_cfg.preos_end_utc) - to_utc_ts(time_cfg.preos_start_utc)).total_seconds() / 86400.0)
        days_os = max(1.0, (to_utc_ts(time_cfg.backtest_end_utc) - to_utc_ts(time_cfg.os_start_utc)).total_seconds() / 86400.0)

        # DD governor audit (required)
        rsc = meta_final.get("risk_scale_changes") or []
        scales = []
        down_n = 0
        up_n = 0
        for it in rsc:
            try:
                scales.append(float(it.get("risk_scale")))
            except Exception:
                continue
            evv = str(it.get("event") or "")
            if evv == "down":
                down_n += 1
            elif evv == "up":
                up_n += 1
        scales2 = [s for s in scales if np.isfinite(s)]
        risk_scale_summary = {
            "changes_n": int(len(rsc)),
            "down_n": int(down_n),
            "up_n": int(up_n),
            "unique_scales": sorted({float(s) for s in scales2}) if scales2 else [],
            "final_scale": float(scales2[-1]) if scales2 else float("nan"),
        }

        execution_audit = {
            "version": "20260112_012",
            "run_status": str(meta_final.get("run_status", "")),
            "stop_out_ts": meta_final.get("stop_out_ts"),
            "equity_floor_usd": float(risk_s.equity_floor_usd),
            # drawdown meta (authoritative; from final simulator)
            "max_dd_usd_preos": float(meta_final.get("max_dd_usd_preos", float("nan"))),
            "max_dd_pct_preos": float(meta_final.get("max_dd_pct_preos", float("nan"))),
            "max_dd_usd_os": float(meta_final.get("max_dd_usd_os", float("nan"))),
            "max_dd_pct_os": float(meta_final.get("max_dd_pct_os", float("nan"))),
            "max_dd_usd_all": float(meta_final.get("max_dd_usd", float("nan"))),
            "max_dd_pct_all": float(meta_final.get("max_dd_pct", float("nan"))),
            "dd_trigger_count": int(meta_final.get("dd_trigger_count") or 0),
            "dd_stop_skip": int(meta_final.get("dd_stop_skip") or 0),
            "risk_scale_levels": meta_final.get("risk_scale_levels") or [],
            "risk_scale_changes": rsc,
            "risk_scale_summary": risk_scale_summary,
            # raw events (fixed mode4 signal, before econ/filters)
            "raw_events_preos": raw_pre,
            "raw_events_os": raw_os,
            "raw_events_epd_preos": float(raw_pre / days_pre),
            "raw_events_epd_os": float(raw_os / days_os),
            # required aliases (spec asks raw_events_epd/take_rate without suffix)
            "raw_events_epd": float(raw_pre / days_pre),
            "take_rate": float(pre_signals_taken / max(1, raw_pre)),
            # take rate against raw events
            "take_rate_preos": float(pre_signals_taken / max(1, raw_pre)),
            "take_rate_os": float(os_signals_taken / max(1, raw_os)),
            # econ prune diagnostics (raw-event level)
            "tp1_over_cost_k": float(econ_k),
            "tp1_atr_mult_selected": float(ex.tpslh.tp1_atr_mult),
            "econ_pruned_events_final": int(econ_pruned_final),
            "events_in_total": int(out_meta.get("events_in", 0)),
            "econ_pruned_preos": int(np.sum(econ_prune_mask & pre_mask_raw.to_numpy(dtype=bool))),
            "econ_pruned_os": int(np.sum(econ_prune_mask & os_mask_raw.to_numpy(dtype=bool))),
            "events_after_econ_preos": int(
                len(scored_all[(scored_all["_entry_ts"] >= to_utc_ts(time_cfg.preos_start_utc)) & (scored_all["_entry_ts"] <= to_utc_ts(time_cfg.preos_end_utc))])
            )
            if not scored_all.empty
            else 0,
            "events_after_econ_os": int(len(scored_all[scored_all["_entry_ts"] >= to_utc_ts(time_cfg.os_start_utc)])) if not scored_all.empty else 0,
            # required skip counters (pre-OS; keep OS too for diagnosis)
            "skipped_over_risk_cap": int(audit_c.get("skipped_over_risk_cap_preOS", 0)),
            "skipped_min_lot": int(audit_c.get("skipped_min_lot_preOS", 0)),
            "skipped_over_risk_cap_os": int(audit_c.get("skipped_over_risk_cap_OS", 0)),
            "skipped_min_lot_os": int(audit_c.get("skipped_min_lot_OS", 0)),
            # full audit counters (for deeper debugging)
            "audit_counters": dict(audit_c),
        }
        write_json(paths.artifacts_dir / "execution_audit.json", execution_audit)

        selected_config = {
            "version": "20260112_012",
            "strict_pass": bool(strict_pass),
            "selected_row": selected_row,
            "best_effort_row": best_effort_row,
            "final_row": final_row,
            "scenario": {"name": str(final_row.get("scenario")), "min_lot": float(mkt_s.min_lot), "lot_step": float(mkt_s.lot_step)},
            "market": dataclasses.asdict(mkt_s),
            "risk": dataclasses.asdict(risk_s),
            "exit": dataclasses.asdict(ex),
            "econ_prune": {"k": float(econ_k), "rule": "tp1_r >= k*cost_r; cost_r=(roundtrip_cost_price+slippage_buffer_price)/sl_dist"},
            "strategy": {
                "risk_cap_usd": float(strat.risk_cap_usd),
                "daily_stop_loss_usd": float(strat.daily_stop_loss_usd),
                "max_parallel_same_dir": int(strat.max_parallel_same_dir),
                "tickets_per_signal": int(strat.tickets_per_signal),
                "cooldown_bars": int(strat.cooldown_bars),
            },
            "filter": dataclasses.asdict(strat.filt),
            "purge_embargo": {"purge_bars": int(cv_cfg.purge_bars), "embargo_bars": int(cv_cfg.embargo_bars)},
        }
        write_json(paths.artifacts_dir / "selected_config.json", selected_config)

        write_json(paths.artifacts_dir / "filter_report.json", filter_report_out)

    else:
        # no final row (empty candidates or cannot satisfy OS_epd>0 feasibility)
        write_json(paths.artifacts_dir / "selected_config.json", {"ok": False, "strict_pass": bool(strict_pass), "selected_row": None, "best_effort_row": None})
        write_json(paths.artifacts_dir / "filter_report.json", {"ok": False, "reason": "no_final_row"})
        write_json(paths.artifacts_dir / "thresholds_walkforward.json", {"ok": False, "reason": "no_final_row"})
        write_json(paths.artifacts_dir / "execution_audit.json", {"ok": False, "reason": "no_final_row"})
        pd.DataFrame().to_csv(paths.artifacts_dir / "backtest_mode4_trades.csv", index=False)

    # Copy script to Desktop + artifacts (hard delivery requirement)
    shutil.copy2(script_path, paths.desktop_script_copy)

    # Create artifacts subdirectories and copy required files (hard delivery requirement)
    artifact_map = {
        "manifest.json": "manifest",
        "leakage_audit.json": "leakage_audit",
        "candidates.csv": "candidates",
        "selected_config.json": "selected",
        "thresholds_walkforward.json": "thresholds",
        "filter_report.json": "filter",
        "regime_report.json": "regime",
        "backtest_mode4_trades.csv": "trades",
        "execution_audit.json": "execution_audit",
        "lot_math_audit.json": "lot_math_audit",
        paths.desktop_script_copy.name: "脚本副本",
    }
    for fname, subdir in artifact_map.items():
        src = paths.artifacts_dir / str(fname)
        if src.exists():
            copy2_into_dir(src, paths.artifacts_dir / str(subdir))

    # =============================
    # Write report 012.txt (required)
    # =============================
    # FINAL_PRINT_BLOCK must appear at the top.
    pre_metrics: Dict[str, Any] = {}
    os_metrics: Dict[str, Any] = {}
    all_metrics: Dict[str, Any] = {}
    pre_dd_usd = float("nan")
    pre_dd_pct = float("nan")
    os_dd_usd = float("nan")
    os_dd_pct = float("nan")
    all_dd_usd = float("nan")
    all_dd_pct = float("nan")

    if final_row is not None and (paths.artifacts_dir / "backtest_mode4_trades.csv").exists():
        tr = pd.read_csv(paths.artifacts_dir / "backtest_mode4_trades.csv")
        mkt_r = dataclasses.replace(mkt, min_lot=float(final_row.get("min_lot") or mkt.min_lot), lot_step=0.01)
        pre_tr = slice_segment(tr, start=time_cfg.preos_start_utc, end=time_cfg.preos_end_utc)
        os_tr = slice_segment(tr, start=time_cfg.os_start_utc, end=time_cfg.backtest_end_utc)
        all_tr = slice_segment(tr, start=time_cfg.backtest_start_utc, end=time_cfg.backtest_end_utc)
        pre_metrics = metrics_from_trades(time_cfg, mkt_r, trades=pre_tr, start_utc=time_cfg.preos_start_utc, end_utc=time_cfg.preos_end_utc)
        os_metrics = metrics_from_trades(time_cfg, mkt_r, trades=os_tr, start_utc=time_cfg.os_start_utc, end_utc=time_cfg.backtest_end_utc)
        all_metrics = metrics_from_trades(time_cfg, mkt_r, trades=all_tr, start_utc=time_cfg.backtest_start_utc, end_utc=time_cfg.backtest_end_utc)

        pre_dd_usd = float(execution_audit.get("max_dd_usd_preos", final_row.get("maxDD_usd", float("nan"))))
        pre_dd_pct = float(execution_audit.get("max_dd_pct_preos", final_row.get("maxDD_pct", float("nan"))))
        os_dd_usd = float(execution_audit.get("max_dd_usd_os", final_row.get("os_maxDD_usd", float("nan"))))
        os_dd_pct = float(execution_audit.get("max_dd_pct_os", final_row.get("os_maxDD_pct", float("nan"))))
        all_dd_usd = float(execution_audit.get("max_dd_usd_all", final_row.get("all_maxDD_usd", float("nan"))))
        all_dd_pct = float(execution_audit.get("max_dd_pct_all", final_row.get("all_maxDD_pct", float("nan"))))

    lines: List[str] = []
    lines.append("==============================")
    lines.append("FINAL_PRINT_BLOCK")
    lines.append("==============================")
    if final_row is None:
        lines.append("- preOS: NA")
        lines.append("- OS:   NA")
        lines.append("- All:  NA")
    else:
        lines.append(
            f"- preOS: epd={fmt(pre_metrics.get('epd'),4)}, tpd={fmt(pre_metrics.get('tpd'),4)}, hit@TP1={fmt(pre_metrics.get('hit_tp1'),4)}, hit@TP2={fmt(pre_metrics.get('hit_tp2'),4)}, PF={fmt(pre_metrics.get('pf'),4)}, ev_r={fmt(pre_metrics.get('ev_r'),4)}, maxDD_usd={fmt(pre_dd_usd,2)}, maxDD%={fmt(pre_dd_pct,2)}"
        )
        lines.append(
            f"- OS:   epd={fmt(os_metrics.get('epd'),4)}, tpd={fmt(os_metrics.get('tpd'),4)}, hit@TP1={fmt(os_metrics.get('hit_tp1'),4)}, hit@TP2={fmt(os_metrics.get('hit_tp2'),4)}, PF={fmt(os_metrics.get('pf'),4)}, ev_r={fmt(os_metrics.get('ev_r'),4)}, maxDD_usd={fmt(os_dd_usd,2)}, maxDD%={fmt(os_dd_pct,2)}"
        )
        lines.append(
            f"- All:  epd={fmt(all_metrics.get('epd'),4)}, tpd={fmt(all_metrics.get('tpd'),4)}, hit@TP1={fmt(all_metrics.get('hit_tp1'),4)}, hit@TP2={fmt(all_metrics.get('hit_tp2'),4)}, PF={fmt(all_metrics.get('pf'),4)}, ev_r={fmt(all_metrics.get('ev_r'),4)}, maxDD_usd={fmt(all_dd_usd,2)}, maxDD%={fmt(all_dd_pct,2)}"
        )
    lines.append("")

    lines.append("==============================")
    lines.append("011→012 STRUCTURAL_CHANGES（改了什么/为什么）")
    lines.append("==============================")
    lines.append("- 引入 TicketPolicy（拆单）：同一信号同方向可并行 N 张（1/2/3），按 risk_share 分摊风险；票据级 exit：TP1 快出+BE / TP2 / TAIL（trailing/时间止盈）。")
    lines.append("- 频率口径切换为 tpd>=1.5（tickets/day），同时保留并输出 epd；candidates.csv 增加 tpd_preos/tpd_os/tpd_all。")
    lines.append("- 候选搜索加入经济性下界剪枝：tp1_r >= k*cost_r（k∈{1.2,1.5,2.0}），避免“高 hit 但结构必亏”。")
    lines.append("- 目标函数两阶段：Phase-1 先找可行域（PF/EV/DD/OS_epd），Phase-2 在可行域内冲刺 hit@TP1 且约束 tpd>=1.5。")
    lines.append("- 过滤器升级：mode4_long/mode4_short 独立 walk-forward；两级门控（Gate-1 y_tail 剔除大亏 + take_rate>=0.6，Gate-2 y 提升 PF/EV）。")
    lines.append("- 稳健性防 multiple-testing：pre-OS 按年与 ADX 三档分桶评估，并加入稳定性约束 ev_r_preos_p25>=0（写入 regime_report.json 且本报告展开）。")
    lines.append("")

    lines.append("==============================")
    lines.append("COST_MODEL（剪枝口径）")
    lines.append("==============================")
    total_cost_px = float(mkt.roundtrip_cost_price) + float(mkt.slippage_buffer_price)
    lines.append(
        f"- roundtrip_cost_price={fmt(mkt.roundtrip_cost_price,6)} | slippage_buffer_price={fmt(mkt.slippage_buffer_price,6)} | total_cost_px={fmt(total_cost_px,6)}"
    )
    lines.append("- sl_dist = atr_ref * sl_atr_mult（价格单位）；cost_r = total_cost_px / sl_dist（R 单位）。")
    lines.append("- tp1_r = tp1_atr_mult / sl_atr_mult（R 单位）；剪枝：tp1_r >= k*cost_r（k∈{1.2,1.5,2.0}）。")
    lines.append("")

    lines.append("==============================")
    lines.append("SCENARIO_COMPARE（pre-OS 选型）")
    lines.append("==============================")
    for scen in ("S1", "S2"):
        sub = candidates_df[candidates_df.get("scenario") == scen].copy() if not candidates_df.empty else pd.DataFrame()
        pass_sub = sub[sub.get("fail_reason", "").astype(str) == ""].copy() if not sub.empty else pd.DataFrame()
        if (not pass_sub.empty) and ("ev_r_preos_p25" in pass_sub.columns):
            pass_sub["ev_r_preos_p25"] = pd.to_numeric(pass_sub["ev_r_preos_p25"], errors="coerce")
            pass_sub = pass_sub[pass_sub["ev_r_preos_p25"].fillna(-float("inf")) >= 0.0].copy()
        if not pass_sub.empty:
            pass_sub = pass_sub.sort_values(["pre_ev_r", "pre_pf", "pre_tpd"], ascending=[False, False, False], kind="mergesort")
            row = pass_sub.iloc[0].to_dict()
            lines.append(
                f"- {scen}: PASS pre_epd={fmt(row.get('pre_epd'),4)}, pre_tpd={fmt(row.get('pre_tpd'),4)}, hit@TP1={fmt(row.get('pre_hit_tp1'),4)}, PF={fmt(row.get('pre_pf'),4)}, ev_r={fmt(row.get('pre_ev_r'),4)}, maxDD_usd={fmt(row.get('maxDD_usd'),2)}, ev_r_preos_p25={fmt(row.get('ev_r_preos_p25'),4)}"
            )
            continue

        if "os_epd" in sub.columns:
            sub2 = sub[pd.to_numeric(sub["os_epd"], errors="coerce").fillna(0.0) > 0.0].copy()
            if not sub2.empty:
                sub = sub2
        if sub.empty:
            lines.append(f"- {scen}: no candidates")
            continue
        sub = sub.sort_values(["pre_ev_r", "pre_pf", "pre_hit_tp1", "pre_tpd", "maxDD_usd"], ascending=[False, False, False, False, True], kind="mergesort")
        row = sub.iloc[0].to_dict()
        lines.append(
            f"- {scen}: BEST_EFFORT pre_epd={fmt(row.get('pre_epd'),4)}, pre_tpd={fmt(row.get('pre_tpd'),4)}, hit@TP1={fmt(row.get('pre_hit_tp1'),4)}, PF={fmt(row.get('pre_pf'),4)}, ev_r={fmt(row.get('pre_ev_r'),4)}, maxDD_usd={fmt(row.get('maxDD_usd'),2)}, ev_r_preos_p25={fmt(row.get('ev_r_preos_p25'),4)} | fail_reason={str(row.get('fail_reason',''))}"
        )
    lines.append("")

    lines.append("==============================")
    lines.append("SELECTION")
    lines.append("==============================")
    lines.append(f"strict_pass={bool(strict_pass)}")
    if final_row is None:
        lines.append("- selected: NA")
    else:
        lines.append(
            f"- scenario={str(final_row.get('scenario'))} | min_lot={fmt(final_row.get('min_lot'),2)} | tp1_over_cost_k={fmt(final_row.get('tp1_over_cost_k'),2)}"
        )
        lines.append(
            f"- entry={str(final_row.get('entry'))} | H={int(final_row.get('H',0))} | tp1_atr_mult={fmt(final_row.get('tp1_atr_mult'),2)} | sl_atr_mult={fmt(final_row.get('sl_atr_mult'),2)} | tp1_close_frac={fmt(final_row.get('tp1_close_frac'),2)} | tp2_mult={fmt(final_row.get('tp2_mult'),2)}"
        )
        lines.append(
            f"- risk_cap_usd={fmt(final_row.get('risk_cap_usd'),2)} | daily_stop_loss_usd={fmt(final_row.get('daily_stop_loss_usd'),2)} | max_parallel_same_dir={int(final_row.get('max_parallel_same_dir',0))} | tickets_per_signal={int(final_row.get('tickets_per_signal',1))} | cooldown_bars={int(final_row.get('cooldown_bars',0))} | q={fmt(final_row.get('q'),2)}"
        )
        lines.append(
            f"- run_status={str(execution_audit.get('run_status',''))} | dd_trigger_count={int(execution_audit.get('dd_trigger_count',0))} | dd_stop_skip={int(execution_audit.get('dd_stop_skip',0))}"
        )
    lines.append("")

    lines.append("==============================")
    lines.append("ROBUSTNESS_BREAKDOWN（pre-OS year/regime；每个候选）")
    lines.append("==============================")
    if candidates_df.empty:
        lines.append("- NA（candidates.csv 为空）")
    else:
        # keep the same order as candidates.csv (already score-sorted)
        for _, r in candidates_df.iterrows():
            cid = str(r.get("candidate_id", ""))
            lines.append(
                f"- candidate_id={cid} | scenario={str(r.get('scenario'))} | pre_tpd={fmt(r.get('pre_tpd'),4)} | hit@TP1={fmt(r.get('pre_hit_tp1'),4)} | PF={fmt(r.get('pre_pf'),4)} | ev_r={fmt(r.get('pre_ev_r'),4)} | ev_r_preos_p25={fmt(r.get('ev_r_preos_p25'),4)}"
            )
            bd = kept_breakdowns.get(str(cid)) or {}
            yt = bd.get("preos_year_table") or []
            rt = bd.get("preos_regime_adx3_table") or []
            lines.append("  YEAR_TABLE")
            if yt:
                for yy in yt:
                    lines.append(
                        f"  - year={int(yy.get('year',0))} | signals={int(yy.get('signals',0))} | tickets={int(yy.get('tickets',0))} | epd={fmt(yy.get('epd'),4)} | tpd={fmt(yy.get('tpd'),4)} | hit@TP1={fmt(yy.get('hit_tp1'),4)} | PF={fmt(yy.get('pf'),4)} | ev_r={fmt(yy.get('ev_r'),4)}"
                    )
            else:
                lines.append("  - (empty)")
            lines.append("  REGIME_ADX3_TABLE")
            if rt:
                for rr in rt:
                    lines.append(
                        f"  - {str(rr.get('adx_label','NA'))} | signals={int(rr.get('signals',0))} | tickets={int(rr.get('tickets',0))} | epd={fmt(rr.get('epd'),4)} | tpd={fmt(rr.get('tpd'),4)} | hit@TP1={fmt(rr.get('hit_tp1'),4)} | PF={fmt(rr.get('pf'),4)} | ev_r={fmt(rr.get('ev_r'),4)}"
                    )
            else:
                lines.append("  - (empty)")
    lines.append("")

    if not strict_pass:
        lines.append("==============================")
        lines.append("INFEASIBILITY_EVIDENCE（新搜索空间上界）")
        lines.append("==============================")
        if candidates_df.empty:
            lines.append("- candidates.csv 为空，无法提供统计证据。")
        else:
            dfc = candidates_df.copy()
            dfc["fail_reason"] = dfc.get("fail_reason", "").fillna("").astype(str)
            vc = dfc["fail_reason"].replace("", "pass").value_counts().to_dict()
            lines.append(f"- fail_reason_top10: {list(vc.items())[:10]}")
            for c in ("pre_epd", "pre_tpd", "pre_hit_tp1", "pre_pf", "pre_ev_r", "maxDD_usd", "os_epd"):
                if c in dfc.columns:
                    dfc[c] = pd.to_numeric(dfc[c], errors="coerce")
            try:
                epd_max = float(dfc["pre_epd"].max()) if "pre_epd" in dfc.columns else float("nan")
                tpd_max = float(dfc["pre_tpd"].max()) if "pre_tpd" in dfc.columns else float("nan")
                hit_max = float(dfc["pre_hit_tp1"].max())
                pf_max0 = float(dfc["pre_pf"].max())
                ev_max0 = float(dfc["pre_ev_r"].max())
                dd_min0 = float(dfc["maxDD_usd"].min())
                lines.append(
                    f"- global_bounds(preOS): tpd_max={fmt(tpd_max,4)}, epd_max={fmt(epd_max,4)}, hit@TP1_max={fmt(hit_max,4)}, PF_max={fmt(pf_max0,4)}, ev_r_max={fmt(ev_max0,4)}, maxDD_usd_min={fmt(dd_min0,2)}"
                )
            except Exception:
                lines.append("- global_bounds: NA（统计失败）")

            hit_gate = (dfc.get("pre_hit_tp1") >= 0.73) & (dfc.get("pre_tpd") >= 1.5) & (dfc.get("os_epd") > 0)
            sub = dfc[hit_gate].copy()
            if sub.empty:
                lines.append("- tightest_conflict_hint: hit@TP1 与 tpd/OS_epd 交集为空")
            else:
                pf_max = float(sub["pre_pf"].max()) if "pre_pf" in sub.columns else float("nan")
                ev_max = float(sub["pre_ev_r"].max()) if "pre_ev_r" in sub.columns else float("nan")
                dd_min = float(sub["maxDD_usd"].min()) if "maxDD_usd" in sub.columns else float("nan")
                lines.append(f"- 在 hit>=0.73 & tpd>=1.5 & OS_epd>0 区域: PF_max={fmt(pf_max,4)}, ev_r_max={fmt(ev_max,4)}, min(maxDD_usd)={fmt(dd_min,2)}")
    lines.append("")

    lines.append("==============================")
    lines.append("ARTIFACTS")
    lines.append("==============================")
    for name in (
        "manifest.json",
        "leakage_audit.json",
        "candidates.csv",
        "selected_config.json",
        "thresholds_walkforward.json",
        "filter_report.json",
        "regime_report.json",
        "backtest_mode4_trades.csv",
        "execution_audit.json",
        "lot_math_audit.json",
        paths.desktop_script_copy.name,
    ):
        lines.append(f"- {str(paths.artifacts_dir / name)}")
    lines.append(f"- report: {str(paths.report_path)}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0

    # =============================
    # Write report 010.txt
    # =============================
    # Load 009 for delta (read-only)
    ref009_txt = Path.home() / "Desktop" / "20260112" / "009.txt"
    ref009_lines = ref009_txt.read_text(encoding="utf-8").splitlines() if ref009_txt.exists() else []
    ref009_summary = {"pre_epd": None, "pre_tpd": None, "pre_hit_tp1": None, "pre_hit_tp2": None, "pre_pf": None, "pre_ev": None, "pre_dd": None}
    for ln in ref009_lines:
        if ln.strip().startswith("- preOS:"):
            # quick parse (best-effort)
            parts = ln.replace("- preOS:", "").split(",")
            for p in parts:
                if "epd=" in p:
                    ref009_summary["pre_epd"] = float(p.split("epd=")[1].strip())
                if "tpd=" in p:
                    ref009_summary["pre_tpd"] = float(p.split("tpd=")[1].strip())
                if "hit@TP1=" in p:
                    ref009_summary["pre_hit_tp1"] = float(p.split("hit@TP1=")[1].strip())
                if "hit@TP2=" in p:
                    ref009_summary["pre_hit_tp2"] = float(p.split("hit@TP2=")[1].strip())
                if "PF=" in p:
                    ref009_summary["pre_pf"] = float(p.split("PF=")[1].strip())
                if "ev_r=" in p:
                    ref009_summary["pre_ev"] = float(p.split("ev_r=")[1].strip())
                if "maxDD_usd=" in p:
                    ref009_summary["pre_dd"] = float(p.split("maxDD_usd=")[1].strip())

    # Final metrics (based on final_row run; strict_pass decides whether it satisfies hard constraints)
    pre_metrics: Dict[str, Any] = {}
    os_metrics: Dict[str, Any] = {}
    all_metrics: Dict[str, Any] = {}
    pre_dd_usd = float("nan")
    pre_dd_pct = float("nan")
    os_dd_usd = float("nan")
    os_dd_pct = float("nan")
    all_dd_usd = float("nan")
    all_dd_pct = float("nan")
    os_has_trades = False
    exec_a: Dict[str, Any] = {}
    if final_row is not None and (paths.artifacts_dir / "backtest_mode4_trades.csv").exists():
        tr = pd.read_csv(paths.artifacts_dir / "backtest_mode4_trades.csv")
        pre_tr = slice_segment(tr, start=time_cfg.preos_start_utc, end=time_cfg.preos_end_utc)
        os_tr = slice_segment(tr, start=time_cfg.os_start_utc, end=time_cfg.backtest_end_utc)
        all_tr = slice_segment(tr, start=time_cfg.backtest_start_utc, end=time_cfg.backtest_end_utc)
        pre_metrics = metrics_from_trades(time_cfg, mkt, trades=pre_tr, start_utc=time_cfg.preos_start_utc, end_utc=time_cfg.preos_end_utc)
        os_metrics = metrics_from_trades(time_cfg, mkt, trades=os_tr, start_utc=time_cfg.os_start_utc, end_utc=time_cfg.backtest_end_utc)
        all_metrics = metrics_from_trades(time_cfg, mkt, trades=all_tr, start_utc=time_cfg.backtest_start_utc, end_utc=time_cfg.backtest_end_utc)
        # execution audit for take_rate attribution
        try:
            exec_a = json.loads((paths.artifacts_dir / "execution_audit.json").read_text(encoding="utf-8"))
        except Exception:
            exec_a = {}
        # DDs are stored on candidate row (pre-OS selection uses pre-OS DD only)
        pre_dd_usd = float(final_row.get("maxDD_usd", float("nan")))
        pre_dd_pct = float(final_row.get("maxDD_pct", float("nan")))
        os_dd_usd = float(final_row.get("os_maxDD_usd", float("nan")))
        os_dd_pct = float(final_row.get("os_maxDD_pct", float("nan")))
        all_dd_usd = float(final_row.get("all_maxDD_usd", float("nan")))
        all_dd_pct = float(final_row.get("all_maxDD_pct", float("nan")))
        os_has_trades = bool(os_metrics.get("epd", 0.0) > 0)

    lines: List[str] = []
    lines.append("==============================")
    lines.append("HEADER")
    lines.append("==============================")
    lines.append(f"时间：{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"脚本：{str(paths.desktop_script_copy)}")
    lines.append(f"仓库：{str(Path(__file__).resolve().parent.parent)}")
    lines.append(f"数据：{str(data_path)}")
    lines.append(f"数据覆盖：{str(df0.index.min())} 至 {str(df0.index.max())}")
    lines.append(f"回测区间：{time_cfg.backtest_start_utc} 至 {min(to_utc_ts(time_cfg.backtest_end_utc), pd.to_datetime(df0.index.max(), utc=True))}")
    lines.append(f"pre-OS（选型）：{time_cfg.preos_start_utc} 至 {time_cfg.preos_end_utc}")
    lines.append(f"OS（仅验收）：{time_cfg.os_start_utc} 至 {min(to_utc_ts(time_cfg.backtest_end_utc), pd.to_datetime(df0.index.max(), utc=True))}")
    lines.append(f"purge/embargo：{cv_cfg.purge_bars}/{cv_cfg.embargo_bars} bars (M5)")
    lines.append(f"initial_capital_usd={fmt(mkt.initial_capital_usd,2)} | cost(roundtrip)={fmt(mkt.roundtrip_cost_price,3)} | slippage_buffer={fmt(mkt.slippage_buffer_price,3)}")
    lines.append("")
    lines.append("==============================")
    lines.append("A) 可行性预扫（take_rate结构修复）")
    lines.append("==============================")
    a1 = lot_audit_sl1
    amax = lot_audit_slmax
    lines.append(f"- raw_events_preOS: {a1.get('preos_events')}")
    lines.append(f"- 009口径(sl_atr_mult=1.00): atr_ref_p80={fmt(a1.get('atr_ref_p80'),4)} | sl_dist_p80={fmt(a1.get('sl_dist_p80'),4)}")
    ml_002 = a1.get("min_lot_branches", {}).get("0.02", {})
    ml_001 = a1.get("min_lot_branches", {}).get("0.01", {})
    lines.append(
        f"- min_lot=0.02 risk_usd P50/P80/P95(含cost+slip): {fmt(ml_002.get('risk_usd_p50'),2)}/{fmt(ml_002.get('risk_usd_p80'),2)}/{fmt(ml_002.get('risk_usd_p95'),2)}"
    )
    lines.append(
        f"- min_lot=0.01 risk_usd P50/P80/P95(含cost+slip): {fmt(ml_001.get('risk_usd_p50'),2)}/{fmt(ml_001.get('risk_usd_p80'),2)}/{fmt(ml_001.get('risk_usd_p95'),2)}"
    )
    lines.append(
        f"- worst-case(sl_atr_mult={float(max(esc.sl_atr_mult_grid)):.2f}) risk_usd_p80(min_lot=0.02): {fmt(amax.get('min_lot_branches', {}).get('0.02', {}).get('risk_usd_p80'),2)}"
    )
    lines.append(f"- auto max_risk_usd_per_trade_grid(用于防止min_lot被cap跳过): {list(risk_grid)}")
    lines.append("")
    lines.append("==============================")
    lines.append("A2) Execution（take_rate 归因）")
    lines.append("==============================")
    if isinstance(exec_a, dict) and ("raw_events_preos" in exec_a):
        audit_c = exec_a.get("audit_counters") or {}
        raw_pre = int(exec_a.get("raw_events_preos", 0))
        after_econ_pre = int(exec_a.get("events_after_econ_preos", 0))
        econ_pruned_pre = int(exec_a.get("econ_pruned_preos", 0))
        opened_pre = int(audit_c.get("trades_opened_preOS", 0))
        lines.append(f"- raw_events_epd_preOS≈{fmt(exec_a.get('raw_events_epd_preos'),4)} | take_rate_preOS≈{fmt(exec_a.get('take_rate_preos'),4)}")
        lines.append(f"- take_rate_preOS = opened_preOS/raw_events_preOS = {opened_pre}/{raw_pre}")
        if raw_pre > 0:
            lines.append(f"- econ_prune: pruned_preOS={econ_pruned_pre} ({fmt(econ_pruned_pre/raw_pre,4)}) | after_econ_preOS={after_econ_pre} ({fmt(after_econ_pre/raw_pre,4)})")
        lines.append(
            f"- skips_preOS: threshold={int(audit_c.get('skipped_threshold_preOS',0))}, over_risk_cap={int(exec_a.get('skipped_over_risk_cap',0))}, min_lot={int(exec_a.get('skipped_min_lot',0))}, dd_stop={int(exec_a.get('skipped_dd_stop',0))}"
        )
        lines.append(
            f"- skips_preOS: daily_stop={int(audit_c.get('skipped_daily_stop_preOS',0))}, max_parallel={int(audit_c.get('skipped_max_parallel_preOS',0))}, dir_conflict={int(audit_c.get('skipped_open_direction_conflict_preOS',0))}"
        )
        lines.append(f"- dd_stop_triggers={int(exec_a.get('dd_stop_triggers',0))} | equity_floor_hits={int(exec_a.get('equity_floor_hits',0))}")
    else:
        lines.append("- execution_audit 不可用（无最终回放或未生成 execution_audit.json）")
    lines.append("")
    lines.append("==============================")
    lines.append("AUDIT（防泄露）")
    lines.append("==============================")
    lines.append(f"- manifest: {str(paths.artifacts_dir / 'manifest.json')}")
    lines.append(f"- leakage_audit: {str(paths.artifacts_dir / 'leakage_audit.json')} (ok={bool(leak_ok)})")
    lines.append("")
    lines.append("==============================")
    lines.append("C) 过滤器（pre-OS OOF）")
    lines.append("==============================")
    try:
        fr = json.loads((paths.artifacts_dir / "filter_report.json").read_text(encoding="utf-8"))
    except Exception:
        fr = {}
    sides = fr.get("sides") if isinstance(fr, dict) else None
    if not isinstance(sides, dict) or not sides:
        lines.append("- filter_report 不可用（无最终候选或未生成）")
    else:
        for side in ("long", "short"):
            s = sides.get(side) or {}
            bs = (s.get("base_stats") or {})
            lgbm = bs.get("lgbm_oof") or {}
            lr = bs.get("logreg_oof") or {}
            calib = (s.get("calibration") or {})
            best = (calib.get("best") or {})
            lines.append(
                f"- {side}: LGBM(AUC={fmt(lgbm.get('auc'),4)}, Brier={fmt(lgbm.get('brier'),4)}) | "
                f"LogReg(AUC={fmt(lr.get('auc'),4)}, Brier={fmt(lr.get('brier'),4)}) | "
                f"calib_best={str(best.get('method','none'))}"
            )
            rt = s.get("regime_oof_table") or []
            if isinstance(rt, list) and rt:
                # stable ordering
                rt2 = sorted(rt, key=lambda r: (int(r.get("vol_regime", -1)), int(r.get("trend_regime", -1))))
                for rr in rt2:
                    lines.append(
                        f"  - regime(v{int(rr.get('vol_regime',-1))},t{int(rr.get('trend_regime',-1))}): "
                        f"n={int(rr.get('n',0))}, AUC={fmt(rr.get('auc'),4)}, Brier={fmt(rr.get('brier'),4)}"
                    )
    lines.append("")
    lines.append("==============================")
    lines.append("SEARCH_RESULT（pre-OS选型；OS仅OS_epd>0约束）")
    lines.append("==============================")
    if final_row is None:
        lines.append("未生成任何候选（candidates.csv 为空）。")
    else:
        if strict_pass:
            lines.append("SELECTED_CONFIG（PASS）：")
        else:
            lines.append("未找到满足全部硬约束的候选（strict_pass=False）。以下 best_effort 仅用于不可行性证据：")
        lines.append(
            f"- entry={final_row['entry']} | H={int(final_row['H'])} | tp1_atr_mult={fmt(final_row['tp1_atr_mult'],2)} | sl_atr_mult={fmt(final_row['sl_atr_mult'],2)}"
        )
        lines.append(
            f"- tp1_close_frac={fmt(final_row['tp1_close_frac'],2)} | tp2_mult={fmt(final_row['tp2_mult'],2)} | tp1_over_cost_k={fmt(final_row.get('tp1_over_cost_k'),2)}"
        )
        lines.append(
            f"- risk_cap_usd={fmt(final_row['risk_cap_usd'],2)} | daily_stop_loss_usd={fmt(final_row['daily_stop_loss_usd'],2)} | max_parallel_same_dir={int(final_row.get('max_parallel_same_dir',0))} | tickets_per_signal={int(final_row.get('tickets_per_signal',1))} | cooldown_bars={int(final_row.get('cooldown_bars',0))}"
        )
        lines.append(
            f"- q(threshold_quantile)={fmt(final_row['q'],2)} | OS_has_trades={bool(os_has_trades)} | strict_pass={bool(strict_pass)} | fail_reason={str(final_row.get('fail_reason',''))}"
        )
    lines.append("")
    lines.append("==============================")
    lines.append("FINAL_PRINT_BLOCK")
    lines.append("==============================")
    if final_row is None:
        lines.append("- preOS: NA")
        lines.append("- OS:   NA")
        lines.append("- All:  NA")
    else:
        lines.append(
            f"- preOS: epd={fmt(pre_metrics.get('epd'),4)}, tpd={fmt(pre_metrics.get('tpd'),4)}, hit@TP1={fmt(pre_metrics.get('hit_tp1'),4)}, hit@TP2={fmt(pre_metrics.get('hit_tp2'),4)}, PF={fmt(pre_metrics.get('pf'),4)}, ev_r={fmt(pre_metrics.get('ev_r'),4)}, maxDD_usd={fmt(pre_dd_usd,2)}, maxDD%={fmt(pre_dd_pct,2)}"
        )
        lines.append(
            f"- OS:   epd={fmt(os_metrics.get('epd'),4)}, tpd={fmt(os_metrics.get('tpd'),4)}, hit@TP1={fmt(os_metrics.get('hit_tp1'),4)}, hit@TP2={fmt(os_metrics.get('hit_tp2'),4)}, PF={fmt(os_metrics.get('pf'),4)}, ev_r={fmt(os_metrics.get('ev_r'),4)}, maxDD_usd={fmt(os_dd_usd,2)}, maxDD%={fmt(os_dd_pct,2)}"
        )
        lines.append(
            f"- All:  epd={fmt(all_metrics.get('epd'),4)}, tpd={fmt(all_metrics.get('tpd'),4)}, hit@TP1={fmt(all_metrics.get('hit_tp1'),4)}, hit@TP2={fmt(all_metrics.get('hit_tp2'),4)}, PF={fmt(all_metrics.get('pf'),4)}, ev_r={fmt(all_metrics.get('ev_r'),4)}, maxDD_usd={fmt(all_dd_usd,2)}, maxDD%={fmt(all_dd_pct,2)}"
        )
    lines.append("")
    lines.append("VS_009（delta，pre-OS）")
    if final_row is None or ref009_summary["pre_epd"] is None:
        lines.append("- 无法解析 009.txt 的 pre-OS 对照行，或本轮无可用回放。")
    else:
        lines.append(
            f"- 009_preOS: epd={fmt(ref009_summary['pre_epd'],4)}, tpd={fmt(ref009_summary['pre_tpd'],4)}, hit@TP1={fmt(ref009_summary['pre_hit_tp1'],4)}, hit@TP2={fmt(ref009_summary['pre_hit_tp2'],4)}, PF={fmt(ref009_summary['pre_pf'],4)}, ev_r={fmt(ref009_summary['pre_ev'],4)}, maxDD_usd={fmt(ref009_summary['pre_dd'],2)}"
        )
        lines.append(
            f"- 010_preOS: epd={fmt(pre_metrics.get('epd'),4)}, tpd={fmt(pre_metrics.get('tpd'),4)}, hit@TP1={fmt(pre_metrics.get('hit_tp1'),4)}, hit@TP2={fmt(pre_metrics.get('hit_tp2'),4)}, PF={fmt(pre_metrics.get('pf'),4)}, ev_r={fmt(pre_metrics.get('ev_r'),4)}, maxDD_usd={fmt(pre_dd_usd,2)}"
        )
        try:
            lines.append(
                f"- delta: epdΔ={fmt(float(pre_metrics.get('epd',0.0)) - float(ref009_summary['pre_epd']),4)}, "
                f"tpdΔ={fmt(float(pre_metrics.get('tpd',0.0)) - float(ref009_summary.get('pre_tpd') or 0.0),4)}, "
                f"hit1Δ={fmt(float(pre_metrics.get('hit_tp1',0.0)) - float(ref009_summary.get('pre_hit_tp1') or 0.0),4)}, "
                f"hit2Δ={fmt(float(pre_metrics.get('hit_tp2',0.0)) - float(ref009_summary.get('pre_hit_tp2') or 0.0),4)}, "
                f"PFΔ={fmt(float(pre_metrics.get('pf',0.0)) - float(ref009_summary.get('pre_pf') or 0.0),4)}, "
                f"evΔ={fmt(float(pre_metrics.get('ev_r',0.0)) - float(ref009_summary.get('pre_ev') or 0.0),4)}, "
                f"ddΔ={fmt(float(pre_dd_usd) - float(ref009_summary.get('pre_dd') or 0.0),2)}"
            )
        except Exception:
            lines.append("- delta: NA（解析失败）")
    lines.append("")
    if not strict_pass:
        lines.append("==============================")
        lines.append("INFEASIBILITY_EVIDENCE（若未PASS则必须给证据）")
        lines.append("==============================")
        if candidates_df.empty:
            lines.append("- candidates.csv 为空，无法提供统计证据。")
        else:
            dfc = candidates_df.copy()
            dfc["fail_reason"] = dfc.get("fail_reason", "").fillna("").astype(str)
            vc = dfc["fail_reason"].replace("", "pass").value_counts().to_dict()
            # top reasons
            top_items = list(vc.items())[:10]
            lines.append(f"- fail_reason_top10: {top_items}")

            # helper views (pre-OS constraints only; OS only uses os_epd>0 feasibility)
            def _num(s: pd.Series) -> pd.Series:
                return pd.to_numeric(s, errors="coerce")

            for c in ("pre_epd", "pre_hit_tp1", "pre_pf", "pre_ev_r", "maxDD_usd", "os_epd"):
                if c in dfc.columns:
                    dfc[c] = _num(dfc[c])

            # global bounds vs hard constraints (pre-OS only; OS only uses os_epd>0)
            try:
                epd_max = float(dfc["pre_epd"].max())
                hit_max = float(dfc["pre_hit_tp1"].max())
                pf_max0 = float(dfc["pre_pf"].max())
                ev_max0 = float(dfc["pre_ev_r"].max())
                dd_min0 = float(dfc["maxDD_usd"].min())
                lines.append(
                    f"- global_bounds(preOS): epd_max={fmt(epd_max,4)}, hit@TP1_max={fmt(hit_max,4)}, PF_max={fmt(pf_max0,4)}, ev_r_max={fmt(ev_max0,4)}, maxDD_usd_min={fmt(dd_min0,2)}"
                )
            except Exception:
                lines.append("- global_bounds: NA（统计失败）")
            try:
                n_total = int(len(dfc))
                cnt_os = int(np.sum((dfc["os_epd"] > 0).to_numpy(dtype=bool)))
                cnt_epd = int(np.sum((dfc["pre_epd"] >= 1.5).to_numpy(dtype=bool)))
                cnt_hit = int(np.sum((dfc["pre_hit_tp1"] >= 0.73).to_numpy(dtype=bool)))
                cnt_pf = int(np.sum((dfc["pre_pf"] >= 1.05).to_numpy(dtype=bool)))
                cnt_ev = int(np.sum((dfc["pre_ev_r"] >= 0.0).to_numpy(dtype=bool)))
                cnt_dd = int(np.sum((dfc["maxDD_usd"] <= 60.0).to_numpy(dtype=bool)))
                lines.append(
                    f"- satisfy_counts: OS_epd>0={cnt_os}/{n_total}, epd>=1.5={cnt_epd}, hit>=0.73={cnt_hit}, PF>=1.05={cnt_pf}, ev>=0={cnt_ev}, maxDD<=60={cnt_dd}"
                )
            except Exception:
                lines.append("- satisfy_counts: NA（统计失败）")

            hit_gate = (dfc.get("pre_hit_tp1") >= 0.73) & (dfc.get("pre_epd") >= 1.5) & (dfc.get("os_epd") > 0)
            sub = dfc[hit_gate].copy()
            if sub.empty:
                lines.append("- 证据: 无候选同时满足 hit@TP1>=0.73 & epd>=1.5 & OS_epd>0（命中/密度与结构约束冲突）。")
            else:
                pf_max = float(sub["pre_pf"].max()) if "pre_pf" in sub.columns else float("nan")
                ev_max = float(sub["pre_ev_r"].max()) if "pre_ev_r" in sub.columns else float("nan")
                dd_min = float(sub["maxDD_usd"].min()) if "maxDD_usd" in sub.columns else float("nan")
                lines.append(f"- 在 hit>=0.73 & epd>=1.5 & OS_epd>0 区域: PF_max={fmt(pf_max,4)}, ev_r_max={fmt(ev_max,4)}, min(maxDD_usd)={fmt(dd_min,2)}")
                if np.isfinite(pf_max) and pf_max < 1.05 - 1e-12:
                    lines.append("- 结论: PF 约束与 hit/epd 可行区冲突（成本/TP1/runner 结构仍偏亏）。")
                if np.isfinite(ev_max) and ev_max < 0.0 - 1e-12:
                    lines.append("- 结论: ev_r 约束与 hit/epd 可行区冲突（收益结构无法覆盖成本）。")
                if np.isfinite(dd_min) and dd_min > 60.0 + 1e-12:
                    lines.append("- 结论: maxDD_usd 约束与 hit/epd 可行区冲突（风控/结构导致回撤下界仍>60）。")

            pf_gate = (dfc.get("pre_pf") >= 1.05) & (dfc.get("pre_ev_r") >= 0.0) & (dfc.get("maxDD_usd") <= 60.0) & (dfc.get("os_epd") > 0)
            sub2 = dfc[pf_gate].copy()
            if sub2.empty:
                lines.append("- 证据: 无候选同时满足 PF>=1.05 & ev_r>=0 & maxDD<=60 & OS_epd>0（盈利/回撤硬约束过紧或结构不匹配）。")
            else:
                hit_max = float(sub2["pre_hit_tp1"].max()) if "pre_hit_tp1" in sub2.columns else float("nan")
                epd_max = float(sub2["pre_epd"].max()) if "pre_epd" in sub2.columns else float("nan")
                lines.append(f"- 在 PF/EV/DD/OS 可行区: hit@TP1_max={fmt(hit_max,4)}, epd_max={fmt(epd_max,4)}")
                if np.isfinite(hit_max) and hit_max < 0.73 - 1e-12:
                    lines.append("- 结论: hit@TP1 约束与盈利/回撤可行区冲突（提高命中会牺牲 EV/PF 或反之）。")
                if np.isfinite(epd_max) and epd_max < 1.5 - 1e-12:
                    lines.append("- 结论: epd 约束与盈利/回撤可行区冲突（过滤强度导致密度不足）。")
        lines.append("")
    lines.append("==============================")
    lines.append("ARTIFACTS")
    lines.append("==============================")
    for name in (
        "manifest.json",
        "leakage_audit.json",
        "candidates.csv",
        "selected_config.json",
        "thresholds_walkforward.json",
        "filter_report.json",
        "regime_report.json",
        "backtest_mode4_trades.csv",
        "execution_audit.json",
        "lot_math_audit.json",
        script_path.name,
    ):
        lines.append(f"- {str(paths.artifacts_dir / name)}")
    lines.append(f"- report: {str(paths.report_path)}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 0


def run_round_019_mode4() -> int:
    # Paths
    base_candidates = ["D:/projectmt5", "/mnt/d/projectmt5", "D:\\projectmt5"]
    base: Optional[Path] = None
    for p in base_candidates:
        pp = Path(p)
        if pp.exists():
            base = pp
            break
    if base is None:
        raise FileNotFoundError(f"BASE not found in candidates: {base_candidates}")

    out_dir = base / "20260112"
    artifacts_dir = out_dir / "019_artifacts"
    report_path = out_dir / "019.txt"
    desktop_script_copy = artifacts_dir / "20260112_019_Mode4_TP2_ConfidenceMax.py"
    paths = Paths(out_dir=out_dir, artifacts_dir=artifacts_dir, report_path=report_path, desktop_script_copy=desktop_script_copy)
    ensure_dir(paths.out_dir)
    ensure_dir(paths.artifacts_dir)

    # Repo auto-locate (Win/WSL compatible; per spec)
    repo_root: Optional[Path] = None
    for cand in (base / "trend_project", Path.home() / "trend_project"):
        if cand.exists():
            repo_root = cand
            break
    if repo_root is None:
        try:
            for p in base.rglob("trend_project/.git"):
                repo_root = p.parent
                break
        except Exception:
            repo_root = None
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    # 016 frozen config (non-TP2 fields must match)
    ref_cfg_path = out_dir / "016_artifacts" / "selected_config.json"
    if not ref_cfg_path.exists():
        raise FileNotFoundError(f"missing 016 selected_config: {ref_cfg_path}")
    ref_cfg = json.loads(ref_cfg_path.read_text(encoding="utf-8"))
    ref_metrics = {
        "pre_epd": float(ref_cfg.get("pre_epd", float("nan"))),
        "pre_hit_tp1": float(ref_cfg.get("pre_hit_tp1", float("nan"))),
        "pre_ev_r": float(ref_cfg.get("pre_ev_r", float("nan"))),
        "pre_maxdd_usd": float(ref_cfg.get("pre_maxdd_usd", float("nan"))),
        "os_epd": float(ref_cfg.get("os_epd", float("nan"))),
        "posterior_tp2": float(ref_cfg.get("posterior_tp2", 0.0)),
    }

    # 016 thresholds (auto-locate; used for freeze audit only)
    thresholds_016_path: Optional[Path] = None
    thresholds_candidates = [
        out_dir / "016_artifacts" / "thresholds_walkforward.json",
        base / "20260112_mode12" / "016_artifacts" / "thresholds_walkforward.json",
    ]
    for cand in thresholds_candidates:
        if cand.exists():
            thresholds_016_path = cand
            break
    if thresholds_016_path is None:
        try:
            for p in base.rglob("016_artifacts/thresholds_walkforward.json"):
                thresholds_016_path = p
                break
        except Exception:
            thresholds_016_path = None
    thresholds_016_sha256 = sha256_file(thresholds_016_path) if thresholds_016_path and thresholds_016_path.exists() else "NA"

    # Configs
    time_cfg = TimeConfig()
    mkt = MarketConfig()
    base_sig = Mode4SignalConfig()
    sig_search = SignalSearchConfig()
    cv_cfg = CVConfig()
    thr_cfg = ThresholdConfig()
    esc = ExitSearchConfig()
    mdl_cfg = ModelConfig()
    risk_cfg = RiskConfig(
        max_risk_usd_per_trade_grid=(2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0),
        daily_stop_loss_usd_grid=(4.0, 5.0, 6.0, 8.0, 10.0, 12.0),
        dd_trigger_usd_grid=(25.0, 30.0, 35.0, 40.0, 45.0, 60.0),
        risk_scale_min_grid=(0.05, 0.10, 0.15),
        equity_floor_usd=float(mkt.initial_capital_usd) - 45.0,
    )

    # Data (auto locate; prefer BASE/42swam for Win/WSL portability)
    data_root_candidates = [
        base / "42swam",
        Path.home() / "Desktop" / "42swam",
    ]
    data_root = next((p for p in data_root_candidates if p.exists()), None)
    if data_root is None:
        raise FileNotFoundError(f"data root not found in candidates: {[str(p) for p in data_root_candidates]}")
    data_path = locate_xauusd_m5(Path(data_root))
    df0 = pd.read_csv(data_path)
    if "datetime" not in df0.columns:
        raise ValueError("数据缺少 datetime 列")
    df0["datetime"] = pd.to_datetime(df0["datetime"], utc=True, errors="coerce")
    df0 = df0.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates(subset=["datetime"])
    df0 = df0.set_index("datetime")
    df0 = df0.rename(columns={c: c.lower() for c in df0.columns})
    need_cols = {"open", "high", "low", "close"}
    if not need_cols.issubset(set(df0.columns)):
        raise ValueError(f"数据缺少列: {sorted(list(need_cols - set(df0.columns)))}")
    df0 = df0.loc[(df0.index >= to_utc_ts(time_cfg.start_utc)) & (df0.index <= to_utc_ts(time_cfg.end_utc))].copy()

    bt0 = to_utc_ts(time_cfg.backtest_start_utc)
    bt1 = min(to_utc_ts(time_cfg.backtest_end_utc), pd.to_datetime(df0.index.max(), utc=True))
    pre0 = to_utc_ts(time_cfg.preos_start_utc)
    pre1 = to_utc_ts(time_cfg.preos_end_utc)
    os0 = to_utc_ts(time_cfg.os_start_utc)

    days_pre = max(1.0, float((pre1 - pre0).total_seconds() / 86400.0))
    days_os = max(1.0, float((bt1 - os0).total_seconds() / 86400.0))
    days_all = max(1.0, float((bt1 - bt0).total_seconds() / 86400.0))

    r0_start = to_utc_ts("2015-01-01")
    r0_end = to_utc_ts("2017-12-31 23:59:59")

    price_scale = float(np.nanmedian(df0["close"].to_numpy(dtype=float)))
    if not np.isfinite(price_scale) or price_scale <= 0:
        price_scale = float(df0["close"].iloc[-1])

    # Precompute by zero_eps
    ind_cache: Dict[float, Dict[str, Any]] = {}
    ctx_cache: Dict[float, Dict[str, np.ndarray]] = {}
    regime_cache: Dict[float, Dict[str, np.ndarray]] = {}
    for eps_mult in sig_search.zero_eps_grid:
        zero_eps = float(eps_mult) * float(price_scale)
        ind = precompute_indicators(df0, zero_eps=float(max(1e-12, zero_eps)))
        ctx = compute_feature_context(df0, ind)
        regimes = build_regimes_2x2(df0, ctx, window_bars=252)
        ind_cache[float(eps_mult)] = ind
        ctx_cache[float(eps_mult)] = ctx
        regime_cache[float(eps_mult)] = regimes

    # Leakage audit (entry-time features)
    leak_gate = leakage_audit_by_truncation(seed=cv_cfg.seed, df_full=df0, feature_cols=list(FEATURE_COLS), sample_n=10)
    leak_gate_ok = bool(leak_gate.get("ok", False)) and int(leak_gate.get("failures_n", 0)) == 0
    if not leak_gate_ok:
        raise RuntimeError(f"leakage_audit 失败：failures_n={int(leak_gate.get('failures_n', 999))}（必须为0）")

    # Manifest
    script_path = Path(__file__).resolve()
    manifest = {
        "version": "20260112_018",
        "generated_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "paths": {
            "data_path": str(data_path),
            "script_path": str(script_path),
            "repo_root": str(repo_root),
            "out_dir": str(paths.out_dir),
            "artifacts_dir": str(paths.artifacts_dir),
        },
        "frozen_016": {
            "selected_config_path": str(ref_cfg_path),
            "selected_config_sha256": sha256_file(ref_cfg_path),
            "thresholds_path": str(thresholds_016_path) if thresholds_016_path is not None else "NA",
            "thresholds_sha256": str(thresholds_016_sha256),
        },
        "data_file": {
            "sha256": sha256_file(data_path),
            "bytes": int(data_path.stat().st_size),
            "mtime_utc": pd.Timestamp(data_path.stat().st_mtime, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
            "rows": int(len(df0)),
            "start": str(df0.index.min()),
            "end": str(df0.index.max()),
        },
        "params": {
            "purge_bars": int(cv_cfg.purge_bars),
            "embargo_bars": int(cv_cfg.embargo_bars),
            "initial_capital_usd": float(mkt.initial_capital_usd),
            "roundtrip_cost_price": float(mkt.roundtrip_cost_price),
            "slippage_buffer_price": float(mkt.slippage_buffer_price),
            "signal_grid": {
                "entry_delay": list(sig_search.entry_delay_grid),
                "confirm_window": list(sig_search.confirm_window_grid),
                "fast_abs_ratio": list(sig_search.fast_abs_ratio_grid),
                "zero_eps_mult": list(sig_search.zero_eps_grid),
            },
            "price_scale": float(price_scale),
            "risk_grid": {
                "risk_cap_usd": list(risk_cfg.max_risk_usd_per_trade_grid),
                "daily_stop_loss_usd": list(risk_cfg.daily_stop_loss_usd_grid),
                "dd_trigger_usd_grid": list(risk_cfg.dd_trigger_usd_grid),
                "dd_stop_cooldown_bars_grid": list(risk_cfg.dd_stop_cooldown_bars_grid),
                "risk_scale_min_grid": list(risk_cfg.risk_scale_min_grid),
            },
            "exit_grid": {
                "H": list(esc.H_grid),
                "H2": list(esc.H2_grid),
                "tp1_q": list(esc.tp1_q_grid),
                "sl_q": list(esc.sl_q_grid),
                "tp2_q": list(esc.tp2_q_grid),
                "tp1_over_cost_k": list(esc.tp1_over_cost_k_grid),
                "tp2_n1": list(esc.tp2_n1_grid),
                "tp2_n2": list(esc.tp2_n2_grid),
            },
        },
        "env": {"python": sys.version.split()[0]},
    }
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
    except Exception:
        git_hash = "NA"
    manifest["git_hash"] = str(git_hash)
    manifest["outputs"] = {
        "report": str(paths.report_path),
        "artifacts": [
            str(paths.artifacts_dir / "manifest.json"),
            str(paths.artifacts_dir / "leakage_audit.json"),
            str(paths.artifacts_dir / "tp2_policy.json"),
            str(paths.artifacts_dir / "tp2_candidates.csv"),
            str(paths.artifacts_dir / "tp2_bucket_stats.csv"),
            str(paths.artifacts_dir / "tp2_calibration.json"),
            str(paths.artifacts_dir / "backtest_mode4_trades.csv"),
            str(paths.artifacts_dir / "execution_audit.json"),
            str(paths.artifacts_dir / "lot_math_audit.json"),
            str(paths.artifacts_dir / "selected_config.json"),
            str(paths.artifacts_dir / "candidates.csv"),
            str(paths.artifacts_dir / script_path.name),
        ],
    }
    write_json(paths.artifacts_dir / "manifest.json", manifest)

    # Helpers
    def _epd(ts: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[int, float]:
        if ts.empty:
            return 0, 0.0
        t = pd.to_datetime(ts, utc=True, errors="coerce")
        m = (t >= start) & (t <= end)
        n = int(np.sum(m))
        days = max(1.0, float((end - start).total_seconds() / 86400.0))
        return n, float(n / days)

    def _prepare_X(df: pd.DataFrame, cols: Sequence[str], med: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        X = df.loc[:, list(cols)].to_numpy(dtype=float)
        if med is None:
            med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        X = np.where(np.isfinite(X), X, med)
        return X, med

    def _calibrated_predict(
        *,
        estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        entry_i: np.ndarray,
        exit_i: np.ndarray,
    ) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
        from sklearn.calibration import CalibratedClassifierCV

        y = np.asarray(y, dtype=int)
        gap = int(cv_cfg.purge_bars + cv_cfg.embargo_bars)
        cv = PurgedTimeSeriesSplit(n_splits=max(3, int(cv_cfg.calib_cv_splits)), entry_i=entry_i, exit_i=exit_i, gap=gap)
        cal = CalibratedClassifierCV(estimator, method="sigmoid", cv=cv)
        cal.fit(X, y)
        p = cal.predict_proba(X)[:, 1].astype(float)
        meta = {"method": "sigmoid", "folds": int(cv.n_splits), "n": int(len(y))}
        return p, cal, meta

    def _make_lgbm_classifier(y: np.ndarray) -> Any:
        import lightgbm as lgb

        y = np.asarray(y, dtype=int)
        pos = int(np.sum(y == 1))
        neg = int(np.sum(y == 0))
        spw = float(neg / max(1, pos))
        return lgb.LGBMClassifier(
            **{
                **mdl_cfg.lgbm_base_params,
                "num_leaves": int(mdl_cfg.num_leaves_grid[0]),
                "max_depth": int(mdl_cfg.max_depth_grid[0]),
                "min_data_in_leaf": int(mdl_cfg.min_data_in_leaf_grid[0]),
                "scale_pos_weight": float(spw),
            }
        )

    def _make_logreg_classifier() -> Any:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=int(cv_cfg.seed),
        )

    def _make_rf_classifier() -> Any:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=120,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=int(cv_cfg.seed),
            n_jobs=1,
        )

    def _make_xgb_classifier(y: np.ndarray) -> Optional[Any]:
        xgb = _maybe_import_xgb()
        if xgb is None:
            return None
        y = np.asarray(y, dtype=int)
        pos = int(np.sum(y == 1))
        neg = int(np.sum(y == 0))
        spw = float(neg / max(1, pos))
        return xgb.XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=float(spw),
            random_state=int(cv_cfg.seed),
            n_jobs=1,
        )

    def _big_loss_labels(ds_side: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        if ds_side.empty:
            return np.zeros(0, dtype=int), {"base_rate": float("nan")}
        tp1_hit = pd.to_numeric(ds_side.get("tp1_hit"), errors="coerce").fillna(0).astype(int).to_numpy()
        fib_hit = pd.to_numeric(ds_side.get("tp1_fib10_hit"), errors="coerce").fillna(0).astype(int).to_numpy()
        exit_type = ds_side.get("exit_type")
        exit_type = exit_type.astype(str).to_numpy() if exit_type is not None else np.full(int(len(ds_side)), "NA")
        failure_base = (tp1_hit == 0) & (fib_hit == 0) & (exit_type == "SL")

        post1_dn = pd.to_numeric(ds_side.get("path_post1_max_down_r"), errors="coerce").to_numpy(dtype=float)
        post2_dn = pd.to_numeric(ds_side.get("path_post2_max_down_r"), errors="coerce").to_numpy(dtype=float)
        thr1 = float("nan")
        thr2 = float("nan")
        if int(np.sum(failure_base)) >= 50:
            try:
                thr1 = float(np.nanquantile(post1_dn[failure_base], 0.75))
                thr2 = float(np.nanquantile(post2_dn[failure_base], 0.75))
            except Exception:
                thr1 = float("nan")
                thr2 = float("nan")
        if np.isfinite(thr1) and np.isfinite(thr2):
            failure_path = failure_base & ((post1_dn >= float(thr1)) | (post2_dn >= float(thr2)))
        else:
            failure_path = failure_base
        base_rate = float(np.mean(failure_base)) if failure_base.size else float("nan")
        path_rate = float(np.mean(failure_path)) if failure_path.size else float("nan")
        return failure_path.astype(int), {"base_rate": base_rate, "path_rate": path_rate, "thr_post1": float(thr1), "thr_post2": float(thr2)}

    def _apply_failure_path(ds_side: pd.DataFrame, *, thr_post1: float, thr_post2: float) -> np.ndarray:
        if ds_side.empty:
            return np.zeros(0, dtype=int)
        tp1_hit = pd.to_numeric(ds_side.get("tp1_hit"), errors="coerce").fillna(0).astype(int).to_numpy()
        fib_hit = pd.to_numeric(ds_side.get("tp1_fib10_hit"), errors="coerce").fillna(0).astype(int).to_numpy()
        exit_type = ds_side.get("exit_type")
        exit_type = exit_type.astype(str).to_numpy() if exit_type is not None else np.full(int(len(ds_side)), "NA")
        failure_base = (tp1_hit == 0) & (fib_hit == 0) & (exit_type == "SL")
        post1_dn = pd.to_numeric(ds_side.get("path_post1_max_down_r"), errors="coerce").to_numpy(dtype=float)
        post2_dn = pd.to_numeric(ds_side.get("path_post2_max_down_r"), errors="coerce").to_numpy(dtype=float)
        if np.isfinite(thr_post1) and np.isfinite(thr_post2):
            failure_path = failure_base & ((post1_dn >= float(thr_post1)) | (post2_dn >= float(thr_post2)))
        else:
            failure_path = failure_base
        return failure_path.astype(int)

    def _big_loss_feature_stats(ds_side: pd.DataFrame, y: np.ndarray) -> List[Dict[str, Any]]:
        return _feature_stats_summary(ds_side, y, feature_cols=list(GATE_FEATURE_COLS), seed=int(cv_cfg.seed))

    def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size == 0 or b.size == 0:
            return float("nan")
        a = np.sort(a)
        b = np.sort(b)
        data = np.concatenate([a, b])
        cdf_a = np.searchsorted(a, data, side="right") / max(1, a.size)
        cdf_b = np.searchsorted(b, data, side="right") / max(1, b.size)
        return float(np.nanmax(np.abs(cdf_a - cdf_b)))

    def _feature_stats_summary(
        ds: pd.DataFrame,
        y: np.ndarray,
        *,
        feature_cols: Sequence[str],
        seed: int,
    ) -> List[Dict[str, Any]]:
        if ds.empty:
            return []
        y = np.asarray(y, dtype=int)
        rows: List[Dict[str, Any]] = []
        cols = [c for c in feature_cols if c in ds.columns]
        if len(cols) == 0:
            return []
        X_raw = ds.loc[:, cols]
        y_pos = (y == 1)
        y_neg = (y == 0)

        mi = np.full(int(len(cols)), float("nan"), dtype=float)
        perm = np.full(int(len(cols)), float("nan"), dtype=float)
        try:
            from sklearn.feature_selection import mutual_info_classif

            X_mi, _med = _prepare_X(ds, cols)
            if int(np.unique(y).size) >= 2 and X_mi.size > 0:
                mi = mutual_info_classif(X_mi, y, discrete_features=False, random_state=int(seed))
        except Exception:
            pass

        try:
            from sklearn.inspection import permutation_importance

            X_perm, _med = _prepare_X(ds, cols)
            if int(np.unique(y).size) >= 2 and int(len(ds)) >= 300:
                lr = fit_logreg_full(cv_cfg, X=X_perm, y=y)
                perm_res = permutation_importance(
                    lr,
                    X_perm,
                    y,
                    n_repeats=3,
                    random_state=int(seed),
                    scoring="roc_auc",
                )
                perm = perm_res.importances_mean.astype(float)
        except Exception:
            pass

        for i, f in enumerate(cols):
            a = pd.to_numeric(X_raw.loc[y_pos, f], errors="coerce")
            b = pd.to_numeric(X_raw.loc[y_neg, f], errors="coerce")
            if a.empty or b.empty:
                continue
            rows.append(
                {
                    "feature": str(f),
                    "pos_mean": float(a.mean()),
                    "pos_std": float(a.std(ddof=0)),
                    "neg_mean": float(b.mean()),
                    "neg_std": float(b.std(ddof=0)),
                    "ks": _ks_statistic(a.to_numpy(dtype=float), b.to_numpy(dtype=float)),
                    "mi": float(mi[int(i)]) if int(i) < len(mi) and np.isfinite(mi[int(i)]) else float("nan"),
                    "perm_importance": float(perm[int(i)]) if int(i) < len(perm) and np.isfinite(perm[int(i)]) else float("nan"),
                    "coverage_pos": float(np.mean(np.isfinite(a))) if len(a) else float("nan"),
                }
            )
        return rows

    def _regime_threshold_stats(ds_pre_side: pd.DataFrame, gate_mask: np.ndarray) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
        stats: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        if ds_pre_side.empty or gate_mask.size == 0:
            return stats
        for key, g in ds_pre_side.groupby(["vol_regime", "trend_regime", "internal_regime"]):
            idx = g.index.to_numpy(dtype=int)
            mask = gate_mask[idx]
            n_pass = int(np.sum(mask))
            epd = float(n_pass / max(1.0, days_pre))
            if n_pass == 0:
                stats[tuple(int(x) for x in key)] = {"n": 0, "epd": 0.0, "hit_tp1": float("nan"), "pf": float("nan"), "dd_trade_count": 0}
                continue
            sub = g.loc[mask]
            hit_tp1 = float(np.mean(pd.to_numeric(sub.get("tp1_hit"), errors="coerce").fillna(0).astype(int)))
            net_r = pd.to_numeric(sub.get("net_r"), errors="coerce").to_numpy(dtype=float)
            pos = net_r[np.isfinite(net_r) & (net_r > 0)]
            neg = net_r[np.isfinite(net_r) & (net_r < 0)]
            pf = float(np.sum(pos) / max(1e-12, abs(np.sum(neg)))) if pos.size and neg.size else float("nan")
            exit_type = sub.get("exit_type")
            if exit_type is None:
                dd_trade_count = 0
            else:
                dd_trade_count = int(np.sum(exit_type.astype(str) == "SL"))
            stats[tuple(int(x) for x in key)] = {
                "n": int(n_pass),
                "epd": float(epd),
                "hit_tp1": float(hit_tp1),
                "pf": float(pf),
                "dd_trade_count": int(dd_trade_count),
            }
        return stats

    def _path_feature_stats(ds_pre: pd.DataFrame) -> Dict[str, Any]:
        if ds_pre.empty:
            return {"counts": {}, "ks_top": [], "prob_top": [], "mean_std": {}}
        tp1_hit = pd.to_numeric(ds_pre.get("tp1_hit"), errors="coerce").fillna(0).astype(int).to_numpy()
        fib_hit = pd.to_numeric(ds_pre.get("tp1_fib10_hit"), errors="coerce").fillna(0).astype(int).to_numpy()
        exit_type = ds_pre.get("exit_type")
        exit_type = exit_type.astype(str).to_numpy() if exit_type is not None else np.full(int(len(ds_pre)), "NA")
        success_mask = tp1_hit == 1
        fib_mask = (tp1_hit == 0) & (fib_hit == 1)
        failure_mask = (tp1_hit == 0) & (fib_hit == 0) & (exit_type == "SL")

        counts = {
            "success_tp1": int(np.sum(success_mask)),
            "success_fib10": int(np.sum(fib_mask)),
            "failure_sl": int(np.sum(failure_mask)),
            "total": int(len(ds_pre)),
        }

        mean_std: Dict[str, Dict[str, Any]] = {}
        ks_rows: List[Dict[str, Any]] = []
        prob_rows: List[Dict[str, Any]] = []
        for f in PATH_FEATURE_COLS:
            if f not in ds_pre.columns:
                continue
            vals = pd.to_numeric(ds_pre[f], errors="coerce").to_numpy(dtype=float)
            a = vals[success_mask]
            b = vals[failure_mask]
            if a.size and b.size:
                ks = _ks_statistic(a, b)
                ks_rows.append({"feature": str(f), "ks": float(ks)})
            if np.isfinite(vals).sum() >= 200:
                try:
                    q1 = float(np.nanquantile(vals, 0.25))
                    q3 = float(np.nanquantile(vals, 0.75))
                except Exception:
                    q1 = float("nan")
                    q3 = float("nan")
                if np.isfinite(q1) and np.isfinite(q3):
                    hi = vals >= q3
                    lo = vals <= q1
                    p_hi = float(np.mean(success_mask[hi])) if np.any(hi) else float("nan")
                    p_lo = float(np.mean(success_mask[lo])) if np.any(lo) else float("nan")
                    prob_rows.append({"feature": str(f), "p_hi": float(p_hi), "p_lo": float(p_lo), "delta": float(p_hi - p_lo)})
            try:
                mean_std[str(f)] = {
                    "success_tp1": {"mean": float(np.nanmean(vals[success_mask])), "std": float(np.nanstd(vals[success_mask]))},
                    "success_fib10": {"mean": float(np.nanmean(vals[fib_mask])), "std": float(np.nanstd(vals[fib_mask]))},
                    "failure_sl": {"mean": float(np.nanmean(vals[failure_mask])), "std": float(np.nanstd(vals[failure_mask]))},
                }
            except Exception:
                pass

        ks_top = sorted(ks_rows, key=lambda x: float(x.get("ks", 0.0)), reverse=True)[:8]
        prob_top = sorted(prob_rows, key=lambda x: float(x.get("delta", 0.0)), reverse=True)[:8]
        return {"counts": counts, "ks_top": ks_top, "prob_top": prob_top, "mean_std": mean_std}

    def _select_tp1_threshold(
        *,
        p_tp1: np.ndarray,
        y_tp1: np.ndarray,
        min_take_rate: float,
        target_p: float = 0.70,
        min_posterior: float = 0.70,
        q_grid: Optional[Sequence[float]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        p = np.asarray(p_tp1, dtype=float)
        y = np.asarray(y_tp1, dtype=int)
        best_thr = 0.5
        best_meta: Dict[str, Any] = {"ok": False, "reason": "no_candidate"}
        best_take = -1.0
        p_ok = p[np.isfinite(p)]
        q_vals = list(q_grid) if q_grid else list(np.linspace(0.30, 0.90, 13))
        for q in q_vals:
            if p_ok.size == 0:
                continue
            thr = float(np.quantile(p_ok, float(q)))
            mask = p >= float(thr)
            take_rate = float(np.mean(mask)) if mask.size else 0.0
            if take_rate < float(min_take_rate):
                continue
            n = int(np.sum(mask))
            k = int(np.sum(y[mask])) if n > 0 else 0
            post = float(beta_posterior_prob_ge(k, n, float(target_p))) if n > 0 else float("nan")
            if np.isfinite(post) and post >= float(min_posterior) - 1e-12 and take_rate >= best_take:
                best_thr = float(thr)
                best_take = float(take_rate)
                best_meta = {"ok": True, "take_rate": float(take_rate), "n": int(n), "k": int(k), "posterior": float(post), "q": float(q)}
        if not best_meta.get("ok"):
            # fallback: pick threshold with max posterior subject to min_take_rate
            best_post = -1.0
            for q in q_vals:
                if p_ok.size == 0:
                    continue
                thr = float(np.quantile(p_ok, float(q)))
                mask = p >= float(thr)
                take_rate = float(np.mean(mask)) if mask.size else 0.0
                if take_rate < float(min_take_rate):
                    continue
                n = int(np.sum(mask))
                k = int(np.sum(y[mask])) if n > 0 else 0
                post = float(beta_posterior_prob_ge(k, n, float(target_p))) if n > 0 else float("nan")
                if np.isfinite(post) and post > best_post:
                    best_post = float(post)
                    best_thr = float(thr)
                    best_meta = {"ok": False, "take_rate": float(take_rate), "n": int(n), "k": int(k), "posterior": float(post), "q": float(q)}
        return float(best_thr), best_meta

    def _apply_regime_thresholds(
        ds_side: pd.DataFrame,
        *,
        p: np.ndarray,
        thr_default: float,
        thr_regime: Dict[Tuple[int, int, int], float],
        accept_if_leq: bool,
    ) -> np.ndarray:
        out = np.zeros(int(len(ds_side)), dtype=bool)
        for i, (_, r) in enumerate(ds_side.iterrows()):
            key = regime_key_from_row(r)
            thr = float(thr_regime.get(tuple(key), float(thr_default)))
            pv = float(p[int(i)]) if int(i) < len(p) else float("nan")
            if not np.isfinite(pv):
                out[i] = False
            else:
                if bool(accept_if_leq):
                    out[i] = bool(pv <= thr + 1e-12)
                else:
                    out[i] = bool(pv >= thr - 1e-12)
        return out

    # Raw signal density scan
    raw_rows: List[Dict[str, Any]] = []
    for eps_mult in sig_search.zero_eps_grid:
        ind = ind_cache.get(float(eps_mult))
        if ind is None:
            continue
        for entry_delay in sig_search.entry_delay_grid:
            for confirm_window in sig_search.confirm_window_grid:
                for r in sig_search.fast_abs_ratio_grid:
                    sig_cfg = dataclasses.replace(
                        base_sig,
                        entry_delay=int(entry_delay),
                        confirm_window=int(confirm_window),
                        fast_abs_ratio=float(r),
                        zero_eps_mult=float(eps_mult),
                    )
                    ev_raw = generate_mode4_events(sig_cfg, df=df0, ind=ind)
                    if ev_raw.empty:
                        continue
                    n_pre, epd_pre = _epd(ev_raw["_entry_ts"], pre0, pre1)
                    n_os, epd_os = _epd(ev_raw["_entry_ts"], os0, bt1)
                    n_all = int(len(ev_raw))
                    epd_all = float(n_all / max(1.0, days_all))
                    raw_rows.append(
                        {
                            "entry_delay": int(entry_delay),
                            "confirm_window": int(confirm_window),
                            "fast_abs_ratio": float(r),
                            "zero_eps_mult": float(eps_mult),
                            "pre_raw_n": int(n_pre),
                            "pre_raw_epd": float(epd_pre),
                            "os_raw_n": int(n_os),
                            "os_raw_epd": float(epd_os),
                            "all_raw_n": int(n_all),
                            "all_raw_epd": float(epd_all),
                        }
                    )

    if not raw_rows:
        raise RuntimeError("raw signal 扫描为空，请检查信号逻辑或数据")

    raw_df = pd.DataFrame(raw_rows).sort_values("pre_raw_epd", ascending=False).reset_index(drop=True)
    raw_df.to_csv(paths.artifacts_dir / "raw_signal_scan.csv", index=False)

    raw_ok = raw_df[raw_df["pre_raw_epd"] >= 0.8].copy()
    if raw_ok.empty:
        sig_pick = raw_df.head(1).copy()
    else:
        sig_pick = raw_ok.head(1).copy()

    # Successive halving contexts
    def _build_ctx(
        *,
        bt_start: pd.Timestamp,
        bt_end: pd.Timestamp,
        pre_start: pd.Timestamp,
        pre_end: pd.Timestamp,
        os_start: pd.Timestamp,
    ) -> FastSimCtx:
        days_pre_ctx = max(1.0, float((pre_end - pre_start).total_seconds() / 86400.0))
        days_os_ctx = max(1.0, float((bt_end - os_start).total_seconds() / 86400.0))
        days_all_ctx = max(1.0, float((bt_end - bt_start).total_seconds() / 86400.0))
        return FastSimCtx(
            idx_ts_ns=_dt_index_to_ns_utc(df0.index),
            year_by_bar=df0.index.year.to_numpy(dtype=np.int16),
            bt_start_ns=int(bt_start.value),
            bt_end_ns=int(bt_end.value),
            pre_start_ns=int(pre_start.value),
            pre_end_ns=int(pre_end.value),
            os_start_ns=int(os_start.value),
            days_pre=float(days_pre_ctx),
            days_os=float(days_os_ctx),
            days_all=float(days_all_ctx),
        )

    r0_start = to_utc_ts("2015-01-01")
    r0_end = to_utc_ts("2017-12-31 23:59:59")
    r1_end = to_utc_ts("2020-12-31 23:59:59")
    ctx_r0 = _build_ctx(bt_start=r0_start, bt_end=r0_end, pre_start=r0_start, pre_end=r0_end, os_start=r0_end + pd.Timedelta(minutes=5))
    ctx_r1 = _build_ctx(bt_start=r0_start, bt_end=r1_end, pre_start=r0_start, pre_end=r1_end, os_start=r1_end + pd.Timedelta(minutes=5))
    ctx_full = _build_ctx(bt_start=bt0, bt_end=bt1, pre_start=pre0, pre_end=pre1, os_start=os0)

    rng = np.random.default_rng(int(cv_cfg.seed))
    ref_entry_delay = int(ref_cfg.get("entry_delay", 0))
    ref_confirm_window = int(ref_cfg.get("confirm_window", 0))
    ref_fast_abs = float(ref_cfg.get("fast_abs_ratio", 1.0))
    ref_zero_eps = float(ref_cfg.get("zero_eps_mult", 0.0))
    ref_h1 = int(ref_cfg.get("H1", esc.H_grid[0]))
    ref_h2 = int(ref_cfg.get("H2", max(ref_h1, esc.H2_grid[0])))
    ref_tp1_q = float(ref_cfg.get("tp1_q", esc.tp1_q))
    ref_sl_q = float(ref_cfg.get("sl_q", esc.sl_q))
    ref_tp2_q = float(ref_cfg.get("tp2_q", esc.tp2_q))
    ref_tp2_n1 = int(ref_cfg.get("tp2_n1", 0))
    ref_tp2_n2 = int(ref_cfg.get("tp2_n2", 0))
    ref_state_thr = float(ref_cfg.get("state_thr", 0.55))
    ref_k_cost = float(ref_cfg.get("k_cost", esc.tp1_over_cost_k_grid[0]))

    h_pairs = [(int(ref_h1), int(ref_h2))]
    q_combos = [(float(ref_tp1_q), float(ref_sl_q), float(ref_tp2_q))]
    tp2_pairs = [(int(ref_tp2_n1), int(ref_tp2_n2))]
    thr_state_grid = (float(ref_state_thr),)
    k_cost_grid = (float(ref_k_cost),)

    risk_fixed = {
        "risk_cap_usd": float(ref_cfg.get("risk_cap_usd", risk_cfg.max_risk_usd_per_trade_grid[0])),
        "daily_stop_loss_usd": float(ref_cfg.get("daily_stop_loss_usd", risk_cfg.daily_stop_loss_usd_grid[0])),
        "dd_trigger_usd": float(ref_cfg.get("dd_trigger_usd", risk_cfg.dd_trigger_usd)),
        "dd_stop_cooldown_bars": int(ref_cfg.get("dd_stop_cooldown_bars", risk_cfg.dd_stop_cooldown_bars)),
        "risk_scale_min": float(ref_cfg.get("risk_scale_min", risk_cfg.risk_scale_min)),
    }
    risk_param_grid = [
        (
            float(risk_fixed["risk_cap_usd"]),
            float(risk_fixed["daily_stop_loss_usd"]),
            float(risk_fixed["dd_trigger_usd"]),
            int(risk_fixed["dd_stop_cooldown_bars"]),
            float(risk_fixed["risk_scale_min"]),
        )
    ]

    max_sig_trials = 1
    sig_candidates = raw_df[
        (raw_df["entry_delay"].astype(int) == int(ref_entry_delay))
        & (raw_df["confirm_window"].astype(int) == int(ref_confirm_window))
        & (raw_df["fast_abs_ratio"].astype(float) == float(ref_fast_abs))
        & (raw_df["zero_eps_mult"].astype(float) == float(ref_zero_eps))
    ].head(int(max_sig_trials))
    if sig_candidates.empty:
        raise RuntimeError("016 frozen signal config not found in raw_signal_scan")

    candidates_rows: List[Dict[str, Any]] = []
    best_row: Optional[Dict[str, Any]] = None
    best_ds_final: Optional[pd.DataFrame] = None
    best_ds_base: Optional[pd.DataFrame] = None
    best_meta: Dict[str, Any] = {}
    best_sig_cfg: Optional[Mode4SignalConfig] = None
    best_raw_counts: Dict[str, int] = {}
    tp2_candidates_rows: List[Dict[str, Any]] = []
    tp2_selected: Optional[Dict[str, Any]] = None
    tp2_calibration: Dict[str, Any] = {}
    tp2_model_meta: Dict[str, Any] = {}
    tp2_leak: Dict[str, Any] = {}
    tp2_seq_meta: Dict[str, Any] = {}
    tp2_seq_built = False
    tp2_regime_report: Dict[str, Any] = {}
    tp2_selection_status: str = "NA"
    tp2_model_obj: Optional[Any] = None
    frozen_diff: Dict[str, Any] = {}
    tp2_selected_config: Dict[str, Any] = {}
    repro_016: Dict[str, Any] = {}

    # Rung0 heap (keep only top-N by score to limit memory)
    max_r0_keep = 8
    r0_heap: List[Tuple[float, int, Dict[str, Any]]] = []
    r0_seq = 0

    for _, sig_row in sig_candidates.iterrows():
        eps_mult = float(sig_row["zero_eps_mult"])
        ind = ind_cache.get(eps_mult)
        ctx = ctx_cache.get(eps_mult)
        regimes = regime_cache.get(eps_mult)
        if ind is None or ctx is None or regimes is None:
            continue
        sig_cfg = dataclasses.replace(
            base_sig,
            entry_delay=int(sig_row["entry_delay"]),
            confirm_window=int(sig_row["confirm_window"]),
            fast_abs_ratio=float(sig_row["fast_abs_ratio"]),
            zero_eps_mult=float(eps_mult),
        )
        ev_raw = generate_mode4_events(sig_cfg, df=df0, ind=ind)
        if ev_raw.empty:
            continue
        ev_feat = attach_event_features(ev_raw, df=df0, ctx=ctx)
        path_feat = compute_path_features(df0, ctx=ctx, ev=ev_feat)
        for c in PATH_FEATURE_COLS:
            if c in path_feat.columns:
                ev_feat[c] = path_feat[c].to_numpy(dtype=float)
        entry_i = ev_feat["entry_i"].astype(int).to_numpy()
        ev_feat["vol_regime"] = np.asarray(regimes["vol_regime"], dtype=int)[entry_i]
        ev_feat["trend_regime"] = np.asarray(regimes["trend_regime"], dtype=int)[entry_i]
        ev_feat["internal_regime"] = np.asarray(regimes["internal_regime"], dtype=int)[entry_i]

        raw_pre_epd = float(sig_row["pre_raw_epd"])
        raw_pre_n = int(sig_row["pre_raw_n"])
        raw_os_epd = float(sig_row["os_raw_epd"])

        for H1, H2 in h_pairs:
            ex_base = ExitConfig(
                entry="event",
                tpslh=TPSLH(H=int(H1), tp1_atr_mult=1.0, sl_atr_mult=1.0),
                tp1_close_frac=float(esc.tp1_close_frac_grid[0]),
                tp2_mult=float(esc.tp2_mult_grid[0]),
            )
            ds_base, _meta_base = compute_event_outcomes(mkt, df=df0, ev=ev_feat, ex=ex_base)
            if ds_base.empty:
                continue
            ds_base["_entry_ts"] = pd.to_datetime(ds_base["_entry_ts"], utc=True, errors="coerce")
            ds_base = ds_base.sort_values(["entry_i", "_entry_ts"], kind="mergesort").reset_index(drop=True)
            pre_mask = (ds_base["_entry_ts"] >= pre0) & (ds_base["_entry_ts"] <= pre1)
            pre_mask_arr = pre_mask.to_numpy(dtype=bool)
            ds_base["big_loss"] = 0
            for side in ("long", "short"):
                ds_side = ds_base[ds_base["side"] == side].copy()
                if ds_side.empty:
                    continue
                pre_mask_side = (ds_side["_entry_ts"] >= pre0) & (ds_side["_entry_ts"] <= pre1)
                ds_pre_side = ds_side[pre_mask_side].copy().reset_index(drop=True)
                if ds_pre_side.empty:
                    continue
                y_tail, y_meta = _big_loss_labels(ds_pre_side)
                thr_post1 = float(y_meta.get("thr_post1", float("nan")))
                thr_post2 = float(y_meta.get("thr_post2", float("nan")))
                y_all = _apply_failure_path(ds_side, thr_post1=thr_post1, thr_post2=thr_post2)
                ds_base.loc[ds_base["side"] == side, "big_loss"] = y_all.astype(int)

            # Big loss gate (LGBM + LR ensemble, regime-weighted)
            p_tail_all = np.full(int(len(ds_base)), np.nan, dtype=float)
            thr_tail_side: Dict[str, float] = {}
            thr_tail_regime: Dict[str, Dict[Tuple[int, int, int], float]] = {"long": {}, "short": {}}
            big_loss_report: Dict[str, Any] = {"sides": {}}
            for side in ("long", "short"):
                ds_all_side = ds_base[ds_base["side"] == side].copy()
                if ds_all_side.empty:
                    continue
                idx_all = ds_all_side.index.to_numpy(dtype=int)
                pre_mask_side = (ds_all_side["_entry_ts"] >= pre0) & (ds_all_side["_entry_ts"] <= pre1)
                ds_pre_side = ds_all_side[pre_mask_side].copy().reset_index(drop=True)
                if ds_pre_side.empty:
                    continue
                ds_all_side = ds_all_side.reset_index(drop=True)
                if int(len(ds_pre_side)) > 1500:
                    ds_pre_side = (
                        ds_pre_side.sample(n=1500, random_state=int(cv_cfg.seed))
                        .sort_values("_entry_ts", kind="mergesort")
                        .reset_index(drop=True)
                    )
                y_tail, y_meta = _big_loss_labels(ds_pre_side)
                if y_tail.size == 0:
                    continue
                X_pre, med = _prepare_X(ds_pre_side, GATE_FEATURE_COLS)
                entry_i_pre = ds_pre_side["entry_i"].astype(int).to_numpy()
                exit_i_pre = ds_pre_side["exit_i"].astype(int).to_numpy()
                p_lgb_pre, cal_lgb, meta_lgb = _calibrated_predict(
                    estimator=_make_lgbm_classifier(y_tail),
                    X=X_pre,
                    y=y_tail,
                    entry_i=entry_i_pre,
                    exit_i=exit_i_pre,
                )
                p_lr_pre, cal_lr, meta_lr = _calibrated_predict(
                    estimator=_make_logreg_classifier(),
                    X=X_pre,
                    y=y_tail,
                    entry_i=entry_i_pre,
                    exit_i=exit_i_pre,
                )
                brier_lgb = float(brier_score(y_tail, p_lgb_pre))
                brier_lr = float(brier_score(y_tail, p_lr_pre))
                weights_fallback = {"lgb": float(1.0 / max(1e-6, brier_lgb)), "lr": float(1.0 / max(1e-6, brier_lr))}
                wsum = float(np.sum(list(weights_fallback.values())))
                weights_fallback = {k: float(v / wsum) for k, v in weights_fallback.items()} if wsum > 0 else {"lgb": 0.5, "lr": 0.5}
                weights_regime = build_regime_weights(ds_pre=ds_pre_side, y=y_tail, p_map={"lgb": p_lgb_pre, "lr": p_lr_pre}, min_n=200)
                p_tail_pre = combine_ensemble_probs(ds_pre_side, p_map={"lgb": p_lgb_pre, "lr": p_lr_pre}, weights_regime=weights_regime, weights_fallback=weights_fallback)
                min_take_rate = float(max(thr_cfg.gate1_take_rate_min, min(0.95, 0.8 / max(1e-9, raw_pre_epd))))
                thr, thr_meta = choose_big_loss_threshold(
                    p_big_loss=p_tail_pre,
                    y_big_loss=y_tail,
                    min_take_rate=min_take_rate,
                    target_reduction=0.5,
                    q_grid=thr_cfg.q_tail_grid,
                )
                thr_tail_side[side] = float(thr)
                for key, sub in ds_pre_side.groupby(["vol_regime", "trend_regime", "internal_regime"]):
                    if int(len(sub)) < 200:
                        continue
                    pos = sub.index.to_numpy(dtype=int)
                    p_sub = p_tail_pre[pos]
                    y_sub = np.asarray(y_tail, dtype=int)[pos]
                    thr_r, _meta_r = choose_big_loss_threshold(
                        p_big_loss=p_sub,
                        y_big_loss=y_sub,
                        min_take_rate=min_take_rate,
                        target_reduction=0.5,
                        q_grid=thr_cfg.q_tail_grid,
                    )
                    thr_tail_regime[side][tuple(int(x) for x in key)] = float(thr_r)
                X_all, _med_all = _prepare_X(ds_all_side, GATE_FEATURE_COLS, med=med)
                if int(np.unique(y_tail).size) >= 2 and int(len(ds_pre_side)) >= 200:
                    p_lgb_all = cal_lgb.predict_proba(X_all)[:, 1].astype(float)
                    p_lr_all = cal_lr.predict_proba(X_all)[:, 1].astype(float)
                else:
                    p_lgb_all = np.full(int(len(ds_all_side)), float(np.nanmean(p_lgb_pre)), dtype=float)
                    p_lr_all = np.full(int(len(ds_all_side)), float(np.nanmean(p_lr_pre)), dtype=float)
                p_tail = combine_ensemble_probs(ds_all_side, p_map={"lgb": p_lgb_all, "lr": p_lr_all}, weights_regime=weights_regime, weights_fallback=weights_fallback)
                p_tail_all[idx_all] = p_tail
                stable_feats = stable_feature_importance_by_year(mdl_cfg, ds_pre=ds_pre_side, y=y_tail, feature_cols=list(GATE_FEATURE_COLS))
                rules = summarize_big_loss_rules(ds_pre_side, y=y_tail, side=str(side))
                model_auc = {
                    "lgb": float(roc_auc(y_tail, p_lgb_pre)),
                    "lr": float(roc_auc(y_tail, p_lr_pre)),
                }
                big_loss_report["sides"][side] = {
                    "brier_preos": float(brier_score(y_tail, p_tail_pre)),
                    "model_brier": {"lgb": float(brier_lgb), "lr": float(brier_lr)},
                    "model_auc": model_auc,
                    "threshold": float(thr),
                    "threshold_meta": thr_meta,
                    "threshold_by_regime": thr_tail_regime.get(side, {}),
                    "regime_threshold_stats": _regime_threshold_stats(ds_pre_side, _apply_regime_thresholds(ds_pre_side, p=p_tail_pre, thr_default=float(thr), thr_regime=thr_tail_regime.get(side, {}), accept_if_leq=True)),
                    "weights_fallback": weights_fallback,
                    "weights_by_regime": weights_regime,
                    "oof": {"lgb": meta_lgb, "lr": meta_lr},
                    "stable_features": stable_feats,
                    "rules": rules,
                    "feature_stats": _big_loss_feature_stats(ds_pre_side, y_tail),
                    "failure_path_meta": y_meta,
                }

            gate_bigloss = np.zeros(int(len(ds_base)), dtype=bool)
            for side in ("long", "short"):
                ds_side = ds_base[ds_base["side"] == side].copy().reset_index(drop=True)
                if ds_side.empty:
                    continue
                idx = ds_base[ds_base["side"] == side].index.to_numpy(dtype=int)
                p_side = p_tail_all[idx]
                thr_def = float(thr_tail_side.get(side, 1.0))
                thr_reg = thr_tail_regime.get(side, {})
                mask = _apply_regime_thresholds(ds_side, p=p_side, thr_default=thr_def, thr_regime=thr_reg, accept_if_leq=True)
                gate_bigloss[idx] = mask
            ds_base["p_tail"] = p_tail_all
            ds_base["gate_bigloss"] = gate_bigloss

            # TP1 success gate (LGBM + RF + LR ensemble, regime-weighted)
            p_tp1_all = np.full(int(len(ds_base)), np.nan, dtype=float)
            thr_tp1_side: Dict[str, float] = {}
            thr_tp1_regime: Dict[str, Dict[Tuple[int, int, int], float]] = {"long": {}, "short": {}}
            tp1_report: Dict[str, Any] = {"sides": {}}
            for side in ("long", "short"):
                ds_all_side = ds_base[ds_base["side"] == side].copy()
                if ds_all_side.empty:
                    continue
                idx_all = ds_all_side.index.to_numpy(dtype=int)
                pre_mask_side = (ds_all_side["_entry_ts"] >= pre0) & (ds_all_side["_entry_ts"] <= pre1)
                ds_pre_side = ds_all_side[pre_mask_side & gate_bigloss[idx_all]].copy().reset_index(drop=True)
                if ds_pre_side.empty:
                    continue
                ds_all_side = ds_all_side.reset_index(drop=True)
                if int(len(ds_pre_side)) > 1500:
                    ds_pre_side = (
                        ds_pre_side.sample(n=1500, random_state=int(cv_cfg.seed))
                        .sort_values("_entry_ts", kind="mergesort")
                        .reset_index(drop=True)
                    )
                if "y_success" in ds_pre_side:
                    y_tp1 = pd.to_numeric(ds_pre_side["y_success"], errors="coerce").fillna(0).astype(int).to_numpy()
                else:
                    y_tp1 = ds_pre_side["tp1_hit"].astype(int).to_numpy()
                if int(np.unique(y_tp1).size) < 2:
                    continue
                X_pre, med = _prepare_X(ds_pre_side, GATE_FEATURE_COLS)
                entry_i_pre = ds_pre_side["entry_i"].astype(int).to_numpy()
                exit_i_pre = ds_pre_side["exit_i"].astype(int).to_numpy()
                p_lgb_pre, cal_lgb, meta_lgb = _calibrated_predict(
                    estimator=_make_lgbm_classifier(y_tp1),
                    X=X_pre,
                    y=y_tp1,
                    entry_i=entry_i_pre,
                    exit_i=exit_i_pre,
                )
                p_rf_pre, cal_rf, meta_rf = _calibrated_predict(
                    estimator=_make_rf_classifier(),
                    X=X_pre,
                    y=y_tp1,
                    entry_i=entry_i_pre,
                    exit_i=exit_i_pre,
                )
                xgb_est = _make_xgb_classifier(y_tp1)
                if xgb_est is not None:
                    p_xgb_pre, cal_xgb, meta_xgb = _calibrated_predict(
                        estimator=xgb_est,
                        X=X_pre,
                        y=y_tp1,
                        entry_i=entry_i_pre,
                        exit_i=exit_i_pre,
                    )
                    meta_xgb["ok"] = True
                else:
                    p_xgb_pre = None
                    cal_xgb = None
                    meta_xgb = {"ok": False, "reason": "xgboost_not_available"}

                base_cols = [p_lgb_pre, p_rf_pre]
                if p_xgb_pre is not None:
                    base_cols.append(p_xgb_pre)
                Z_pre = np.column_stack(base_cols)
                p_stack_pre, cal_stack, meta_stack = _calibrated_predict(
                    estimator=_make_logreg_classifier(),
                    X=Z_pre,
                    y=y_tp1,
                    entry_i=entry_i_pre,
                    exit_i=exit_i_pre,
                )

                brier_lgb = float(brier_score(y_tp1, p_lgb_pre))
                brier_rf = float(brier_score(y_tp1, p_rf_pre))
                brier_xgb = float(brier_score(y_tp1, p_xgb_pre)) if p_xgb_pre is not None else float("nan")
                p_tp1_pre = p_stack_pre
                min_take_rate = float(max(thr_cfg.gate1_take_rate_min, min(0.95, 0.8 / max(1e-9, raw_pre_epd))))
                thr_tp1, thr_meta = _select_tp1_threshold(
                    p_tp1=p_tp1_pre,
                    y_tp1=y_tp1,
                    min_take_rate=min_take_rate,
                    target_p=float(thr_cfg.tp1_target_p),
                    min_posterior=float(thr_cfg.tp1_min_posterior),
                    q_grid=thr_cfg.q_grid,
                )
                thr_tp1_side[side] = float(thr_tp1)
                for key, sub in ds_pre_side.groupby(["vol_regime", "trend_regime", "internal_regime"]):
                    if int(len(sub)) < 200:
                        continue
                    pos = sub.index.to_numpy(dtype=int)
                    p_sub = p_tp1_pre[pos]
                    y_sub = np.asarray(y_tp1, dtype=int)[pos]
                    thr_r, _meta_r = _select_tp1_threshold(
                        p_tp1=p_sub,
                        y_tp1=y_sub,
                        min_take_rate=min_take_rate,
                        target_p=float(thr_cfg.tp1_target_p),
                        min_posterior=float(thr_cfg.tp1_min_posterior),
                        q_grid=thr_cfg.q_grid,
                    )
                    thr_tp1_regime[side][tuple(int(x) for x in key)] = float(thr_r)
                X_all, _med_all = _prepare_X(ds_all_side, GATE_FEATURE_COLS, med=med)
                if int(np.unique(y_tp1).size) >= 2 and int(len(ds_pre_side)) >= 200:
                    p_lgb_all = cal_lgb.predict_proba(X_all)[:, 1].astype(float)
                    p_rf_all = cal_rf.predict_proba(X_all)[:, 1].astype(float)
                    if cal_xgb is not None:
                        p_xgb_all = cal_xgb.predict_proba(X_all)[:, 1].astype(float)
                    else:
                        p_xgb_all = None
                else:
                    p_lgb_all = np.full(int(len(ds_all_side)), float(np.nanmean(p_lgb_pre)), dtype=float)
                    p_rf_all = np.full(int(len(ds_all_side)), float(np.nanmean(p_rf_pre)), dtype=float)
                    p_xgb_all = None
                Z_all = np.column_stack([p_lgb_all, p_rf_all] + ([p_xgb_all] if p_xgb_all is not None else []))
                p_tp1 = cal_stack.predict_proba(Z_all)[:, 1].astype(float)
                p_tp1_all[idx_all] = p_tp1
                model_brier = {"lgb": float(brier_lgb), "rf": float(brier_rf), "stack": float(brier_score(y_tp1, p_stack_pre))}
                if p_xgb_pre is not None:
                    model_brier["xgb"] = float(brier_xgb)
                model_auc = {
                    "lgb": float(roc_auc(y_tp1, p_lgb_pre)),
                    "rf": float(roc_auc(y_tp1, p_rf_pre)),
                    "stack": float(roc_auc(y_tp1, p_stack_pre)),
                }
                if p_xgb_pre is not None:
                    model_auc["xgb"] = float(roc_auc(y_tp1, p_xgb_pre))
                stable_feats_tp1 = stable_feature_importance_by_year(mdl_cfg, ds_pre=ds_pre_side, y=y_tp1, feature_cols=list(GATE_FEATURE_COLS))
                tp1_report["sides"][side] = {
                    "brier_preos": float(brier_score(y_tp1, p_tp1_pre)),
                    "model_brier": model_brier,
                    "model_auc": model_auc,
                    "threshold": float(thr_tp1),
                    "threshold_meta": thr_meta,
                    "threshold_by_regime": thr_tp1_regime.get(side, {}),
                    "regime_threshold_stats": _regime_threshold_stats(ds_pre_side, _apply_regime_thresholds(ds_pre_side, p=p_tp1_pre, thr_default=float(thr_tp1), thr_regime=thr_tp1_regime.get(side, {}), accept_if_leq=False)),
                    "stacker": meta_stack,
                    "oof": {"lgb": meta_lgb, "rf": meta_rf, "xgb": meta_xgb, "stack": meta_stack},
                    "stable_features": stable_feats_tp1,
                }

            gate_tp1 = np.zeros(int(len(ds_base)), dtype=bool)
            for side in ("long", "short"):
                ds_side = ds_base[ds_base["side"] == side].copy().reset_index(drop=True)
                if ds_side.empty:
                    continue
                idx = ds_base[ds_base["side"] == side].index.to_numpy(dtype=int)
                p_side = p_tp1_all[idx]
                thr_def = float(thr_tp1_side.get(side, 0.0))
                thr_reg = thr_tp1_regime.get(side, {})
                mask = _apply_regime_thresholds(ds_side, p=p_side, thr_default=thr_def, thr_regime=thr_reg, accept_if_leq=False)
                gate_tp1[idx] = mask
            ds_base["p_success"] = p_tp1_all
            ds_base["gate_success"] = gate_tp1

            for tp1_q, sl_q, tp2_q in q_combos:
                tp1_r_dyn = np.full(int(len(ds_base)), float("nan"), dtype=float)
                sl_r_dyn = np.full(int(len(ds_base)), float("nan"), dtype=float)
                quant_report: Dict[str, Any] = {"sides": {}}
                for side in ("long", "short"):
                    ds_all_side = ds_base[ds_base["side"] == side].copy()
                    if ds_all_side.empty:
                        continue
                    idx_all = ds_all_side.index.to_numpy(dtype=int)
                    pre_mask_side = (ds_all_side["_entry_ts"] >= pre0) & (ds_all_side["_entry_ts"] <= pre1)
                    ds_pre_side = ds_all_side[pre_mask_side & gate_bigloss[idx_all] & gate_tp1[idx_all]].copy().reset_index(drop=True)
                    if ds_pre_side.empty:
                        continue
                    ds_all_side = ds_all_side.reset_index(drop=True)
                    if int(len(ds_pre_side)) > 1200:
                        ds_pre_side = (
                            ds_pre_side.sample(n=1200, random_state=int(cv_cfg.seed))
                            .sort_values("_entry_ts", kind="mergesort")
                            .reset_index(drop=True)
                        )
                    y_mfe = np.maximum(pd.to_numeric(ds_pre_side.get("mfe_r"), errors="coerce").to_numpy(dtype=float), 0.0)
                    y_mae = np.maximum(-pd.to_numeric(ds_pre_side.get("mae_r"), errors="coerce").to_numpy(dtype=float), 0.0)
                    if y_mfe.size == 0 or y_mae.size == 0:
                        continue
                    X_pre, med = _prepare_X(ds_pre_side, MODEL_FEATURE_COLS)
                    entry_i_pre = ds_pre_side["entry_i"].astype(int).to_numpy()
                    exit_i_pre = ds_pre_side["exit_i"].astype(int).to_numpy()
                    p_mfe_oof, meta_mfe = oof_predict_purged_lgbm_quantile(cv_cfg, mdl_cfg, X=X_pre, y=y_mfe, entry_i=entry_i_pre, exit_i=exit_i_pre, alpha=float(tp1_q))
                    p_mae_oof, meta_mae = oof_predict_purged_lgbm_quantile(cv_cfg, mdl_cfg, X=X_pre, y=y_mae, entry_i=entry_i_pre, exit_i=exit_i_pre, alpha=float(sl_q))
                    X_all, _med_all = _prepare_X(ds_all_side, MODEL_FEATURE_COLS, med=med)
                    if int(len(ds_pre_side)) >= 200:
                        mdl_mfe = fit_lgbm_quantile_full(mdl_cfg, X=X_pre, y=y_mfe, alpha=float(tp1_q))
                        mdl_mae = fit_lgbm_quantile_full(mdl_cfg, X=X_pre, y=y_mae, alpha=float(sl_q))
                        pred_mfe = mdl_mfe.predict(X_all).astype(float)
                        pred_mae = mdl_mae.predict(X_all).astype(float)
                    else:
                        pred_mfe = np.full(int(len(ds_all_side)), float(np.nanmedian(y_mfe)), dtype=float)
                        pred_mae = np.full(int(len(ds_all_side)), float(np.nanmedian(y_mae)), dtype=float)
                    tp1_r_dyn[idx_all] = pred_mfe
                    sl_r_dyn[idx_all] = pred_mae
                    quant_report["sides"][side] = {"mfe_oof": meta_mfe, "mae_oof": meta_mae, "tp1_q": float(tp1_q), "sl_q": float(sl_q)}

                tp1_r_dyn = np.clip(tp1_r_dyn, float(esc.min_tp1_r), float(esc.max_tp1_r))
                sl_r_dyn = np.clip(sl_r_dyn, float(esc.min_sl_r), float(esc.max_sl_r))

                # TP1 dynamic adjustment: cost pressure + sideways penalty
                cost_r_base = pd.to_numeric(ds_base.get("cost_r"), errors="coerce").to_numpy(dtype=float)
                atr_rel_base = pd.to_numeric(ds_base.get("atr_rel"), errors="coerce").to_numpy(dtype=float)
                roll_vol_base = pd.to_numeric(ds_base.get("rolling_vol_20"), errors="coerce").to_numpy(dtype=float)
                try:
                    atr_rel_thr = float(np.nanquantile(atr_rel_base[pre_mask_arr], 0.40))
                except Exception:
                    atr_rel_thr = float(np.nanquantile(atr_rel_base, 0.40)) if np.any(np.isfinite(atr_rel_base)) else 1.0
                try:
                    roll_vol_thr = float(np.nanquantile(roll_vol_base[pre_mask_arr], 0.40))
                except Exception:
                    roll_vol_thr = float(np.nanquantile(roll_vol_base, 0.40)) if np.any(np.isfinite(roll_vol_base)) else 0.0
                sideways = (atr_rel_base <= float(atr_rel_thr)) | (roll_vol_base <= float(roll_vol_thr))
                cost_adj = 1.0 + 0.03 * np.maximum(np.nan_to_num(cost_r_base, nan=0.0) - 0.4, 0.0)
                cost_adj = np.clip(cost_adj, 1.0, 1.05)
                tp1_r_dyn = np.clip(tp1_r_dyn * cost_adj * np.where(sideways, 0.9, 1.0), float(esc.min_tp1_r), float(esc.max_tp1_r))

                ds_stage1_in = ds_base.copy()
                ds_stage1_in["H1"] = int(H1)
                ds_stage1_in["H2"] = int(H2)
                ds_stage1_in["tp1_r_dyn"] = tp1_r_dyn
                ds_stage1_in["sl_r_dyn"] = sl_r_dyn
                ds_stage1_in["tp1_close_frac"] = float(esc.tp1_close_frac_grid[0])
                ds_stage1_in["tp2_r_dyn"] = np.clip(tp1_r_dyn + 0.8, tp1_r_dyn + 0.1, tp1_r_dyn + 6.0)
                ds_stage1_in["trail_mult"] = 0.4

                ds_stage1 = compute_event_outcomes_dynamic(mkt, df=df0, ev=ds_stage1_in, base_sl_atr_mult=1.0)
                if isinstance(ds_stage1, pd.Series):
                    ds_stage1 = ds_stage1.to_frame().T
                if ds_stage1.empty:
                    continue

                # TP2-only candidate generation (post-TP1)
                ds_stage1 = ds_stage1.sort_values(["entry_i", "_entry_ts"], kind="mergesort").reset_index(drop=True)
                ds_stage1["event_id"] = np.arange(int(len(ds_stage1)))

                gate_bigloss_s = ds_stage1.get("gate_bigloss")
                gate_tp1_s = ds_stage1.get("gate_success")
                gate_bigloss_s = gate_bigloss_s.astype(bool).to_numpy() if gate_bigloss_s is not None else gate_bigloss
                gate_tp1_s = gate_tp1_s.astype(bool).to_numpy() if gate_tp1_s is not None else gate_tp1
                cost_r_stage1 = pd.to_numeric(ds_stage1.get("cost_r"), errors="coerce").to_numpy(dtype=float)
                tp1_r_stage1 = pd.to_numeric(ds_stage1.get("tp1_r_dyn"), errors="coerce").to_numpy(dtype=float)
                gate_cost = np.isfinite(tp1_r_stage1) & np.isfinite(cost_r_stage1) & (tp1_r_stage1 >= float(ref_k_cost) * cost_r_stage1)
                gate_pass = gate_bigloss_s & gate_tp1_s & gate_cost
                ds_stage1["gate_pass"] = gate_pass

                tp2_macd = compute_tp2_macd_features(df0, ctx=ctx)
                if not bool(tp2_seq_built):
                    trades_018_path = out_dir / "018_artifacts" / "backtest_mode4_trades.csv"
                    tp2_seq_out = paths.artifacts_dir / "expanded_tp2_features.parquet"
                    tp2_seq_meta = build_tp2_sequence_dataset_preos(
                        df_prices=df0,
                        ctx=ctx,
                        regimes=regimes,
                        tp2_macd=tp2_macd,
                        trades_csv_path=trades_018_path,
                        pre_start=pre0,
                        pre_end=pre1,
                        H2_max=int(TP2_SEQ_H2_MAX),
                        out_path=tp2_seq_out,
                    )
                    tp2_seq_built = bool(tp2_seq_meta.get("ok", False))
                    if not bool(tp2_seq_built):
                        raise RuntimeError(f"expanded_tp2_features.parquet 生成失败：{tp2_seq_meta}")

                # TP2_019_DEEP_BLOCK_START
                use_tp2_deep_019 = True
                if bool(use_tp2_deep_019):
                    out_tp2 = tp2_deep_optimize_019(
                        paths=paths,
                        time_cfg=time_cfg,
                        cv_cfg=cv_cfg,
                        mdl_cfg=mdl_cfg,
                        mkt=mkt,
                        risk=risk_cfg,
                        esc=esc,
                        df_prices=df0,
                        ctx=ctx,
                        regimes=regimes,
                        tp2_macd=tp2_macd,
                        ds_stage1=ds_stage1,
                        gate_pass=gate_pass,
                        ctx_r0=ctx_r0,
                        ctx_r1=ctx_r1,
                        ctx_full=ctx_full,
                        risk_fixed=risk_fixed,
                        risk_trial=dataclasses.replace(
                            risk_cfg,
                            dd_trigger_usd=float(risk_fixed["dd_trigger_usd"]),
                            dd_trigger_usd_year=float(risk_fixed["dd_trigger_usd"]),
                            dd_trigger_usd_quarter=float(risk_fixed["dd_trigger_usd"]),
                            dd_stop_cooldown_bars=int(risk_fixed["dd_stop_cooldown_bars"]),
                            risk_scale_min=float(risk_fixed["risk_scale_min"]),
                        ),
                    )
                    tp2_selected = out_tp2.get("tp2_selected")
                    tp2_model_obj = out_tp2.get("tp2_model_obj")
                    tp2_model_meta = out_tp2.get("tp2_model_meta") or {}
                    tp2_calibration = out_tp2.get("tp2_calibration") or {}
                    tp2_leak = out_tp2.get("tp2_leak") or {}
                    tp2_regime_report = out_tp2.get("tp2_regime_report") or {}
                    tp2_selection_status = str(out_tp2.get("tp2_selection_status", "ok"))
                    best_row = out_tp2.get("best_row")
                    best_ds_final = out_tp2.get("best_ds_final")
                    best_ds_base = out_tp2.get("best_ds_base")
                    best_meta = out_tp2.get("best_meta") or {}
                    best_sig_cfg = out_tp2.get("best_sig_cfg")
                    candidates_rows = out_tp2.get("candidates_rows") or []
                    # Skip legacy TP2 (018) block.
                    continue
                # TP2_019_DEEP_BLOCK_END
                tp2_ds_all = ds_stage1[gate_pass & (ds_stage1["tp1_hit"] == True)].copy()
                if tp2_ds_all.empty:
                    continue
                tp2_ds_all = tp2_ds_all.reset_index(drop=True)

                tp2_feat = build_tp2_feature_frame(
                    df=df0,
                    ctx=ctx,
                    regimes=regimes,
                    tp1_idx=tp2_ds_all["tp1_hit_i"].astype(int).to_numpy(),
                    tp1_r_dyn=tp2_ds_all["tp1_r_dyn"].to_numpy(dtype=float),
                    sl_r_dyn=tp2_ds_all["sl_r_dyn"].to_numpy(dtype=float),
                    tp1_dist_ratio=tp2_ds_all["tp1_dist_ratio"].to_numpy(dtype=float),
                    macd_extra=tp2_macd,
                )
                tp2_ds_all = pd.concat([tp2_ds_all, tp2_feat], axis=1)
                # Avoid duplicate columns after concat; prefer TP1-time features from tp2_feat (keep last).
                tp2_ds_all = tp2_ds_all.loc[:, ~tp2_ds_all.columns.duplicated(keep="last")].copy()

                high = df0["high"].to_numpy(dtype=float)
                low = df0["low"].to_numpy(dtype=float)
                close = df0["close"].to_numpy(dtype=float)
                cost_total_px = float(mkt.roundtrip_cost_price) + float(mkt.slippage_buffer_price)

                def _tp2_label_for_mult(
                    df_tp1: pd.DataFrame,
                    tp2_mult_arr: np.ndarray,
                    trail_mult_arr: np.ndarray,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                    n = int(len(df_tp1))
                    y = np.zeros(n, dtype=int)
                    time_to = np.full(n, -1, dtype=int)
                    censor = np.ones(n, dtype=int)
                    for i, r in enumerate(df_tp1.itertuples(index=False)):
                        tp1_i = int(getattr(r, "tp1_hit_i"))
                        if tp1_i < 0:
                            continue
                        entry_px = float(getattr(r, "entry_price"))
                        direction = int(getattr(r, "direction"))
                        tp1_dist = float(getattr(r, "tp1_dist"))
                        atr_ref = float(getattr(r, "atr_ref"))
                        H2i = int(getattr(r, "H2"))
                        tp2_mult = float(tp2_mult_arr[i])
                        trail_mult = float(trail_mult_arr[i])
                        if not (np.isfinite(tp1_dist) and np.isfinite(entry_px) and np.isfinite(tp2_mult)):
                            continue
                        tp2_price = float(entry_px + float(direction) * float(tp1_dist) * float(tp2_mult))
                        be_price = float(entry_px + float(direction) * float(cost_total_px))
                        end_i = int(min(len(close) - 1, int(tp1_i) + int(H2i)))
                        trail_stop_px = float(max(0.0, float(trail_mult) * float(atr_ref)))
                        runner = runner_after_tp1_dynamic(
                            high=high,
                            low=low,
                            close=close,
                            direction=int(direction),
                            tp1_hit_i=int(tp1_i),
                            entry_price=float(entry_px),
                            be_price=float(be_price),
                            tp2_price=float(tp2_price),
                            trail_stop_px=float(trail_stop_px),
                            end_i=int(end_i),
                            schedule=None,
                        )
                        hit = bool(runner.get("tp2_hit", False))
                        exit_i = int(runner.get("runner_exit_i", tp1_i))
                        if hit:
                            y[i] = 1
                            censor[i] = 0
                        if int(exit_i) >= int(tp1_i):
                            time_to[i] = int(exit_i - tp1_i)
                    return y, time_to, censor

                tp2_model_base_mult = 1.6
                tp2_model_base_trail = 0.4
                base_mult_arr = np.full(int(len(tp2_ds_all)), float(tp2_model_base_mult), dtype=float)
                base_trail_arr = np.full(int(len(tp2_ds_all)), float(tp2_model_base_trail), dtype=float)
                y_base, time_base, censor_base = _tp2_label_for_mult(tp2_ds_all, base_mult_arr, base_trail_arr)
                tp2_ds_all["y_tp2_base"] = y_base
                tp2_ds_all["time_to_tp2"] = time_base
                tp2_ds_all["censor"] = censor_base

                tp2_pre_mask = (tp2_ds_all["_entry_ts"] >= pre0) & (tp2_ds_all["_entry_ts"] <= pre1)
                tp2_pre = tp2_ds_all[tp2_pre_mask].copy()
                if tp2_pre.empty:
                    continue

                tp2_leak = leakage_audit_tp2_features(seed=int(cv_cfg.seed), df_full=df0, tp1_indices=tp2_ds_all["tp1_hit_i"].to_numpy(dtype=int))
                if not bool(tp2_leak.get("ok", False)) or int(tp2_leak.get("failures_n", 1)) != 0:
                    raise RuntimeError(f"tp2 leakage_audit 失败：failures_n={int(tp2_leak.get('failures_n', 999))}")

                y_pre = tp2_pre["y_tp2_base"].astype(int).to_numpy()
                X_pre, med_tp2 = _prepare_X(tp2_pre, TP2_FEATURE_COLS)
                entry_i_tp2 = tp2_pre["tp1_hit_i"].astype(int).to_numpy()
                exit_i_tp2 = np.clip(entry_i_tp2 + int(H2), 0, len(df0) - 1)

                calib_rows: List[Dict[str, Any]] = []
                p_oof_raw, meta_oof = oof_predict_purged_sklearn(
                    cv_cfg,
                    X=X_pre,
                    y=y_pre,
                    entry_i=entry_i_tp2,
                    exit_i=exit_i_tp2,
                    estimator="logreg",
                )
                base_brier = float(brier_score(y_pre, p_oof_raw))
                base_ece, base_table = expected_calibration_error(y_pre, p_oof_raw, n_bins=10)
                calib_rows.append(
                    {
                        "method": "none",
                        "brier": float(base_brier),
                        "ece": float(base_ece),
                        "oof_meta": meta_oof,
                    }
                )

                best_method = "none"
                best_brier = float(base_brier)
                best_ece = float(base_ece)
                best_oof = np.asarray(p_oof_raw, dtype=float)
                best_table: List[Dict[str, Any]] = list(base_table)

                cal = calibrate_platt_isotonic(p_oof=p_oof_raw, y=y_pre)
                if bool(cal.get("ok", False)):
                    for method in mdl_cfg.calib_methods:
                        cal_best = dict(cal)
                        cal_best["best"] = {"method": str(method)}
                        p_cal = apply_calibration(p_oof_raw, cal=cal_best)
                        brier = float(brier_score(y_pre, p_cal))
                        ece, ece_table = expected_calibration_error(y_pre, p_cal, n_bins=10)
                        calib_rows.append(
                            {
                                "method": str(method),
                                "brier": float(brier),
                                "ece": float(ece),
                                "oof_meta": meta_oof,
                            }
                        )
                        if float(brier) < float(best_brier):
                            best_brier = float(brier)
                            best_ece = float(ece)
                            best_method = str(method)
                            best_oof = np.asarray(p_cal, dtype=float)
                            best_table = list(ece_table)
                else:
                    cal = {"ok": False, "reason": str(cal.get("reason", "unknown")), "best": {"method": "none"}, "models": {}}

                tp2_base_model = fit_logreg_full(cv_cfg, X=X_pre, y=y_pre)
                tp2_model_obj = {"base_model": tp2_base_model, "calibration": cal, "best_method": str(best_method)}
                X_all, _med_all = _prepare_X(tp2_ds_all, TP2_FEATURE_COLS, med=med_tp2)
                p_all_raw = np.asarray(tp2_base_model.predict_proba(X_all)[:, 1], dtype=float)
                if bool(cal.get("ok", False)) and str(best_method) in ("sigmoid", "isotonic"):
                    cal_best = dict(cal)
                    cal_best["best"] = {"method": str(best_method)}
                    p_all = apply_calibration(p_all_raw, cal=cal_best)
                else:
                    p_all = p_all_raw
                p_decision = np.asarray(p_all, dtype=float).copy()
                p_decision[np.asarray(tp2_pre_mask, dtype=bool)] = np.asarray(best_oof, dtype=float)
                tp2_ds_all["tp2_p"] = p_decision

                tp2_calibration = {
                    "base_mult": float(tp2_model_base_mult),
                    "method": str(best_method),
                    "brier_oof": float(best_brier),
                    "ece_oof": float(best_ece),
                    "ece_table": best_table,
                    "candidates": calib_rows,
                    "oof_meta": meta_oof,
                    "cal_ok": bool(cal.get("ok", False)),
                    "n_pre": int(len(tp2_pre)),
                    "base_rate": float(np.mean(y_pre)) if len(y_pre) else float("nan"),
                }
                tp2_model_meta = {"features": list(TP2_FEATURE_COLS), "median": [float(x) for x in np.asarray(med_tp2, dtype=float)]}

                tp2_map_idx = tp2_ds_all["event_id"].astype(int).to_numpy()
                p_map = np.full(int(len(ds_stage1)), np.nan, dtype=float)
                p_map[tp2_map_idx] = p_decision
                pre_mask_stage1 = (ds_stage1["_entry_ts"] >= pre0) & (ds_stage1["_entry_ts"] <= pre1)
                os_mask_stage1 = ds_stage1["_entry_ts"] >= os0

                risk_trial = dataclasses.replace(
                    risk_cfg,
                    dd_trigger_usd=float(risk_fixed["dd_trigger_usd"]),
                    dd_trigger_usd_year=float(risk_fixed["dd_trigger_usd"]),
                    dd_trigger_usd_quarter=float(risk_fixed["dd_trigger_usd"]),
                    dd_stop_cooldown_bars=int(risk_fixed["dd_stop_cooldown_bars"]),
                    risk_scale_min=float(risk_fixed["risk_scale_min"]),
                )

                def _eval_candidate(
                    name: str,
                    tp2_mult_tp1: np.ndarray,
                    trail_mult_tp1: np.ndarray,
                    *,
                    thr: Optional[float] = None,
                    H2_tp1: Optional[np.ndarray] = None,
                ) -> Tuple[Dict[str, Any], pd.DataFrame]:
                    tp2_mult_all = np.full(int(len(ds_stage1)), float(tp2_model_base_mult), dtype=float)
                    tp2_mult_all[tp2_map_idx] = np.asarray(tp2_mult_tp1, dtype=float)
                    trail_mult_all = np.full(int(len(ds_stage1)), float(tp2_model_base_trail), dtype=float)
                    trail_mult_all[tp2_map_idx] = np.asarray(trail_mult_tp1, dtype=float)

                    tp2_r_dyn = tp2_mult_all * tp1_r_stage1
                    tp2_r_dyn = np.where(np.isfinite(tp2_r_dyn), tp2_r_dyn, tp1_r_stage1 + 0.8)

                    ds_final_in = ds_stage1.copy()
                    if H2_tp1 is not None:
                        H2_tp1 = np.asarray(H2_tp1, dtype=int)
                        if int(H2_tp1.size) == int(tp2_map_idx.size):
                            ds_final_in.loc[tp2_map_idx, "H2"] = H2_tp1
                    ds_final_in["tp2_r_dyn"] = tp2_r_dyn
                    ds_final_in["tp2_r_n0"] = tp2_r_dyn
                    ds_final_in["tp2_n1"] = -1
                    ds_final_in["tp2_n2"] = -1
                    ds_final_in["trail_mult"] = trail_mult_all
                    ds_final_in["trail_mult_n0"] = trail_mult_all
                    ds_final = compute_event_outcomes_dynamic(mkt, df=df0, ev=ds_final_in, base_sl_atr_mult=1.0)
                    if isinstance(ds_final, pd.Series):
                        ds_final = ds_final.to_frame().T
                    if ds_final.empty:
                        return {"name": str(name), "ok": False}, ds_final
                    ds_final = ds_final.sort_values(["entry_i", "_entry_ts"], kind="mergesort").reset_index(drop=True)

                    eid = ds_final["event_id"].astype(int).to_numpy()
                    ds_final["tp2_mult"] = tp2_mult_all[eid]
                    ds_final["tp2_p"] = p_map[eid]
                    gate_pass_final = gate_pass[eid]
                    ds_final["gate_pass"] = gate_pass_final
                    pre_mask_final = pre_mask_stage1.to_numpy(dtype=bool)[eid]
                    os_mask_final = os_mask_stage1.to_numpy(dtype=bool)[eid]
                    if "signal_i" not in ds_final.columns:
                        ds_final["signal_i"] = -1
                    if "p_score" not in ds_final.columns:
                        ds_final["p_score"] = 1.0
                    if "p_tail" not in ds_final.columns:
                        ds_final["p_tail"] = 0.0
                    if "vol_regime" not in ds_final.columns:
                        ds_final["vol_regime"] = -1
                    if "trend_regime" not in ds_final.columns:
                        ds_final["trend_regime"] = -1

                    pre_sel = ds_final[pre_mask_final & gate_pass_final].copy()
                    tp1_sel = pre_sel[pre_sel["tp1_hit"] == True].copy()
                    n_tp1 = int(len(tp1_sel))
                    k_tp2 = int(np.sum(tp1_sel["tp2_hit"].astype(int))) if n_tp1 > 0 else 0
                    tp2_cond_hit = float(k_tp2 / max(1, n_tp1)) if n_tp1 > 0 else float("nan")
                    p_tp2_post = float(beta_posterior_prob_ge(k_tp2, n_tp1, 0.60)) if n_tp1 > 0 else float("nan")
                    runner_ev = float(np.nanmean(pd.to_numeric(tp1_sel.get("runner_cash_r"), errors="coerce"))) if n_tp1 > 0 else float("nan")

                    arr = build_scored_event_arrays(ds_final, mkt=mkt)
                    pass_indices = np.where(gate_pass_final)[0]
                    lot_max = lot_max_for_risk_cap(mkt, sl_dist_risk=arr.sl_dist_risk, risk_cap_usd=float(risk_fixed["risk_cap_usd"]))
                    pre_m, os_m, all_m, meta = simulate_trading_fast_metrics(
                        ctx_full,
                        mkt,
                        risk_trial,
                        arr=arr,
                        pass_indices=pass_indices,
                        lot_max_by_ticket=[lot_max],
                        daily_stop_loss_usd=float(risk_fixed["daily_stop_loss_usd"]),
                        max_parallel_same_dir=1,
                        tickets_per_signal=1,
                        tp1_close_frac=float(esc.tp1_close_frac_grid[0]),
                        cooldown_bars=0,
                        with_breakdowns=False,
                    )

                    pre_dd = float(meta.get("max_dd_usd_preos", float("nan")))
                    constraints_ok = (
                        float(pre_m.get("hit_tp1", 0.0)) >= float(ref_metrics["pre_hit_tp1"]) - 0.01
                        and float(pre_m.get("ev_r", 0.0)) >= float(ref_metrics["pre_ev_r"]) - 0.02
                        and float(pre_m.get("epd", 0.0)) >= float(ref_metrics["pre_epd"]) - 0.10
                        and float(pre_dd) <= float(ref_metrics["pre_maxdd_usd"]) + 5.0
                        and float(os_m.get("epd", 0.0)) >= float(ref_metrics["os_epd"]) - 0.10
                    )
                    posterior_improved = float(p_tp2_post) > float(ref_metrics.get("posterior_tp2", 0.0)) + 1e-9

                    cand = {
                        "name": str(name),
                        "tp2_kind": "tiered" if thr is not None else "fixed",
                        "tp2_thr": float(thr) if thr is not None else float("nan"),
                        "tp2_mult_low": float(np.nanmin(tp2_mult_tp1)) if tp2_mult_tp1.size else float("nan"),
                        "tp2_mult_high": float(np.nanmax(tp2_mult_tp1)) if tp2_mult_tp1.size else float("nan"),
                        "pre_tp2_cond_hit": float(tp2_cond_hit),
                        "posterior_tp2": float(p_tp2_post),
                        "n_tp1": int(n_tp1),
                        "k_tp2": int(k_tp2),
                        "pre_runner_ev": float(runner_ev),
                        "pre_epd": float(pre_m.get("epd", float("nan"))),
                        "pre_pf": float(pre_m.get("pf", float("nan"))),
                        "pre_ev_r": float(pre_m.get("ev_r", float("nan"))),
                        "pre_hit_tp1": float(pre_m.get("hit_tp1", float("nan"))),
                        "pre_hit_tp2": float(pre_m.get("hit_tp2", float("nan"))),
                        "pre_maxdd_usd": float(pre_dd),
                        "os_epd": float(os_m.get("epd", float("nan"))),
                        "all_epd": float(all_m.get("epd", float("nan"))),
                        "all_ev_r": float(all_m.get("ev_r", float("nan"))),
                        "constraints_ok": bool(constraints_ok),
                        "posterior_improved": bool(posterior_improved),
                    }
                    return cand, ds_final

                # =============================
                # 018 TP2 policy: bucketed Beta-Binomial on extra_MFE_R (pre-OS only)
                # =============================

                tp2_policy: Dict[str, Any] = {
                    "type": "bucket_beta_binomial",
                    "p0": float(TP2_P0),
                    "posterior_target": float(TP2_POSTERIOR_TARGET),
                    "min_bucket_n": int(TP2_MIN_BUCKET_N),
                    "extra_r_grid": [float(x) for x in TP2_EXTRA_R_GRID],
                    "H2_grid": [int(x) for x in esc.H2_grid],
                    "vol_cut_q": [float(TP2_VOL_Q_LO), float(TP2_VOL_Q_HI)],
                    "trend_cut_q": [float(TP2_TREND_Q_LO), float(TP2_TREND_Q_HI)],
                }

                vol_pre = pd.to_numeric(tp2_pre.get("atr_rel"), errors="coerce").to_numpy(dtype=float)
                trend_pre = pd.to_numeric(tp2_pre.get("ema20_slope_atr"), errors="coerce").to_numpy(dtype=float)
                vol_q_lo = float(np.nanquantile(vol_pre, float(TP2_VOL_Q_LO))) if np.any(np.isfinite(vol_pre)) else 1.0
                vol_q_hi = float(np.nanquantile(vol_pre, float(TP2_VOL_Q_HI))) if np.any(np.isfinite(vol_pre)) else 1.0
                trend_q_lo = float(np.nanquantile(trend_pre, float(TP2_TREND_Q_LO))) if np.any(np.isfinite(trend_pre)) else 0.0
                trend_q_hi = float(np.nanquantile(trend_pre, float(TP2_TREND_Q_HI))) if np.any(np.isfinite(trend_pre)) else 0.0
                tp2_policy["bucket_cuts"] = {
                    "atr_rel": {"q_lo": float(vol_q_lo), "q_hi": float(vol_q_hi)},
                    "ema20_slope_atr": {"q_lo": float(trend_q_lo), "q_hi": float(trend_q_hi)},
                }

                tp2_pre = tp2_pre.copy()
                tp2_ds_all = tp2_ds_all.copy()
                tp2_pre["bucket"] = build_tp2_buckets(tp2_pre, vol_q_lo=vol_q_lo, vol_q_hi=vol_q_hi, trend_q_lo=trend_q_lo, trend_q_hi=trend_q_hi)
                tp2_ds_all["bucket"] = build_tp2_buckets(tp2_ds_all, vol_q_lo=vol_q_lo, vol_q_hi=vol_q_hi, trend_q_lo=trend_q_lo, trend_q_hi=trend_q_hi)

                # extra_MFE_R by H2 on pre-OS (label-only; no OS leakage)
                high = df0["high"].to_numpy(dtype=float)
                low = df0["low"].to_numpy(dtype=float)
                extra_by_h2: Dict[int, np.ndarray] = {}
                for H2i in esc.H2_grid:
                    vals = np.zeros(int(len(tp2_pre)), dtype=float)
                    for j, r in enumerate(tp2_pre.itertuples(index=False)):
                        vals[int(j)] = tp2_extra_mfe_r(
                            high=high,
                            low=low,
                            direction=int(getattr(r, "direction")),
                            tp1_hit_i=int(getattr(r, "tp1_hit_i")),
                            entry_price=float(getattr(r, "entry_price")),
                            tp1_dist=float(getattr(r, "tp1_dist")),
                            sl_dist=float(getattr(r, "sl_dist")),
                            H2=int(H2i),
                            cost_total_px=float(cost_total_px),
                        )
                    extra_by_h2[int(H2i)] = np.asarray(vals, dtype=float)

                bucket_arr = tp2_pre["bucket"].astype(str).to_numpy()
                bucket_to_idx: Dict[str, np.ndarray] = {}
                for b in np.unique(bucket_arr):
                    bucket_to_idx[str(b)] = np.where(bucket_arr == str(b))[0]

                stats_rows: List[Dict[str, Any]] = []
                for bucket, idx in bucket_to_idx.items():
                    n = int(idx.size)
                    for H2i, extra_vec in extra_by_h2.items():
                        extra_b = np.asarray(extra_vec[idx], dtype=float)
                        for extra_r in TP2_EXTRA_R_GRID:
                            k = int(np.sum(extra_b >= float(extra_r)))
                            post = float(beta_posterior_prob_ge(k, n, float(TP2_P0))) if n > 0 else float("nan")
                            stats_rows.append(
                                {
                                    "bucket": str(bucket),
                                    "H2": int(H2i),
                                    "extra_r": float(extra_r),
                                    "n": int(n),
                                    "k": int(k),
                                    "k_over_n": float(k / max(1, n)),
                                    "posterior": float(post),
                                    "eligible": bool(n >= int(TP2_MIN_BUCKET_N) and float(post) >= float(TP2_POSTERIOR_TARGET)),
                                }
                            )

                tp2_bucket_stats_df = pd.DataFrame(stats_rows)
                tp2_bucket_stats_df.to_csv(paths.artifacts_dir / "tp2_bucket_stats.csv", index=False)

                policy_map: Dict[str, Dict[str, Any]] = {}
                policy_rows: List[Dict[str, Any]] = []
                for bucket, g in tp2_bucket_stats_df.groupby("bucket", sort=False):
                    g_ok = g[g["eligible"] == True].copy()
                    if g_ok.empty:
                        continue
                    best = g_ok.sort_values(["extra_r", "posterior", "H2"], ascending=[False, False, True]).iloc[0]
                    pol = {
                        "extra_r": float(best["extra_r"]),
                        "H2": int(best["H2"]),
                        "posterior": float(best["posterior"]),
                        "n": int(best["n"]),
                        "k": int(best["k"]),
                    }
                    policy_map[str(bucket)] = pol
                    policy_rows.append({"bucket": str(bucket), **pol, "k_over_n": float(best["k_over_n"])})

                tp2_candidates_df = pd.DataFrame(policy_rows).sort_values(["extra_r", "posterior", "n"], ascending=[False, False, False]).reset_index(drop=True)
                tp2_candidates_df.to_csv(paths.artifacts_dir / "tp2_candidates.csv", index=False)
                tp2_policy["buckets"] = policy_map
                write_json(paths.artifacts_dir / "tp2_policy.json", tp2_policy)

                # Summary on pre-OS TP1-hit set (attempt-all within enabled buckets)
                attempted_n_pre = int(sum(int(v.get("n", 0)) for v in policy_map.values()))
                attempted_k_pre = int(sum(int(v.get("k", 0)) for v in policy_map.values()))
                attempt_rate_pre = float(attempted_n_pre / max(1, int(len(tp2_pre))))
                tp2_cond_hit_pre = float(attempted_k_pre / max(1, attempted_n_pre)) if attempted_n_pre > 0 else float("nan")
                post_vals = np.asarray([float(v.get("posterior", float("nan"))) for v in policy_map.values()], dtype=float)
                post_vals = post_vals[np.isfinite(post_vals)]
                post_min = float(np.nanmin(post_vals)) if post_vals.size else float("nan")
                post_p05 = float(np.nanquantile(post_vals, 0.05)) if post_vals.size else float("nan")
                post_p50 = float(np.nanquantile(post_vals, 0.50)) if post_vals.size else float("nan")
                post_p95 = float(np.nanquantile(post_vals, 0.95)) if post_vals.size else float("nan")

                if attempted_n_pre > 0:
                    extra_rep = np.concatenate([np.full(int(v["n"]), float(v["extra_r"]), dtype=float) for v in policy_map.values()])
                    extra_mean = float(np.nanmean(extra_rep))
                    extra_median = float(np.nanmedian(extra_rep))
                else:
                    extra_mean = float("nan")
                    extra_median = float("nan")

                # Reproduce 016 (mode4, frozen non-TP2) before applying any TP2 policy.
                if not repro_016:
                    ds_stage1_base = ds_stage1.copy()
                    if "signal_i" not in ds_stage1_base.columns:
                        ds_stage1_base["signal_i"] = -1
                    if "p_score" not in ds_stage1_base.columns:
                        ds_stage1_base["p_score"] = 1.0
                    if "p_tail" not in ds_stage1_base.columns:
                        ds_stage1_base["p_tail"] = 0.0
                    if "vol_regime" not in ds_stage1_base.columns:
                        ds_stage1_base["vol_regime"] = -1
                    if "trend_regime" not in ds_stage1_base.columns:
                        ds_stage1_base["trend_regime"] = -1
                    base_arr = build_scored_event_arrays(ds_stage1_base, mkt=mkt)
                    base_pass_indices = np.where(gate_pass)[0]
                    base_lot_max = lot_max_for_risk_cap(mkt, sl_dist_risk=base_arr.sl_dist_risk, risk_cap_usd=float(risk_fixed["risk_cap_usd"]))
                    base_pre_m, base_os_m, base_all_m, base_meta = simulate_trading_fast_metrics(
                        ctx_full,
                        mkt,
                        risk_trial,
                        arr=base_arr,
                        pass_indices=base_pass_indices,
                        lot_max_by_ticket=[base_lot_max],
                        daily_stop_loss_usd=float(risk_fixed["daily_stop_loss_usd"]),
                        max_parallel_same_dir=1,
                        tickets_per_signal=1,
                        tp1_close_frac=float(esc.tp1_close_frac_grid[0]),
                        cooldown_bars=0,
                        with_breakdowns=False,
                    )
                    base_dd_pre = float(base_meta.get("max_dd_usd_preos", float("nan")))
                    repro_016 = {
                        "pre": dict(base_pre_m),
                        "os": dict(base_os_m),
                        "all": dict(base_all_m),
                        "dd_pre": float(base_dd_pre),
                        "dd_os": float(base_meta.get("max_dd_usd_os", float("nan"))),
                        "dd_all": float(base_meta.get("max_dd_usd", float("nan"))),
                    }
                    ok = (
                        float(base_pre_m.get("epd", 0.0)) >= 0.8
                        and float(base_pre_m.get("hit_tp1", 0.0)) >= 0.70
                        and float(base_pre_m.get("ev_r", -1.0)) > 0.0
                        and np.isfinite(base_dd_pre)
                        and float(base_dd_pre) <= 45.0
                        and float(base_os_m.get("epd", 0.0)) > 0.0
                    )
                    repro_016["ok"] = bool(ok)
                    if not ok:
                        raise RuntimeError(
                            "016 reproduction failed: "
                            f"pre_epd={fmt(base_pre_m.get('epd'),4)}, "
                            f"pre_hit_tp1={fmt(base_pre_m.get('hit_tp1'),4)}, "
                            f"pre_ev_r={fmt(base_pre_m.get('ev_r'),4)}, "
                            f"pre_maxDD_usd={fmt(base_dd_pre,2)}, "
                            f"os_epd={fmt(base_os_m.get('epd'),4)}"
                        )

                # Apply TP2 policy to all TP1-hit events (all periods), recompute outcomes, then evaluate via fast simulator.
                ds_final_in = ds_stage1.copy()
                # Ensure runner schedule fields are defined for all rows (avoid NaN->int crashes).
                ds_final_in["tp2_n1"] = -1
                ds_final_in["tp2_n2"] = -1
                ds_final_in["tp2_attempt"] = False
                ds_final_in["tp2_extra_r"] = np.nan
                ds_final_in["tp2_h2_policy"] = -1
                tp2_map_idx_full = tp2_ds_all["event_id"].astype(int).to_numpy()
                tp2_bucket_all = tp2_ds_all["bucket"].astype(str).to_numpy()

                tp1_r_dyn_all = pd.to_numeric(ds_final_in.get("tp1_r_dyn"), errors="coerce").to_numpy(dtype=float)
                sl_r_dyn_all = pd.to_numeric(ds_final_in.get("sl_r_dyn"), errors="coerce").to_numpy(dtype=float)
                for i, eid in enumerate(tp2_map_idx_full):
                    pol = policy_map.get(str(tp2_bucket_all[int(i)]))
                    if pol is None:
                        continue
                    exr = float(pol["extra_r"])
                    h2v = int(pol["H2"])
                    if not (np.isfinite(exr) and float(TP2_EXTRA_R_CLIP[0]) - 1e-12 <= exr <= float(TP2_EXTRA_R_CLIP[1]) + 1e-12 and h2v > 0):
                        continue
                    if not (0 <= int(eid) < int(len(ds_final_in))):
                        continue
                    base_tp1 = float(tp1_r_dyn_all[int(eid)])
                    base_sl = float(sl_r_dyn_all[int(eid)])
                    if not (np.isfinite(base_tp1) and np.isfinite(base_sl) and base_tp1 > 1e-12):
                        continue
                    ds_final_in.at[int(eid), "H2"] = int(h2v)
                    ds_final_in.at[int(eid), "tp2_r_dyn"] = float(base_tp1 + float(exr) * float(base_sl))
                    ds_final_in.at[int(eid), "trail_mult"] = float(TP2_DISABLE_TRAIL_MULT)
                    ds_final_in.at[int(eid), "tp2_n1"] = -1
                    ds_final_in.at[int(eid), "tp2_n2"] = -1
                    ds_final_in.at[int(eid), "tp2_attempt"] = True
                    ds_final_in.at[int(eid), "tp2_extra_r"] = float(exr)
                    ds_final_in.at[int(eid), "tp2_h2_policy"] = int(h2v)

                ds_final = compute_event_outcomes_dynamic(mkt, df=df0, ev=ds_final_in, base_sl_atr_mult=1.0)
                if isinstance(ds_final, pd.Series):
                    ds_final = ds_final.to_frame().T
                ds_final = ds_final.sort_values(["entry_i", "_entry_ts"], kind="mergesort").reset_index(drop=True)
                if "signal_i" not in ds_final.columns:
                    ds_final["signal_i"] = -1
                if "p_score" not in ds_final.columns:
                    ds_final["p_score"] = 1.0
                if "p_tail" not in ds_final.columns:
                    ds_final["p_tail"] = 0.0
                if "vol_regime" not in ds_final.columns:
                    ds_final["vol_regime"] = -1
                if "trend_regime" not in ds_final.columns:
                    ds_final["trend_regime"] = -1

                eid = ds_final["event_id"].astype(int).to_numpy()
                gate_pass_final = gate_pass[eid]
                ds_final["gate_pass"] = gate_pass_final

                pre_mask_final = pre_mask_stage1.to_numpy(dtype=bool)[eid]
                os_mask_final = os_mask_stage1.to_numpy(dtype=bool)[eid]

                pre_sel = ds_final[pre_mask_final & gate_pass_final].copy()
                tp1_sel = pre_sel[pre_sel["tp1_hit"] == True].copy()
                n_tp1 = int(len(tp1_sel))
                att = tp1_sel.get("tp2_attempt")
                if att is None:
                    att_mask = np.zeros(n_tp1, dtype=bool)
                else:
                    att_mask = att.astype(bool).to_numpy()
                n_att = int(np.sum(att_mask))
                k_att = int(np.sum((tp1_sel["tp2_hit"].astype(bool).to_numpy() & att_mask))) if n_att > 0 else 0
                tp2_cond_hit_att = float(k_att / max(1, n_att)) if n_att > 0 else float("nan")

                # Evaluate trading metrics (fast) with unchanged risk controls
                arr = build_scored_event_arrays(ds_final, mkt=mkt)
                pass_indices = np.where(gate_pass_final)[0]
                lot_max = lot_max_for_risk_cap(mkt, sl_dist_risk=arr.sl_dist_risk, risk_cap_usd=float(risk_fixed["risk_cap_usd"]))
                pre_m, os_m, all_m, meta = simulate_trading_fast_metrics(
                    ctx_full,
                    mkt,
                    risk_trial,
                    arr=arr,
                    pass_indices=pass_indices,
                    lot_max_by_ticket=[lot_max],
                    daily_stop_loss_usd=float(risk_fixed["daily_stop_loss_usd"]),
                    max_parallel_same_dir=1,
                    tickets_per_signal=1,
                    tp1_close_frac=float(esc.tp1_close_frac_grid[0]),
                    cooldown_bars=0,
                    with_breakdowns=False,
                )
                pre_dd = float(meta.get("max_dd_usd_preos", float("nan")))

                tp2_selected_row = {
                    "name": "bucket_confmax",
                    "tp2_kind": "bucket",
                    "tp2_thr": float("nan"),
                    "tp2_mult_low": float("nan"),
                    "tp2_mult_high": float("nan"),
                    "tp2_attempt_rate_pre": float(attempt_rate_pre),
                    "pre_tp2_cond_hit": float(tp2_cond_hit_att),
                    "posterior_tp2": float(post_min) if np.isfinite(post_min) else 0.0,
                    "posterior_min": float(post_min),
                    "posterior_p05": float(post_p05),
                    "posterior_p50": float(post_p50),
                    "posterior_p95": float(post_p95),
                    "extra_r_mean": float(extra_mean),
                    "extra_r_median": float(extra_median),
                    "n_tp1": int(n_att),
                    "k_tp2": int(k_att),
                    "pre_epd": float(pre_m.get("epd", float("nan"))),
                    "pre_pf": float(pre_m.get("pf", float("nan"))),
                    "pre_ev_r": float(pre_m.get("ev_r", float("nan"))),
                    "pre_hit_tp1": float(pre_m.get("hit_tp1", float("nan"))),
                    "pre_hit_tp2": float(pre_m.get("hit_tp2", float("nan"))),
                    "pre_maxdd_usd": float(pre_dd),
                    "os_epd": float(os_m.get("epd", float("nan"))),
                    "all_epd": float(all_m.get("epd", float("nan"))),
                    "all_ev_r": float(all_m.get("ev_r", float("nan"))),
                    "constraints_ok": bool(np.isfinite(pre_dd) and float(pre_dd) <= 45.0 and float(pre_m.get("epd", 0.0)) >= 0.8 and float(pre_m.get("hit_tp1", 0.0)) >= 0.70 and float(pre_m.get("ev_r", -1.0)) >= 0.0 and float(os_m.get("epd", 0.0)) > 0.0),
                    "posterior_improved": bool(np.isfinite(post_min) and float(post_min) > float(ref_metrics.get("posterior_tp2", 0.0)) + 1e-12),
                }
                tp2_candidates_rows.append(tp2_selected_row)
                best_candidate = tp2_selected_row
                best_ds = ds_final
                tp2_selection_status = "ok" if policy_map else "disabled"

                if best_candidate is None or best_ds is None:
                    continue

                tp2_selected = best_candidate
                candidates_rows = list(tp2_candidates_rows)
                best_row = {
                    "pre_epd": float(best_candidate.get("pre_epd", float("nan"))),
                    "pre_pf": float(best_candidate.get("pre_pf", float("nan"))),
                    "pre_ev_r": float(best_candidate.get("pre_ev_r", float("nan"))),
                    "pre_hit_tp1": float(best_candidate.get("pre_hit_tp1", float("nan"))),
                    "pre_hit_tp2": float(best_candidate.get("pre_hit_tp2", float("nan"))),
                    "pre_maxdd_usd": float(best_candidate.get("pre_maxdd_usd", float("nan"))),
                    "os_epd": float(best_candidate.get("os_epd", float("nan"))),
                    "all_epd": float(best_candidate.get("all_epd", float("nan"))),
                    "all_pf": float(best_candidate.get("pre_pf", float("nan"))),
                    "all_ev_r": float(best_candidate.get("all_ev_r", float("nan"))),
                    "all_maxdd_usd": float(best_candidate.get("pre_maxdd_usd", float("nan"))),
                    "entry_delay": int(sig_row.get("entry_delay", 0)),
                    "confirm_window": int(sig_row.get("confirm_window", 0)),
                    "fast_abs_ratio": float(sig_row.get("fast_abs_ratio", 1.0)),
                    "zero_eps_mult": float(sig_row.get("zero_eps_mult", 0.0)),
                    "H1": int(H1),
                    "H2": int(H2),
                    "tp1_q": float(tp1_q),
                    "sl_q": float(sl_q),
                    "tp2_q": float(ref_tp2_q),
                    "k_cost": float(ref_k_cost),
                    "risk_cap_usd": float(risk_fixed["risk_cap_usd"]),
                    "daily_stop_loss_usd": float(risk_fixed["daily_stop_loss_usd"]),
                    "dd_trigger_usd": float(risk_fixed["dd_trigger_usd"]),
                    "dd_stop_cooldown_bars": int(risk_fixed["dd_stop_cooldown_bars"]),
                    "risk_scale_min": float(risk_fixed["risk_scale_min"]),
                    "tp2_n1": int(ref_tp2_n1),
                    "tp2_n2": int(ref_tp2_n2),
                    "state_thr": float(ref_state_thr),
                    "posterior_tp1": float(ref_cfg.get("posterior_tp1", float("nan"))),
                    "posterior_tp2": float(best_candidate.get("posterior_tp2", float("nan"))),
                    "posterior_sl": float(ref_cfg.get("posterior_sl", float("nan"))),
                    "raw_pre_epd": float(sig_row.get("pre_raw_epd", float("nan"))),
                    "raw_os_epd": float(sig_row.get("os_raw_epd", float("nan"))),
                    "constraints_ok": bool(best_candidate.get("constraints_ok")),
                    "score": float(best_candidate.get("pre_ev_r", float("nan"))),
                }
                best_ds_final = best_ds
                best_ds_base = ds_base.copy()
                best_meta = {
                    "big_loss_report": big_loss_report,
                    "tp1_report": tp1_report,
                    "quant_report": quant_report,
                    "tp2_calibration": tp2_calibration,
                    "tp2_model_meta": tp2_model_meta,
                    "tp2_selection_status": tp2_selection_status,
                    "posterior": {
                        "tp1": float(ref_cfg.get("posterior_tp1", float("nan"))),
                        "sl": float(ref_cfg.get("posterior_sl", float("nan"))),
                        "tp2": float(best_candidate.get("posterior_tp2", float("nan"))),
                        "n_tp2": int(best_candidate.get("n_tp1", 0)),
                        "k_tp2": int(best_candidate.get("k_tp2", 0)),
                    },
                }
                best_sig_cfg = sig_cfg
                sig_row_map = sig_row.to_dict()
                best_raw_counts = {
                    "pre": int(sig_row_map.get("pre_raw_n", 0)),
                    "os": int(sig_row_map.get("os_raw_n", 0)),
                    "all": int(sig_row_map.get("all_raw_n", 0)),
                }

    rung0_keep = [c for _, _, c in sorted(r0_heap, key=lambda x: x[0], reverse=True)]

    # Rung1: 2015-2020
    rung1_keep: List[Dict[str, Any]] = []
    for cand in rung0_keep:
        arr = cand["arr"]
        pass_indices = cand["pass_indices"]
        best_r1 = {"score": -1e18}
        best_r1_any = {"score": -1e18}
        for risk_cap, dsl, dd_thr, dd_cd, rs in risk_param_grid:
            risk_trial = dataclasses.replace(
                risk_cfg,
                dd_trigger_usd=float(dd_thr),
                dd_trigger_usd_year=float(dd_thr),
                dd_trigger_usd_quarter=float(dd_thr),
                dd_stop_cooldown_bars=int(dd_cd),
                risk_scale_min=float(rs),
            )
            lot_max = lot_max_for_risk_cap(mkt, sl_dist_risk=arr.sl_dist_risk, risk_cap_usd=float(risk_cap))
            pre_m, _os_m, _all_m, meta = simulate_trading_fast_metrics(
                ctx_r1,
                mkt,
                risk_trial,
                arr=arr,
                pass_indices=pass_indices,
                lot_max_by_ticket=[lot_max],
                daily_stop_loss_usd=float(dsl),
                max_parallel_same_dir=1,
                tickets_per_signal=1,
                tp1_close_frac=float(esc.tp1_close_frac_grid[0]),
                cooldown_bars=0,
                with_breakdowns=False,
            )
            pre_dd1 = float(meta.get("max_dd_usd_preos", float("inf")))
            stage1_ok = (
                float(pre_m.get("epd", 0.0)) >= 0.7
                and float(pre_m.get("hit_tp1", 0.0)) >= 0.65
                and float(pre_dd1) <= 50.0
            )
            score = float(pre_m["ev_r"]) + 0.2 * float(pre_m["epd"]) + 0.1 * float(pre_m["hit_tp1"]) - 0.003 * float(pre_dd1)
            if float(score) > float(best_r1_any.get("score", -1e18)):
                best_r1_any = {
                    "score": float(score),
                    "risk_cap": float(risk_cap),
                    "dsl": float(dsl),
                    "dd_trigger_usd": float(dd_thr),
                    "dd_stop_cooldown_bars": int(dd_cd),
                    "risk_scale_min": float(rs),
                    "pre_m": pre_m,
                    "meta": meta,
                }
            if float(score) > float(best_r1.get("score", -1e18)):
                best_r1 = {
                    "score": float(score),
                    "risk_cap": float(risk_cap),
                    "dsl": float(dsl),
                    "dd_trigger_usd": float(dd_thr),
                    "dd_stop_cooldown_bars": int(dd_cd),
                    "risk_scale_min": float(rs),
                    "pre_m": pre_m,
                    "meta": meta,
                } if stage1_ok else best_r1
        if float(best_r1.get("score", -1e18)) <= -1e17:
            best_r1 = best_r1_any
            best_r1["stage1_ok"] = False
        else:
            best_r1["stage1_ok"] = True
        if float(best_r1_any.get("score", -1e18)) <= -1e17:
            continue
        cand["best_r1"] = best_r1
        rung1_keep.append(cand)

    rung1_keep = sorted(rung1_keep, key=lambda x: float((x.get("best_r1") or {}).get("score", -1e18)), reverse=True)
    rung1_keep = rung1_keep[: max(6, int(len(rung1_keep) * 0.5))]

    # Rung2: full pre-OS
    for cand in rung1_keep:
        arr = cand["arr"]
        pass_indices = cand["pass_indices"]
        best_r2 = {"score": -1e18}
        best_r2_any = {"score": -1e18}
        for risk_cap, dsl, dd_thr, dd_cd, rs in risk_param_grid:
            risk_trial = dataclasses.replace(
                risk_cfg,
                dd_trigger_usd=float(dd_thr),
                dd_trigger_usd_year=float(dd_thr),
                dd_trigger_usd_quarter=float(dd_thr),
                dd_stop_cooldown_bars=int(dd_cd),
                risk_scale_min=float(rs),
            )
            lot_max = lot_max_for_risk_cap(mkt, sl_dist_risk=arr.sl_dist_risk, risk_cap_usd=float(risk_cap))
            pre_m, os_m, all_m, meta = simulate_trading_fast_metrics(
                ctx_full,
                mkt,
                risk_trial,
                arr=arr,
                pass_indices=pass_indices,
                lot_max_by_ticket=[lot_max],
                daily_stop_loss_usd=float(dsl),
                max_parallel_same_dir=1,
                tickets_per_signal=1,
                tp1_close_frac=float(esc.tp1_close_frac_grid[0]),
                cooldown_bars=0,
                with_breakdowns=False,
            )
            pre_dd2 = float(meta.get("max_dd_usd_preos", float("inf")))
            stage2_ok = (
                float(pre_m.get("epd", 0.0)) >= 0.8
                and float(pre_m.get("hit_tp1", 0.0)) >= 0.70
                and float(pre_m.get("ev_r", -1.0)) >= 0.0
                and float(pre_dd2) <= 45.0
            )
            score = float(pre_m["ev_r"]) + 0.2 * float(pre_m["epd"]) + 0.1 * float(pre_m["hit_tp1"]) - 0.003 * float(pre_dd2)
            if float(score) > float(best_r2_any.get("score", -1e18)):
                best_r2_any = {
                    "score": float(score),
                    "risk_cap": float(risk_cap),
                    "dsl": float(dsl),
                    "dd_trigger_usd": float(dd_thr),
                    "dd_stop_cooldown_bars": int(dd_cd),
                    "risk_scale_min": float(rs),
                    "pre_m": pre_m,
                    "os_m": os_m,
                    "all_m": all_m,
                    "meta": meta,
                }
            if float(score) > float(best_r2.get("score", -1e18)):
                best_r2 = {
                    "score": float(score),
                    "risk_cap": float(risk_cap),
                    "dsl": float(dsl),
                    "dd_trigger_usd": float(dd_thr),
                    "dd_stop_cooldown_bars": int(dd_cd),
                    "risk_scale_min": float(rs),
                    "pre_m": pre_m,
                    "os_m": os_m,
                    "all_m": all_m,
                    "meta": meta,
                } if stage2_ok else best_r2
        if float(best_r2.get("score", -1e18)) <= -1e17:
            best_r2 = best_r2_any
            best_r2["stage2_ok"] = False
        else:
            best_r2["stage2_ok"] = True
        if float(best_r2_any.get("score", -1e18)) <= -1e17:
            continue
        cand["best_r2"] = best_r2

        pre_m = best_r2.get("pre_m") or {}
        os_m = best_r2.get("os_m") or {}
        all_m = best_r2.get("all_m") or {}
        meta = best_r2.get("meta") or {}
        sig_row = cand.get("sig_row") or {}
        post = cand.get("posterior") or {}

        row = {
            "pre_epd": float(pre_m.get("epd", float("nan"))),
            "pre_pf": float(pre_m.get("pf", float("nan"))),
            "pre_ev_r": float(pre_m.get("ev_r", float("nan"))),
            "pre_hit_tp1": float(pre_m.get("hit_tp1", float("nan"))),
            "pre_hit_tp2": float(pre_m.get("hit_tp2", float("nan"))),
            "pre_maxdd_usd": float(meta.get("max_dd_usd_preos", float("nan"))),
            "os_epd": float(os_m.get("epd", float("nan"))),
            "all_epd": float(all_m.get("epd", float("nan"))),
            "all_pf": float(all_m.get("pf", float("nan"))),
            "all_ev_r": float(all_m.get("ev_r", float("nan"))),
            "all_maxdd_usd": float(meta.get("max_dd_usd", float("nan"))),
            "entry_delay": int(sig_row.get("entry_delay", 0)),
            "confirm_window": int(sig_row.get("confirm_window", 0)),
            "fast_abs_ratio": float(sig_row.get("fast_abs_ratio", 1.0)),
            "zero_eps_mult": float(sig_row.get("zero_eps_mult", 0.0)),
            "H1": int(cand.get("H1", 0)),
            "H2": int(cand.get("H2", 0)),
            "tp1_q": float(cand.get("tp1_q", float("nan"))),
            "sl_q": float(cand.get("sl_q", float("nan"))),
            "tp2_q": float(cand.get("tp2_q", float("nan"))),
            "k_cost": float(cand.get("k_cost", float("nan"))),
            "risk_cap_usd": float(best_r2.get("risk_cap", float("nan"))),
            "daily_stop_loss_usd": float(best_r2.get("dsl", float("nan"))),
            "dd_trigger_usd": float(best_r2.get("dd_trigger_usd", float("nan"))),
            "dd_stop_cooldown_bars": int(best_r2.get("dd_stop_cooldown_bars", 0)),
            "risk_scale_min": float(best_r2.get("risk_scale_min", float("nan"))),
            "tp2_n1": int(cand.get("tp2_n1", 0)),
            "tp2_n2": int(cand.get("tp2_n2", 0)),
            "state_thr": float(cand.get("state_thr", float("nan"))),
            "posterior_tp1": float(post.get("tp1", float("nan"))),
            "posterior_tp2": float(post.get("tp2", float("nan"))),
            "posterior_sl": float(post.get("sl", float("nan"))),
            "raw_pre_epd": float(sig_row.get("pre_raw_epd", float("nan"))),
            "raw_os_epd": float(sig_row.get("os_raw_epd", float("nan"))),
        }
        candidates_rows.append(row)

        constraints_ok = (
            float(pre_m.get("ev_r", -1.0)) >= 0.0
            and float(pre_m.get("epd", 0.0)) >= 0.8
            and float(pre_m.get("hit_tp1", 0.0)) >= 0.70
            and float(meta.get("max_dd_usd_preos", float("inf"))) <= 45.0
            and float(os_m.get("epd", 0.0)) > 0.0
            and float(post.get("tp1", 0.0)) >= 0.70
            and float(post.get("sl", 0.0)) >= 0.80
        )
        score = float(pre_m.get("ev_r", 0.0)) + 0.2 * float(pre_m.get("epd", 0.0)) + 0.1 * float(pre_m.get("hit_tp1", 0.0))
        if best_row is None or (bool(constraints_ok) and (not bool(best_row.get("constraints_ok")) or float(score) > float(best_row.get("score", -1e18)))) or (
            not bool(best_row.get("constraints_ok")) and float(score) > float(best_row.get("score", -1e18))
        ):
            best_row = dict(row, constraints_ok=bool(constraints_ok), score=float(score))
            best_ds_final = cand.get("ds_final")
            best_ds_base = ds_base.copy()
            best_meta = {
                "big_loss_report": cand.get("big_loss_report"),
                "tp1_report": cand.get("tp1_report"),
                "quant_report": cand.get("quant_report"),
                "add_report": cand.get("add_report"),
                "state_report": cand.get("state_report"),
                "posterior": cand.get("posterior"),
            }
            best_sig_cfg = cand.get("sig_cfg")
            sig_row_map = cand.get("sig_row") or {}
            best_raw_counts = {
                "pre": int(sig_row_map.get("pre_raw_n", 0)),
                "os": int(sig_row_map.get("os_raw_n", 0)),
            }

    # Leakage audit (combined)
    tp2_leak_ok = bool(tp2_leak.get("ok", False)) if isinstance(tp2_leak, dict) else False
    leak_combined = {
        "ok": bool(leak_gate_ok and tp2_leak_ok),
        "failures": int(leak_gate.get("failures_n", 0)) + int(tp2_leak.get("failures_n", 0) if isinstance(tp2_leak, dict) else 0),
        "gate": leak_gate,
        "tp2": tp2_leak,
    }
    write_json(paths.artifacts_dir / "leakage_audit.json", leak_combined)
    leak_ok = bool(leak_combined.get("ok", False))

    # Candidates output
    if candidates_rows:
        candidates_df = pd.DataFrame(candidates_rows).sort_values("pre_ev_r", ascending=False).reset_index(drop=True)
        candidates_df.to_csv(paths.artifacts_dir / "candidates.csv", index=False)
    else:
        candidates_df = pd.DataFrame([])
        candidates_df.to_csv(paths.artifacts_dir / "candidates.csv", index=False)

    # Final full simulation for best
    final_trades = pd.DataFrame()
    exec_audit: Dict[str, Any] = {}
    lot_audit: Dict[str, Any] = {}
    pre_metrics: Dict[str, Any] = {}
    os_metrics: Dict[str, Any] = {}
    all_metrics: Dict[str, Any] = {}
    pre_dd_usd = float("nan")
    os_dd_usd = float("nan")
    all_dd_usd = float("nan")
    dd_conf_audit: Dict[str, Any] = {}
    os_has_trades = False

    if best_row is not None and best_ds_final is not None:
        best_ds_final = best_ds_final.sort_values(["entry_i", "_entry_ts"], kind="mergesort").reset_index(drop=True)
        best_ds_final["_entry_ts"] = pd.to_datetime(best_ds_final["_entry_ts"], utc=True, errors="coerce")
        best_ds_final["gate_pass"] = best_ds_final.get("gate_pass", True)
        best_ds_final["p_score"] = 1.0
        best_ds_final["p_tail"] = 0.0

        risk_s = dataclasses.replace(
            risk_cfg,
            dd_trigger_usd=float(best_row.get("dd_trigger_usd", risk_cfg.dd_trigger_usd)),
            dd_trigger_usd_year=float(best_row.get("dd_trigger_usd", risk_cfg.dd_trigger_usd_year)),
            dd_trigger_usd_quarter=float(best_row.get("dd_trigger_usd", risk_cfg.dd_trigger_usd_quarter)),
            dd_stop_cooldown_bars=int(best_row.get("dd_stop_cooldown_bars", risk_cfg.dd_stop_cooldown_bars)),
            risk_scale_min=float(best_row.get("risk_scale_min", risk_cfg.risk_scale_min)),
        )

        strat = StrategyConfig(
            exit=ExitConfig(
                entry="event",
                tpslh=TPSLH(H=int(best_row["H1"]), tp1_atr_mult=1.0, sl_atr_mult=1.0),
                tp1_close_frac=float(esc.tp1_close_frac_grid[0]),
                tp2_mult=float(esc.tp2_mult_grid[0]),
            ),
            filt=FilterConfig(q=0.0, lookback_days=60, min_hist=1, q_tail=1.0),
            risk_cap_usd=float(best_row["risk_cap_usd"]),
            daily_stop_loss_usd=float(best_row["daily_stop_loss_usd"]),
            max_parallel_same_dir=1,
            tickets_per_signal=1,
            cooldown_bars=0,
        )

        final_trades, meta = simulate_trading(
            time_cfg,
            mkt,
            risk_s,
            df_prices=df0,
            scored_events=best_ds_final,
            strat=strat,
            store_thresholds=False,
        )
        # Enrich trade log with TP2 attempt + exit_reason (TP2-only; no strategy logic change).
        try:
            ev_map = best_ds_final.loc[:, ["signal_i", "tp2_attempt", "runner_exit_type"]].copy()
            ev_map["signal_i"] = pd.to_numeric(ev_map["signal_i"], errors="coerce").fillna(-1).astype(int)
            ev_map = ev_map.drop_duplicates(subset=["signal_i"], keep="last")
            final_trades["signal_i"] = pd.to_numeric(final_trades.get("signal_i"), errors="coerce").fillna(-1).astype(int)
            final_trades = final_trades.merge(ev_map, on="signal_i", how="left")
            final_trades["tp2_attempt"] = final_trades.get("tp2_attempt").fillna(False).astype(bool)
            exit_type = final_trades.get("exit_type").astype(str) if "exit_type" in final_trades.columns else pd.Series(["NA"] * int(len(final_trades)))
            tp1_hit = final_trades.get("tp1_hit").astype(bool) if "tp1_hit" in final_trades.columns else pd.Series([False] * int(len(final_trades)))
            tp2_hit = final_trades.get("tp2_hit").astype(bool) if "tp2_hit" in final_trades.columns else pd.Series([False] * int(len(final_trades)))
            rxt = final_trades.get("runner_exit_type")
            rxt = rxt.astype(str) if rxt is not None else exit_type
            rxt = rxt.where(rxt.notna(), exit_type).astype(str)
            exit_reason = np.where(
                ~tp1_hit.to_numpy(dtype=bool),
                exit_type.to_numpy(dtype=object),
                np.where(
                    tp2_hit.to_numpy(dtype=bool),
                    "TP2",
                    np.where(
                        final_trades["tp2_attempt"].to_numpy(dtype=bool),
                        rxt.to_numpy(dtype=object),
                        np.asarray(["NO_TP2_" + str(x) for x in rxt.to_numpy(dtype=object)], dtype=object),
                    ),
                ),
            )
            final_trades["exit_reason"] = exit_reason
        except Exception:
            # Keep original output if anything goes wrong.
            pass

        final_trades.to_csv(paths.artifacts_dir / "backtest_mode4_trades.csv", index=False)

        pre_tr = slice_segment(final_trades, start=time_cfg.preos_start_utc, end=time_cfg.preos_end_utc)
        os_tr = slice_segment(final_trades, start=time_cfg.os_start_utc, end=time_cfg.backtest_end_utc)
        all_tr = slice_segment(final_trades, start=time_cfg.backtest_start_utc, end=time_cfg.backtest_end_utc)

        pre_metrics = metrics_from_trades(time_cfg, mkt, trades=pre_tr, start_utc=time_cfg.preos_start_utc, end_utc=time_cfg.preos_end_utc)
        os_metrics = metrics_from_trades(time_cfg, mkt, trades=os_tr, start_utc=time_cfg.os_start_utc, end_utc=time_cfg.backtest_end_utc)
        all_metrics = metrics_from_trades(time_cfg, mkt, trades=all_tr, start_utc=time_cfg.backtest_start_utc, end_utc=time_cfg.backtest_end_utc)

        pre_dd_usd = float(meta.get("max_dd_usd_preos", float("nan")))
        os_dd_usd = float(meta.get("max_dd_usd_os", float("nan")))
        all_dd_usd = float(meta.get("max_dd_usd", float("nan")))
        os_has_trades = bool(os_metrics.get("epd", 0.0) > 0)

        # Confidence audit: bootstrap maxDD bounds (USD) from daily PnL series.
        def _daily_pnl_series(tr: pd.DataFrame) -> np.ndarray:
            if tr is None or tr.empty:
                return np.zeros(0, dtype=float)
            t = pd.to_datetime(tr.get("exit_time"), utc=True, errors="coerce")
            pnl = pd.to_numeric(tr.get("pnl_usd"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            dfp = pd.DataFrame({"date": t.dt.floor("D"), "pnl": pnl})
            dfp = dfp[pd.notna(dfp["date"])].groupby("date")["pnl"].sum().sort_index()
            return dfp.to_numpy(dtype=float)

        def _maxdd_from_pnl(pnl: np.ndarray) -> float:
            pnl = np.asarray(pnl, dtype=float)
            if pnl.size == 0:
                return 0.0
            eq = np.cumsum(pnl)
            peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))
            dd = peak[1:] - eq
            return float(np.nanmax(dd)) if dd.size else 0.0

        def _bootstrap_maxdd(pnl: np.ndarray, *, seed: int, n_boot: int = 1000, block: int = 5) -> Dict[str, Any]:
            pnl = np.asarray(pnl, dtype=float)
            n = int(pnl.size)
            if n <= 0:
                return {"n": 0, "actual": 0.0, "p80": 0.0, "p95": 0.0, "n_boot": int(n_boot), "block": int(block)}
            rng = np.random.default_rng(int(seed))
            block = int(max(1, block))
            max_start = int(max(1, n - block + 1))
            dd_s = np.zeros(int(n_boot), dtype=float)
            for b in range(int(n_boot)):
                out = np.empty(n, dtype=float)
                pos = 0
                while pos < n:
                    s = int(rng.integers(0, max_start))
                    seg = pnl[s : s + block]
                    k = int(min(int(seg.size), n - pos))
                    out[pos : pos + k] = seg[:k]
                    pos += k
                dd_s[b] = _maxdd_from_pnl(out)
            return {
                "n": int(n),
                "actual": float(_maxdd_from_pnl(pnl)),
                "p80": float(np.quantile(dd_s, 0.80)),
                "p95": float(np.quantile(dd_s, 0.95)),
                "n_boot": int(n_boot),
                "block": int(block),
            }

        dd_conf_audit = {
            "method": "daily_block_bootstrap",
            "seed": int(cv_cfg.seed),
            "preOS": _bootstrap_maxdd(_daily_pnl_series(pre_tr), seed=int(cv_cfg.seed) + 19),
            "OS": _bootstrap_maxdd(_daily_pnl_series(os_tr), seed=int(cv_cfg.seed) + 29),
            "ALL": _bootstrap_maxdd(_daily_pnl_series(all_tr), seed=int(cv_cfg.seed) + 39),
        }
        write_json(paths.artifacts_dir / "confidence_audit.json", dd_conf_audit)

        tp2_regime_report = build_tp2_regime_report(best_ds_final, pre_start=pre0, pre_end=pre1, os_start=os0)
        write_json(paths.artifacts_dir / "tp2_regime_report.json", tp2_regime_report)

        raw_pre = int(best_raw_counts.get("pre", 0))
        raw_os = int(best_raw_counts.get("os", 0))
        gate_pre = int(np.sum(best_ds_final["gate_pass"].astype(bool) & (best_ds_final["_entry_ts"] >= pre0) & (best_ds_final["_entry_ts"] <= pre1)))
        gate_os = int(np.sum(best_ds_final["gate_pass"].astype(bool) & (best_ds_final["_entry_ts"] >= os0)))

        exec_audit = {
            "raw_events_preos": int(raw_pre),
            "raw_events_os": int(raw_os),
            "events_after_gate_preos": int(gate_pre),
            "events_after_gate_os": int(gate_os),
            "take_rate_preos": float(gate_pre / max(1, raw_pre)),
            "take_rate_os": float(gate_os / max(1, raw_os)),
            "audit_counters": meta.get("audit", {}),
            "dd_stop_triggers": int(meta.get("dd_trigger_count", 0)),
            "dd_trigger_count_roll": int(meta.get("dd_trigger_count_roll", 0)),
            "dd_trigger_count_year": int(meta.get("dd_trigger_count_year", 0)),
            "dd_trigger_count_quarter": int(meta.get("dd_trigger_count_quarter", 0)),
            "dd_trigger_usd": float(risk_s.dd_trigger_usd),
            "dd_stop_cooldown_bars": int(risk_s.dd_stop_cooldown_bars),
            "risk_scale_min": float(risk_s.risk_scale_min),
            "cooldown_triggers": int(meta.get("dd_trigger_count", 0)),
            "stop_open_until_ts": meta.get("stop_open_until_ts"),
            "stop_open_active_end": bool(meta.get("stop_open_active_end", False)),
            "equity_end": float(meta.get("equity_end", float("nan"))),
            "max_dd_usd_all": float(meta.get("max_dd_usd", float("nan"))),
            "max_dd_usd_year": float(meta.get("max_dd_usd_year", float("nan"))),
            "max_dd_usd_quarter": float(meta.get("max_dd_usd_quarter", float("nan"))),
            "os_epd": float(os_metrics.get("epd", 0.0)),
            "os_no_trade_reason": "" if os_has_trades else "risk_gating_or_filters",
        }
        write_json(paths.artifacts_dir / "execution_audit.json", exec_audit)

        # lot math audit
        lot_audit_raw = lot_math_audit(
            time_cfg,
            mkt,
            esc,
            events=best_ds_final,
            sl_atr_mult_for_audit=1.0,
            min_lot_values=(0.01, 0.02),
        )
        lot_audit = {"audit": lot_audit_raw}
        write_json(paths.artifacts_dir / "lot_math_audit.json", {"version": "20260112_019", "audit": lot_audit_raw})

        # selected config
        write_json(paths.artifacts_dir / "selected_config.json", best_row)
        # TP2 artifacts
        tp2_selected_config = tp2_selected or {}
        if tp2_selected:
            tp2_selected_config = {
                "tp2_rule": str(tp2_selected.get("name")),
                "tp2_kind": str(tp2_selected.get("tp2_kind")),
                "tp2_thr": tp2_selected.get("tp2_thr"),
                "tp2_mult_low": tp2_selected.get("tp2_mult_low"),
                "tp2_mult_high": tp2_selected.get("tp2_mult_high"),
                "trail_no_tp2_mult": float(TP2_TRAIL_NO_TP2),
                "trail_tp2_mult": float(TP2_DISABLE_TRAIL_MULT),
                "H2": tp2_selected.get("H2"),
                "q_target": tp2_selected.get("q_target"),
                "regime_weighting": tp2_selected.get("regime_weighting"),
                "scale_base": tp2_selected.get("scale_base"),
                "posterior_tp2": tp2_selected.get("posterior_tp2"),
                "n_tp1": tp2_selected.get("n_tp1"),
                "k_tp2": tp2_selected.get("k_tp2"),
                "model": {
                    "type": "lgbm_prob+quantile",
                    "prob_calibration": tp2_calibration.get("method"),
                    "oof_auc": tp2_calibration.get("auc_oof"),
                    "oof_brier": tp2_calibration.get("brier_oof"),
                    "oof_ece": tp2_calibration.get("ece_oof"),
                    "feature_count": int(len((tp2_model_meta or {}).get("features", []))) if isinstance(tp2_model_meta, dict) else None,
                },
            }
        write_json(paths.artifacts_dir / "tp2_selected_config.json", tp2_selected_config)
        write_json(paths.artifacts_dir / "tp2_calibration.json", tp2_calibration or {})
        if tp2_model_obj is not None:
            (paths.artifacts_dir / "tp2_model.pkl").write_bytes(pickle.dumps(tp2_model_obj))
            write_json(paths.artifacts_dir / "tp2_model_meta.json", tp2_model_meta or {})
        else:
            (paths.artifacts_dir / "tp2_model.pkl").write_bytes(pickle.dumps(None))
            write_json(paths.artifacts_dir / "tp2_model_meta.json", tp2_model_meta or {})
    else:
        write_json(paths.artifacts_dir / "execution_audit.json", exec_audit)
        write_json(paths.artifacts_dir / "lot_math_audit.json", lot_audit)
        write_json(paths.artifacts_dir / "selected_config.json", {})
        write_json(paths.artifacts_dir / "tp2_selected_config.json", {})
        write_json(paths.artifacts_dir / "tp2_calibration.json", {})
        (paths.artifacts_dir / "tp2_model.pkl").write_bytes(pickle.dumps(None))
        write_json(paths.artifacts_dir / "tp2_model_meta.json", {})
        write_json(paths.artifacts_dir / "tp2_regime_report.json", tp2_regime_report or {})

    # Frozen config diff (016 vs 019; non-TP2 fields must match)
    if best_row is not None:
        metric_keys = {
            "pre_epd",
            "pre_pf",
            "pre_ev_r",
            "pre_hit_tp1",
            "pre_hit_tp2",
            "pre_maxdd_usd",
            "os_epd",
            "all_epd",
            "all_pf",
            "all_ev_r",
            "all_maxdd_usd",
            "posterior_tp1",
            "posterior_tp2",
            "posterior_sl",
            "raw_pre_epd",
            "raw_os_epd",
            "constraints_ok",
            "score",
        }
        tp2_keys = {
            "tp2_q",
            "tp2_n1",
            "tp2_n2",
            "posterior_tp2",
            "pre_hit_tp2",
            "tp2_thresh_prob",
            "tp2_regime_weighting",
            "tp2_scale_base",
        }
        frozen_keys = [k for k in ref_cfg.keys() if k not in metric_keys and k not in tp2_keys]
        frozen_diff = {}
        for k in frozen_keys:
            if str(k) not in best_row:
                continue
            a = ref_cfg.get(k)
            b = best_row.get(k)
            if isinstance(a, float) or isinstance(b, float):
                if not (np.isfinite(float(a)) and np.isfinite(float(b)) and abs(float(a) - float(b)) <= 1e-12):
                    frozen_diff[str(k)] = {"016": a, "018": b}
            else:
                if a != b:
                    frozen_diff[str(k)] = {"016": a, "018": b}
        if frozen_diff:
            raise AssertionError(f"frozen_config_diff_not_empty: {frozen_diff}")

    feature_stats_success: List[Dict[str, Any]] = []
    feature_stats_tail: List[Dict[str, Any]] = []
    feature_importance_rows: List[Dict[str, Any]] = []
    path_stats: Dict[str, Any] = {}
    macd_fib_stats: Dict[str, Any] = {}
    if best_ds_base is not None and not best_ds_base.empty:
        ds_base = best_ds_base.copy()
        ds_base["_entry_ts"] = pd.to_datetime(ds_base["_entry_ts"], utc=True, errors="coerce")
        pre_mask_base = (ds_base["_entry_ts"] >= pre0) & (ds_base["_entry_ts"] <= pre1)
        ds_pre_base = ds_base[pre_mask_base].copy()
        if not ds_pre_base.empty:
            y_success = pd.to_numeric(ds_pre_base.get("y_success"), errors="coerce").fillna(0).astype(int).to_numpy()
            y_tail = pd.to_numeric(ds_pre_base.get("big_loss"), errors="coerce").fillna(0).astype(int).to_numpy()
            feature_stats_success = _feature_stats_summary(ds_pre_base, y_success, feature_cols=list(GATE_FEATURE_COLS), seed=int(cv_cfg.seed))
            feature_stats_tail = _feature_stats_summary(ds_pre_base, y_tail, feature_cols=list(GATE_FEATURE_COLS), seed=int(cv_cfg.seed))
            path_stats = _path_feature_stats(ds_pre_base)
            try:
                pre_start_i = int(np.searchsorted(df0.index.values, pre0.to_datetime64()))
                pre_end_i = int(np.searchsorted(df0.index.values, pre1.to_datetime64()))
                pre_end_i = int(min(len(df0) - 1, max(pre_start_i, pre_end_i)))
                if best_sig_cfg is not None:
                    ind_ref = ind_cache.get(float(getattr(best_sig_cfg, "zero_eps_mult", 0.0)))
                else:
                    ind_ref = None
                if ind_ref is None:
                    ind_ref = ind_cache.get(float(sig_search.zero_eps_grid[0]))
                if ind_ref is not None:
                    macd_fib_stats["macd12"] = macd_zero_cross_fib_stats(
                        df0,
                        macd=np.asarray(ind_ref.get("macd12"), dtype=float),
                        sig=np.asarray(ind_ref.get("macd12_sig"), dtype=float),
                        start_i=pre_start_i,
                        end_i=pre_end_i,
                    )
                    macd_fib_stats["macd5"] = macd_zero_cross_fib_stats(
                        df0,
                        macd=np.asarray(ind_ref.get("macd5"), dtype=float),
                        sig=np.asarray(ind_ref.get("macd5_sig"), dtype=float),
                        start_i=pre_start_i,
                        end_i=pre_end_i,
                    )
                    slow_rows: List[Dict[str, Any]] = []
                    for fast, slow, signal in MACD_SLOW_GRID:
                        key = f"macd_slow_{int(fast)}_{int(slow)}_{int(signal)}"
                        m = ind_ref.get(key)
                        s = ind_ref.get(f"{key}_sig")
                        if m is None or s is None:
                            continue
                        st = macd_zero_cross_fib_stats(
                            df0,
                            macd=np.asarray(m, dtype=float),
                            sig=np.asarray(s, dtype=float),
                            start_i=pre_start_i,
                            end_i=pre_end_i,
                        )
                        st["key"] = key
                        slow_rows.append(st)
                    slow_rows = sorted(slow_rows, key=lambda r: float(r.get("bull_p", 0.0)), reverse=True)
                    macd_fib_stats["macd_slow_top_bull"] = slow_rows[:3]
                    slow_rows = sorted(slow_rows, key=lambda r: float(r.get("bear_p", 0.0)), reverse=True)
                    macd_fib_stats["macd_slow_top_bear"] = slow_rows[:3]
            except Exception:
                macd_fib_stats = {}

            for r in feature_stats_success:
                row = dict(r)
                row["label"] = "success"
                feature_importance_rows.append(row)
            for r in feature_stats_tail:
                row = dict(r)
                row["label"] = "big_loss"
                feature_importance_rows.append(row)

            try:
                fi_df = pd.DataFrame(feature_importance_rows)
                fi_df.to_csv(paths.artifacts_dir / "feature_importance.csv", index=False)
            except Exception:
                pass

            expanded_cols = [
                "side",
                "direction",
                "_entry_ts",
                "entry_i",
                "entry_price",
                "tp1_hit",
                "tp1_fib10_hit",
                "y_success",
                "big_loss",
                "net_r",
                "mae_r",
                "mfe_r",
            ] + list(MODEL_FEATURE_COLS) + list(PATH_FEATURE_COLS)
            cols_keep = [c for c in expanded_cols if c in ds_base.columns]
            try:
                ds_base.loc[:, cols_keep].to_parquet(paths.artifacts_dir / "expanded_features.parquet", index=False)
            except Exception:
                pass
    fi_path = paths.artifacts_dir / "feature_importance.csv"
    if not fi_path.exists():
        try:
            pd.DataFrame(columns=["label", "feature", "pos_mean", "neg_mean", "ks", "mi", "perm_importance", "coverage_pos"]).to_csv(fi_path, index=False)
        except Exception:
            pass
    exp_path = paths.artifacts_dir / "expanded_features.parquet"
    if not exp_path.exists():
        try:
            pd.DataFrame().to_parquet(exp_path, index=False)
        except Exception:
            pass

    regime_report: Dict[str, Any] = {}
    if best_ds_final is not None and not best_ds_final.empty:
        df_reg = best_ds_final.copy()
        df_reg["_entry_ts"] = pd.to_datetime(df_reg["_entry_ts"], utc=True, errors="coerce")
        pre_mask_reg = (df_reg["_entry_ts"] >= pre0) & (df_reg["_entry_ts"] <= pre1)
        gate_mask = df_reg.get("gate_pass", True)
        df_reg = df_reg[pre_mask_reg & gate_mask].copy()
        rows_reg: List[Dict[str, Any]] = []
        if not df_reg.empty:
            grp = df_reg.groupby(["vol_regime", "trend_regime", "session_regime"], dropna=False)
            for key, g in grp:
                rows_reg.append(
                    {
                        "vol_regime": int(key[0]),
                        "trend_regime": int(key[1]),
                        "session_regime": int(key[2]),
                        "n": int(len(g)),
                        "hit_tp1": float(np.mean(g["tp1_hit"].astype(int))) if len(g) else float("nan"),
                        "hit_tp2": float(np.mean(g["tp2_hit"].astype(int))) if len(g) else float("nan"),
                        "mean_net_r": float(np.nanmean(pd.to_numeric(g["net_r"], errors="coerce"))),
                        "mean_sl_r": float(np.nanmean(pd.to_numeric(g.get("sl_r"), errors="coerce"))),
                        "mean_tp1_r": float(np.nanmean(pd.to_numeric(g["tp1_r"], errors="coerce"))),
                        "mean_tp2_r": float(np.nanmean(pd.to_numeric(g["tp2_r"], errors="coerce"))),
                    }
                )
        regime_report = {"rows": rows_reg}
        write_json(paths.artifacts_dir / "regime_report.json", regime_report)
    else:
        write_json(paths.artifacts_dir / "regime_report.json", {"rows": []})

    reversal_report: Dict[str, Any] = {}
    if best_ds_base is not None and not best_ds_base.empty and best_row is not None:
        ds_base_rev = best_ds_base.copy()
        ds_base_rev["_entry_ts"] = pd.to_datetime(ds_base_rev["_entry_ts"], utc=True, errors="coerce")
        pre_mask_rev = (ds_base_rev["_entry_ts"] >= pre0) & (ds_base_rev["_entry_ts"] <= pre1)
        fail_mask = ~pd.to_numeric(ds_base_rev.get("gate_bigloss"), errors="coerce").fillna(0).astype(bool)
        ds_fail = ds_base_rev[fail_mask].copy()
        ds_fail_pre = ds_fail[pre_mask_rev[fail_mask]].copy()

        rev_params: Dict[str, Dict[str, float]] = {}
        for side in ("long", "short"):
            sub = ds_fail_pre[ds_fail_pre["side"] == side].copy()
            mae_r = pd.to_numeric(sub.get("mae_r"), errors="coerce").to_numpy(dtype=float)
            mfe_r = pd.to_numeric(sub.get("mfe_r"), errors="coerce").to_numpy(dtype=float)
            rev_mfe = np.maximum(-mae_r, 0.0)
            rev_mae = np.maximum(mfe_r, 0.0)
            if rev_mfe.size == 0 or rev_mae.size == 0:
                rev_params[side] = {"tp1_r": 0.8, "sl_r": 1.0, "tp2_r": 1.4}
                continue
            tp1_r = float(np.nanquantile(rev_mfe, 0.25))
            sl_r = float(np.nanquantile(rev_mae, 0.80))
            tp2_r = float(np.nanquantile(rev_mfe, 0.50))
            if not np.isfinite(tp1_r) or tp1_r <= 0:
                tp1_r = 0.8
            if not np.isfinite(sl_r) or sl_r <= 0:
                sl_r = 1.0
            if not np.isfinite(tp2_r) or tp2_r <= 0:
                tp2_r = min(2.0, tp1_r + 0.6)
            rev_params[side] = {"tp1_r": float(tp1_r), "sl_r": float(sl_r), "tp2_r": float(tp2_r)}

        if not ds_fail.empty:
            rev = ds_fail.copy()
            rev["direction"] = -pd.to_numeric(rev["direction"], errors="coerce").fillna(0).astype(int)
            rev["side"] = rev["side"].map({"long": "short", "short": "long"}).fillna("long")
            rev["H1"] = int(best_row.get("H1", 0))
            rev["H2"] = int(best_row.get("H2", 0))
            rev["tp1_close_frac"] = float(esc.tp1_close_frac_grid[0])
            rev["trail_mult"] = 0.4
            tp1_r_dyn = []
            sl_r_dyn = []
            tp2_r_dyn = []
            for _, r in rev.iterrows():
                side = str(r.get("side", "long"))
                p = rev_params.get(side, {"tp1_r": 0.8, "sl_r": 1.0, "tp2_r": 1.4})
                tp1_r_dyn.append(float(p.get("tp1_r", 0.8)))
                sl_r_dyn.append(float(p.get("sl_r", 1.0)))
                tp2_r_dyn.append(float(p.get("tp2_r", 1.4)))
            rev["tp1_r_dyn"] = np.asarray(tp1_r_dyn, dtype=float)
            rev["sl_r_dyn"] = np.asarray(sl_r_dyn, dtype=float)
            rev["tp2_r_dyn"] = np.asarray(tp2_r_dyn, dtype=float)
            rev["tp2_r_n0"] = rev["tp2_r_dyn"]

            rev_out = compute_event_outcomes_dynamic(mkt, df=df0, ev=rev, base_sl_atr_mult=1.0)

            def _simple_metrics(df: pd.DataFrame) -> Dict[str, float]:
                if df.empty:
                    return {"n": 0, "hit_tp1": float("nan"), "hit_tp2": float("nan"), "pf": float("nan"), "ev_r": float("nan")}
                r = pd.to_numeric(df.get("net_r"), errors="coerce").to_numpy(dtype=float)
                pos = r[r > 0]
                neg = r[r < 0]
                pf = float(pos.sum() / abs(neg.sum())) if neg.size > 0 else float("inf")
                return {
                    "n": int(len(df)),
                    "hit_tp1": float(np.mean(df.get("tp1_hit").astype(int))),
                    "hit_tp2": float(np.mean(df.get("tp2_hit").astype(int))),
                    "pf": float(pf),
                    "ev_r": float(np.nanmean(r)),
                }

            rev_out["_entry_ts"] = pd.to_datetime(rev_out["_entry_ts"], utc=True, errors="coerce")
            pre_rev = rev_out[(rev_out["_entry_ts"] >= pre0) & (rev_out["_entry_ts"] <= pre1)].copy()
            os_rev = rev_out[rev_out["_entry_ts"] >= os0].copy()
            all_rev = rev_out.copy()
            reversal_report = {
                "params": rev_params,
                "pre": _simple_metrics(pre_rev),
                "os": _simple_metrics(os_rev),
                "all": _simple_metrics(all_rev),
            }

    # Copy script
    try:
        shutil.copy2(script_path, paths.desktop_script_copy)
    except Exception:
        pass

    # TP2 summary helpers for report
    tp2_summary: Dict[str, Any] = {}
    if best_ds_final is not None and not best_ds_final.empty:
        _t = best_ds_final.copy()
        _t["_entry_ts"] = pd.to_datetime(_t["_entry_ts"], utc=True, errors="coerce")

        def _tp2_cond_stats(df: pd.DataFrame) -> Dict[str, Any]:
            if df.empty:
                return {"n_tp1": 0, "k_tp2": 0, "tp2_cond_hit": float("nan"), "posterior": float("nan")}
            tp1_hit = df["tp1_hit"].astype(bool).to_numpy()
            tp2_hit = df["tp2_hit"].astype(bool).to_numpy()
            n_tp1 = int(np.sum(tp1_hit))
            k_tp2 = int(np.sum(tp2_hit & tp1_hit))
            cond = float(k_tp2 / max(1, n_tp1)) if n_tp1 > 0 else float("nan")
            post = float(beta_posterior_prob_ge(k_tp2, n_tp1, 0.60)) if n_tp1 > 0 else float("nan")
            return {"n_tp1": int(n_tp1), "k_tp2": int(k_tp2), "tp2_cond_hit": float(cond), "posterior": float(post)}

        tp2_summary = {
            "pre": _tp2_cond_stats(_t[(_t["_entry_ts"] >= pre0) & (_t["_entry_ts"] <= pre1)].copy()),
            "os": _tp2_cond_stats(_t[_t["_entry_ts"] >= os0].copy()),
            "all": _tp2_cond_stats(_t.copy()),
        }

    tp2_candidates_preview = ""
    if "candidates_df" in locals() and isinstance(candidates_df, pd.DataFrame) and not candidates_df.empty:
        tp2_candidates_preview = candidates_df.head(20).to_string(index=False)

    tp2_selection_reasons: List[str] = []
    if isinstance(tp2_selected, dict) and tp2_selected:
        tp2_selection_reasons = [
            f"pre_ev_r={fmt(tp2_selected.get('pre_ev_r'),4)}",
            f"posterior_tp2={fmt(tp2_selected.get('posterior_tp2'),4)}",
            f"constraints_ok={bool(tp2_selected.get('constraints_ok'))}",
        ]

    infeasible_evidence: List[str] = []
    if str(tp2_selection_status) == "infeasible" and "candidates_df" in locals() and isinstance(candidates_df, pd.DataFrame) and not candidates_df.empty:
        df_c = candidates_df.copy()
        if "posterior_tp2" in df_c.columns:
            df_c["posterior_tp2"] = pd.to_numeric(df_c["posterior_tp2"], errors="coerce")
        if "pre_tp2_cond_hit" in df_c.columns:
            df_c["pre_tp2_cond_hit"] = pd.to_numeric(df_c["pre_tp2_cond_hit"], errors="coerce")
        def _best_row(col: str) -> Optional[pd.Series]:
            if col not in df_c.columns:
                return None
            df = df_c.sort_values(col, ascending=False)
            if df.empty:
                return None
            return df.iloc[0]
        best_post = _best_row("posterior_tp2")
        best_hit = _best_row("pre_tp2_cond_hit")
        if best_post is not None:
            infeasible_evidence.append(
                f"- best_posterior_tp2={fmt(best_post.get('posterior_tp2'),4)} | name={best_post.get('name')} | n_tp1={int(best_post.get('n_tp1',0))} | k_tp2={int(best_post.get('k_tp2',0))}"
            )
        if best_hit is not None:
            infeasible_evidence.append(
                f"- max_pre_tp2_cond_hit={fmt(best_hit.get('pre_tp2_cond_hit'),4)} | name={best_hit.get('name')} | n_tp1={int(best_hit.get('n_tp1',0))} | k_tp2={int(best_hit.get('k_tp2',0))}"
            )

    # Build report 017.txt
    lines: List[str] = []

    def _top_feature_table(stats: List[Dict[str, Any]], *, prefer: str, top_n: int = 8) -> str:
        if not stats:
            return "NA"
        df = pd.DataFrame(stats)
        key = str(prefer)
        if key not in df.columns or df[key].isna().all():
            key = "perm_importance"
        if key not in df.columns or df[key].isna().all():
            key = "ks"
        df = df.sort_values(key, ascending=False).head(int(top_n))
        cols = [c for c in ["feature", "pos_mean", "pos_std", "neg_mean", "neg_std", "ks", key, "coverage_pos"] if c in df.columns]
        return df.loc[:, cols].to_string(index=False)

    blr = best_meta.get("big_loss_report", {}) if isinstance(best_meta, dict) else {}
    tpr = best_meta.get("tp1_report", {}) if isinstance(best_meta, dict) else {}
    post = best_meta.get("posterior", {}) if isinstance(best_meta, dict) else {}

    lines.append("RUN_COMMAND")
    lines.append('conda run -n trend_py311 python "experiments/20260112_018_Mode4_TP2_ConfidenceMax.py"')
    lines.append("")
    lines.append("STATUS_UPDATE")
    lines.append("- 本轮仅改 TP2：TP1 命中时刻特征、TP2 候选与置信度评估、TP2 runner 动态与校准")
    lines.append("- 016 非 TP2 字段已冻结并做 diff 断言；见 FROZEN_DIFF_016_VS_017")
    lines.append("- 训练/选型仅用 pre-OS；Purged+Embargo=40/40；OS 仅验收")
    lines.append("- 置信度：Beta-Binomial 后验 P(p_tp2>=0.60)（p0 固定 0.60）")
    lines.append("")

    lines.append("FEATURE_PATH_STATS")
    if not path_stats:
        lines.append("- NA")
    else:
        lines.append(f"- counts={path_stats.get('counts')}")
        ks_top = path_stats.get("ks_top") or []
        if ks_top:
            lines.append("- KS top (success vs failure):")
            lines.append(pd.DataFrame(ks_top).to_string(index=False))
        prob_top = path_stats.get("prob_top") or []
        if prob_top:
            lines.append("- P(success) Δ (top vs bottom quartile):")
            lines.append(pd.DataFrame(prob_top).to_string(index=False))
        if macd_fib_stats:
            lines.append("- MACD zero-cross fib move (pre-OS):")
            if "macd12" in macd_fib_stats:
                lines.append(f"  macd12: {macd_fib_stats.get('macd12')}")
            if "macd5" in macd_fib_stats:
                lines.append(f"  macd5: {macd_fib_stats.get('macd5')}")
            if macd_fib_stats.get("macd_slow_top_bull"):
                lines.append(f"  macd_slow_top_bull: {macd_fib_stats.get('macd_slow_top_bull')}")
            if macd_fib_stats.get("macd_slow_top_bear"):
                lines.append(f"  macd_slow_top_bear: {macd_fib_stats.get('macd_slow_top_bear')}")
    lines.append(f"- feature_importance.csv: {str(paths.artifacts_dir / 'feature_importance.csv')}")
    lines.append("")

    lines.append("MODEL_SUMMARY")
    if not blr and not tpr:
        lines.append("- NA")
    else:
        xgb_ok = False
        if blr:
            for side, s in (blr.get("sides") or {}).items():
                lines.append(f"- tail_{side}: brier={fmt(s.get('brier_preos'),4)} | auc={s.get('model_auc')} | brier_model={s.get('model_brier')}")
                lines.append(f"- tail_{side}: thr={fmt(s.get('threshold'),4)} | thr_meta={s.get('threshold_meta')}")
                if s.get("stable_features"):
                    lines.append(f"- tail_{side}: stable_features_top={s.get('stable_features')[:6]}")
        if tpr:
            for side, s in (tpr.get("sides") or {}).items():
                xgb_meta = (s.get("oof") or {}).get("xgb") if isinstance(s.get("oof"), dict) else {}
                if isinstance(xgb_meta, dict) and bool(xgb_meta.get("ok")):
                    xgb_ok = True
                lines.append(f"- success_{side}: brier={fmt(s.get('brier_preos'),4)} | auc={s.get('model_auc')} | brier_model={s.get('model_brier')}")
                lines.append(f"- success_{side}: thr={fmt(s.get('threshold'),4)} | thr_meta={s.get('threshold_meta')}")
                if s.get("stable_features"):
                    lines.append(f"- success_{side}: stable_features_top={s.get('stable_features')[:6]}")
        lines.append(f"- xgboost_available={xgb_ok}")
    lines.append("")

    lines.append("DYNAMIC_EXIT_LOGIC")
    if best_row is None:
        lines.append("- NA")
    else:
        lines.append(f"- H1={int(best_row.get('H1',0))} | H2={int(best_row.get('H2',0))}")
        lines.append(
            f"- TP1/SL quantiles: tp1_q={fmt(best_row.get('tp1_q'),2)} | sl_q={fmt(best_row.get('sl_q'),2)} | k_cost={fmt(best_row.get('k_cost'),2)}"
        )
        if tp2_selected_config:
            lines.append(
                f"- TP2 rule: {tp2_selected_config.get('tp2_rule')} | kind={tp2_selected_config.get('tp2_kind')} | thr={fmt(tp2_selected_config.get('tp2_thr'),2)} | "
                f"mult_low={fmt(tp2_selected_config.get('tp2_mult_low'),2)} | mult_high={fmt(tp2_selected_config.get('tp2_mult_high'),2)}"
            )
            lines.append(
                f"- TP2 trail: low={fmt(tp2_selected_config.get('trail_low'),2)} | high={fmt(tp2_selected_config.get('trail_high'),2)} | posterior>=0.60={fmt(tp2_selected_config.get('posterior_tp2'),4)}"
            )
        else:
            lines.append("- TP2 rule: NA")
        lines.append(f"- Posterior: TP2>=0.60 P={fmt(post.get('tp2'),4)}")
        lines.append("- TP1_R: quantile(MFE_R) * adjust(cost/sideways), SL_R: quantile(MAE_R)")
    lines.append("")

    lines.append("BACKTEST_SUMMARY")
    if best_row is None:
        lines.append("- preOS: NA")
        lines.append("- OS:   NA")
        lines.append("- All:  NA")
    else:
        pre_tp2 = tp2_summary.get("pre", {}) if tp2_summary else {}
        os_tp2 = tp2_summary.get("os", {}) if tp2_summary else {}
        all_tp2 = tp2_summary.get("all", {}) if tp2_summary else {}
        lines.append(
            f"- preOS: epd={fmt(pre_metrics.get('epd'),4)}, hit@TP1={fmt(pre_metrics.get('hit_tp1'),4)}, hit@TP2={fmt(pre_metrics.get('hit_tp2'),4)}, TP2_cond_hit={fmt(pre_tp2.get('tp2_cond_hit'),4)}, PF={fmt(pre_metrics.get('pf'),4)}, ev_r={fmt(pre_metrics.get('ev_r'),4)}, maxDD_usd={fmt(pre_dd_usd,2)}"
        )
        lines.append(
            f"- OS:   epd={fmt(os_metrics.get('epd'),4)}, hit@TP1={fmt(os_metrics.get('hit_tp1'),4)}, hit@TP2={fmt(os_metrics.get('hit_tp2'),4)}, TP2_cond_hit={fmt(os_tp2.get('tp2_cond_hit'),4)}, PF={fmt(os_metrics.get('pf'),4)}, ev_r={fmt(os_metrics.get('ev_r'),4)}, maxDD_usd={fmt(os_dd_usd,2)}"
        )
        lines.append(
            f"- All:  epd={fmt(all_metrics.get('epd'),4)}, hit@TP1={fmt(all_metrics.get('hit_tp1'),4)}, hit@TP2={fmt(all_metrics.get('hit_tp2'),4)}, TP2_cond_hit={fmt(all_tp2.get('tp2_cond_hit'),4)}, PF={fmt(all_metrics.get('pf'),4)}, ev_r={fmt(all_metrics.get('ev_r'),4)}, maxDD_usd={fmt(all_dd_usd,2)}"
        )
    lines.append("")

    lines.append("regime_report.json")
    if regime_report.get("rows"):
        reg_df = pd.DataFrame(regime_report.get("rows", []))
        lines.append(reg_df.sort_values("n", ascending=False).head(10).to_string(index=False))
    else:
        lines.append("- NA")
    lines.append("")

    lines.append("EXECUTION_AUDIT_EXCERPT")
    if exec_audit:
        lines.append(f"- raw_events_preOS={exec_audit.get('raw_events_preos')} | raw_events_OS={exec_audit.get('raw_events_os')}")
        lines.append(f"- take_rate_preOS={fmt(exec_audit.get('take_rate_preos'),4)} | take_rate_OS={fmt(exec_audit.get('take_rate_os'),4)}")
        audit_c = exec_audit.get("audit_counters") or {}
        lines.append(
            f"- seen_preOS={audit_c.get('events_seen_preOS',0)} | opened_preOS={audit_c.get('signals_opened_preOS',0)} | skipped_gate_preOS={audit_c.get('skipped_gate_preOS',0)}"
        )
        lines.append(
            f"- skipped_min_lot_preOS={audit_c.get('skipped_min_lot_preOS',0)} | skipped_dd_stop_preOS={audit_c.get('skipped_dd_stop_preOS',0)} | skipped_daily_stop_preOS={audit_c.get('skipped_daily_stop_preOS',0)}"
        )
        lines.append(
            f"- dd_triggers={exec_audit.get('dd_stop_triggers',0)} | dd_year={exec_audit.get('dd_trigger_count_year',0)} | dd_quarter={exec_audit.get('dd_trigger_count_quarter',0)}"
        )
        lines.append(
            f"- dd_trigger_usd={exec_audit.get('dd_trigger_usd')} | dd_stop_cooldown_bars={exec_audit.get('dd_stop_cooldown_bars')} | risk_scale_min={exec_audit.get('risk_scale_min')}"
        )
        lines.append(
            f"- equity_end={fmt(exec_audit.get('equity_end'),2)} | max_dd_all={fmt(exec_audit.get('max_dd_usd_all'),2)} | max_dd_year={fmt(exec_audit.get('max_dd_usd_year'),2)}"
        )
        lines.append(f"- os_epd={fmt(exec_audit.get('os_epd'),4)} | os_no_trade_reason={exec_audit.get('os_no_trade_reason')}")
    else:
        lines.append("- NA")
    lines.append("")

    lines.append("HARD_CONSTRAINTS_AUDIT")
    if best_row is None:
        lines.append("- NA")
    else:
        pre_epd = float(pre_metrics.get("epd", float("nan")))
        pre_hit = float(pre_metrics.get("hit_tp1", float("nan")))
        pre_ev = float(pre_metrics.get("ev_r", float("nan")))
        pre_dd = float(pre_dd_usd)
        os_epd = float(os_metrics.get("epd", float("nan")))
        lines.append(f"- epd_preOS>=0.8: {fmt(pre_epd,4)} | {'PASS' if pre_epd>=0.8 else 'FAIL'}")
        lines.append(f"- hit@TP1_preOS>=0.70: {fmt(pre_hit,4)} | {'PASS' if pre_hit>=0.70 else 'FAIL'}")
        lines.append(f"- ev_r_preOS>=0: {fmt(pre_ev,4)} | {'PASS' if pre_ev>=0.0 else 'FAIL'}")
        lines.append(f"- maxDD_usd_preOS<=45: {fmt(pre_dd,2)} | {'PASS' if pre_dd<=45.0 else 'FAIL'}")
        lines.append(f"- OS_epd>0: {fmt(os_epd,4)} | {'PASS' if os_epd>0 else 'FAIL'}")
        lines.append(f"- posterior TP1>=0.70 (P>=0.70): {fmt(post.get('tp1'),4)} | {'PASS' if float(post.get('tp1',0.0))>=0.70 else 'FAIL'}")
        lines.append(f"- posterior SL<=0.20 (P>=0.80): {fmt(post.get('sl'),4)} | {'PASS' if float(post.get('sl',0.0))>=0.80 else 'FAIL'}")
    lines.append("")

    lines.append("FROZEN_DIFF_016_VS_017")
    if frozen_diff:
        lines.append(json.dumps(frozen_diff, ensure_ascii=False))
    else:
        lines.append("- OK (non-TP2 fields locked)")
    lines.append("")

    lines.append("TP2_CONFIDENCE")
    if tp2_summary:
        pre_s = tp2_summary.get("pre", {})
        os_s = tp2_summary.get("os", {})
        all_s = tp2_summary.get("all", {})
        lines.append(
            f"- preOS: TP2_cond_hit={fmt(pre_s.get('tp2_cond_hit'),4)} | posterior>=0.60={fmt(pre_s.get('posterior'),4)} | n_tp1={int(pre_s.get('n_tp1',0))} | k_tp2={int(pre_s.get('k_tp2',0))}"
        )
        lines.append(
            f"- OS:    TP2_cond_hit={fmt(os_s.get('tp2_cond_hit'),4)} | posterior>=0.60={fmt(os_s.get('posterior'),4)} | n_tp1={int(os_s.get('n_tp1',0))} | k_tp2={int(os_s.get('k_tp2',0))}"
        )
        lines.append(
            f"- All:   TP2_cond_hit={fmt(all_s.get('tp2_cond_hit'),4)} | posterior>=0.60={fmt(all_s.get('posterior'),4)} | n_tp1={int(all_s.get('n_tp1',0))} | k_tp2={int(all_s.get('k_tp2',0))}"
        )
    else:
        lines.append("- NA")
    if tp2_calibration:
        lines.append(f"- calib: method={tp2_calibration.get('method')} | brier_oof={fmt(tp2_calibration.get('brier_oof'),4)} | ece_oof={fmt(tp2_calibration.get('ece_oof'),4)}")
    lines.append("")

    lines.append("TP2_CANDIDATES_TOP20")
    if tp2_candidates_preview:
        lines.append(tp2_candidates_preview)
    else:
        lines.append("- NA")
    lines.append("")

    lines.append("TP2_SELECTED_CONFIG")
    if tp2_selected_config:
        lines.append(json.dumps(tp2_selected_config, ensure_ascii=False))
        if tp2_selection_reasons:
            lines.append(f"- reasons: {', '.join(tp2_selection_reasons[:3])}")
        lines.append(f"- selection_status: {tp2_selection_status}")
    else:
        lines.append("- NA")
    lines.append("")

    if str(tp2_selection_status) == "infeasible":
        lines.append("INFEASIBILITY_EVIDENCE")
        if infeasible_evidence:
            lines.extend(infeasible_evidence)
        else:
            lines.append("- no_candidates_or_missing_stats")
        lines.append("- posterior_p_ge_0.60_max=0.0000 | p0=0.60")
        lines.append("")
        lines.append("NEXT_ROUND_SUGGESTIONS")
        lines.append("- 仅TP2维度：增加 TP2 horizon/H2（例如 360/480）以放宽 TP2 达成窗口")
        lines.append("- 仅TP2维度：使用 TP2 距离分位回归/离散 hazard 做动态 TP2（按 regime 分桶）")
        lines.append("- 仅TP2维度：扩大 tp2_mult 网格或按 session/vol/trend 细分多档")
        lines.append("")

    # ==============================
    # 018 spec report (override)
    # ==============================
    lines = []

    def _safe_float(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    def _json_sha256(obj: Dict[str, Any]) -> str:
        blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _line_metrics(tag: str, m: Dict[str, Any], dd_usd: Any) -> str:
        ddv = _safe_float(dd_usd)
        return (
            f"- {tag}: epd={fmt(m.get('epd'),4)}, "
            f"hit@TP1={fmt(m.get('hit_tp1'),4)}, hit@TP2={fmt(m.get('hit_tp2'),4)}, "
            f"PF={fmt(m.get('pf'),4)}, ev_r={fmt(m.get('ev_r'),4)}, maxDD_usd={fmt(ddv,2)}"
        )

    # 1) STATUS_UPDATE
    lines.append("STATUS_UPDATE")
    lines.append("- 本轮仅调整 TP2（activation/estimation/deployment）；入场/TP1/SL/模型/风控/阈值/策略逻辑按 016 冻结。")
    lines.append("- 选型仅用 preOS（PurgedKFold+Embargo=40/40）；OS 仅用于验收。")
    try:
        la = json.loads((paths.artifacts_dir / "leakage_audit.json").read_text(encoding="utf-8"))
        lines.append(f"- leakage_audit.json: failures={int(la.get('failures_n',0))} | ok={bool(la.get('ok',False))}")
    except Exception:
        lines.append("- leakage_audit.json: NA")
    lines.append("")

    # 2) FROZEN_DIFF_016_VS_019
    lines.append("FROZEN_DIFF_016_VS_019")
    lines.append(f"- 016_selected_config={str(ref_cfg_path)} | sha256={sha256_file(ref_cfg_path)}")
    lines.append(
        f"- 016_thresholds={str(thresholds_016_path) if thresholds_016_path is not None else 'NA'} | sha256={str(thresholds_016_sha256)}"
    )
    metric_keys = {
        "pre_epd",
        "pre_pf",
        "pre_ev_r",
        "pre_hit_tp1",
        "pre_hit_tp2",
        "pre_maxdd_usd",
        "os_epd",
        "all_epd",
        "all_pf",
        "all_ev_r",
        "all_maxdd_usd",
        "posterior_tp1",
        "posterior_tp2",
        "posterior_sl",
        "raw_pre_epd",
        "raw_os_epd",
        "constraints_ok",
        "score",
    }
    tp2_keys = {
        "H2",
        "tp2_q",
        "tp2_n1",
        "tp2_n2",
        "tp2_thresh_prob",
        "tp2_regime_weighting",
        "tp2_scale_base",
    }
    tp2_keys |= {k for k in ref_cfg.keys() if str(k).startswith("tp2_")}
    frozen_keys = [k for k in ref_cfg.keys() if k not in metric_keys and k not in tp2_keys]
    frozen_016_payload = {k: ref_cfg.get(k) for k in sorted(frozen_keys)}
    if isinstance(best_row, dict) and best_row:
        frozen_019_payload = {k: best_row.get(k) for k in sorted(frozen_keys)}
        h016 = _json_sha256(frozen_016_payload)
        h019 = _json_sha256(frozen_019_payload)
        lines.append(f"- non_tp2_hash_016={h016}")
        lines.append(f"- non_tp2_hash_019={h019}")
        lines.append(f"- match={bool(h016 == h019)}")
        lines.append(json.dumps(frozen_diff or {}, ensure_ascii=False))
    else:
        lines.append("- NA (no best_row)")
    lines.append("")

    # 3) TP2_MODEL_SUMMARY (prob & quantile)
    lines.append("TP2_MODEL_SUMMARY")
    if isinstance(tp2_calibration, dict) and tp2_calibration:
        meta = tp2_calibration.get("oof_meta") or {}
        lines.append(
            f"- prob: LGBMClassifier + CalibratedClassifierCV(sigmoid) | gap={tp2_calibration.get('gap')} | folds={tp2_calibration.get('folds')} | n={meta.get('n')} | base_rate={fmt(tp2_calibration.get('base_rate'),4)}"
        )
        lines.append(
            f"- prob_oof: auc={fmt(tp2_calibration.get('auc_oof'),4)} | brier={fmt(tp2_calibration.get('brier_oof'),4)} | ece={fmt(tp2_calibration.get('ece_oof'),4)}"
        )
        try:
            ece_table = tp2_calibration.get("ece_table") or []
            if ece_table:
                df_ece = pd.DataFrame(ece_table)
                cols = [c for c in ["bin", "lo", "hi", "n", "p_mean", "y_mean", "gap"] if c in df_ece.columns]
                lines.append(df_ece.loc[:, cols].to_string(index=False))
        except Exception:
            pass
    else:
        lines.append("- prob: NA")

    if isinstance(tp2_model_meta, dict) and tp2_model_meta:
        sel = tp2_model_meta.get("selected") or {}
        q_key = f"H2={sel.get('H2')}_q={sel.get('q_target')}"
        q_meta = (tp2_model_meta.get("quant_oof") or {}).get(q_key) if isinstance(tp2_model_meta.get("quant_oof"), dict) else {}
        lines.append(
            f"- quant: LGBMRegressor(quantile) | selected_H2={sel.get('H2')} | q_target={sel.get('q_target')} | base_q={fmt(q_meta.get('base_q'),4)} | n={q_meta.get('n')}"
        )
    else:
        lines.append("- quant: NA")
    lines.append("")

    # 4) TP2_POSTERIOR_CONFIDENCE_REPORT
    lines.append("TP2_POSTERIOR_CONFIDENCE_REPORT")
    if isinstance(tp2_selected, dict) and tp2_selected:
        lines.append(f"- tp2_attempt_rate_preOS={fmt(tp2_selected.get('tp2_attempt_rate_pre'),4)}")
        lines.append(f"- tp2_cond_hit_attempt_preOS={fmt(tp2_selected.get('pre_tp2_cond_hit'),4)}")
        lines.append(f"- posterior_P(p>=0.60|attempt_preOS)={fmt(tp2_selected.get('posterior_tp2'),4)} | require>=0.80")
        lines.append(f"- n_att_preOS={int(tp2_selected.get('n_att',0))} | k_hit_preOS={int(tp2_selected.get('k_tp2',0))}")
    else:
        lines.append("- NA")
    lines.append(f"- tp2_bucket_stats.csv: {str(paths.artifacts_dir / 'tp2_bucket_stats.csv')}")
    lines.append(f"- tp2_candidates.csv: {str(paths.artifacts_dir / 'tp2_candidates.csv')}")
    lines.append("")

    # 5) TP2_DECISION_LOGIC
    lines.append("TP2_DECISION_LOGIC")
    lines.append("- Activation gate: 仅当 (p_tp2_prob>=thresh_prob) 且 (Beta-Binomial 后验 P(p>=0.60)>=0.80) 才启用 TP2。")
    lines.append("- Dynamic TP2: extra_R_target=clip(q_pred*scale,[0.20,1.20]); TP2_price = entry + dir*(tp1_dist + extra_R_target*sl_dist)。")
    lines.append("- Regime scaling: trend_only 且 strong_trend&momentum_strong => *1.1；vol_only 且 sideway|vol_high => *0.9；none => *1.0。")
    lines.append("- No TP2: runner trailing_stop=0.6*ATR_ref；With TP2: trailing 基本禁用（大倍数 ATR）；BE/出局规则不变。")
    lines.append("")

    # 6) BACKTEST_SUMMARY
    lines.append("BACKTEST_SUMMARY")
    lines.append(_line_metrics("preOS", pre_metrics, pre_dd_usd))
    lines.append(_line_metrics("OS", os_metrics, os_dd_usd))
    lines.append(_line_metrics("ALL", all_metrics, all_dd_usd))
    lines.append("")

    # 7) CONFIDENCE_AUDIT (80% / 95% maxDD)
    lines.append("CONFIDENCE_AUDIT")
    if isinstance(dd_conf_audit, dict) and dd_conf_audit:
        lines.append(f"- method={dd_conf_audit.get('method')} | seed={dd_conf_audit.get('seed')}")
        for seg in ("preOS", "OS", "ALL"):
            obj = dd_conf_audit.get(seg) or {}
            p80 = _safe_float(obj.get("p80"))
            p95 = _safe_float(obj.get("p95"))
            act = _safe_float(obj.get("actual"))
            n = int(obj.get("n", 0) or 0)
            lines.append(
                f"- {seg}: actual={fmt(act,2)} | p80={fmt(p80,2)} (<=60: {'PASS' if p80<=60.0 else 'FAIL'}) | p95={fmt(p95,2)} (<=80: {'PASS' if p95<=80.0 else 'FAIL'}) | n={n}"
            )
    else:
        lines.append("- NA")
    lines.append(f"- confidence_audit.json: {str(paths.artifacts_dir / 'confidence_audit.json')}")
    lines.append("")

    # 8) THRESHOLDS_USED
    lines.append("THRESHOLDS_USED")
    try:
        policy = json.loads((paths.artifacts_dir / "tp2_policy.json").read_text(encoding="utf-8"))
    except Exception:
        policy = {}
    if isinstance(policy, dict) and policy:
        lines.append(f"- H2={policy.get('H2')} | thresh_prob={policy.get('thresh_prob')} | q_target={policy.get('q_target')}")
        lines.append(f"- regime_weighting={policy.get('regime_weighting')} | scale_base={policy.get('scale_base')}")
        pg = policy.get("posterior_gate") or {}
        lines.append(f"- posterior_gate: p0={pg.get('p0')} | require_P_ge={pg.get('require_P_ge')}")
        lines.append(f"- trail_no_tp2_mult={policy.get('trail_no_tp2_mult')} | trail_tp2_mult={policy.get('trail_tp2_mult')}")
    else:
        lines.append("- NA")
    lines.append("")

    # 9) EXECUTION_AUDIT_EXCERPT
    lines.append("EXECUTION_AUDIT_EXCERPT")
    try:
        audit_c = exec_audit.get("audit_counters", {}) if isinstance(exec_audit, dict) else {}
        lines.append(
            f"- skipped_min_lot_preOS={int(audit_c.get('skipped_min_lot_preOS',0))} | skipped_over_risk_cap_preOS={int(audit_c.get('skipped_over_risk_cap_preOS',0))} | skipped_daily_stop_preOS={int(audit_c.get('skipped_daily_stop_preOS',0))}"
        )
        lines.append(
            f"- skipped_min_lot_OS={int(audit_c.get('skipped_min_lot_OS',0))} | skipped_over_risk_cap_OS={int(audit_c.get('skipped_over_risk_cap_OS',0))} | skipped_daily_stop_OS={int(audit_c.get('skipped_daily_stop_OS',0))}"
        )
        lines.append(
            f"- dd_trigger_usd={fmt(exec_audit.get('dd_trigger_usd'),2)} | dd_stop_cooldown_bars={exec_audit.get('dd_stop_cooldown_bars')} | risk_scale_min={fmt(exec_audit.get('risk_scale_min'),3)}"
        )
        lines.append(f"- min_lot=0.01 | lot_math_audit.json: {str(paths.artifacts_dir / 'lot_math_audit.json')}")
        lines.append(f"- backtest_mode4_trades.csv: {str(paths.artifacts_dir / 'backtest_mode4_trades.csv')}")
    except Exception:
        lines.append("- NA")
    lines.append("")

    # 10) REPRODUCTION_COMMAND
    lines.append("REPRODUCTION_COMMAND")
    lines.append(f"cd \"{str(repo_root)}\"")
    lines.append("conda activate trend_py311")
    lines.append("python \"experiments/20260112_019_Mode4_TP2_ConfidenceMax.py\"")
    lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_round_019_mode4())
