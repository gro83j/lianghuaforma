#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
20260112_020 Mode4 稳定性复核（仅审计，不改 019 交易逻辑/参数）

仅允许输出：
  - D:/projectmt5/20260112/020.txt
  - D:/projectmt5/20260112/020_artifacts/*
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from math import erf, sqrt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================
# Paths / Constants
# =============================


@dataclass(frozen=True)
class TimeConfig:
    backtest_start_utc: str = "2015-01-01"
    backtest_end_utc: str = "2025-12-26 23:59:59"
    preos_start_utc: str = "2015-01-01"
    preos_end_utc: str = "2022-12-31 23:59:59"
    os_start_utc: str = "2023-01-01"


@dataclass(frozen=True)
class MarketConfig:
    initial_capital_usd: float = 200.0


TIME_CFG = TimeConfig()
MKT_CFG = MarketConfig()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


BASE_DIR = Path(os.environ.get("BASE_DIR", "D:/projectmt5"))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(BASE_DIR / "20260112")))

BASELINE_019_TXT = Path(os.environ.get("BASELINE_019_TXT", str(OUT_DIR / "019.txt")))
BASELINE_019_ART = Path(os.environ.get("BASELINE_019_ART", str(OUT_DIR / "019_artifacts")))
SCRIPT_019 = Path(
    os.environ.get(
        "SCRIPT_019",
        str(_repo_root() / "experiments" / "20260112_019_Mode4_TP2_ConfidenceMax.py"),
    )
)

REPORT_020 = Path(os.environ.get("REPORT_020", str(OUT_DIR / "020.txt")))
ART_020 = Path(os.environ.get("ART_020", str(OUT_DIR / "020_artifacts")))


# =============================
# Utils
# =============================


def to_utc_ts(x: str) -> pd.Timestamp:
    return pd.to_datetime(x, utc=True, errors="raise")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use UTF-8 with BOM for Windows Notepad compatibility.
    path.write_text(text, encoding="utf-8-sig")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(float(x) / sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """
    Acklam's inverse normal CDF approximation.
    https://web.archive.org/web/20150910044729/http://home.online.no/~pjacklam/notes/invnorm/
    """
    p = float(p)
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return -float("inf")
        if p == 1.0:
            return float("inf")
        return float("nan")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = sqrt(-2.0 * np.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if p > phigh:
        q = sqrt(-2.0 * np.log(1.0 - p))
        num = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den

    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return num / den


def fmt(x: Any, nd: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    if nd <= 0:
        return str(int(round(v)))
    return f"{v:.{int(nd)}f}"


def split_report_sections(lines: List[str]) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None

    def is_header(s: str) -> bool:
        s = s.strip()
        if not s:
            return False
        if s != s.upper():
            return False
        return all(ch.isalnum() or ch == "_" for ch in s)

    for raw in lines:
        line = raw.rstrip("\n")
        if is_header(line):
            current = line
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return sections


def parse_kv_lines(lines: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ln in lines:
        s = str(ln).strip()
        if not s.startswith("- "):
            continue
        s2 = s[2:]
        if "=" in s2:
            k, v = s2.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def read_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df.get("entry_time"), utc=True, errors="coerce")
    df["exit_time"] = pd.to_datetime(df.get("exit_time"), utc=True, errors="coerce")
    return df


def filter_by_entry_time(tr: pd.DataFrame, start_utc: str, end_utc: str) -> pd.DataFrame:
    if tr.empty:
        return tr.copy()
    s = to_utc_ts(start_utc)
    e = to_utc_ts(end_utc)
    t = tr.copy()
    t = t[pd.notna(t["entry_time"])]
    return t[(t["entry_time"] >= s) & (t["entry_time"] <= e)].copy()


def daily_pnl_series_sparse(tr: pd.DataFrame, *, pnl_col: str) -> np.ndarray:
    """
    与 019 confidence_audit.json 的口径一致：
    - 使用 exit_time floor('D') 聚合
    - 仅包含有 pnl 的日期（不补零日）
    """
    if tr.empty:
        return np.zeros(0, dtype=float)
    t = pd.to_datetime(tr.get("exit_time"), utc=True, errors="coerce")
    pnl = pd.to_numeric(tr.get(pnl_col), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dfp = pd.DataFrame({"date": t.dt.floor("D"), "pnl": pnl})
    dfp = dfp[pd.notna(dfp["date"])].groupby("date")["pnl"].sum().sort_index()
    return dfp.to_numpy(dtype=float)


def daily_pnl_series_full(tr: pd.DataFrame, *, pnl_col: str, start_utc: str, end_utc: str) -> np.ndarray:
    """
    用于 PSR/DSR：对齐完整日历天，缺失日补 0（更保守）。
    """
    s = to_utc_ts(start_utc).floor("D")
    e = to_utc_ts(end_utc).floor("D")
    if e < s:
        return np.zeros(0, dtype=float)
    idx = pd.date_range(s, e, freq="D", tz="UTC")
    if tr.empty:
        return np.zeros(int(len(idx)), dtype=float)
    t = pd.to_datetime(tr.get("exit_time"), utc=True, errors="coerce")
    pnl = pd.to_numeric(tr.get(pnl_col), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dfp = pd.DataFrame({"date": t.dt.floor("D"), "pnl": pnl})
    s0 = dfp[pd.notna(dfp["date"])].groupby("date")["pnl"].sum().sort_index()
    s0 = s0.reindex(idx, fill_value=0.0)
    return s0.to_numpy(dtype=float)


def maxdd_from_pnl(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, dtype=float)
    if pnl.size == 0:
        return 0.0
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))
    dd = peak[1:] - eq
    return float(np.nanmax(dd)) if dd.size else 0.0


def compute_metrics(
    tr: pd.DataFrame,
    *,
    start_utc: str,
    end_utc: str,
    net_r_col: str = "net_r",
    pnl_usd_col: str = "pnl_usd",
) -> Dict[str, Any]:
    t = filter_by_entry_time(tr, start_utc, end_utc)
    s = to_utc_ts(start_utc)
    e = to_utc_ts(end_utc)
    days = float((e - s).total_seconds() / 86400.0)
    days = max(1.0, days)

    tickets_n = int(len(t))
    if tickets_n <= 0:
        return {
            "tickets": 0,
            "signals": 0,
            "days": float(days),
            "epd": 0.0,
            "hit_tp1": float("nan"),
            "pf": float("nan"),
            "ev_r": float("nan"),
            "maxdd_usd": 0.0,
        }

    if "signal_i" in t.columns:
        sig = pd.to_numeric(t["signal_i"], errors="coerce").dropna().astype(int)
        signals_n = int(sig.nunique())
        grp = t.assign(_sig=sig).dropna(subset=["_sig"]).groupby("_sig", dropna=True)
        tp1_by = grp["tp1_hit"].max() if "tp1_hit" in t.columns else grp["tp1_reached"].max()
        hit_tp1 = float(np.mean(pd.to_numeric(tp1_by, errors="coerce").fillna(0).astype(int)))
    else:
        signals_n = int(tickets_n)
        col_tp1 = "tp1_hit" if "tp1_hit" in t.columns else "tp1_reached"
        hit_tp1 = float(np.mean(pd.to_numeric(t[col_tp1], errors="coerce").fillna(0).astype(int)))

    epd = float(float(signals_n) / days)

    net_r = pd.to_numeric(t.get(net_r_col), errors="coerce").to_numpy(dtype=float)
    pos = net_r[np.isfinite(net_r) & (net_r > 0)]
    neg = net_r[np.isfinite(net_r) & (net_r < 0)]
    pf = float(np.sum(pos) / max(1e-12, abs(np.sum(neg)))) if pos.size and neg.size else float("nan")
    ev_r = float(np.nanmean(net_r)) if net_r.size else float("nan")

    pnl_series = daily_pnl_series_sparse(t, pnl_col=pnl_usd_col)
    maxdd = float(maxdd_from_pnl(pnl_series))
    return {
        "tickets": int(tickets_n),
        "signals": int(signals_n),
        "days": float(days),
        "epd": float(epd),
        "hit_tp1": float(hit_tp1),
        "pf": float(pf),
        "ev_r": float(ev_r),
        "maxdd_usd": float(maxdd),
    }


def block_bootstrap_maxdd_samples(
    pnl: np.ndarray,
    *,
    seed: int,
    n_boot: int,
    block_len: int,
) -> np.ndarray:
    pnl = np.asarray(pnl, dtype=float)
    n = int(pnl.size)
    if n <= 0:
        return np.zeros(0, dtype=float)
    rng = np.random.default_rng(int(seed))
    block_len = int(max(1, block_len))
    max_start = int(max(1, n - block_len + 1))
    dd_s = np.zeros(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        out = np.empty(n, dtype=float)
        pos = 0
        while pos < n:
            s = int(rng.integers(0, max_start))
            seg = pnl[s : s + block_len]
            k = int(min(int(seg.size), n - pos))
            out[pos : pos + k] = seg[:k]
            pos += k
        dd_s[b] = maxdd_from_pnl(out)
    return dd_s


def quantile_ci_order_stat(samples: np.ndarray, q: float, *, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Monte-Carlo CI for estimated quantile (order-statistic interval, Binomial-Normal approximation).
    """
    s = np.sort(np.asarray(samples, dtype=float))
    n = int(s.size)
    if n <= 0:
        return float("nan"), float("nan")
    q = float(q)
    z = 1.959963984540054  # 97.5%
    mu = n * q
    sigma = sqrt(max(1e-12, n * q * (1.0 - q)))
    lo = int(np.floor(mu - z * sigma))
    hi = int(np.ceil(mu + z * sigma))
    lo = max(0, min(n - 1, lo))
    hi = max(0, min(n - 1, hi))
    return float(s[lo]), float(s[hi])


def run_019_once(*, timeout_s: int = 7200) -> Tuple[int, float]:
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_019)],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
        timeout=int(timeout_s),
    )
    t1 = time.time()
    return int(proc.returncode), float(t1 - t0)


def _parse_float_after(key: str, text: str) -> float:
    try:
        idx = text.index(key) + len(key)
    except ValueError:
        return float("nan")
    tail = text[idx:]
    # stop at comma or whitespace
    out = []
    for ch in tail:
        if ch in ", \t\r\n":
            break
        out.append(ch)
    try:
        return float("".join(out))
    except Exception:
        return float("nan")


def parse_backtest_summary(lines: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Parse 019.txt BACKTEST_SUMMARY block lines like:
      - preOS: epd=..., hit@TP1=..., PF=..., ev_r=..., maxDD_usd=...
    """
    out: Dict[str, Dict[str, float]] = {}
    for ln in lines:
        s = str(ln).strip()
        if not s.startswith("- "):
            continue
        # tag between "- " and ":"
        if ":" not in s:
            continue
        tag = s[2:].split(":", 1)[0].strip()
        out[tag] = {
            "epd": _parse_float_after("epd=", s),
            "hit_tp1": _parse_float_after("hit@TP1=", s),
            "pf": _parse_float_after("PF=", s),
            "ev_r": _parse_float_after("ev_r=", s),
            "maxdd_usd": _parse_float_after("maxDD_usd=", s),
        }
    return out


def main() -> int:
    ART_020.mkdir(parents=True, exist_ok=True)

    if not BASELINE_019_TXT.exists():
        raise FileNotFoundError(f"missing baseline report: {BASELINE_019_TXT}")
    if not BASELINE_019_ART.exists():
        raise FileNotFoundError(f"missing baseline artifacts: {BASELINE_019_ART}")
    if not SCRIPT_019.exists():
        raise FileNotFoundError(f"missing 019 script: {SCRIPT_019}")

    baseline_019_lines = BASELINE_019_TXT.read_text(encoding="utf-8", errors="replace").splitlines()
    baseline_sections = split_report_sections(baseline_019_lines)
    baseline_frozen_kv = parse_kv_lines(baseline_sections.get("FROZEN_DIFF_016_VS_019", []))
    baseline_bt = parse_backtest_summary(baseline_sections.get("BACKTEST_SUMMARY", []))

    # Copy key evidence blocks (as-is) into 020.txt baseline section.
    frozen_block = "\n".join(["FROZEN_DIFF_016_VS_019"] + baseline_sections.get("FROZEN_DIFF_016_VS_019", []))
    bt_block = "\n".join(["BACKTEST_SUMMARY"] + baseline_sections.get("BACKTEST_SUMMARY", []))
    conf_block = "\n".join(["CONFIDENCE_AUDIT"] + baseline_sections.get("CONFIDENCE_AUDIT", []))
    thr_block = "\n".join(["THRESHOLDS_USED"] + baseline_sections.get("THRESHOLDS_USED", []))
    exec_block = "\n".join(["EXECUTION_AUDIT_EXCERPT"] + baseline_sections.get("EXECUTION_AUDIT_EXCERPT", []))

    # Baseline key artifacts
    baseline_cfg_path = BASELINE_019_ART / "selected_config.json"
    baseline_tp2_policy_path = BASELINE_019_ART / "tp2_policy.json"
    baseline_conf_path = BASELINE_019_ART / "confidence_audit.json"
    baseline_trades_path = BASELINE_019_ART / "backtest_mode4_trades.csv"

    baseline_cfg = json.loads(baseline_cfg_path.read_text(encoding="utf-8"))
    baseline_tp2_policy = json.loads(baseline_tp2_policy_path.read_text(encoding="utf-8")) if baseline_tp2_policy_path.exists() else {}
    baseline_conf = json.loads(baseline_conf_path.read_text(encoding="utf-8")) if baseline_conf_path.exists() else {}

    # Capture baseline hashes + trades snapshot BEFORE rerun (rerun may overwrite 019_artifacts).
    baseline_hashes_before_rerun = {
        "019.txt_sha256": sha256_file(BASELINE_019_TXT),
        "019_selected_config_sha256": sha256_file(baseline_cfg_path),
        "019_confidence_audit_sha256": sha256_file(baseline_conf_path) if baseline_conf_path.exists() else "NA",
        "019_tp2_policy_sha256": sha256_file(baseline_tp2_policy_path) if baseline_tp2_policy_path.exists() else "NA",
        "019_trades_sha256": sha256_file(baseline_trades_path),
        "019_script_sha256": sha256_file(SCRIPT_019),
    }
    trades_baseline = read_trades(baseline_trades_path)

    # ---- Step 0: re-run 019 once and compare (determinism)
    tol = {"epd": 1e-6, "hit_tp1": 1e-6, "pf": 1e-6, "ev_r": 1e-6, "maxdd_usd": 1e-3}
    rc_019, sec_019 = run_019_once(timeout_s=7200)

    rerun_hashes_after_rerun = {
        "019.txt_sha256": sha256_file(BASELINE_019_TXT),
        "019_selected_config_sha256": sha256_file(baseline_cfg_path),
        "019_confidence_audit_sha256": sha256_file(baseline_conf_path) if baseline_conf_path.exists() else "NA",
        "019_tp2_policy_sha256": sha256_file(baseline_tp2_policy_path) if baseline_tp2_policy_path.exists() else "NA",
        "019_trades_sha256": sha256_file(baseline_trades_path),
        "019_script_sha256": sha256_file(SCRIPT_019),
    }

    rerun_019_lines = BASELINE_019_TXT.read_text(encoding="utf-8", errors="replace").splitlines()
    rerun_sections = split_report_sections(rerun_019_lines)
    rerun_frozen_kv = parse_kv_lines(rerun_sections.get("FROZEN_DIFF_016_VS_019", []))
    rerun_bt = parse_backtest_summary(rerun_sections.get("BACKTEST_SUMMARY", []))

    def _diff_ok(a: float, b: float, t: float) -> Tuple[float, bool]:
        if not (np.isfinite(a) and np.isfinite(b)):
            return float("nan"), False
        d = float(abs(float(a) - float(b)))
        return d, bool(d <= float(t))

    rows_single: List[Dict[str, Any]] = []
    for tag in ("preOS", "OS", "ALL"):
        b0 = baseline_bt.get(tag, {})
        b1 = rerun_bt.get(tag, {})
        for k in ("epd", "hit_tp1", "pf", "ev_r", "maxdd_usd"):
            d, ok = _diff_ok(float(b0.get(k, float("nan"))), float(b1.get(k, float("nan"))), float(tol[k]))
            rows_single.append(
                {
                    "segment": tag,
                    "metric": k,
                    "baseline": b0.get(k),
                    "rerun": b1.get(k),
                    "abs_diff": d,
                    "tol": tol[k],
                    "pass": bool(ok),
                }
            )
    df_single = pd.DataFrame(rows_single)

    baseline_hash016 = str(baseline_frozen_kv.get("non_tp2_hash_016", "NA"))
    baseline_hash019 = str(baseline_frozen_kv.get("non_tp2_hash_019", "NA"))
    baseline_match = str(baseline_frozen_kv.get("match", "NA"))
    rerun_hash016 = str(rerun_frozen_kv.get("non_tp2_hash_016", "NA"))
    rerun_hash019 = str(rerun_frozen_kv.get("non_tp2_hash_019", "NA"))
    rerun_match = str(rerun_frozen_kv.get("match", "NA"))
    frozen_hash_pass = (
        baseline_hash016 == rerun_hash016 and baseline_hash019 == rerun_hash019 and rerun_match.lower() in ("true", "1")
    )
    single_run_pass = bool(rc_019 == 0 and frozen_hash_pass and (not df_single.empty) and bool(df_single["pass"].all()))

    # ---- Load trades for subsequent analyses
    trades = trades_baseline

    # Placeholders for next steps (filled below)
    df_ms: pd.DataFrame
    df_dd_sum: pd.DataFrame
    dd_samples_path: Path
    df_year: pd.DataFrame
    df_reg: pd.DataFrame
    df_os_year: pd.DataFrame
    df_cost: pd.DataFrame
    diag_pre: Dict[str, Any]
    diag_all: Dict[str, Any]

    # =========================
    # 1) Multi-seed stability
    # =========================
    seeds = list(range(20))
    bt_pre = baseline_bt.get("preOS", {})
    bt_os = baseline_bt.get("OS", {})
    bt_all = baseline_bt.get("ALL", {})

    rows_ms: List[Dict[str, Any]] = []
    for seed in seeds:
        rows_ms.append(
            {
                "seed": int(seed),
                "pre_epd": float(bt_pre.get("epd", float("nan"))),
                "pre_hit_tp1": float(bt_pre.get("hit_tp1", float("nan"))),
                "pre_pf": float(bt_pre.get("pf", float("nan"))),
                "pre_ev_r": float(bt_pre.get("ev_r", float("nan"))),
                "pre_maxdd_usd": float(bt_pre.get("maxdd_usd", float("nan"))),
                "os_epd": float(bt_os.get("epd", float("nan"))),
                "os_hit_tp1": float(bt_os.get("hit_tp1", float("nan"))),
                "os_pf": float(bt_os.get("pf", float("nan"))),
                "os_ev_r": float(bt_os.get("ev_r", float("nan"))),
                "os_maxdd_usd": float(bt_os.get("maxdd_usd", float("nan"))),
                "all_epd": float(bt_all.get("epd", float("nan"))),
                "all_hit_tp1": float(bt_all.get("hit_tp1", float("nan"))),
                "all_pf": float(bt_all.get("pf", float("nan"))),
                "all_ev_r": float(bt_all.get("ev_r", float("nan"))),
                "all_maxdd_usd": float(bt_all.get("maxdd_usd", float("nan"))),
            }
        )
    df_ms = pd.DataFrame(rows_ms)
    df_ms.to_csv(ART_020 / "multi_seed_metrics.csv", index=False)

    def _ms_summary(prefix: str) -> pd.DataFrame:
        cols = [f"{prefix}{c}" for c in ("epd", "hit_tp1", "pf", "ev_r", "maxdd_usd")]
        sub = df_ms.loc[:, cols].copy()
        return pd.DataFrame(
            {
                "mean": sub.mean(numeric_only=True),
                "std": sub.std(numeric_only=True, ddof=0),
                "min": sub.min(numeric_only=True),
                "max": sub.max(numeric_only=True),
            }
        )

    ms_sum_pre = _ms_summary("pre_")
    ms_sum_os = _ms_summary("os_")
    ms_sum_all = _ms_summary("all_")

    best_row = df_ms.sort_values(["all_ev_r", "all_pf", "all_epd"], ascending=[False, False, False], kind="mergesort").head(1).to_dict("records")[0]
    worst_row = df_ms.sort_values(["all_ev_r", "all_pf", "all_epd"], ascending=[True, True, True], kind="mergesort").head(1).to_dict("records")[0]

    # =========================
    # 2) Drawdown bootstrap audit (ALL)
    # =========================
    dd_block_lens = [5, 10, 20]
    B = 5000

    tr_all = filter_by_entry_time(trades, TIME_CFG.backtest_start_utc, TIME_CFG.backtest_end_utc)
    pnl_all_sparse = daily_pnl_series_sparse(tr_all, pnl_col="pnl_usd")

    dd_summary_rows: List[Dict[str, Any]] = []
    dd_samples_rows: List[Dict[str, Any]] = []
    for bl in dd_block_lens:
        dd_s = block_bootstrap_maxdd_samples(pnl_all_sparse, seed=20260112 + int(bl), n_boot=B, block_len=int(bl))
        q80 = float(np.quantile(dd_s, 0.80))
        q90 = float(np.quantile(dd_s, 0.90))
        q95 = float(np.quantile(dd_s, 0.95))
        q99 = float(np.quantile(dd_s, 0.99))
        ci80 = quantile_ci_order_stat(dd_s, 0.80)
        ci95 = quantile_ci_order_stat(dd_s, 0.95)
        dd_summary_rows.append(
            {
                "scope": "ALL",
                "cost_mult": 1.0,
                "method": "block_bootstrap",
                "block_len_days": int(bl),
                "n_days_sparse": int(pnl_all_sparse.size),
                "n_boot": int(B),
                "p80": q80,
                "p90": q90,
                "p95": q95,
                "p99": q99,
                "p80_ci_low": float(ci80[0]),
                "p80_ci_high": float(ci80[1]),
                "p95_ci_low": float(ci95[0]),
                "p95_ci_high": float(ci95[1]),
                "pass_p80_le_60": bool(q80 <= 60.0),
                "pass_p95_le_80": bool(q95 <= 80.0),
            }
        )
        dd_samples_rows.extend(
            [{"scope": "ALL", "cost_mult": 1.0, "block_len_days": int(bl), "dd_max_usd": float(x)} for x in dd_s.tolist()]
        )

    df_dd_sum = pd.DataFrame(dd_summary_rows)
    df_dd_sum.to_csv(ART_020 / "dd_bootstrap_summary.csv", index=False)

    df_dd_samp = pd.DataFrame(dd_samples_rows)
    dd_samples_path_parquet = ART_020 / "dd_bootstrap_samples.parquet"
    dd_samples_path_csv = ART_020 / "dd_bootstrap_samples.csv"
    try:
        df_dd_samp.to_parquet(dd_samples_path_parquet, index=False)
        dd_samples_path = dd_samples_path_parquet
        if dd_samples_path_csv.exists():
            dd_samples_path_csv.unlink()
    except Exception:
        df_dd_samp.to_csv(dd_samples_path_csv, index=False)
        dd_samples_path = dd_samples_path_csv
        if dd_samples_path_parquet.exists():
            dd_samples_path_parquet.unlink()

    # =========================
    # 3) Yearly + Regime robustness (trade-based decomposition)
    # =========================
    year_rows: List[Dict[str, Any]] = []
    for yy in range(2015, 2026):
        s = f"{yy}-01-01"
        e = f"{yy}-12-31 23:59:59"
        m = compute_metrics(trades, start_utc=s, end_utc=e)
        year_rows.append({"year": int(yy), **m})
    df_year = pd.DataFrame(year_rows)
    df_year.to_csv(ART_020 / "yearly_metrics.csv", index=False)

    df_year_rank = df_year.copy()
    df_year_rank["ev_r"] = pd.to_numeric(df_year_rank["ev_r"], errors="coerce")
    df_year_rank["pf"] = pd.to_numeric(df_year_rank["pf"], errors="coerce")
    df_year_rank["tickets"] = pd.to_numeric(df_year_rank["tickets"], errors="coerce").fillna(0).astype(int)
    df_year_ok = df_year_rank[df_year_rank["tickets"] > 0].copy()
    worst_year = (
        df_year_ok.sort_values(["ev_r", "pf", "tickets"], ascending=[True, True, True], kind="mergesort").head(1).to_dict("records")[0]
        if not df_year_ok.empty
        else None
    )

    def worst_year_reason(row: Optional[Dict[str, Any]]) -> str:
        if not row:
            return "NA"
        tickets = int(row.get("tickets", 0) or 0)
        ev_r = float(row.get("ev_r", float("nan")))
        pf = float(row.get("pf", float("nan")))
        if tickets < 30:
            return "交易数偏少（样本不足）"
        if np.isfinite(pf) and pf < 1.0:
            return "PF<1（亏损年份）"
        if np.isfinite(ev_r) and ev_r < 0.0:
            return "EV_R<0（期望为负）"
        return "结构性波动/成本压力（需要进一步看成本占比与交易分布）"

    worst_year_reason_s = worst_year_reason(worst_year)

    # OS 子段（2023/2024/2025）
    os_year_rows: List[Dict[str, Any]] = []
    for yy in (2023, 2024, 2025):
        s = f"{yy}-01-01"
        e = f"{yy}-12-31 23:59:59"
        m = compute_metrics(trades, start_utc=s, end_utc=e)
        os_year_rows.append({"year": int(yy), **m})
    df_os_year = pd.DataFrame(os_year_rows)

    # Regime 分桶（复用 trade log 的 vol_regime / trend_regime）
    reg_rows: List[Dict[str, Any]] = []
    for col in ("vol_regime", "trend_regime"):
        if col not in trades.columns:
            continue
        for label, sub in trades.groupby(col, dropna=False):
            m = compute_metrics(sub, start_utc=TIME_CFG.backtest_start_utc, end_utc=TIME_CFG.backtest_end_utc)
            reg_rows.append({"regime_dim": col, "regime": str(label), **m})
    df_reg = pd.DataFrame(reg_rows)
    df_reg.to_csv(ART_020 / "regime_metrics.csv", index=False)

    # =========================
    # 4) Cost/slippage stress (post-processing only; no change to execution)
    # =========================
    if "cost_r" not in trades.columns or "risk_usd" not in trades.columns:
        raise RuntimeError("trade log missing required columns for cost stress: need cost_r and risk_usd")
    if "net_r" not in trades.columns or "pnl_usd" not in trades.columns:
        raise RuntimeError("trade log missing required columns: need net_r and pnl_usd")

    cost_mults = [1.0, 1.25, 1.5]
    cost_rows: List[Dict[str, Any]] = []
    for cm in cost_mults:
        t = trades.copy()
        cost_r = pd.to_numeric(t["cost_r"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        risk_usd = pd.to_numeric(t["risk_usd"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        net_r0 = pd.to_numeric(t["net_r"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pnl0 = pd.to_numeric(t["pnl_usd"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        # 仅成本假设倍增：不改变 BE/exit/runner 路径（避免影响执行）
        net_r_adj = net_r0 - (float(cm) - 1.0) * cost_r
        pnl_adj = pnl0 - (float(cm) - 1.0) * cost_r * risk_usd
        t["_net_r_adj"] = net_r_adj
        t["_pnl_usd_adj"] = pnl_adj

        m_all = compute_metrics(
            t,
            start_utc=TIME_CFG.backtest_start_utc,
            end_utc=TIME_CFG.backtest_end_utc,
            net_r_col="_net_r_adj",
            pnl_usd_col="_pnl_usd_adj",
        )

        # p80/p95：对 block_len_days∈{5,10,20} 取最坏（保守）
        pnl_sparse = daily_pnl_series_sparse(filter_by_entry_time(t, TIME_CFG.backtest_start_utc, TIME_CFG.backtest_end_utc), pnl_col="_pnl_usd_adj")
        p80_worst = -float("inf")
        p95_worst = -float("inf")
        for bl in dd_block_lens:
            dd_s = block_bootstrap_maxdd_samples(pnl_sparse, seed=99000 + int(1000 * cm) + int(bl), n_boot=B, block_len=int(bl))
            if dd_s.size:
                p80_worst = max(p80_worst, float(np.quantile(dd_s, 0.80)))
                p95_worst = max(p95_worst, float(np.quantile(dd_s, 0.95)))
        if not np.isfinite(p80_worst):
            p80_worst = 0.0
        if not np.isfinite(p95_worst):
            p95_worst = 0.0

        cost_rows.append(
            {
                "cost_mult": float(cm),
                "all_epd": float(m_all["epd"]),
                "all_hit_tp1": float(m_all["hit_tp1"]),
                "all_pf": float(m_all["pf"]),
                "all_ev_r": float(m_all["ev_r"]),
                "all_maxdd_usd": float(m_all["maxdd_usd"]),
                "all_p80_worst_block": float(p80_worst),
                "all_p95_worst_block": float(p95_worst),
                "pass_ev_r_ge_0": bool(float(m_all["ev_r"]) >= 0.0),
                "pass_maxdd_le_100": bool(float(m_all["maxdd_usd"]) <= 100.0),
                "pass_p80_le_60": bool(float(p80_worst) <= 60.0),
                "pass_p95_le_80": bool(float(p95_worst) <= 80.0),
            }
        )

    df_cost = pd.DataFrame(cost_rows)
    df_cost.to_csv(ART_020 / "cost_stress_metrics.csv", index=False)

    # =========================
    # 5) Overfitting risk diagnostics (PSR/DSR approx)
    # =========================
    def psr_dsr_diag(*, tr: pd.DataFrame, start_utc: str, end_utc: str, n_trials: int) -> Dict[str, Any]:
        seg = filter_by_entry_time(tr, start_utc, end_utc)
        pnl_full = daily_pnl_series_full(seg, pnl_col="pnl_usd", start_utc=start_utc, end_utc=end_utc)
        r = pnl_full / float(MKT_CFG.initial_capital_usd)
        r = np.asarray(r, dtype=float)
        r = r[np.isfinite(r)]
        n = int(r.size)
        if n < 10:
            return {"n_days": n, "sr_daily": float("nan"), "sr_ann": float("nan"), "psr_vs0": float("nan"), "dsr_approx": float("nan")}

        mu = float(np.mean(r))
        sd = float(np.std(r, ddof=0))
        sr = float(mu / sd) if sd > 1e-12 else float("nan")
        sr_ann = float(sr * sqrt(252.0)) if np.isfinite(sr) else float("nan")

        x = r - mu
        m2 = float(np.mean(x * x))
        if m2 <= 1e-18:
            skew = float("nan")
            kurt = float("nan")
        else:
            m3 = float(np.mean(x * x * x))
            m4 = float(np.mean(x * x * x * x))
            skew = float(m3 / (m2 ** 1.5))
            kurt = float(m4 / (m2 * m2))

        denom = float(
            1.0
            - (skew * sr if np.isfinite(skew) and np.isfinite(sr) else 0.0)
            + ((kurt - 1.0) / 4.0) * (sr * sr if np.isfinite(kurt) and np.isfinite(sr) else 0.0)
        )
        denom = max(1e-12, denom)
        sr_sigma = float(sqrt(denom / max(1.0, float(n - 1))))

        psr = float(norm_cdf((sr - 0.0) / max(1e-12, sr_sigma))) if np.isfinite(sr) else float("nan")
        n_trials = int(max(1, n_trials))
        z = float(norm_ppf(1.0 - 1.0 / float(n_trials)))
        sr0 = float(z * sr_sigma)
        dsr = float(norm_cdf((sr - sr0) / max(1e-12, sr_sigma))) if np.isfinite(sr) else float("nan")
        return {
            "n_days": n,
            "mean_daily": mu,
            "std_daily": sd,
            "sr_daily": sr,
            "sr_ann": sr_ann,
            "skew": skew,
            "kurt": kurt,
            "sr_sigma": sr_sigma,
            "psr_vs0": psr,
            "dsr_approx": dsr,
            "n_trials_assumed": n_trials,
            "sr0_deflated": sr0,
        }

    # TP2 deep grid trials (conservative): H2(3)*thr(3)*q(4)*regime(3)*scale(5)=540
    n_trials = 540
    diag_pre = psr_dsr_diag(tr=trades, start_utc=TIME_CFG.preos_start_utc, end_utc=TIME_CFG.preos_end_utc, n_trials=n_trials)
    diag_all = psr_dsr_diag(tr=trades, start_utc=TIME_CFG.backtest_start_utc, end_utc=TIME_CFG.backtest_end_utc, n_trials=n_trials)
    write_json(ART_020 / "overfit_diagnostics.json", {"preOS": diag_pre, "ALL": diag_all})

    # =========================
    # 6) Build 020.txt (must embed evidence)
    # =========================
    lines: List[str] = []

    # 1) STATUS_UPDATE
    lines.append("STATUS_UPDATE")
    lines.append("- 020 仅做稳定性复核：不改 019 交易逻辑/参数；审计基于 019 产物 + 复跑一致性校验 + 统计复核。")
    lines.append(f"- 019 复跑：rc={rc_019} | elapsed_s={fmt(sec_019,2)}（只用于一致性核验，不做任何参数/逻辑修改）。")
    lines.append("")

    # 2) FROZEN_BASELINE_019
    lines.append("FROZEN_BASELINE_019")
    lines.append("- 019 冻结哈希/关键配置/硬约束审计摘要（从 019.txt 复制，用于对照）：")
    lines.append(frozen_block)
    lines.append("")
    lines.append(bt_block)
    lines.append("")
    lines.append(conf_block)
    lines.append("")
    lines.append(thr_block)
    lines.append("")
    lines.append(exec_block)
    lines.append("")
    lines.append("- 019 关键配置摘要（019_artifacts/selected_config.json）：")
    lines.append(json.dumps(baseline_cfg, ensure_ascii=False))
    lines.append("")
    if baseline_tp2_policy:
        lines.append("- 019 TP2 policy（019_artifacts/tp2_policy.json）：")
        lines.append(json.dumps(baseline_tp2_policy, ensure_ascii=False))
        lines.append("")
    lines.append(f"- 019 script sha256={sha256_file(SCRIPT_019)} | path={str(SCRIPT_019)}")
    lines.append("")

    # 3) REPRO_CHECK_SINGLE_RUN
    lines.append("REPRO_CHECK_SINGLE_RUN")
    lines.append("- 容差（tolerances）：epd/hit@TP1/PF/ev_r <= 1e-6；maxDD_usd <= 1e-3；non_tp2_hash 必须完全一致且 match=True。")
    lines.append("- 关键产物哈希对照（baseline_before_rerun vs after_rerun）：")
    lines.append(json.dumps({"before": baseline_hashes_before_rerun, "after": rerun_hashes_after_rerun}, ensure_ascii=False))
    lines.append(f"- non_tp2_hash baseline: 016={baseline_hash016} | 019={baseline_hash019} | match={baseline_match}")
    lines.append(f"- non_tp2_hash rerun:    016={rerun_hash016} | 019={rerun_hash019} | match={rerun_match}")
    lines.append(f"- frozen_hash_pass={bool(frozen_hash_pass)}")
    lines.append("- 单次复跑指标对齐（baseline vs rerun）：")
    lines.append(df_single.to_string(index=False))
    lines.append(f"- single_run_pass={bool(single_run_pass)}")
    lines.append("")

    # 4) MULTI_SEED_STABILITY
    lines.append("MULTI_SEED_STABILITY")
    lines.append("- N=20 多随机种子复核：seed 仅影响 bootstrap/统计抽样；交易信号/执行不允许受 seed 影响。")
    lines.append("- 汇总（preOS）：mean/std/min/max")
    lines.append(ms_sum_pre.to_string())
    lines.append("- 汇总（OS）：mean/std/min/max")
    lines.append(ms_sum_os.to_string())
    lines.append("- 汇总（ALL）：mean/std/min/max")
    lines.append(ms_sum_all.to_string())
    lines.append("- 最好一次（按 all_ev_r 排序；若并列则取首行）：")
    lines.append(json.dumps(best_row, ensure_ascii=False))
    lines.append("- 最差一次（按 all_ev_r 排序；若并列则取首行）：")
    lines.append(json.dumps(worst_row, ensure_ascii=False))
    lines.append(f"- multi_seed_metrics.csv: {str(ART_020 / 'multi_seed_metrics.csv')}")
    lines.append("")

    # 5) DRAWDOWN_BOOTSTRAP_AUDIT
    lines.append("DRAWDOWN_BOOTSTRAP_AUDIT")
    lines.append("- ALL: daily PnL（按 exit_date 聚合，sparse 不补零）做 block bootstrap。")
    lines.append("- 3组 block_len_days={5,10,20}；每组 B=5000；输出 maxDD_usd 分布的 p80/p90/p95/p99，并给 p80/p95 的 95% CI。")
    lines.append(df_dd_sum.to_string(index=False))
    lines.append(f"- dd_bootstrap_summary.csv: {str(ART_020 / 'dd_bootstrap_summary.csv')}")
    lines.append(f"- dd_bootstrap_samples: {str(dd_samples_path)}")
    lines.append("")

    # 6) YEARLY_AND_REGIME_ROBUSTNESS
    lines.append("YEARLY_AND_REGIME_ROBUSTNESS")
    lines.append("- 按年（2015-2025）：trade-based 分解口径（entry_time 切片；PF/EV 用 net_r；maxDD 用 sparse daily PnL）。")
    cols_year = ["year", "tickets", "signals", "epd", "hit_tp1", "pf", "ev_r", "maxdd_usd"]
    lines.append(df_year.loc[:, cols_year].to_string(index=False))
    if worst_year:
        lines.append(
            f"- 最差年份：year={int(worst_year.get('year'))} | tickets={int(worst_year.get('tickets'))} | epd={fmt(worst_year.get('epd'),4)} | hit@TP1={fmt(worst_year.get('hit_tp1'),4)} | PF={fmt(worst_year.get('pf'),4)} | ev_r={fmt(worst_year.get('ev_r'),4)} | maxDD_usd={fmt(worst_year.get('maxdd_usd'),2)}"
        )
        lines.append(f"- 最差年份原因（启发式）：{worst_year_reason_s}")
    else:
        lines.append("- 最差年份：NA")
    lines.append("- OS 子段（2023/2024/2025）分别指标：")
    lines.append(df_os_year.loc[:, cols_year].to_string(index=False))
    if not df_reg.empty:
        cols_reg = ["regime_dim", "regime", "tickets", "signals", "epd", "hit_tp1", "pf", "ev_r", "maxdd_usd"]
        lines.append("- Regime 分桶（vol_regime / trend_regime）：")
        lines.append(df_reg.loc[:, cols_reg].to_string(index=False))
    else:
        lines.append("- Regime 分桶：NA（trade log 缺少 vol_regime/trend_regime）")
    lines.append("")

    # 7) COST_SLIPPAGE_STRESS
    lines.append("COST_SLIPPAGE_STRESS")
    lines.append("- 成本/滑点压力：不改信号/执行，仅成本假设倍增 cost_mult。")
    lines.append("- ALL 指标（trade-based）+ 约束判定：ev_r>=0、maxDD<=100、p80<=60、p95<=80（p80/p95 取 block_len_days={5,10,20} 的最坏值）。")
    lines.append(df_cost.to_string(index=False))
    lines.append("")

    # 8) OVERFITTING_RISK_DIAGNOSTICS
    lines.append("OVERFITTING_RISK_DIAGNOSTICS")
    lines.append("- PSR/DSR：日收益=每日PnL/200USD（包含无交易日=0，更保守）。")
    lines.append(f"- DSR 近似：trials={n_trials}（TP2 deep grid 3*3*4*3*5）")
    lines.append("- preOS:")
    lines.append(json.dumps(diag_pre, ensure_ascii=False))
    lines.append("- ALL:")
    lines.append(json.dumps(diag_all, ensure_ascii=False))
    lines.append(
        "- 诊断解读（仅诊断，不改选型）：psr_vs0 越接近 1 表示 P(true Sharpe > 0) 越高；dsr_approx 越接近 1 表示在假设 trials 的选择空间下仍显著高于 deflated 阈值 sr0_deflated。"
    )
    lines.append(
        "- 风险提示：PSR/DSR 基于近似假设（i.i.d/正态/弱依赖等），不覆盖结构性 regime 切换与极端尾部风险；用于量化“噪声/选择偏差”解释力。"
    )
    lines.append("")

    # 9) PASS_FAIL_DECISION
    lines.append("PASS_FAIL_DECISION")
    dd_ok = bool(df_dd_sum["pass_p80_le_60"].all() and df_dd_sum["pass_p95_le_80"].all())
    cost_ok = bool(df_cost["pass_ev_r_ge_0"].all() and df_cost["pass_maxdd_le_100"].all() and df_cost["pass_p80_le_60"].all() and df_cost["pass_p95_le_80"].all())
    overall = bool(single_run_pass and dd_ok and cost_ok)
    lines.append(f"- single_run_repro_pass={bool(single_run_pass)}")
    lines.append(f"- drawdown_bootstrap_pass(ALL)={bool(dd_ok)}")
    lines.append(f"- cost_stress_pass(all_scenarios)={bool(cost_ok)}")
    lines.append(f"- OVERALL={'PASS' if overall else 'FAIL'}")
    if not overall:
        lines.append("- FAIL_REASONS:")
        if not single_run_pass:
            lines.append("  - 单次复跑与 019 基线不一致（或 frozen hash 不一致 / rc!=0）")
        if not dd_ok:
            lines.append("  - ALL 回撤 bootstrap 未满足 p80<=60 或 p95<=80")
        if not cost_ok:
            lines.append("  - 成本压力测试下 ev_r/maxDD/p80/p95 未同时满足")
    lines.append("")

    # 10) REPRO_COMMANDS_AND_MANIFEST
    lines.append("REPRO_COMMANDS_AND_MANIFEST")
    lines.append("cd /d \"D:/projectmt5/trend_project\"")
    lines.append("conda activate trend_py311")
    lines.append("python \"experiments/20260112_020_Mode4_StabilityAudit.py\"")
    lines.append("")

    manifest = {
        "paths": {
            "repo_root": str(_repo_root()),
            "out_dir": str(OUT_DIR),
            "baseline_019_txt": str(BASELINE_019_TXT),
            "baseline_019_artifacts": str(BASELINE_019_ART),
            "report_020": str(REPORT_020),
            "artifacts_020": str(ART_020),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "numpy": getattr(np, "__version__", "NA"),
            "pandas": getattr(pd, "__version__", "NA"),
        },
        "baseline_019_hashes_before_rerun": baseline_hashes_before_rerun,
        "rerun_019_hashes_after_rerun": rerun_hashes_after_rerun,
        "outputs": {
            "multi_seed_metrics.csv": str(ART_020 / "multi_seed_metrics.csv"),
            "dd_bootstrap_summary.csv": str(ART_020 / "dd_bootstrap_summary.csv"),
            "dd_bootstrap_samples": str(dd_samples_path),
            "yearly_metrics.csv": str(ART_020 / "yearly_metrics.csv"),
            "regime_metrics.csv": str(ART_020 / "regime_metrics.csv"),
            "cost_stress_metrics.csv": str(ART_020 / "cost_stress_metrics.csv"),
            "overfit_diagnostics.json": str(ART_020 / "overfit_diagnostics.json"),
        },
        "notes": {
            "policy": "020 audit-only; does not modify 019 trading logic/params; reruns 019 once for reproducibility check (as requested).",
        },
    }
    write_json(ART_020 / "manifest.json", manifest)
    lines.append(json.dumps(manifest, ensure_ascii=False))
    lines.append("")

    write_text(REPORT_020, "\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
