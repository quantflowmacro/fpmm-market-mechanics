#!/usr/bin/env python3
"""
QuantFlow Macro - Working Paper 01 Official Replication & Verification Suite (V1.1)
Title: First-Principles Market Mechanics: Kinematic Regime Classification via
       Potential-Well Topology and Amortized Overhead Resistance
Author: Kevin Z. Qiu (QuantFlow Macro Research Institute)
SSRN Archive: https://ssrn.com/abstract=7667578
DOI:          10.2139/ssrn.7667578
Repository:   https://github.com/quantflowmacro/fpmm-market-mechanics
"""

import io
from pathlib import Path
import sys
import time
import urllib.request
import zipfile
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

SP100_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "AVGO", "TSLA", "JPM", "UNH",
    "XOM", "LLY", "V", "PG", "COST",
    "HD", "MA", "ABBV", "MRK", "PEP",
]


def load_or_fetch_us_sample(cache_path: Path, force_refresh: bool = False) -> pd.DataFrame:
    """加载或通过 yfinance 拉取 20 只 S&P 100 流动性核心成分股真实日线 (含 2019 预热期)。"""
    if cache_path.exists() and not force_refresh:
        df = pd.read_parquet(cache_path)
        if "market" in df.columns:
            df = df[df["market"] == "US"].copy()
        if len(df) > 1000:
            print(f"[1/5] Loaded {len(df):,} cached US stock-days from {cache_path}")
            return df

    import yfinance as yf
    print("[1/5] Fetching 20-stock S&P 100 liquid core sample via yfinance ...")
    records = []
    for tk in SP100_SAMPLE:
        try:
            t_obj = yf.Ticker(tk)
            hist = t_obj.history(start="2019-01-01", end="2026-11-01", auto_adjust=True)
            if hist.empty or len(hist) < 300:
                continue
            try:
                info = t_obj.info or {}
                s_float = float(info.get("floatShares") or info.get("sharesOutstanding") or 1e9)
            except Exception:
                s_float = 1e9

            hist = hist.reset_index()
            records.append(pd.DataFrame({
                "date": pd.to_datetime(hist["Date"]).dt.tz_localize(None),
                "ticker": tk,
                "open": hist["Open"].astype(float),
                "high": hist["High"].astype(float),
                "low": hist["Low"].astype(float),
                "close": hist["Close"].astype(float),
                "volume": hist["Volume"].astype(float),
                "s_float": s_float,
            }))
            time.sleep(0.2)
        except Exception as exc:
            print(f"  [WARN] Skip {tk}: {exc}")

    if not records:
        raise RuntimeError("No data fetched from yfinance. Please check internet connection.")

    df_all = pd.concat(records, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(cache_path, index=False)
    print(f"  [OK] Saved {len(df_all):,} real stock-days to {cache_path}")
    return df_all


def compute_wp01_physics(df: pd.DataFrame) -> pd.DataFrame:
    """严格复现 WP-01 Section 2 至 Section 6 全部物理算子与 7 大 MECE 相态分类。"""
    df = df.sort_values(["ticker", "date"]).copy()

    # 1. 重心均价与活跃换手率 FT_real (%)
    df["vwap"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    df["ft_real"] = (df["volume"] / df["s_float"]) * 100.0
    df["dollar_vol"] = df["close"] * df["volume"]

    # 2. 宏观看板水深乘数 K_macro(t) = clip(V_ma5 / V_ma20, 0.25, 1.25) (严格 t-1 滞后)
    mkt_vol = df.groupby("date")["dollar_vol"].sum().sort_index()
    v_ma5 = mkt_vol.rolling(5, min_periods=1).mean().shift(1)
    v_ma20 = mkt_vol.rolling(20, min_periods=5).mean().shift(1)
    k_macro_s = (v_ma5 / v_ma20).clip(0.25, 1.25).fillna(1.0)
    df["k_macro"] = df["date"].map(k_macro_s)

    # 3. 重心做功效率 VWAP_PDE (下四分位阻尼底线 gamma_min = 0.80%)
    gamma_min = 0.80
    denom = np.maximum(df["ft_real"], gamma_min)
    df["vwap_pde"] = (((df["close"] - df["vwap"]) / df["vwap"]) * 100.0) / denom

    # 4. K线形态几何特征
    hl_span = np.maximum(df["high"] - df["low"], 1e-6)
    upper_wick = df["high"] - np.maximum(df["open"], df["close"])
    df["wick_ratio"] = upper_wick / hl_span
    prev_close = df.groupby("ticker")["close"].shift(1)
    df["ret_1d"] = (df["close"] - prev_close) / prev_close

    # 5. 250日指数衰减头顶阻力密度 RD (lambda=0.985) 与相对势能坐标 Phi
    lam = 0.985
    weights = lam ** np.arange(249, -1, -1)

    def _per_stock_kernel(g: pd.DataFrame) -> pd.DataFrame:
        p_typ = ((g["high"] + g["low"] + g["close"]) / 3.0).to_numpy()
        vol = g["volume"].to_numpy()
        c_arr = g["close"].to_numpy()
        lo_arr = g["low"].to_numpy()
        hi_arr = g["high"].to_numpy()
        n = len(g)

        rd = np.full(n, np.nan)
        phi = np.full(n, np.nan)
        for i in range(249, n):
            p_w = p_typ[i - 249 : i + 1]
            v_w = vol[i - 249 : i + 1]
            c_now = c_arr[i]
            tot_v = np.dot(weights, v_w)
            if tot_v > 0:
                rd[i] = (np.dot(weights * (p_w > c_now), v_w) / tot_v) * 100.0
            lo_min = np.min(lo_arr[i - 249 : i + 1])
            hi_max = np.max(hi_arr[i - 249 : i + 1])
            if hi_max > lo_min:
                phi[i] = np.clip((c_now - lo_min) / (hi_max - lo_min), 0.0, 1.0)

        out = g.copy()
        out["rd"] = rd
        out["phi"] = phi
        return out

    df = pd.concat([_per_stock_kernel(g) for _, g in df.groupby("ticker", sort=False)], ignore_index=True)

    # 6. 动态临界换手包络 FT_max(Phi) 与做功阻抗比 Z_imp (a = 4.50%, b = 2.80%)
    df["ft_max"] = 4.5 - 2.8 * df["phi"]
    df["z_imp"] = np.where(df["vwap_pde"] > 0, df["rd"] / np.maximum(df["vwap_pde"], 0.05), np.inf)

    # 7. Theorem 6.3 严格互斥完备 (MECE) 7 大热力学相态分类决策树
    cond_r4 = (df["ft_real"] > df["ft_max"]) & ((df["wick_ratio"] > 0.45) | (df["vwap_pde"] <= 0))
    cond_r5 = (df["vwap_pde"] < -0.15) & (df["ret_1d"] < -0.015)
    cond_r6 = (df["ret_1d"] >= 0.095) & (df["wick_ratio"] < 0.05)
    cond_r1 = (df["vwap_pde"] > 0.15) & (df["z_imp"] <= 120.0) & (df["ft_real"] <= df["ft_max"])
    cond_r2 = (df["vwap_pde"] > 0.05) & (df["z_imp"] <= 450.0) & (df["ft_real"] <= df["ft_max"])
    cond_r3 = (df["phi"] <= 0.35) & (df["ft_real"] <= df["ft_max"])

    regime = np.full(len(df), "R7_KINETIC_NEUTRAL", dtype=object)
    regime = np.where(cond_r3, "R3_POTENTIAL_WELL_ACCUM", regime)
    regime = np.where(cond_r2, "R2_CONTROLLED_ESCAPE", regime)
    regime = np.where(cond_r1, "R1_SUPERCONDUCTING_ESCAPE", regime)
    regime = np.where(cond_r6, "R6_BOUNDARY_LOCK", regime)
    regime = np.where(cond_r5, "R5_GRAVITATIONAL_COLLAPSE", regime)
    regime = np.where(cond_r4, "R4_CROCODILE_DISSIPATION", regime)
    df["regime"] = regime

    # 样本外因果前瞻收益率 (t+1 开盘至 t+2/t+6/t+11 开盘)
    open_t1 = df.groupby("ticker")["open"].shift(-1)
    df["fwd_1d"] = (df.groupby("ticker")["open"].shift(-2) - open_t1) / open_t1
    df["fwd_5d"] = (df.groupby("ticker")["open"].shift(-6) - open_t1) / open_t1
    df["fwd_10d"] = (df.groupby("ticker")["open"].shift(-11) - open_t1) / open_t1
    return df


def fetch_ff6_factors() -> pd.DataFrame | None:
    """从 Kenneth French 公开数据库直连拉取 FF5 + Momentum 日度因子表。"""
    ff5_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

    def _read_kf_zip(url: str) -> pd.DataFrame:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
            csv_name = zf.namelist()[0]
            raw_lines = zf.read(csv_name).decode("latin-1").splitlines()
        data_rows = []
        header = None
        for line in raw_lines:
            parts = [p.strip() for p in line.split(",")]
            if header is None and len(parts) >= 2 and (("Mkt-RF" in parts) or (len(parts) == 2 and parts[1] == "Mom")):
                header = ["date"] + parts[1:]
                continue
            if header is not None and len(parts) == len(header) and parts[0].isdigit() and len(parts[0]) == 8:
                data_rows.append(parts)
        df_f = pd.DataFrame(data_rows, columns=header)
        df_f["date"] = pd.to_datetime(df_f["date"], format="%Y%m%d")
        for col in header[1:]:
            df_f[col] = pd.to_numeric(df_f[col], errors="coerce") / 100.0
        return df_f.dropna()

    try:
        df5 = _read_kf_zip(ff5_url)
        dfm = _read_kf_zip(mom_url)
        return pd.merge(df5, dfm, on="date", how="inner")
    except Exception as exc:
        print(f"  [WARN] Kenneth French online fetch skipped ({exc}); using cross-sectional market proxy.")
        return None


def run_factor_regression(daily_ret: pd.Series, df_oos: pd.DataFrame) -> tuple[str, str]:
    """使用 statsmodels 执行 HAC Newey-West (maxlags=5) 因子回归，输出年化 Alpha 与 NW t-Stat。"""
    try:
        import statsmodels.api as sm
    except ImportError:
        return "N/A (install statsmodels)", "N/A"

    ff_df = fetch_ff6_factors()
    ret_df = daily_ret.rename("port_ret").reset_index()

    if ff_df is not None:
        m = pd.merge(ret_df, ff_df, on="date", how="inner")
        if len(m) > 100:
            y = m["port_ret"] - m["RF"]
            factor_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"] if c in m.columns]
            X = sm.add_constant(m[factor_cols])
            model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            ann_alpha = model.params["const"] * 252.0
            t_stat = model.tvalues["const"]
            return f"{ann_alpha * 100:+.2f}% (FF6)", f"{t_stat:.2f}"

    mkt_proxy = df_oos.groupby("date")["fwd_1d"].mean().rename("mkt_ret").reset_index()
    m = pd.merge(ret_df, mkt_proxy, on="date", how="inner").dropna()
    rf_daily = 0.035 / 252.0
    y = m["port_ret"] - rf_daily
    X = sm.add_constant(m["mkt_ret"] - rf_daily)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    ann_alpha = model.params["const"] * 252.0
    t_stat = model.tvalues["const"]
    return f"{ann_alpha * 100:+.2f}% (CAPM)", f"{t_stat:.2f}"


def run_full_wp01_verification(df: pd.DataFrame, out_dir: Path):
    """一次性输出 WP-01 全部实证表格、Newey-West Alpha、转移矩阵、宏观门控消融与年度稳健性。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    df_oos = df[(df["date"] >= "2020-01-02") & df["phi"].notna()].copy()

    # 1. Table 1 (Paper V1.1): 7 大 MECE 热力学相态占比与前瞻超额收益
    print("\n" + "=" * 100)
    print("=== [1/5 · Table 1] 7 MECE Thermodynamic Regimes: Cross-Sectional Share & Forward Returns ===")
    print("=" * 100)
    order = [
        "R1_SUPERCONDUCTING_ESCAPE", "R2_CONTROLLED_ESCAPE", "R3_POTENTIAL_WELL_ACCUM",
        "R7_KINETIC_NEUTRAL", "R4_CROCODILE_DISSIPATION", "R5_GRAVITATIONAL_COLLAPSE", "R6_BOUNDARY_LOCK"
    ]
    t1_rows = []
    tot_n = len(df_oos)
    share_sum = 0.0
    for r_name in order:
        sub = df_oos[df_oos["regime"] == r_name]
        sh = len(sub) / tot_n * 100.0
        share_sum += sh
        mean_z = sub["z_imp"].replace(np.inf, np.nan).mean()
        z_str = "Blocked (+inf)" if (pd.isna(mean_z) or r_name == "R5_GRAVITATIONAL_COLLAPSE") else f"{mean_z:.1f}"
        t1_rows.append({
            "Thermodynamic Regime": r_name,
            "Share (%)": f"{sh:.1f}%",
            "Mean Phi": f"{sub['phi'].mean():.2f}" if len(sub) else "-",
            "Mean Z_imp": z_str,
            "Fwd 1D": f"{sub['fwd_1d'].mean() * 100:+.2f}%" if len(sub) else "-",
            "Fwd 5D": f"{sub['fwd_5d'].mean() * 100:+.2f}%" if len(sub) else "-",
            "Fwd 10D": f"{sub['fwd_10d'].mean() * 100:+.2f}%" if len(sub) else "-",
        })
    t1_df = pd.DataFrame(t1_rows)
    t1_df.to_latex(out_dir / "WP01_Table1_MECE_Regimes.tex", index=False)
    print(t1_df.to_string(index=False))
    print(f"--> MECE Partition Total Share Verification: {share_sum:.1f}% (Exact 100.0% Partition)")

    # 2. Table 2 (Paper V1.1): 状态滞后波段持有协议 + 四档交易摩擦压力测试 + Newey-West Alpha
    pos_list = []
    for _, g in df_oos.groupby("ticker", sort=False):
        regs = g["regime"].to_numpy()
        pos = np.zeros(len(g), dtype=int)
        in_p = 0
        for i, r in enumerate(regs):
            if in_p == 0 and r in ("R1_SUPERCONDUCTING_ESCAPE", "R2_CONTROLLED_ESCAPE"):
                in_p = 1
            elif in_p == 1 and r in ("R4_CROCODILE_DISSIPATION", "R5_GRAVITATIONAL_COLLAPSE"):
                in_p = 0
            pos[i] = in_p
        pos_list.append(pd.Series(pos, index=g.index))
    df_oos["pos"] = pd.concat(pos_list).sort_index()

    entries = (df_oos.groupby("ticker")["pos"].diff().fillna(0) == 1).sum()
    avg_hold = df_oos["pos"].sum() / max(entries, 1)

    print("\n" + "=" * 100)
    print(f"=== [2/5 · Table 2] State-Hysteresis Portfolio & Friction Stress Test (Avg Hold: {avg_hold:.1f} Days) ===")
    print("=" * 100)
    t2_rows = []
    pos_chg = np.abs(df_oos["pos"] - df_oos.groupby("ticker")["pos"].shift(1).fillna(0))

    for bps in [5.0, 10.0, 15.0, 20.0]:
        eff_weight = df_oos["pos"] * df_oos["k_macro"]
        net_r = eff_weight * df_oos["fwd_1d"] - pos_chg * df_oos["k_macro"] * (bps / 10000.0)
        col_name = f"net_r_{int(bps)}"
        df_oos[col_name] = net_r.fillna(0.0)

        daily_ret = df_oos.groupby("date")[col_name].mean()
        ann_ret = daily_ret.mean() * 252.0
        ann_vol = daily_ret.std(ddof=1) * np.sqrt(252.0)
        sharpe = (ann_ret - 0.035) / ann_vol if ann_vol > 0 else 0.0
        nav = (1.0 + daily_ret).cumprod()
        max_dd = (nav / nav.cummax() - 1.0).min()
        alpha_str, t_str = run_factor_regression(daily_ret, df_oos) if bps in (5.0, 10.0) else ("-", "-")

        t2_rows.append({
            "One-Way Cost": f"{bps:.0f} bps",
            "Ann. Return": f"{ann_ret * 100:+.1f}%",
            "Ann. Vol": f"{ann_vol * 100:.1f}%",
            "Max DD": f"{max_dd * 100:.1f}%",
            "Sharpe (Rf=3.5%)": f"{sharpe:.2f}",
            "Net Alpha": alpha_str,
            "NW t-Stat": t_str,
        })

    t2_df = pd.DataFrame(t2_rows)
    t2_df.to_latex(out_dir / "WP01_Table2_Hysteresis_Backtest.tex", index=False)
    print(t2_df.to_string(index=False))

    # 3. 7x7 日度相态转移概率矩阵 (Markov Transition Matrix)
    df_oos["next_regime"] = df_oos.groupby("ticker")["regime"].shift(-1)
    valid_tr = df_oos.dropna(subset=["next_regime"])
    trans_mat = (pd.crosstab(valid_tr["regime"], valid_tr["next_regime"], normalize="index") * 100.0).round(1)
    trans_mat.to_latex(out_dir / "WP01_Diag1_Transition_Matrix.tex")
    print("\n" + "=" * 100)
    print("=== [3/5 · Diagnostic 1] Daily Regime Markov Transition Probability Matrix (%) ===")
    print("=" * 100)
    print(trans_mat.to_string())

    # 4. K_macro 宏观水深乘数消融对照
    df_oos["ret_unscaled_5"] = (df_oos["pos"] * df_oos["fwd_1d"] - pos_chg * (5.0 / 10000.0)).fillna(0.0)
    gate_rows = []
    for label, col in [
        ("Unscaled Hysteresis (Fixed 1.0x Weight)", "ret_unscaled_5"),
        ("Macro-Gated Hysteresis (K_macro in [0.25, 1.25])", "net_r_5"),
    ]:
        d_ret = df_oos.groupby("date")[col].mean()
        ann_r = d_ret.mean() * 252.0
        ann_v = d_ret.std(ddof=1) * np.sqrt(252.0)
        sr = (ann_r - 0.035) / ann_v if ann_v > 0 else 0.0
        nav = (1.0 + d_ret).cumprod()
        mdd = (nav / nav.cummax() - 1.0).min()
        gate_rows.append({
            "Portfolio Configuration": label,
            "Ann. Return": f"{ann_r * 100:+.1f}%",
            "Ann. Vol": f"{ann_v * 100:.1f}%",
            "Max DD": f"{mdd * 100:.1f}%",
            "Sharpe (Rf=3.5%)": f"{sr:.2f}",
        })

    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_latex(out_dir / "WP01_Diag2_KMacro_Ablation.tex", index=False)
    print("\n" + "=" * 100)
    print("=== [4/5 · Diagnostic 2] Macro Liquidity Multiplier (K_macro) Ablation ===")
    print("=" * 100)
    print(gate_df.to_string(index=False))

    # 5. 分年度子周期稳健性检验 (2020 - 2026)
    df_oos["year"] = pd.to_datetime(df_oos["date"]).dt.year
    yr_rows = []
    for yr, grp in df_oos.groupby("year"):
        d_ret = grp.groupby("date")["net_r_5"].mean()
        if len(d_ret) < 20:
            continue
        ann_r = d_ret.mean() * 252.0
        ann_v = d_ret.std(ddof=1) * np.sqrt(252.0)
        nav = (1.0 + d_ret).cumprod()
        mdd = (nav / nav.cummax() - 1.0).min()
        yr_rows.append({
            "Calendar Year": yr,
            "Trading Days": len(d_ret),
            "Ann. Return": f"{ann_r * 100:+.1f}%",
            "Ann. Vol": f"{ann_v * 100:.1f}%",
            "Max DD": f"{mdd * 100:.1f}%",
        })

    yr_df = pd.DataFrame(yr_rows)
    yr_df.to_latex(out_dir / "WP01_Diag3_SubPeriod_Stability.tex", index=False)
    print("\n" + "=" * 100)
    print("=== [5/5 · Diagnostic 3] Annual Sub-Period Robustness (2020 - 2026) ===")
    print("=" * 100)
    print(yr_df.to_string(index=False))
    print(f"\n[SUCCESS] All WP-01 LaTeX verification tables exported to: {out_dir.resolve()}")


if __name__ == "__main__":
    sample_file = root_dir / "data" / "us_sp100_wp01_sample.parquet"
    out_dir = root_dir / "output_wp01"
    raw_df = load_or_fetch_us_sample(sample_file, force_refresh=False)
    print("[2/5] Computing WP-01 potential-well topology & amortized resistance ...")
    feat_df = compute_wp01_physics(raw_df)
    run_full_wp01_verification(feat_df, out_dir)
