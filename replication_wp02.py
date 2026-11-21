#!/usr/bin/env python3
"""
QuantFlow Macro - Working Paper 02 Official Replication & Verification Suite
Title: Microstructure Dark-Pool Decomposition, Kinetic Envelope Expansion,
       and 2D State-Phase Topology in US Equity Markets
Author: Kevin Z. Qiu (QuantFlow Macro Research Institute)
SSRN Archive: https://ssrn.com/abstract=7667578
DOI:          10.2139/ssrn.7667578
Repository:   https://github.com/quantflowmacro/fpmm-market-mechanics

Academic Replication Notice:
This open-source replication suite implements the canonical 2D state-phase topology
and empirical ablation protocols defined in Qiu (2026), Working Paper 02.
Stylized surrogate proxies and state-driven phase transitions are used for
reproducible scientific demonstration; institutional high-frequency dark-pool models,
intraday trade flow execution pipelines, and proprietary fund hyperparameters
remain strictly decoupled and confidential under QuantFlow Macro IP guidelines.
"""

from dataclasses import dataclass
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


# ==============================================================================
# 0. 全局配置与样本池定义 (S&P 100 20只核心样本)
# ==============================================================================
SP100_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "AVGO", "TSLA", "JPM", "UNH",
    "XOM", "LLY", "V", "PG", "COST",
    "HD", "MA", "ABBV", "MRK", "PEP",
]

CACHE_DIR = root_dir / "data"
CACHE_FILE = CACHE_DIR / "us_sp100_wp01_sample.parquet"
OUTPUT_DIR = root_dir / "output_wp02"

START_DATE = "2020-01-01"
END_DATE = "2026-10-25"


@dataclass(frozen=True)
class CanonicalWP02Config:
    """
    Stylized Academic Baseline Configuration for WP-02 Open-Source Replication.
    Note: Production execution thresholds, adaptive sizing, and proprietary data
    pipelines are omitted in accordance with institutional IP guidelines.
    """
    # Theoretical phase boundaries published in WP-02 Section 5.2
    theta_pde: float = 0.14
    theta_pure_dark: float = 0.70
    theta_co_dark: float = 0.40

    # Stylized symmetric weights for Level-1 OHLCV dark-pool surrogate
    w_compression: float = 0.40
    w_clv: float = 0.40
    w_smoothness: float = 0.20

    # Empirical US FINRA TRF bounds [15%, 72%]
    dark_floor: float = 0.15
    dark_ceil: float = 0.72

    # Dark-pool capacity multiplier parameters (WP-02 Section 5.2)
    lambda_dark: float = 0.47
    m_dark_max: float = 1.45

    # Standard swing-holding hysteresis horizon
    max_hold_days: int = 10


# ==============================================================================
# 1. 外部基准因子加载 (Kenneth French FF5 + Momentum) 与鲁棒下载器
# ==============================================================================
def fetch_ff6_factors() -> pd.DataFrame | None:
    """修复列表索引解析与表头匹配的 Kenneth French FF5 + Momentum 日度因子下载器。"""
    ff5_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

    def _read_kf_zip(url: str) -> pd.DataFrame:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
            csv_name = zf.namelist()[0]
            raw_lines = zf.read(csv_name).decode("latin-1").splitlines()
        data_rows = []
        header = None
        for line in raw_lines:
            parts = [p.strip() for p in line.split(",")]
            if header is None and len(parts) >= 2 and (parts[0] == "" or not parts[0].isdigit()) and ("Mkt-RF" in parts or "Mom" in parts):
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


# ==============================================================================
# 2. 数据加载与预处理 (优先复用 WP-01 缓存)
# ==============================================================================
def load_or_fetch_data() -> pd.DataFrame:
    """
    优先复用本地缓存 data/us_sp100_wp01_sample.parquet；
    若无缓存则通过 yfinance 自动拉取 SP100_SAMPLE 历史日线并生成缓存。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        print(f"[INFO] 检测到本地缓存，直接加载: {CACHE_FILE}")
        df = pd.read_parquet(CACHE_FILE)
        col_map = {c: c.lower() for c in df.columns}
        df = df.rename(columns=col_map)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        return df

    import yfinance as yf
    print(f"[INFO] 未检测到本地缓存，正在通过 yfinance 拉取 {len(SP100_SAMPLE)} 只样本股票数据...")
    raw = yf.download(
        tickers=SP100_SAMPLE,
        start=START_DATE,
        end=END_DATE,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    records = []
    for ticker in SP100_SAMPLE:
        if ticker not in raw.columns.get_level_values(0):
            continue
        sub = raw[ticker].dropna(how="all").copy()
        if sub.empty:
            continue
        sub["ticker"] = ticker
        sub = sub.reset_index()
        date_col = "Date" if "Date" in sub.columns else sub.columns[0]
        sub = sub.rename(columns={
            date_col: "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        records.append(sub[["date", "ticker", "open", "high", "low", "close", "volume"]])

    df = pd.concat(records, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["s_float"] = 1e9
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df.to_parquet(CACHE_FILE, index=False)
    print(f"[INFO] 数据拉取完成并已缓存至: {CACHE_FILE} (共 {len(df):,} 条记录)")
    return df


# ==============================================================================
# 3. 核心微结构暗池算子与 2D 相态拓扑计算 (WP-02 Canonical Open-Source)
# ==============================================================================
def compute_stylized_dark_surrogate(
    df: pd.DataFrame, cfg: CanonicalWP02Config = CanonicalWP02Config()
) -> pd.DataFrame:
    """
    Computes a stylized Level-1 daily OHLCV proxy for Dark_Ratio and M_dark.
    Stylized symmetric weights for reproducible open-source demonstration.
    """
    out = df.copy()
    hi = out["high"]
    lo = out["low"]
    cl = out["close"]
    op = out["open"]

    # 1. Intra-bar range compression (normalized against 20-day rolling range)
    out["_bar_range"] = (hi - lo) / cl.replace(0, np.nan)
    avg_range = out.groupby("ticker")["_bar_range"].transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )
    compression = np.clip(1.0 - (out["_bar_range"] / avg_range.replace(0, np.nan)), 0.0, 1.0).fillna(0.5)
    out.drop(columns=["_bar_range"], inplace=True)

    # 2. Close Location Value (CLV) mapped to [0, 1]
    hl_span = (hi - lo).replace(0, np.nan)
    clv_raw = ((cl - lo) - (hi - cl)) / hl_span
    close_loc = np.clip((clv_raw.fillna(0.0) + 1.0) / 2.0, 0.0, 1.0)

    # 3. Body-to-range smoothness (institutional VWAP/TWAP execution signature)
    body_ratio = (cl - op).abs() / hl_span
    smoothness = np.clip(1.0 - body_ratio.fillna(0.5), 0.0, 1.0)

    # Stylized symmetric combination mapped onto [15%, 72%]
    raw_surrogate = (
        cfg.w_compression * compression
        + cfg.w_clv * close_loc
        + cfg.w_smoothness * smoothness
    )
    out["dark_ratio"] = np.clip(
        cfg.dark_floor + (cfg.dark_ceil - cfg.dark_floor) * raw_surrogate,
        cfg.dark_floor,
        cfg.dark_ceil,
    )

    # Dynamic dark-pool capacity multiplier M_dark (WP-02 Eq. 14)
    # Active only when net directional work is non-negative
    active_dark = np.where(out["vwap_pde"] >= 0.0, out["dark_ratio"], 0.0)
    out["m_dark"] = np.clip(1.0 + cfg.lambda_dark * active_dark, 1.0, cfg.m_dark_max)

    return out


def classify_2d_microstructure_state(
    df: pd.DataFrame, cfg: CanonicalWP02Config = CanonicalWP02Config()
) -> tuple[pd.Series, pd.Series]:
    """
    Assigns canonical 2D microstructure phase states (S0-S5) defined in WP-02 Section 5.2.
    Returns (short_phase_codes, full_phase_labels).
    """
    pde = df["vwap_pde"]
    dr = df["dark_ratio"]

    conditions = [
        (pde < -cfg.theta_pde),                                              # S5: Distribution / Negative Work
        (pde >= cfg.theta_pde) & (dr >= cfg.theta_pure_dark),                # S1: Pure Dark Propulsion
        (pde >= cfg.theta_pde) & (dr >= cfg.theta_co_dark),                  # S2: Co-Propulsion (Lit + Dark)
        (pde >= cfg.theta_pde) & (dr < cfg.theta_co_dark),                   # S4: Lit-Dominated Push
        (pde.abs() < cfg.theta_pde) & (dr >= cfg.theta_co_dark),             # S3: Covert Dark Accumulation
    ]
    code_choices = ["S5", "S1", "S2", "S4", "S3"]
    label_choices = [
        "S5: Distribution / Negative Work",
        "S1: Pure Dark Propulsion",
        "S2: Co-Propulsion (Lit + Dark)",
        "S4: Lit-Dominated Push",
        "S3: Covert Dark Accumulation",
    ]

    phase_code = pd.Series(np.select(conditions, code_choices, default="S0"), index=df.index)
    phase_label = pd.Series(np.select(conditions, label_choices, default="S0: Laminar Neutral"), index=df.index)
    return phase_code, phase_label


def generate_ablation_signals(
    df_oos: pd.DataFrame, use_dark_gate: bool = True
) -> tuple[pd.Series, pd.Series]:
    """
    Pure state-driven signal generation for open-source ablation replication.
    Eliminates proprietary micro-thresholds and dynamic leverage scaling.
    """
    states = df_oos["phase_state"]

    # Baseline (Lit-Only): enters on any positive propulsion state (S1, S2, S4)
    # WP-02 Dark-Gated: filters out retail-dominated lit breakouts (S4), keeping S1 & S2
    if use_dark_gate:
        entry_signal = states.isin(["S1", "S2"])
    else:
        entry_signal = states.isin(["S1", "S2", "S4"])

    # Exit when microstructure transitions into negative-work distribution (S5)
    exit_signal = states == "S5"

    return entry_signal, exit_signal


def compute_equal_weight_position(
    entry_signal: pd.Series, exit_signal: pd.Series, max_hold: int = 10
) -> np.ndarray:
    """
    Standard unlevered (1.0x) state-hysteresis holding protocol.
    Replaces proprietary dynamic leverage sizing.
    """
    n = len(entry_signal)
    weights = np.zeros(n, dtype=float)
    in_pos = False
    hold_days = 0

    ent_arr = entry_signal.to_numpy(dtype=bool)
    ext_arr = exit_signal.to_numpy(dtype=bool)

    for i in range(n):
        if not in_pos:
            if ent_arr[i]:
                in_pos = True
                hold_days = 1
                weights[i] = 1.0  # Strictly unlevered 1.0x unit weight
        else:
            hold_days += 1
            if ext_arr[i] or hold_days > max_hold:
                in_pos = False
                hold_days = 0
                weights[i] = 0.0
            else:
                weights[i] = 1.0

    return weights


def compute_wp02_dark_pool_operators(
    df: pd.DataFrame, cfg: CanonicalWP02Config = CanonicalWP02Config()
) -> pd.DataFrame:
    """校准 Level-1 暗池纯度分位数分布、阻力卷积与 2D 微观拓扑相态"""
    df = df.sort_values(["ticker", "date"]).copy()

    # 1. 基础物理量 (与 WP-01 严格对齐)
    df["vwap"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    if "s_float" in df.columns and df["s_float"].notna().all():
        df["ft_real"] = (df["volume"] / df["s_float"]) * 100.0
    else:
        vol_med_60 = df.groupby("ticker")["volume"].transform(lambda s: s.rolling(60, min_periods=10).median())
        df["ft_real"] = (df["volume"] / (vol_med_60 + 1e-8)) * 0.80

    gamma_min = 0.80
    df["vwap_pde"] = (((df["close"] - df["vwap"]) / df["vwap"]) * 100.0) / np.maximum(df["ft_real"], gamma_min)

    # 2. Level-1 暗池代理与容量扩张乘数 M_dark (WP-02 Section 5.2)
    df = compute_stylized_dark_surrogate(df, cfg=cfg)

    # 3. 250日阻力密度 RD 与势能 Phi (论文 Section 3 严格物理卷积)
    lam = 0.985
    weights = lam ** np.arange(249, -1, -1)

    def _per_stock(g: pd.DataFrame) -> pd.DataFrame:
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
            lo_m = np.min(lo_arr[i - 249 : i + 1])
            hi_m = np.max(hi_arr[i - 249 : i + 1])
            if hi_m > lo_m:
                phi[i] = np.clip((c_now - lo_m) / (hi_m - lo_m), 0.0, 1.0)
        out = g.copy()
        out["rd"] = rd
        out["phi"] = phi
        return out

    df = pd.concat([_per_stock(g) for _, g in df.groupby("ticker", sort=False)], ignore_index=True)
    df["ft_max_lit"] = 4.5 - 2.8 * df["phi"]
    df["ft_max_full"] = df["ft_max_lit"] * df["m_dark"]
    df["z_imp"] = np.where(df["vwap_pde"] > 0, df["rd"] / np.maximum(df["vwap_pde"], 0.05), np.inf)

    # 4. 2D 力学状态相图 (WP-02 Section 5.2 理论边界)
    p_code, p_label = classify_2d_microstructure_state(df, cfg=cfg)
    df["phase_state"] = p_code
    df["state_2d"] = p_label

    open_t1 = df.groupby("ticker")["open"].shift(-1)
    df["fwd_1d"] = (df.groupby("ticker")["open"].shift(-2) - open_t1) / open_t1
    df["fwd_5d"] = (df.groupby("ticker")["open"].shift(-6) - open_t1) / open_t1
    df["fwd_10d"] = (df.groupby("ticker")["open"].shift(-11) - open_t1) / open_t1
    return df


# ==============================================================================
# 4. 表格生成与回测消融验证 (继承 WP-01 状态滞后波段持有协议)
# ==============================================================================
def run_hysteresis_backtest(
    df_oos: pd.DataFrame,
    ent_mask: pd.Series,
    exit_mask: pd.Series,
    max_hold: int = 10,
    cost_bps: float = 5.0,
) -> tuple[pd.Series, float]:
    """严格继承 WP-01 状态滞后波段持有引擎 (标准 1.0x 等权持有)"""
    cost_rate = cost_bps / 10000.0
    pos_list = []
    hold_list = []

    for ticker, sub in df_oos.groupby("ticker", sort=False):
        e_sub = ent_mask.loc[sub.index]
        x_sub = exit_mask.loc[sub.index]
        weights = compute_equal_weight_position(e_sub, x_sub, max_hold=max_hold)

        in_p = False
        start_idx = 0
        for i in range(len(weights)):
            if not in_p and weights[i] > 0:
                in_p = True
                start_idx = i
            elif in_p and (weights[i] == 0 or i == len(weights) - 1):
                in_p = False
                hold_list.append(i - start_idx)

        s_out = pd.DataFrame(
            {"date": sub["date"], "fwd_1d": sub["fwd_1d"], "pos": weights},
            index=sub.index,
        )
        pos_list.append(s_out)

    panel = pd.concat(pos_list).sort_index()
    pos_chg = np.abs(panel["pos"] - panel.groupby(df_oos["ticker"])["pos"].shift(1).fillna(0))
    net_ret = panel["pos"] * panel["fwd_1d"] - pos_chg * cost_rate
    daily = net_ret.groupby(panel["date"]).mean()
    avg_h = float(np.mean(hold_list)) if hold_list else 0.0
    return daily, avg_h


def compute_metrics(daily_ret: pd.Series, avg_hold: float) -> dict:
    daily_ret = daily_ret.dropna()
    ann_r = daily_ret.mean() * 252.0
    ann_v = daily_ret.std(ddof=1) * np.sqrt(252.0)
    cum = (1.0 + daily_ret).cumprod()
    mdd = (cum / cum.cummax() - 1.0).min() if len(cum) > 0 else 0.0
    sr = (ann_r - 0.035) / ann_v if ann_v > 0 else 0.0
    return {
        "Ann. Return (%)": f"{ann_r * 100:+.2f}%",
        "Ann. Vol (%)": f"{ann_v * 100:.2f}%",
        "Max Drawdown (%)": f"{mdd * 100:.2f}%",
        "Sharpe Ratio": f"{sr:.2f}",
        "Avg Hold (Days)": f"{avg_hold:.1f}",
    }


def verify_wp02_tables(df: pd.DataFrame, cfg: CanonicalWP02Config = CanonicalWP02Config()) -> None:
    """生成、打印并导出 WP-02 两张核心实证 LaTeX 表格至 output_wp02/"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_oos = df[(df["date"] >= "2020-01-02") & df["phi"].notna()].copy()

    # --- 表 1：2D 力学状态相图与前瞻收益表 (均衡分布 + NaN 兜底) ---
    valid_df = df_oos.dropna(subset=["fwd_1d", "fwd_5d", "fwd_10d"]).copy()
    total_obs = max(len(valid_df), 1)

    state_order = [
        "S1: Pure Dark Propulsion",
        "S2: Co-Propulsion (Lit + Dark)",
        "S3: Covert Dark Accumulation",
        "S4: Lit-Dominated Push",
        "S5: Distribution / Negative Work",
        "S0: Laminar Neutral",
    ]

    t1_rows = []
    for st in state_order:
        sub = valid_df[valid_df["state_2d"] == st]
        n_sub = len(sub)
        t1_rows.append({
            "2D Microstructure State": st,
            "Sample Share (%)": f"{n_sub / total_obs * 100:.2f}%" if n_sub > 0 else "-",
            "Mean VWAP_PDE": f"{sub['vwap_pde'].mean():.3f}" if n_sub > 0 else "-",
            "Mean Dark_Ratio (%)": f"{sub['dark_ratio'].mean() * 100:.1f}%" if n_sub > 0 else "-",
            "Mean M_dark": f"{sub['m_dark'].mean():.3f}" if n_sub > 0 else "-",
            "Fwd 1D Ret (%)": f"{sub['fwd_1d'].mean() * 100:+.3f}%" if n_sub > 0 else "-",
            "Fwd 5D Ret (%)": f"{sub['fwd_5d'].mean() * 100:+.3f}%" if n_sub > 0 else "-",
            "Fwd 10D Ret (%)": f"{sub['fwd_10d'].mean() * 100:+.3f}%" if n_sub > 0 else "-",
        })

    table1_df = pd.DataFrame(t1_rows)
    print("\n" + "=" * 96)
    print("Table 1: 2D Microstructure Phase Diagram & Causal t+1 Open Forward Returns")
    print("=" * 96)
    print(table1_df.to_string(index=False))

    tex1_path = OUTPUT_DIR / "WP02_Table1_2D_States.tex"
    try:
        tex1_body = table1_df.to_latex(index=False)
    except Exception:
        tex1_body = table1_df.to_string(index=False)
    tex1_str = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{WP-02: 2D Microstructure Phase Diagram (VWAP\\_PDE $\\times$ Dark\\_Ratio) and Forward Returns}\n"
        "\\label{tab:wp02_2d_states}\n"
        f"{tex1_body}\n"
        "\\end{table}\n"
    )
    tex1_path.write_text(tex1_str, encoding="utf-8")

    # --- 表 2：暗池包络扩张单变量消融实验对比表 (纯相态枚举驱动 + 无杠杆等权持有) ---
    ent_lit, exit_lit = generate_ablation_signals(df_oos, use_dark_gate=False)
    ent_wp02, exit_wp02 = generate_ablation_signals(df_oos, use_dark_gate=True)

    # 叠加包络线物理容量门控 (Baseline 限 Lit 包络，WP-02 允许 M_dark 扩张包络)
    ent_lit = ent_lit & (df_oos["ft_real"] <= df_oos["ft_max_lit"])
    ent_wp02 = ent_wp02 & (df_oos["ft_real"] <= df_oos["ft_max_full"])

    d_lit, h_lit = run_hysteresis_backtest(df_oos, ent_lit, exit_lit, max_hold=cfg.max_hold_days, cost_bps=5.0)
    d_wp02, h_wp02 = run_hysteresis_backtest(df_oos, ent_wp02, exit_wp02, max_hold=cfg.max_hold_days, cost_bps=5.0)
    d_stress, h_stress = run_hysteresis_backtest(df_oos, ent_wp02, exit_wp02, max_hold=cfg.max_hold_days, cost_bps=15.0)

    m_lit = compute_metrics(d_lit, h_lit)
    m_wp02 = compute_metrics(d_wp02, h_wp02)
    m_stress = compute_metrics(d_stress, h_stress)

    alpha_wp02, t_wp02 = run_factor_regression(d_wp02, df_oos)
    alpha_lit, t_lit = run_factor_regression(d_lit, df_oos)

    table2_df = pd.DataFrame([
        {
            "Strategy Specification": "Baseline: Lit-Only Envelope (No Dark Filter)",
            **m_lit,
            "Net Alpha": alpha_lit,
            "NW t-Stat": t_lit,
        },
        {
            "Strategy Specification": "WP-02: Dark-Pool M_dark Engine (5 bps)",
            **m_wp02,
            "Net Alpha": alpha_wp02,
            "NW t-Stat": t_wp02,
        },
        {
            "Strategy Specification": "WP-02: Dark-Pool M_dark Stressed (15 bps)",
            **m_stress,
            "Net Alpha": "-",
            "NW t-Stat": "-",
        },
    ])

    print("\n" + "=" * 96)
    print("Table 2: Dark-Pool Decomposition & M_dark Envelope Ablation Study (t+1 Open Execution)")
    print("=" * 96)
    print(table2_df.to_string(index=False))

    tex2_path = OUTPUT_DIR / "WP02_Table2_DarkPool_Ablation.tex"
    try:
        tex2_body = table2_df.to_latex(index=False)
    except Exception:
        tex2_body = table2_df.to_string(index=False)
    tex2_str = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{WP-02: Ablation Study of Dark-Pool Decomposition and Envelope Multiplier ($M_{\\text{dark}}$)}\n"
        "\\label{tab:wp02_ablation}\n"
        f"{tex2_body}\n"
        "\\end{table}\n"
    )
    tex2_path.write_text(tex2_str, encoding="utf-8")

    print(f"\n[SUCCESS] 全部 LaTeX 表格已成功导出至目录: {OUTPUT_DIR.resolve()}/")
    print(f"  -> {tex1_path}")
    print(f"  -> {tex2_path}")


# ==============================================================================
# 5. 主入口
# ==============================================================================
def main():
    t0 = time.perf_counter()
    print(">>> 启动 QuantFlow Macro WP-02 暗池分解与 2D 力学状态相图实证复现脚本...")
    cfg = CanonicalWP02Config()
    df_raw = load_or_fetch_data()
    df_ops = compute_wp02_dark_pool_operators(df_raw, cfg=cfg)
    verify_wp02_tables(df_ops, cfg=cfg)
    elapsed = time.perf_counter() - t0
    print(f">>> 全部实证检验与表格导出完成，总耗时: {elapsed:.2f} 秒。")


if __name__ == "__main__":
    main()
