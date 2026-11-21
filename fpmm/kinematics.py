"""
First-Principles Market Mechanics (FPMM) - Kinematic Foundations
QuantFlow Macro Research Institute (c) 2026

Formulates equity price formation as particle dynamics within a non-linear
gravitational potential well constrained by institutional order flow.
Strictly implements definitions from Qiu (2026), Section 2-4.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd


def compute_velocity(series_close: pd.Series, window: int = 5, normalized: bool = True) -> pd.Series:
    """
    Computes price velocity v = dx / dt on dimensionless log-price manifold.

    Parameters
    ----------
    series_close : pd.Series
        Asset close price series.
    window : int, default 5
        Smoothing window for instantaneous velocity approximation.
    normalized : bool, default True
        If True, returns logarithmic returns scaled per unit time.

    Returns
    -------
    pd.Series
        Instantaneous price velocity.
    """
    if normalized:
        log_price = np.log(series_close)
        velocity = (log_price - log_price.shift(window)) / window
    else:
        velocity = (series_close - series_close.shift(window)) / window
    return velocity


def compute_potential_well(
    df: pd.DataFrame,
    window: int = 250,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    """
    Computes the dimensionless range coordinate Phi(P_now) in [0, 1]
    normalized across the cyclical channel lookback (Equation 3.3 in Qiu, 2026):

        Phi(P_now) = clip((P_now - min(Low_N)) / (max(High_N) - min(Low_N)), 0, 1)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame containing at least close, high, low columns.
    window : int, default 250
        Lookback window in trading sessions (canonical N = 250 days, 52-week channel).
    price_col : str, default 'close'
        Current clearing price column name.
    high_col : str, default 'high'
        Period high price column name.
    low_col : str, default 'low'
        Period low price column name.

    Returns
    -------
    pd.Series
        Dimensionless potential coordinate Phi in [0, 1].
    """
    p_now = df[price_col]
    if high_col in df.columns and low_col in df.columns:
        rolling_high = df[high_col].rolling(window=window, min_periods=1).max()
        rolling_low = df[low_col].rolling(window=window, min_periods=1).min()
    else:
        rolling_high = p_now.rolling(window=window, min_periods=1).max()
        rolling_low = p_now.rolling(window=window, min_periods=1).min()

    denom = (rolling_high - rolling_low).replace(0, 1e-6)
    phi = (p_now - rolling_low) / denom
    return phi.clip(0.0, 1.0)


def compute_overhead_resistance_density(
    df: pd.DataFrame,
    window: int = 250,
    lambda_decay: float = 0.985,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
) -> pd.Series:
    """
    Computes measure-theoretic Overhead Resistance Density RD(P_now) with
    exponential memory decay (Equation 3.2 in Qiu, 2026):

        RD(P_now) = sum_{k=1}^N lambda^{N-k} * V_k * I(P_typical,k > P_now)
                    / sum_{k=1}^N lambda^{N-k} * V_k * 100%

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    window : int, default 250
        Lookback window N.
    lambda_decay : float, default 0.985
        Exponential memory decay constant (T_1/2 = 45.86 trading days).
    price_col, high_col, low_col, volume_col : str
        Column names.

    Returns
    -------
    pd.Series
        Overhead resistance density percentage RD in [0.0%, 100.0%].
    """
    close = df[price_col].values
    vol = df[volume_col].values if volume_col in df.columns else np.ones(len(df))

    if high_col in df.columns and low_col in df.columns:
        p_typ = (df[high_col].values + df[low_col].values + close) / 3.0
    else:
        p_typ = close

    n = len(df)
    rd_values = np.zeros(n, dtype=float)

    # Precompute decay powers for max window size
    weights_full = np.power(lambda_decay, np.arange(window - 1, -1, -1))

    for i in range(n):
        start_idx = max(0, i - window + 1)
        w_slice = weights_full[-(i - start_idx + 1):]
        v_slice = vol[start_idx : i + 1]
        typ_slice = p_typ[start_idx : i + 1]
        p_now = close[i]

        eff_vol = w_slice * v_slice
        denom = np.sum(eff_vol)
        if denom <= 0:
            rd_values[i] = 0.0
        else:
            numer = np.sum(eff_vol[typ_slice > p_now])
            rd_values[i] = (numer / denom) * 100.0

    return pd.Series(rd_values, index=df.index, name="RD")


def compute_vwap_pde(
    df: pd.DataFrame,
    gamma_min: float = 0.80,
    vwap_col: Optional[str] = None,
    turnover_col: Optional[str] = None,
) -> pd.Series:
    """
    Computes dimensionless Barycentric Mechanical Efficiency (VWAP_PDE / eta_W)
    (Definition 2.1 in Qiu, 2026):

        VWAP_PDE = [((P_close - P_VWAP) / P_VWAP) * 100%] / max(FT_real, gamma_min)

    Dimensional Unit Consistency Note:
    ----------------------------------
    Both numerator displacement and denominator active turnover (FT_real) are evaluated
    in percentage points (%):
      - Displacement: ((P_close - VWAP) / VWAP) * 100% (e.g., +1.20 for +1.20%).
      - Active Turnover (FT_real): Volume / S_float * 100% (e.g., 0.86 for 0.86%).
      - Damping Floor (gamma_min): 0.80 for 0.80% (canonical S&P 100 lower quartile).
    If decimal inputs (e.g. 0.0080 for 0.80%) are supplied, gamma_min must be scaled
    proportionally (0.0080) to preserve dimensional invariance eta_W in O(1).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    gamma_min : float, default 0.80
        Canonical lower-quartile damping floor (0.80% for S&P 100).
    vwap_col : str, optional
        Name of precomputed VWAP column. If None, computes typical price session VWAP.
    turnover_col : str, optional
        Name of active turnover rate column (in percentage points, e.g. 0.86 for 0.86%).
        If None, approximates via volume relative to rolling 20-session baseline.

    Returns
    -------
    pd.Series
        Barycentric propulsion deviation ratio VWAP_PDE.
    """
    close = df["close"]

    if vwap_col and vwap_col in df.columns:
        vwap = df[vwap_col]
    elif "high" in df.columns and "low" in df.columns and "volume" in df.columns:
        typical_price = (df["high"] + df["low"] + close) / 3.0
        pv = typical_price * df["volume"]
        vwap = pv.rolling(window=20, min_periods=1).sum() / df["volume"].rolling(window=20, min_periods=1).sum().replace(0, 1e-6)
    else:
        vwap = close.rolling(window=20, min_periods=1).mean()

    # Percentage displacement from VWAP (e.g., 0.0122 -> 1.22%)
    disp_pct = ((close - vwap) / vwap.replace(0, 1e-6)) * 100.0

    # Active turnover rate FT_real in percentage points
    if turnover_col and turnover_col in df.columns:
        ft_real = df[turnover_col]
    elif "volume" in df.columns:
        rolling_vol = df["volume"].rolling(window=20, min_periods=1).mean().replace(0, 1e-6)
        ft_real = (df["volume"] / rolling_vol) * 1.0  # normalized around 1.0%
    else:
        ft_real = pd.Series(1.0, index=df.index)

    damping_floor = np.maximum(ft_real, gamma_min)
    vwap_pde = disp_pct / damping_floor
    return pd.Series(vwap_pde, index=df.index, name="VWAP_PDE")


def compute_critical_turnover_envelope(
    phi: Union[pd.Series, np.ndarray, float],
    is_dark_pool: Union[bool, pd.Series, np.ndarray] = False,
    kappa: float = 0.20,
    beta_sigma: float = 1.00,
    base_a: float = 4.50,
    base_b: float = 2.80,
    dark_ratio: Optional[Union[pd.Series, np.ndarray, float]] = None,
    a: Optional[float] = None,
    b: Optional[float] = None,
) -> Union[pd.Series, np.ndarray, float]:
    """
    Computes Multiplicative Dynamic Critical Turnover Envelope FT_max(Phi)
    (Equation 4.2 in Qiu, 2026). Aligned with WP-01 (V1.1) S&P 100 Level-1 baseline (a = 4.50%, b = 2.80%).

        FT_max(Phi) = (1 + kappa * I_dark) * beta_sigma * (a - b * Phi)

    Parameters
    ----------
    phi : pd.Series or np.ndarray or float
        Dimensionless potential range coordinate in [0, 1].
    is_dark_pool : bool or pd.Series or np.ndarray, default False
        Boolean indicator or dark pool flag for institutional execution expansion.
    kappa : float, default 0.20
        Institutional dark-pool expansion multiplier (e.g. 0.20 for 1.20x expansion).
    beta_sigma : float, default 1.00
        Idiosyncratic volatility elasticity scalar.
    base_a : float, default 4.50
        Base valley turnover ceiling at Phi = 0 (4.50% for US Large-Cap S&P 100).
    base_b : float, default 2.80
        Linear altitude compression slope (2.80%).
    dark_ratio : pd.Series or np.ndarray or float, optional
        Off-exchange volume percentage (if provided and >= 40.0%, is_dark_pool evaluates to True).
    a, b : float, optional
        Aliases for base_a and base_b.

    Returns
    -------
    pd.Series or np.ndarray or float
        Dynamic critical turnover ceiling FT_max in percentage points.
    """
    actual_a = base_a if a is None else a
    actual_b = base_b if b is None else b

    if dark_ratio is not None:
        if isinstance(dark_ratio, (pd.Series, np.ndarray)):
            i_dark = np.where(dark_ratio >= 40.0, 1.0, 0.0)
        else:
            i_dark = 1.0 if dark_ratio >= 40.0 else 0.0
    elif isinstance(is_dark_pool, (pd.Series, np.ndarray)):
        i_dark = np.asarray(is_dark_pool, dtype=float)
    else:
        i_dark = 1.0 if bool(is_dark_pool) else 0.0

    dark_mult = 1.0 + kappa * i_dark

    if isinstance(phi, pd.Series):
        phi_val = phi.clip(0.0, 1.0)
        envelope = dark_mult * beta_sigma * (actual_a - actual_b * phi_val)
        return envelope.clip(lower=0.25)
    else:
        phi_val = np.clip(phi, 0.0, 1.0)
        envelope = dark_mult * beta_sigma * (actual_a - actual_b * phi_val)
        return np.maximum(envelope, 0.25)


# Backward-compatible alias
compute_turnover_envelope = compute_critical_turnover_envelope


def compute_work_impedance(
    rd: Union[pd.Series, np.ndarray],
    vwap_pde: Union[pd.Series, np.ndarray],
) -> Union[pd.Series, np.ndarray]:
    """
    Computes Work Impedance Ratio Z (Equation 5.1 in Qiu, 2026):

        Z = RD / VWAP_PDE  if VWAP_PDE > 0 else BLOCKED (+inf)

    Parameters
    ----------
    rd : pd.Series or np.ndarray
        Overhead resistance density percentage in [0, 100].
    vwap_pde : pd.Series or np.ndarray
        Barycentric propulsion deviation ratio.

    Returns
    -------
    pd.Series or np.ndarray
        Work impedance value Z (or np.inf if blocked).
    """
    is_series = isinstance(vwap_pde, pd.Series)
    pde_arr = vwap_pde.values if is_series else np.asarray(vwap_pde)
    rd_arr = rd.values if isinstance(rd, pd.Series) else np.asarray(rd)

    z_vals = np.where(pde_arr > 0, rd_arr / np.maximum(pde_arr, 1e-6), np.inf)

    if is_series:
        return pd.Series(z_vals, index=vwap_pde.index, name="Z_Imp")
    return z_vals
