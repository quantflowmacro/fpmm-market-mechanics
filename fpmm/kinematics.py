"""
First-Principles Market Mechanics (FPMM) - Kinematic Foundations
QuantFlow Macro Research Institute (c) 2026

Formulates equity price formation as particle dynamics within a non-linear
gravitational potential well constrained by institutional order flow.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd


def compute_velocity(series_close: pd.Series, window: int = 5, normalized: bool = True) -> pd.Series:
    """
    Computes price velocity v = dx / dt.

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


def compute_potential_well(series_close: pd.Series, window: int = 20) -> pd.Series:
    """
    Computes the dimensionless potential well depth Phi(x).
    Harmonic oscillator approximation: Phi(x) ~ 0.5 * z^2.

    Parameters
    ----------
    series_close : pd.Series
        Asset close price series.
    window : int, default 20
        Lookback window for equilibrium mean and standard deviation.

    Returns
    -------
    pd.Series
        Dimensionless potential energy Phi.
    """
    rolling_mean = series_close.rolling(window=window).mean()
    rolling_std = series_close.rolling(window=window).std().replace(0, 1e-6)
    z_score = (series_close - rolling_mean) / rolling_std
    phi = 0.5 * np.square(z_score)
    return phi


def compute_turnover_envelope(phi: Union[pd.Series, np.ndarray], alpha: float = 1.8, beta: float = 0.4) -> Union[pd.Series, np.ndarray]:
    """
    Computes critical turnover envelope FT_max(Phi).
    Represents the escape velocity threshold required for a particle to transcend
    potential barriers under institutional frictional damping.

    Parameters
    ----------
    phi : pd.Series or np.ndarray
        Dimensionless potential energy series.
    alpha : float, default 1.8
        Base critical turnover threshold factor.
    beta : float, default 0.4
        Coupling coefficient scaling with potential depth.

    Returns
    -------
    pd.Series or np.ndarray
        Dynamic critical turnover envelope FT_max.
    """
    return alpha + beta * np.sqrt(np.maximum(0, phi))


def compute_vwap_work(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculates institutional VWAP displacement work W_VWAP = (P - VWAP) * Volume.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame containing ['close', 'volume'] (and optionally 'high', 'low').
    window : int, default 20
        Rolling window for VWAP baseline calculation.

    Returns
    -------
    pd.Series
        Normalized institutional displacement work.
    """
    if "high" in df.columns and "low" in df.columns:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    else:
        typical_price = df["close"]

    pv = typical_price * df["volume"]
    rolling_pv = pv.rolling(window=window).sum()
    rolling_vol = df["volume"].rolling(window=window).sum().replace(0, 1e-6)
    vwap = rolling_pv / rolling_vol

    displacement = (df["close"] - vwap) / vwap
    return displacement
