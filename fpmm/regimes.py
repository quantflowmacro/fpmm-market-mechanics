"""
First-Principles Market Mechanics (FPMM) - Baseline Regime Classifier
QuantFlow Macro Research Institute (c) 2026
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def compute_potential_well(series_close: pd.Series, window: int = 20) -> pd.Series:
    """
    Computes the dimensionless potential well depth Phi(x).
    Harmonic oscillator approximation: Phi(x) ~ 0.5 * z^2.
    """
    rolling_mean = series_close.rolling(window=window).mean()
    rolling_std = series_close.rolling(window=window).std().replace(0, 1e-6)
    z_score = (series_close - rolling_mean) / rolling_std
    phi = 0.5 * np.square(z_score)
    return phi


def classify_market_regime(df: pd.DataFrame) -> pd.Series:
    """
    Classifies market microstructure into 7 discrete thermodynamic phase regimes.
    Requires columns: ['close', 'volume']
    """
    close = df["close"]
    volume = df["volume"]

    phi = compute_potential_well(close)
    turnover_ratio = volume / volume.rolling(window=20).mean().replace(0, 1e-6)
    returns = close.pct_change()
    volatility = returns.rolling(window=10).std()

    regimes: List[str] = []
    for p, tr, ret, vol in zip(phi, turnover_ratio, returns, volatility):
        if np.isnan(p) or np.isnan(tr):
            regimes.append("Regime VII: Kinetic Neutral")
            continue

        if p < 0.35 and ret > 0:
            regimes.append("Regime I: Superconducting Propulsion")
        elif p >= 0.75 and tr > 1.8 and ret > 0:
            regimes.append("Regime II: Potential Well Escape")
        elif vol < 0.005 and tr < 0.8:
            regimes.append("Regime III: Cryogenic Equilibrium")
        elif tr > 2.0 and abs(ret) < 0.002:
            regimes.append("Regime IV: Crocodile Turbulent Dissipation")
        elif p > 1.2 and ret < -0.015:
            regimes.append("Regime V: Gravitational Collapse")
        elif 0.35 <= p <= 0.75:
            regimes.append("Regime VI: Potential Well Oscillation")
        else:
            regimes.append("Regime VII: Kinetic Neutral")

    return pd.Series(regimes, index=df.index, name="FPMM_Regime")
