"""
First-Principles Market Mechanics (FPMM) - Naturally Orthogonal Regime Classifier
QuantFlow Macro Research Institute (c) 2026

Implements the 7 mutually exclusive and collectively exhaustive (MECE)
thermodynamic phase regimes defined in Qiu (2026), Section 5 & Theorem 6.3.
Decoupled architecture utilizing a parameterized configuration dataclass.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
import numpy as np
import pandas as pd
from .kinematics import (
    compute_critical_turnover_envelope,
    compute_overhead_resistance_density,
    compute_potential_well,
    compute_vwap_pde,
    compute_work_impedance,
)


class Regime(str, Enum):
    """
    Theorem 6.3 - 7 Orthogonal MECE Thermodynamic Trading Regimes.
    Aligned with WP-01 (V1.1) official monograph.
    """
    R1_SUPERCONDUCTING_ESCAPE = "R1_SUPERCONDUCTING_ESCAPE"
    R2_CONTROLLED_ESCAPE = "R2_CONTROLLED_ESCAPE"
    R3_POTENTIAL_WELL_ACCUM = "R3_POTENTIAL_WELL_ACCUM"
    R4_CROCODILE_DISSIPATION = "R4_CROCODILE_DISSIPATION"
    R5_GRAVITATIONAL_COLLAPSE = "R5_GRAVITATIONAL_COLLAPSE"
    R6_BOUNDARY_LOCK = "R6_BOUNDARY_LOCK"
    R7_KINETIC_NEUTRAL = "R7_KINETIC_NEUTRAL"


@dataclass
class RegimeConfig:
    """
    Parametric configuration for thermodynamic phase regime state-space partitioning.
    Allows researchers to specify customized or estimated calibration priors.

    Parameters
    ----------
    gamma_min : float, default 0.80
        Lower-quartile damping floor for VWAP displacement to eliminate low-liquidity singularities.
    phi_valley_bound : float, default 0.35
        Potential coordinate boundary separating the lower valley from the upper corridor.
    pde_super_min : float, default 0.15
        Minimum barycentric mechanical efficiency required for superconducting propulsion.
    pde_escape_min : float, default 0.05
        Minimum barycentric mechanical efficiency required for controlled well escape.
    pde_collapse_max : float, default -0.15
        Maximum negative barycentric efficiency threshold triggering gravitational collapse.
    ret_collapse_max : float, default -0.015
        Maximum 1-day return threshold confirming downward kinetic failure (-1.5%).
    ret_lock_min : float, default 0.095
        Minimum 1-day return threshold triggering boundary lock state (+9.5%).
    wick_lock_max : float, default 0.05
        Maximum upper shadow wick ratio for boundary liquidity lock.
    wick_turb_min : float, default 0.45
        Minimum upper shadow wick ratio indicating fluid shear collision.
    z_crit_super : float, default 120.0
        Upper critical impedance bound for superconducting propulsion.
    z_crit_escape : float, default 450.0
        Upper critical impedance bound for controlled well escape.
    base_a : float, default 4.50
        Base valley turnover ceiling at Phi = 0 (4.50% for US Large-Cap S&P 100).
    base_b : float, default 2.80
        Linear altitude compression slope (2.80%).
    kappa : float, default 0.20
        Institutional dark-pool expansion multiplier.
    beta_sigma : float, default 1.00
        Idiosyncratic volatility elasticity scalar.
    """
    gamma_min: float = 0.80
    phi_valley_bound: float = 0.35
    pde_super_min: float = 0.15
    pde_escape_min: float = 0.05
    pde_collapse_max: float = -0.15
    ret_collapse_max: float = -0.015
    ret_lock_min: float = 0.095
    wick_lock_max: float = 0.05
    wick_turb_min: float = 0.45
    z_crit_super: float = 120.0
    z_crit_escape: float = 450.0
    base_a: float = 4.50
    base_b: float = 2.80
    kappa: float = 0.20
    beta_sigma: float = 1.00


def classify_market_regime(
    df: pd.DataFrame,
    config: Optional[RegimeConfig] = None,
    turnover_col: Optional[str] = None,
    dark_ratio_col: Optional[str] = None,
    phi_series: Optional[pd.Series] = None,
    rd_series: Optional[pd.Series] = None,
    vwap_pde_series: Optional[pd.Series] = None,
    ft_max_series: Optional[Union[pd.Series, np.ndarray]] = None,
    z_imp_series: Optional[Union[pd.Series, np.ndarray]] = None,
) -> pd.Series:
    """
    Classifies market microstructure state vector Psi_t into 7 discrete,
    mutually exclusive, and collectively exhaustive (MECE) thermodynamic phase regimes.

    State space is partitioned according to Theorem 6.3 in Qiu (2026),
    strictly aligned with WP-01 (V1.1) empirical benchmarks.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with ['open', 'high', 'low', 'close', 'volume'].
    config : RegimeConfig, optional
        Calibration parameter configuration instance. If None, uses default baseline parameters.
    turnover_col : str, optional
        Column name for active free-float turnover percentage.
    dark_ratio_col : str, optional
        Column name for off-exchange dark pool volume percentage.
    phi_series, rd_series, vwap_pde_series, ft_max_series, z_imp_series : pd.Series, optional
        Precomputed kinematic observables. If None, computes on the fly.

    Returns
    -------
    pd.Series
        Series of classified FPMM regimes indexed by DataFrame index.
    """
    cfg = config or RegimeConfig()

    close = df["close"]
    open_p = df["open"] if "open" in df.columns else close.shift(1).fillna(close)
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close

    # 1. Kinematic Observables
    phi = phi_series if phi_series is not None else compute_potential_well(df)
    rd = rd_series if rd_series is not None else compute_overhead_resistance_density(df)
    vwap_pde = vwap_pde_series if vwap_pde_series is not None else compute_vwap_pde(df, gamma_min=cfg.gamma_min, turnover_col=turnover_col)
    z_imp = z_imp_series if z_imp_series is not None else compute_work_impedance(rd, vwap_pde)

    # Active turnover rate FT_real
    if turnover_col and turnover_col in df.columns:
        ft_real = df[turnover_col]
    elif "volume" in df.columns:
        rolling_vol = df["volume"].rolling(window=20, min_periods=1).mean().replace(0, 1e-6)
        ft_real = (df["volume"] / rolling_vol) * 1.0
    else:
        ft_real = pd.Series(1.0, index=df.index)

    dark_ratio = df[dark_ratio_col] if dark_ratio_col and dark_ratio_col in df.columns else pd.Series(0.0, index=df.index)
    if ft_max_series is not None:
        ft_max = ft_max_series
    else:
        ft_max = compute_critical_turnover_envelope(
            phi,
            dark_ratio=dark_ratio,
            base_a=cfg.base_a,
            base_b=cfg.base_b,
            kappa=cfg.kappa,
            beta_sigma=cfg.beta_sigma,
        )

    # 2. Geometric Shape Characteristics
    hl_span = np.maximum(high - low, 1e-6)
    upper_wick = high - np.maximum(open_p, close)
    wick_ratio = upper_wick / hl_span
    prev_close = close.shift(1).fillna(close)
    ret_1d = (close - prev_close) / prev_close.replace(0, 1e-6)

    # 3. Theorem 6.3 MECE Decision Tree
    cond_r4 = (ft_real > ft_max) & ((wick_ratio > cfg.wick_turb_min) | (vwap_pde <= 0))
    cond_r5 = (vwap_pde < cfg.pde_collapse_max) & (ret_1d < cfg.ret_collapse_max)
    cond_r6 = (ret_1d >= cfg.ret_lock_min) & (wick_ratio < cfg.wick_lock_max)
    cond_r1 = (vwap_pde > cfg.pde_super_min) & (z_imp <= cfg.z_crit_super) & (ft_real <= ft_max)
    cond_r2 = (vwap_pde > cfg.pde_escape_min) & (z_imp <= cfg.z_crit_escape) & (ft_real <= ft_max)
    cond_r3 = (phi <= cfg.phi_valley_bound) & (ft_real <= ft_max)

    regimes = np.full(len(df), Regime.R7_KINETIC_NEUTRAL.value, dtype=object)
    regimes = np.where(cond_r3, Regime.R3_POTENTIAL_WELL_ACCUM.value, regimes)
    regimes = np.where(cond_r2, Regime.R2_CONTROLLED_ESCAPE.value, regimes)
    regimes = np.where(cond_r1, Regime.R1_SUPERCONDUCTING_ESCAPE.value, regimes)
    regimes = np.where(cond_r6, Regime.R6_BOUNDARY_LOCK.value, regimes)
    regimes = np.where(cond_r5, Regime.R5_GRAVITATIONAL_COLLAPSE.value, regimes)
    regimes = np.where(cond_r4, Regime.R4_CROCODILE_DISSIPATION.value, regimes)

    # Guard for NaN burn-in intervals
    nan_mask = phi.isna() | vwap_pde.isna()
    regimes = np.where(nan_mask, Regime.R7_KINETIC_NEUTRAL.value, regimes)

    return pd.Series(regimes, index=df.index, name="FPMM_Regime")


# Backward-compatible and functional alias
classify_regime = classify_market_regime
