"""
First-Principles Market Mechanics (FPMM) - Naturally Orthogonal Regime Classifier
QuantFlow Macro Research Institute (c) 2026

Implements the 7 mutually exclusive and collectively exhaustive (MECE)
thermodynamic phase regimes defined in Qiu (2026), Section 5 & Theorem 6.3.
Decoupled architecture utilizing a parameterized configuration dataclass.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
from .kinematics import (
    compute_overhead_resistance_density,
    compute_potential_well,
    compute_turnover_envelope,
    compute_vwap_pde,
    compute_work_impedance,
)


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
    phi_collapse_min : float, default 0.50
        Minimum potential coordinate required for gravitational collapse state.
    pde_super_min : float, default 0.40
        Minimum barycentric mechanical efficiency required for superconducting propulsion.
    pde_escape_min : float, default 0.14
        Minimum barycentric mechanical efficiency required for potential well escape.
    pde_collapse_max : float, default -0.15
        Maximum negative barycentric efficiency threshold triggering gravitational collapse.
    z_crit_super : float, default 120.0
        Upper critical impedance bound for superconducting propulsion.
    z_crit_escape : float, default 450.0
        Upper critical impedance bound for potential well escape.
    ft_cryo_max : float, default 1.20
        Maximum active turnover percentage permitted in cryogenic equilibrium.
    ft_min_active : float, default 0.20
        Minimum active turnover threshold for non-quiescent regimes.
    ret_impulse_min : float, default 1.00
        Minimum percentage price displacement for directional kinetic impulse.
    ret_cryo_max : float, default 0.80
        Maximum absolute price displacement permitted during cryogenic basal absorption.
    dark_cryo_min : float, default 40.0
        Minimum off-exchange volume purity confirmation threshold.
    """
    gamma_min: float = 0.80
    phi_valley_bound: float = 0.35
    phi_collapse_min: float = 0.50
    pde_super_min: float = 0.40
    pde_escape_min: float = 0.14
    pde_collapse_max: float = -0.15
    z_crit_super: float = 120.0
    z_crit_escape: float = 450.0
    ft_cryo_max: float = 1.20
    ft_min_active: float = 0.20
    ret_impulse_min: float = 1.00
    ret_cryo_max: float = 0.80
    dark_cryo_min: float = 40.0


def classify_market_regime(
    df: pd.DataFrame,
    config: Optional[RegimeConfig] = None,
    turnover_col: Optional[str] = None,
    dark_ratio_col: Optional[str] = None,
) -> pd.Series:
    """
    Classifies market microstructure state vector Psi_t into 7 discrete,
    mutually exclusive, and collectively exhaustive (MECE) thermodynamic phase regimes.

    The state space is partitioned into two natural thermodynamic half-spaces:
      - Positive Mechanical Work Half-Space (VWAP_PDE > 0)
      - Negative/Zero Mechanical Work Half-Space (VWAP_PDE <= 0, Z = BLOCKED)

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

    # 1. State Vector Observables
    phi = compute_potential_well(df)
    rd = compute_overhead_resistance_density(df)
    vwap_pde = compute_vwap_pde(df, gamma_min=cfg.gamma_min, turnover_col=turnover_col)
    z_imp = compute_work_impedance(rd, vwap_pde)

    # Active turnover rate FT_real
    if turnover_col and turnover_col in df.columns:
        ft_real = df[turnover_col]
    elif "volume" in df.columns:
        rolling_vol = df["volume"].rolling(window=20, min_periods=1).mean().replace(0, 1e-6)
        ft_real = (df["volume"] / rolling_vol) * 1.0
    else:
        ft_real = pd.Series(1.0, index=df.index)

    dark_ratio = df[dark_ratio_col] if dark_ratio_col and dark_ratio_col in df.columns else pd.Series(0.0, index=df.index)
    ft_max = compute_turnover_envelope(phi, dark_ratio=dark_ratio)

    delta_p = (close.pct_change() * 100.0).fillna(0.0)
    entity = (np.abs(close - open_p) / open_p.replace(0, 1e-6)) * 100.0
    upper_shadow = ((high - np.maximum(open_p, close)) / open_p.replace(0, 1e-6)) * 100.0

    regimes: List[str] = []
    n = len(df)

    for i in range(n):
        p_phi = phi.iloc[i]
        p_pde = vwap_pde.iloc[i]
        p_ft = ft_real.iloc[i]
        p_ft_max = ft_max.iloc[i] if isinstance(ft_max, pd.Series) else ft_max[i]
        p_z = z_imp.iloc[i]
        p_dp = delta_p.iloc[i]
        p_ent = entity.iloc[i]
        p_ushad = upper_shadow.iloc[i]
        p_dark = dark_ratio.iloc[i]

        # NaN guard
        if np.isnan(p_phi) or np.isnan(p_pde):
            regimes.append("Regime VII: Kinetic Neutral")
            continue

        # Branch A: Positive Mechanical Work Half-Space (VWAP_PDE > 0)
        if p_pde > 0:
            # 1. Regime IV: Crocodile Turbulent Dissipation
            if p_ft > p_ft_max and (p_ent <= 0.35 or p_ushad >= 2.0 * p_ent):
                regimes.append("Regime IV: Crocodile Turbulent Dissipation")

            # 2. Regime I: Superconducting Propulsion
            elif (
                cfg.phi_valley_bound <= p_phi <= 1.00
                and cfg.ft_min_active <= p_ft <= p_ft_max
                and p_dp > cfg.ret_impulse_min
                and p_pde >= cfg.pde_super_min
                and p_z <= cfg.z_crit_super
            ):
                regimes.append("Regime I: Superconducting Propulsion")

            # 3. Regime II: Potential Well Escape
            elif (
                p_phi < cfg.phi_valley_bound
                and cfg.ft_min_active <= p_ft <= p_ft_max
                and p_dp > cfg.ret_impulse_min
                and p_pde >= cfg.pde_escape_min
                and p_z <= cfg.z_crit_escape
            ):
                regimes.append("Regime II: Potential Well Escape")

            # 4. Regime III: Cryogenic Equilibrium
            elif (
                p_phi < cfg.phi_valley_bound
                and 0.10 <= p_ft <= cfg.ft_cryo_max
                and abs(p_dp) <= cfg.ret_cryo_max
                and 0.0 < p_pde < cfg.pde_escape_min
                and (p_dark >= cfg.dark_cryo_min or dark_ratio_col is None)
            ):
                regimes.append("Regime III: Cryogenic Equilibrium")

            # 5. Regime VII: Kinetic Neutral Equilibrium
            else:
                regimes.append("Regime VII: Kinetic Neutral")

        # Branch B: Negative or Zero Mechanical Work Half-Space (VWAP_PDE <= 0, Z = BLOCKED)
        else:
            # 6. Regime V: Gravitational Collapse
            if p_phi >= cfg.phi_collapse_min and p_pde <= cfg.pde_collapse_max:
                regimes.append("Regime V: Gravitational Collapse")

            # 7. Regime VI: Potential Well Oscillation
            else:
                regimes.append("Regime VI: Potential Well Oscillation")

    return pd.Series(regimes, index=df.index, name="FPMM_Regime")
