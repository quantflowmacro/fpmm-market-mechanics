"""
First-Principles Market Mechanics (FPMM)
Minimal Phase Detection Demo
QuantFlow Macro Research Institute (c) 2026
"""

from pathlib import Path
import sys

# Ensure repository root is in sys.path for direct script execution
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
from fpmm.kinematics import (
    compute_overhead_resistance_density,
    compute_potential_well,
    compute_turnover_envelope,
    compute_velocity,
    compute_vwap_pde,
    compute_work_impedance,
)
from fpmm.regimes import classify_market_regime


def run_demo():
    data_path = root_dir / "data" / "sample_ohlcv.csv"

    print("=" * 78)
    print("FPMM: First-Principles Market Mechanics Phase Detection Demo (v0.2.0)")
    print("=" * 78)

    # 1. Load standardized sample OHLCV data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} bars from {data_path.name}")

    # 2. Compute kinematic observables according to Qiu (2026)
    df["phi"] = compute_potential_well(df, window=250)
    df["rd"] = compute_overhead_resistance_density(df, window=250)
    df["velocity"] = compute_velocity(df["close"], window=5)
    df["vwap_pde"] = compute_vwap_pde(df, turnover_col="turnover_pct" if "turnover_pct" in df.columns else None)
    df["ft_max"] = compute_turnover_envelope(df["phi"])
    df["z_imp"] = compute_work_impedance(df["rd"], df["vwap_pde"])

    # 3. Detect thermodynamic phase regimes (7 MECE Orthogonal Regimes)
    df["regime"] = classify_market_regime(
        df,
        turnover_col="turnover_pct" if "turnover_pct" in df.columns else None,
        dark_ratio_col="dark_ratio" if "dark_ratio" in df.columns else None,
    )

    # 4. Display Regime Distribution Summary
    print("\n--- Detected Regime Distribution ---")
    regime_counts = df["regime"].value_counts()
    for regime, count in regime_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {regime:<45} : {count:>3} bars ({pct:5.1f}%)")

    # 5. Display Recent State Transitions
    print("\n--- Recent Market Microstructure States (Last 5 Bars) ---")
    view_cols = ["timestamp", "close", "phi", "vwap_pde", "rd", "regime"]
    display_df = df[view_cols].tail(5)
    for _, row in display_df.iterrows():
        pde_str = f"{row['vwap_pde']:>+6.2f}"
        print(
            f"  [{row['timestamp']}] Close: {row['close']:>6.2f} | "
            f"Phi: {row['phi']:>4.2f} | PDE: {pde_str} | RD: {row['rd']:>5.1f}% | {row['regime']}"
        )

    print("\n" + "=" * 78)
    print(f"Current Market Phase: {df['regime'].iloc[-1]}")
    print("=" * 78)


if __name__ == "__main__":
    run_demo()
