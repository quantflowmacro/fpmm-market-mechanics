"""
First-Principles Market Mechanics (FPMM)
Minimal Phase Detection Demo
QuantFlow Macro Research Institute (c) 2026
"""

from pathlib import Path
import pandas as pd
from fpmm.kinematics import compute_potential_well, compute_velocity
from fpmm.regimes import classify_market_regime


def run_demo():
    # Resolve project root and data path
    root_dir = Path(__file__).resolve().parent.parent
    data_path = root_dir / "data" / "sample_ohlcv.csv"

    print("=" * 60)
    print("FPMM: First-Principles Market Mechanics Phase Detection Demo")
    print("=" * 60)

    # 1. Load standardized sample OHLCV data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} bars from {data_path.name}")

    # 2. Compute kinematic observables
    df["phi"] = compute_potential_well(df["close"], window=20)
    df["velocity"] = compute_velocity(df["close"], window=5)

    # 3. Detect thermodynamic phase regimes
    df["regime"] = classify_market_regime(df)

    # 4. Display Regime Distribution Summary
    print("\n--- Detected Regime Distribution ---")
    regime_counts = df["regime"].value_counts()
    for regime, count in regime_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {regime:<42} : {count:>3} bars ({pct:5.1f}%)")

    # 5. Display Recent State Transitions
    print("\n--- Recent Market Microstructure States (Last 5 Bars) ---")
    view_cols = ["timestamp", "close", "volume", "phi", "regime"]
    display_df = df[view_cols].tail(5)
    for _, row in display_df.iterrows():
        print(
            f"  [{row['timestamp']}] Close: {row['close']:>6.2f} | "
            f"Phi: {row['phi']:>5.3f} | {row['regime']}"
        )

    print("\n" + "=" * 60)
    print(f"Current Market Phase: {df['regime'].iloc[-1]}")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
