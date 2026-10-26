# First-Principles Market Mechanics (FPMM)

[![SSRN Paper](https://img.shields.io/badge/SSRN-Abstract_7667578-blue.svg)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7667578)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0004--1469--2690-green.svg)](https://orcid.org/0009-0004-1469-2690)
[![Website](https://img.shields.io/badge/Research-QuantFlow_Macro-black.svg)](https://quantflowmacro.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official baseline implementation of **First-Principles Market Mechanics (FPMM)**, a continuous physics-based quantitative asset pricing and microstructure framework formulated by **Kevin Z. Qiu** at the **QuantFlow Macro Research Institute** (2026).

---

## Overview

FPMM models equity price formation not as a memoryless geometric random walk, but as a discrete mass particle traversing a non-linear gravitational potential well $\Phi(x)$ constrained by institutional order flow dynamics:

$$\frac{dv_t}{dt} = -\nabla \Phi(x_t) - \gamma v_t + \eta(t)$$

By evaluating the dynamical balance between instantaneous price acceleration and the critical turnover envelope $FT_{\max}(\Phi)$, the framework classifies equity state-space into **7 discrete thermodynamic phase regimes**:

1. **Regime I: Superconducting Propulsion** ($\Phi < 0.35, \mathcal{W}_{\text{VWAP}} > 0$)
2. **Regime II: Potential Well Escape** (Breakout with turnover exceeding $FT_{\max}$)
3. **Regime III: Cryogenic Equilibrium** (Volatility contraction, boundary compression)
4. **Regime IV: Crocodile Turbulent Dissipation** (High volume, minimal displacement)
5. **Regime V: Gravitational Collapse** (Kinetic exhaustion at elevated potential)
6. **Regime VI: Potential Well Oscillation** (Harmonic damping across boundary zones)
7. **Regime VII: Kinetic Neutral** (Laminar baseline state)

---

## Repository Structure

```plaintext
fpmm-market-mechanics/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI workflow
├── data/
│   └── sample_ohlcv.csv         # Desensitized sample high-frequency/daily market data
├── fpmm/
│   ├── __init__.py              # Core package exports
│   ├── kinematics.py            # Velocity v, dimensionless potential Phi, envelope FT_max
│   └── regimes.py               # 7-phase thermodynamic state machine
├── examples/
│   └── demo_phase_detection.py  # Out-of-the-box regime detection demo
├── README.md                    # Core documentation & mathematical specifications
├── requirements.txt             # numpy, pandas, scipy, matplotlib
└── LICENSE                      # MIT License
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/your-username/fpmm-market-mechanics.git
cd fpmm-market-mechanics
pip install -r requirements.txt
```

### Minimal Phase Detection Demo

```python
import pandas as pd
from fpmm.regimes import classify_market_regime

# Load standardized sample OHLCV
df = pd.read_csv("data/sample_ohlcv.csv")

# Detect thermodynamic regime
regime_series = classify_market_regime(df)
print(f"Current Market Phase: {regime_series.iloc[-1]}")
```

Run the complete packaged example:

```bash
python examples/demo_phase_detection.py
```

---

## Research & Documentation

- **Theoretical Monograph**: [Read Full Working Paper at QuantFlow Macro](https://quantflowmacro.com/first-principles-market-mechanics-particle-dynamics/)
- **Academic Pre-print**: [SSRN Abstract 7667578](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7667578)
- **Institution**: [QuantFlow Macro Research Institute](https://quantflowmacro.com)

---

## Citation

If you utilize this framework, codebase, or mathematical methodology in your research, please cite:

```bibtex
@article{qiu2026firstprinciples,
  title={First-Principles Market Mechanics: Particle Kinematics, Potential Well Escape, and Thermodynamic Phase Regimes in Financial Markets},
  author={Qiu, Kevin Z.},
  journal={QuantFlow Macro Working Paper Series},
  volume={2026},
  number={01},
  pages={QFM--2026--01},
  year={2026},
  publisher={QuantFlow Macro Research Institute},
  url={https://quantflowmacro.com/first-principles-market-mechanics-particle-dynamics/},
  note={SSRN: \url{https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7667578}}
}
```

---

## Regulatory Compliance & Notice

Operates under the SEC Publisher's Exemption (15 U.S.C. § 80b-2(a)(11)(D)). Educational and institutional quantitative research only.
