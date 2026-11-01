# First-Principles Market Mechanics (FPMM)

[![SSRN Paper](https://img.shields.io/badge/SSRN-7667578-blue.svg)](https://ssrn.com/abstract=7667578)
[![DOI](https://img.shields.io/badge/DOI-10.2139%2Fssrn.7667578-blue.svg)](https://doi.org/10.2139/ssrn.7667578)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0004--1469--2690-green.svg)](https://orcid.org/0009-0004-1469-2690)
[![Website](https://img.shields.io/badge/Research-QuantFlow_Macro-black.svg)](https://quantflowmacro.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official baseline implementation of **First-Principles Market Mechanics (FPMM)**, a continuous physics-based quantitative asset pricing and microstructure framework formulated by **Kevin Z. Qiu** at the **QuantFlow Macro Research Institute** (2026).

---

## Overview

FPMM models equity price formation not as a memoryless geometric random walk, but as a discrete mass particle traversing a non-linear gravitational potential well $\Phi(x)$ constrained by institutional order flow dynamics under the overdamped Smoluchowski-Kramers limit:

```math
\mathbb{E}_t\left[\frac{dx(t)}{dt}\right] = \frac{F_{\text{ext}}(t) - \dfrac{\partial U_{\text{grav}}(x, t)}{\partial x} - F_{\text{bar}}(x, \dot{x}, t)}{\gamma\big(FT_{\text{real}}(t)\big)}
```

Where:
- $\Phi(P_{\text{now}}) \in [0, 1]$: Dimensionless cyclical range coordinate mapping spatial altitude.
- $RD(P_{\text{now}}) \in [0, 100\%]$: Measure-theoretic overhead trapped supply density under continuous memory decay.
- `VWAP_PDE` ($\eta_W$): Dimensionless barycentric mechanical propulsion efficiency.
- $FT_{\max}(\Phi)$: Altitude-dependent critical turnover envelope constraining sustainable volume throughput.
- $Z = RD / \text{VWAP-PDE}$: Work impedance ratio (conduction locks to `BLOCKED` under zero/negative mechanical work).

By evaluating the dynamical balance between instantaneous price acceleration, work impedance, and the critical turnover envelope, the framework partitions equity state-space into **7 naturally orthogonal thermodynamic phase regimes (MECE)**:

1. **Regime I: Superconducting Propulsion**: Frictionless momentum expansion characterized by high-altitude potential well traversal under minimal work impedance and robust positive barycentric work.
2. **Regime II: Potential Well Escape**: Structural breakout confirmation indicating particle kinetic impulse successfully overcoming deep valley potential well barriers.
3. **Regime III: Cryogenic Equilibrium**: Basal institutional accumulation phase exhibiting compressed price volatility, low turnover, and covert off-exchange volume absorption.
4. **Regime IV: Crocodile Turbulent Dissipation**: High-energy shear collision and inventory distribution where excessive volume breaches critical turnover limits without generating net directional displacement.
5. **Regime V: Gravitational Collapse**: Severe kinetic failure and uncompensated gravitational descent at elevated potential coordinates under material negative barycentric work.
6. **Regime VI: Potential Well Oscillation**: Harmonically damped negative-work state representing valley-floor equilibrium or shallow consolidation pauses.
7. **Regime VII: Kinetic Neutral Equilibrium**: Complementary laminar transition state across the positive mechanical work half-space.

---

## Repository Structure

```plaintext
fpmm-market-mechanics/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI workflow
├── data/
│   └── sample_ohlcv.csv         # Desensitized sample market microstructure data
├── fpmm/
│   ├── __init__.py              # Core package exports
│   ├── kinematics.py            # Potential coordinate Phi, resistance RD, PDE, envelope FT_max, impedance Z
│   └── regimes.py               # 7-phase MECE thermodynamic state machine & RegimeConfig
├── examples/
│   └── demo_phase_detection.py  # Out-of-the-box regime detection demo
├── README.md                    # Core documentation & qualitative physical specifications
├── requirements.txt             # numpy, pandas, scipy, matplotlib
└── LICENSE                      # MIT License
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/quantflowmacro/fpmm-market-mechanics.git
cd fpmm-market-mechanics
pip install -r requirements.txt
```

### Minimal Phase Detection Demo

```python
import pandas as pd
from fpmm.regimes import classify_market_regime

# Load standardized sample OHLCV
df = pd.read_csv("data/sample_ohlcv.csv")

# Detect thermodynamic regime using baseline configuration
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
- **SSRN Working Paper**: [SSRN Abstract 7667578](https://ssrn.com/abstract=7667578)
- **Digital Object Identifier (DOI)**: [doi:10.2139/ssrn.7667578](https://doi.org/10.2139/ssrn.7667578)
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
  month={10},
  publisher={QuantFlow Macro Research Institute},
  doi={10.2139/ssrn.7667578},
  url={https://ssrn.com/abstract=7667578},
  note={Available at SSRN: \url{https://ssrn.com/abstract=7667578} or \url{https://doi.org/10.2139/ssrn.7667578}}
}
```

Standard Academic Reference:
> Qiu, Kevin Z., *First-Principles Market Mechanics: Particle Kinematics, Potential Well Escape, and Thermodynamic Phase Regimes in Financial Markets* (October 24, 2026). Available at SSRN: https://ssrn.com/abstract=7667578 or http://dx.doi.org/10.2139/ssrn.7667578

---

## Regulatory Compliance & Notice

Operates under the SEC Publisher's Exemption (15 U.S.C. § 80b-2(a)(11)(D)). Educational and institutional quantitative research only.
