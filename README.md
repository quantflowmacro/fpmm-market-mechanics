# First-Principles Market Mechanics (FPMM) — Official Replication Repository

[![SSRN Paper](https://img.shields.io/badge/SSRN-7667578-blue.svg)](https://ssrn.com/abstract=7667578)
[![DOI](https://img.shields.io/badge/DOI-10.2139%2Fssrn.7667578-blue.svg)](https://doi.org/10.2139/ssrn.7667578)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0004--1469--2690-green.svg)](https://orcid.org/0009-0004-1469-2690)
[![Website](https://img.shields.io/badge/Research-QuantFlow_Macro-black.svg)](https://quantflowmacro.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official baseline implementation and empirical replication suite for **First-Principles Market Mechanics (FPMM)**, a continuous physics-based quantitative asset pricing and market microstructure framework formulated by **Kevin Z. Qiu** at the **QuantFlow Macro Research Institute** (2026).

**Working Paper 01 (SSRN #7667578):** *First-Principles Market Mechanics: Particle Kinematics, Potential Well Escape, and Thermodynamic Phase Regimes in Financial Markets* (Series No. 2026-01, V1.1)

---

## 1. Overview & Theoretical Foundations

FPMM models equity price formation not as a memoryless geometric random walk, but as a discrete mass particle traversing a non-linear gravitational potential well $\Phi(x)$ constrained by institutional order flow dynamics under the overdamped Smoluchowski-Kramers limit:

```math
\mathbb{E}_t\left[\frac{dx(t)}{dt}\right] = \frac{F_{\text{ext}}(t) - \dfrac{\partial U_{\text{grav}}(x, t)}{\partial x} - F_{\text{bar}}(x, \dot{x}, t)}{\gamma\big(FT_{\text{real}}(t)\big)}
```

Where:
- $\Phi(P_{\text{now}}) \in [0, 1]$: Dimensionless cyclical range coordinate mapping spatial altitude along the 250-day lookback channel.
- $RD(P_{\text{now}}) \in [0, 100\%]$: Measure-theoretic overhead trapped supply density under continuous exponential memory decay ($\lambda = 0.985$, half-life $T_{1/2} = 45.86$ trading sessions).
- `VWAP_PDE` ($\eta_W$): Dimensionless barycentric mechanical propulsion efficiency regularized by lower-quartile damping ($\gamma_{\min} = 0.80\%$).
- $FT_{\max}(\Phi)$: Multiplicative dynamic critical turnover envelope contracting monotonically with spatial potential coordinate $\Phi$.
- $Z_{\text{imp}} = RD / \eta_W$: Work impedance ratio (locks to `BLOCKED` under zero or negative barycentric mechanical work).
- $K_{\text{macro}}(t) \in [0.25, 1.25]$: Strictly causal ($t-1$) cross-sectional liquidity multiplier gating effective portfolio exposure.

The framework partitions equity state-space into **7 mutually exclusive and collectively exhaustive (MECE) thermodynamic phase regimes**:

1. **Regime I: Superconducting Propulsion ($\mathcal{R}_{\text{I}}$)**: Frictionless momentum expansion characterized by high-altitude potential well traversal under minimal work impedance and robust positive barycentric work.
2. **Regime II: Potential Well Escape ($\mathcal{R}_{\text{II}}$)**: Structural breakout confirmation where particle kinetic impulse successfully overcomes deep valley potential well barriers.
3. **Regime III: Potential-Well Accumulation ($\mathcal{R}_{\text{III}}$)**: Basal institutional absorption phase exhibiting compressed price volatility, low turnover, and covert off-exchange volume accumulation.
4. **Regime IV: Crocodile Turbulent Dissipation ($\mathcal{R}_{\text{IV}}$)**: High-energy fluid shear collision and inventory distribution where excessive volume breaches critical turnover limits without generating net directional displacement.
5. **Regime V: Gravitational Collapse ($\mathcal{R}_{\text{V}}$)**: Kinetic failure and uncompensated gravitational descent at elevated potential coordinates under material negative barycentric work.
6. **Regime VI: Boundary Queue Lock ($\mathcal{R}_{\text{VI}}$)**: Liquidity-exhaustion lock state exhibiting directional boundary pinning under vanishing upper wicks.
7. **Regime VII: Kinetic Neutral Equilibrium ($\mathcal{R}_{\text{VII}}$)**: Complementary laminar transition state across the positive mechanical work half-space.

---

## 2. Quickstart & Turnkey Replication

### Installation

```bash
git clone https://github.com/quantflowmacro/fpmm-market-mechanics.git
cd fpmm-market-mechanics
pip install -r requirements.txt
```

### Reproduce All WP-01 Paper Tables (~15 Seconds)

```bash
python replication_wp01.py
```

Running `replication_wp01.py` executes the complete end-to-end replication pipeline:
1. **Real Market Sample Retrieval**: Loads the cached 20-stock S&P 100 liquid core sample (`2019-01-01` to `2026-10-31`, 39,380 stock-days including a 250-day burn-in) from `data/us_sp100_wp01_sample.parquet`, or fetches it dynamically via `yfinance`.
2. **Causal Physics Feature Computation**: Calculates 250-day exponential decay overhead resistance ($RD$), dimensionless potential coordinate ($\Phi$), barycentric work efficiency (`VWAP_PDE`), critical turnover envelope ($FT_{\max}$), and work impedance ($Z_{\text{imp}}$).
3. **Out-of-Sample Empirical Diagnostics (`2020-01-02` to `2026-10-31`)**: Exports publication-ready LaTeX tables (`WP01_Table1` through `WP01_Diag3`) directly into `output_wp01/`.

### Reproduce All WP-02 Paper Tables (~1 Second)

```bash
python replication_wp02.py
```

Running `replication_wp02.py` reproduces the 2D Microstructure Phase Topology ($S_0 \sim S_5$) and Dark-Pool Envelope Expansion ablation study reported in **WP-02**:
- Implements stylized symmetric Level-1 proxies and unlevered 1.0x state-hysteresis holding protocols for open-source academic replication; institutional high-frequency TRF micro-weights and fund execution parameters remain strictly decoupled.
- Exports publication-ready LaTeX tables (`WP02_Table1_2D_States.tex` and `WP02_Table2_DarkPool_Ablation.tex`) directly into `output_wp02/`.

### Minimal Phase Detection Demo

```bash
python examples/demo_phase_detection.py
```

---

## 3. Verified Reference Output (20-Stock S&P 100 Liquid Core Sub-Sample)

Executing `python replication_wp01.py` reproduces the exact numerical results reported in **WP-01 (V1.1)**:

### Table 1: MECE Thermodynamic Regime Partition (`100.0%` Exact Sum)

| Thermodynamic Regime | Sample Share | Mean $\Phi$ | Mean $Z_{\text{imp}}$ | Fwd 1D | Fwd 5D | Fwd 10D |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| $\mathcal{R}_{\text{I}}$: Superconducting Escape | 35.7% | 0.75 | 32.5 | +0.03% | +0.29% | +0.70% |
| $\mathcal{R}_{\text{II}}$: Controlled Escape | 9.7% | 0.59 | 187.7 | +0.08% | +0.45% | +1.03% |
| $\mathcal{R}_{\text{III}}$: Potential-Well Accumulation | 5.7% | 0.19 | 1004.9 | +0.20% | +0.81% | +1.22% |
| $\mathcal{R}_{\text{VII}}$: Kinetic Neutral | 33.9% | 0.76 | 409.8 | +0.12% | +0.49% | +0.99% |
| $\mathcal{R}_{\text{IV}}$: Crocodile Dissipation | 2.1% | 0.70 | 356.1 | +0.46% | +1.97% | +3.23% |
| $\mathcal{R}_{\text{V}}$: Gravitational Collapse | 12.8% | 0.57 | Blocked (+inf) | +0.17% | +0.78% | +1.33% |
| $\mathcal{R}_{\text{VI}}$: Boundary Queue Lock | 0.1% | 0.64 | 24.6 | -0.10% | -3.99% | +0.15% |

### Table 2: State-Hysteresis Backtest & Fama-French 6-Factor Alpha (`Avg Hold: 7.6 Days`)

| One-Way Friction Cost | Ann. Return | Ann. Vol | Max DD | Sharpe ($R_f=3.5\%$) | FF6 Net $\alpha$ | NW $t$-Stat |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5 bps (Baseline)** | **+11.4%** | **13.2%** | **-17.2%** | **0.60** | **+9.35%** | **1.97** |
| **10 bps** | +8.9% | 13.2% | -17.8% | 0.41 | +6.81% | 1.43 |
| **15 bps** | +6.4% | 13.2% | -18.3% | 0.22 | — | — |
| **20 bps** | +3.9% | 13.3% | -18.9% | 0.03 | — | — |

---

## 4. Repository Structure

```plaintext
fpmm-market-mechanics/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI matrix (Python 3.10, 3.11, 3.12)
├── data/
│   ├── sample_ohlcv.csv         # Desensitized sample market microstructure data
│   └── us_sp100_wp01_sample.parquet # Pre-cached 20-stock S&P 100 sample (2019-2026)
├── fpmm/
│   ├── __init__.py              # Core package exports
│   ├── kinematics.py            # Coordinate Phi, resistance RD, PDE, envelope FT_max, impedance Z
│   └── regimes.py               # 7-phase MECE thermodynamic state machine & RegimeConfig
├── examples/
│   └── demo_phase_detection.py  # Minimal out-of-the-box regime detection demo
├── replication_wp01.py          # WP-01 turnkey empirical verification suite (Tables 1-2 & Diags)
├── README.md                    # Core documentation & empirical reproduction benchmarks
├── requirements.txt             # Environment dependencies
└── LICENSE                      # MIT License
```

---

## 5. Academic Citation

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
> Qiu, Kevin Z., *First-Principles Market Mechanics: Particle Kinematics, Potential Well Escape, and Thermodynamic Phase Regimes in Financial Markets* (October 24, 2026). Available at SSRN: https://ssrn.com/abstract=7667578 or https://doi.org/10.2139/ssrn.7667578

---

## 6. Regulatory Compliance & Disclaimer

Operates strictly under the U.S. Supreme Court's Lowe v. SEC Publisher's Exemption (15 U.S.C. § 80b-2(a)(11)(D)). All code, models, and empirical outputs provided in this repository are for quantitative academic research, open-source peer review, and educational reproducibility only. Nothing herein constitutes personalized investment advice or an offer to buy or sell securities.
