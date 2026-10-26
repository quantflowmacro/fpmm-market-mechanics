"""
First-Principles Market Mechanics (FPMM)
Official baseline implementation of physics-based continuous asset pricing
and thermodynamic regime detection.
"""

from .kinematics import (
    compute_velocity,
    compute_potential_well,
    compute_turnover_envelope,
    compute_vwap_work,
)
from .regimes import classify_market_regime

__version__ = "0.1.0"
__author__ = "Kevin Z. Qiu"
__institution__ = "QuantFlow Macro Research Institute"

__all__ = [
    "compute_velocity",
    "compute_potential_well",
    "compute_turnover_envelope",
    "compute_vwap_work",
    "classify_market_regime",
]
