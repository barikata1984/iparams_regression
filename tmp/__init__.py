# Recursive Least Squares for Inertial Parameter Estimation
"""
This package provides implementations for online estimation of inertial parameters
using recursive least squares approaches:
- Ordinary Least Squares (OLS) / Recursive Least Squares (RLS)
- Total Least Squares (TLS) / Recursive Total Least Squares (RTLS)
"""

from .recursive_ols import RecursiveOLS
from .recursive_tls import RecursiveTLS
from .inertial_estimator import InertialParameterEstimator

__all__ = [
    "RecursiveOLS",
    "RecursiveTLS",
    "InertialParameterEstimator",
]
