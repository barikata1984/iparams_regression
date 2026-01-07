# iparams_regression package
"""
Inertial Parameter Estimation Package

This package provides:
- Regressor matrix computation for inertial parameter estimation
- Recursive Ordinary Least Squares (ROLS) estimator
- Recursive Total Least Squares (RTLS) estimator
- High-level InertialParameterEstimator interface
"""

from .numerical_differentiator import (
    NumericalDifferentiator,
)
from .recursive_ols import RecursiveOLS, solve_ols_batch
from .recursive_tls import RecursiveTLS, solve_tls_batch
from .inertial_parameters import InertialParameters

__all__ = [
    # Regressor matrix
    "NumericalDifferentiator",
    # OLS
    "RecursiveOLS",
    "solve_ols_batch",
    # TLS
    "RecursiveTLS",
    "solve_tls_batch",
    # High-level estimator
    "InertialParameters",
]
