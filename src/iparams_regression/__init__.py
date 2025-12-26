# iparams_regression package
"""
Inertial Parameter Estimation Package

This package provides:
- Regressor matrix computation for inertial parameter estimation
- Recursive Ordinary Least Squares (ROLS) estimator
- Recursive Total Least Squares (RTLS) estimator
- High-level InertialParameterEstimator interface
"""

from .regressor_matrix import (
    compute_regressor_matrix,
    compute_gravity_in_sensor_frame,
    NumericalDifferentiator,
)
from .recursive_ols import RecursiveOLS, solve_ols_batch
from .recursive_tls import RecursiveTLS, solve_tls_batch
from .inertial_estimator import (
    InertialParameters,
    InertialParameterEstimator,
    estimate_inertial_parameters_batch,
)

__all__ = [
    # Regressor matrix
    "compute_regressor_matrix",
    "compute_gravity_in_sensor_frame",
    "NumericalDifferentiator",
    # OLS
    "RecursiveOLS",
    "solve_ols_batch",
    # TLS
    "RecursiveTLS",
    "solve_tls_batch",
    # High-level estimator
    "InertialParameters",
    "InertialParameterEstimator",
    "estimate_inertial_parameters_batch",
]
