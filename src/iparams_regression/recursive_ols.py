"""
Recursive Ordinary Least Squares (RLS) Implementation

This module provides a recursive implementation of ordinary least squares
for online parameter estimation. RLS minimizes the error in the output
measurements only, assuming the input data (regressor matrix) is noise-free.

Model: y = A @ x + noise
where:
    y: output vector (measurements)
    A: regressor matrix (data matrix)
    x: parameter vector to be estimated
"""

import numpy as np
from numpy.typing import NDArray


class RecursiveOLS:
    """
    Recursive Ordinary Least Squares (RLS) estimator.

    This implementation uses the matrix inversion lemma (Sherman-Morrison-Woodbury)
    for efficient recursive updates of the covariance matrix.

    Attributes:
        n_params: Number of parameters to estimate
        forgetting_factor: Forgetting factor (0 < λ <= 1) for exponential weighting
        P: Covariance matrix (inverse of the information matrix)
        x: Current parameter estimate
    """

    def __init__(
        self,
        n_params: int,
        forgetting_factor: float = 1.0,
        initial_covariance: float = 1e6,
    ):
        """
        Initialize the RLS estimator.

        Args:
            n_params: Number of parameters to estimate
            forgetting_factor: Forgetting factor λ (0 < λ <= 1).
                              λ = 1 gives equal weight to all data.
                              λ < 1 gives more weight to recent data.
            initial_covariance: Initial value for diagonal of covariance matrix.
                               Large values indicate high initial uncertainty.
        """
        if not 0 < forgetting_factor <= 1:
            raise ValueError("Forgetting factor must be in (0, 1]")

        self.n_params = n_params
        self.forgetting_factor = forgetting_factor

        # Initialize covariance matrix P = δI (large initial uncertainty)
        self.P = initial_covariance * np.eye(n_params)

        # Initialize parameter estimate to zeros
        self.x = np.zeros(n_params)

        # Track number of updates
        self.n_updates = 0

    def update(self, a: NDArray[np.floating], y: float) -> NDArray[np.floating]:
        """
        Perform a single recursive update with a new measurement.

        Args:
            a: Regressor vector (1D array of shape (n_params,))
            y: Scalar measurement

        Returns:
            Updated parameter estimate
        """
        a = np.asarray(a).flatten()
        if len(a) != self.n_params:
            raise ValueError(f"Regressor vector must have {self.n_params} elements")

        λ = self.forgetting_factor

        # Compute Kalman gain: K = P @ a / (λ + a.T @ P @ a)
        Pa = self.P @ a
        denominator = λ + a @ Pa
        K = Pa / denominator

        # Compute prediction error
        y_pred = a @ self.x
        error = y - y_pred

        # Update parameter estimate: x = x + K * error
        self.x = self.x + K * error

        # Update covariance matrix: P = (P - K @ a.T @ P) / λ
        self.P = (self.P - np.outer(K, Pa)) / λ

        # Ensure symmetry (numerical stability)
        self.P = (self.P + self.P.T) / 2

        self.n_updates += 1

        return self.x.copy()

    def update_batch(
        self,
        A: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Perform batch recursive updates with multiple measurements.

        Args:
            A: Regressor matrix (n_samples, n_params)
            y: Measurement vector (n_samples,)

        Returns:
            Updated parameter estimate
        """
        A = np.asarray(A)
        y = np.asarray(y).flatten()

        if A.ndim == 1:
            A = A.reshape(1, -1)

        if len(y) != A.shape[0]:
            raise ValueError("Number of measurements must match number of rows in A")

        for i in range(len(y)):
            self.update(A[i], y[i])

        return self.x.copy()

    def get_estimate(self) -> NDArray[np.floating]:
        """Return the current parameter estimate."""
        return self.x.copy()

    def get_covariance(self) -> NDArray[np.floating]:
        """Return the current covariance matrix."""
        return self.P.copy()

    def get_std(self) -> NDArray[np.floating]:
        """Return the standard deviation of parameter estimates."""
        return np.sqrt(np.diag(self.P))

    def reset(self, initial_covariance: float = 1e6) -> None:
        """Reset the estimator to initial state."""
        self.P = initial_covariance * np.eye(self.n_params)
        self.x = np.zeros(self.n_params)
        self.n_updates = 0


def solve_ols_batch(
    A: NDArray[np.floating],
    y: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Solve ordinary least squares problem in batch mode.

    Solves: min_x ||y - A @ x||^2

    Args:
        A: Regressor matrix (n_samples, n_params)
        y: Measurement vector (n_samples,)

    Returns:
        Parameter estimate x
    """
    A = np.asarray(A)
    y = np.asarray(y).flatten()

    # Use numpy's least squares solver (uses SVD internally)
    x, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

    return x
