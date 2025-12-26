"""
Inertial Parameter Estimator for Robot Dynamics

This module provides a high-level interface for estimating inertial parameters
(mass, center of mass, inertia tensor) of a rigid body or robot link using
recursive least squares approaches.

The dynamics of a rigid body can be written as:
    F = M @ ẍ + h(x, ẋ)

where the parameters to be estimated are typically:
    - m: mass
    - m*cx, m*cy, m*cz: first moments of mass (mass * center of mass)
    - Ixx, Ixy, Ixz, Iyy, Iyz, Izz: inertia tensor elements

The regressor matrix is constructed such that:
    τ = Y(q, q̇, q̈) @ θ

where θ is the vector of inertial parameters.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple
from dataclasses import dataclass

from .recursive_ols import RecursiveOLS, solve_ols_batch
from .recursive_tls import RecursiveTLS, solve_tls_batch


@dataclass
class InertialParameters:
    """
    Container for inertial parameters of a rigid body.

    Attributes:
        mass: Mass of the body [kg]
        com: Center of mass position [m] (3D vector)
        inertia: Inertia tensor [kg*m^2] (3x3 symmetric matrix)
    """

    mass: float
    com: NDArray[np.floating]  # (3,)
    inertia: NDArray[np.floating]  # (3, 3)

    @classmethod
    def from_parameter_vector(cls, theta: NDArray[np.floating]) -> "InertialParameters":
        """
        Create InertialParameters from a 10-element parameter vector.

        The parameter vector is:
            [m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]

        Args:
            theta: 10-element parameter vector

        Returns:
            InertialParameters instance
        """
        if len(theta) != 10:
            raise ValueError("Parameter vector must have 10 elements")

        m = theta[0]

        if abs(m) < 1e-10:
            # Avoid division by zero
            com = np.zeros(3)
        else:
            com = theta[1:4] / m

        # Construct symmetric inertia tensor
        inertia = np.array(
            [
                [theta[4], theta[5], theta[6]],
                [theta[5], theta[7], theta[8]],
                [theta[6], theta[8], theta[9]],
            ]
        )

        return cls(mass=m, com=com, inertia=inertia)

    def to_parameter_vector(self) -> NDArray[np.floating]:
        """
        Convert to a 10-element parameter vector.

        Returns:
            Parameter vector [m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
        """
        m = self.mass
        mc = m * self.com
        inertia_matrix = self.inertia

        return np.array(
            [
                m,
                mc[0],
                mc[1],
                mc[2],
                inertia_matrix[0, 0],
                inertia_matrix[0, 1],
                inertia_matrix[0, 2],
                inertia_matrix[1, 1],
                inertia_matrix[1, 2],
                inertia_matrix[2, 2],
            ]
        )

    def __repr__(self) -> str:
        return (
            f"InertialParameters(\n"
            f"  mass={self.mass:.4f} kg,\n"
            f"  com={self.com},\n"
            f"  inertia=\n{self.inertia}\n"
            f")"
        )


class InertialParameterEstimator:
    """
    Estimator for inertial parameters using OLS or TLS.

    This class provides methods for constructing the regressor matrix
    from motion and force/torque data, and estimating inertial parameters
    using either ordinary least squares or total least squares.

    Example usage:
        estimator = InertialParameterEstimator(method='tls')

        for t in range(n_samples):
            # Get measurements
            accel = get_linear_acceleration()
            omega = get_angular_velocity()
            alpha = get_angular_acceleration()
            force = get_measured_force()
            torque = get_measured_torque()

            # Update estimate
            params = estimator.update(accel, omega, alpha, force, torque)
    """

    def __init__(
        self,
        method: Literal["ols", "tls"] = "tls",
        forgetting_factor: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the inertial parameter estimator.

        Args:
            method: Estimation method ("ols" or "tls")
            forgetting_factor: Forgetting factor for recursive updates
            **kwargs: Additional arguments passed to the underlying estimator
        """
        self.method = method
        self.n_params = 10  # m, mc_x, mc_y, mc_z, Ixx, Ixy, Ixz, Iyy, Iyz, Izz

        if method == "ols":
            self.estimator = RecursiveOLS(
                n_params=self.n_params,
                forgetting_factor=forgetting_factor,
                **kwargs,
            )
        elif method == "tls":
            self.estimator = RecursiveTLS(
                n_params=self.n_params,
                forgetting_factor=forgetting_factor,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'ols' or 'tls'.")

    @staticmethod
    def build_regressor_row(
        accel: NDArray[np.floating],
        omega: NDArray[np.floating],
        alpha: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Build a row of the regressor matrix from motion data.

        The Newton-Euler equations for a rigid body are:
            F = m * a + ω × (ω × (m * c))
            τ = I @ α + ω × (I @ ω) + c × F

        This function builds the regressor Y such that:
            [F; τ] = Y @ θ

        where θ = [m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]

        Args:
            accel: Linear acceleration (3,)
            omega: Angular velocity (3,)
            alpha: Angular acceleration (3,)

        Returns:
            Tuple of (force_regressor, torque_regressor), each (3, 10)
        """
        a = np.asarray(accel)
        w = np.asarray(omega)
        dw = np.asarray(alpha)

        # Skew-symmetric matrix
        def skew(v):
            return np.array(
                [
                    [0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0],
                ]
            )

        S_w = skew(w)
        S_a = skew(a)
        # Note: S_dw = skew(dw) is available if needed for extended formulations

        # Force regressor: F = m*a + ω×(ω×(m*c)) = m*a - S_w @ S_w @ (m*c)
        Y_force = np.zeros((3, 10))
        Y_force[:, 0] = a  # m
        Y_force[:, 1:4] = -S_w @ S_w  # m*c terms

        # Torque regressor: τ = I @ α + ω × (I @ ω) + c × (m*a - ω×(ω×(m*c)))
        # This is more complex, simplified version:
        Y_torque = np.zeros((3, 10))

        # c × F = -S_c @ F, where F = m*a (simplified, ignoring ω×ω×c term for clarity)
        # Actually: c × (m*a) = (m*c) × a = -a × (m*c) = S_a @ (m*c)
        Y_torque[:, 1:4] = (
            S_a - S_w @ S_w @ S_w.T
        )  # Approximate first moment contribution

        # Inertia contributions: I @ α + ω × (I @ ω)
        # For symmetric I = [[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]]

        # α contribution: I @ α
        # Row 0: Ixx*α[0] + Ixy*α[1] + Ixz*α[2]
        # Row 1: Ixy*α[0] + Iyy*α[1] + Iyz*α[2]
        # Row 2: Ixz*α[0] + Iyz*α[1] + Izz*α[2]

        # Build inertia regressor (columns 4-9 correspond to Ixx, Ixy, Ixz, Iyy, Iyz, Izz)
        # This is a simplified regressor - full derivation requires careful handling

        # I @ α contribution
        Y_inertia_alpha = np.array(
            [
                [dw[0], dw[1], dw[2], 0, 0, 0],  # Row 0
                [0, dw[0], 0, dw[1], dw[2], 0],  # Row 1
                [0, 0, dw[0], 0, dw[1], dw[2]],  # Row 2
            ]
        )

        # ω × (I @ ω) contribution (gyroscopic term)
        # Note: The full expansion of I @ ω would be:
        # [Ixx*wx + Ixy*wy + Ixz*wz, Ixy*wx + Iyy*wy + Iyz*wz, Ixz*wx + Iyz*wy + Izz*wz]

        # ω × (I @ ω): build the regressor for this term
        # [ω]× @ I @ ω where I is parameterized by [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
        Y_gyro = np.array(
            [
                [
                    0,
                    -w[2] * w[0] + w[1] * w[1],
                    -w[2] * w[1],
                    w[1] * w[2],
                    -w[1] * w[1] + w[2] * w[2],
                    -w[2] * w[1],
                ],
                [
                    w[2] * w[0] - w[0] * w[0],
                    w[2] * w[1],
                    w[2] * w[2] - w[0] * w[0],
                    -w[0] * w[1],
                    -w[0] * w[2],
                    w[2] * w[2],
                ],
                [
                    -w[1] * w[0],
                    -w[1] * w[1] + w[0] * w[0],
                    w[0] * w[1],
                    w[0] * w[2],
                    w[0] * w[0] - w[1] * w[1],
                    -w[1] * w[2],
                ],
            ]
        )

        Y_torque[:, 4:10] = Y_inertia_alpha + Y_gyro

        return Y_force, Y_torque

    def update(
        self,
        accel: NDArray[np.floating],
        omega: NDArray[np.floating],
        alpha: NDArray[np.floating],
        force: NDArray[np.floating],
        torque: NDArray[np.floating],
    ) -> InertialParameters:
        """
        Update the parameter estimate with new measurements.

        Args:
            accel: Linear acceleration (3,)
            omega: Angular velocity (3,)
            alpha: Angular acceleration (3,)
            force: Measured force (3,)
            torque: Measured torque (3,)

        Returns:
            Current estimate of inertial parameters
        """
        Y_force, Y_torque = self.build_regressor_row(accel, omega, alpha)

        # Stack force and torque equations
        Y = np.vstack([Y_force, Y_torque])  # (6, 10)
        y = np.concatenate([force, torque])  # (6,)

        # Update estimator with each row
        for i in range(6):
            self.estimator.update(Y[i], y[i])

        return self.get_parameters()

    def get_parameters(self) -> InertialParameters:
        """Get the current parameter estimate."""
        theta = self.estimator.get_estimate()
        return InertialParameters.from_parameter_vector(theta)

    def get_parameter_vector(self) -> NDArray[np.floating]:
        """Get the raw parameter vector."""
        return self.estimator.get_estimate()

    def reset(self) -> None:
        """Reset the estimator."""
        self.estimator.reset()


def estimate_inertial_parameters_batch(
    accels: NDArray[np.floating],
    omegas: NDArray[np.floating],
    alphas: NDArray[np.floating],
    forces: NDArray[np.floating],
    torques: NDArray[np.floating],
    method: Literal["ols", "tls"] = "tls",
) -> InertialParameters:
    """
    Estimate inertial parameters from batch data.

    Args:
        accels: Linear accelerations (n_samples, 3)
        omegas: Angular velocities (n_samples, 3)
        alphas: Angular accelerations (n_samples, 3)
        forces: Measured forces (n_samples, 3)
        torques: Measured torques (n_samples, 3)
        method: "ols" or "tls"

    Returns:
        Estimated inertial parameters
    """
    n_samples = len(accels)

    # Build full regressor matrix
    Y_list = []
    y_list = []

    for i in range(n_samples):
        Y_force, Y_torque = InertialParameterEstimator.build_regressor_row(
            accels[i], omegas[i], alphas[i]
        )
        Y_list.append(Y_force)
        Y_list.append(Y_torque)
        y_list.append(forces[i])
        y_list.append(torques[i])

    Y = np.vstack(Y_list)
    y = np.concatenate(y_list)

    if method == "ols":
        theta = solve_ols_batch(Y, y)
    else:
        theta, _ = solve_tls_batch(Y, y)

    return InertialParameters.from_parameter_vector(theta)
