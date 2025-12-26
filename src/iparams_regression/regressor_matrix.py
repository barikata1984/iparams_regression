#!/usr/bin/env python3
"""
Regressor Matrix Calculation for Inertial Parameter Estimation

Based on Equation 6 from:
"On-line estimation of inertial parameters using a recursive total least-squares approach"
Kubus, Kröger, Wahl (2008)

The regressor matrix A is a 6x10 matrix that relates:
- Forces and torques (wrench) measured by the sensor
- To the inertial parameters of the load

Equation: [f; tau] = A * phi

where phi = [m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]^T
"""

import numpy as np


def compute_regressor_matrix(
    omega: np.ndarray, alpha: np.ndarray, a: np.ndarray, g: np.ndarray
) -> np.ndarray:
    """
    Compute the 6x10 regressor matrix A from Equation 6.

    Parameters
    ----------
    omega : np.ndarray
        Angular velocity [omega_x, omega_y, omega_z] in sensor frame (rad/s)
    alpha : np.ndarray
        Angular acceleration [alpha_x, alpha_y, alpha_z] in sensor frame (rad/s^2)
    a : np.ndarray
        Linear acceleration [a_x, a_y, a_z] in sensor frame (m/s^2)
    g : np.ndarray
        Gravity vector [g_x, g_y, g_z] in sensor frame (m/s^2)

    Returns
    -------
    np.ndarray
        6x10 regressor matrix A

    Notes
    -----
    The parameter vector phi is ordered as:
    [m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]

    The output wrench is ordered as:
    [fx, fy, fz, tau_x, tau_y, tau_z]
    """
    # Unpack vectors
    wx, wy, wz = omega
    ax_dot, ay_dot, az_dot = alpha  # alpha = d(omega)/dt
    ax, ay, az = a
    gx, gy, gz = g

    # Initialize 6x10 matrix
    A = np.zeros((6, 10))

    # Row 1: fx
    # fx = m*(ax-gx) + m*cx*(-wy^2-wz^2) + m*cy*(wx*wy-az_dot) + m*cz*(wx*wz+ay_dot)
    A[0, 0] = ax - gx  # m
    A[0, 1] = -(wy**2) - wz**2  # m*cx
    A[0, 2] = wx * wy - az_dot  # m*cy
    A[0, 3] = wx * wz + ay_dot  # m*cz

    # Row 2: fy
    # fy = m*(ay-gy) + m*cx*(wx*wy+az_dot) + m*cy*(-wx^2-wz^2) + m*cz*(wy*wz-ax_dot)
    A[1, 0] = ay - gy  # m
    A[1, 1] = wx * wy + az_dot  # m*cx
    A[1, 2] = -(wx**2) - wz**2  # m*cy
    A[1, 3] = wy * wz - ax_dot  # m*cz

    # Row 3: fz
    # fz = m*(az-gz) + m*cx*(wx*wz-ay_dot) + m*cy*(wy*wz+ax_dot) + m*cz*(-wy^2-wx^2)
    A[2, 0] = az - gz  # m
    A[2, 1] = wx * wz - ay_dot  # m*cx
    A[2, 2] = wy * wz + ax_dot  # m*cy
    A[2, 3] = -(wy**2) - wx**2  # m*cz

    # Row 4: tau_x
    # tau_x = m*cx*0 + m*cy*(gz-az) + m*cz*(ay-gy)
    #       + Ixx*ax_dot + Ixy*(ay_dot-wx*wz) + Ixz*(az_dot+wx*wy) + Iyz*(-wy*wz)
    A[3, 0] = 0  # m
    A[3, 1] = gz - az  # m*cx
    A[3, 2] = ay - gy  # m*cy
    A[3, 3] = 0  # m*cz (from image: this appears to be missing)
    A[3, 4] = ax_dot  # Ixx
    A[3, 5] = ay_dot - wx * wz  # Ixy
    A[3, 6] = az_dot + wx * wy  # Ixz
    A[3, 7] = -wy * wz  # Iyy
    A[3, 8] = 0  # Iyz
    A[3, 9] = 0  # Izz

    # Row 5: tau_y
    # tau_y = m*cx*(az-gz) + m*cy*0 + m*cz*(gx-ax)
    #       + Ixx*(wx*wz) + Ixy*(ax_dot+wy*wz) + Ixz*(wz^2-wx^2) + Iyy*ay_dot + Iyz*(az_dot+wx*wy) + Izz*(-wx*wy)
    A[4, 0] = 0  # m
    A[4, 1] = az - gz  # m*cx
    A[4, 2] = 0  # m*cy
    A[4, 3] = gx - ax  # m*cz
    A[4, 4] = wx * wz  # Ixx
    A[4, 5] = ax_dot + wy * wz  # Ixy
    A[4, 6] = wz**2 - wx**2  # Ixz
    A[4, 7] = ay_dot  # Iyy
    A[4, 8] = az_dot + wx * wy  # Iyz
    A[4, 9] = -wx * wy  # Izz

    # Row 6: tau_z
    # tau_z = m*cx*(gy-ay) + m*cy*(gx-ax) + m*cz*0
    #       + Ixx*(-wx*wy) + Ixy*(wx^2-wy^2) + Ixz*(ax_dot-wy*wz) + Iyy*(wx*wy) + Iyz*ay_dot + Izz*az_dot
    A[5, 0] = 0  # m
    A[5, 1] = gy - ay  # m*cx
    A[5, 2] = gx - ax  # m*cy
    A[5, 3] = 0  # m*cz
    A[5, 4] = -wx * wy  # Ixx
    A[5, 5] = wx**2 - wy**2  # Ixy
    A[5, 6] = ax_dot - wy * wz  # Ixz
    A[5, 7] = wx * wy  # Iyy
    A[5, 8] = ay_dot  # Iyz
    A[5, 9] = az_dot  # Izz

    return A


def compute_gravity_in_sensor_frame(
    orientation_quat: np.ndarray, g_world: np.ndarray = np.array([0, 0, -9.81])
) -> np.ndarray:
    """
    Compute the gravity vector in the sensor frame.

    Parameters
    ----------
    orientation_quat : np.ndarray
        Quaternion [x, y, z, w] representing the orientation of the sensor frame
        relative to the world frame.
    g_world : np.ndarray
        Gravity vector in world frame (default: [0, 0, -9.81])

    Returns
    -------
    np.ndarray
        Gravity vector [gx, gy, gz] in sensor frame
    """
    from scipy.spatial.transform import Rotation

    R = Rotation.from_quat(orientation_quat).as_matrix()
    g_sensor = R.T @ g_world
    return g_sensor


class NumericalDifferentiator:
    """
    Simple numerical differentiator with low-pass filtering.
    """

    def __init__(self, cutoff_freq: float = 10.0):
        """
        Parameters
        ----------
        cutoff_freq : float
            Cutoff frequency for low-pass filter (Hz)
        """
        self.prev_value = None
        self.prev_time = None
        self.prev_derivative = None
        self.cutoff_freq = cutoff_freq

    def update(self, value: np.ndarray, time: float) -> np.ndarray:
        """
        Compute the derivative of the input signal.

        Parameters
        ----------
        value : np.ndarray
            Current value
        time : float
            Current timestamp

        Returns
        -------
        np.ndarray
            Filtered derivative
        """
        if self.prev_value is None:
            self.prev_value = value
            self.prev_time = time
            self.prev_derivative = np.zeros_like(value)
            return self.prev_derivative

        dt = time - self.prev_time
        if dt <= 0:
            return self.prev_derivative

        # Raw derivative
        raw_derivative = (value - self.prev_value) / dt

        # Low-pass filter (first-order)
        alpha = dt * self.cutoff_freq / (1 + dt * self.cutoff_freq)
        filtered_derivative = (
            alpha * raw_derivative + (1 - alpha) * self.prev_derivative
        )

        # Update state
        self.prev_value = value
        self.prev_time = time
        self.prev_derivative = filtered_derivative

        return filtered_derivative

    def reset(self):
        """Reset the differentiator state."""
        self.prev_value = None
        self.prev_time = None
        self.prev_derivative = None
