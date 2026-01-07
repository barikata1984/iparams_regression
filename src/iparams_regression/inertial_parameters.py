"""
Inertial Parameters Data Structure

This module provides the InertialParameters dataclass for representing
inertial parameters (mass, center of mass, inertia tensor).
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


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
