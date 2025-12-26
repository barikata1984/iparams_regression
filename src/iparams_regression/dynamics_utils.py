"""
Utility functions for dynamics calculations, using pymlg for SE3 operations.
Includes the regressor matrix calculation based on Lynch & Park (Modern Robotics) logic.

Requires: pymlg (https://github.com/decargroup/pymlg)
"""

import numpy as np
from pymlg import SE3, SO3


def bullet(vec3: np.ndarray) -> np.ndarray:
    """
    Computes the co-adjoint representation related matrix (or similar structure)
    that maps inertial parameters to wrench.

    Equivalent to the 'bullet' operator in the reference implementation.
    Output is (6, 6) but based on the provided dynamics.py implementation,
    it seems to return a specific block structure used for regressor construction.

    In dynamics.py:
    return np.block([
        [x, 0, 0],  # ixx
        [0, y, 0],  # iyy
        [0, 0, z],  # izz
        [y, x, 0],  # ixy
        [0, z, y],  # iyz
        [z, 0, x],  # izx
    ]).T

    This maps [Ixx, Iyy, Izz, Ixy, Iyz, Izx] to the torque components.
    """
    x, y, z = vec3
    return np.block(
        [
            [x, 0, 0],  # ixx
            [0, y, 0],  # iyy
            [0, 0, z],  # izz
            [y, x, 0],  # ixy
            [0, z, y],  # iyz
            [z, 0, x],  # izx
        ]
    ).T


def get_regressor_matrix(twist: np.ndarray, dtwist: np.ndarray) -> np.ndarray:
    """
    Compute the regressor matrix Y(V, dV) such that F = Y @ theta.

    Based on Lynch & Park (Modern Robotics) formulation / dynamics.py implementation.

    Parameters
    ----------
    twist : np.ndarray
        Spatial velocity (twist) in Body Frame. Shape (6,) [v, w] or similar based on definition.
        In dynamics.py, twist is split into v (linear) and w (angular).
        Note: The order in dynamics.py is `v, w = np.split(twist, 2)` -> twist = [v, w].
        However, standard Modern Robotics notation often uses [w, v].
        Based on dynamics.py:
        v, w = np.split(twist, 2) implies twist[:3] is linear, twist[3:] is angular.

    dtwist : np.ndarray
        Spatial acceleration in Body Frame. Shape (6,) [dv, dw].

    Returns
    -------
    np.ndarray
        6x10 Regressor matrix.
        Parameter order assumed: [m, Ixx, Iyy, Izz, Ixy, Iyz, Izx, mcx, mcy, mcz]
        (Wait, let's verify the order from the block construction)

        Block construction in dynamics.py:
        regressor = np.block([[x.reshape((-1, 1)), X, np.zeros((3, 6))], [np.zeros((3, 1)), -wedge_x, Y]])

        Where:
        x (3x1): associated with mass (linear force part)
        X (3x6): associated with ?? (Wait, X comes from wedge_dw + wedge_w @ wedge_w)
                 This looks like the term for m*c (first moment of inertia) if using Steiner's theorem terms,
                 but usually mass params are [m, mcx, mcy, mcz, Ixx...]

        Let's look at the structure more carefully.
        F = m*a - f_ext ...
        The top 3 rows are Force equations.
        The bottom 3 rows are Torque equations.

        Column 0: x (3x1) -> Force = x * theta[0] (mass?)
           x = dv + w x v (linear acceleration) -> F = m * a. Correct.
           So theta[0] is Mass.

        Columns 1-6 (X):
           X is 3x6. ???
           Usually, the parameters are 10.
           dynamics.py regressor is 6x(1+6+6)?? No.

           Let's check dynamics.py again.
           bullet(vec3) returns shape (6, 3). No, (3, 6) because of .T.
           bullet input is size 3. Output is 3x6.

           regressor = np.block([[x (3x1), X (3x3?), zeros?]])

           Wait, in dynamics.py:
           bullet_w = bullet(w) -> 3x6
           Y = bullet_dw + wedge_w @ bullet_w -> 3x6

           Structure:
           [[x (3x1), X (3x?), zeros(3, ?)]]

           In dynamics.py:
           X = wedge_dw + wedge_w @ wedge_w  -> (3,3)

           regressor = np.block([[x.reshape((-1, 1)), X, np.zeros((3, 6))],
                                [np.zeros((3, 1)), -wedge_x, Y]])

           Top row: [3x1, 3x3, 3x6] -> Total cols: 1 + 3 + 6 = 10. Correct.

           Theta structure:
           1 (mass)
           3 ( ??? first moment? ) -> X corresponds to this.
             X = [domega]x + [omega]x[omega]x.
             Force = m*a - ... skew(domega)*mc + ...
             This corresponds to m*c (first moment of mass).
           6 ( Inertia ) -> Zero in top row. Inertia doesn't affect Force directly (if CM is reference)?
             Wait, Newton's law: F = m * (dv/dt + ...).
             If reference is not CM, there are coupling terms.

           Bottom row:
           [3x1, 3x3, 3x6]
           Col 0 (Mass): Zero. (Torque doesn't depend on mass directly if purely rotational? No, gravity/Coriolis on CM)
           Col 1-3: -wedge_x = -[a]x. Torque = ... - m * [a]x * c ?? = m * c x a.
             Matches first moment.
           Col 4-9 (Y): Inertia terms.

           So parameter order is:
           [m, (3 elements of m*c), (6 elements of Inertia)]

           m*c order: x, y, z (vector)
           Inertia order: Ixx, Iyy, Izz, Ixy, Iyz, Izx (from bullet function)

           Final Order: [m, mcx, mcy, mcz, Ixx, Iyy, Izz, Ixy, Iyz, Izx]
    """
    # dynamics.py: v, w = np.split(twist, 2)
    # This implies twist is [v; w] (linear, angular)
    v, w = np.split(twist, 2)
    dv, dw = np.split(dtwist, 2)

    # SO3.wedge is the skew-symmetric operator [.]x
    wedge_w = SO3.wedge(w)
    wedge_dw = SO3.wedge(dw)

    bullet_w = bullet(w)
    bullet_dw = bullet(dw)

    # Linear part
    # x = dv + w x v
    x = dv + wedge_w @ v

    wedge_x = SO3.wedge(x)

    # Angular/Coupling terms
    X = wedge_dw + wedge_w @ wedge_w
    Y = bullet_dw + wedge_w @ bullet_w

    # Construct Regressor (6x10)
    # [ x   X   0 ]
    # [ 0  -x^  Y ]
    regressor = np.block(
        [[x.reshape((-1, 1)), X, np.zeros((3, 6))], [np.zeros((3, 1)), -wedge_x, Y]]
    )

    return regressor


def convert_twist_world_to_body(
    twist_world: np.ndarray, rotation_matrix_wb: np.ndarray
) -> np.ndarray:
    """
    Convert a twist from World frame to Body frame.

    Assumptions:
    - world_twist is [v_w; w_w] (Linear; Angular) expressed in World Frame.
    - Reference Point is the Body Origin (at current location).
    - rotation_matrix_wb is R_wb (Rotation from Body to World).

    Formula:
    V_b = [ R_wb^T  0      ]  V_w
          [ 0       R_wb^T ]

    This is valid because the reference point of V_w and V_b is the same (Body Origin),
    only the basis vectors are rotated.

    Parameters
    ----------
    twist_world : (6,) [v, w] in World Frame
    rotation_matrix_wb : (3, 3) R_wb

    Returns
    -------
    twist_body : (6,) [v, w] in Body Frame
    """
    R_bw = rotation_matrix_wb.T

    v_w = twist_world[:3]
    w_w = twist_world[3:]

    v_b = R_bw @ v_w
    w_b = R_bw @ w_w

    return np.concatenate([v_b, w_b])
