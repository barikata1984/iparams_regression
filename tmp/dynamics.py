from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
from liegroups.numpy import SE3, SO3
from mujoco._functions import mjd_transitionFD
from mujoco._structs import MjData, MjModel, MjOption
from numpy.typing import NDArray

from transformations import Poses, homogenize
from utilities import get_element_id


@dataclass
class StateSpaceConfig:
    epsilon: float = 1e-8
    centered: bool = True


class StateSpace:
    def __init__(
        self,
        cfg: StateSpaceConfig,
        m: MjModel,
        d: MjData,
    ) -> None:
        self.epsilon = cfg.epsilon
        self.centered = cfg.centered

        self.ns = 2 * m.nv + m.na  # Number of dimensions of state space
        self.nsensordata = m.nsensordata  # Number of sensor ourputs

        self.A = np.zeros((self.ns, self.ns))  # State transition matrix
        self.B = np.zeros((self.ns, m.nu))  # Input2state matrix
        self.C = np.zeros((m.nsensordata, self.ns))  # State2output matrix
        self.D = np.zeros((m.nsensordata, m.nu))  # Input2output matrix

        # Populate the matrices
        self.update_matrices(m, d)

    def update_matrices(
        self,
        m: MjModel,
        d: MjData,
    ) -> None:
        mjd_transitionFD(m, d, self.epsilon, self.centered, self.A, self.B, self.C, self.D)


def _get_simat(
    mass: float,
    diag_i: list[float] | NDArray,
) -> NDArray:
    imat = np.diag(diag_i)  # inertia matrix
    return np.block([[mass * np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), imat]])


def get_spatial_inertia_matrix(
    mass,
    diagonal_inertia,
):
    assert len(mass) == len(diagonal_inertia), (
        "Lenght of 'mass' of the bodies and that of 'diagonal_inertia' vectors must match."
    )
    return np.array([_get_simat(m, di) for m, di in zip(mass, diagonal_inertia, strict=False)])


def transfer_simat(
    pose: SE3 | Sequence[SE3],
    simat: NDArray,
) -> NDArray:
    r"""Transfer the frame to which a spatial inertia tensor is desribed.

    Assuming, a sparial inertia tensor (\mathfrak{g}_a)is defined with
    reference to a frame {a}, this method converts the frame to which
    the tensor is described to another frame {b} given an input pose
    representing the configuration of {b} with respect to {a} (T_{ab})
    """

    single_pose = False
    single_simat = False

    # Add a batch dimension to handle a single pose as a pose set
    if isinstance(pose, SE3):
        # poses is an instance of liegroups.numpy.se3.SE3Matrix if this block hit
        single_pose = True
        pose = [pose]

    # Add a batch dimension to handle a single simat as a simat set
    if 2 == simat.ndim:
        single_simat = True
        simat = np.expand_dims(simat, 0)

    assert len(pose) == len(simat), ValueError(
        "The numbers of spatial inertia tensors and SE3 instances do not match."
    )

    adjoint = [p.inv().adjoint() for p in pose]  #
    # Ad is assumed to be described
    transfered = np.array([Ad.T @ sim @ Ad for Ad, sim in zip(adjoint, simat, strict=False)])  # Eq. 8.42 in MR

    return transfered[0] if single_pose or single_simat else transfered


def inverse(
    traj: np.ndarray,
    hposes_body_parent,
    simats_body: np.ndarray,
    uscrews_body: np.ndarray,
    twist_0: np.ndarray,
    dtwist_0: np.ndarray,
    wrench_tip: np.ndarray = np.zeros(6),
    pose_tip_ee: NDArray = SE3.identity(),
):
    # Prepare lie group, twist, and dtwist storage arrays
    poses = []  # T_{i, i - 1} in Modern Robotics
    twists = [twist_0]  # \mathcal{V}
    dtwists = [dtwist_0]  # \dot{\mathcal{V}}

    # Forward iterations
    for i, (h_p, us) in enumerate(zip(hposes_body_parent[1:], uscrews_body, strict=False)):
        poses.append(SE3.exp(-1 * us * traj[0, i]).dot(h_p))  # Eq. 8.50
        # Compute twist (Eq. 8.51 in Modern Robotics)
        tw = poses[-1].adjoint() @ twists[-1] + us * traj[1, i]
        # Compute the derivatife of twist (Eq. 8.52 in Modern Robotics)
        dtw = poses[-1].adjoint() @ dtwists[-1] + SE3.curlywedge(tw) @ us * traj[1, i] + us * traj[2, i]
        # Add the twist and its derivative to their storage arrays
        twists.append(tw)
        dtwists.append(dtw)

    # Backward iterations
    wrench = [wrench_tip]
    poses.append(pose_tip_ee)
    # Let m the # of joint/actuator axes, the backward iteration should be
    # performed from index m to 1. So, the range is set like below.
    for i in range(len(uscrews_body), 0, -1):
        # Compute wrench (Eq. 8.53 in Modern Robotics)
        w = (
            poses[i].adjoint().T @ wrench[-1]
            + simats_body[i] @ dtwists[i]
            + -1 * SE3.curlywedge(twists[i]).T @ simats_body[i] @ twists[i]
        )
        wrench.append(w)

    wrench.reverse()

    # An uscrew is a set of one-hot vectors, where the hot flag indicates the force
    # or torque element of the wrench that each screw corresponds to. Therefore,
    # the hadamarrd product below extracts the control signals, which are the
    # magnitude of target force or torque signals, from wrench array
    ctrl_mat = wrench[:-1] * uscrews_body  # Eq. 8.54

    return ctrl_mat.sum(axis=1), poses, twists, dtwists


def get_regressor_matrix(
    twist: NDArray,
    dtwist: NDArray,
) -> NDArray:
    def bullet(
        vec3: NDArray,
    ) -> NDArray:
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

    v, w = np.split(twist, 2)
    dv, dw = np.split(dtwist, 2)

    wedge_w = SO3.wedge(w)
    wedge_dw = SO3.wedge(dw)
    bullet_w = bullet(w)
    bullet_dw = bullet(dw)

    x = dv + wedge_w @ v
    # x = np.expand_dims(_x, axis=1)
    wedge_x = SO3.wedge(x)
    X = wedge_dw + wedge_w @ wedge_w
    Y = bullet_dw + wedge_w @ bullet_w
    regressor = np.block([[x.reshape((-1, 1)), X, np.zeros((3, 6))], [np.zeros((3, 1)), -wedge_x, Y]])

    return regressor


def coordinate_transfer_imat(pose_target_current, imat_current, mass):
    rot = pose_target_current.rot.as_matrix()
    trans = np.expand_dims(pose_target_current.trans, axis=1)
    imat = rot @ imat_current @ rot.T + mass * (trans.T @ trans * np.eye(3) - trans @ trans.T)

    return imat


def coordinate_transfer_simat(pose_target_current, simat_current):
    simat = pose_target_current.adjoint() @ simat_current @ pose_target_current.adjoint().T

    return simat


# NEW METHODS; USE BELOW FROM NOW ON
def get_linvel(
    twist: NDArray,
    pose: SE3,
    homogeneous: bool = False,
) -> NDArray:
    """
    Given a twist and the pose to a target coordinate frame w.r.t the reference
    frame of the twist, extract the linear velocity whose reference framae is
    converted to the target frame.

    Refer to Chap. 8.2.1 in Modern Robotics (Lynch and Park, 2017) and check the
    second paragraph.

    Parameters
    ----------
    twist: NDArray
        Twist vector
    pose: SE3(Matrix)
        Pose of the target coordinate frame w.r.t the reference frame of the twist
    """
    _linvel = SE3.wedge(twist) @ homogenize(pose.trans)

    return _linvel if homogeneous else _linvel[:3]


def get_linacc(
    twist: NDArray,
    dtwist: NDArray,
    pose: SE3,
    homogeneous: bool = False,
) -> NDArray:
    """
    Given a twist, its time-derivative, and the pose to a target coordinate frame
    w.r.t the reference frame of the twist, extract the linear acceleration whose
    reference framae is converted to the target frame.

    Refer to Chap. 8.2.1 in Modern Robotics (Lynch and Park, 2017) and check the
    second paragraph.


    Parameters
    ----------
    twist: NDArray
        Twist vector
    dtwist: NDArray
        Time-derivative of the twist
    pose: SE3(Matrix)
        Pose of the target coordinate frame w.r.t the reference frame of the twist
    """
    _linvel = get_linvel(twist, pose, homogeneous=True)
    _linacc = SE3.wedge(dtwist) @ homogenize(pose.trans) + SE3.wedge(twist) @ _linvel

    return _linacc if homogeneous else _linacc[:3]


def setup_robot_dynamics_parameters(
    m: MjModel,
    d: MjData,
    ee_body_name: str = "link6",
):
    poses = Poses(m, d)
    id_ll = get_element_id(m, "body", ee_body_name)
    id_x2ll = slice(0, id_ll + 1)
    pose_ll_llj = poses.l_lj[id_ll]

    uscrews_lj = []
    for t, ax in zip(m.jnt_type, m.jnt_axis, strict=False):
        us_lj = np.zeros(6)
        if 2 == t:  # slider joint
            us_lj[:3] += ax
        elif 3 == t:  # hinge joint
            us_lj[3:] += ax
        else:
            raise TypeError(
                "Only slide or hinge joints, represented as 2 or 3 for an element of m.jnt_type, are supported."
            )
        uscrews_lj.append(us_lj)
    uscrews_lj = np.array(uscrews_lj)

    simats_bi_b = get_spatial_inertia_matrix(
        m.body_mass,
        m.body_inertia,
    )
    simats_lj_l = []
    for pose_lj_li, simat_li_l in zip(poses.lj_li, simats_bi_b[id_x2ll], strict=False):
        simats_lj_l.append(transfer_simat(pose_lj_li, simat_li_l))
    simats_lj_l = np.array(simats_lj_l)

    pose_x_ll = poses.x_b[id_ll]
    for pose_x_bi, simat_bi_b in zip(poses.x_bi[id_ll + 1 :], simats_bi_b[id_ll + 1 :], strict=False):
        pose_x_llj = pose_x_ll.dot(pose_ll_llj)
        pose_bi_llj = pose_x_bi.inv().dot(pose_x_llj)
        simat_llj_b = transfer_simat(pose_bi_llj.inv(), simat_bi_b)
        simats_lj_l[id_ll] += simat_llj_b

    hposes_lj_kj = [SE3.identity()]
    for k in range(m.njnt):
        hpose_kj_k = poses.l_lj[k].inv()
        hpose_l_lj = poses.l_lj[k + 1]
        hpose_k_l = poses.a_b[k + 1]
        hpose_kj_lj = hpose_kj_k.dot(hpose_k_l.dot(hpose_l_lj))
        hposes_lj_kj.append(hpose_kj_lj.inv())

    gacc_x = -1 * np.array([*MjOption().gravity, 0, 0, 0])
    inverse_dynamics = partial(
        inverse,
        hposes_body_parent=hposes_lj_kj,
        simats_body=simats_lj_l,
        uscrews_body=uscrews_lj,
        twist_0=np.zeros(6),
        dtwist_0=gacc_x,
    )
    return poses, id_ll, pose_ll_llj, uscrews_lj, simats_lj_l, hposes_lj_kj, inverse_dynamics


def calculate_frame_dynamics(
    act_traj: NDArray,
    inverse_dynamics_partial_func,
    id_ll: int,
    pose_x_ll: SE3,
    pose_ll_llj: SE3,
    pose_x_sen: SE3,
) -> tuple[NDArray, NDArray, NDArray]:
    _, _, twists_lj_l, dtwists_lj_l = inverse_dynamics_partial_func(act_traj)
    twist_llj = twists_lj_l[id_ll]
    pose_sen_llj = pose_x_sen.inv().dot(pose_x_ll.dot(pose_ll_llj))
    twist_sen = pose_sen_llj.adjoint() @ twist_llj
    dtwist_llj = dtwists_lj_l[id_ll]
    pose_sen_llj_dadjoint = SE3.curlywedge(twist_sen) @ pose_sen_llj.adjoint()
    dtwist_sen = pose_sen_llj_dadjoint @ twist_llj + pose_sen_llj.adjoint() @ dtwist_llj

    regressor = get_regressor_matrix(twist_sen, dtwist_sen)
    return twist_sen, dtwist_sen, regressor
