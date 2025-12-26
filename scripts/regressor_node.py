#!/usr/bin/env python3
"""
ROS Node for computing the regressor matrix for inertial parameter estimation.

Uses joint_states and tf instead of cartesian_compliance_controller topics,
so it works with the default position controller.

Subscribes to:
- /joint_states (sensor_msgs/JointState)
- /wrench/filtered (geometry_msgs/WrenchStamped)

Uses:
- ur_pykdl for forward kinematics and velocity computation
- tf2 for orientation lookup
- iparams_regression.dynamics_utils for regressor calculation (Lynch & Park formulation)

Publishes:
- /iparams_regression/regressor_matrix (std_msgs/Float64MultiArray)
- /iparams_regression/wrench (geometry_msgs/WrenchStamped) - passthrough of sensor wrench
"""

import rospy
import numpy as np
import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

from iparams_regression.dynamics_utils import (
    get_regressor_matrix,
    convert_twist_world_to_body,
)
from iparams_regression.regressor_matrix import NumericalDifferentiator

# Import ur_pykdl for kinematics
try:
    from ur_pykdl.ur_pykdl import ur_kinematics
except ImportError:
    rospy.logwarn("ur_pykdl not found, falling back to manual jacobian computation")
    ur_kinematics = None


class RegressorMatrixNode:
    """
    ROS node that computes the regressor matrix for inertial parameter estimation.
    Uses joint_states instead of cartesian_compliance_controller topics.
    """

    # UR5e joint names in order (must match ur_pykdl JOINT_ORDER)
    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    def __init__(self):
        rospy.init_node("regressor_matrix_node")

        # Parameters
        self.cutoff_freq = rospy.get_param("~cutoff_freq", 10.0)
        self.joint_states_topic = rospy.get_param(
            "~joint_states_topic", "/joint_states"
        )
        self.wrench_topic = rospy.get_param("~wrench_topic", "/wrench/filtered")
        self.base_link = rospy.get_param("~base_link", "base_link")
        self.ee_link = rospy.get_param("~ee_link", "tool0")

        # Initialize ur_pykdl kinematics
        if ur_kinematics is not None:
            try:
                self.kin = ur_kinematics(base_link=self.base_link, ee_link=self.ee_link)
                rospy.loginfo("ur_pykdl kinematics initialized successfully")
            except Exception as e:
                rospy.logerr(f"Failed to initialize ur_pykdl: {e}")
                self.kin = None
        else:
            self.kin = None

        # TF2 buffer for orientation lookup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Numerical differentiators for acceleration computation
        self.omega_diff = NumericalDifferentiator(cutoff_freq=self.cutoff_freq)
        self.v_diff = NumericalDifferentiator(cutoff_freq=self.cutoff_freq)

        # State storage
        self.last_wrench_msg = None
        self.last_wrench_time = None

        # Publishers
        self.regressor_pub = rospy.Publisher(
            "~regressor_matrix", Float64MultiArray, queue_size=10
        )
        self.wrench_pub = rospy.Publisher("~wrench", WrenchStamped, queue_size=10)

        # Subscribers
        self.joint_states_sub = rospy.Subscriber(
            self.joint_states_topic, JointState, self.joint_states_callback
        )
        self.wrench_sub = rospy.Subscriber(
            self.wrench_topic, WrenchStamped, self.wrench_callback
        )

        rospy.loginfo("Regressor matrix node initialized")
        rospy.loginfo(f"  Joint states topic: {self.joint_states_topic}")
        rospy.loginfo(f"  Wrench topic: {self.wrench_topic}")
        rospy.loginfo(f"  Cutoff frequency: {self.cutoff_freq} Hz")
        rospy.loginfo(f"  Base link: {self.base_link}")
        rospy.loginfo(f"  EE link: {self.ee_link}")

    def wrench_callback(self, msg: WrenchStamped):
        """Store latest wrench message."""
        self.last_wrench_msg = msg
        self.last_wrench_time = msg.header.stamp.to_sec()

    def joint_states_callback(self, msg: JointState):
        """
        Callback for joint states.
        Computes twist from joint velocities using ur_pykdl,
        converts to Body Frame, and computes the regressor matrix.
        """
        if self.kin is None:
            rospy.logwarn_throttle(5.0, "Kinematics not initialized")
            return

        if self.last_wrench_msg is None:
            rospy.logwarn_throttle(5.0, "No wrench data received yet")
            return

        # Check if this message contains all required joints
        if not all(name in msg.name for name in self.JOINT_NAMES):
            return

        # Extract joint positions and velocities
        try:
            joint_positions = []
            joint_velocities = []
            for name in self.JOINT_NAMES:
                idx = msg.name.index(name)
                joint_positions.append(msg.position[idx])
                joint_velocities.append(msg.velocity[idx])
            joint_positions = np.array(joint_positions)
            joint_velocities = np.array(joint_velocities)
        except (ValueError, IndexError) as e:
            rospy.logwarn_throttle(5.0, f"Failed to extract joint data: {e}")
            return

        # Current timestamp
        t = msg.header.stamp.to_sec()

        # 1. Compute Twist in World Frame (Reference Point: Tool0)
        # Returns [vx, vy, vz, wx, wy, wz]
        twist_w = self.kin.forward_velocity(joint_positions, joint_velocities)
        v_w = np.array(twist_w[:3])
        omega_w = np.array(twist_w[3:])

        # 2. Compute Acceleration in World Frame (Numerical Differentiation)
        alpha_w = self.omega_diff.update(omega_w, t)  # Angular accel
        a_w = self.v_diff.update(v_w, t)  # Linear accel

        # 3. Get Rotation Matrix (World to Body)
        # FK returns 4x4 Homogeneous matrix.
        pose_homog = self.kin.forward_position_kinematics(joint_positions)

        from scipy.spatial.transform import Rotation

        quat = pose_homog[3:]  # [qx, qy, qz, qw]
        R_wb = Rotation.from_quat(quat).as_matrix()  # Rotation from Body to World

        # 4. Convert Twist and Acceleration to Body Frame
        # Prepare 6D vectors [v, w]
        twist_w_vec = np.concatenate([v_w, omega_w])
        dtwist_w_vec = np.concatenate([a_w, alpha_w])

        twist_b = convert_twist_world_to_body(twist_w_vec, R_wb)

        # 5. Gravity Compensation and Acceleration
        # Ideally, dtwist_b should include the "proper acceleration" (gravity effect).
        # We model gravity as an upward acceleration of the base frame.
        # g_w = [0, 0, 9.81] (Upwards)
        g_w = np.array([0, 0, 9.81])
        dtwist_w_vec[:3] += g_w

        dtwist_b = convert_twist_world_to_body(dtwist_w_vec, R_wb)

        # 6. Compute Regressor
        # dynamics_utils.get_regressor_matrix expects [v, w] order for input.
        A = get_regressor_matrix(twist_b, dtwist_b)

        # Publish regressor matrix
        self.publish_regressor_matrix(A, msg.header)

        # Passthrough wrench (assumed to be already in Body/Sensor Frame)
        self.wrench_pub.publish(self.last_wrench_msg)

    def publish_regressor_matrix(self, A: np.ndarray, header):
        """
        Publish the regressor matrix as a Float64MultiArray.
        """
        msg = Float64MultiArray()

        # Set dimensions (6 rows, 10 columns)
        msg.layout.dim = [
            MultiArrayDimension(label="row", size=6, stride=60),
            MultiArrayDimension(label="col", size=10, stride=10),
        ]
        msg.layout.data_offset = 0

        # Flatten row-major
        msg.data = A.flatten().tolist()

        self.regressor_pub.publish(msg)

    def run(self):
        """
        Main loop.
        """
        rospy.spin()


def main():
    try:
        node = RegressorMatrixNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
