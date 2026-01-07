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
from geometry_msgs.msg import WrenchStamped

from scipy import constants
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

from iparams_regression.dynamics_utils import (
    get_regressor_matrix,
    convert_twist_world_to_body,
    compute_body_twist_and_derivative,
)
from iparams_regression.numerical_differentiator import NumericalDifferentiator

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
        # self.omega_diff = NumericalDifferentiator(cutoff_freq=self.cutoff_freq)
        # self.v_diff = NumericalDifferentiator(cutoff_freq=self.cutoff_freq)
        self.t_diff = NumericalDifferentiator(cutoff_freq=self.cutoff_freq)

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

        # 1. Compute Twist and DTwist in Body Frame (using logic moved to dynamics_utils)
        tw_tool0_tool0, dtw_tool0_tool0 = compute_body_twist_and_derivative(
            self.kin, self.t_diff, joint_positions, joint_velocities, t
        )

        # 6. Compute Regressor
        # dynamics_utils.get_regressor_matrix expects [v, w] order for input.
        A = get_regressor_matrix(tw_tool0_tool0, dtw_tool0_tool0)

        ## DEBUG: Log values at 1Hz
        # rospy.loginfo_throttle(
        #    1.0,
        #    f"\n[DEBUG] Regressor Node Status:\n"
        #    f"  Twist Body: {tw_tool0_tool0}\n"
        #    f"  DTwist Body: {dtw_tool0_tool0}\n"
        #    f"  A (Reg Matrix) Sample Row 0: {A[0]}\n"
        #    f"  A has NaN: {np.isnan(A).any()}\n"
        #    f"  A has Inf: {np.isinf(A).any()}\n",
        # )

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
