#!/usr/bin/env python3
"""
ROS Node for online inertial parameter estimation.

Subscribes to:
- /regressor_matrix_node/regressor_matrix (std_msgs/Float64MultiArray)
- /regressor_matrix_node/wrench (geometry_msgs/WrenchStamped)

Publishes:
- ~parameters (std_msgs/Float64MultiArray) - 10-element parameter vector

The parameter vector is:
    [m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
"""

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from geometry_msgs.msg import WrenchStamped

from iparams_regression import RecursiveOLS, RecursiveTLS, InertialParameters


class InertialEstimatorNode:
    """
    ROS node for online inertial parameter estimation using ROLS or RTLS.
    """

    PARAM_NAMES = [
        "mass",
        "m*cx",
        "m*cy",
        "m*cz",
        "Ixx",
        "Ixy",
        "Ixz",
        "Iyy",
        "Iyz",
        "Izz",
    ]

    def __init__(self):
        rospy.init_node("inertial_estimator_node")

        # Parameters
        self.method = rospy.get_param("~method", "tls")  # "ols" or "tls"
        self.forgetting_factor = rospy.get_param("~forgetting_factor", 0.99)
        self.deadzone = rospy.get_param("~deadzone", 0.1)
        self.regressor_topic = rospy.get_param(
            "~regressor_topic", "/regressor_matrix_node/regressor_matrix"
        )
        self.wrench_topic = rospy.get_param(
            "~wrench_topic", "/regressor_matrix_node/wrench"
        )

        # Initialize estimator
        self.n_params = 10
        if self.method == "ols":
            self.estimator = RecursiveOLS(
                n_params=self.n_params,
                forgetting_factor=self.forgetting_factor,
                deadzone=self.deadzone,
            )
        else:
            self.estimator = RecursiveTLS(
                n_params=self.n_params,
                forgetting_factor=self.forgetting_factor,
                deadzone=self.deadzone,
            )

        # State storage
        self.last_wrench = None

        # Publishers
        self.params_pub = rospy.Publisher(
            "~parameters", Float64MultiArray, queue_size=10
        )

        # Subscribers (not synchronized - use latest wrench for each regressor)
        self.wrench_sub = rospy.Subscriber(
            self.wrench_topic, WrenchStamped, self.wrench_callback
        )
        self.regressor_sub = rospy.Subscriber(
            self.regressor_topic, Float64MultiArray, self.regressor_callback
        )

        # For periodic logging
        self.n_updates = 0
        self.last_log_time = rospy.Time.now()

        rospy.loginfo("Inertial Estimator Node initialized")
        rospy.loginfo(f"  Method: {self.method}")
        rospy.loginfo(f"  Forgetting factor: {self.forgetting_factor}")
        rospy.loginfo(f"  Deadzone: {self.deadzone}")
        rospy.loginfo(f"  Regressor topic: {self.regressor_topic}")
        rospy.loginfo(f"  Wrench topic: {self.wrench_topic}")

    def wrench_callback(self, msg: WrenchStamped):
        """Store the latest wrench message."""
        self.last_wrench = np.array(
            [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z,
            ]
        )

    def regressor_callback(self, regressor_msg: Float64MultiArray):
        """
        Callback for regressor matrix messages.
        Updates the estimator with latest wrench and publishes the estimate.
        """
        if self.last_wrench is None:
            rospy.logwarn_throttle(5.0, "No wrench data received yet")
            return

        # Extract regressor matrix (6x10)
        regressor = np.array(regressor_msg.data).reshape(6, 10)
        wrench = self.last_wrench

        # Update estimator with each row
        for i in range(6):
            self.estimator.update(regressor[i], wrench[i])

        self.n_updates += 1

        # Publish current estimate
        self.publish_parameters()

        # Periodic logging
        now = rospy.Time.now()
        if (now - self.last_log_time).to_sec() > 5.0:
            self.log_parameters()
            self.last_log_time = now

    def publish_parameters(self):
        """Publish the current parameter estimate."""
        theta = self.estimator.get_estimate()

        msg = Float64MultiArray()
        msg.layout.dim = [MultiArrayDimension(label="params", size=10, stride=10)]
        msg.data = theta.tolist()

        self.params_pub.publish(msg)

    def log_parameters(self):
        """Log current parameter estimates to console."""
        theta = self.estimator.get_estimate()
        params = InertialParameters.from_parameter_vector(theta)

        rospy.loginfo("=" * 50)
        rospy.loginfo(f"Inertial Parameter Estimate (n_updates={self.n_updates})")
        rospy.loginfo(f"  Mass: {params.mass:.4f} kg")
        rospy.loginfo(
            f"  CoM:  [{params.com[0]:.4f}, {params.com[1]:.4f}, {params.com[2]:.4f}] m"
        )
        rospy.loginfo(
            f"  Inertia diagonal: [{params.inertia[0, 0]:.6f}, {params.inertia[1, 1]:.6f}, {params.inertia[2, 2]:.6f}]"
        )
        rospy.loginfo("=" * 50)

    def run(self):
        """Main loop."""
        rospy.spin()


def main():
    try:
        node = InertialEstimatorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
