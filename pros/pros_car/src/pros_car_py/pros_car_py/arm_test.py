import math
import time
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.parameter import Parameter

PARAMETER_INTEGER_TYPE = 3
class ArmTestNode(Node):
    """
    A ROS2 node for testing the robotic arm by publishing joint angles.

    This node publishes joint angles to simulate moving the robotic arm
    joints from 0 to 180 degrees.

    Attributes:
        joint_count (int): Number of joints to control.
        joint_trajectory_publisher_ (Publisher): Publisher to send joint angle messages.
    """

    def __init__(self):
        """
        Initializes the ArmTestNode.

        The constructor declares a parameter for the number of joints and sets
        up the publisher to publish joint angles to the 'robot_arm' topic.

        References:
            https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Using-Parameters-In-A-Class-Python.html
        """
        super().__init__("arm_test_node")

        # Declare the 'joints' parameter with a descriptor for metadata
        joint_param_descriptor = ParameterDescriptor(
            description="Number of joints to control.",
            type=PARAMETER_INTEGER_TYPE  # Correct way to specify the type
        )
        self.declare_parameter("joint_count", 4, joint_param_descriptor)  # Default is 4 joints
        self.joint_count = self.get_parameter("joint_count").get_parameter_value().integer_value

        self.joint_trajectory_publisher_ = self.create_publisher(
            JointTrajectoryPoint, "robot_arm", 10
        )

    def move_joints(self):
        """
        Publishes messages to move each joint from 0 to 180 degrees, with a 0.5 second interval.

        This method sends joint angle positions for all joints to the `robot_arm` topic.
        Each joint moves from 0 degrees to 180 degrees in steps of 10 degrees.
        """
        for angle in range(0, 181, 10):  # From 0 to 180 degrees in steps of 10 degrees
            msg = JointTrajectoryPoint()
            msg.positions = [
                math.radians(angle)
            ] * self.joint_count  # Set all joints to the same angle
            msg.velocities = [0.0] * self.joint_count  # Initialize velocities to 0
            self.joint_trajectory_publisher_.publish(msg)
            self.get_logger().info(
                f"Published joint angles: {angle} degrees to {self.joint_count} joints."
            )
            time.sleep(0.5)  # Pause for 0.5 seconds between each message


def main(args=None):
    """
    The main function to initialize and run the ArmTestNode.

    This function initializes the ROS 2 environment, retrieves the number of joints
    from the parameter, and starts moving the joints by calling the `move_joints` method.
    """
    rclpy.init(args=args)

    # Initialize the node
    arm_test_node = ArmTestNode()

    try:
        arm_test_node.move_joints()  # Start moving the joints
    except KeyboardInterrupt:
        pass

    # Clean up when finished
    arm_test_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
