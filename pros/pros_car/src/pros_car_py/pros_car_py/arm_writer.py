"""
PROS Arm Commercial License
Copyright 2023 帕亞科技 (PAIA-Tech) . All rights reserved.
Primary Author: 陳麒麟(Kylin Chen)
Contributor(s): 蘇文鈺(Alvin Su) ,
This software and associated documentation files are proprietary to 帕亞科技 (PAIA-Tech),
and are not to be copied, reproduced, or disclosed to any third party, in whole or in part, without
express prior written permission. Use and modification of this software is permitted for licensed
users only, and must always remain proprietary to 帕亞科技 (PAIA-Tech).


This code is a node to write control signal to specific arm.
It will receive topic /robot_arm from ROS and
transform to servo control signal for serial device.

"""
import math
import orjson
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from serial import Serial
import std_msgs.msg
from trajectory_msgs.msg import JointTrajectoryPoint
from pros_car_py.car_models import *
from .env import ARM_SERIAL_PORT_DEFAULT


class ArmSerialWriter(Node):
    """
    A ROS2 node that handles sending joint trajectory points to an ESP32 over a serial connection.

    This class subscribes to joint trajectory point messages and sends the converted
    degree-based servo target angles to the ESP32 via serial communication.

    Attributes:
        _serial (Serial): The serial connection to the ESP32.
        _subscriber (Subscription): The subscription to the JointTrajectoryPoint messages.

    Methods:
        listener_callback(msg): Handles incoming messages and sends the corresponding servo target angles to ESP32.
    """
    def __init__(self):
        """
        Initializes the ArmSerialWriter node.

        The constructor sets up the serial connection and creates a subscription to
        listen for JointTrajectoryPoint messages.
        """
        super().__init__("arm_serial_writer")

        # Set up the serial connection
        serial_port = self.declare_parameter(
            "serial_port", ARM_SERIAL_PORT_DEFAULT
        ).value
        self._serial = Serial(serial_port, 115200, timeout=0)

        # Subscribe to JointTrajectoryPoint messages
        self._subscriber = self.create_subscription(
            JointTrajectoryPoint,
            DeviceDataTypeEnum.robot_arm,
            self.listener_callback,
            10,
        )

    def listener_callback(self, msg: JointTrajectoryPoint):
        """
        Callback function triggered when a new JointTrajectoryPoint message is received.

        This function converts the radian positions from the message into degrees, formats
        them as a JSON string, and sends them to the ESP32 over the serial connection.

        Args:
            msg (JointTrajectoryPoint): The incoming message containing radian positions.

        Raises:
            orjson.JSONEncodeError: If an error occurs during JSON encoding.
        """
        # TODO send pos to esp32
        # Extract the radian positions from the message
        radian_positions = msg.positions
        self.get_logger().info(f"receive {radian_positions}")

        # Convert radian positions to degrees
        degree_positions = [math.degrees(rad) % 360 for rad in radian_positions]
        ctrl_json = {"servo_target_angles": degree_positions}
        try:
            ctrl_str = orjson.dumps(ctrl_json, option=orjson.OPT_APPEND_NEWLINE)
            self._serial.write(ctrl_str)
        except orjson.JSONEncodeError as error:
            self.get_logger().error(f"Json encode error when recv message: {msg}")
            return
        # Log the output sent to ESP32
        self.get_logger().info(f"{ctrl_str}")


def main(args=None):
    """
    Main function to initialize and run the ArmSerialWriter node.
    
    This function initializes the ROS2 environment, starts the node, and keeps it running.
    """
    rclpy.init(args=args)
    serial_writer = ArmSerialWriter()
    rclpy.spin(serial_writer)

    serial_writer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
