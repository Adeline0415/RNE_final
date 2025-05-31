import orjson
import math
from .env_crane import ARM_SERIAL_PORT_DEFAULT
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import std_msgs.msg
from trajectory_msgs.msg import JointTrajectoryPoint

from serial import Serial


class CraneSerialWriter(Node):
    def __init__(self):
        super().__init__('arm_serial_writer')

        # Set up the serial connection
        serial_port = self.declare_parameter('serial_port', ARM_SERIAL_PORT_DEFAULT).value
        self._serial = Serial(serial_port, 115200, timeout=0)

        #  subscribe
        self.joint_trajectory_subscriber = self.create_subscription(
            JointTrajectoryPoint,
            'joint_trajectory_point',
            self.joint_trajectory_listener_callback,
            10
        )

        self.crane_state_subscriber = self.create_subscription(
            std_msgs.msg.String,
            'crane_state',
            self.crane_state_listener_callback,
            10
        )

    def joint_trajectory_listener_callback(self, msg: JointTrajectoryPoint):
        radian_positions = msg.positions
        self.get_logger().info(f"receive {radian_positions}")

        # radian to degree
        degree_positions = [math.degrees(rad) % 360 for rad in radian_positions]
        ctrl_json = {"servo_target_angles": degree_positions}
        try:
            ctrl_str = orjson.dumps(ctrl_json, option=orjson.OPT_APPEND_NEWLINE)

            self._serial.write(ctrl_str)
        except orjson.JSONEncodeError as error:
            self.get_logger().error(f"Json encode error when recv message: {msg}")
            return
        # log
        self.get_logger().info(f"{ctrl_str}")

    def crane_state_listener_callback(self, msg: std_msgs.msg.String):
        crane_state_dict = orjson.loads(msg.data)
        self.get_logger().info(f"receive {crane_state_dict}")
                                
        try:
            crane_state = crane_state_dict.get('data').get('crane_state')
                      
            ctrl_json = {"crane_state": crane_state}             
            ctrl_str = orjson.dumps(ctrl_json, option=orjson.OPT_APPEND_NEWLINE)
            ctrl_str = orjson.dumps(ctrl_json, option=orjson.OPT_APPEND_NEWLINE)
            self.get_logger().info(f"{ctrl_str}")
            ctrl_str = orjson.dumps(ctrl_json, option=orjson.OPT_APPEND_NEWLINE)            
            self.get_logger().info(f"{ctrl_str}")
            self._serial.write(ctrl_str)
        except orjson.JSONEncodeError as error:
            self.get_logger().error(f"Json encode error when recv message: {msg}")
            return
        # log
        self.get_logger().info(f"{ctrl_str}")

def main(args=None):
    rclpy.init(args=args)
    serial_writer = CraneSerialWriter()
    rclpy.spin(serial_writer)

    serial_writer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
