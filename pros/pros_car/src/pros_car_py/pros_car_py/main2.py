import urwid
import os
import threading
import rclpy
import time
import io
import sys
from pros_car_py.joint_config import JOINT_UPDATES_POSITIVE, JOINT_UPDATES_NEGATIVE
from pros_car_py.car_controller import CarController
from pros_car_py.arm_controller import ArmController
from pros_car_py.data_processor import DataProcessor
from pros_car_py.nav_processing import Nav2Processing
from pros_car_py.ros_communicator import RosCommunicator
from pros_car_py.crane_controller import CraneController
from pros_car_py.custom_control import CustomControl
from pros_car_py.ik_solver import PybulletRobotController
from pros_car_py.mode_app import ModeApp


def init_ros_node():
    rclpy.init()
    node = RosCommunicator()
    thread = threading.Thread(target=rclpy.spin, args=(node,))
    thread.start()
    return node, thread


def main():
    ros_communicator, ros_thread = init_ros_node()
    data_processor = DataProcessor(ros_communicator)
    nav2_processing = Nav2Processing(ros_communicator, data_processor)
    ik_solver = PybulletRobotController(end_eff_index=5)
    car_controller = CarController(ros_communicator, nav2_processing)
    arm_controller = ArmController(
        ros_communicator, data_processor, ik_solver, num_joints=5
    )
    crane_controller = CraneController(
        ros_communicator, data_processor, ik_solver, num_joints=7
    )
    custom_control = CustomControl(car_controller, arm_controller)
    app = ModeApp(car_controller, arm_controller, custom_control, crane_controller)

    try:
        app.main()
    finally:
        rclpy.shutdown()
        ros_thread.join()


if __name__ == "__main__":
    main()
