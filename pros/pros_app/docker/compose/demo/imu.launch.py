from launch import LaunchDescription
from launch.actions import TimerAction, LogInfo
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    current_dir = os.path.dirname(__file__)
    ekf_config_file = os.path.join(current_dir, "ekf_config.yaml")

    wit_ros2_imu_node = Node(
        package="wit_ros2_imu",
        executable="wit_ros2_imu",
        name="wit_ros2_imu",
        output="screen",
    )

    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[ekf_config_file],
        remappings=[("odometry", "odometry"), ("imu/data", "imu/data")],
    )

    wit_ros2_imu_success_message = LogInfo(msg="wit_ros2_imu 啟動成功!")

    ekf_startup_message = TimerAction(
        period=5.0, actions=[ekf_node, LogInfo(msg="ekf_node 啟動成功!")]
    )

    return LaunchDescription(
        [
            wit_ros2_imu_node,
            wit_ros2_imu_success_message,
            ekf_startup_message,
        ]
    )
