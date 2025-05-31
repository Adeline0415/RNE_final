from math import pi
from typing import Tuple
import numpy as np
from math import atan2, degrees
import math

"""
Calculate wheel speeds based on the given command velocity.

Parameters :
    cmd_vel_nav: ROS2 Topic data

Returns :
    A tuple of two floats representing the left and right wheel speeds in PID (revolutions per minute).

Example :
    car_data = self.node.wait_for_data()
    self.pwm_left, self.pwm_right = self.calculate_wheel_speeds(car_data["navigation_data"])
"""


def get_yaw_from_quaternion(z, w):
    """四位數的z、w取得偏行角"""
    return np.degrees(2 * np.arctan2(z, w))


def get_direction_vector(current_position, target_position):
    """計算目前車體位置指向目標的vector"""
    return np.array(target_position) - np.array(current_position)


def get_angle_to_target(car_yaw, direction_vector):
    """計算car與target之間的角度差"""
    target_yaw = np.arctan2(direction_vector[1], direction_vector[0])
    angle_diff = target_yaw - np.radians(car_yaw)

    return (np.degrees(angle_diff)) % 360


def calculate_angle_point(car_quaternion_1, car_quaternion_2, car_pos, target_pos):
    """回傳車頭面對目標的角度, 左轉是0~-180, 右轉是0~180, 越接近0代表車頭越正面於目標"""
    car_yaw = get_yaw_from_quaternion(car_quaternion_1, car_quaternion_2)
    direction_vector = get_direction_vector(car_pos, target_pos)
    angle_to_target = get_angle_to_target(car_yaw, direction_vector)
    angle_diff = angle_to_target
    if angle_diff > 180:
        angle_diff -= 360
    return angle_diff


def quaternion_to_euler(z, w):
    t0 = +2.0 * (w * z + 0.0 * 0.0)
    t1 = +1.0 - 2.0 * (0.0 * 0.0 + z * z)
    yaw = atan2(t0, t1)
    return yaw


def calculate_angle_to_target(vehicle_pos, target_pos, vehicle_orientation):
    # 計算目標點相對於車子位置的向量
    dx = target_pos[0] - vehicle_pos[0]
    dy = target_pos[1] - vehicle_pos[1]

    # 計算車子到目標點的方向角度
    target_angle = atan2(dy, dx)

    # 將四元數轉換成偏航角
    vehicle_yaw = quaternion_to_euler(vehicle_orientation[0], vehicle_orientation[1])

    # 計算車子當前朝向與目標方向之間的角度差
    angle_difference = target_angle - vehicle_yaw

    # 規定角度在 [-π, π] 區間
    angle_difference = atan2(math.sin(angle_difference), math.cos(angle_difference))

    return degrees(angle_difference)


def round_to_decimal_places(data_list, decimal_places=3):
    return [round(num, decimal_places) for num in data_list]


def cal_distance(car_pos, target_pos):
    return math.hypot(car_pos[0] - target_pos[0], car_pos[1] - target_pos[1])
