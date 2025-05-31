# -*- coding: utf-8 -*-

# config.py

"""
vel, rotate_vel為自走車PID數值, 可於arduino程式碼查看

於ros_receive_and_data_processing/AI_node.py使用

前左、前右、後左、後右
"""
vel = 8.0
vel_slow = 5.0
rotate_vel = 8.0
rotate_vel_slow = 3.0
rotate_vel_median = 5.0
ACTION_MAPPINGS = {
    "FORWARD": [vel, vel, vel, vel],  # 前進
    "FORWARD_SLOW": [vel_slow, vel_slow, vel_slow, vel_slow],  # 前進
    "LEFT_FRONT": [rotate_vel, rotate_vel * 1.2, rotate_vel, rotate_vel * 1.2],  # 左前
    "COUNTERCLOCKWISE_ROTATION": [
        -rotate_vel,
        rotate_vel,
        -rotate_vel,
        rotate_vel,
    ],  # 左自轉
    "COUNTERCLOCKWISE_ROTATION_SLOW": [
        -rotate_vel_slow,
        rotate_vel_slow,
        -rotate_vel_slow,
        rotate_vel_slow,
    ],  # 慢左自轉
    "COUNTERCLOCKWISE_ROTATION_MEDIAN": [
        -rotate_vel_median,
        rotate_vel_median,
        -rotate_vel_median,
        rotate_vel_median,
    ],  # 中速左自轉
    "BACKWARD": [-vel, -vel, -vel, -vel],  # 後退
    "BACKWARD_SLOW": [-vel_slow, -vel_slow, -vel_slow, -vel_slow],  # 後退
    "CLOCKWISE_ROTATION": [rotate_vel, -rotate_vel, rotate_vel, -rotate_vel],  # 右自轉
    "CLOCKWISE_ROTATION_SLOW": [
        rotate_vel_slow,
        -rotate_vel_slow,
        rotate_vel_slow,
        -rotate_vel_slow,
    ],  # 右慢自轉
    "CLOCKWISE_ROTATION_MEDIAN": [
        rotate_vel_median,
        -rotate_vel_median,
        rotate_vel_median,
        -rotate_vel_median,
    ],  # 中右自轉
    "RIGHT_FRONT": [rotate_vel * 1.2, rotate_vel, rotate_vel * 1.2, rotate_vel],  # 右前
    "RIGHT_SHIFT": [rotate_vel, -rotate_vel, -rotate_vel, rotate_vel],
    "LEFT_SHIFT": [-rotate_vel, rotate_vel, rotate_vel, -rotate_vel],
    "STOP": [0.0, 0.0, 0.0, 0.0],
}
