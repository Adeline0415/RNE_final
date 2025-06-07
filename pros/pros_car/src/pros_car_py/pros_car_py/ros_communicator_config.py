# -*- coding: utf-8 -*-

# config.py

"""
vel, rotate_vel為自走車PID數值, 可於arduino程式碼查看

於ros_receive_and_data_processing/AI_node.py使用

前左、前右、後左、後右
"""
vel = 12.0
vel_fast = 24.0
vel_run = 30.0
vel_slow = 5.0
rotate_vel = 8.0
rotate_vel_slow = 4.5
rotate_vel_median = 5.0
delta = 0.75
ACTION_MAPPINGS = {
    "FORWARD": [vel, vel, vel, vel],  # 前進
    "FORWARD_RUN": [vel_run, vel_run, vel_run, vel_run],  # 前進
    "FORWARD_FAST": [vel_fast, vel_fast, vel_fast, vel_fast],  # 前進
    "FORWARD_SLOW": [vel_slow, vel_slow, vel_slow, vel_slow],  # 前進
    "FORWARD_WITH_LEFT_CORRECTION": [vel, vel + delta, vel, vel + delta],
    "FORWARD_WITH_RIGHT_CORRECTION": [vel + delta, vel, vel + delta, vel],
    "FORWARD_FAST_WITH_LEFT_CORRECTION": [vel_fast, vel_fast + 1, vel_fast, vel_fast + 1],
    "FORWARD_FAST_WITH_RIGHT_CORRECTION": [vel_fast + 1, vel_fast, vel_fast + 1, vel_fast],
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
    "BACKWARD_RUN": [-vel_run, -vel_run, -vel_run, -vel_run],
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
