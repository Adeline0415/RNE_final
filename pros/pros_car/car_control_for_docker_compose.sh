#!/bin/bash
source /opt/ros/humble/setup.bash
colcon build
. ./install/setup.bash
ros2 run pros_car_py carB_keboard
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
# ros2 run pros_car_py carB_writer
tail -f /dev/null