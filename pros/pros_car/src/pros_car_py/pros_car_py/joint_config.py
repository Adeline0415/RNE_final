# joint_config.py

import math

# 定義關節更新的數據
# 給 update_joint_position 專用
JOINT_UPDATES_POSITIVE = [
    (0, math.radians(10), 0, 3600),
    (1, math.radians(10), 0, 3600),
    (2, math.radians(10), 0, 3600),
    (3, math.radians(10), 0, 3600),
]

JOINT_UPDATES_NEGATIVE = [
    (0, math.radians(-10), 0, 3600),
    (1, math.radians(-10), 0, 3600),
    (2, math.radians(-10), 0, 3600),
    (3, math.radians(-10), 0, 3600),
]
