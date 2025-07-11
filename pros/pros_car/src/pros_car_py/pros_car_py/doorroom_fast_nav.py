#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from std_msgs.msg import String, Float32MultiArray
from pros_car_py.ros_communicator_config import ACTION_MAPPINGS
from pros_car_py.car_models import DeviceDataTypeEnum
from sensor_msgs.msg import CompressedImage
from enum import Enum, auto
import time
import cv2
import numpy as np
from cv_bridge import CvBridge
import math

class DoorRoomState(Enum):
    INIT = auto()                 # 初始化
    TURNING_LEFT_90 = auto()      # 逆時鐘轉90度朝左
    TURNING_RIGHT_90 = auto()     # 順時鐘轉90度朝右
    MOVING_TO_LEFT_WALL = auto()  # 移動到最左邊牆壁
    TURNING_RIGHT_180 = auto()    # 轉向面對右邊開始掃描
    TURNING_LEFT_180 = auto()     # 轉向面對左邊開始掃描
    PASSING_THROUGH_DOOR = auto() # 通過門
    MOVING_TO_RIGHT_DOOR = auto() # 移動到右邊下一個門的位置
    MOVING_TO_LEFT_DOOR = auto()  # 移動到最左邊還沒通過的門的位置
    SCANNING_FOR_PIKACHU = auto() # 掃描皮卡丘
    APPROACHING_PIKACHU = auto()  # 接近皮卡丘
    SUCCESS = auto()              # 成功

class CarDirection(Enum):
    FORWARD = auto()
    LEFT = auto()
    RIGHT = auto()
    BACKWARD = auto()

class DoorRoomNav(Node):
    def __init__(self):
        super().__init__('doorroom_nav')
        
        # === 狀態管理 ===
        self.state = DoorRoomState.INIT
        self.state_start_time = None
        self.clock = Clock()
        # === 車頭方向追蹤 ===
        self.car_direction = CarDirection.FORWARD
        self.rotation_angle = 0.0
        self.is_rotating = False
        self.rotation_direction = None
        self.rotation_start_time = None
        self.quarter_rotation_clockwise_time = 2.5 # 2.25 
        self.quarter_rotation_counterclockwise_time = 3.0 # 2.75
        self.quarter_rotation_mid_clockwise_time = 8.75 #7.75 #7.5
        self.quarter_rotation_mid_counterclockwise_time = 12.25 # 11.25

        # === 門的位置管理 ===
        self.doors_passed = 0  # 已通過的門數量
        # self.door_distance_move_time = 2.9  # 移動到下一個門位置的時間 fast
        self.door_distance_move_time = 6.1

        # === 門位置追蹤系統 ===
        self.door_bitmap = [0, 0, 0, 0]  # 確保最先初始化
        self.cur_door = 1.5
        self.last_door = -1
        self.door_positions = [0, 1, 2, 3]
        # self.single_door_move_time = 2.9 #fast
        self.single_door_move_time = 6.1
        
        # === 牆壁檢測 ===
        self.wall_color_range = None  # 將在運行時設定
        self.wall_reached = False  # 是否已到達牆壁
        self.wall_move_duration = 4.2  # 到達牆壁所需時間
        # self.extra_move_duration = 1.5 #fast
        self.extra_move_duration = 4.0 #3.5 #slow
        
        # === 皮卡丘檢測 ===
        self.pikachu_detected = False
        self.delta_x = 0.0  # 像素偏移（左負右正）
        self.pikachu_total_area = 0.0
        self.target_area_threshold = 50000 

        # === 最終接近階段 === 
        self.final_approach_start_time = None
        self.final_approach_duration = 3.0
        self.final_approach_total_duration = 7.5
        
        # === 移動計時 ===
        self.move_start_time = None
        self.scan_start_time = None
        #slow
        # self.turn_duration_left = 13.75  #timeout
        # self.turn_duration_right = 11.0  #timeout
        self.turn_duration_left = 5.0  #timeout
        self.turn_duration_right = 4.5 #timeout


        # === 顏色檢測系統 ===
        self.color_ranges = {
            'blue': {
                'target_bgr': [154, 140, 94],
                'lower': np.array([144, 130, 84]),
                'upper': np.array([200, 180, 140])
            },
            'gray': {
                'target_bgr': [100, 100, 100], 
                'lower': np.array([70, 70, 70]),
                'upper': np.array([145, 140, 140])
            },
            'brown': {
                'target_bgr': [90, 133, 190], #115 138 170 #102 142 191 # 170 138 115 #84 113 150 #122 144 178
                'lower': np.array([65, 95, 135]),
                'upper': np.array([137, 185, 235])
            },
            'white': {
                'target_bgr': [250, 250, 250], 
                'lower': np.array([240, 240, 240]),
                'upper': np.array([255, 255, 255])
            },
            'light_blue': {
                'target_bgr': [222, 203, 136], #229 200 133, 216 197 130
                'lower': np.array([210, 193, 126]),
                'upper': np.array([242, 223, 155])
            },
            'yellow':{
                'target_bgr': [11, 228, 248], 
                'lower': np.array([0, 220, 240]),
                'upper': np.array([20, 245, 255])
            }
        }
                
        # === 設置訂閱者和發布者 ===
        self.setup_subscribers()
        self.setup_publishers()
        
        self.bridge = CvBridge()
        self.current_rgb_image = None
        
        # === 初始化 ===
        self.initialize_mission()
        
        # 主循環定時器
        self.timer = self.create_timer(0.1, self.main_loop)  # 10Hz
        
        self.get_logger().info("DoorRoom導航器啟動")

    def setup_subscribers(self):
        """設置訂閱者"""
        self.yolo_target_info_sub = self.create_subscription(
            Float32MultiArray, '/yolo/target_info', self.yolo_target_info_callback, 10)
        self.rgb_image_sub = self.create_subscription(
            CompressedImage, '/camera/image/compressed', self.rgb_image_callback, 10)

    def setup_publishers(self):
        """設置發布者"""
        # 車輛控制
        self.rear_wheel_pub = self.create_publisher(
            Float32MultiArray, DeviceDataTypeEnum.car_C_rear_wheel, 10)
        self.front_wheel_pub = self.create_publisher(
            Float32MultiArray, DeviceDataTypeEnum.car_C_front_wheel, 10)

    def initialize_mission(self):
        """初始化任務"""
        self.change_state(DoorRoomState.INIT)
        self.get_logger().info("開始DoorRoom導航任務")

    def change_state(self, new_state):
        """改變狀態"""
        old_state = self.state.name if hasattr(self, 'state') else "None"
        self.state = new_state
        self.state_start_time = self.clock.now()
        self.get_logger().info(f"狀態切換: {old_state} → {new_state.name}")

    def publish_car_control(self, action_key):
        """發布車輛控制指令"""
        if action_key not in ACTION_MAPPINGS:
            self.get_logger().warn(f"未知動作: {action_key}")
            return

        # 處理旋轉動作的方向追蹤
        if action_key in ["COUNTERCLOCKWISE_ROTATION_MEDIAN", "COUNTERCLOCKWISE_ROTATION"]:
            if self.is_rotating and self.rotation_direction == "clockwise":
                # 正在順時鐘旋轉，需要先停止
                self.stop_rotation()
            if not self.is_rotating:
                self.start_rotation("counterclockwise")
        elif action_key in ["CLOCKWISE_ROTATION_MEDIAN", "CLOCKWISE_ROTATION"]:
            if self.is_rotating and self.rotation_direction == "counterclockwise":
                # 正在逆時鐘旋轉，需要先停止
                self.stop_rotation()
            if not self.is_rotating:
                self.start_rotation("clockwise")
        else:
            # 停止旋轉動作
            if self.is_rotating:
                self.stop_rotation()

        velocities = ACTION_MAPPINGS[action_key]
        vel1, vel2, vel3, vel4 = velocities
        
        # 發布後輪
        rear_msg = Float32MultiArray()
        rear_msg.data = [vel1, vel2]
        self.rear_wheel_pub.publish(rear_msg)
        
        # 發布前輪
        front_msg = Float32MultiArray()
        front_msg.data = [vel3, vel4]
        self.front_wheel_pub.publish(front_msg)

    # === 回調函數 ===
    def yolo_target_info_callback(self, msg):
        """YOLO目標信息回調"""
        try:
            if len(msg.data) >= 2:
                found_target = bool(msg.data[0])  # 是否找到皮卡丘
                self.delta_x = msg.data[1]       # 像素偏移（左負右正）
                self.pikachu_total_area = msg.data[2] if len(msg.data) >= 3 else 0.0
                self.pikachu_detected = found_target
                
        except Exception as e:
            self.get_logger().error(f"YOLO回調錯誤: {e}")

    def rgb_image_callback(self, msg):
        """RGB圖像回調"""
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            self.current_rgb_image = cv_image
        except Exception as e:
            self.get_logger().error(f"RGB圖像回調錯誤: {e}")

    # === 導航邏輯 ===
    
    def get_target_door_position(self, direction="left"):
        """計算目標門位置 - 統一邏輯
        Args:
            direction (str): "left" 找最左邊未訪問門, "right" 找右邊最近未訪問門
        """
        unvisited_doors = [i for i, visited in enumerate(self.door_bitmap) if visited == 0]
        
        if not unvisited_doors:
            return None
        
        if direction == "left":
            # 返回最左邊的未訪問門
            return min(unvisited_doors)
        
        elif direction == "right":
            # 只考慮比當前門位置更右邊的未訪問門
            right_doors = [door for door in unvisited_doors if door > self.cur_door]
            
            if right_doors:
                # 有右邊的門，選擇最近的（最左邊的）
                return min(right_doors)
            else:
                # 沒有右邊的門，返回最左邊的未訪問門（從頭開始）
                self.get_logger().warn("error 找不到右邊未訪問的門")
                return min(unvisited_doors)
        
        return None
    def calculate_move_time_to_door(self, target_door):
        """計算移動到目標門所需的時間"""
        if self.cur_door == 1.5:
            return self.single_door_move_time  # 第一次移動
        
        distance = abs(target_door - self.cur_door)
        return distance * self.single_door_move_time
    
    def update_door_status(self, door_position, passed=False):
        """更新門的狀態"""
        if 0 <= door_position <= 3:
            if passed:
                self.door_bitmap[door_position] = 1
                self.last_door = door_position
                self.doors_passed += 1
                self.get_logger().info(f"通過門{door_position}，狀態更新: {self.door_bitmap}")
            else:
                self.cur_door = door_position
                self.get_logger().info(f"開始掃描門{door_position}")

    def print_rgb_info(self):
        """印出RGB資訊用於調試"""
        if self.current_rgb_image is not None:
            # 計算平均RGB值
            mean_color = np.mean(self.current_rgb_image, axis=(0, 1))
            b, g, r = mean_color
            self.get_logger().info(f"畫面平均RGB: R={r:.1f}, G={g:.1f}, B={b:.1f}")
            
            # 檢查各種顏色比例
            light_blue_ratio = self.check_color_ratio("light_blue")
            blue_ratio = self.check_color_ratio("blue")
            gray_ratio = self.check_color_ratio("gray")
            brown_ratio = self.check_color_ratio("brown")
            self.get_logger().info(f"顏色比例 - 藍色: {blue_ratio:.3f}, 灰色: {gray_ratio:.3f}, 棕色: {brown_ratio:.3f}, 淺藍色: {light_blue_ratio:.3f}")

        self.get_logger().info(f"現在車頭方向:{self.get_current_direction()}")

    def check_color_ratio(self, color_name):
        """檢查畫面中指定顏色的比例"""
        if self.current_rgb_image is None:
            return 0.0
        
        if color_name not in self.color_ranges:
            self.get_logger().error(f"未知顏色: {color_name}")
            return 0.0
        
        try:
            color_info = self.color_ranges[color_name]
            
            # 創建顏色遮罩 - 檢查每個像素是否在指定BGR範圍內
            mask = cv2.inRange(self.current_rgb_image, 
                             color_info['lower'], 
                             color_info['upper'])
            
            # 計算符合顏色範圍的像素數量
            total_pixels = self.current_rgb_image.shape[0] * self.current_rgb_image.shape[1]
            color_pixels = np.sum(mask > 0)  # mask中白色像素(值=255)的數量
            ratio = color_pixels / total_pixels
            
            # 詳細日誌輸出
            self.get_logger().info(
                f"{color_name}檢測: 總像素={total_pixels}, {color_name}像素={color_pixels}, "
                f"比例={ratio:.3f} ({ratio*100:.1f}%)"
            )
            
            return ratio
        except Exception as e:
            self.get_logger().error(f"檢查{color_name}比例錯誤: {e}")
            return 0.0
        
    def turn_left_90_degrees_first(self):
        """逆時鐘轉90度朝左"""
        if self.move_start_time is None:
            self.move_start_time = self.clock.now()
            self.get_logger().info("開始逆時鐘轉90度朝左")
        
        elapsed = (self.clock.now() - self.move_start_time).nanoseconds / 1e9
        
        if elapsed < self.turn_duration_left:
            self.publish_car_control("COUNTERCLOCKWISE_ROTATION")
            has_line, color, angle = self.detect_horizontal_line()

            if has_line and (self.check_color_ratio("blue") > 0.4) and (color in ["blue", "light_blue"]):
                self.get_logger().info("檢測到水平線，已轉90度朝左")
                self.move_start_time = None
                self.change_state(DoorRoomState.MOVING_TO_LEFT_WALL)

        else:
            # 超時也認為已轉到位
            self.get_logger().info("轉向超時，假設已朝左")
            self.move_start_time = None
            self.change_state(DoorRoomState.MOVING_TO_LEFT_WALL)

    def turn_left_90_degrees(self):
        """逆時鐘轉90度朝左"""
        if self.move_start_time is None:
            self.move_start_time = self.clock.now()
            self.get_logger().info("開始逆時鐘轉90度朝左")
        
        elapsed = (self.clock.now() - self.move_start_time).nanoseconds / 1e9
        
        if elapsed < self.turn_duration_left:
            self.publish_car_control("COUNTERCLOCKWISE_ROTATION")
            has_line, color, angle = self.detect_horizontal_line()
            # 檢查是否看到水平線（表示已轉90度）
            if has_line and (color in ["blue", "light_blue"]):
                self.get_logger().info("檢測到水平線，已轉90度朝左")
                # self.car_direction = CarDirection.LEFT
                self.move_start_time = None
                self.change_state(DoorRoomState.MOVING_TO_LEFT_DOOR)

        else:
            # 超時也認為已轉到位
            self.get_logger().info("轉向超時，假設已朝左")
            # self.car_direction = CarDirection.LEFT
            self.move_start_time = None
            self.change_state(DoorRoomState.MOVING_TO_LEFT_DOOR)

    def turn_right_90_degrees(self):
        """順時鐘轉90度朝右"""
        if self.move_start_time is None:
            self.move_start_time = self.clock.now()
            self.get_logger().info("開始順時鐘轉90度朝右")
        
        elapsed = (self.clock.now() - self.move_start_time).nanoseconds / 1e9
        
        if elapsed < self.turn_duration_right:
            self.publish_car_control("CLOCKWISE_ROTATION")
            has_line, color, angle = self.detect_horizontal_line()
            if has_line and (color in ["blue", "light_blue"]):
                self.get_logger().info("檢測到水平線，已轉90度朝右")
                # self.car_direction = CarDirection.RIGHT
                self.move_start_time = None
                self.change_state(DoorRoomState.MOVING_TO_RIGHT_DOOR)
        else:
            self.get_logger().info("轉向超時，假設已朝右")
            # self.car_direction = CarDirection.RIGHT
            self.move_start_time = None
            self.change_state(DoorRoomState.MOVING_TO_RIGHT_DOOR)

    def move_to_left_wall(self):
        """往前移動直到到達左邊牆壁（畫面都是淺藍色），然後再直走1.5秒"""
        # 印出RGB資訊用於調試
        self.print_rgb_info()
        
        # 檢查是否到達牆壁
        blue_ratio = self.check_color_ratio("blue")
        self.car_direction = CarDirection.LEFT
        
        if not self.wall_reached:
            # 還沒到達牆壁，繼續前進
            self.publish_corrected_forward_control()
            
            if blue_ratio > 0.8:
                self.get_logger().info(f"檢測到左邊牆壁！淺藍色比例: {blue_ratio:.3f}")
                self.wall_reached = True
                self.move_start_time = self.clock.now()  # 開始計時額外移動時間
        else:
            # 已經到達牆壁，繼續直走1.5秒
            if self.move_start_time is None:
                self.move_start_time = self.clock.now()
            
            elapsed = (self.clock.now() - self.move_start_time).nanoseconds / 1e9
            
            if elapsed < self.extra_move_duration:
                self.publish_car_control("FORWARD")
                remaining_time = self.extra_move_duration - elapsed
                self.get_logger().info(f"到達牆壁後額外前進中...剩餘{remaining_time:.1f}秒")
            else:
                self.get_logger().info("完成牆壁移動，開始轉向面對右邊")
                self.update_door_status(0, passed=False)
                self.wall_reached = False  # 重置狀態
                self.move_start_time = None
                self.change_state(DoorRoomState.TURNING_RIGHT_180)

    def turn_right_180_degrees(self):
        """轉向面對右邊準備掃描，轉向過程中就檢測門"""
        if self.move_start_time is None:
            self.move_start_time = self.clock.now()
            self.get_logger().info("開始轉向面對右邊（轉向過程中檢測門）")
        
        elapsed = (self.clock.now() - self.move_start_time).nanoseconds / 1e9
        
        if elapsed < self.turn_duration_right*2:
            self.publish_car_control("CLOCKWISE_ROTATION")
            cur_dir = self.get_current_direction()
            has_line, color, angle = self.detect_horizontal_line()
            # 顏色版本
            if has_line and color == "brown" and (self.check_color_ratio("gray") > 0.05) and (self.doors_passed != 2):
                self.get_logger().info("轉向過程中檢測到門的水平線！")
                # self.car_direction = CarDirection.FORWARD
                self.move_start_time = None
                self.change_state(DoorRoomState.PASSING_THROUGH_DOOR)
                return
            
            if has_line and (self.doors_passed == 2) and (cur_dir != CarDirection.LEFT) and ((self.cur_door == 0) or (self.cur_door == 3)) and ((self.check_color_ratio("brown") < 0.1) or (self.check_color_ratio("light_blue") > 0.3)): ## 最後一層
                self.get_logger().info("轉向過程中檢測到門的水平線！")
                self.move_start_time = None
                self.change_state(DoorRoomState.PASSING_THROUGH_DOOR)
                return
            
            if has_line and (self.doors_passed == 2) and ((self.cur_door == 1) or (self.cur_door == 2)) and ((self.check_color_ratio("yellow") > 0) or (self.check_color_ratio("light_blue")>0.45)): ## 最後一層間一定看得到皮卡丘
                self.get_logger().info("轉向過程中檢測到門的水平線！")
                self.move_start_time = None
                self.change_state(DoorRoomState.PASSING_THROUGH_DOOR)
                return
            
            if has_line and (color in ["blue", "light_blue"]) and (elapsed > self.quarter_rotation_clockwise_time):
                self.get_logger().info("180度轉向完成 檢測到牆壁水平線，往前移動到下一個門位置")
                self.move_start_time = None
                self.change_state(DoorRoomState.MOVING_TO_RIGHT_DOOR)
        else:
            self.get_logger().info("轉向完成但未檢測到門，往前移動到下一個門位置")
            # self.car_direction = CarDirection.RIGHT
            self.move_start_time = None
            self.change_state(DoorRoomState.MOVING_TO_RIGHT_DOOR)

    def turn_left_180_degrees(self):
        """轉向面對左邊準備掃描，轉向過程中就檢測門"""
        if self.move_start_time is None:
            self.move_start_time = self.clock.now()
            self.get_logger().info("開始轉向面對左邊（轉向過程中檢測門）")
        
        elapsed = (self.clock.now() - self.move_start_time).nanoseconds / 1e9
        
        if elapsed < self.turn_duration_left*2:
            self.publish_car_control("COUNTERCLOCKWISE_ROTATION")
            #顏色檢測
            has_line, color, angle = self.detect_horizontal_line()
            cur_dir = self.get_current_direction()
            if has_line and color == "brown" and (self.check_color_ratio("gray") > 0.05) and (self.doors_passed != 2):
                self.get_logger().info("轉向過程中檢測到門的水平線！")
                self.move_start_time = None
                self.change_state(DoorRoomState.PASSING_THROUGH_DOOR)
                return
            
            if has_line and (self.doors_passed == 2) and (cur_dir != CarDirection.RIGHT) and ((self.cur_door == 0) or (self.cur_door == 3)) and ((self.check_color_ratio("brown") < 0.1) or (self.check_color_ratio("light_blue") > 0.3)): ## 最後一層
                self.get_logger().info("轉向過程中檢測到門的水平線！")
                self.move_start_time = None
                self.change_state(DoorRoomState.PASSING_THROUGH_DOOR)
                return
            
            if has_line and (self.doors_passed == 2) and ((self.cur_door == 1) or (self.cur_door == 2)) and ((self.check_color_ratio("yellow") > 0) or (self.check_color_ratio("light_blue")>0.45)): ## 最後一層間一定看得到皮卡丘
                self.get_logger().info("轉向過程中檢測到門的水平線！")
                self.move_start_time = None
                self.change_state(DoorRoomState.PASSING_THROUGH_DOOR)
                return
            
            if has_line and (color in ["blue", "light_blue"]) and (elapsed > self.quarter_rotation_counterclockwise_time):
                self.get_logger().info("180度轉向完成 檢測到牆壁水平線，轉180回右邊")
                self.car_direction = CarDirection.LEFT
                self.move_start_time = None
                self.change_state(DoorRoomState.TURNING_RIGHT_180)
        else:
            self.get_logger().info("轉向完成但未檢測到門，往前移動到下一個門位置")
            self.move_start_time = None
            self.car_direction = CarDirection.LEFT
            self.change_state(DoorRoomState.TURNING_RIGHT_180)

    def pass_through_door(self):
        """通過門"""
        # 檢查是否看不到灰色
        gray_ratio = self.check_color_ratio("gray")
        self.car_direction = CarDirection.FORWARD
        
        if self.move_start_time is None:
            # 還沒開始計時，繼續前進直到看不到灰色
            self.publish_corrected_forward_control()
            
            if gray_ratio == 0 or (self.doors_passed==2):
                # 看不到灰色了，開始計時
                self.move_start_time = self.clock.now()
                self.get_logger().info("已看不到灰色，開始計時通過門")
        else:
            # 已經開始計時，檢查是否達到2秒
            elapsed = (self.clock.now() - self.move_start_time).nanoseconds / 1e9
            
            # if elapsed < 0.65 or ((elapsed < 1.25) and (self.doors_passed==2)):  # 看不到灰色後再走幾秒
            #     self.publish_car_control("FORWARD")
            if ((elapsed < 1.3) and (self.doors_passed==0)) or ((elapsed < 1.75) and (self.doors_passed==1))or ((elapsed < 3.5) and (self.doors_passed==2)):  # 看不到灰色後再走幾秒
                self.publish_car_control("FORWARD")
            else:
                # 完成通過門
                self.update_door_status(self.cur_door, passed=True)
                self.move_start_time = None
                self.get_logger().info(f"已通過{self.doors_passed}道門")
                
                if self.doors_passed >= 3:
                    self.change_state(DoorRoomState.SCANNING_FOR_PIKACHU)
                else:
                    target_door = self.get_target_door_position("left") 
                    if target_door > self.cur_door:
                        self.get_logger().info("通過門後，開始右轉")
                        self.change_state(DoorRoomState.TURNING_RIGHT_90)
                    else:
                        self.get_logger().info("通過門後，開始左轉")
                        self.change_state(DoorRoomState.TURNING_LEFT_90)

    def move_to_right_door(self):
        """移動到下一個門的位置"""
        self.car_direction = CarDirection.RIGHT
        if self.move_start_time is None:
            self.move_start_time = self.clock.now()
            
            # 計算下一個目標門和移動時間
            target_door = self.get_target_door_position("right")
            if target_door is not None:
                move_time = self.calculate_move_time_to_door(target_door)
                self.door_distance_move_time = move_time
                self.get_logger().info(f"移動到右邊下一個門{target_door}的位置，預計移動時間: {move_time:.1f}秒")
                self.update_door_status(target_door, passed=False)
            else:
                self.get_logger().info("沒有更多門需要檢查，開始尋找皮卡丘")
                self.change_state(DoorRoomState.SCANNING_FOR_PIKACHU)
                return
        
        elapsed = (self.clock.now() - self.move_start_time).nanoseconds / 1e9
        
        if elapsed < self.door_distance_move_time:
            self.publish_corrected_forward_control()
        else:
            self.move_start_time = None
            self.change_state(DoorRoomState.TURNING_LEFT_180)

    def move_to_left_door(self):
        """移動到最左邊未訪問的門位置"""
        self.car_direction = CarDirection.LEFT
        if self.move_start_time is None:
            self.move_start_time = self.clock.now()
            
            target_door = self.get_target_door_position("left")  # 使用 "left" 模式
            if target_door is not None:
                move_time = self.calculate_move_time_to_door(target_door)
                self.door_distance_move_time = move_time
                self.get_logger().info(f"移動到最左邊門{target_door}，移動時間: {move_time:.1f}秒")
                self.update_door_status(target_door, passed=False)
            else:
                self.change_state(DoorRoomState.SCANNING_FOR_PIKACHU)
                return
        
        elapsed = (self.clock.now() - self.move_start_time).nanoseconds / 1e9
        
        if elapsed < self.door_distance_move_time:
            self.publish_corrected_forward_control()
        else:
            self.move_start_time = None
            self.change_state(DoorRoomState.TURNING_RIGHT_180)

    def customized_final_approach_time(self):
        if self.last_door == 0 or self.last_door == 3:
            return self.final_approach_duration + 1.0
        else:
            return self.final_approach_duration

    def scan_for_pikachu(self):
        """掃描皮卡丘"""
        if self.last_door < 2:
            self.publish_car_control("CLOCKWISE_ROTATION")
        else:
            self.publish_car_control("COUNTERCLOCKWISE_ROTATION")
        self.get_logger().info("掃描皮卡丘中...")

    def approach_pikachu(self):
        """接近皮卡丘 - 當面積達到閾值時進入最終階段"""
        # 如果已經在最終接近階段，直接執行不受檢測影響
        # 記錄開始接近的時間（只在第一次進入時記錄）
        if not hasattr(self, 'approach_start_time') or self.approach_start_time is None:
            self.approach_start_time = self.clock.now()
            self.get_logger().info("開始接近皮卡丘，啟動計時器")
        
        # 如果已經在最終接近階段，直接執行不受檢測影響
        if self.final_approach_start_time is not None:
            customized_final_approach_time = self.customized_final_approach_time()
            elapsed = (self.clock.now() - self.final_approach_start_time).nanoseconds / 1e9
            if elapsed < customized_final_approach_time:
                self.publish_car_control("FORWARD")
                remaining_time = customized_final_approach_time - elapsed
                self.get_logger().info(f"最終接近中...剩餘{remaining_time:.1f}秒")
            else:
                self.change_state(DoorRoomState.SUCCESS)
            return  

        # 檢查是否滿足進入最終接近的條件
        approach_elapsed = (self.clock.now() - self.approach_start_time).nanoseconds / 1e9
        area_reached = (self.pikachu_total_area > self.target_area_threshold) and ((self.last_door == 0) or (self.last_door == 3))
        timeout_reached = approach_elapsed >= 4.0  # 4秒超時
        
        # 面積達標或超時6秒，進入最終接近階段
        if area_reached or timeout_reached:
            self.final_approach_start_time = self.clock.now()
            if area_reached:
                self.get_logger().info(f"面積達標 ({self.pikachu_total_area:.0f}px²)，進入最終接近階段")
            if timeout_reached:
                self.get_logger().info(f"接近超時 ({approach_elapsed:.1f}秒)，強制進入最終接近階段")
            return
        
        alignment_threshold = 50
        
        if abs(self.delta_x) > alignment_threshold:
            if self.delta_x < 0:
                self.publish_car_control("COUNTERCLOCKWISE_ROTATION")
            else:
                self.publish_car_control("CLOCKWISE_ROTATION")
        else:
            self.publish_car_control("FORWARD_FAST")
            self.get_logger().info(f"前進接近 (面積: {self.pikachu_total_area:.0f}px²)")

    def detect_horizontal_line(self):
        """檢測水平線並返回上方顏色
        Returns:
            tuple: (是否有水平線 bool, 上方主要顏色 str or None)
        """
        if self.current_rgb_image is None:
            return False, None, None
        
        try:
            gray = cv2.cvtColor(self.current_rgb_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=80,
                minLineLength=30,
                maxLineGap=10
            )
        
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # 計算線段長度
                    length = ((x2-x1)**2 + (y2-y1)**2)**0.5
                    
                    # 只檢查夠長的線段
                    if length < 20:  # 最小長度閾值
                        continue
                        
                    # 計算斜率角度
                    if x2 - x1 != 0:
                        slope = (y2 - y1) / (x2 - x1)
                        angle = abs(math.atan(slope) * 180 / math.pi)
                        angle_with_sign = math.atan(slope) * 180 / math.pi  # 保留正負號
                    else:
                        angle = 90  # 垂直線
                        angle_with_sign = 90
                    
                    # 水平線角度應該接近0度
                    if angle <= 0.05:  # 允許5度誤差
                        # 檢測水平線上方的顏色
                        color = self.get_color_above_line(x1, y1, x2, y2)
                        
                        self.get_logger().info(f"檢測到水平線: ({x1},{y1}) -> ({x2},{y2}), 上方顏色: {color}, 傾斜角度: {angle_with_sign:.2f}度")
                        return True, color, angle_with_sign
            
            return False, None, None
            
        except Exception as e:
            self.get_logger().error(f"水平線檢測錯誤: {e}")
            return False, None, None

    def get_color_above_line(self, x1, y1, x2, y2):
        """獲取水平線上方5-15像素區域的主要顏色
        Args:
            x1, y1, x2, y2: 水平線的座標
        Returns:
            str: 主要顏色名稱 ("brown", "blue") 或 None
        """
        try:
            # 確定水平線的範圍
            line_y = min(y1, y2)
            left_x = min(x1, x2)
            right_x = max(x1, x2)
            
            # 定義檢測區域：線上方5-15像素
            upper_y = max(0, line_y - 21)
            lower_y = max(0, line_y - 1)
            
            # 確保座標在圖像範圍內
            height, width = self.current_rgb_image.shape[:2]
            upper_y = max(0, upper_y)
            lower_y = min(height, lower_y)
            left_x = max(0, left_x)
            right_x = min(width, right_x)
            
            # 提取檢測區域
            if upper_y < lower_y and left_x < right_x:
                region = self.current_rgb_image[upper_y:lower_y, left_x:right_x]
                
                if region.size == 0:
                    return None
                
                total_pixels = region.shape[0] * region.shape[1]
                target_colors = ['blue', 'brown', 'light_blue']  # 只考慮這四種顏色
                color_results = {}
                
                # 只檢查指定的四種顏色在區域中的比例
                for color_name in target_colors:
                    if color_name in self.color_ranges:
                        color_info = self.color_ranges[color_name]
                        mask = cv2.inRange(region, color_info['lower'], color_info['upper'])
                        color_pixels = np.sum(mask > 0)
                        ratio = color_pixels / total_pixels
                        color_results[color_name] = ratio
                
                # 找出四種顏色中比例最高的
                if color_results:  # 確保有結果
                    max_color = max(color_results, key=color_results.get)
                    max_ratio = color_results[max_color]
                    
                    self.get_logger().info(f"顏色檢測結果: {color_results}")
                    
                    # 可以設定一個最低閾值，或者直接返回比例最高的
                    if max_ratio > 0.1:  # 降低閾值到10%
                        return max_color
                    else:
                        # 即使比例很低，也返回最高的那個
                        return max_color
            
            return None
            
        except Exception as e:
            self.get_logger().error(f"檢測線上方顏色錯誤: {e}")
        return None
    
    def detect_wide_horizontal_line(self):
        """檢測水平線並返回上方顏色
        Returns:
            tuple: (是否有水平線 bool, 上方主要顏色 str or None)
        """
        if self.current_rgb_image is None:
            return False, None
        
        try:
            gray = cv2.cvtColor(self.current_rgb_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=80,
                minLineLength=30,
                maxLineGap=10
            )
        
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0] # x1 小 x2大 --> x1 左 --> 
                    
                    # 計算線段長度
                    length = ((x2-x1)**2 + (y2-y1)**2)**0.5
                    if length < 20:  # 最小長度閾值
                        continue

                    height = self.current_rgb_image.shape[0]
                    is_ground = ((y1 > height/2) and (y2 > height/2)) 
                    
                    # 計算斜率角度
                    if x2 - x1 != 0:
                        if is_ground: 
                            slope = ((y2 - y1) / (x2 - x1)) #地板
                        else:
                            slope = -((y2 - y1) / (x2 - x1)) #右-左  天空  正: 右邊高往右轉 負:左邊高往左轉 
                        angle = abs(math.atan(slope) * 180 / math.pi)
                        angle_with_sign = math.atan(slope) * 180 / math.pi  # 保留正負號
                    else:
                        angle = 90  # 垂直線
                        angle_with_sign = 90
                    
                    # 水平線角度應該接近0度
                    if angle <= 3:  # 允許5度誤差
                        self.get_logger().info(f"檢測到水平線: ({x1},{y1}) -> ({x2},{y2}), 傾斜角度: {angle_with_sign:.2f}度")
                        return True, angle_with_sign
            
            return False, None
            
        except Exception as e:
            self.get_logger().error(f"水平線檢測錯誤: {e}")
            return False, None
    
    def correct_direction_with_horizontal_line(self):
        """根據水平線傾斜度校正行進方向"""
        has_line, tilt_angle = self.detect_wide_horizontal_line()
        
        if not has_line or tilt_angle is None:
            self.get_logger().info("未檢測到水平線，直接前進")
            return "FORWARD"
        
        angle_threshold = 0.5
        correction_threshold = 0.5
        
        if abs(tilt_angle) <= angle_threshold:
            return "FORWARD"
        elif tilt_angle > correction_threshold: # 正往右轉
            self.get_logger().info(f" {tilt_angle:.2f}度，向右校正")
            return "FORWARD_FAST_WITH_RIGHT_CORRECTION"
        elif tilt_angle < -correction_threshold: #負往左轉
            self.get_logger().info(f"{tilt_angle:.2f}度，向左校正")
            return "FORWARD_FAST_WITH_LEFT_CORRECTION"
        else:
            return "FORWARD"
        
    def publish_corrected_forward_control(self):
        """發布帶有校正的前進指令"""
        correction_action = self.correct_direction_with_horizontal_line()
        
        if correction_action == "FORWARD_FAST_WITH_LEFT_CORRECTION":
            # 前進 + 輕微左轉校正
            self.publish_car_control("FORWARD_FAST_WITH_LEFT_CORRECTION")
        elif correction_action == "FORWARD_FAST_WITH_RIGHT_CORRECTION":
            # 前進 + 輕微右轉校正
            self.publish_car_control("FORWARD_FAST_WITH_RIGHT_CORRECTION")
        else:
            # 正常前進
            self.publish_car_control("FORWARD")

    # === 車頭方向追蹤系統 ===
    def start_rotation(self, direction):
        """開始旋轉並啟動方向追蹤
        Args:
            direction (str): "clockwise" 或 "counterclockwise"
        """
        self.is_rotating = True
        self.rotation_direction = direction
        self.rotation_start_time = self.clock.now()
        self.get_logger().info(f"開始{direction}旋轉，當前方向: {self.car_direction.name}")
    
    def stop_rotation(self):
        """停止旋轉並更新最終方向"""
        if self.is_rotating:
            self.update_car_direction()  # 最後更新一次
            self.is_rotating = False
            self.rotation_direction = None
            self.rotation_start_time = None
            self.get_logger().info(f"停止旋轉，最終方向: {self.car_direction.name}")
    
    def update_car_direction(self):
        """根據旋轉時間更新車頭方向"""
        if not self.is_rotating or self.rotation_start_time is None:
            return
            
        # 計算已旋轉的時間
        current_time = self.clock.now()
        elapsed_time = (current_time - self.rotation_start_time).nanoseconds / 1e9
        
        # 計算已旋轉的角度（以度為單位）
        if self.rotation_direction == "counterclockwise":
            self.rotation_angle = (elapsed_time / self.quarter_rotation_counterclockwise_time) * 90.0
        else:
            self.rotation_angle = (elapsed_time / self.quarter_rotation_clockwise_time) * 90.0
        
        direction_order = [CarDirection.FORWARD, CarDirection.LEFT, 
                         CarDirection.BACKWARD, CarDirection.RIGHT]
        current_index = direction_order.index(self.car_direction)
        
        # 計算需要切換多少次方向
        # 第一次切換在45度，之後每90度切換一次
        # 切換點：45°, 135°, 225°, 315°...
        direction_changes = 0
        if self.rotation_angle >= 45.0:
            # 第一次切換在45度
            direction_changes = 1 + int((self.rotation_angle - 45.0) / 90.0)
        
        if direction_changes > 0:
            if self.rotation_direction == "counterclockwise":
                # 逆時鐘：FORWARD -> LEFT -> BACKWARD -> RIGHT -> FORWARD
                new_index = (current_index + direction_changes) % 4
            else:  # clockwise
                # 順時鐘：FORWARD -> RIGHT -> BACKWARD -> LEFT -> FORWARD
                new_index = (current_index - direction_changes) % 4
            
            old_direction = self.car_direction
            
            self.car_direction = direction_order[new_index]
            
            if old_direction != self.car_direction:
                self.get_logger().info(f"方向更新: {old_direction.name} -> {self.car_direction.name} "
                                     f"(旋轉{self.rotation_angle:.1f}度)")
                # 重置開始時間，記錄已處理的角度
                self.rotation_start_time = current_time

    def get_current_direction(self):
        """獲取當前車頭方向"""
        if self.is_rotating:
            self.update_car_direction()
            self.get_logger().info(f"更新後方向: {self.car_direction}")
        return self.car_direction
    
    def get_current_rotation(self):
        """獲取當前車頭方向"""
        if self.is_rotating:
            self.update_car_direction()
            self.get_logger().info(f"旋轉: {self.rotation_angle:.1f}")
        return self.rotation_angle

    # === 主循環 ===
    def main_loop(self):
        """主循環"""
        # self.publish_car_control("COUNTERCLOCKWISE_ROTATION_MEDIAN")
        # self.publish_car_control("CLOCKWISE_ROTATION_MEDIAN")
        if self.state == DoorRoomState.INIT:
            # 初始化完成，開始轉向左邊
            self.turn_left_90_degrees_first()
            
        elif self.state == DoorRoomState.TURNING_LEFT_90:
            self.turn_left_90_degrees()

        elif self.state == DoorRoomState.TURNING_RIGHT_90:
            self.turn_right_90_degrees()

        elif self.state == DoorRoomState.TURNING_LEFT_180:
            self.turn_left_180_degrees()

        elif self.state == DoorRoomState.TURNING_RIGHT_180:
            self.turn_right_180_degrees()
            
        elif self.state == DoorRoomState.PASSING_THROUGH_DOOR:
            self.pass_through_door()

        elif self.state == DoorRoomState.MOVING_TO_LEFT_WALL:
            self.move_to_left_wall()  

        elif self.state == DoorRoomState.MOVING_TO_LEFT_DOOR:
            self.move_to_left_door()  

        elif self.state == DoorRoomState.MOVING_TO_RIGHT_DOOR:
            self.move_to_right_door()
            
        elif self.state == DoorRoomState.SCANNING_FOR_PIKACHU:
            if self.pikachu_detected:
                self.change_state(DoorRoomState.APPROACHING_PIKACHU)
                self.last_rgb_check_time = self.clock.now()
            else:
                self.scan_for_pikachu()
            
        elif self.state == DoorRoomState.APPROACHING_PIKACHU:
            self.approach_pikachu()
                
        elif self.state == DoorRoomState.SUCCESS:
            self.publish_car_control("STOP")
            self.get_logger().info("成功完成DoorRoom導航！")

def main(args=None):
    rclpy.init(args=args)
    
    node = DoorRoomNav()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("DoorRoom導航節點被中斷")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()