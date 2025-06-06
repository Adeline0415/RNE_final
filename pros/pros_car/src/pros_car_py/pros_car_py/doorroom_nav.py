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
        self.quarter_rotation_time = 8.5 #220/3 = 46.67 => 46.67/4 = 11.67

        # === 門的位置管理 ===
        self.doors_passed = 0  # 已通過的門數量
        self.door_distance_move_time = 6.0  # 移動到下一個門位置的時間

        # === 門位置追蹤系統 ===
        self.door_bitmap = [0, 0, 0, 0]  # 確保最先初始化
        self.cur_door = 1.5
        self.last_door = -1
        self.door_positions = [0, 1, 2, 3]
        self.single_door_move_time = 6.0
        
        # === 牆壁檢測 ===
        self.wall_color_range = None  # 將在運行時設定
        self.wall_reached = False  # 是否已到達牆壁
        self.extra_move_duration = 4.0  # 到達牆壁後額外移動時間
        
        # === 皮卡丘檢測 ===
        self.pikachu_detected = False
        self.delta_x = 0.0  # 像素偏移（左負右正）
        self.pikachu_total_area = 0.0
        
        # === 移動計時 ===
        self.move_start_time = None
        self.scan_start_time = None
        self.turn_duration = 12.0  #timeout

        # === 顏色檢測系統 ===
        self.color_ranges = {
            'blue': {
                'target_bgr': [154, 140, 94],
                'lower': np.array([144, 130, 84]),
                'upper': np.array([190, 176, 130])
            },
            'gray': {
                'target_bgr': [100, 100, 100], 
                'lower': np.array([80, 80, 80]),
                'upper': np.array([120, 120, 120])
            },
            'brown': {
                'target_bgr': [90, 133, 190], 
                'lower': np.array([80, 123, 180]),
                'upper': np.array([110, 163, 220])
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
            blue_ratio = self.check_color_ratio("blue")
            gray_ratio = self.check_color_ratio("gray")
            brown_ratio = self.check_color_ratio("brown")
            self.get_logger().info(f"顏色比例 - 藍色: {blue_ratio:.3f}, 灰色: {gray_ratio:.3f}, 棕色: {brown_ratio:.3f}")

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
        
        if elapsed < self.turn_duration:
            self.publish_car_control("COUNTERCLOCKWISE_ROTATION_MEDIAN")
            
            # 檢查是否看到水平線（表示已轉90度）
            if self.detect_horizontal_line() and (self.check_color_ratio("blue") > 0.4) and (self.get_current_direction() == CarDirection.LEFT):
                self.get_logger().info("檢測到水平線，已轉90度朝左")
                self.car_direction = CarDirection.LEFT
                self.move_start_time = None
                self.change_state(DoorRoomState.MOVING_TO_LEFT_WALL)
        else:
            # 超時也認為已轉到位
            self.get_logger().info("轉向超時，假設已朝左")
            self.car_direction = CarDirection.LEFT
            self.move_start_time = None
            self.change_state(DoorRoomState.MOVING_TO_LEFT_WALL)

    def turn_left_90_degrees(self):
        """逆時鐘轉90度朝左"""
        if self.move_start_time is None:
            self.move_start_time = self.clock.now()
            self.get_logger().info("開始逆時鐘轉90度朝左")
        
        elapsed = (self.clock.now() - self.move_start_time).nanoseconds / 1e9
        
        if elapsed < self.turn_duration:
            self.publish_car_control("COUNTERCLOCKWISE_ROTATION_MEDIAN")
            
            # 檢查是否看到水平線（表示已轉90度）
            if self.detect_horizontal_line() and (self.check_color_ratio("blue") > 0.15) and (self.get_current_direction() == CarDirection.LEFT):
                self.get_logger().info("檢測到水平線，已轉90度朝左")
                self.car_direction = CarDirection.LEFT
                self.move_start_time = None
                self.change_state(DoorRoomState.MOVING_TO_LEFT_DOOR)
        else:
            # 超時也認為已轉到位
            self.get_logger().info("轉向超時，假設已朝左")
            self.car_direction = CarDirection.LEFT
            self.move_start_time = None
            self.change_state(DoorRoomState.MOVING_TO_LEFT_DOOR)

    def turn_right_90_degrees(self):
        """順時鐘轉90度朝右"""
        if self.move_start_time is None:
            self.move_start_time = self.clock.now()
            self.get_logger().info("開始順時鐘轉90度朝右")
        
        elapsed = (self.clock.now() - self.move_start_time).nanoseconds / 1e9
        
        if elapsed < self.turn_duration:
            self.publish_car_control("CLOCKWISE_ROTATION_MEDIAN")
            
            # 檢查是否看到水平線且轉到右邊
            if self.detect_horizontal_line() and (self.get_current_direction() == CarDirection.RIGHT):
                self.get_logger().info("檢測到水平線，已轉90度朝右")
                self.car_direction = CarDirection.RIGHT
                self.move_start_time = None
                self.change_state(DoorRoomState.MOVING_TO_RIGHT_DOOR)
        else:
            self.get_logger().info("轉向超時，假設已朝右")
            self.car_direction = CarDirection.RIGHT
            self.move_start_time = None
            self.change_state(DoorRoomState.MOVING_TO_RIGHT_DOOR)

    def move_to_left_wall(self):
        """往前移動直到到達左邊牆壁（畫面都是淺藍色），然後再直走1.5秒"""
        # 印出RGB資訊用於調試
        self.print_rgb_info()
        
        # 檢查是否到達牆壁
        blue_ratio = self.check_color_ratio("blue")
        
        if not self.wall_reached:
            # 還沒到達牆壁，繼續前進
            self.publish_car_control("FORWARD")
            
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
        
        if elapsed < self.turn_duration*2:
            self.publish_car_control("CLOCKWISE_ROTATION_MEDIAN")
            # 在轉向過程中檢測門
            if self.detect_horizontal_line() and (self.check_color_ratio("brown") > 0.15) and (self.get_current_direction() == CarDirection.FORWARD):
                self.get_logger().info("轉向過程中檢測到門的水平線！")
                self.move_start_time = None
                self.change_state(DoorRoomState.PASSING_THROUGH_DOOR)
                return
            
            if self.detect_horizontal_line() and (self.doors_passed == 2) and (self.get_current_direction() == CarDirection.FORWARD): ## 最後一層
                self.get_logger().info("轉向過程中檢測到門的水平線！")
                self.move_start_time = None
                self.change_state(DoorRoomState.PASSING_THROUGH_DOOR)
                return
            
            if self.detect_horizontal_line() and (self.check_color_ratio("blue") < 0.9) and (self.get_current_direction() == CarDirection.RIGHT):
                self.get_logger().info("180度轉向完成 檢測到牆壁水平線，往前移動到下一個門位置")
                self.car_direction = CarDirection.RIGHT
                self.move_start_time = None
                self.change_state(DoorRoomState.MOVING_TO_RIGHT_DOOR)
        else:
            self.get_logger().info("轉向完成但未檢測到門，往前移動到下一個門位置")
            self.car_direction = CarDirection.RIGHT
            self.move_start_time = None
            self.change_state(DoorRoomState.MOVING_TO_RIGHT_DOOR)

    def turn_left_180_degrees(self):
        """轉向面對左邊準備掃描，轉向過程中就檢測門"""
        if self.move_start_time is None:
            self.move_start_time = self.clock.now()
            self.get_logger().info("開始轉向面對左邊（轉向過程中檢測門）")
        
        elapsed = (self.clock.now() - self.move_start_time).nanoseconds / 1e9
        
        if elapsed < self.turn_duration*2:
            self.publish_car_control("COUNTERCLOCKWISE_ROTATION_MEDIAN")
            # 在轉向過程中檢測門
            if self.detect_horizontal_line() and (self.check_color_ratio("brown") > 0.15) and (self.get_current_direction() == CarDirection.FORWARD):
                self.get_logger().info("轉向過程中檢測到門的水平線！")
                self.move_start_time = None
                self.change_state(DoorRoomState.PASSING_THROUGH_DOOR)
                return
            
            if self.detect_horizontal_line() and (self.doors_passed == 2) and (self.get_current_direction() == CarDirection.FORWARD):
                self.get_logger().info("轉向過程中檢測到門的水平線！")
                self.move_start_time = None
                self.change_state(DoorRoomState.PASSING_THROUGH_DOOR)
                return
            
            if self.detect_horizontal_line() and (self.check_color_ratio("blue") > 0.15) and (self.check_color_ratio("blue") < 0.9) and (self.get_current_direction() == CarDirection.LEFT):
                self.get_logger().info("180度轉向完成 檢測到牆壁水平線，轉180回左邊")
                self.car_direction = CarDirection.LEFT
                self.move_start_time = None
                self.change_state(DoorRoomState.TURNING_RIGHT_180)
        else:
            self.get_logger().info("轉向完成但未檢測到門，往前移動到下一個門位置")
            self.car_direction = CarDirection.LEFT
            self.move_start_time = None
            self.change_state(DoorRoomState.TURNING_RIGHT_180)

    def pass_through_door(self):
        """通過門"""
        # 檢查是否看不到灰色
        gray_ratio = self.check_color_ratio("gray")
        
        if self.move_start_time is None:
            # 還沒開始計時，繼續前進直到看不到灰色
            self.publish_car_control("FORWARD")
            
            if gray_ratio == 0:
                # 看不到灰色了，開始計時
                self.move_start_time = self.clock.now()
                self.get_logger().info("已看不到灰色，開始計時通過門")
        else:
            # 已經開始計時，檢查是否達到2秒
            elapsed = (self.clock.now() - self.move_start_time).nanoseconds / 1e9
            
            if elapsed < 1.0:  # 看不到灰色後再走2秒
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
            self.publish_car_control("FORWARD")
        else:
            self.move_start_time = None
            self.change_state(DoorRoomState.TURNING_LEFT_180)

    def move_to_left_door(self):
        """移動到最左邊未訪問的門位置"""
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
            self.publish_car_control("FORWARD")
        else:
            self.move_start_time = None
            self.change_state(DoorRoomState.TURNING_RIGHT_180)

    def scan_for_pikachu(self):
        """掃描皮卡丘"""
        if self.pikachu_detected:
            self.change_state(DoorRoomState.APPROACHING_PIKACHU)
            return
        
        # 原地旋轉尋找皮卡丘
        self.publish_car_control("COUNTERCLOCKWISE_ROTATION")
        self.get_logger().info("掃描皮卡丘中...")

    def approach_pikachu(self):
        """接近皮卡丘 - 當面積達到閾值時進入最終階段"""
        # 如果已經在最終接近階段，直接執行不受檢測影響
        if self.final_approach_start_time is not None:
            elapsed = (self.clock.now() - self.final_approach_start_time).nanoseconds / 1e9
            if elapsed < self.final_approach_duration:
                self.publish_car_control("FORWARD")
                remaining_time = self.final_approach_duration - elapsed
                self.get_logger().info(f"最終接近中...剩餘{remaining_time:.1f}秒")
            else:
                self.change_state(DoorRoomState.SUCCESS)
            return  

        if not self.pikachu_detected:
            self.final_approach_start_time = self.clock.now()
            return

        # 面積未達標，繼續對準和接近（原有邏輯）
        alignment_threshold = 50
        
        if abs(self.delta_x) > alignment_threshold:
            if self.delta_x < 0:
                self.publish_car_control("COUNTERCLOCKWISE_ROTATION")
            else:
                self.publish_car_control("CLOCKWISE_ROTATION")
        else:
            self.publish_car_control("FORWARD")
            self.get_logger().info(f"前進接近 (面積: {self.pikachu_total_area:.0f}px²)")

    def detect_horizontal_line(self):
        """檢測水平線"""
        if self.current_rgb_image is None:
            return False
        
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
                    if length < 50:  # 最小長度閾值
                        continue
                        
                    # 計算斜率角度
                    if x2 - x1 != 0:
                        slope = (y2 - y1) / (x2 - x1)
                        angle = abs(math.atan(slope) * 180 / math.pi)
                    else:
                        angle = 90  # 垂直線
                    
                    # 水平線角度應該接近0度
                    if angle <= 0.5:  # 允許1度誤差
                        self.get_logger().info(f"檢測到水平線: ({x1},{y1}) -> ({x2},{y2}), 角度: {angle:.1f}°")
                        return True
            
            return False
            
        except Exception as e:
            self.get_logger().error(f"水平線檢測錯誤: {e}")
            return False

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
        self.rotation_angle = (elapsed_time / self.quarter_rotation_time) * 90.0
        
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

        if self.state == DoorRoomState.INIT:
            # 初始化完成，開始轉向左邊
            self.turn_left_90_degrees_first()

        elif self.pikachu_detected:
            self.change_state(DoorRoomState.APPROACHING_PIKACHU)
            
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