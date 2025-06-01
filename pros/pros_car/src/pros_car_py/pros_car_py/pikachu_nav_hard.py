#!/usr/bin/env python3
"""
皮卡丘Hard模式導航 - 支援ArUco + RGB
"""

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from std_msgs.msg import String, Float32MultiArray, Bool
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from pros_car_py.ros_communicator_config import ACTION_MAPPINGS
from pros_car_py.car_models import DeviceDataTypeEnum
from enum import Enum, auto
import cv2
import numpy as np
import json
import time

class FSM(Enum):
    INIT = auto()
    ROOM_DETECTION = auto()
    ARUCO_CALIBRATION = auto()  # Hard模式特有：ArUco標定
    SYSTEMATIC_SEARCH = auto()
    PIKACHU_APPROACH = auto()
    SUCCESS = auto()
    FAILED = auto()

class PikachuNavHard(Node):
    def __init__(self):
        super().__init__('pikachu_nav_hard')
        self.bridge = CvBridge()
        
        # === FSM狀態管理 ===
        self.state = FSM.INIT
        self.state_start_time = None
        
        # === 時間管理 ===
        self.clock = Clock()
        self.mission_start_time = self.clock.now()
        self.total_timeout = 240.0  # 4分鐘總超時
        
        # === Hard模式特有：ArUco檢測 ===
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.detected_markers = {}
        self.room_map_calibrated = False
        
        # === 皮卡丘檢測 ===
        self.pikachu_detected = False
        self.pikachu_position = None
        self.pikachu_last_seen = None
        
        # === 搜索策略 ===
        self.search_grid = self.create_search_grid()
        self.current_target_idx = 0
        self.visited_positions = []
        
        # === 移動控制 ===
        self.current_action = "STOP"
        self.obstacle_detected = False
        
        # === ROS通信設置 ===
        self.setup_subscribers()
        self.setup_publishers()
        
        # === 初始化 ===
        self.initialize_mission()
        
        # 主循環定時器
        self.timer = self.create_timer(0.1, self.main_loop)
        
        self.get_logger().info("🎯 皮卡丘Hard模式導航已啟動 - ArUco + RGB")

    def setup_subscribers(self):
        """設置訂閱者"""
        # YOLO檢測
        self.yolo_status_sub = self.create_subscription(
            Bool, '/yolo/detection/status', self.yolo_status_callback, 10)
        
        self.yolo_position_sub = self.create_subscription(
            PointStamped, '/yolo/detection/position', self.yolo_position_callback, 10)
        
        # RGB圖像 (用於ArUco檢測)
        self.rgb_image_sub = self.create_subscription(
            CompressedImage, '/camera/image/compressed', self.rgb_image_callback, 10)
        
        # 車輛狀態
        self.pose_sub = self.create_subscription(
            Float32MultiArray, 'digital_twin/pose_status_array',
            self.pose_status_callback, 10)

    def setup_publishers(self):
        """設置發布者"""
        # 車輛控制
        self.rear_wheel_pub = self.create_publisher(
            Float32MultiArray, DeviceDataTypeEnum.car_C_rear_wheel, 10)
        self.front_wheel_pub = self.create_publisher(
            Float32MultiArray, DeviceDataTypeEnum.car_C_front_wheel, 10)
        
        # YOLO目標設置
        self.target_label_pub = self.create_publisher(String, '/target_label', 10)
        
        # 任務狀態
        self.status_pub = self.create_publisher(String, '/pikachu_hard_status', 10)

    def initialize_mission(self):
        """初始化任務"""
        self.change_state(FSM.INIT)
        
        # 設置皮卡丘檢測
        target_msg = String()
        target_msg.data = "pikachu"
        self.target_label_pub.publish(target_msg)
        
        self.publish_status("INITIALIZED", "Hard模式初始化完成")

    def create_search_grid(self):
        """創建基於ArUco的搜索網格"""
        # Hard模式可以利用ArUco標記來建立更精確的搜索網格
        # 這裡先使用簡化的網格，後續會基於ArUco調整
        grid_points = [
            (0, 0),    # 中心
            (1, 0),    # 右
            (-1, 0),   # 左
            (0, 1),    # 前
            (0, -1),   # 後
            (1, 1),    # 右前
            (-1, 1),   # 左前
            (1, -1),   # 右後
            (-1, -1),  # 左後
        ]
        return grid_points

    def change_state(self, new_state):
        """改變FSM狀態"""
        self.state = new_state
        self.state_start_time = self.clock.now()
        self.get_logger().info(f"🔄 狀態切換至: {new_state.name}")

    def publish_car_control(self, action_key):
        """發布車輛控制指令"""
        if action_key == self.current_action:
            return
            
        if action_key not in ACTION_MAPPINGS:
            return
            
        self.current_action = action_key
        velocities = ACTION_MAPPINGS[action_key]
        vel1, vel2, vel3, vel4 = velocities
        
        # 發布控制指令
        rear_msg = Float32MultiArray()
        rear_msg.data = [vel1, vel2]
        self.rear_wheel_pub.publish(rear_msg)
        
        front_msg = Float32MultiArray()
        front_msg.data = [vel3, vel4]
        self.front_wheel_pub.publish(front_msg)

    def publish_status(self, status, message=""):
        """發布任務狀態"""
        elapsed = (self.clock.now() - self.mission_start_time).nanoseconds / 1e9
        status_data = {
            "status": status,
            "state": self.state.name,
            "message": message,
            "elapsed_time": elapsed,
            "aruco_markers": len(self.detected_markers),
            "pikachu_detected": self.pikachu_detected
        }
        
        status_msg = String()
        status_msg.data = json.dumps(status_data)
        self.status_pub.publish(status_msg)

    # === 回調函數 ===
    def yolo_status_callback(self, msg):
        """YOLO檢測狀態回調"""
        self.pikachu_detected = msg.data
        if self.pikachu_detected:
            self.pikachu_last_seen = self.clock.now()

    def yolo_position_callback(self, msg):
        """YOLO位置回調"""
        self.pikachu_position = msg

    def rgb_image_callback(self, msg):
        """RGB圖像回調 - 進行ArUco檢測"""
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            self.detect_aruco_markers(cv_image)
        except Exception as e:
            self.get_logger().error(f"圖像處理錯誤: {e}")

    def pose_status_callback(self, msg):
        """位置狀態回調"""
        if len(msg.data) >= 5:
            self.x = msg.data[0]
            self.y = msg.data[1]
            self.yaw = msg.data[2]
            self.obstacle_detected = bool(msg.data[4])

    # === ArUco檢測和定位 ===
    def detect_aruco_markers(self, image):
        """檢測ArUco標記"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                # 儲存檢測到的標記信息
                corner = corners[i][0]
                center = np.mean(corner, axis=0)
                
                self.detected_markers[marker_id] = {
                    'center': center,
                    'corners': corner,
                    'timestamp': self.clock.now().nanoseconds / 1e9
                }
                
            self.get_logger().info(f"🎯 檢測到ArUco標記: {list(ids.flatten())}")
            
            # 如果檢測到足夠的標記，進行房間標定
            if len(self.detected_markers) >= 2 and not self.room_map_calibrated:
                self.calibrate_room_map()

    def calibrate_room_map(self):
        """基於ArUco標記標定房間地圖"""
        if len(self.detected_markers) < 2:
            return
            
        # 簡化的房間標定邏輯
        # 在真實應用中，這裡會建立更精確的座標系統
        marker_positions = {}
        for marker_id, marker_info in self.detected_markers.items():
            marker_positions[marker_id] = marker_info['center']
        
        self.room_map_calibrated = True
        self.update_search_grid_with_aruco()
        
        self.get_logger().info("🗺️  基於ArUco完成房間地圖標定")

    def update_search_grid_with_aruco(self):
        """基於ArUco標記更新搜索網格"""
        # 利用ArUco標記位置優化搜索路徑
        # 這裡實現簡化版本，真實情況下會更複雜
        
        if len(self.detected_markers) >= 2:
            # 基於標記位置計算房間中心和邊界
            marker_centers = [info['center'] for info in self.detected_markers.values()]
            
            # 重新計算搜索網格
            self.search_grid = self.calculate_optimized_grid(marker_centers)
            self.get_logger().info("📍 基於ArUco優化搜索網格")

    def calculate_optimized_grid(self, marker_centers):
        """基於ArUco標記計算優化的搜索網格"""
        # 簡化實現：在標記之間和周圍創建搜索點
        optimized_grid = []
        
        # 添加標記附近的搜索點
        for center in marker_centers:
            # 在每個標記周圍添加搜索點
            offsets = [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]
            for offset_x, offset_y in offsets:
                optimized_grid.append((center[0] + offset_x, center[1] + offset_y))
        
        # 添加中心點
        if len(marker_centers) >= 2:
            center_x = sum(c[0] for c in marker_centers) / len(marker_centers)
            center_y = sum(c[1] for c in marker_centers) / len(marker_centers)
            optimized_grid.append((center_x, center_y))
        
        return optimized_grid

    # === 搜索邏輯 ===
    def execute_aruco_guided_search(self):
        """執行ArUco引導的搜索"""
        if not self.room_map_calibrated:
            # 如果還沒有標定，先進行標定搜索
            self.execute_calibration_search()
        else:
            # 已標定，執行優化搜索
            self.execute_optimized_search()

    def execute_calibration_search(self):
        """執行標定搜索"""
        # 緩慢旋轉以檢測更多ArUco標記
        state_time = (self.clock.now() - self.state_start_time).nanoseconds / 1e9
        
        if state_time < 10.0:  # 前10秒旋轉檢測ArUco
            self.publish_car_control("CLOCKWISE_ROTATION_SLOW")
        elif state_time < 15.0:  # 然後前進一段距離
            self.publish_car_control("FORWARD_SLOW")
        else:
            # 如果還沒有足夠的ArUco標記，繼續搜索
            if len(self.detected_markers) < 2:
                self.change_state(FSM.SYSTEMATIC_SEARCH)
            else:
                self.calibrate_room_map()

    def execute_optimized_search(self):
        """執行優化搜索"""
        if self.current_target_idx >= len(self.search_grid):
            # 重新開始搜索
            self.current_target_idx = 0
            
        # 獲取當前目標點
        target_point = self.search_grid[self.current_target_idx]
        
        # 簡化的導航邏輯：向目標點移動
        if self.navigate_to_point(target_point):
            # 到達目標點，移動到下一個
            self.current_target_idx += 1
            self.visited_positions.append(target_point)

    def navigate_to_point(self, target_point):
        """導航到指定點"""
        # 簡化的導航實現
        # 在真實應用中，這裡會使用更複雜的路徑規劃
        
        if self.obstacle_detected:
            self.publish_car_control("CLOCKWISE_ROTATION")
            return False
        else:
            # 簡單前進
            self.publish_car_control("FORWARD_SLOW")
            return True  # 簡化：假設總是能到達

    def approach_pikachu(self):
        """接近皮卡丘"""
        if not self.pikachu_position:
            self.change_state(FSM.SYSTEMATIC_SEARCH)
            return
            
        x = self.pikachu_position.point.x
        y = self.pikachu_position.point.y
        z = self.pikachu_position.point.z
        
        distance = np.sqrt(x*x + y*y + z*z)
        
        if distance < 0.5:  # 50cm內認為成功
            self.change_state(FSM.SUCCESS)
            return
        
        # 計算接近動作
        if abs(y) > 0.3:  # 需要對準
            if y > 0:
                self.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
            else:
                self.publish_car_control("CLOCKWISE_ROTATION_SLOW")
        elif x > 1.0:  # 距離較遠
            self.publish_car_control("FORWARD")
        elif x > 0.5:  # 距離適中
            self.publish_car_control("FORWARD_SLOW")
        else:
            self.publish_car_control("STOP")

    # === 主循環 ===
    def main_loop(self):
        """主FSM循環"""
        current_time = self.clock.now()
        total_elapsed = (current_time - self.mission_start_time).nanoseconds / 1e9
        
        # 全局超時檢查
        if total_elapsed > self.total_timeout:
            self.change_state(FSM.FAILED)
            return
        
        # 執行當前狀態邏輯
        if self.state == FSM.INIT:
            self.change_state(FSM.ROOM_DETECTION)
            
        elif self.state == FSM.ROOM_DETECTION:
            # 簡單的房間檢測
            state_time = (current_time - self.state_start_time).nanoseconds / 1e9
            if state_time < 3.0:
                self.publish_car_control("FORWARD_SLOW")
            else:
                self.change_state(FSM.ARUCO_CALIBRATION)
                
        elif self.state == FSM.ARUCO_CALIBRATION:
            if self.pikachu_detected:
                self.change_state(FSM.PIKACHU_APPROACH)
            else:
                self.execute_aruco_guided_search()
                
        elif self.state == FSM.SYSTEMATIC_SEARCH:
            if self.pikachu_detected:
                self.change_state(FSM.PIKACHU_APPROACH)
            else:
                if self.room_map_calibrated:
                    self.execute_optimized_search()
                else:
                    # 基本搜索
                    cycle_time = (current_time - self.state_start_time).nanoseconds / 1e9
                    cycle = int(cycle_time) % 4
                    actions = ["FORWARD", "CLOCKWISE_ROTATION", "FORWARD", "COUNTERCLOCKWISE_ROTATION"]
                    self.publish_car_control(actions[cycle])
                    
        elif self.state == FSM.PIKACHU_APPROACH:
            if self.pikachu_detected:
                self.approach_pikachu()
            else:
                self.change_state(FSM.SYSTEMATIC_SEARCH)
                
        elif self.state == FSM.SUCCESS:
            self.publish_car_control("STOP")
            self.publish_status("SUCCESS", f"🎉 成功找到皮卡丘！用時 {total_elapsed:.1f}秒")
            
        elif self.state == FSM.FAILED:
            self.publish_car_control("STOP")
            self.publish_status("FAILED", f"❌ 任務失敗，超時 {total_elapsed:.1f}秒")

def main(args=None):
    rclpy.init(args=args)
    
    node = PikachuNavHard()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Hard模式節點被中斷")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()