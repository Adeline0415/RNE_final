#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from std_msgs.msg import String, Float32MultiArray, Bool
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from pros_car_py.ros_communicator_config import ACTION_MAPPINGS
from pros_car_py.car_models import DeviceDataTypeEnum
from enum import Enum, auto
import cv2
import numpy as np
import time

class FSM(Enum):
    ROOM_DETECT = auto()      # 檢測房間類型
    INIT_SEARCH = auto()      # 初始化搜索
    SEARCH_SWEEP = auto()     # 左右掃描搜索
    PIKACHU_DETECTED = auto() # 檢測到皮卡丘
    APPROACH = auto()         # 接近皮卡丘
    SUCCESS = auto()          # 成功找到
    TIMEOUT = auto()          # 超時失敗

class PikachuSeekerLivingRoom(Node):
    def __init__(self):
        super().__init__('pikachu_seeker_living_room')
        self.bridge = CvBridge()
        
        # FSM狀態管理
        self.state = FSM.ROOM_DETECT
        self.search_direction = 1  # 1: 右轉, -1: 左轉
        self.sweep_count = 0
        self.max_sweeps = 6  # 最多掃描6次
        
        # 時間管理
        self.clock = Clock()
        self.start_time = None
        self.state_start_time = None
        self.search_timeout = 180.0  # 3分鐘搜索超時
        
        # 皮卡丘檢測狀態
        self.pikachu_detected = False
        self.pikachu_position = None
        self.detection_confidence = 0.0
        self.lost_target_time = None
        
        # 車輛位置和狀態 (如果有提供)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.collision = False
        
        # 房間檢測
        self.room_confirmed = False
        self.room_type = "unknown"
        
        # === ROS訂閱者 ===
        # 車輛位置狀態 (如果有的話)
        self.pose_sub = self.create_subscription(
            Float32MultiArray,
            'digital_twin/pose_status_array',
            self.pose_status_callback,
            10
        )
        
        # YOLO檢測狀態
        self.yolo_status_sub = self.create_subscription(
            Bool,
            '/yolo/detection/status',
            self.yolo_status_callback,
            10
        )
        
        # YOLO檢測位置
        self.yolo_position_sub = self.create_subscription(
            PointStamped,
            '/yolo/detection/position',
            self.yolo_position_callback,
            10
        )
        
        # RGB圖像 (用於房間檢測和ArUco)
        self.rgb_image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.rgb_image_callback,
            10
        )
        
        # === ROS發布者 ===
        # 車輛控制
        self.rear_wheel_pub = self.create_publisher(
            Float32MultiArray,
            DeviceDataTypeEnum.car_C_rear_wheel,
            10
        )
        self.forward_wheel_pub = self.create_publisher(
            Float32MultiArray,
            DeviceDataTypeEnum.car_C_front_wheel,
            10
        )
        
        # 設定YOLO檢測目標
        self.target_label_pub = self.create_publisher(
            String,
            '/target_label',
            10
        )
        
        # 任務狀態發布
        self.task_status_pub = self.create_publisher(
            String,
            '/pikachu_task_status',
            10
        )
        
        # 動作指令發布 (調試用)
        self.action_key_pub = self.create_publisher(
            String,
            "/action_key",
            10
        )
        
        # 初始化
        self.start_time = self.clock.now()
        self.state_start_time = self.start_time
        
        # 設定皮卡丘為檢測目標
        self.set_pikachu_target()
        
        self.get_logger().info("🎯 皮卡丘尋找節點已啟動 - Living Room模式")
        self.get_logger().info("📋 支援模式: Hard (ArUco + RGB) / Hell (RGB only)")

    def set_pikachu_target(self):
        """設定皮卡丘為YOLO檢測目標"""
        target_msg = String()
        target_msg.data = "pikachu"  # 或根據你的YOLO模型調整
        self.target_label_pub.publish(target_msg)
        self.get_logger().info("🎯 已設定皮卡丘為檢測目標")

    def publish_car_control(self, action_key, publish_rear=True, publish_front=True):
        """發布車輛控制指令"""
        if action_key not in ACTION_MAPPINGS:
            self.get_logger().warn(f"未知動作指令: {action_key}")
            return
            
        velocities = ACTION_MAPPINGS[action_key]
        vel1, vel2, vel3, vel4 = velocities
        
        # 發布後輪控制
        if publish_rear:
            rear_msg = Float32MultiArray()
            rear_msg.data = [vel1, vel2]
            self.rear_wheel_pub.publish(rear_msg)
            
        # 發布前輪控制
        if publish_front:
            front_msg = Float32MultiArray()
            front_msg.data = [vel3, vel4]
            self.forward_wheel_pub.publish(front_msg)
            
        # 發布動作指令 (調試用)
        action_msg = String()
        action_msg.data = action_key
        self.action_key_pub.publish(action_msg)

    def publish_task_status(self, status, message=""):
        """發布任務狀態"""
        status_msg = String()
        status_data = {
            "status": status,
            "state": self.state.name,
            "message": message,
            "time_elapsed": (self.clock.now() - self.start_time).nanoseconds / 1e9
        }
        status_msg.data = str(status_data)
        self.task_status_pub.publish(status_msg)

    def pose_status_callback(self, msg: Float32MultiArray):
        """車輛位置狀態回調"""
        if len(msg.data) >= 5:
            self.x = msg.data[0]
            self.y = msg.data[1] 
            self.yaw = msg.data[2]
            # road_ahead = bool(msg.data[3])
            self.collision = bool(msg.data[4])

    def yolo_status_callback(self, msg: Bool):
        """YOLO檢測狀態回調"""
        self.pikachu_detected = msg.data
        if not self.pikachu_detected:
            self.lost_target_time = self.clock.now()

    def yolo_position_callback(self, msg: PointStamped):
        """YOLO檢測位置回調"""
        self.pikachu_position = msg
        if self.pikachu_detected:
            self.lost_target_time = None

    def rgb_image_callback(self, msg: CompressedImage):
        """RGB圖像回調 - 用於房間檢測和ArUco檢測"""
        if not self.room_confirmed:
            self.detect_room_type(msg)

    def detect_room_type(self, image_msg: CompressedImage):
        """檢測房間類型"""
        try:
            # 解壓縮圖像
            cv_image = self.bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")
            
            # 簡單的Living Room檢測邏輯
            # 可以基於顏色直方圖、特定物體檢測等
            room_type = self.analyze_living_room_features(cv_image)
            
            if room_type == "living_room":
                self.room_type = "living_room"
                self.room_confirmed = True
                self.get_logger().info("🏠 確認當前房間: Living Room")
                
        except Exception as e:
            self.get_logger().error(f"房間檢測錯誤: {e}")

    def analyze_living_room_features(self, image):
        """分析Living Room特徵"""
        # 簡化的房間檢測邏輯
        # 實際使用時可以檢測沙發、電視、桌子等特徵
        
        # 轉換為HSV進行顏色分析
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 檢測棕色/木色 (客廳常見顏色)
        brown_lower = np.array([10, 50, 50])
        brown_upper = np.array([20, 255, 200])
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        brown_ratio = np.sum(brown_mask > 0) / (image.shape[0] * image.shape[1])
        
        # 如果棕色比例超過閾值，認為是客廳
        if brown_ratio > 0.1:  # 10%以上棕色
            return "living_room"
        else:
            return "unknown"

    def calculate_distance_to_pikachu(self):
        """計算到皮卡丘的距離"""
        if not self.pikachu_position:
            return float('inf')
            
        x = self.pikachu_position.point.x
        y = self.pikachu_position.point.y
        z = self.pikachu_position.point.z
        
        return np.sqrt(x*x + y*y + z*z)

    def calculate_approach_action(self):
        """計算接近皮卡丘的動作"""
        if not self.pikachu_position:
            return "STOP"
            
        x = self.pikachu_position.point.x
        y = self.pikachu_position.point.y
        
        # 基於目標位置決定動作
        if abs(y) > 0.3:  # 需要轉向對準
            if y > 0:
                return "COUNTERCLOCKWISE_ROTATION_SLOW"
            else:
                return "CLOCKWISE_ROTATION_SLOW"
        elif x > 1.0:  # 距離較遠，前進
            return "FORWARD"
        elif x > 0.5:  # 距離適中，慢速前進
            return "FORWARD_SLOW"
        else:
            return "STOP"  # 足夠接近

    def fsm_update(self):
        """FSM狀態更新"""
        current_time = self.clock.now()
        elapsed_time = (current_time - self.start_time).nanoseconds / 1e9
        state_time = (current_time - self.state_start_time).nanoseconds / 1e9
        
        # 檢查全局超時
        if elapsed_time > self.search_timeout:
            self.state = FSM.TIMEOUT
            
        if self.state == FSM.ROOM_DETECT:
            if self.room_confirmed and self.room_type == "living_room":
                self.state = FSM.INIT_SEARCH
                self.state_start_time = current_time
                self.get_logger().info("🔍 開始初始化搜索")
            else:
                # 繼續前進直到確認房間
                self.publish_car_control("FORWARD_SLOW")
                
        elif self.state == FSM.INIT_SEARCH:
            # 初始化階段，稍微前進然後開始掃描
            if state_time < 2.0:
                self.publish_car_control("FORWARD_SLOW")
            else:
                self.state = FSM.SEARCH_SWEEP
                self.state_start_time = current_time
                self.get_logger().info("🔄 開始左右掃描搜索")
                
        elif self.state == FSM.SEARCH_SWEEP:
            if self.pikachu_detected:
                self.state = FSM.PIKACHU_DETECTED
                self.state_start_time = current_time
                self.get_logger().info("🎯 檢測到皮卡丘！")
            else:
                # 左右掃描搜索
                self.execute_sweep_search(state_time)
                
        elif self.state == FSM.PIKACHU_DETECTED:
            if self.pikachu_detected:
                self.state = FSM.APPROACH
                self.state_start_time = current_time
                self.get_logger().info("🚗 開始接近皮卡丘")
            elif state_time > 3.0:  # 3秒後沒有檢測到，回到搜索
                self.state = FSM.SEARCH_SWEEP
                self.state_start_time = current_time
                self.get_logger().info("❌ 丟失目標，回到搜索模式")
                
        elif self.state == FSM.APPROACH:
            if self.pikachu_detected:
                distance = self.calculate_distance_to_pikachu()
                if distance < 0.5:  # 50cm內認為成功
                    self.state = FSM.SUCCESS
                    self.get_logger().info("🎉 成功找到皮卡丘！")
                else:
                    action = self.calculate_approach_action()
                    self.publish_car_control(action)
            else:
                # 丟失目標，回到搜索
                if state_time > 5.0:
                    self.state = FSM.SEARCH_SWEEP
                    self.state_start_time = current_time
                    self.get_logger().info("❌ 接近過程中丟失目標")
                else:
                    self.publish_car_control("STOP")
                    
        elif self.state == FSM.SUCCESS:
            self.publish_car_control("STOP")
            self.publish_task_status("SUCCESS", "成功找到皮卡丘")
            if state_time > 5.0:  # 停止5秒後結束
                self.get_logger().info("✅ 任務完成，節點將保持運行")
                
        elif self.state == FSM.TIMEOUT:
            self.publish_car_control("STOP")
            self.publish_task_status("TIMEOUT", "搜索超時")
            self.get_logger().info("⏰ 搜索超時，任務失敗")

    def execute_sweep_search(self, state_time):
        """執行左右掃描搜索"""
        sweep_duration = 8.0  # 每個方向掃描8秒
        forward_duration = 3.0  # 前進3秒
        
        cycle_time = state_time % (sweep_duration * 2 + forward_duration)
        
        if cycle_time < sweep_duration:
            # 第一個方向掃描
            if self.search_direction == 1:
                self.publish_car_control("CLOCKWISE_ROTATION_SLOW")
            else:
                self.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
        elif cycle_time < sweep_duration * 2:
            # 第二個方向掃描
            if self.search_direction == 1:
                self.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
            else:
                self.publish_car_control("CLOCKWISE_ROTATION_SLOW")
        else:
            # 前進一段距離
            self.publish_car_control("FORWARD")
            
        # 檢查是否完成一個完整周期
        if cycle_time < 1.0 and state_time > sweep_duration * 2 + forward_duration:
            self.sweep_count += 1
            if self.sweep_count >= self.max_sweeps:
                self.state = FSM.TIMEOUT
                self.get_logger().info(f"🔍 完成{self.max_sweeps}次掃描，未找到皮卡丘")

    def timer_callback(self):
        """定時器回調，執行FSM"""
        self.fsm_update()

def main(args=None):
    rclpy.init(args=args)
    
    node = PikachuSeekerLivingRoom()
    
    # 創建定時器來執行FSM
    timer = node.create_timer(0.1, node.timer_callback)  # 10Hz
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 節點被用戶中斷")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()