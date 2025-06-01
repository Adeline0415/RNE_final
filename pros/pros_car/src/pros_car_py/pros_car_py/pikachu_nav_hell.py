#!/usr/bin/env python3
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
    INIT = auto()             # 初始化
    ROOM_DETECTION = auto()   # 房間檢測
    FURNITURE_MAPPING = auto() # 家具地圖構建
    SYSTEMATIC_SEARCH = auto() # 系統性搜索
    PIKACHU_APPROACH = auto() # 接近皮卡丘
    SUCCESS = auto()          # 成功
    EMERGENCY_SEARCH = auto() # 緊急搜索
    FAILED = auto()           # 失敗

class PikachuNavHell(Node):
    def __init__(self):
        super().__init__('pikachu_seeker_hell')
        self.bridge = CvBridge()
        
        # === FSM狀態管理 ===
        self.state = FSM.INIT
        self.prev_state = None
        self.state_start_time = None
        
        # === 時間管理 ===
        self.clock = Clock()
        self.mission_start_time = self.clock.now()
        self.total_timeout = 300.0  # 5分鐘總超時
        
        # === 皮卡丘檢測 ===
        self.pikachu_detected = False
        self.pikachu_position = None
        self.pikachu_last_seen = None
        self.pikachu_confidence = 0.0
        
        # === 家具檢測與地圖 ===
        self.furniture_map = {}  # {家具類型: [位置列表]}
        self.visited_areas = []  # 已訪問區域
        self.current_scan_direction = 1  # 1: 右, -1: 左
        
        # === 房間檢測 ===
        self.room_type = "unknown"
        self.room_confirmed = False
        self.room_detection_samples = []
        
        # === 搜索策略 ===
        self.search_pattern = "spiral"  # spiral, grid, random
        self.search_phase = 0
        self.phase_duration = 15.0  # 每個搜索階段15秒
        
        # === 移動控制 ===
        self.current_action = "STOP"
        self.action_start_time = None
        self.min_action_duration = 0.5  # 最小動作持續時間
        
        # === 障礙物檢測 ===
        self.obstacle_detected = False
        self.last_safe_direction = "FORWARD"
        
        # === ROS訂閱者 ===
        self.setup_subscribers()
        
        # === ROS發布者 ===
        self.setup_publishers()
        
        # === 初始化 ===
        self.initialize_mission()
        
        # 創建主循環定時器
        self.timer = self.create_timer(0.1, self.main_loop)  # 10Hz
        
        self.get_logger().info("🔥 皮卡丘Hell模式搜索已啟動 - 純RGB導航")

    def setup_subscribers(self):
        """設置訂閱者"""
        # YOLO檢測結果
        self.yolo_status_sub = self.create_subscription(
            Bool, '/yolo/detection/status', self.yolo_status_callback, 10)
        
        self.yolo_position_sub = self.create_subscription(
            PointStamped, '/yolo/detection/position', self.yolo_position_callback, 10)
        
        # 多目標檢測結果 (如果你的YOLO能檢測多種物體)
        self.object_offset_sub = self.create_subscription(
            String, '/yolo/object/offset', self.object_offset_callback, 10)
        
        # RGB圖像
        self.rgb_image_sub = self.create_subscription(
            CompressedImage, '/camera/image/compressed', self.rgb_image_callback, 10)
        
        # 車輛狀態 (如果有)
        self.pose_sub = self.create_subscription(
            Float32MultiArray, 'digital_twin/pose_status_array', 
            self.pose_status_callback, 10)
        
        # 深度信息 (用於障礙物檢測)
        self.depth_info_sub = self.create_subscription(
            Float32MultiArray, '/camera/x_multi_depth_values',
            self.depth_info_callback, 10)

    def setup_publishers(self):
        """設置發布者"""
        # 車輛控制
        self.rear_wheel_pub = self.create_publisher(
            Float32MultiArray, DeviceDataTypeEnum.car_C_rear_wheel, 10)
        self.front_wheel_pub = self.create_publisher(
            Float32MultiArray, DeviceDataTypeEnum.car_C_front_wheel, 10)
        
        # YOLO目標設置
        self.target_label_pub = self.create_publisher(
            String, '/target_label', 10)
        
        # 任務狀態
        self.status_pub = self.create_publisher(
            String, '/pikachu_hell_status', 10)
        
        # 調試信息
        self.debug_pub = self.create_publisher(
            String, '/pikachu_hell_debug', 10)

    def initialize_mission(self):
        """初始化任務"""
        self.change_state(FSM.INIT)
        
        # 設置YOLO檢測多個目標
        self.set_detection_targets(["pikachu", "sofa", "table", "tv", "chair"])
        
        self.publish_status("INITIALIZED", "Hell模式初始化完成")
        self.get_logger().info("🎯 設置多目標檢測: 皮卡丘 + 家具")

    def set_detection_targets(self, targets):
        """設置YOLO檢測目標"""
        for target in targets:
            target_msg = String()
            target_msg.data = target
            self.target_label_pub.publish(target_msg)
            time.sleep(0.1)  # 短暫延遲確保消息發送

    def change_state(self, new_state):
        """改變FSM狀態"""
        self.prev_state = self.state
        self.state = new_state
        self.state_start_time = self.clock.now()
        self.get_logger().info(f"🔄 狀態切換: {self.prev_state.name} → {new_state.name}")

    def publish_car_control(self, action_key):
        """發布車輛控制指令"""
        if action_key == self.current_action:
            return  # 避免重複發送相同指令
            
        if action_key not in ACTION_MAPPINGS:
            self.get_logger().warn(f"未知動作: {action_key}")
            return
            
        self.current_action = action_key
        self.action_start_time = self.clock.now()
        
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

    def publish_status(self, status, message=""):
        """發布任務狀態"""
        elapsed = (self.clock.now() - self.mission_start_time).nanoseconds / 1e9
        status_data = {
            "status": status,
            "state": self.state.name,
            "message": message,
            "elapsed_time": elapsed,
            "pikachu_detected": self.pikachu_detected,
            "furniture_count": len(self.furniture_map),
            "visited_areas": len(self.visited_areas)
        }
        
        status_msg = String()
        status_msg.data = json.dumps(status_data)
        self.status_pub.publish(status_msg)

    def publish_debug(self, debug_info):
        """發布調試信息"""
        debug_msg = String()
        debug_msg.data = json.dumps(debug_info)
        self.debug_pub.publish(debug_msg)

    # === 回調函數 ===
    def yolo_status_callback(self, msg):
        """YOLO檢測狀態回調"""
        self.pikachu_detected = msg.data
        if self.pikachu_detected:
            self.pikachu_last_seen = self.clock.now()

    def yolo_position_callback(self, msg):
        """YOLO位置回調"""
        self.pikachu_position = msg

    def object_offset_callback(self, msg):
        """多物體檢測回調"""
        try:
            objects = json.loads(msg.data)
            self.update_furniture_map(objects)
        except json.JSONDecodeError:
            pass

    def rgb_image_callback(self, msg):
        """RGB圖像回調"""
        if not self.room_confirmed:
            self.detect_room_from_image(msg)

    def pose_status_callback(self, msg):
        """位置狀態回調"""
        if len(msg.data) >= 5:
            self.x = msg.data[0]
            self.y = msg.data[1]
            self.yaw = msg.data[2]
            # self.road_ahead = bool(msg.data[3])
            self.obstacle_detected = bool(msg.data[4])

    def depth_info_callback(self, msg):
        """深度信息回調 - 用於障礙物檢測"""
        if len(msg.data) >= 20:
            # 分析前方區域的深度
            forward_depths = msg.data[7:13]  # 前方區域
            valid_depths = [d for d in forward_depths if d > 0]
            
            if valid_depths:
                min_distance = min(valid_depths)
                if min_distance < 0.8:  # 80cm內有障礙物
                    self.obstacle_detected = True
                else:
                    self.obstacle_detected = False

    # === 房間和家具檢測 ===
    def detect_room_from_image(self, image_msg):
        """從RGB圖像檢測房間類型"""
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")
            room_features = self.analyze_room_features(cv_image)
            
            self.room_detection_samples.append(room_features)
            
            if len(self.room_detection_samples) >= 10:
                # 分析多個樣本確定房間類型
                living_room_votes = sum(1 for r in self.room_detection_samples if r == "living_room")
                
                if living_room_votes >= 7:
                    self.room_type = "living_room"
                    self.room_confirmed = True
                    self.get_logger().info("🏠 確認房間類型: Living Room")
                    self.change_state(FSM.FURNITURE_MAPPING)
                    
        except Exception as e:
            self.get_logger().error(f"房間檢測錯誤: {e}")

    def analyze_room_features(self, image):
        """分析房間特徵"""
        # 轉換到HSV色彩空間
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Living Room特徵檢測
        # 1. 木色/棕色檢測 (家具)
        brown_lower = np.array([10, 50, 50])
        brown_upper = np.array([25, 255, 200])
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        brown_ratio = np.sum(brown_mask > 0) / (image.shape[0] * image.shape[1])
        
        # 2. 藍色檢測 (可能的地毯或裝飾)
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        blue_ratio = np.sum(blue_mask > 0) / (image.shape[0] * image.shape[1])
        
        # 3. 邊緣檢測 (家具邊緣)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        
        # Living Room判斷邏輯
        if brown_ratio > 0.08 and edge_density > 0.05:
            return "living_room"
        else:
            return "unknown"

    def update_furniture_map(self, detected_objects):
        """更新家具地圖"""
        for obj in detected_objects:
            label = obj.get('label', 'unknown')
            offset = obj.get('offset_flu', [0, 0, 0])
            
            if label != 'pikachu':  # 只記錄家具
                if label not in self.furniture_map:
                    self.furniture_map[label] = []
                
                # 添加位置信息
                position = {
                    'offset': offset,
                    'timestamp': self.clock.now().nanoseconds / 1e9,
                    'confidence': obj.get('confidence', 0.5)
                }
                self.furniture_map[label].append(position)
                
                # 限制每種家具最多記錄10個位置
                if len(self.furniture_map[label]) > 10:
                    self.furniture_map[label].pop(0)

    # === 搜索策略 ===
    def execute_systematic_search(self):
        """執行系統性搜索"""
        state_time = (self.clock.now() - self.state_start_time).nanoseconds / 1e9
        
        if self.search_pattern == "spiral":
            self.execute_spiral_search(state_time)
        elif self.search_pattern == "grid":
            self.execute_grid_search(state_time)
        else:
            self.execute_adaptive_search(state_time)

    def execute_spiral_search(self, state_time):
        """螺旋搜索"""
        phase_time = state_time % (self.phase_duration * 4)  # 4階段循環
        
        if phase_time < self.phase_duration:
            # 階段1: 向前掃描
            self.scan_and_move("FORWARD")
        elif phase_time < self.phase_duration * 2:
            # 階段2: 右轉掃描
            self.scan_and_move("CLOCKWISE_ROTATION")
        elif phase_time < self.phase_duration * 3:
            # 階段3: 向前掃描
            self.scan_and_move("FORWARD")
        else:
            # 階段4: 左轉掃描
            self.scan_and_move("COUNTERCLOCKWISE_ROTATION")

    def execute_grid_search(self, state_time):
        """網格搜索"""
        # 簡化的網格搜索：左右掃描 + 前進
        cycle_time = state_time % 20.0  # 20秒一個周期
        
        if cycle_time < 6.0:
            self.scan_and_move("COUNTERCLOCKWISE_ROTATION_SLOW")
        elif cycle_time < 12.0:
            self.scan_and_move("CLOCKWISE_ROTATION_SLOW")
        elif cycle_time < 16.0:
            self.scan_and_move("FORWARD")
        else:
            self.scan_and_move("CLOCKWISE_ROTATION")

    def execute_adaptive_search(self, state_time):
        """自適應搜索 - 基於家具地圖"""
        if len(self.furniture_map) == 0:
            # 沒有家具信息，執行基本掃描
            self.execute_spiral_search(state_time)
            return
            
        # 基於家具位置規劃搜索
        # 皮卡丘可能在家具附近
        if self.should_search_near_furniture():
            self.search_near_furniture()
        else:
            self.execute_spiral_search(state_time)

    def should_search_near_furniture(self):
        """判斷是否應該在家具附近搜索"""
        # 如果檢測到沙發、桌子等，皮卡丘可能在附近
        priority_furniture = ['sofa', 'table', 'chair']
        for furniture in priority_furniture:
            if furniture in self.furniture_map and len(self.furniture_map[furniture]) > 0:
                return True
        return False

    def search_near_furniture(self):
        """在家具附近搜索"""
        # 簡化實現：慢速轉向搜索
        if self.current_scan_direction == 1:
            self.publish_car_control("CLOCKWISE_ROTATION_SLOW")
        else:
            self.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
            
        # 定期改變掃描方向
        if self.action_start_time and \
           (self.clock.now() - self.action_start_time).nanoseconds / 1e9 > 5.0:
            self.current_scan_direction *= -1

    def scan_and_move(self, primary_action):
        """掃描並移動"""
        if self.obstacle_detected:
            # 遇到障礙物，嘗試避開
            self.avoid_obstacle()
        else:
            # 正常執行動作
            self.publish_car_control(primary_action)

    def avoid_obstacle(self):
        """避開障礙物"""
        # 簡單的避障策略
        avoid_actions = ["CLOCKWISE_ROTATION", "COUNTERCLOCKWISE_ROTATION", "BACKWARD"]
        
        # 選擇避障動作
        if self.last_safe_direction in avoid_actions:
            action = self.last_safe_direction
        else:
            action = "CLOCKWISE_ROTATION"
            
        self.publish_car_control(action)
        self.last_safe_direction = action

    # === 皮卡丘接近邏輯 ===
    def approach_pikachu(self):
        """接近皮卡丘"""
        if not self.pikachu_position:
            self.change_state(FSM.SYSTEMATIC_SEARCH)
            return
            
        x = self.pikachu_position.point.x
        y = self.pikachu_position.point.y
        z = self.pikachu_position.point.z
        
        distance = np.sqrt(x*x + y*y + z*z)
        
        # 成功條件：距離小於50cm
        if distance < 0.5:
            self.change_state(FSM.SUCCESS)
            return
        
        # 計算接近動作
        if abs(y) > 0.4:  # 需要對準
            if y > 0:
                self.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
            else:
                self.publish_car_control("CLOCKWISE_ROTATION_SLOW")
        elif x > 1.0:  # 距離較遠
            if self.obstacle_detected:
                self.avoid_obstacle()
            else:
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
            # 初始化完成後進入房間檢測
            self.change_state(FSM.ROOM_DETECTION)
            
        elif self.state == FSM.ROOM_DETECTION:
            if self.room_confirmed:
                self.change_state(FSM.FURNITURE_MAPPING)
            else:
                # 緩慢前進進行房間檢測
                self.publish_car_control("FORWARD_SLOW")
                
        elif self.state == FSM.FURNITURE_MAPPING:
            # 快速掃描建立家具地圖
            state_time = (current_time - self.state_start_time).nanoseconds / 1e9
            
            if state_time < 10.0:  # 前10秒建立家具地圖
                if state_time < 5.0:
                    self.publish_car_control("CLOCKWISE_ROTATION_SLOW")
                else:
                    self.publish_car_control("COUNTERCLOCKWISE_ROTATION_SLOW")
            else:
                self.change_state(FSM.SYSTEMATIC_SEARCH)
                
        elif self.state == FSM.SYSTEMATIC_SEARCH:
            if self.pikachu_detected:
                self.change_state(FSM.PIKACHU_APPROACH)
            else:
                self.execute_systematic_search()
                
                # 如果搜索時間過長，切換到緊急搜索
                state_time = (current_time - self.state_start_time).nanoseconds / 1e9
                if state_time > 120.0:  # 2分鐘後切換到緊急搜索
                    self.change_state(FSM.EMERGENCY_SEARCH)
                    
        elif self.state == FSM.PIKACHU_APPROACH:
            if self.pikachu_detected:
                self.approach_pikachu()
            else:
                # 丟失目標，回到搜索
                self.change_state(FSM.SYSTEMATIC_SEARCH)
                
        elif self.state == FSM.EMERGENCY_SEARCH:
            # 緊急搜索：快速隨機移動
            state_time = (current_time - self.state_start_time).nanoseconds / 1e9
            if self.pikachu_detected:
                self.change_state(FSM.PIKACHU_APPROACH)
            elif state_time > 60.0:  # 緊急搜索1分鐘
                self.change_state(FSM.FAILED)
            else:
                # 快速隨機搜索
                cycle = int(state_time) % 4
                actions = ["FORWARD", "CLOCKWISE_ROTATION", "BACKWARD", "COUNTERCLOCKWISE_ROTATION"]
                self.publish_car_control(actions[cycle])
                
        elif self.state == FSM.SUCCESS:
            self.publish_car_control("STOP")
            self.publish_status("SUCCESS", f"🎉 成功找到皮卡丘！用時 {total_elapsed:.1f}秒")
            
        elif self.state == FSM.FAILED:
            self.publish_car_control("STOP")
            self.publish_status("FAILED", f"❌ 任務失敗，超時 {total_elapsed:.1f}秒")
        
        # 定期發布狀態
        if int(total_elapsed) % 10 == 0:  # 每10秒發布一次狀態
            self.publish_status("RUNNING", f"狀態: {self.state.name}")

def main(args=None):
    rclpy.init(args=args)
    
    node = PikachuSeekerHell()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Hell模式節點被中斷")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()