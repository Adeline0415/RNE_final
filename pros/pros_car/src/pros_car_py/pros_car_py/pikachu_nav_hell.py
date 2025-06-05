#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from std_msgs.msg import String, Float32MultiArray, Bool
from geometry_msgs.msg import PointStamped
from pros_car_py.ros_communicator_config import ACTION_MAPPINGS
from pros_car_py.car_models import DeviceDataTypeEnum
from enum import Enum, auto
import json
import time

class SimpleState(Enum):
    INIT = auto()              # 初始化
    SCANNING = auto()          # 原地掃描
    APPROACHING = auto()       # 接近皮卡丘
    AVOIDING_OBSTACLE = auto() # 避障
    SUCCESS = auto()           # 成功
    FAILED = auto()            # 失敗

class PikachuNavHell(Node):
    def __init__(self):
        super().__init__('pikachu_nav_hell')
        
        # === 狀態管理 ===
        self.state = SimpleState.INIT
        self.state_start_time = None
        self.clock = Clock()
        
        # === 皮卡丘檢測 ===
        self.pikachu_detected = False
        self.pikachu_position = None  # [forward, left, up] 格式
        self.detection_status = False
        
        # === 避障邏輯 ===
        self.obstacle_detected = False
        self.avoid_step = 0  # 避障步驟: 0=右轉90度, 1=前進, 2=左轉90度
        
        # === 掃描計時 ===
        self.scan_start_time = None
        self.scan_duration = 8.0  # 原地轉一圈預估8秒
        
        # === 設置訂閱者 ===
        self.setup_subscribers()
        
        # === 設置發布者 ===
        self.setup_publishers()
        
        # === 初始化 ===
        self.initialize_mission()
        
        # 主循環定時器
        self.timer = self.create_timer(0.1, self.main_loop)  # 10Hz
        
        self.get_logger().info("🚀 皮卡丘Hell模式搜索器啟動")

    def setup_subscribers(self):
        """設置訂閱者"""
        # YOLO物體偏移資訊 (主要數據來源)
        self.object_offset_sub = self.create_subscription(
            String, '/yolo/object/offset', self.object_offset_callback, 10)
        
        # YOLO檢測狀態
        self.detection_status_sub = self.create_subscription(
            Bool, '/yolo/detection/status', self.detection_status_callback, 10)
        
        # 深度信息用於避障
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

    def initialize_mission(self):
        """初始化任務"""
        self.change_state(SimpleState.INIT)
        
        # 設置YOLO檢測皮卡丘
        self.set_detection_target("pikachu")
        
        self.get_logger().info("🎯 設置檢測目標: 皮卡丘")

    def set_detection_target(self, target):
        """設置YOLO檢測目標"""
        target_msg = String()
        target_msg.data = target
        
        # 多次發布確保YOLO收到
        for i in range(3):
            self.target_label_pub.publish(target_msg)
            time.sleep(0.1)
        
        self.get_logger().info(f"📡 設置檢測目標: {target} (已發布3次確保收到)")

    def change_state(self, new_state):
        """改變狀態"""
        old_state = self.state.name if hasattr(self, 'state') else "None"
        self.state = new_state
        self.state_start_time = self.clock.now()
        self.get_logger().info(f"🔄 狀態切換: {old_state} → {new_state.name}")

    def publish_car_control(self, action_key):
        """發布車輛控制指令"""
        if action_key not in ACTION_MAPPINGS:
            self.get_logger().warn(f"未知動作: {action_key}")
            return
            
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
        
        # 改成info級別，確保能看到車輛控制指令
        self.get_logger().info(f"🚗 執行動作: {action_key}, 速度: [{vel1}, {vel2}, {vel3}, {vel4}]")
        
        # 檢查發布者是否正常
        rear_sub_count = self.rear_wheel_pub.get_subscription_count()
        front_sub_count = self.front_wheel_pub.get_subscription_count()
        self.get_logger().info(f"📡 後輪訂閱者: {rear_sub_count}, 前輪訂閱者: {front_sub_count}")

    # === 回調函數 ===
    def object_offset_callback(self, msg):
        """YOLO物體偏移回調 - 主要數據來源"""
        self.get_logger().info("📥 收到YOLO offset數據！")  # 確認有收到回調
        
        try:
            # 檢查是否有數據
            if not msg.data or msg.data.strip() == "":
                self.get_logger().info("📡 收到空的offset數據")
                return
            
            # 檢查是否是空陣列
            if msg.data.strip() == "[]":
                self.pikachu_detected = False
                self.pikachu_position = None
                self.get_logger().info("📡 YOLO未檢測到任何物體")
                return
            
            objects = json.loads(msg.data)
            
            # 重置檢測狀態
            self.pikachu_detected = False
            self.pikachu_position = None
            
            # 打印所有檢測到的物體（調試用）
            if objects:
                labels = [obj.get('label', 'unknown') for obj in objects]
                self.get_logger().info(f"🔍 檢測到物體: {labels}")
            
            # 尋找皮卡丘 (不區分大小寫，多種可能的名稱)
            pikachu_names = ['pikachu']
            for obj in objects:
                obj_label = obj.get('label', '')
                
                if any(name in obj_label for name in pikachu_names):
                    self.pikachu_detected = True
                    
                    # 獲取FLU座標系位置 [forward, left, up]
                    offset_flu = obj.get('offset_flu', [0, 0, 0])
                    self.pikachu_position = offset_flu
                    
                    self.get_logger().info(
                        f"🎯 檢測到皮卡丘！標籤: '{obj_label}', 位置: F={offset_flu[0]:.2f}, L={offset_flu[1]:.2f}, U={offset_flu[2]:.2f}")
                    break
                    
            if not self.pikachu_detected:
                self.get_logger().info("👀 本次掃描未檢測到皮卡丘，繼續搜索...")
                
        except json.JSONDecodeError as e:
            self.get_logger().info(f"⚠️  JSON解析問題: {e}")
            self.get_logger().info(f"📄 原始數據: {msg.data}")
        except Exception as e:
            self.get_logger().info(f"⚠️  數據處理問題: {e}")
            self.get_logger().info(f"📄 原始數據: {msg.data}")

    def detection_status_callback(self, msg):
        """YOLO檢測狀態回調"""
        self.detection_status = msg.data

    def depth_info_callback(self, msg):
        """深度信息回調 - 用於避障"""
        if len(msg.data) >= 20:
            # 分析前方區域的深度
            forward_depths = msg.data[7:13]  # 前方區域
            valid_depths = [d for d in forward_depths if d > 0]
            
            if valid_depths:
                min_distance = min(valid_depths)
                self.obstacle_detected = min_distance < 0.6  # 60cm內有障礙物
            else:
                self.obstacle_detected = False

    # === 主要邏輯 ===
    def scan_for_pikachu(self):
        """原地掃描皮卡丘"""
        if self.scan_start_time is None:
            self.scan_start_time = self.clock.now()
            self.get_logger().info("🔍 開始原地掃描...")
        
        elapsed = (self.clock.now() - self.scan_start_time).nanoseconds / 1e9
        
        self.get_logger().info(f"⏰ 掃描進度: {elapsed:.1f}/{self.scan_duration:.1f} 秒")
        
        if elapsed < self.scan_duration:
            # 最慢速度逆時鐘旋轉
            self.publish_car_control("COUNTERCLOCKWISE_ROTATION")
        else:
            # 掃描完一圈仍未找到，延長搜索時間或重新開始
            self.get_logger().info("⚠️  第一輪掃描完成，未找到皮卡丘，重新開始掃描...")
            self.scan_start_time = None  # 重置掃描時間，重新開始
            # 不要直接進入失敗狀態，而是重新掃描

    def approach_pikachu(self):
        """接近皮卡丘"""
        if not self.pikachu_detected or not self.pikachu_position:
            # 丟失目標，回到掃描
            self.change_state(SimpleState.SCANNING)
            return
        
        forward_dist = self.pikachu_position[0]  # 前方距離
        left_offset = self.pikachu_position[1]   # 左右偏移
        
        # 檢查是否到達目標
        if forward_dist < 0.5:  # 50cm內算成功
            self.change_state(SimpleState.SUCCESS)
            return
        
        # 檢查是否有障礙物
        if self.obstacle_detected:
            self.change_state(SimpleState.AVOIDING_OBSTACLE)
            return
        
        # 決定移動方向
        if abs(left_offset) > 0.3:  # 需要對準
            if left_offset > 0:  # 皮卡丘在左邊
                self.publish_car_control("COUNTERCLOCKWISE_ROTATION")
            else:  # 皮卡丘在右邊
                self.publish_car_control("CLOCKWISE_ROTATIONW")
        else:  # 已對準，直接前進
            if forward_dist > 1.5:
                self.publish_car_control("FORWARD")  # 距離遠，正常速度
            else:
                self.publish_car_control("FORWARD_SLOW")  # 距離近，慢速

    def avoid_obstacle(self):
        """簡單避障邏輯：右轉90度→前進→左轉90度"""
        state_time = (self.clock.now() - self.state_start_time).nanoseconds / 1e9
        
        if self.avoid_step == 0:  # 右轉90度
            if state_time < 2.0:  # 轉2秒
                self.publish_car_control("CLOCKWISE_ROTATION")
            else:
                self.avoid_step = 1
                self.state_start_time = self.clock.now()
        
        elif self.avoid_step == 1:  # 前進一點
            if state_time < 1.5:  # 前進1.5秒
                self.publish_car_control("FORWARD")
            else:
                self.avoid_step = 2
                self.state_start_time = self.clock.now()
        
        elif self.avoid_step == 2:  # 左轉90度
            if state_time < 2.0:  # 轉2秒
                self.publish_car_control("COUNTERCLOCKWISE_ROTATION")
            else:
                # 避障完成，重置並回到接近模式
                self.avoid_step = 0
                self.change_state(SimpleState.APPROACHING)

    # === 主循環 ===
    def main_loop(self):
        """主循環"""
        if self.state == SimpleState.INIT:
            # 初始化完成，開始掃描
            time.sleep(1)  # 等待1秒讓系統穩定
            self.change_state(SimpleState.SCANNING)
            
        elif self.state == SimpleState.SCANNING:
            if self.pikachu_detected:
                self.change_state(SimpleState.APPROACHING)
            else:
                self.scan_for_pikachu()
                
        elif self.state == SimpleState.APPROACHING:
            if self.pikachu_detected:
                self.approach_pikachu()
            else:
                # 丟失目標，回到掃描
                self.scan_start_time = None  # 重置掃描時間
                self.change_state(SimpleState.SCANNING)
                
        elif self.state == SimpleState.AVOIDING_OBSTACLE:
            self.avoid_obstacle()
            
        elif self.state == SimpleState.SUCCESS:
            self.publish_car_control("STOP")
            self.get_logger().info("🎉 成功找到並接近皮卡丘！")
            
        elif self.state == SimpleState.FAILED:
            self.publish_car_control("STOP")
            self.get_logger().info("📝 任務狀態：未能在規定時間內找到皮卡丘")
            self.get_logger().info("💡 建議檢查：1) YOLO節點是否運行 2) 皮卡丘是否在視野內 3) 模型是否包含皮卡丘類別")
        
        # 定期檢查YOLO連接狀態
        current_time = self.clock.now()
        if hasattr(self, '_last_check_time'):
            time_since_check = (current_time - self._last_check_time).nanoseconds / 1e9
        else:
            time_since_check = 0
            self._last_check_time = current_time
        
        if time_since_check > 5.0:  # 每5秒檢查一次
            self._last_check_time = current_time
            # 檢查訂閱者數量
            offset_sub_count = len([node for node in self.get_topic_names_and_types() if '/yolo/object/offset' in node[0]])
            self.get_logger().info(f"🔗 YOLO連接檢查 - offset話題存在: {'是' if offset_sub_count > 0 else '否'}")
            
            # 重新發送目標設置
            self.set_detection_target("pikachu")

def main(args=None):
    rclpy.init(args=args)
    
    node = PikachuNavHell()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Hell模式節點被中斷")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()