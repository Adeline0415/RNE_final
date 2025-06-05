#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from std_msgs.msg import String, Float32MultiArray
from pros_car_py.ros_communicator_config import ACTION_MAPPINGS
from pros_car_py.car_models import DeviceDataTypeEnum
from enum import Enum, auto
import time

class SimpleState(Enum):
    INIT = auto()              # 初始化
    SCANNING = auto()          # 原地掃描
    MOVING_TO_CENTER = auto()  # 移動到房間中央
    APPROACHING = auto()       # 接近皮卡丘
    SUCCESS = auto()           # 成功

class PikachuNavHell(Node):
    def __init__(self):
        super().__init__('pikachu_nav_hell')
        
        # === 狀態管理 ===
        self.state = SimpleState.INIT
        self.state_start_time = None
        self.clock = Clock()
        
        # === 皮卡丘檢測 ===
        self.pikachu_detected = False
        self.delta_x = 0.0  # 像素偏移（左負右正）
        
        # === 掃描計時 ===
        self.scan_start_time = None
        self.scan_duration = 8.0  # 原地轉一圈預估8秒
        self.scan_count = 0  # 掃描次數
        
        # === 移動到中央 ===
        self.move_start_time = None
        self.move_duration = 3.0  # 往前移動3秒到房間中央
        
        # === 設置訂閱者和發布者 ===
        self.setup_subscribers()
        self.setup_publishers()
        
        # === 初始化 ===
        self.initialize_mission()
        
        # 主循環定時器
        self.timer = self.create_timer(0.1, self.main_loop)  # 10Hz
        
        self.get_logger().info("🚀 皮卡丘Hell模式搜索器啟動")

    def setup_subscribers(self):
        """設置訂閱者"""
        self.yolo_target_info_sub = self.create_subscription(
            Float32MultiArray, '/yolo/target_info', self.yolo_target_info_callback, 10)

    def setup_publishers(self):
        """設置發布者"""
        # 車輛控制
        self.rear_wheel_pub = self.create_publisher(
            Float32MultiArray, DeviceDataTypeEnum.car_C_rear_wheel, 10)
        self.front_wheel_pub = self.create_publisher(
            Float32MultiArray, DeviceDataTypeEnum.car_C_front_wheel, 10)

    def initialize_mission(self):
        """初始化任務"""
        self.change_state(SimpleState.INIT)
        self.get_logger().info("🎯 開始皮卡丘搜索任務")

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
        
        self.get_logger().info(f"🚗 執行動作: {action_key}")

    # === 回調函數 ===
    def yolo_target_info_callback(self, msg):
        """YOLO目標信息回調"""
        try:
            if len(msg.data) >= 2:
                found_target = bool(msg.data[0])  # 是否找到皮卡丘
                self.delta_x = msg.data[1]       # 像素偏移（左負右正）
                
                self.pikachu_detected = found_target
                
                if self.pikachu_detected:
                    self.get_logger().info(
                        f"🎯 檢測到皮卡丘！像素偏移: {self.delta_x:.0f}px")
                
        except Exception as e:
            self.get_logger().error(f"YOLO回調錯誤: {e}")

    # === 主要邏輯 ===
    def scan_for_pikachu(self):
        """原地掃描皮卡丘"""
        if self.scan_start_time is None:
            self.scan_start_time = self.clock.now()
            self.get_logger().info(f"🔍 開始第{self.scan_count + 1}次掃描...")
        
        elapsed = (self.clock.now() - self.scan_start_time).nanoseconds / 1e9
        
        if elapsed < self.scan_duration:
            # 逆時鐘旋轉掃描
            self.publish_car_control("COUNTERCLOCKWISE_ROTATION")
        else:
            # 掃描完成
            self.scan_count += 1
            self.scan_start_time = None
            
            if self.scan_count == 1:
                # 第一次掃描完成，移動到房間中央
                self.get_logger().info("🏃 第一次掃描完成，移動到房間中央...")
                self.change_state(SimpleState.MOVING_TO_CENTER)
            else:
                # 第二次掃描也沒找到，停止任務
                self.get_logger().info("😞 兩次掃描都未找到皮卡丘，任務結束")
                self.publish_car_control("STOP")

    def move_to_center(self):
        """移動到房間中央"""
        if self.move_start_time is None:
            self.move_start_time = self.clock.now()
        
        elapsed = (self.clock.now() - self.move_start_time).nanoseconds / 1e9
        
        if elapsed < self.move_duration:
            # 直線前進
            self.publish_car_control("FORWARD")
        else:
            # 移動完成，開始第二次掃描
            self.get_logger().info("🔍 到達房間中央，開始第二次掃描...")
            self.move_start_time = None
            self.change_state(SimpleState.SCANNING)

    def approach_pikachu(self):
        """接近皮卡丘 - 讓皮卡丘在畫面中央"""
        # 轉換像素偏移為米（可調整係數）
        left_offset_m = self.delta_x / 300.0
        
        # 設定對準閾值（像素）
        alignment_threshold = 50  # 50像素內算對準
        
        if abs(self.delta_x) > alignment_threshold:
            # 需要轉向對準
            if self.delta_x < 0:  # 皮卡丘在左邊
                self.publish_car_control("COUNTERCLOCKWISE_ROTATION")
                self.get_logger().info(f"🔄 皮卡丘在左邊，左轉對準 (偏移: {self.delta_x:.0f}px)")
            else:  # 皮卡丘在右邊
                self.publish_car_control("CLOCKWISE_ROTATION")
                self.get_logger().info(f"🔄 皮卡丘在右邊，右轉對準 (偏移: {self.delta_x:.0f}px)")
        else:
            # 已對準，直接前進
            self.publish_car_control("FORWARD")
            self.get_logger().info(f"➡️  已對準皮卡丘，前進接近 (偏移: {self.delta_x:.0f}px)")
            
            # 設定成功條件：前進一段時間後算成功
            approach_time = (self.clock.now() - self.state_start_time).nanoseconds / 1e9
            if approach_time > 20.0:  # 接近20秒後算成功
                self.change_state(SimpleState.SUCCESS)

    # === 主循環 ===
    def main_loop(self):
        """主循環"""
        if self.state == SimpleState.INIT:
            # 初始化完成，開始第一次掃描
            time.sleep(1)  # 等待1秒讓系統穩定
            self.change_state(SimpleState.SCANNING)
            
        elif self.state == SimpleState.SCANNING:
            if self.pikachu_detected:
                self.change_state(SimpleState.APPROACHING)
            else:
                self.scan_for_pikachu()
                
        elif self.state == SimpleState.MOVING_TO_CENTER:
            if self.pikachu_detected:
                self.change_state(SimpleState.APPROACHING)
            else:
                self.move_to_center()
                
        elif self.state == SimpleState.APPROACHING:
            if self.pikachu_detected:
                self.approach_pikachu()
            else:
                # 丟失目標，回到掃描
                self.get_logger().info("⚠️  丟失皮卡丘，回到掃描模式")
                self.scan_start_time = None
                self.change_state(SimpleState.SCANNING)
                
        elif self.state == SimpleState.SUCCESS:
            self.publish_car_control("STOP")
            self.get_logger().info("🎉 成功找到並接近皮卡丘！")

def main(args=None):
    rclpy.init(args=args)
    
    node = PikachuNavHell()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Hell模式節點被中斷")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()