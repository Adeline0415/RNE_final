#!/usr/bin/env python3
"""
系統檢查腳本 - 診斷各個話題的狀態
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray, Bool
from pros_car_py.car_models import DeviceDataTypeEnum
import time

class SystemChecker(Node):
    def __init__(self):
        super().__init__('system_checker')
        
        # 記錄各話題的狀態
        self.topic_status = {
            'yolo_offset': False,
            'yolo_status': False,
            'car_rear': False,
            'car_front': False,
            'target_label': False
        }
        
        # 訂閱各個關鍵話題
        self.offset_sub = self.create_subscription(
            String, '/yolo/object/offset', self.offset_callback, 10)
        
        self.status_sub = self.create_subscription(
            Bool, '/yolo/detection/status', self.status_callback, 10)
        
        # 測試車輛控制話題的訂閱者數量
        self.rear_pub = self.create_publisher(
            Float32MultiArray, DeviceDataTypeEnum.car_C_rear_wheel, 10)
        self.front_pub = self.create_publisher(
            Float32MultiArray, DeviceDataTypeEnum.car_C_front_wheel, 10)
        
        # 測試目標設置
        self.target_pub = self.create_publisher(String, '/target_label', 10)
        
        # 定期檢查
        self.timer = self.create_timer(3.0, self.check_system)
        
        self.get_logger().info("🔧 系統檢查器啟動")

    def offset_callback(self, msg):
        self.topic_status['yolo_offset'] = True
        self.get_logger().info(f"✅ 收到YOLO offset數據: {msg.data[:100]}...")

    def status_callback(self, msg):
        self.topic_status['yolo_status'] = True
        status = "檢測到物體" if msg.data else "未檢測到物體"
        self.get_logger().info(f"✅ 收到YOLO狀態: {status}")

    def check_system(self):
        """系統檢查"""
        self.get_logger().info("🔍 === 系統狀態檢查 ===")
        
        # 檢查話題列表
        topic_list = self.get_topic_names_and_types()
        important_topics = [
            '/yolo/object/offset',
            '/yolo/detection/status', 
            '/target_label',
            DeviceDataTypeEnum.car_C_rear_wheel,
            DeviceDataTypeEnum.car_C_front_wheel
        ]
        
        for topic in important_topics:
            exists = any(topic in t[0] for t in topic_list)
            status = "✅ 存在" if exists else "❌ 不存在"
            self.get_logger().info(f"📡 話題 {topic}: {status}")
        
        # 檢查訂閱者數量
        rear_subs = self.rear_pub.get_subscription_count()
        front_subs = self.front_pub.get_subscription_count()
        target_subs = self.target_pub.get_subscription_count()
        
        self.get_logger().info(f"🚗 後輪話題訂閱者: {rear_subs}")
        self.get_logger().info(f"🚗 前輪話題訂閱者: {front_subs}")
        self.get_logger().info(f"🎯 目標設置話題訂閱者: {target_subs}")
        
        # 測試發送目標設置
        target_msg = String()
        target_msg.data = "pikachu"
        self.target_pub.publish(target_msg)
        self.get_logger().info("📤 發送目標設置: pikachu")
        
        # 測試車輛控制（停止指令）
        stop_msg = Float32MultiArray()
        stop_msg.data = [0.0, 0.0]
        self.rear_pub.publish(stop_msg)
        self.front_pub.publish(stop_msg)
        self.get_logger().info("📤 發送車輛停止指令")
        
        self.get_logger().info("🔍 === 檢查完成 ===\n")

def main(args=None):
    rclpy.init(args=args)
    
    node = SystemChecker()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 系統檢查器被中斷")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()