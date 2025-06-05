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

class SimpleState(Enum):
    INIT = auto()               # 初始化
    SCANNING = auto()           # 原地掃描
    APPROACHING = auto()        # 接近皮卡丘
    OBSTACLE_AVOIDANCE = auto() # 避障
    SUCCESS = auto()            # 成功

class PikachuNavHell(Node):
    def __init__(self):
        super().__init__('pikachu_nav_hell')
        
        # === 狀態管理 ===
        self.state = SimpleState.INIT
        self.state_start_time = None
        self.scan_phase = 0  # 掃描階段
        self.clock = Clock()
        
        # === 皮卡丘檢測 ===
        self.pikachu_detected = False
        self.delta_x = 0.0  # 像素偏移（左負右正）
        
        # === 掃描計時 ===
        self.scan_start_time = None
        self.scan_duration = 11.0  # 原地轉一圈
        self.scan_count = 0  # 掃描次數
        
        # === 移動到中央 ===
        self.move_start_time = None
        self.move_duration = 4.5  # 往前移動5秒到房間中央
        
        # === 設置訂閱者和發布者 ===
        self.setup_subscribers()
        self.setup_publishers()
        
        self.bridge = CvBridge()
        self.pikachu_total_area = 0.0
        self.target_area_threshold = 52000 
        
        # === RGB圖像監聽 === 
        self.current_rgb_image = None
        self.previous_rgb_image = None
        self.last_rgb_check_time = None
        self.rgb_check_interval = 1.0
        self.image_similarity_threshold = 0.85
        self.consecutive_similar_count = 0
        self.max_consecutive_similar = 3  # 連續2次相似才判定為撞到
        
        # === 避障模式 === 
        self.obstacle_start_time = None
        self.obstacle_phase = 0
        self.obstacle_durations = [2.0, 1.5, 2.0]

        # 皮卡丘面積變化檢測
        self.pikachu_area_history = []  # 存儲歷史面積數據
        self.area_check_interval = 2.0  # 1秒檢查一次
        self.last_area_check_time = None
        self.area_change_threshold = 0.05  # 面積變化小於5%視為撞到
            
        # === 最終接近階段 === 
        self.final_approach_start_time = None
        self.final_approach_duration = 1.5

        self.current_action = "STOP"  # 初始狀態為停止

        # === 初始化 ===
        self.initialize_mission()
        
        # 主循環定時器
        self.timer = self.create_timer(0.1, self.main_loop)  # 10Hz
        
        self.get_logger().info("皮卡丘Hell模式搜索器啟動")

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
        self.change_state(SimpleState.INIT)
        self.get_logger().info("開始皮卡丘搜索任務")

    def change_state(self, new_state):
        """改變狀態"""
        old_state = self.state.name if hasattr(self, 'state') else "None"
        self.state = new_state
        self.state_start_time = self.clock.now()
        self.get_logger().info(f"狀態切換: {old_state} → {new_state.name}")

    def publish_car_control(self, action_key):
        """發布車輛控制指令"""
        # 記錄當前動作
        self.current_action = action_key
                
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
        
        self.get_logger().info(f"執行動作: {action_key}")

    # === 回調函數 ===
    def yolo_target_info_callback(self, msg):
        """YOLO目標信息回調"""
        try:
            if len(msg.data) >= 2:
                found_target = bool(msg.data[0])  # 是否找到皮卡丘
                self.delta_x = msg.data[1]       # 像素偏移（左負右正）
                self.pikachu_total_area = msg.data[2] if len(msg.data) == 3 else 0.0
                self.pikachu_detected = found_target
                
        except Exception as e:
            self.get_logger().error(f"YOLO回調錯誤: {e}")

    def rgb_image_callback(self, msg):
        """RGB圖像回調"""
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            self.previous_rgb_image = self.current_rgb_image
            self.current_rgb_image = cv_image
        except Exception as e:
            self.get_logger().error(f"RGB圖像回調錯誤: {e}")

    # === 主要邏輯 ===
    def scan_for_pikachu(self):
        """掃描皮卡丘 - 左轉100度→右轉50度→前進到中央→轉圈"""
        if self.scan_start_time is None:
            self.scan_start_time = self.clock.now()
            self.scan_phase = 0  # 0:左轉100度, 1:右轉50度, 2:前進到中央, 3:中央轉圈
            self.get_logger().info("開始掃描序列...")
        
        elapsed = (self.clock.now() - self.scan_start_time).nanoseconds / 1e9
        rotate_100 = 5.75
        rotate_50 = 3.2
        
        if self.scan_phase == 0:
            # 階段0: 左轉100度 (預估需要約6-7秒)
            if elapsed < rotate_100:
                self.publish_car_control("COUNTERCLOCKWISE_ROTATION")
                if elapsed < 1:
                    self.get_logger().info("階段1: 左轉100度掃描中...")
            else:
                self.scan_phase = 1
                self.scan_start_time = self.clock.now()  # 重置計時器
                self.get_logger().info("階段2: 右轉50度回到左斜前...")
        
        elif self.scan_phase == 1:
            # 階段1: 右轉50度回到左斜前 (預估需要約3秒)
            if elapsed < rotate_50:
                self.publish_car_control("CLOCKWISE_ROTATION")
            else:
                self.scan_phase = 2
                self.scan_start_time = self.clock.now()  # 重置計時器
                self.get_logger().info("階段3: 直線前進到房間中央...")
        
        elif self.scan_phase == 2:
            # 階段2: 直線前進到房間中央
            self.last_rgb_check_time = self.clock.now()
            if elapsed < self.move_duration:  # 使用原本的 move_duration (3秒)
                self.publish_car_control("FORWARD")
            else:
                self.scan_phase = 3
                self.scan_start_time = self.clock.now()  # 重置計時器
                self.get_logger().info("階段4: 在中央位置轉圈掃描...")
        
        elif self.scan_phase == 3:
            # 階段3: 在中央轉一圈
            if elapsed < self.scan_duration:  # 使用原本的 scan_duration (8秒)
                self.publish_car_control("COUNTERCLOCKWISE_ROTATION")
            else:
                # 所有階段完成，任務結束
                self.get_logger().info("完整掃描序列結束，未找到皮卡丘，任務結束")
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
            self.get_logger().info("到達房間中央，開始第二次掃描...")
            self.move_start_time = None
            self.change_state(SimpleState.SCANNING)

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
                self.change_state(SimpleState.SUCCESS)
            return  
        
        # 檢查是否達到面積閾值（首次觸發）
        if self.pikachu_total_area >= self.target_area_threshold:
            self.final_approach_start_time = self.clock.now()
            self.get_logger().info(f"皮卡丘面積達標({self.pikachu_total_area:.0f}px²)！開始最終接近...")
            return  # 返回，下一輪循環會執行上面的邏輯
        
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

    def calculate_image_similarity(self, img1, img2):
        """計算兩張圖像的相似度"""
        if img1 is None or img2 is None:
            return 0.0
        
        try:
            h, w = img1.shape[:2]
            small_size = (w//4, h//4)
            
            img1_small = cv2.resize(img1, small_size)
            img2_small = cv2.resize(img2, small_size)
            
            gray1 = cv2.cvtColor(img1_small, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2_small, cv2.COLOR_BGR2GRAY)
            
            mse = np.mean((gray1 - gray2) ** 2)
            if mse == 0:
                return 1.0
            
            max_pixel_value = 255.0
            psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
            similarity = min(1.0, psnr / 50.0)
            return similarity
        except Exception as e:
            self.get_logger().error(f"圖像相似度計算錯誤: {e}")
            return 0.0

    def check_obstacle_collision(self):
        """檢查是否撞到障礙物 - RGB相似度 OR 皮卡丘面積停滯"""
        current_time = self.clock.now()
        
        # 方法1: RGB相似度檢測（原有邏輯）
        rgb_collision = self.check_rgb_similarity_collision(current_time)
        
        # 方法2: 皮卡丘面積停滯檢測（新增）
        area_collision = self.check_pikachu_area_stagnation(current_time)
        
        # OR 關係：任一方法檢測到碰撞就判定為撞到
        return rgb_collision or area_collision

    def check_rgb_similarity_collision(self, current_time):
        """RGB相似度檢測方法"""
        time_diff = (current_time - self.last_rgb_check_time).nanoseconds / 1e9
        
        if time_diff >= self.rgb_check_interval:
            if self.current_rgb_image is not None and self.previous_rgb_image is not None:
                similarity = self.calculate_image_similarity(
                    self.current_rgb_image, self.previous_rgb_image)
                
                if similarity > self.image_similarity_threshold:
                    self.consecutive_similar_count += 1
                    self.get_logger().info(f"⚠️ RGB相似度高: {similarity:.3f} (連續{self.consecutive_similar_count}次)")
                    
                    if self.consecutive_similar_count >= self.max_consecutive_similar:
                        self.get_logger().warn(f"🚧 RGB檢測到障礙物！連續{self.consecutive_similar_count}次相似")
                        self.consecutive_similar_count = 0
                        return True
                else:
                    self.consecutive_similar_count = 0
                    self.get_logger().info(f"✅ RGB正常，相似度: {similarity:.3f}")
            
            self.last_rgb_check_time = current_time
        
        return False

    def check_pikachu_area_stagnation(self, current_time):
        """皮卡丘面積停滯檢測方法"""
        # 只有在檢測到皮卡丘且正在前進時才檢查
        if not self.pikachu_detected or self.current_action != "FORWARD":
            return False
        
        # 初始化面積檢查時間
        if self.last_area_check_time is None:
            self.last_area_check_time = current_time
            return False
        
        time_diff = (current_time - self.last_area_check_time).nanoseconds / 1e9
        
        if time_diff >= self.area_check_interval:
            # 記錄當前面積和時間戳
            current_area_data = {
                'area': self.pikachu_total_area,
                'timestamp': current_time
            }
            self.pikachu_area_history.append(current_area_data)
            
            # 保持歷史記錄不超過10個（約5秒的歷史）
            if len(self.pikachu_area_history) > 10:
                self.pikachu_area_history.pop(0)
            
            # 檢查0.5秒前的面積
            if len(self.pikachu_area_history) >= 2:
                previous_area_data = self.pikachu_area_history[-2]  # 0.5秒前的數據
                current_area = current_area_data['area']
                previous_area = previous_area_data['area']
                
                # 計算面積變化比例
                if previous_area > 0:  # 避免除零
                    area_change_ratio = (current_area - previous_area) / previous_area
                    
                    self.get_logger().info(
                        f"📐 面積變化: {previous_area:.0f} → {current_area:.0f} "
                        f"(變化率: {area_change_ratio:.3f})"
                    )
                    
                    # 如果面積變化太小（小於閾值），判定為撞到
                    if abs(area_change_ratio) < self.area_change_threshold:
                        self.get_logger().warn(
                            f"🚧 皮卡丘面積停滯檢測到障礙物！"
                            f"變化率僅 {area_change_ratio:.3f} < {self.area_change_threshold}"
                        )
                        return True
            
            self.last_area_check_time = current_time
        
        return False

    def obstacle_avoidance(self):
        """避障模式：順時鐘→前進→逆時鐘"""
        if self.obstacle_start_time is None:
            self.obstacle_start_time = self.clock.now()
            self.obstacle_phase = 0
            self.get_logger().info("開始避障序列...")
        
        elapsed = (self.clock.now() - self.obstacle_start_time).nanoseconds / 1e9
        current_duration = self.obstacle_durations[self.obstacle_phase]
        
        if elapsed < current_duration:
            if self.obstacle_phase == 0:
                self.publish_car_control("CLOCKWISE_ROTATION")
            elif self.obstacle_phase == 1:
                self.publish_car_control("FORWARD")
            elif self.obstacle_phase == 2:
                self.publish_car_control("COUNTERCLOCKWISE_ROTATION")
        else:
            if self.obstacle_phase < 2:
                self.obstacle_phase += 1
                self.obstacle_start_time = self.clock.now()
            else:
                self.get_logger().info("✅避障完成，返回掃描模式")
                self.obstacle_start_time = None
                self.obstacle_phase = 0
                self.change_state(SimpleState.SCANNING)

    # === 主循環 ===
    def main_loop(self):
        """主循環"""
        if (self.state in [SimpleState.APPROACHING,  SimpleState.SCANNING]
            and self.last_rgb_check_time is not None):
            if self.check_obstacle_collision():
                self.change_state(SimpleState.OBSTACLE_AVOIDANCE)
                return

        if self.state == SimpleState.INIT:
            # 初始化完成，開始第一次掃描
            time.sleep(1)  # 等待1秒讓系統穩定
            self.change_state(SimpleState.SCANNING)
            
        elif self.state == SimpleState.SCANNING:
            if self.pikachu_detected:
                self.change_state(SimpleState.APPROACHING)
                self.last_rgb_check_time = self.clock.now()
            else:
                self.scan_for_pikachu()
                
        elif self.state == SimpleState.APPROACHING:
            self.approach_pikachu()
                
        elif self.state == SimpleState.SUCCESS:
            self.publish_car_control("STOP")
            self.get_logger().info("成功找到並接近皮卡丘！")

        elif self.state == SimpleState.OBSTACLE_AVOIDANCE:
            self.obstacle_avoidance()

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