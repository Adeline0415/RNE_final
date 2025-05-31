import os
import cv2
import numpy as np
import roslibpy
from ultralytics import YOLO
import base64

class YOLOProcessor:
    def __init__(self, model_path, ros_host, ros_port):
        # 初始化 YOLO 模型並設置到 GPU
        self.model = YOLO(model_path)
        self.model.to('cuda')

        # 連接 rosbridge
        self.client = roslibpy.Ros(host=ros_host, port=ros_port)
        self.client.run()

        # 檢查 ROS 連線狀態
        if self.client.is_connected:
            print(f"Successfully connected to rosbridge server at ws://{ros_host}:{ros_port}")

        # 設定訂閱與發布 topic
        self.image_listener = roslibpy.Topic(self.client, '/camera/color/image_raw/compressed', 'sensor_msgs/msg/CompressedImage')
        self.yolo_publisher = roslibpy.Topic(self.client, '/yolo/detection/compressed', 'sensor_msgs/msg/CompressedImage')

    def process_image(self, data):
        # 解碼接收的影像數據
        compressed_img = data['data']
        if isinstance(compressed_img, str):
            compressed_img = base64.b64decode(compressed_img)

        img_data = np.frombuffer(compressed_img, np.uint8)
        img_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

        if img_bgr is None:
            print("Failed to decode image.")
            return

        # 使用 YOLO 進行物體辨識並獲得帶有 bounding box 的影像
        img_with_boxes = self.detect_objects(img_bgr)

        # 將處理後的影像壓縮成 JPEG 格式
        _, compressed_img = cv2.imencode('.jpg', img_with_boxes)
        compressed_data = compressed_img.tobytes()

        # 將二進位數據轉換為 Base64 字串
        base64_data = base64.b64encode(compressed_data).decode('utf-8')

        # 發布處理後的影像到新的 topic，使用 Base64 編碼的數據
        self.yolo_publisher.publish(roslibpy.Message({'format': 'jpeg', 'data': base64_data}))
        print("Published processed image with bounding boxes.")

    def detect_objects(self, img):
        # 使用 YOLO 模型進行物體檢測
        results = self.model(img)

        # 畫出 bounding box
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{int(box.cls)}: {float(box.conf):.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img

    def start_processing(self):
        # 訂閱影像訊息
        self.image_listener.subscribe(self.process_image)
        print("Waiting for images...")

    def stop_processing(self):
        # 取消訂閱並關閉連接
        self.image_listener.unsubscribe()
        self.yolo_publisher.unadvertise()
        self.client.terminate()
        print("Disconnected")

# 主程式
if __name__ == "__main__":
    # 設置模型路徑和 rosbridge 參數
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_nano_auto_augu_super_close_fire.pt')
    ROS_HOST = '192.168.0.211'
    ROS_PORT = 9090

    # 創建 YOLOProcessor 物件並開始處理影像
    yolo_processor = YOLOProcessor(MODEL_PATH, ROS_HOST, ROS_PORT)
    try:
        yolo_processor.start_processing()
        while yolo_processor.client.is_connected:
            pass
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        yolo_processor.stop_processing()
