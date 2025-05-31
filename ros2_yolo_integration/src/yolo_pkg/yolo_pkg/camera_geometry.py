import numpy as np
from yolo_pkg.camera_parameters import CameraParameters
import json  # Import the json module


class CameraGeometry:
    def __init__(self, yolo_depth_extractor):
        self.camera_intrinsics = CameraParameters().get_camera_intrinsics()
        self.yolo_depth_extractor = yolo_depth_extractor

    def calculate_3d_position(self):
        """
        計算每個物體在 IMU 坐標系中的 3D 位置。
        Returns:
            list: 每個物體的 3D 位置向量。
        """
        return self._process_objects(self._calculate_position, "position")

    def calculate_movement_to_center_crosshair(self):
        """
        計算需要的 3D 位移，使機械手臂的末端移動，讓畫面中心十字架對準物體中心點。
        Returns:
            list: 包含每個物體中心對齊的 3D 位移向量。
        """
        # 注意：_calculate_offset 返回的是相機坐標系的偏移，可能需要轉換到IMU坐標系
        return self._process_objects(self._calculate_offset, "movement")

    def calculate_offset_from_crosshair_2d(self):
        """
        計算每個物體中心相對於畫面中心十字架的 3D 偏移，
        並以右手系 (Forward, Left, Up) 表示：
          x: 前方 (Depth)
          y: 左方
          z: 上方
        Returns:
            list: 每個物體的 {'label': str, 'offset_flu': np.ndarray([x, y, z])} 列表。
        """
        return self._process_objects(self._calculate_offset_flu, "offset_flu")

    def _calculate_offset_flu(self, x, y, depth, fx, fy, cx, cy):
        """
        將畫面座標 (x, y) 與深度轉到 FLU 右手座標系。

        Args:
            x, y: 物體中心像素座標
            depth: 深度 (公尺)
            fx, fy, cx, cy: 相機內參

        Returns:
            np.ndarray([x_f, y_f, z_f]):
              x_f = +Z_cam （前方）
              y_f = -X_cam （左方）
              z_f = -Y_cam （上方）
        """
        # 先算出在光學相機座標系裡的 XYZ
        X_cam = (x - cx) * depth / fx  # 右正
        Y_cam = (y - cy) * depth / fy  # 下正
        Z_cam = depth  # 前正

        # 轉到 FLU 右手系
        x_f = Z_cam  # + 前
        y_f = -X_cam  # + 左
        z_f = -Y_cam  # + 上

        return np.array([x_f, y_f, z_f])

    def _process_objects(self, calculation_fn, result_key):
        """
        通用處理函數，計算每個物體的相關 3D 信息。

        Args:
            calculation_fn (function): 用於計算 3D 信息的函數。
            result_key (str): 在結果字典中使用的鍵名。

        Returns:
            str: 包含每個物體計算結果（數值四捨五入到小數點後三位）的 JSON 字符串。
                 如果沒有物體，返回空列表的 JSON 字符串 '[]'。
        """
        fx = self.camera_intrinsics["fx"]
        fy = self.camera_intrinsics["fy"]
        cx = self.camera_intrinsics["cx"]  # 十字架中心的 X 像素座標
        cy = self.camera_intrinsics["cy"]  # 十字架中心的 Y 像素座標

        yolo_objects = self.yolo_depth_extractor.get_yolo_object_depth()

        results = []  # Initialize results as a list

        if not yolo_objects:
            print("No YOLO objects detected or depth data unavailable.")
            # Return an empty list as a JSON string
            return json.dumps(results)

        for obj in yolo_objects:
            label = obj["label"]
            x1, y1, x2, y2 = obj["box"]
            # 確保深度值是公尺 (假設 yolo_depth_extractor 返回的是公尺)
            depth_value = obj["depth"]

            # 如果深度值無效或非正數，跳過該物體
            if depth_value is None or depth_value <= 0:
                # print(f"Invalid or non-positive depth for object {label}. Skipping.")
                continue

            # 計算物體中心點像素座標
            object_center_x = (x1 + x2) / 2
            object_center_y = (y1 + y2) / 2

            # 使用傳入的計算函數處理物體
            result = calculation_fn(
                object_center_x, object_center_y, depth_value, fx, fy, cx, cy
            )

            # Round numerical results to 3 decimal places
            rounded_result = None
            if isinstance(result, np.ndarray):
                # Round each element in the numpy array and convert to list
                rounded_result = [round(val, 3) for val in result.tolist()]
            elif isinstance(result, (list, tuple)):
                # Round each element in the list or tuple
                rounded_result = [round(val, 3) for val in result]
            elif isinstance(result, (int, float)):
                # Round a single numerical value
                rounded_result = round(result, 3)
            else:
                # Keep non-numeric results as is (though unlikely for coordinates)
                rounded_result = result

            results.append({"label": label, result_key: rounded_result})

        # Convert the list of dictionaries to a JSON string
        return json.dumps(results)

    def _calculate_position(self, x, y, depth, fx, fy, cx, cy):
        """
        計算物體在相機坐標系中的 3D 位置，然後轉換到 IMU 坐標系。
        (假設 IMU 坐標系 = ROS 標準相機坐標系: Z 前, X 右, Y 下 -> Z 前, -X 左, -Y 上)
        實際轉換可能依賴於你的硬體設置。

        Args:
            x (float): 物體中心的水平像素座標。
            y (float): 物體中心的垂直像素座標。
            depth (float): 深度值 (公尺)。
            fx, fy, cx, cy (float): 相機內參數。

        Returns:
            np.ndarray: 物體在 IMU 坐標系中的 3D 位置向量 [Z_imu, X_imu, Y_imu]。
        """
        # 相機坐標系中的位置
        X_cam = (x - cx) * depth / fx
        Y_cam = (y - cy) * depth / fy
        Z_cam = depth
        # 轉換到 IMU 坐標系 (這裡假設 Z 軸對齊，X/Y 軸反向)
        # 請根據你的實際 TF 調整此轉換
        return np.array([Z_cam, -X_cam, -Y_cam])

    def _calculate_offset(self, x, y, depth, fx, fy, cx, cy):
        """
        計算物體中心相對於畫面中心的 3D 偏移量 (在相機坐標系中)。

        Args:
            x (float): 物體中心的水平像素座標。
            y (float): 物體中心的垂直像素座標。
            depth (float): 深度值 (公尺)。
            fx, fy, cx, cy (float): 相機內參數。

        Returns:
            np.ndarray: 物體在相機坐標系中的 3D 偏移向量 [X_cam, Y_cam, Z_cam]。
        """
        x_offset = (x - cx) * depth / fx
        y_offset = (y - cy) * depth / fy
        z_offset = depth  # 這裡的 Z 偏移量通常就是物體的深度
        # 返回相機坐標系的偏移量
        return np.array([x_offset, y_offset, z_offset])

    def _calculate_real_offset_2d(self, x, y, depth, fx, fy, cx, cy):
        """
        計算物體中心相對於畫面中心十字架的真實世界水平(X)和垂直(Y)距離差。
        結果是在相機的右手座標系 (+X 右, +Y 下) 中表示，單位為公尺。

        Args:
            x (float): 物體中心的水平像素座標。
            y (float): 物體中心的垂直像素座標。
            depth (float): 深度值 (公尺)。
            fx, fy, cx, cy (float): 相機內參數。

        Returns:
            tuple: (real_x_offset, real_y_offset) in meters.
                   real_x_offset: 水平偏移 (右為正)
                   real_y_offset: 垂直偏移 (下為正)
        """
        real_x_offset = (x - cx) * depth / fx  # 水平偏移 (Camera X-axis)
        real_y_offset = (y - cy) * depth / fy  # 垂直偏移 (Camera Y-axis)
        return (real_x_offset, real_y_offset)
