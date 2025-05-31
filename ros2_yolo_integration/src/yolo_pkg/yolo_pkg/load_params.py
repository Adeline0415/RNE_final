import os
import yaml
import torch
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO


class LoadParams:
    def __init__(self, package_name="yolo_pkg"):
        self.package_name = package_name
        self.params = {}
        self._load_parameters()
        self._yolo_detection_model = None
        self._yolo_segmentation_model = None

    def _load_parameters(self):
        try:
            package_path = os.path.join(
                get_package_share_directory(self.package_name), "config"
            )
            yaml_file_path = os.path.join(package_path, "yolo_params.yaml")

            with open(yaml_file_path, "r") as file:
                self.params = yaml.safe_load(file)

            print(f"Parameters loaded from {yaml_file_path}")

        except Exception as e:
            print(f"Warning: Failed to load parameters: {e}")
            self.params = {
                "yolo": {
                    "conf": 0.6,
                    "obb_model": "yolov8n.pt",
                    "seg_model": "yolo11n-seg.pt",
                },
                "image": {"use_compressed": True, "screenshot_fps": 5},
            }
            print("Using default parameters.")

    def get_params(self):
        return self.params

    def get_yolo_params(self):
        return self.params.get("yolo", {})

    def get_image_params(self):
        return self.params.get("image", {})

    def get_confidence_threshold(self):
        return self.get_yolo_params().get("conf", 0.6)

    def get_use_compressed(self):
        return self.get_image_params().get("use_compressed", True)

    def get_screenshot_fps(self):
        return self.get_image_params().get("screenshot_fps", 5)

    def get_detection_model(self):
        if self._yolo_detection_model is None:
            model_name = self.get_yolo_params().get("obb_model", "yolov8n.pt")

            model_path = os.path.join(
                get_package_share_directory(self.package_name), "models", model_name
            )
            print()
            print("*" * 10)
            print(f"Loading detection model: {model_name}")

            model = YOLO(model_path)

            if torch.cuda.is_available():
                print("CUDA is available. Using GPU.")
                model.to("cuda")
            else:
                print("CUDA is not available. Using CPU.")
                model.to("cpu")
            print("*" * 10)

            self._yolo_detection_model = model

        return self._yolo_detection_model

    def get_segmentation_model(self):
        if self._yolo_segmentation_model is None:
            model_name = self.get_yolo_params().get("seg_model", "yolo11n-seg.pt")

            model_path = os.path.join(
                get_package_share_directory(self.package_name), "models", model_name
            )

            print()
            print("*" * 10)
            print(f"Loading segmentation model: {model_name}")

            model = YOLO(model_path)

            if torch.cuda.is_available():
                print("CUDA is available. Using GPU.")
                model.to("cuda")
            else:
                print("CUDA is not available. Using CPU.")
                model.to("cpu")
            print("*" * 10)

            self._yolo_segmentation_model = model

        return self._yolo_segmentation_model
