import cv2
import numpy as np
import os
from datetime import datetime
import time
import json
import colorsys


class BoundingBoxVisualizer:
    def __init__(self, image_processor, yolo_bounding_box, ros_communicator):
        self.ros_communicator = ros_communicator
        self.image_processor = image_processor
        self.yolo_bounding_box = yolo_bounding_box
        self.last_screenshot_time = 0
        self.screenshot_interval = 1 / 5

        self.label_colors = {}
        self._hue_next = 0.0

    def _get_color_for_label(self, label):

        if label not in self.label_colors:
            golden_ratio = 0.618033988749895
            self._hue_next = (self._hue_next + golden_ratio) % 1.0
            h = self._hue_next
            s = 0.9 + 0.1 * np.random.rand()
            v = 0.9 + 0.1 * np.random.rand()
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            color = (int(b * 255), int(g * 255), int(r * 255))
            self.label_colors[label] = color
        return self.label_colors[label]

    def _draw_object_offsets(self, image, offsets_3d_json):
        if not offsets_3d_json:
            return False
        try:
            offsets_data = json.loads(offsets_3d_json)
            if not offsets_data:
                return False
            detected_objects = self.yolo_bounding_box.get_tags_and_boxes()
            if not detected_objects:
                return False
            label_to_box = {obj["label"]: obj["box"] for obj in detected_objects}
            for offset_obj in offsets_data:
                label = offset_obj["label"]
                offset = offset_obj.get("offset_flu")
                if not offset or label not in label_to_box:
                    continue
                x1, y1, x2, y2 = label_to_box[label]
                color = self._get_color_for_label(label)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(image, (center_x, center_y), 5, color, -1)
                offset_text = f"F:{offset[0]}m L:{offset[1]}m U:{offset[2]}m"
                text_y = y2 + 20
                cv2.putText(
                    image,
                    offset_text,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
            return True
        except json.JSONDecodeError:
            print("Error decoding JSON from offsets_3d")
            return False
        except Exception as e:
            print(f"Error drawing offset info: {e}")
            return False

    def draw_offset_info(self, offsets_3d_json):
        image = self.image_processor.get_rgb_cv_image()
        if image is None:
            print("Error: No image received from image_processor")
            return
        self._draw_object_offsets(image, offsets_3d_json)
        try:
            ros_image = self.image_processor.get_rgb_ros_image(image)
            self.ros_communicator.publish_data("yolo_image", ros_image)
        except Exception as e:
            print(f"Failed to publish image with offsets: {e}")

    def save_fps_screenshot(self, save_folder="fps_screenshots"):
        image = self.image_processor.get_rgb_cv_image()
        if image is None:
            print("Error: No image received from image_processor")
            return
        current_time = time.time()
        if current_time - self.last_screenshot_time < self.screenshot_interval:
            return
        self.last_screenshot_time = current_time
        os.makedirs(save_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = os.path.join(save_folder, f"full_frame_{timestamp}.png")
        cv2.imwrite(path, image)

    def draw_bounding_boxes(
        self,
        draw_crosshair=False,
        screenshot=False,
        save_folder="screenshots",
        segmentation_status=False,
        bounding_status=False,
        offsets_3d_json=None,
    ):
        image = self.image_processor.get_rgb_cv_image()
        if image is None:
            print("Error: No image received from image_processor")
            return
        if segmentation_status:
            segmentation_objects = self.yolo_bounding_box.get_segmentation_data()
            for obj in segmentation_objects:
                mask = obj["mask"]
                label = obj["label"]
                x1, y1, x2, y2 = obj["box"]
                color = self._get_color_for_label(label)
                mask_colored = np.zeros_like(image, dtype=np.uint8)
                mask_colored[mask > 0] = color
                image = cv2.addWeighted(image, 1, mask_colored, 0.5, 0)
                if bounding_status:
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        image,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
        if bounding_status:
            detected_objects = self.yolo_bounding_box.get_tags_and_boxes()
            for obj in detected_objects:
                label = obj["label"]
                confidence = obj["confidence"]
                x1, y1, x2, y2 = obj["box"]
                color = self._get_color_for_label(label)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                text = f"{label} ({confidence:.2f})"
                cv2.putText(
                    image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
                if screenshot:
                    cropped = image[y1:y2, x1:x2]
                    if cropped.size > 0:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        p = os.path.join(save_folder, f"{label}_{ts}.png")
                        cv2.imwrite(p, cropped)
        if draw_crosshair:
            self._draw_crosshair(image)
        if offsets_3d_json:
            self._draw_object_offsets(image, offsets_3d_json)
        if not isinstance(image, np.ndarray):
            print("Processed image is not a valid numpy array.")
            return
        try:
            ros_image = self.image_processor.get_rgb_ros_image(image)
            self.ros_communicator.publish_data("yolo_image", ros_image)
        except Exception as e:
            print(f"Failed to publish final image: {e}")

    def _draw_crosshair(self, image):
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.line(image, (cx - 20, cy), (cx + 20, cy), (0, 0, 255), 2)
        cv2.line(image, (cx, cy - 20), (cx, cy + 20), (0, 0, 255), 2)
