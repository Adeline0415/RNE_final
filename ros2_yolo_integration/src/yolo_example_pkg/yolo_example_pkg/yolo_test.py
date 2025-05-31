import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import CompressedImage, Image # Import Image for depth/raw color
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory
import torch


class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__("yolo_detection_node")

        # --- Parameters ---
        self.declare_parameter("use_depth", False) # True to process depth, False for color
        self.declare_parameter("use_compressed_color", True) # Only relevant if use_depth is False
        self.declare_parameter("color_topic", "/camera/color/image_raw") # Base topic for color
        self.declare_parameter("depth_topic", "/camera/depth/image_rect_raw") # Topic for depth
        self.declare_parameter("depth_encoding", "16UC1") # Common depth encoding, adjust if needed

        self.use_depth = self.get_parameter("use_depth").get_parameter_value().bool_value
        self.use_compressed_color = self.get_parameter("use_compressed_color").get_parameter_value().bool_value
        self.color_topic_base = self.get_parameter("color_topic").get_parameter_value().string_value
        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.depth_encoding = self.get_parameter("depth_encoding").get_parameter_value().string_value

        # Add parameter callback for dynamic changes
        self.add_on_set_parameters_callback(self.parameters_callback)

        # --- Initialization ---
        self.bridge = CvBridge()

        # Load YOLO model
        model_path = os.path.join(
            get_package_share_directory("yolo_example_pkg"), "models", "yolov8n.pt"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {device}")
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            raise

        # Setup subscription and publisher based on parameters
        self.image_sub = None # Initialize subscriber variable
        self.image_pub = None # Initialize publisher variable
        self.setup_communication()

        self.get_logger().info(f"YOLO Node started. Processing {'Depth' if self.use_depth else 'Color'}.")
        if not self.use_depth:
             self.get_logger().info(f"Color processing using compressed: {self.use_compressed_color}")

    def parameters_callback(self, params):
        """Callback for dynamic parameter changes."""
        needs_reset = False
        for param in params:
            if param.name in ["use_depth", "use_compressed_color", "color_topic", "depth_topic", "depth_encoding"]:
                # Update internal state immediately for logging/checks
                if param.name == "use_depth":
                    self.use_depth = param.value
                elif param.name == "use_compressed_color":
                    self.use_compressed_color = param.value
                elif param.name == "color_topic":
                    self.color_topic_base = param.value
                elif param.name == "depth_topic":
                    self.depth_topic = param.value
                elif param.name == "depth_encoding":
                    self.depth_encoding = param.value
                needs_reset = True

        if needs_reset:
            self.get_logger().info("Parameter change detected, resetting communication.")
            # Destroy existing sub/pub before creating new ones
            if self.image_sub:
                self.destroy_subscription(self.image_sub)
                self.image_sub = None
            if self.image_pub:
                self.destroy_publisher(self.image_pub)
                self.image_pub = None
            # Re-setup communication with potentially updated parameters
            self.setup_communication()
            self.get_logger().info(f"Communication reset. Now processing {'Depth' if self.use_depth else 'Color'}.")

        return SetParametersResult(successful=True)

    def setup_communication(self):
        """Sets up subscriber and publisher based on parameters."""
        if self.use_depth:
            # Subscribe to Depth Image
            self.image_sub = self.create_subscription(
                Image, self.depth_topic, self.depth_image_callback, 10
            )
            # Publish processed depth (visualized as BGR8)
            self.image_pub = self.create_publisher(
                Image, "/yolo/depth_detection", 10
            )
            self.get_logger().info(f"Subscribing to Depth: {self.depth_topic} ({self.depth_encoding})")
            self.get_logger().info(f"Publishing processed Depth to: /yolo/depth_detection (BGR8)")
        else:
            # Subscribe to Color Image (Compressed or Raw)
            if self.use_compressed_color:
                topic = self.color_topic_base + "/compressed"
                self.image_sub = self.create_subscription(
                    CompressedImage, topic, self.compressed_color_callback, 10
                )
                # Publish processed color (Compressed)
                self.image_pub = self.create_publisher(
                    CompressedImage, "/yolo/color_detection/compressed", 10
                )
                self.get_logger().info(f"Subscribing to Compressed Color: {topic}")
                self.get_logger().info(f"Publishing processed Compressed Color to: /yolo/color_detection/compressed")
            else:
                topic = self.color_topic_base
                self.image_sub = self.create_subscription(
                    Image, topic, self.raw_color_callback, 10
                )
                # Publish processed color (Raw BGR8)
                self.image_pub = self.create_publisher(
                    Image, "/yolo/color_detection", 10
                )
                self.get_logger().info(f"Subscribing to Raw Color: {topic}")
                self.get_logger().info(f"Publishing processed Raw Color to: /yolo/color_detection (BGR8)")

    def process_image_for_yolo(self, cv_image_bgr):
        """Common image processing logic using YOLO on a BGR image."""
        # Use YOLO model
        try:
            results = self.model(cv_image_bgr, verbose=False)
        except Exception as e:
            self.get_logger().error(f"Error during YOLO detection: {e}")
            return None

        # Draw Bounding Box
        processed_image = self.draw_bounding_boxes(cv_image_bgr, results)
        return processed_image

    def depth_image_callback(self, msg: Image):
        """Callback for depth images."""
        try:
            # Convert depth image using specified encoding
            cv_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.depth_encoding)
            # Handle potential passthrough case where encoding might not be directly convertible
            # For example, if desired_encoding='passthrough' was needed. Check cv_bridge docs.
        except CvBridgeError as e:
            self.get_logger().error(f"Could not convert depth image: {e}")
            return
        except Exception as e:
             self.get_logger().error(f"Unexpected error converting depth image: {e}")
             return

        # --- Preprocessing Depth for YOLO (Example: Normalize and Convert to BGR) ---
        # This part is crucial and may need significant adjustment based on depth range and desired outcome.
        # Basic normalization to 0-255 for visualization. Handle 0 values (often invalid depth).
        # Replace 0s with a large value (max range) before normalization, or handle separately.
        max_depth = np.max(cv_depth_image) # Find max depth in this specific image, or use a known max range
        if max_depth == 0: max_depth = 1.0 # Avoid division by zero if image is all zeros

        # Normalize to 0-1, then scale to 0-255. Convert NaNs or Infs if they exist.
        cv_depth_image_float = cv_depth_image.astype(np.float32)
        cv_depth_image_float[cv_depth_image_float == 0] = max_depth # Treat 0 as max distance for this normalization
        cv_depth_image_float = np.nan_to_num(cv_depth_image_float, nan=max_depth, posinf=max_depth, neginf=0) # Handle NaN/Inf

        # Normalize (example: linear scaling)
        normalized_depth = cv2.normalize(cv_depth_image_float, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Convert grayscale depth to BGR for YOLO input (results likely poor)
        cv_depth_bgr = cv2.cvtColor(normalized_depth, cv2.COLOR_GRAY2BGR)
        # --- End Preprocessing ---

        # Process the BGR-converted depth image with YOLO
        processed_bgr_image = self.process_image_for_yolo(cv_depth_bgr)

        if processed_bgr_image is not None:
            # Publish the processed BGR image (visualized depth with boxes)
            self.publish_raw_image(processed_bgr_image, msg.header)

    def compressed_color_callback(self, msg: CompressedImage):
        """Callback for compressed color images."""
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Could not convert compressed color image: {e}")
            return

        processed_image = self.process_image_for_yolo(cv_image)
        if processed_image is not None:
            self.publish_compressed_image(processed_image, msg.header)

    def raw_color_callback(self, msg: Image):
        """Callback for raw color images."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Could not convert raw color image: {e}")
            return

        processed_image = self.process_image_for_yolo(cv_image)
        if processed_image is not None:
            self.publish_raw_image(processed_image, msg.header)

    def draw_bounding_boxes(self, image, results):
        """Draws YOLO bounding boxes on the image."""
        # This function remains the same, operating on a BGR image
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

    def publish_compressed_image(self, image, header):
        """Publishes processed image as CompressedImage."""
        if self.image_pub is None: return # Check if publisher exists
        try:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(image)
            compressed_msg.header = header
            self.image_pub.publish(compressed_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Could not publish compressed image: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error publishing compressed image: {e}")


    def publish_raw_image(self, image, header):
        """Publishes processed image as raw Image (BGR8)."""
        if self.image_pub is None: return # Check if publisher exists
        try:
            raw_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            raw_msg.header = header
            self.image_pub.publish(raw_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Could not publish raw image: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error publishing raw image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()