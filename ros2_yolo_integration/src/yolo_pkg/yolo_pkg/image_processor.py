import struct
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge


class ImageProcessor:
    def __init__(self, ros_communicator, load_params):
        self.bridge = CvBridge()
        self.ros_communicator = ros_communicator
        self.latest_valid_depth_image = None  # Cache the last valid image
        self.load_params = load_params

    def _convert_imgmsg_to_cv2(self, img, encoding):
        return self.bridge.imgmsg_to_cv2(img, desired_encoding=encoding)

    def _decode_compressed_depth(self, img_msg):
        """
        Decodes a ROS CompressedImage with format '16UC1; compressedDepth'.

        This function checks if the data starts with the PNG signature.
        If it does, it decodes the PNG directly.
        Otherwise, it assumes a custom header (8 bytes: two floats) is present.
        """
        png_signature = bytes([137, 80, 78, 71, 13, 10, 26, 10])
        data_bytes = bytes(img_msg.data)

        if data_bytes.startswith(png_signature):
            # No header present; decode PNG directly.
            png_data = data_bytes
            depth_quant_a = 1.0  # Default scaling factor
            depth_quant_b = 0.0  # Default offset
        else:
            # If there's a custom header, extract it.
            if len(data_bytes) < 8:
                raise ValueError("Compressed depth message is too short!")
            header = data_bytes[:8]
            depth_quant_a, depth_quant_b = struct.unpack("ff", header)
            png_data = data_bytes[8:]

        np_arr = np.frombuffer(png_data, np.uint8)
        depth_png = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if depth_png is None:
            raise ValueError("Failed to decode PNG depth image")

        # If a header was provided, use it to compute real depth.
        # Otherwise, assume the PNG directly encodes 16UC1 data in millimeters.
        if depth_quant_a != 1.0 or depth_quant_b != 0.0:
            depth_float = np.zeros_like(depth_png, dtype=np.float32)
            valid = depth_png > 0
            depth_float[valid] = depth_quant_a / (
                depth_png[valid].astype(np.float32) - depth_quant_b
            )
        else:
            # Convert from uint16 (assumed in millimeters) to float32 in meters.
            depth_float = depth_png.astype(np.float32) / 1000.0  # Convert mm to meters

        return depth_float

    def _convert_image_from_ros_to_cv(self, img, mode):
        """Converts ROS image to OpenCV format (np.ndarray)."""
        try:
            if img is None:
                print("Received None image message.")
                return None

            if isinstance(img, CompressedImage):
                # Handle compressed RGB or specific compressed depth formats
                if img.format.startswith("16UC1; compressedDepth"):
                    cv_image = self._decode_compressed_depth(
                        img
                    )  # Returns float32 meters
                elif mode == "rgb":
                    # Standard compressed RGB (e.g., jpeg, png)
                    cv_image = self.bridge.compressed_imgmsg_to_cv2(
                        img, desired_encoding="bgr8"
                    )
                else:
                    # Other compressed formats might need specific handling
                    # For now, try generic decoding if not RGB
                    np_arr = np.frombuffer(img.data, np.uint8)
                    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                    # If it's depth, assume millimeters and convert to meters
                    if (
                        mode == "depth"
                        and cv_image is not None
                        and cv_image.dtype == np.uint16
                    ):
                        cv_image = cv_image.astype(np.float32) / 1000.0

            elif isinstance(img, Image):
                # Handle raw Image messages
                if mode == "rgb":
                    encoding = "bgr8"
                    cv_image = self._convert_imgmsg_to_cv2(img, encoding)
                elif mode == "depth":
                    # Common raw depth encodings: 16UC1 (mm), 32FC1 (meters)
                    if img.encoding == "16UC1":
                        cv_image = self._convert_imgmsg_to_cv2(img, "16UC1")
                        # Convert mm to meters
                        cv_image = cv_image.astype(np.float32) / 1000.0
                    elif img.encoding == "32FC1":
                        cv_image = self._convert_imgmsg_to_cv2(
                            img, "32FC1"
                        )  # Already in meters
                    else:
                        # Try passthrough if encoding is unknown but might be compatible
                        cv_image = self._convert_imgmsg_to_cv2(img, "passthrough")
                        # Attempt conversion if it looks like mm
                        if cv_image is not None and cv_image.dtype == np.uint16:
                            cv_image = cv_image.astype(np.float32) / 1000.0
                else:
                    # Fallback for unknown mode
                    cv_image = self._convert_imgmsg_to_cv2(img, "passthrough")
            else:
                print(f"Unsupported image type received: {type(img)}")
                return None  # Return None for unsupported types

            # Final check if conversion resulted in a valid numpy array
            if not isinstance(cv_image, np.ndarray):
                print(f"Conversion failed for image type {type(img)} and mode {mode}.")
                return None

            # Cache the valid image if it's for depth
            if mode == "depth":
                self.latest_valid_depth_image = cv_image

            return cv_image

        except CvBridgeError as e:
            print(f"CvBridge Error converting image: {e}")
            return None
        except ValueError as e:
            print(f"Value Error converting image: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error converting image: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _convert_image_from_cv_to_ros(self, img):
        return self.bridge.cv2_to_compressed_imgmsg(img)

    def get_yolo_target_label(self):
        return self.ros_communicator.get_latest_data("target_label")

    def get_rgb_ros_image(self, img):
        ros_img = self._convert_image_from_cv_to_ros(img)
        return ros_img

    def get_depth_cv_image(self, use_compressed=None):
        """
        Fetch and convert the depth image from ROS to OpenCV format (float32 meters).

        Args:
            use_compressed (bool): If True, fetches the compressed depth image.
                                   If False (default), fetches the raw depth image.

        Returns:
            np.ndarray or None: The depth image in OpenCV format (meters) or None on failure.
        """
        if use_compressed is None:
            use_compressed = self.load_params.get_use_compressed()

        if use_compressed:
            ros_key = "depth_image_compress"
        else:
            ros_key = "depth_image"

        image_msg = self.ros_communicator.get_latest_data(ros_key)

        if image_msg is None:
            print(f"No data received for key: {ros_key}. Trying the other format...")
            # Try the alternative format if the primary one failed
            fallback_key = "depth_image" if use_compressed else "depth_image_compress"
            image_msg = self.ros_communicator.get_latest_data(fallback_key)
            if image_msg is None:
                print(f"No data received for fallback key: {fallback_key} either.")
                # If both fail, return the cached image if available
                if self.latest_valid_depth_image is not None:
                    print("Returning cached depth image.")
                    return self.latest_valid_depth_image
                else:
                    print("No depth image available.")
                    return None

        # Convert the fetched message
        converted_image = self._convert_image_from_ros_to_cv(image_msg, mode="depth")

        # If conversion fails, return the cached image
        if converted_image is None and self.latest_valid_depth_image is not None:
            print("Conversion failed, returning cached depth image.")
            return self.latest_valid_depth_image

        return converted_image

    def get_rgb_cv_image(self):
        image = self.ros_communicator.get_latest_data("rgb_compress")
        return self._convert_image_from_ros_to_cv(image, mode="rgb")
