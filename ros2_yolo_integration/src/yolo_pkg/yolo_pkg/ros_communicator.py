from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Imu, Image
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PointStamped


class RosCommunicator(Node):
    def __init__(self):
        super().__init__("RosCommunicator")

        # --- Subscriber and Publisher Initialization ---
        self.subscriber_dict = {
            "rgb_compress": {
                "topic": "/camera/image/compressed",
                "msg_type": CompressedImage,
                "callback": self._image_sub_callback,
            },
            "imu": {
                "topic": "/imu/data",
                "msg_type": Imu,
                "callback": self._imu_sub_callback,
            },
            "depth_image_compress": {
                "topic": "/camera/depth/compressed",
                "msg_type": CompressedImage,
                "callback": self._depth_image_compress_sub_callback,
            },
            "depth_image": {
                "topic": "/camera/depth/image_raw",
                "msg_type": Image,
                "callback": self._depth_image_sub_callback,
            },
            "target_label": {
                "topic": "/target_label",
                "msg_type": String,
                "callback": self._target_label_sub_callback,
            },
        }

        self.publisher_dict = {
            "yolo_image": {
                "topic": "/yolo/detection/compressed",
                "msg_type": CompressedImage,
            },
            "point": {
                "topic": "/yolo/detection/position",
                "msg_type": PointStamped,
            },
            "object_offset": {
                "topic": "/yolo/object/offset",
                "msg_type": String,
            },
            "detection_status": {
                "topic": "/yolo/detection/status",
                "msg_type": Bool,
            },
        }

        # Initialize Subscribers
        self.latest_data = {}
        for key, sub in self.subscriber_dict.items():
            self.latest_data[key] = None
            msg_type = sub["msg_type"]
            topic = sub["topic"]
            callback = sub["callback"]
            self.create_subscription(msg_type, topic, callback, 10)

        # Initialize Publishers
        self.publisher_instances = {}
        for key, pub in self.publisher_dict.items():
            self.publisher_instances[key] = self.create_publisher(
                pub["msg_type"], pub["topic"], 10
            )

    # --- Callback Functions ---
    def _image_sub_callback(self, msg):
        self.latest_data["rgb_compress"] = msg

    def _imu_sub_callback(self, msg):
        self.latest_data["imu"] = msg

    def _depth_image_sub_callback(self, msg):
        self.latest_data["depth_image"] = msg

    def _depth_image_compress_sub_callback(self, msg):
        self.latest_data["depth_image_compress"] = msg

    def _target_label_sub_callback(self, msg):
        self.latest_data["target_label"] = msg

    # --- Getter Functions ---
    def get_latest_data(self, key):
        return self.latest_data.get(key)

    # --- Publisher Functions ---
    def publish_data(self, key, data):
        try:
            publisher = self.publisher_instances.get(key)
            if publisher:
                publisher.publish(data)
            else:
                self.get_logger().error(f"No publisher found for key: {key}")
        except Exception as e:
            self.get_logger().error(f"Could not publish data for {key}: {e}")
