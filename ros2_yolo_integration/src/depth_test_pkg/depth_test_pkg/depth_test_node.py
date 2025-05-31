#!/usr/bin/env python3
# filepath: /home/user/workspace/pros/ros2_yolo_integration/src/depth_test_pkg/depth_test_pkg/depth_test_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32

class DepthCenterPixelNode(Node):
    def __init__(self):
        super().__init__('depth_center_pixel_node')

        # Declare parameters
        self.declare_parameter('depth_topic', '/unity_camera_1/depth')
        self.declare_parameter('publish_center_value', True)

        # Get parameters
        depth_topic = self.get_parameter('depth_topic').value
        publish_center_value = self.get_parameter('publish_center_value').value

        # Create a subscription to the depth image topic
        self.subscription = self.create_subscription(
            Image,
            depth_topic,
            self.depth_image_callback,
            10)

        # Publisher for center pixel value (optional)
        if publish_center_value:
            self.publisher = self.create_publisher(
                Float32,
                'depth_center_value',
                10)
        else:
            self.publisher = None

        # Initialize the CvBridge
        self.bridge = CvBridge()

        self.get_logger().info(f'Depth Center Pixel Node started')
        self.get_logger().info(f'Subscribing to topic: {depth_topic}')

    def depth_image_callback(self, msg):
        print("dsads")
        try:
            # Convert ROS Image message to OpenCV image
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # Get image dimensions
            height, width = depth_image.shape

            # Calculate center coordinates
            center_y = height // 2
            center_x = width // 2

            # Get center pixel value
            center_value = float(depth_image[center_y, center_x])

            # Log the center pixel value
            self.get_logger().info(f'Center pixel ({center_x}, {center_y}) value: {center_value:.3f}m')

            # Publish center value if configured to do so
            if self.publisher:
                value_msg = Float32()
                value_msg.data = center_value
                self.publisher.publish(value_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    depth_center_pixel_node = DepthCenterPixelNode()
    rclpy.spin(depth_center_pixel_node)
    depth_center_pixel_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()