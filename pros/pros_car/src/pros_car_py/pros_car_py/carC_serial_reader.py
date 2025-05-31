from pros_car_py.env import SERIAL_DEV_DEFAULT, SERIAL_DEV_FORWARD_DEFAULT
from pros_car_py.car_models import *
import rclpy
from rclpy.node import Node
import orjson
from std_msgs.msg import String
from serial import Serial
from rclpy.duration import Duration


class CarCSerialReader(Node):
    def __init__(self):
        super().__init__('car_c_serial_reader')

        serial_port = self.declare_parameter('serial_port', SERIAL_DEV_DEFAULT).value
        serial_port_forward = self.declare_parameter('serial_port_forward', SERIAL_DEV_FORWARD_DEFAULT).value

        self._serial = Serial(serial_port, 115200, timeout=0)
        self._serial_forward = Serial(serial_port_forward, 115200, timeout=0)

        self.publisher = self.create_publisher(
            String,
            DeviceDataTypeEnum.car_C_state,
            10
        )

        self.publisher_forward = self.create_publisher(
            String,
            DeviceDataTypeEnum.car_C_state_front,
            10
        )

        # Create a timer to read from the serial port and publish state every 100 ms
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.timer_forward = self.create_timer(0.01, self.timer_callback_forward)

        self.log_interval = Duration(seconds=1)  # Log every 1 seconds
        current_time = self.get_clock().now()
        self.last_log_time = current_time

    def timer_callback(self):
        if self._serial.in_waiting > 0:
            incoming_data = self._serial.readline()
            current_time = self.get_clock().now()
            if current_time - self.last_log_time >= self.log_interval:
                self.get_logger().info(f'Receive from car esp32: {incoming_data}')
                self.last_log_time = current_time

            try:
                # Assuming the incoming data is already in the required JSON format
                state_msg = String()
                # validation should be customized
                state_data = dict(CarCState(**orjson.loads(incoming_data)))
                state_msg.data = orjson.dumps(
                    dict(DeviceData(type=DeviceDataTypeEnum.car_C_state,
                                    data=state_data))
                ).decode()
                self.publisher.publish(state_msg)
            except orjson.JSONDecodeError as e:
                self.get_logger().error(f'JSON decode error: {e}')

    def timer_callback_forward(self):
        if self._serial_forward.in_waiting > 0:
            incoming_data_forward = self._serial_forward.readline()
            current_time = self.get_clock().now()
            if current_time - self.last_log_time >= self.log_interval:
                self.get_logger().info(f'Receive from car esp32: {incoming_data_forward}')
                self.last_log_time = current_time

            try:
                state_msg = String()
                state_data = dict(CarCState(**orjson.loads(incoming_data_forward)))
                state_msg.data = orjson.dumps(
                    dict(DeviceData(type=DeviceDataTypeEnum.car_C_state,
                                    data=state_data))
                ).decode()
                self.publisher_forward.publish(state_msg)
            except orjson.JSONDecodeError as e:
                self.get_logger().error(f'JSON decode error: {e}')


def main(args=None):
    rclpy.init(args=args)
    car_c_state_publisher = CarCSerialReader()
    rclpy.spin(car_c_state_publisher)
    car_c_state_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
