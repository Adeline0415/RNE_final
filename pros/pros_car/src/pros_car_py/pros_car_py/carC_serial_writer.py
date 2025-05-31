from pros_car_py.env import SERIAL_DEV_DEFAULT, SERIAL_DEV_FORWARD_DEFAULT
from pros_car_py.car_models import *
import rclpy
from rclpy.node import Node
import orjson
from std_msgs.msg import String
from serial import Serial


class CarCControlSubscriber(Node):
    def __init__(self):
        super().__init__("car_c_control_subscriber")
        self.subscription = self.create_subscription(
            String,
            DeviceDataTypeEnum.car_C_rear_wheel,  # topic name
            self.listener_callback,
            10,
        )
        serial_port = self.declare_parameter("serial_port", SERIAL_DEV_DEFAULT).value

        self.subscription  # prevent unused variable warning
        self._serial = Serial(serial_port, 115200, timeout=0)
        # ------------------------------------------------------------------
        self.subscription_forward = self.create_subscription(
            String,
            DeviceDataTypeEnum.car_C_front_wheel,  # topic name
            self.listener_callback_forward,
            10,
        )
        serial_port_forward = self.declare_parameter(
            "serial_port_forward", SERIAL_DEV_FORWARD_DEFAULT
        ).value

        self.subscription_forward  # prevent unused variable warning
        self._serial_forward = Serial(serial_port_forward, 115200, timeout=0)

    # -------------------------------------------------------------------
    def listener_callback(self, msg):
        try:
            control_data = orjson.loads(msg.data)
            # TODO use more clear method to write
            # TODO divide serial data and data validation
            if control_data.get("type") == DeviceDataTypeEnum.car_C_rear_wheel:
                self.process_control_data(CarCControl, control_data.get("data", {}))
        except orjson.JSONDecodeError as e:
            self.get_logger().error("JSON decode error: {}".format(e))
        except KeyError as e:
            self.get_logger().error("Missing key in JSON data: {}".format(e))

    def listener_callback_forward(self, msg):
        try:
            control_data_forward = orjson.loads(msg.data)
            # TODO use more clear method to write
            # TODO divide serial data and data validation
            if control_data_forward.get("type") == DeviceDataTypeEnum.car_C_front_wheel:
                self.process_control_data_forward(
                    CarCControl, control_data_forward.get("data", {})
                )
        except orjson.JSONDecodeError as e:
            self.get_logger().error("JSON decode error: {}".format(e))
        except KeyError as e:
            self.get_logger().error("Missing key in JSON data: {}".format(e))

    def process_control_data(self, type_cls, data: dict):
        # Process the control data as needed
        # direction = data.get('direction')
        # target_vel = data.get('target_vel')
        # TODO should be customized
        control_signal = type_cls(**data)

        self._serial.write(
            orjson.dumps(dict(control_signal), option=orjson.OPT_APPEND_NEWLINE)
        )
        # self.get_logger().info(f'Received type:{type(data)} data: {data}')
        self.get_logger().info(f"Received {control_signal}")

    def process_control_data_forward(self, type_cls, data: dict):
        # Process the control data as needed
        # direction = data.get('direction')
        # target_vel = data.get('target_vel')
        # TODO should be customized
        control_signal_forward = type_cls(**data)

        self._serial_forward.write(
            orjson.dumps(dict(control_signal_forward), option=orjson.OPT_APPEND_NEWLINE)
        )
        # self.get_logger().info(f'Received type:{type(data)} data: {data}')
        self.get_logger().info(f"Received {control_signal_forward}")


def main(args=None):
    rclpy.init(args=args)
    car_c_control_subscriber = CarCControlSubscriber()
    rclpy.spin(car_c_control_subscriber)
    car_c_control_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
