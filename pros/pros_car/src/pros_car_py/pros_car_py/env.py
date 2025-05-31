import os

SERIAL_DEV_DEFAULT = os.getenv(
    "SERIAL_DEV_DEFAULT", "/dev/usb_rear_wheel"
)  # replace with your default value
SERIAL_DEV_FORWARD_DEFAULT = os.getenv(
    "SERIAL_DEV_FORWARD_DEFAULT", "/dev/usb_front_wheel"
)  # replace with your default value
ARM_SERIAL_PORT_DEFAULT = os.getenv("ARM_SERIAL_PORT", "/dev/usb_robot_arm")
WHEEL_SPEED = os.getenv("WHEEL_SPEED")
ARM_ROTATE_ANGLE = os.getenv("ARM_ROTATE_ANGLE")
