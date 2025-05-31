from enum import Enum, auto
from typing import List
import pydantic


class StringEnum(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    def __eq__(self, other):
        if isinstance(other, StringEnum):
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other

        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.value

    def __hash__(self):
        return hash(self.value)


class DeviceDataTypeEnum(StringEnum):
    car_A_state = auto()
    car_A_control = auto()
    car_B_state = auto()
    car_B_control = auto()
    car_C_state = auto()
    car_C_state_front = auto()
    car_C_front_wheel = auto()
    car_C_rear_wheel = auto()
    robot_arm = auto()


class DeviceData(pydantic.BaseModel):
    type: DeviceDataTypeEnum
    data: dict


class CarAState(pydantic.BaseModel):
    vels: List[float] = []
    encoders: List[int] = []
    direction: int


class CarAControl(pydantic.BaseModel):
    target_vel: List[float] = []
    direction: int = 90


class CarBState(pydantic.BaseModel):
    vels: List[float] = []
    encoders: List[int] = []


class CarBControl(pydantic.BaseModel):
    target_vel: List[float] = []


class CarCState(pydantic.BaseModel):
    vels: List[float] = []
    encoders: List[int] = []


class CarCControl(pydantic.BaseModel):
    target_vel: List[float] = []


class TwoWheelAndServoControlSignal(pydantic.BaseModel):
    target_vel: List[float] = []
    direction: int = None


class TwoWheelAndServoState(pydantic.BaseModel):
    motor_count: int = 2
    vels: List[float] = []
    encoders: List[int] = []
    direction: int


class TwoWheelControlSignal(pydantic.BaseModel):
    target_vel: List[float] = []


class TwoWheelState(pydantic.BaseModel):
    motor_count: int = 2
    vels: List[float] = []
    encoders: List[int] = []
