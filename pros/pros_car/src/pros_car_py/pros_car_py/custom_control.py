
class CustomControl:
    def __init__(self, car_controller, arm_controller):
        self.car_controller = car_controller
        self.arm_controller = arm_controller
        
    def manual_control(self, key):
        self.car_controller.manual_control(key)
        self.arm_controller.manual_control(key)
