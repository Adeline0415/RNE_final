import urwid
from pros_car_py.base_mode import BaseMode
import threading
import time


class VehicleMode(BaseMode):
    def enter(self):
        text = urwid.Text("Vehicle Mode\nPress 'q' to return to main menu.")
        filler = urwid.Filler(text, valign="top")

        self.app.loop.widget = filler
        self.app.loop.unhandled_input = self.handle_input

    def handle_input(self, key):
        if key == "q":
            self.app.car_controller.manual_control(key)
            self.app.main_menu()
        else:
            self.app.car_controller.manual_control(key)


class ArmMode(BaseMode):
    submodes = ["0", "1", "2", "3", "4"]

    def enter(self):
        self.app.horizontal_select(self.submodes, self.handle_submode_select)

    def handle_submode_select(self, submode):
        def on_key(key):
            self.app.arm_controller.manual_control(int(submode), key)

        self.show_submode_screen(
            message=f"Arm Mode: Submode {submode}\nPress 'q' to go back.", on_key=on_key
        )


class CraneMode(BaseMode):
    submodes = ["0", "1", "2", "3", "4", "5", "6", "99"]

    def enter(self):
        self.app.horizontal_select(self.submodes, self.handle_submode_select)

    def handle_submode_select(self, submode):
        def on_key(key):
            self.app.crane_controller.manual_control(int(submode), key)

        self.show_submode_screen(
            message=f"Crane Mode: Submode {submode}\nPress 'q' to go back.",
            on_key=on_key,
        )


class AutoNavMode(BaseMode):
    submodes = ["manual_auto_nav", "target_auto_nav", "custom_nav"]

    def enter(self):
        self.app.horizontal_select(self.submodes, self.handle_submode_select)

    def handle_submode_select(self, submode):
        def on_key(key):
            self.app.car_controller.auto_control(submode, key)
            if key == "q":
                self.app.car_controller.auto_control(submode, key=key)

        self.show_submode_screen(
            message=f"AutoNav Mode: Submode {submode}\nPress 'q' to go back.",
            on_key=on_key,
        )


class AutoArmMode(BaseMode):
    submodes = ["auto_arm_human"]

    def enter(self):
        self.app.horizontal_select(self.submodes, self.handle_submode_select)

    def handle_submode_select(self, submode):
        def on_key(key):
            self.app.arm_controller.auto_control(mode=submode, key=key)
            if key == "q":
                self.app.arm_controller.auto_control(mode=submode, key=key)

        self.show_submode_screen(
            message=f"AutoArm Mode: Submode {submode}\nPress 'q' to go back.",
            on_key=on_key,
        )
