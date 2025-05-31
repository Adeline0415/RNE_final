# mode_app.py
import urwid
from pros_car_py.mode_manager import (
    VehicleMode,
    ArmMode,
    CraneMode,
    AutoNavMode,
    AutoArmMode,
)

MODES_REGISTRY = {
    "mode_vehicle": ("Control Vehicle", VehicleMode),
    "mode_arm": ("Manual Arm Control", ArmMode),
    "mode_crane": ("Manual Crane Control", CraneMode),
    "mode_auto_nav": ("Auto Navigation", AutoNavMode),
    "mode_auto_arm": ("Automatic Arm Mode", AutoArmMode),
}


class ModeApp:
    def __init__(
        self, car_controller, arm_controller, custom_control, crane_controller
    ):
        self.car_controller = car_controller
        self.arm_controller = arm_controller
        self.custom_control = custom_control
        self.crane_controller = crane_controller

        self.palette = [("reversed", "standout", "")]
        self.loop = urwid.MainLoop(None, palette=self.palette)

        self.current_mode = None

    def main(self):
        self.main_menu()
        self.loop.run()

    def main_menu(self):
        menu_items = [urwid.Text("Main Menu:")]

        for mode_name, (label, mode_class) in MODES_REGISTRY.items():
            button = urwid.Button(
                label, on_press=lambda _, m=mode_name: self.switch_mode(m)
            )
            menu_items.append(urwid.AttrMap(button, None, focus_map="reversed"))

        exit_button = urwid.Button("Exit", on_press=lambda _: self.exit_program())
        menu_items.append(urwid.AttrMap(exit_button, None, focus_map="reversed"))

        menu_list = urwid.ListBox(urwid.SimpleFocusListWalker(menu_items))
        self.loop.widget = menu_list
        # self.loop.unhandled_input = None

    def switch_mode(self, mode_name):
        mode_info = MODES_REGISTRY.get(mode_name)
        if not mode_info:
            print(f"Mode '{mode_name}' not found.")
            return

        if self.current_mode:
            self.current_mode.exit()

        _, ModeClass = mode_info
        self.current_mode = ModeClass(self)
        self.current_mode.enter()

    def horizontal_select(self, options, on_select):
        index = [0]

        def render():
            items = [
                urwid.Text(("reversed", f"[{opt}]") if i == index[0] else f" {opt} ")
                for i, opt in enumerate(options)
            ]
            return urwid.Columns(items, dividechars=1)

        def key_handler(key):
            if key == "left":
                index[0] = (index[0] - 1) % len(options)
                self.loop.widget = urwid.Filler(render(), valign="top")
            elif key == "right":
                index[0] = (index[0] + 1) % len(options)
                self.loop.widget = urwid.Filler(render(), valign="top")
            elif key == "enter":
                on_select(options[index[0]])
            elif key == "q":
                self.main_menu()

        self.loop.widget = urwid.Filler(render(), valign="top")
        self.loop.unhandled_input = key_handler

    def exit_program(self):
        print("Exiting program.")
        raise urwid.ExitMainLoop()
