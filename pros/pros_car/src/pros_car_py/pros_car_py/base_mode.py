# base_mode.py
import urwid
class BaseMode:
    """所有模式的基底類別，定義通用的介面。"""
    def __init__(self, app):
        """
        app: 這裡會是 ModeApp 的實例，裡面含有
             - loop (urwid.MainLoop)
             - 其他 Controller (car_controller, arm_controller, ...)
             - 也有一些輔助方法 (如 horizontal_select、main_menu 等)
        """
        self.app = app

    def clean_terminal(self):
        """
        清空終端畫面（印出 ANSI 控制碼）。
        """
        print("\033[2J\033[H", end="")

    def show_submode_screen(self, message, on_key=None, on_quit=None):
        """
        顯示子模式畫面，並處理鍵盤事件。
        - message: 要顯示的文字
        - on_key(key): 當使用者按下非 'q' 的按鍵時要做的事
        - on_quit(): 當使用者按下 'q' 時要做的事 (若不指定則預設呼叫 self.enter())
        """
        text = urwid.Text(message)
        filler = urwid.Filler(text, valign='top')
        self.app.loop.widget = filler

        def input_handler(key):
            if on_key:
                on_key(key)
            if key == 'q' and on_quit:
                self.clean_terminal()
                on_quit()
            elif key == 'q' and not on_quit:
                self.clean_terminal()
                self.enter()

        self.app.loop.unhandled_input = input_handler

    def enter(self):
        """進入模式時要顯示的畫面、初始化等。"""
        self.clean_terminal()

    def handle_input(self, key):
        """處理鍵盤輸入。"""
        pass

    def exit(self):
        """若模式需要釋放或關閉什麼資源，可以在這裡做。"""
        pass
