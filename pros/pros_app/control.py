import os
import sys
import signal
import subprocess
import argparse

# List of shell scripts to choose from
scripts = [
    "./slam.sh",
    "./slam_unity.sh",
    "./slam_ydlidar.sh",
    "./store_map.sh",
    "./localization.sh",
    "./localization_unity.sh",
    "./localization_ydlidar.sh",
    "./camera_astra.sh",
    "./camera_dabai.sh",
    "./rosbridge_server.sh",
    "./gps.sh",
    "./camera_calibration_unity.sh",
]


# Keep track of child process PIDs
child_pids = []


def clear_screen():
    """Clear console screen (Windows or Unix)."""
    os.system("cls" if os.name == "nt" else "clear")


def show_menu():
    """Display the main menu."""
    clear_screen()
    print("Choose a script to run:")
    for i, script in enumerate(scripts, start=1):
        print(f"{i}. {script}")
    print("s. Show running processes")
    print("d. Shutdown all child processes")
    print("q. Quit")


def show_running_processes():
    """Display running PIDs and their commands."""
    global child_pids
    if not child_pids:
        print("No child processes are currently running.")
        return

    print("Running processes:")
    alive_pids = []
    for pid in child_pids:
        # 檢查此 PID 是否仍存活
        try:
            os.kill(pid, 0)  # 成功則存活, 失敗會丟 OSError
        except OSError:
            # 該 PID 已不存在，跳過
            continue

        # 取得該 PID 的指令 (僅在類 Unix 下可行)
        # Windows 下可能需要其他方式或第三方套件 (如 psutil)
        cmd = subprocess.getoutput(f"ps -p {pid} -o args=")
        print(f"PID: {pid} - Command: {cmd}")
        alive_pids.append(pid)

    # 更新 child_pids: 移除已死掉的 PID
    child_pids = alive_pids
    if not alive_pids:
        print("No child processes are currently running.")


def shutdown_all_children():
    """Send SIGINT to all child PIDs and clear the list."""
    global child_pids
    if not child_pids:
        print("No child processes are running.")
        return

    print("Shutting down all child processes...")
    for pid in child_pids:
        # 先檢查是否還存活
        try:
            os.kill(pid, 0)
        except OSError:
            continue  # 該 PID 不存在了，跳過

        print(f"Terminating process: {pid}")
        try:
            os.kill(pid, signal.SIGINT)
        except OSError as e:
            print(f"[警告] 無法終止 PID={pid}: {e}")
        # 可以等待該子行程結束，但若它很快就結束了 wait() 會很快
        # 如果該 PID 已經死掉，wait 會出錯，可用 try-except
        try:
            pid_wait, _ = os.waitpid(pid, 0)  # 等它結束
        except ChildProcessError:
            pass

    child_pids.clear()


def run_script(script, print_logs=True):
    """
    Run the specified script in the background.
    If print_logs=False, redirect stdout/stderr to /dev/null.
    Then wait for user input:
      - 'q': kill the process
      - 'b': return to menu (process stays running in background if still alive)
    """
    if print_logs:
        print(f"Running {script} with logs...")
        p = subprocess.Popen(["/bin/bash", script])
    else:
        print(f"Running {script} without logs...")
        p = subprocess.Popen(
            ["/bin/bash", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    # Record the PID
    script_pid = p.pid
    child_pids.append(script_pid)
    print(f"PID of {script}: {script_pid}")
    print(
        "Press 'b' to go back to menu (keep process running), or 'q' to quit and terminate the process."
    )

    while True:
        user_input = input().strip().lower()
        if user_input == "q":
            print(f"Terminating {script}: {script_pid}...")
            # Send SIGINT
            try:
                os.kill(script_pid, signal.SIGINT)
            except OSError as e:
                print(f"[警告] 無法終止 PID={script_pid}: {e}")
            # 等待結束
            try:
                os.waitpid(script_pid, 0)
            except ChildProcessError:
                pass

            # Remove from child_pids
            if script_pid in child_pids:
                child_pids.remove(script_pid)
            break

        elif user_input == "b":
            print(
                f"Going back to the menu. The process {script_pid} will continue running in background."
            )
            break
        else:
            print("[提示] 請輸入 'b' 或 'q'。")


def check_and_pull_docker_image(script_name):
    """
    檢查腳本中是否有需要的 Docker 映像，若不存在則執行 docker pull 並顯示進度。
    """
    with open(script_name, "r") as script_file:
        script_content = script_file.read()

    # 提取 docker run 或 docker pull 中的映像名稱
    images = []
    for line in script_content.splitlines():
        if "docker run" in line or "docker pull" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part in ["docker", "run", "pull"] and i + 1 < len(parts):
                    images.append(parts[i + 1])

    for image in images:
        print(f"[檢查] Docker 映像檔: {image}")
        result = subprocess.run(
            ["docker", "images", "-q", image],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if not result.stdout.strip():
            print(f"[缺失] 映像檔 {image} 不存在，開始拉取...")
            pull_process = subprocess.Popen(
                ["docker", "pull", image], stdout=sys.stdout, stderr=sys.stderr
            )
            try:
                pull_process.wait()  # 等待拉取完成
            except KeyboardInterrupt:
                print("\n[訊息] 拉取過程中偵測到 Ctrl + C，正在取消...")
                pull_process.terminate()
                stop_all_docker_containers()
                sys.exit(1)
            if pull_process.returncode == 0:
                print(f"[成功] 映像檔 {image} 拉取完成！")
            else:
                print(f"[錯誤] 拉取映像檔 {image} 失敗！")
        else:
            print(f"[成功] 映像檔 {image} 已存在！")


def main():

    parser = argparse.ArgumentParser(description="A Python menu to run .sh scripts.")
    parser.add_argument(
        "-s", "--silent", action="store_true", help="Run in silent mode (no logs)."
    )
    args = parser.parse_args()

    silent_mode = args.silent

    while True:

        show_menu()
        if silent_mode:
            print("===== Running in silent mode =====")

        choice = input("Enter your choice: ").strip().lower()

        if choice == "q":
            print("Shutting down all child processes...")
            shutdown_all_children()
            print("Exiting...")
            break

        elif choice == "d":
            shutdown_all_children()
            print("Press any key to continue...")
            input()
        elif choice == "s":
            show_running_processes()
            print("Press any key to continue...")
            input()
        elif choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(scripts):
                selected_script = scripts[idx - 1]
                # 若非 silent_mode，或剛好是 store_map.sh，則印 logs
                if not silent_mode or selected_script == "./store_map.sh":
                    run_script(selected_script, print_logs=True)
                else:
                    run_script(selected_script, print_logs=False)
            else:
                print("Invalid choice. Please try again.")
                print("Press any key to continue...")
                input()
        else:
            print("Invalid choice. Please try again.")
            print("Press any key to continue...")
            input()


if __name__ == "__main__":
    # 掛載 Ctrl+C handler，用來清理子行程
    def signal_handler(sig, frame):
        print("\n[訊息] 偵測到 Ctrl + C，開始終止所有子行程並退出程式...")
        shutdown_all_children()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    main()
