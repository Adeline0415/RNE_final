import os
import yaml
import subprocess
import signal
import sys


def list_docker_images_from_all_compose():
    """
    從 compose 資料夾中的所有 docker-compose.yml 文件解析出所有的 image 名稱。
    """
    compose_dir = os.path.join("docker", "compose")
    if not os.path.exists(compose_dir):
        print("[錯誤] 找不到 compose 資料夾！")
        print(f"[提示] 當前工作目錄: {os.getcwd()}")
        sys.exit(1)

    images = []
    for root, _, files in os.walk(compose_dir):
        for file in files:
            if file.endswith(".yml") or file.endswith(".yaml"):
                compose_file = os.path.join(root, file)
                try:
                    with open(compose_file, "r") as f:
                        compose_content = yaml.safe_load(f)

                    services = compose_content.get("services", {})
                    for service, config in services.items():
                        image = config.get("image")
                        if image and image not in images:
                            images.append(image)
                except Exception as e:
                    print(f"[警告] 無法解析 {compose_file}: {e}")

    return images


def pull_docker_image(image):
    """
    拉取指定的 Docker 映像。
    """
    print(f"[拉取中] {image} ...")
    try:
        pull_process = subprocess.Popen(
            ["docker", "pull", image], stdout=sys.stdout, stderr=sys.stderr
        )
        pull_process.wait()
        if pull_process.returncode == 0:
            print(f"[成功] 映像檔 {image} 拉取完成！")
        else:
            print(f"[錯誤] 拉取映像檔 {image} 失敗！")
    except KeyboardInterrupt:
        print(f"\n[中斷] 拉取映像檔 {image} 已取消。")
        pull_process.terminate()
        sys.exit(1)


def pull_all_images(images):
    """
    拉取所有的 Docker 映像。
    """
    for image in images:
        pull_docker_image(image)


def display_menu(images):
    """
    顯示選單，讓使用者選擇拉取哪個映像。
    """
    while True:
        print("\n========== Docker 映像選單 ==========")
        for idx, image in enumerate(images, start=1):
            print(f"{idx}. 拉取 {image}")
        print(f"{len(images) + 1}. 全部拉取")
        print("q. 離開程式")
        print("===================================")

        choice = input("請選擇要拉取的映像：").strip().lower()

        if choice.isdigit():
            choice = int(choice)
            if 1 <= choice <= len(images):
                pull_docker_image(images[choice - 1])
            elif choice == len(images) + 1:
                pull_all_images(images)
            else:
                print("[錯誤] 無效的選項，請重新輸入。")
        elif choice == "q":
            print("[訊息] 程式結束。")
            sys.exit(0)
        else:
            print("[錯誤] 無效的選項，請重新輸入。")


def signal_handler(sig, frame):
    """
    捕捉 Ctrl + C (SIGINT)，提示用戶程序已中止。
    """
    print("\n[訊息] 偵測到 Ctrl + C，程式即將結束。")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)

    images = list_docker_images_from_all_compose()
    if not images:
        print("[訊息] 沒有在任何 docker-compose 文件中找到映像。")
        sys.exit(0)

    display_menu(images)


if __name__ == "__main__":
    main()
