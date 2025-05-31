import subprocess


def stop_and_remove_all_docker_containers():
    try:
        # 停止所有正在運行的容器
        print("Stopping all running Docker containers...")
        subprocess.run(
            "docker stop $(docker ps -aq)",
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # 刪除所有容器（包括已停止的）
        print("Removing all Docker containers...")
        result = subprocess.run(
            "docker rm $(docker ps -aq)",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("All Docker containers have been removed successfully.")
        print("Output:", result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print("Error occurred while stopping or removing Docker containers.")
        print("Error message:", e.stderr.strip())


if __name__ == "__main__":
    stop_and_remove_all_docker_containers()
