#!/bin/bash

# Port mapping check
PORT_MAPPING=""
if [ "$1" = "--port" ] && [ -n "$2" ] && [ -n "$3" ]; then
    PORT_MAPPING="-p $2:$3"
    shift 3  # Remove the first three arguments
fi

# 檢查系統架構與作業系統
ARCH=$(uname -m)
OS=$(uname -s)

# 初始化 GPU 相關變數
GPU_FLAGS=""
USE_GPU=false

# 檢查是否為 Linux 並且支援 NVIDIA GPU
if [ "$OS" = "Linux" ]; then
    if [ -f "/etc/nv_tegra_release" ]; then
        GPU_FLAGS="--runtime=nvidia"
        USE_GPU=true
    elif docker info --format '{{json .}}' | grep -q '"Runtimes".*nvidia'; then
        GPU_FLAGS="--gpus all"
        USE_GPU=true
    fi
fi

# 測試 GPU 是否可用
if [ "$USE_GPU" = true ]; then
    echo "Testing Docker run with GPU..."
    docker run --rm $GPU_FLAGS ghcr.io/screamlab/pros_car_docker_image:latest /bin/bash -c "echo GPU test" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "GPU not supported or failed, disabling GPU flags."
        GPU_FLAGS=""
        USE_GPU=false
    fi
fi

echo "Detected OS: $OS, Architecture: $ARCH"
echo "GPU Flags: $GPU_FLAGS"

# 設定適當的 Docker 參數
device_options=""

# 檢查設備並加入 --device 參數
if [ -e /dev/usb_front_wheel ]; then
    device_options+=" --device=/dev/usb_front_wheel"
fi
if [ -e /dev/usb_rear_wheel ]; then
    device_options+=" --device=/dev/usb_rear_wheel"
fi
if [ -e /dev/usb_robot_arm ]; then
    device_options+=" --device=/dev/usb_robot_arm"
fi

# 根據不同架構選擇適當的 Docker 圖像
if [ "$ARCH" = "aarch64" ]; then
    echo "Detected architecture: arm64"
    docker run -it --rm \
        --network compose_my_bridge_network \
        $PORT_MAPPING \
        $device_options \
        --runtime=nvidia \
        --env-file .env \
        -v "$(pwd)/src:/workspaces/src" \
        ghcr.io/screamlab/pros_car_docker_image:latest \
        /bin/bash

elif [ "$ARCH" = "x86_64" ] || ([ "$ARCH" = "arm64" ] && [ "$OS" = "Darwin" ]); then
    echo "Detected architecture: amd64 or macOS arm64"

    if [ "$OS" = "Darwin" ]; then
        echo "Running Docker on macOS (without GPU support)..."
        docker run -it --rm \
            --network compose_my_bridge_network \
            $PORT_MAPPING \
            $device_options \
            --env-file .env \
            -v "$(pwd)/src:/workspaces/src" \
            -v "$(pwd)/screenshots:/workspaces/screenshots" \
            ghcr.io/screamlab/pros_car_docker_image:latest \
            /bin/bash
    else
        echo "Trying to run with GPU support..."
        docker run -it --rm \
            --network compose_my_bridge_network \
            $PORT_MAPPING \
            $GPU_FLAGS \
            $device_options \
            --env-file .env \
            -v "$(pwd)/src:/workspaces/src" \
            -v "$(pwd)/screenshots:/workspaces/screenshots" \
            ghcr.io/screamlab/pros_car_docker_image:latest \
            /bin/bash

        # 如果 GPU 啟動失敗，回退到 CPU 模式
        if [ $? -ne 0 ]; then
            echo "GPU not supported or failed, falling back to CPU mode..."
            docker run -it --rm \
                --network compose_my_bridge_network \
                $PORT_MAPPING \
                --env-file .env \
                $device_options \
                -v "$(pwd)/src:/workspaces/src" \
                -v "$(pwd)/screenshots:/workspaces/screenshots" \
                ghcr.io/screamlab/pros_car_docker_image:latest \
                /bin/bash
        fi
    fi
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi
