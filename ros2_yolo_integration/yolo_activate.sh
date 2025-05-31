#!/bin/bash

PORT_MAPPING=""
if [ "$1" = "--port" ] && [ -n "$2" ] && [ -n "$3" ]; then
    PORT_MAPPING="-p $2:$3"
    shift 3  # Remove the first three arguments
fi

# 檢查系統架構
ARCH=$(uname -m)
OS=$(uname -s)

# 適用於 x86_64 或 macOS 上的 arm64
if [ "$ARCH" = "aarch64" ]; then
    echo "Detected architecture: arm64"
    docker run -it --rm \
        --network compose_my_bridge_network \
        $PORT_MAPPING \
        --runtime=nvidia \
        --env-file .env \
        -v "$(pwd)/src:/workspace/src" \
        registry.screamtrumpet.csie.ncku.edu.tw/screamlab/ros2_yolo_opencv_image:latest \
        /bin/bash
elif [ "$ARCH" = "x86_64" ] || ([ "$ARCH" = "arm64" ] && [ "$OS" = "Darwin" ]); then
    echo "Detected architecture: amd64 or macOS arm64"
    if [ "$OS" = "Darwin" ]; then
        # macOS 版本（不使用 --gpus all）
        docker run -it --rm \
            --network compose_my_bridge_network \
            $PORT_MAPPING \
            --env-file .env \
            -v "$(pwd)/src:/workspaces/src" \
            -v "$(pwd)/screenshots:/workspaces/screenshots" \
            -v "$(pwd)/fps_screenshots:/workspaces/fps_screenshots" \
            registry.screamtrumpet.csie.ncku.edu.tw/screamlab/pros_cameraapi:0.0.2 \
            /bin/bash
    else
        echo "Trying to run with GPU support..."
        docker run -it --rm \
            --network compose_my_bridge_network \
            $PORT_MAPPING \
            --gpus all \
            --env-file .env \
            -v "$(pwd)/src:/workspaces/src" \
            -v "$(pwd)/screenshots:/workspaces/screenshots" \
            -v "$(pwd)/fps_screenshots:/workspaces/fps_screenshots" \
            registry.screamtrumpet.csie.ncku.edu.tw/screamlab/pros_cameraapi:0.0.2 \
            /bin/bash

        # 如果上一個指令失敗，則改用不帶 GPU 的版本
        if [ $? -ne 0 ]; then
            echo "GPU not supported or failed, falling back to CPU mode..."
            docker run -it --rm \
                --network compose_my_bridge_network \
                $PORT_MAPPING \
                --env-file .env \
                -v "$(pwd)/src:/workspaces/src" \
                -v "$(pwd)/screenshots:/workspaces/screenshots" \
                -v "$(pwd)/fps_screenshots:/workspaces/fps_screenshots" \
                registry.screamtrumpet.csie.ncku.edu.tw/screamlab/pros_cameraapi:0.0.2 \
                /bin/bash
        fi
    fi
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi
