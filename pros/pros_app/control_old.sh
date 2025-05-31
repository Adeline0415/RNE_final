#!/bin/bash

SLAM_SCRIPT="./slam_unity.sh"
STORE_MAP_SCRIPT="./store_map.sh"
LOCALIZATION_SCRIPT="./localization_unity.sh"

SLAM_COMPOSE="./docker/compose/docker-compose_slam_unity.yml"
LOCALIZATION_COMPOSE="./docker/compose/docker-compose_localization_unity.yml"
NAVIGATION_COMPOSE="./docker/compose/docker-compose_navigation_unity.yml"
STORE_MAP_COMPOSE="./docker/compose/docker-compose_store_map.yml"
RPLIDAR_COMPOSE="./docker/compose/docker-compose_rplidar_unity.yml"

# Determine which docker compose command to use
if command -v docker-compose &> /dev/null
then
    DOCKER_COMPOSE_COMMAND="docker-compose"
elif docker compose version &> /dev/null
then
    DOCKER_COMPOSE_COMMAND="docker compose"
else
    echo "Neither 'docker-compose' nor 'docker compose' is installed. Please install Docker Compose."
    exit 1
fi

cleanup() {
    echo "正在快速關閉所有 Docker 容器..."
    $DOCKER_COMPOSE_COMMAND -f "$RPLIDAR_COMPOSE" down --timeout 0 > /dev/null 2>&1
    $DOCKER_COMPOSE_COMMAND -f "$SLAM_COMPOSE" down --timeout 0 > /dev/null 2>&1
    $DOCKER_COMPOSE_COMMAND -f "$STORE_MAP_COMPOSE" down --timeout 0 > /dev/null 2>&1
    $DOCKER_COMPOSE_COMMAND -f "$LOCALIZATION_COMPOSE" down --timeout 0 > /dev/null 2>&1
    $DOCKER_COMPOSE_COMMAND -f "$NAVIGATION_COMPOSE" down --timeout 0 > /dev/null 2>&1
    # # 強制關閉所有名稱中包含 `scripts-navigation` 的容器
    # echo "強制關閉所有名稱中包含 'scripts-navigation' 的容器..."
    # containers=$(docker ps -q --filter "name=scripts-navigation")
    # if [ -n "$containers" ]; then
    #     docker stop $containers
    #     echo "容器已強制關閉：$containers"
    # fi
}

stop_container() {
    local compose_file="$1"
    echo "正在關閉 $compose_file 中的容器..."
    $DOCKER_COMPOSE_COMMAND -f "$compose_file" down --timeout 0 > /dev/null 2>&1

    # 再次檢查並強制關閉特定容器
    echo "檢查並強制關閉 $compose_file 中的容器..."
    project_name=$(get_compose_project_name "$compose_file")
    containers=$(docker ps -q --filter "label=com.docker.compose.project=$project_name")
    if [ -n "$containers" ]; then
        docker kill $containers
        echo "$compose_file 容器已強制關閉：$containers"
    fi
}

# 獲取項目名稱（與 Docker 項目名稱保持一致）
get_compose_project_name() {
    local compose_file="$1"
    basename "$compose_file" .yml
}

# 檢查項目中的容器是否存在
is_project_running() {
    local compose_file="$1"
    $DOCKER_COMPOSE_COMMAND -f "$compose_file" ps --services --filter "status=running" | grep -q .
}

trap cleanup SIGINT

run_script_with_monitor() {
    local script_path="$1"
    local compose_file="$2"
    local auto_exit="$3"
    local allow_stop="$4"
    local cleanup_on_exit="$5"

    setsid bash "$script_path" > /dev/null 2>&1 &
    SCRIPT_PID=$!
    SCRIPT_GROUP_PID=$(ps -o pgid= -p $SCRIPT_PID | grep -o '[0-9]*')

    if [ "$auto_exit" == "true" ]; then
        wait "$SCRIPT_PID"
    else
        echo "按下 'q' 返回選單。"

        while true; do
            read -rsn1 -t 1 key
            if [[ $key == 'q' ]]; then
                echo "正在返回選單..."
                if [ "$allow_stop" == "true" ]; then
                    # 停止對應的 Docker 容器
                    stop_container "$compose_file"
                    # 向進程組發送 SIGTERM 信號，殺掉子腳本及其子進程
                    kill -TERM -$SCRIPT_GROUP_PID 2>/dev/null
                    wait "$SCRIPT_PID" 2>/dev/null
                fi
                if [ "$cleanup_on_exit" == "true" ]; then
                    cleanup
                fi
                break
            fi
            # 檢查子腳本是否已經退出
            if ! kill -0 "$SCRIPT_PID" 2>/dev/null; then
                echo "子腳本已退出。"
                break
            fi
        done
    fi
}

while true; do
    clear
    echo "請選擇要執行的腳本："
    echo "1. slam_unity.sh"
    echo "2. store_map.sh （需要在選項1執行的情況下使用）"
    echo "3. localization_unity.sh"
    echo "4. 關閉所有容器"
    echo "5. 退出"
    read -p "請輸入選項 (1-4): " choice

    case $choice in
        1)
            # 關閉選項 3 對應的容器
            # stop_container "$LOCALIZATION_COMPOSE"
            cleanup
            echo "啟動 slam 中..."
            # 啟動 slam_unity.sh，不允許按下 'q' 停止腳本
            run_script_with_monitor "$SLAM_SCRIPT" "$SLAM_COMPOSE" "false" "false" "false"
            ;;
        2)
            # 檢查選項 1 是否正在運行
            echo "正在檢查選項 1 是否正在運行..."
            if is_project_running "$SLAM_COMPOSE"; then
                echo "選項 1 正在運行，啟動 store_map.sh。"
                # 啟動 store_map.sh，不需要人為控制，按下 'q' 不會停止腳本
                run_script_with_monitor "$STORE_MAP_SCRIPT" "$STORE_MAP_COMPOSE" "false" "false" "false"
            else
                echo "請先啟動選項1（slam_unity.sh）。"
                sleep 2
            fi
            ;;
        3)
            # # 關閉選項 1 和選項 2 對應的容器
            # echo "關閉選項 1 的容器..."
            # stop_container "$SLAM_COMPOSE"
            # echo "關閉選項 2 的容器..."
            # stop_container "$STORE_MAP_COMPOSE"
            cleanup
            echo "啟動 localization 中..."
            # 檢查是否有當前的 localization_unity.sh 正在運行
            # if is_project_running "$LOCALIZATION_COMPOSE"; then
            #     echo "localization_unity.sh 的容器已在運行，重新啟動..."
            #     # stop_container "$LOCALIZATION_COMPOSE"
            #     cleanup
            #     sleep 2  # 等待容器完全停止
            # fi

            # 啟動 localization_unity.sh，允許按下 'q' 停止腳本並調用 cleanup
            run_script_with_monitor "$LOCALIZATION_SCRIPT" "$LOCALIZATION_COMPOSE" "false" "true" "true"
            ;;
        4)
            echo "關閉所有容器中..."
            cleanup
            ;;
        5)
            echo "退出程序。"
            # 調用清理函數，停止所有 Docker 容器
            cleanup
            exit 0
            ;;
        *)
            echo "無效的選項，請輸入1-4。"
            sleep 2
            ;;
    esac
done
