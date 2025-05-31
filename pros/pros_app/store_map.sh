#!/bin/bash

# 確定使用哪個 docker compose 指令
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

# 清理 docker-compose
cleanup() {
    echo "Shutting down docker-compose services..."
    $DOCKER_COMPOSE_COMMAND -f ./docker/compose/docker-compose_store_map.yml down --timeout 0 > /dev/null 2>&1
}

trap 'cleanup; exit 0' SIGINT

# 重試計數器
MAX_RETRIES=5
retry_count=0
success_flag=false

while [ $retry_count -lt $MAX_RETRIES ]; do
    echo "Starting docker-compose services... (Attempt $((retry_count+1))/$MAX_RETRIES)"
    $DOCKER_COMPOSE_COMMAND -f ./docker/compose/docker-compose_store_map.yml up -d > /dev/null 2>&1

    echo "Monitoring logs for errors..."

    # 監控 docker-compose 的 logs
    while read -r line; do
        if echo "$line" | grep -q "Failed to spin map subscription"; then
            echo "Error detected in logs: Failed to spin map subscription"
            cleanup
            echo "Retrying..."
            sleep 2  # 等待短時間後重試
            break  # 偵測到錯誤，離開 log 監控，重新開始過程
        fi

        if echo "$line" | grep -q "success"; then
            echo "Success detected."
            success_flag=true
            break  # 偵測到成功，離開 log 監控
        fi
    done < <($DOCKER_COMPOSE_COMMAND -f ./docker/compose/docker-compose_store_map.yml logs -f)

    # 如果偵測到成功，則跳出主循環
    if [ "$success_flag" = true ]; then
        echo "儲存成功"
        cleanup
        exit 0
    fi

    # 增加重試計數
    retry_count=$((retry_count+1))
done

# 超過最大重試次數
echo "儲存失敗，已超過最大重試次數 ($MAX_RETRIES)."
cleanup
exit 1
