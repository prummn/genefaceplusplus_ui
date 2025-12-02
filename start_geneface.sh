#!/bin/bash
# start_geneface.sh - 启动 GeneFace++ Docker 服务

PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
CONTAINER_NAME="geneface"

echo "=========================================="
echo "  GeneFace++ Docker 启动脚本"
echo "=========================================="
echo "项目目录: $PROJECT_DIR"

# 检查容器状态
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "✓ GeneFace++ 已在运行"
    exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "启动已存在的容器..."
    docker start $CONTAINER_NAME
else
    echo "创建新容器..."
    docker run -d \
        --name $CONTAINER_NAME \
        -p 7869:7860 \
        --gpus all \
        -v ~/.cache:/root/.cache \
        -v ${PROJECT_DIR}/GeneFace:/data/geneface \
        --restart unless-stopped \
        genfaceplus:0219 \
        python /data/geneface/api_server.py
fi

# 等待服务就绪
echo "等待服务启动..."
for i in {1..30}; do
    if curl -s http://localhost:7869/health > /dev/null 2>&1; then
        echo "✓ GeneFace++ 服务已就绪"
        echo "  API 地址: http://localhost:7869"
        exit 0
    fi
    sleep 1
    echo -n "."
done

echo ""
echo "✗ 服务启动超时，请检查日志: docker logs $CONTAINER_NAME"
exit 1