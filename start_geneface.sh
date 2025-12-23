#!/usr/bin/env bash
set -e

echo "=========================================="
echo "  GeneFace++ Docker Launcher (Linux)"
echo "=========================================="

# 获取脚本所在目录（绝对路径）
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="geneface"

echo "Project directory: ${PROJECT_DIR}"

# ----------------------------------------
# 检查容器是否正在运行
# ----------------------------------------
RUNNING=0
if docker ps -q -f "name=^${CONTAINER_NAME}$" | grep -q .; then
    RUNNING=1
fi

if [ "$RUNNING" -eq 1 ]; then
    echo "[OK] Container is already running"
    goto_check_service=true
else
    goto_check_service=false
fi

# ----------------------------------------
# 检查是否存在已停止的容器
# ----------------------------------------
if [ "$goto_check_service" = false ]; then
    EXISTS=0
    if docker ps -aq -f "name=^${CONTAINER_NAME}$" | grep -q .; then
        EXISTS=1
    fi

    if [ "$EXISTS" -eq 1 ]; then
        echo "Starting existing container..."
        docker start "${CONTAINER_NAME}"
        sleep 2

        if docker ps -q -f "name=^${CONTAINER_NAME}$" | grep -q .; then
            echo "[OK] Container started"
        else
            echo "Container failed to start, removing and recreating..."
            docker rm -f "${CONTAINER_NAME}"
            EXISTS=0
        fi
    fi

    # ----------------------------------------
    # 创建新容器
    # ----------------------------------------
    if [ "$EXISTS" -eq 0 ]; then
        echo "Creating new container..."
        docker run -d \
            -e PYTHONUNBUFFERED=1 \
            --name "${CONTAINER_NAME}" \
            -p 7869:7860 \
            --gpus all \
            -v "${HOME}/.cache:/root/.cache" \
            -v "${PROJECT_DIR}/GeneFace:/data/geneface" \
            --restart unless-stopped \
            genfaceplus:0219 \
            bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate pytorch && cd /data/geneface && python api_server.py"
    fi
fi

# ----------------------------------------
# 等待服务启动
# ----------------------------------------
echo "Waiting for service..."

count=0
while true; do
    if curl -s http://localhost:7869/health >/dev/null 2>&1; then
        echo
        echo "[OK] GeneFace++ service is ready"
        echo "API URL: http://localhost:7869"
        break
    fi

    count=$((count + 1))
    if [ "$count" -ge 30 ]; then
        echo
        echo "[FAIL] Startup timeout"
        echo "View logs: docker logs ${CONTAINER_NAME}"
        docker ps -a -f "name=${CONTAINER_NAME}"
        break
    fi

    printf "."
    sleep 1
done

# ----------------------------------------
# 如果容器已运行，检查服务状态
# ----------------------------------------
if [ "$goto_check_service" = true ]; then
    if curl -s http://localhost:7869/health >/dev/null 2>&1; then
        echo "API URL: http://localhost:7869"
    else
        echo "Container running but service not responding"
        echo "View logs: docker logs ${CONTAINER_NAME}"
    fi
fi
