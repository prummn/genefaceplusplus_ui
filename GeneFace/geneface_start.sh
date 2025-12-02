#!/bin/bash

CONTAINER_NAME="geneface"
WORKSPACE_DIR="/data/geneface/"

echo "1. 启动 Docker 容器: $CONTAINER_NAME"
docker start $CONTAINER_NAME

echo "2. 进入容器并配置环境..."
docker exec -it $CONTAINER_NAME bash -c "
source ~/.bashrc && 
conda activate pytorch && 
cd $WORKSPACE_DIR && 
export PYTHONPATH=./ && 
export HF_ENDPOINT=https://hf-mirror.com && 
exec bash"

# 如果容器启动失败，脚本会在这里暂停或退出
if [ $? -ne 0 ]; then
    echo "警告：进入容器失败，请手动检查容器状态。"
fi