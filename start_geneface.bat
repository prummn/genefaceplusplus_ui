@echo off
setlocal enabledelayedexpansion

chcp 65001 >nul
echo ==========================================
echo   GeneFace++ Docker 启动脚本 (Windows)
echo ==========================================

set PROJECT_DIR=%~dp0
set PROJECT_DIR=%PROJECT_DIR:~0,-1%
set CONTAINER_NAME=geneface

echo 项目目录: %PROJECT_DIR%

:: 检查容器是否正在运行
docker ps --format "{{.Names}}" | findstr /x "%CONTAINER_NAME%" >nul 2>&1
if %errorlevel%==0 (
    echo ✓ GeneFace++ 已在运行
    goto :CHECK_SERVICE
)

:: 检查容器是否存在但未运行
docker ps -a --format "{{.Names}}" | findstr /x "%CONTAINER_NAME%" >nul 2>&1
if %errorlevel%==0 (
    echo 启动已存在的容器...
    docker start %CONTAINER_NAME%
    goto :WAIT_SERVICE
)

:: 创建新容器
echo 创建新容器...
docker run -d ^
    --name %CONTAINER_NAME% ^
    -p 7869:7860 ^
    --gpus all ^
    -v "%USERPROFILE%\.cache":/root/.cache ^
    -v "%PROJECT_DIR%\GeneFace":/data/geneface ^
    --restart unless-stopped ^
    genfaceplus:0219 ^
    bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate pytorch && cd /data/geneface && python api_server.py"

:WAIT_SERVICE
echo 等待服务启动...

set /a count=0
:CHECK_LOOP
curl -s http://localhost:7869/health >nul 2>&1
if %errorlevel%==0 (
    echo.
    echo ✓ GeneFace++ 服务已就绪
    echo API 地址: http://localhost:7869
    goto :END
)

set /a count+=1
if %count% GEQ 30 (
    echo.
    echo ✗ 启动超时
    echo 查看日志: docker logs %CONTAINER_NAME%
    echo.
    echo 尝试查看容器状态:
    docker ps -a | findstr %CONTAINER_NAME%
    goto :END
)

<nul set /p=.
timeout /t 1 >nul
goto CHECK_LOOP

:CHECK_SERVICE
curl -s http://localhost:7869/health >nul 2>&1
if %errorlevel%==0 (
    echo API 地址: http://localhost:7869
) else (
    echo 容器运行中但服务未响应，查看日志: docker logs %CONTAINER_NAME%
)

:END
endlocal