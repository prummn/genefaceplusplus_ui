@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo   GeneFace++ Docker Launcher (Windows)
echo ==========================================

set PROJECT_DIR=%~dp0
set PROJECT_DIR=%PROJECT_DIR:~0,-1%
set CONTAINER_NAME=geneface

echo Project directory: %PROJECT_DIR%

:: Check if container is running
set RUNNING=0
for /f %%i in ('docker ps -q -f "name=^%CONTAINER_NAME%$" 2^>nul') do set RUNNING=1

if %RUNNING%==1 (
    echo [OK] Container is already running
    goto :CHECK_SERVICE
)

:: Check if container exists but stopped
set EXISTS=0
for /f %%i in ('docker ps -aq -f "name=^%CONTAINER_NAME%$" 2^>nul') do set EXISTS=1

if %EXISTS%==1 (
    echo Starting existing container...
    docker start %CONTAINER_NAME%
    timeout /t 2 >nul

    :: Verify it actually started
    set STARTED=0
    for /f %%i in ('docker ps -q -f "name=^%CONTAINER_NAME%$" 2^>nul') do set STARTED=1

    if !STARTED!==1 (
        echo [OK] Container started
        goto :WAIT_SERVICE
    )

    echo Container failed to start, removing and recreating...
    docker rm -f %CONTAINER_NAME%
)

:: Create new container
echo Creating new container...
docker run -d ^
    -e PYTHONUNBUFFERED=1 ^
    --name %CONTAINER_NAME% ^
    -p 7869:7860 ^
    --gpus all ^
    -v "%USERPROFILE%\.cache":/root/.cache ^
    -v "%PROJECT_DIR%\GeneFace":/data/geneface ^
    --restart unless-stopped ^
    genefaceplus:0219 ^
    bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate pytorch && cd /data/geneface && python api_server.py"

:WAIT_SERVICE
echo Waiting for service...

set /a count=0
:CHECK_LOOP
curl -s http://localhost:7869/health >nul 2>&1
if %errorlevel%==0 (
    echo.
    echo [OK] GeneFace++ service is ready
    echo API URL: http://localhost:7869
    goto :END
)

set /a count+=1
if %count% GEQ 30 (
    echo.
    echo [FAIL] Startup timeout
    echo View logs: docker logs %CONTAINER_NAME%
    docker ps -a -f "name=%CONTAINER_NAME%"
    goto :END
)

<nul set /p=.
timeout /t 1 >nul
goto CHECK_LOOP

:CHECK_SERVICE
curl -s http://localhost:7869/health >nul 2>&1
if %errorlevel%==0 (
    echo API URL: http://localhost:7869
) else (
    echo Container running but service not responding
    echo View logs: docker logs %CONTAINER_NAME%
)

:END
endlocal