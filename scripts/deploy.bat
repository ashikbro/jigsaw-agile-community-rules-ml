@echo off
REM Content Moderation System Deployment Script for Windows
REM This script helps deploy and manage the containerized content moderation system

setlocal enabledelayedexpansion

REM Check if Docker is running
:check_docker
echo [INFO] Checking Docker status...
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker Desktop and try again.
    exit /b 1
)
echo [SUCCESS] Docker is running

REM Check if docker-compose is available
:check_docker_compose
docker-compose --version >nul 2>&1
if errorlevel 1 (
    docker compose version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Docker Compose is not available. Please install Docker Compose.
        exit /b 1
    ) else (
        set DOCKER_COMPOSE=docker compose
    )
) else (
    set DOCKER_COMPOSE=docker-compose
)
echo [SUCCESS] Docker Compose is available: !DOCKER_COMPOSE!

REM Parse command line arguments
if "%1"=="" goto show_help
if "%1"=="help" goto show_help
if "%1"=="--help" goto show_help
if "%1"=="-h" goto show_help
if "%1"=="build" goto build_images
if "%1"=="start" goto start_services
if "%1"=="stop" goto stop_services
if "%1"=="restart" goto restart_services
if "%1"=="status" goto show_status
if "%1"=="logs" goto show_logs
if "%1"=="minimal" goto deploy_minimal
if "%1"=="full" goto deploy_full
if "%1"=="cleanup" goto cleanup
if "%1"=="health" goto check_health

echo [ERROR] Unknown command: %1. Use 'help' for usage information.
exit /b 1

:build_images
echo [INFO] Building Docker images...
!DOCKER_COMPOSE! build --no-cache
if errorlevel 1 (
    echo [ERROR] Failed to build images
    exit /b 1
)
echo [SUCCESS] Images built successfully
goto :eof

:start_services
echo [INFO] Starting services...
!DOCKER_COMPOSE! up -d
if errorlevel 1 (
    echo [ERROR] Failed to start services
    exit /b 1
)
echo [INFO] Waiting for services to be ready...
timeout /t 10 /nobreak >nul
call :check_service_health
goto show_status

:stop_services
echo [INFO] Stopping services...
!DOCKER_COMPOSE! down
echo [SUCCESS] Services stopped
goto :eof

:restart_services
echo [INFO] Restarting services...
!DOCKER_COMPOSE! restart
timeout /t 5 /nobreak >nul
call :check_service_health
goto show_status

:show_logs
if "%2"=="" (
    echo [INFO] Showing logs for all services
    !DOCKER_COMPOSE! logs -f
) else (
    echo [INFO] Showing logs for service: %2
    !DOCKER_COMPOSE! logs -f %2
)
goto :eof

:show_status
echo [INFO] Service status:
!DOCKER_COMPOSE! ps
echo.
echo [INFO] Service URLs:
echo   API Documentation: http://localhost:8000/docs
echo   API Health Check:  http://localhost:8000/health
echo   Streamlit Demo:    http://localhost:8501
echo   Nginx Proxy:       http://localhost:80
echo   Prometheus:        http://localhost:9090
echo   Grafana:           http://localhost:3000 (admin/admin123)
goto :eof

:deploy_minimal
echo [INFO] Deploying minimal setup (API + Streamlit only)...
!DOCKER_COMPOSE! up -d api streamlit
timeout /t 10 /nobreak >nul
call :check_service_health
goto show_status

:deploy_full
echo [INFO] Deploying full setup with monitoring...
!DOCKER_COMPOSE! up -d
timeout /t 15 /nobreak >nul
call :check_service_health
goto show_status

:cleanup
echo [INFO] Cleaning up Docker resources...
!DOCKER_COMPOSE! down -v --remove-orphans
docker system prune -f
echo [SUCCESS] Cleanup completed
goto :eof

:check_service_health
echo [INFO] Checking service health...

REM Check API health
set /a counter=0
:api_health_loop
set /a counter+=1
curl -f http://localhost:8000/health >nul 2>&1
if not errorlevel 1 (
    echo [SUCCESS] API service is healthy
    goto check_streamlit
)
if !counter! geq 30 (
    echo [ERROR] API service failed to start
    goto :eof
)
timeout /t 2 /nobreak >nul
goto api_health_loop

:check_streamlit
set /a counter=0
:streamlit_health_loop
set /a counter+=1
curl -f http://localhost:8501 >nul 2>&1
if not errorlevel 1 (
    echo [SUCCESS] Streamlit service is healthy
    goto :eof
)
if !counter! geq 30 (
    echo [WARNING] Streamlit service may not be ready yet
    goto :eof
)
timeout /t 2 /nobreak >nul
goto streamlit_health_loop

:check_health
call :check_service_health
goto :eof

:show_help
echo Content Moderation System Deployment Script for Windows
echo.
echo Usage: %0 [COMMAND]
echo.
echo Commands:
echo   build          Build Docker images
echo   start          Start all services
echo   stop           Stop all services
echo   restart        Restart all services
echo   status         Show service status and URLs
echo   logs [service] Show logs (optionally for specific service)
echo   minimal        Deploy minimal setup (API + Streamlit)
echo   full           Deploy full setup with monitoring
echo   cleanup        Stop services and clean up resources
echo   health         Check service health
echo   help           Show this help message
echo.
echo Examples:
echo   %0 start                 # Start all services
echo   %0 logs api              # Show API logs
echo   %0 minimal               # Deploy minimal setup
goto :eof