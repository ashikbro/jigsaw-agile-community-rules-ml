#!/bin/bash

# Content Moderation System Deployment Script
# This script helps deploy and manage the containerized content moderation system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if docker-compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        if ! docker compose version &> /dev/null; then
            print_error "Docker Compose is not available. Please install Docker Compose."
            exit 1
        else
            DOCKER_COMPOSE="docker compose"
        fi
    else
        DOCKER_COMPOSE="docker-compose"
    fi
    print_success "Docker Compose is available: $DOCKER_COMPOSE"
}

# Function to build images
build_images() {
    print_status "Building Docker images..."
    $DOCKER_COMPOSE build --no-cache
    print_success "Images built successfully"
}

# Function to start services
start_services() {
    print_status "Starting services..."
    $DOCKER_COMPOSE up -d
    
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    check_service_health
}

# Function to check service health
check_service_health() {
    print_status "Checking service health..."
    
    # Check API health
    for i in {1..30}; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            print_success "API service is healthy"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "API service failed to start"
            return 1
        fi
        sleep 2
    done
    
    # Check Streamlit
    for i in {1..30}; do
        if curl -f http://localhost:8501 > /dev/null 2>&1; then
            print_success "Streamlit service is healthy"
            break
        fi
        if [ $i -eq 30 ]; then
            print_warning "Streamlit service may not be ready yet"
        fi
        sleep 2
    done
}

# Function to stop services
stop_services() {
    print_status "Stopping services..."
    $DOCKER_COMPOSE down
    print_success "Services stopped"
}

# Function to restart services
restart_services() {
    print_status "Restarting services..."
    $DOCKER_COMPOSE restart
    sleep 5
    check_service_health
}

# Function to show logs
show_logs() {
    if [ -n "$1" ]; then
        print_status "Showing logs for service: $1"
        $DOCKER_COMPOSE logs -f "$1"
    else
        print_status "Showing logs for all services"
        $DOCKER_COMPOSE logs -f
    fi
}

# Function to show service status
show_status() {
    print_status "Service status:"
    $DOCKER_COMPOSE ps
    
    echo
    print_status "Service URLs:"
    echo "  API Documentation: http://localhost:8000/docs"
    echo "  API Health Check:  http://localhost:8000/health"
    echo "  Streamlit Demo:    http://localhost:8501"
    echo "  Nginx Proxy:       http://localhost:80"
    echo "  Prometheus:        http://localhost:9090"
    echo "  Grafana:           http://localhost:3000 (admin/admin123)"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    $DOCKER_COMPOSE down -v --remove-orphans
    docker system prune -f
    print_success "Cleanup completed"
}

# Function to deploy minimal setup (API + Streamlit only)
deploy_minimal() {
    print_status "Deploying minimal setup (API + Streamlit only)..."
    $DOCKER_COMPOSE up -d api streamlit
    sleep 10
    check_service_health
}

# Function to deploy full setup
deploy_full() {
    print_status "Deploying full setup with monitoring..."
    $DOCKER_COMPOSE up -d
    sleep 15
    check_service_health
}

# Function to show help
show_help() {
    echo "Content Moderation System Deployment Script"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  build          Build Docker images"
    echo "  start          Start all services"
    echo "  stop           Stop all services"
    echo "  restart        Restart all services"
    echo "  status         Show service status and URLs"
    echo "  logs [service] Show logs (optionally for specific service)"
    echo "  minimal        Deploy minimal setup (API + Streamlit)"
    echo "  full           Deploy full setup with monitoring"
    echo "  cleanup        Stop services and clean up resources"
    echo "  health         Check service health"
    echo "  help           Show this help message"
    echo
    echo "Examples:"
    echo "  $0 start                 # Start all services"
    echo "  $0 logs api              # Show API logs"
    echo "  $0 minimal               # Deploy minimal setup"
}

# Main script logic
case "$1" in
    build)
        check_docker
        check_docker_compose
        build_images
        ;;
    start)
        check_docker
        check_docker_compose
        start_services
        show_status
        ;;
    stop)
        check_docker_compose
        stop_services
        ;;
    restart)
        check_docker
        check_docker_compose
        restart_services
        show_status
        ;;
    status)
        check_docker_compose
        show_status
        ;;
    logs)
        check_docker_compose
        show_logs "$2"
        ;;
    minimal)
        check_docker
        check_docker_compose
        deploy_minimal
        show_status
        ;;
    full)
        check_docker
        check_docker_compose
        deploy_full
        show_status
        ;;
    cleanup)
        check_docker_compose
        cleanup
        ;;
    health)
        check_service_health
        ;;
    help|--help|-h)
        show_help
        ;;
    "")
        print_error "No command specified. Use 'help' for usage information."
        exit 1
        ;;
    *)
        print_error "Unknown command: $1. Use 'help' for usage information."
        exit 1
        ;;
esac