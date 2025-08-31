# Docker Deployment Guide

This guide explains how to deploy the Content Moderation System using Docker containers for easy deployment, scaling, and management.

## 🐳 Overview

The system is containerized with the following services:

- **API Server**: FastAPI-based REST API for content moderation
- **Streamlit Demo**: Interactive web interface for testing
- **Nginx**: Reverse proxy and load balancer
- **Redis**: Caching and rate limiting (optional)
- **Prometheus**: Metrics collection (optional)
- **Grafana**: Monitoring dashboard (optional)

## 📋 Prerequisites

- Docker Desktop or Docker Engine (20.10+)
- Docker Compose (2.0+)
- At least 4GB RAM available
- Ports 80, 8000, 8501 available

## 🚀 Quick Start

### Option 1: Minimal Setup (Recommended for testing)

```bash
# Linux/macOS
./deploy.sh minimal

# Windows
deploy.bat minimal
```

### Option 2: Full Setup (With monitoring)

```bash
# Linux/macOS
./deploy.sh full

# Windows
deploy.bat full
```

### Option 3: Manual Docker Compose

```bash
# Build images
docker-compose build

# Start minimal services
docker-compose up -d api streamlit

# Or start all services
docker-compose up -d
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file to customize deployment:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Streamlit Configuration
STREAMLIT_PORT=8501

# Redis Configuration (optional)
REDIS_URL=redis://redis:6379/0

# Security
API_SECRET_KEY=your-secret-key-here
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8501

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=admin123
```

### Volume Mounts

The following directories are mounted as volumes:

- `./models:/app/models:ro` - Model files (read-only)
- `./logs:/app/logs` - Application logs
- `./outputs:/app/outputs` - Generated outputs

## 🌐 Service URLs

After deployment, access the services at:

| Service | URL | Description |
|---------|-----|-------------|
| API Docs | http://localhost:8000/docs | Interactive API documentation |
| API Health | http://localhost:8000/health | Health check endpoint |
| Streamlit | http://localhost:8501 | Interactive demo interface |
| Nginx Proxy | http://localhost:80 | Load balancer (routes to API) |
| Prometheus | http://localhost:9090 | Metrics collection |
| Grafana | http://localhost:3000 | Monitoring dashboard |

## 📊 Monitoring Setup

### Prometheus Metrics

The API server exposes metrics at `/metrics` endpoint:

- Request count and duration
- Model prediction latency
- Error rates
- System resource usage

### Grafana Dashboards

Default login: `admin` / `admin123`

Pre-configured dashboards include:
- API Performance
- Model Accuracy Trends
- System Resource Usage
- Error Rate Analysis

## 🛠 Management Commands

### Using Deployment Scripts

```bash
# Check service status
./deploy.sh status

# View logs
./deploy.sh logs
./deploy.sh logs api  # Specific service

# Restart services
./deploy.sh restart

# Stop services
./deploy.sh stop

# Clean up everything
./deploy.sh cleanup
```

### Using Docker Compose Directly

```bash
# View running services
docker-compose ps

# View logs
docker-compose logs -f
docker-compose logs -f api

# Scale services
docker-compose up -d --scale api=3

# Update and restart
docker-compose pull
docker-compose up -d

# Stop and remove
docker-compose down

# Remove volumes too
docker-compose down -v
```

## 🔒 Security Considerations

### Production Deployment

1. **Use HTTPS**: Configure SSL certificates in nginx
2. **Secure Redis**: Enable authentication and use secure connections
3. **API Authentication**: Implement proper JWT token validation
4. **Network Security**: Use Docker networks and firewall rules
5. **Resource Limits**: Set memory and CPU limits for containers

### SSL Configuration

To enable HTTPS:

1. Place SSL certificates in `./ssl/` directory
2. Uncomment HTTPS server block in `nginx.conf`
3. Update docker-compose.yml to mount SSL certificates

## 🐛 Troubleshooting

### Common Issues

**Port conflicts:**
```bash
# Check what's using the port
netstat -tulpn | grep :8000

# Kill process or change port in docker-compose.yml
```

**Memory issues:**
```bash
# Check Docker memory usage
docker stats

# Increase Docker Desktop memory limit
# Or add memory limits to services
```

**Model loading errors:**
```bash
# Check if models directory exists and has correct permissions
ls -la ./models/

# Check API logs
docker-compose logs api
```

**Service health checks failing:**
```bash
# Check service logs
docker-compose logs api

# Test health endpoint manually
curl http://localhost:8000/health
```

### Debug Mode

Enable debug logging:

```yaml
# In docker-compose.yml
services:
  api:
    environment:
      - LOG_LEVEL=DEBUG
      - PYTHONUNBUFFERED=1
```

## 📈 Performance Tuning

### Scaling

```bash
# Scale API service
docker-compose up -d --scale api=3

# Use load balancer
# Nginx will automatically distribute requests
```

### Resource Limits

```yaml
# In docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

### Caching

Enable Redis caching:

```yaml
# Uncomment redis service in docker-compose.yml
# Update API to use Redis for caching predictions
```

## 🔄 Updates and Maintenance

### Updating the System

```bash
# Pull latest changes
git pull

# Rebuild images
docker-compose build --no-cache

# Restart services
docker-compose up -d
```

### Backup and Restore

```bash
# Backup volumes
docker run --rm -v content-moderation_redis_data:/data -v $(pwd):/backup alpine tar czf /backup/redis_backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v content-moderation_redis_data:/data -v $(pwd):/backup alpine tar xzf /backup/redis_backup.tar.gz -C /data
```

## 📝 Logs and Monitoring

### Log Locations

- Application logs: `./logs/`
- Docker logs: `docker-compose logs`
- Nginx logs: Inside nginx container at `/var/log/nginx/`

### Log Rotation

Configure log rotation in docker-compose.yml:

```yaml
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## 🤝 Contributing

To contribute to the Docker setup:

1. Test changes locally with `docker-compose up --build`
2. Update documentation if needed
3. Test both minimal and full deployments
4. Ensure all health checks pass

## 📞 Support

For deployment issues:

1. Check the troubleshooting section above
2. Review service logs: `docker-compose logs`
3. Verify system requirements are met
4. Check Docker and Docker Compose versions

---

**Note**: This deployment is optimized for development and testing. For production use, additional security hardening and performance optimization may be required.