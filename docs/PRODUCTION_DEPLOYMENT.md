# EMP v1.0 Production Deployment Guide

## Overview
This guide provides comprehensive instructions for deploying the Evolving Market Predator (EMP) v1.0 system in a production environment. The system is now production-ready with enhanced stability, monitoring, and operational capabilities.

## System Architecture

### Core Components
- **Sensory Cortex**: Multi-dimensional market analysis engine
- **Pattern Memory**: Long-term memory for trading contexts
- **Chaos Engine**: Adversarial stress testing
- **Structured Logging**: Comprehensive audit trails
- **Real-time Dashboard**: Live monitoring interface
- **Token Manager**: Automated cTrader token refresh
- **PostgreSQL Integration**: Robust strategy persistence

### Infrastructure Stack
- **Backend**: Python 3.9+ with asyncio
- **Database**: PostgreSQL 14+ with Redis caching
- **Monitoring**: Prometheus + Grafana
- **Containerization**: Docker + Docker Compose
- **Web Interface**: Real-time dashboard with WebSocket support

## Pre-Deployment Checklist

### 1. Environment Setup
```bash
# Verify Python version
python --version  # Should be 3.9+

# Install system dependencies
pip install -r requirements.txt

# Verify Docker installation
docker --version
docker-compose --version
```

### 2. Configuration Files
Ensure all configuration files are properly set:

#### `.env` Configuration
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/emp_prod
REDIS_URL=redis://localhost:6379/0

# cTrader API
CTRADER_CLIENT_ID=your_client_id
CTRADER_CLIENT_SECRET=your_client_secret
CTRADER_REFRESH_TOKEN=your_refresh_token
CTRADER_ACCOUNT_ID=your_account_id

# External APIs
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FRED_API_KEY=your_fred_key
NEWS_API_KEY=your_news_api_key

# Security
JWT_SECRET_KEY=your_jwt_secret_key
ENCRYPTION_KEY=your_encryption_key
```

#### `config.yaml` Production Settings
```yaml
system:
  name: "EMP Production"
  version: "1.0.0"
  environment: "production"
  debug: false

trading:
  max_positions: 10
  max_daily_loss: 1000.0
  risk_per_trade: 0.02
  slippage_tolerance: 0.001

data:
  sources:
    - yahoo_finance
    - alpha_vantage
    - fred
    - newsapi
  
  refresh_interval: 60  # seconds
  cache_duration: 300   # seconds

monitoring:
  health_check_interval: 30
  metrics_retention_days: 30
  log_level: "INFO"
```

## Deployment Steps

### 1. Database Setup

#### PostgreSQL Setup
```bash
# Create database
createdb emp_prod

# Run migrations
psql -d emp_prod -f scripts/migrations/001_initial_schema.sql

# Verify connection
psql -d emp_prod -c "SELECT version();"
```

#### Redis Setup
```bash
# Start Redis
redis-server --daemonize yes

# Test connection
redis-cli ping  # Should return PONG
```

### 2. Docker Deployment

#### Build and Start Services
```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Verify services
docker-compose ps
```

#### Docker Compose Configuration
```yaml
version: '3.8'

services:
  emp-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://emp:emp@postgres:5432/emp_prod
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data

  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: emp_prod
      POSTGRES_USER: emp
      POSTGRES_PASSWORD: emp
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus:/etc/prometheus
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### 3. Security Configuration

#### SSL/TLS Setup
```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Update nginx configuration
cp config/nginx/production.conf /etc/nginx/sites-available/emp
```

#### API Security
```python
# Rate limiting
from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@limiter.limit("100/minute")
@app.get("/api/trades")
async def get_trades(request: Request):
    pass
```

### 4. Monitoring Setup

#### Prometheus Configuration
```yaml
# config/prometheus/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'emp-app'
    static_configs:
      - targets: ['emp-app:8000']
    metrics_path: '/metrics'
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
```

#### Grafana Dashboards
Import the provided dashboards:
- `dashboards/emp-overview.json`
- `dashboards/trading-metrics.json`
- `dashboards/system-health.json`

### 5. Health Checks

#### Application Health Check
```bash
# Check application health
curl http://localhost:8000/health

# Expected response
{
  "status": "healthy",
  "timestamp": "2024-07-20T10:30:00Z",
  "services": {
    "database": "ok",
    "redis": "ok",
    "broker": "ok"
  }
}
```

#### Database Health Check
```bash
# Check database connection
psql -d emp_prod -c "SELECT COUNT(*) FROM strategies;"
```

## Production Features

### 1. Token Management
The system now includes automatic cTrader token refresh:
- Tokens are refreshed 1 hour before expiry
- Refresh tokens are securely stored in Redis
- Failed refreshes trigger alerts

### 2. Pattern Memory
Enhanced memory system for trading contexts:
- Stores historical trading patterns
- Provides similarity-based recall
- Tracks outcomes for learning

### 3. Chaos Testing
Built-in adversarial testing:
- Configurable chaos injection
- Black swan event simulation
- Network failure testing
- Memory pressure testing

### 4. Structured Logging
Comprehensive logging with:
- Correlation IDs for request tracking
- Performance metrics
- Audit trails for trading actions
- Real-time log streaming

### 5. Real-time Dashboard
Live monitoring interface:
- WebSocket-based real-time updates
- Portfolio status
- Active trades
- System health
- Pattern memory insights

## Operational Procedures

### Daily Operations
```bash
# Check system health
./scripts/health_check.sh

# Review logs
./scripts/review_logs.sh

# Check trading performance
./scripts/performance_report.sh
```

### Backup Procedures
```bash
# Database backup
pg_dump emp_prod > backups/emp_prod_$(date +%Y%m%d).sql

# Redis backup
redis-cli BGSAVE

# Configuration backup
tar -czf backups/config_$(date +%Y%m%d).tar.gz config/
```

### Monitoring Alerts
Set up alerts for:
- High CPU/memory usage
- Database connection failures
- Trading losses exceeding thresholds
- Token refresh failures
- Chaos events

## Troubleshooting

### Common Issues

#### 1. Database Connection Issues
```bash
# Check PostgreSQL status
systemctl status postgresql

# Check connection
psql -d emp_prod -c "SELECT 1;"

# Restart if needed
sudo systemctl restart postgresql
```

#### 2. Redis Connection Issues
```bash
# Check Redis status
redis-cli ping

# Check memory usage
redis-cli info memory

# Restart if needed
sudo systemctl restart redis
```

#### 3. Token Refresh Failures
```bash
# Check token status
curl http://localhost:8000/api/token/status

# Manual refresh
curl -X POST http://localhost:8000/api/token/refresh
```

### Performance Tuning

#### Database Optimization
```sql
-- Create indexes for performance
CREATE INDEX idx_strategies_created_at ON strategies(created_at);
CREATE INDEX idx_trades_symbol_timestamp ON trades(symbol, timestamp);
CREATE INDEX idx_events_type_timestamp ON events(type, timestamp);
```

#### Redis Optimization
```bash
# Set memory limits
redis-cli config set maxmemory 1gb
redis-cli config set maxmemory-policy allkeys-lru
```

## Security Best Practices

### 1. Network Security
- Use HTTPS for all external communications
- Implement rate limiting
- Use VPN for database access
- Regular security audits

### 2. Data Security
- Encrypt sensitive data at rest
- Use secure key management
- Regular security updates
- Access logging

### 3. API Security
- Implement OAuth2 authentication
- Use API keys for external services
- Regular token rotation
- Request signing

## Scaling Considerations

### Horizontal Scaling
- Use load balancers for multiple instances
- Implement database read replicas
- Use Redis clustering
- CDN for static assets

### Vertical Scaling
- Monitor resource usage
- Optimize database queries
- Use connection pooling
- Implement caching strategies

## Support and Maintenance

### Regular Maintenance Tasks
- Daily: Health checks, log review
- Weekly: Performance analysis, security updates
- Monthly: Database optimization, backup verification
- Quarterly: Security audit, capacity planning

### Emergency Procedures
- Incident response plan
- Rollback procedures
- Communication protocols
- Recovery procedures

## Contact Information
- **Technical Support**: support@emp-trading.com
- **Emergency Hotline**: +1-800-EMP-HELP
- **Documentation**: https://docs.emp-trading.com
- **Status Page**: https://status.emp-trading.com

---

**Note**: This deployment guide assumes a production environment with proper security measures. Always test in a staging environment before deploying to production.
