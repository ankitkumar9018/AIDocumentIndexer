# Deployment Guide

Guide for deploying AIDocumentIndexer to production environments.

## Overview

AIDocumentIndexer can be deployed using:
- Docker Compose (recommended for small deployments)
- Kubernetes (recommended for production)
- Cloud platforms (AWS, GCP, Azure)

---

## Docker Compose Production

### Prerequisites

- Docker 24+
- Docker Compose 2+
- 4GB+ RAM
- 20GB+ storage

### System Dependencies in Docker

Ensure your Docker images include Tesseract OCR for scanned PDF support:

```dockerfile
# In your Dockerfile for the backend
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*
```

### 1. Configuration

Create production environment file:

```bash
cp .env.example .env.production
```

Edit `.env.production` with production values:

```bash
# Database
DATABASE_URL=postgresql://user:STRONG_PASSWORD@db:5432/aidocindexer
POSTGRES_PASSWORD=STRONG_PASSWORD

# Redis
REDIS_URL=redis://:REDIS_PASSWORD@redis:6379

# Security
JWT_SECRET=GENERATE_STRONG_SECRET_KEY
ALLOWED_ORIGINS=https://yourdomain.com

# LLM
OPENAI_API_KEY=sk-...

# Production settings
DEBUG=false
LOG_LEVEL=INFO
```

### 2. Deploy

```bash
# Pull latest images
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull

# Start services
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

### 3. Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE support
        proxy_buffering off;
        proxy_read_timeout 86400;
    }
}
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.28+)
- kubectl configured
- Helm 3+

### 1. Create Namespace

```bash
kubectl create namespace aidocindexer
```

### 2. Create Secrets

```bash
kubectl create secret generic aidocindexer-secrets \
  --from-literal=database-url='postgresql://...' \
  --from-literal=jwt-secret='...' \
  --from-literal=openai-api-key='sk-...' \
  -n aidocindexer
```

### 3. Deploy PostgreSQL

```yaml
# postgres-deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: aidocindexer
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: pgvector/pgvector:pg15
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: aidocindexer
        - name: POSTGRES_USER
          value: postgres
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: aidocindexer-secrets
              key: postgres-password
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi
```

### 4. Deploy Redis

```yaml
# redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: aidocindexer
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        command: ["redis-server", "--requirepass", "$(REDIS_PASSWORD)"]
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: aidocindexer-secrets
              key: redis-password
```

### 5. Deploy Backend

```yaml
# backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: aidocindexer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: aidocindexer/backend:latest
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: aidocindexer-secrets
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

### 6. Deploy Frontend

```yaml
# frontend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: aidocindexer
spec:
  replicas: 2
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: frontend
        image: aidocindexer/frontend:latest
        ports:
        - containerPort: 3000
        env:
        - name: NEXT_PUBLIC_API_URL
          value: "https://api.yourdomain.com"
        resources:
          requests:
            cpu: "200m"
            memory: "256Mi"
          limits:
            cpu: "1000m"
            memory: "1Gi"
```

### 7. Create Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: aidocindexer
  namespace: aidocindexer
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - yourdomain.com
    - api.yourdomain.com
    secretName: aidocindexer-tls
  rules:
  - host: yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend
            port:
              number: 3000
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend
            port:
              number: 8000
```

### 8. Apply Configuration

```bash
kubectl apply -f k8s/
```

---

## AWS Deployment

### Using ECS with Fargate

1. Create ECR repositories
2. Push Docker images
3. Create ECS cluster
4. Deploy services with task definitions
5. Configure ALB for load balancing
6. Set up RDS for PostgreSQL
7. Configure ElastiCache for Redis

### Using EKS

Follow the Kubernetes deployment guide with AWS-specific configurations:
- Use EBS for persistent storage
- Use RDS for PostgreSQL
- Use ElastiCache for Redis
- Use ALB Ingress Controller

---

## GCP Deployment

### Using Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/aidocindexer-backend

# Deploy
gcloud run deploy aidocindexer-backend \
  --image gcr.io/PROJECT_ID/aidocindexer-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Using GKE

Follow the Kubernetes deployment guide with GCP-specific configurations.

---

## Monitoring

### Health Checks

```bash
# Backend health
curl https://api.yourdomain.com/health

# Expected response
{
  "status": "healthy",
  "database": "connected",
  "redis": "connected",
  "version": "0.1.0"
}
```

### Prometheus Metrics

Add Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: 'aidocindexer-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: /metrics
```

### Grafana Dashboard

Import the dashboard from `monitoring/grafana/dashboard.json`.

---

## Backup & Recovery

### Database Backup

```bash
# Manual backup
docker compose exec db pg_dump -U postgres aidocindexer > backup.sql

# Scheduled backup with cron
0 2 * * * docker compose exec -T db pg_dump -U postgres aidocindexer > /backups/$(date +\%Y\%m\%d).sql
```

### Restore

```bash
docker compose exec -T db psql -U postgres aidocindexer < backup.sql
```

---

## Security Checklist

- [ ] Use strong passwords for all services
- [ ] Enable HTTPS with valid certificates
- [ ] Configure firewall rules
- [ ] Set up rate limiting
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Rotate secrets regularly
- [ ] Enable database encryption at rest
- [ ] Configure CORS properly
- [ ] Set secure HTTP headers
- [ ] Configure DSPy optimization timeout for production workloads
- [ ] Review BYOK key storage model (client-side localStorage)

---

## Scaling

### Horizontal Scaling

```bash
# Docker Compose
docker compose up -d --scale backend=3

# Kubernetes
kubectl scale deployment backend --replicas=5 -n aidocindexer
```

### Vertical Scaling

Adjust resource limits in deployment configurations.

### Database Scaling

- Add read replicas for PostgreSQL
- Use connection pooling (PgBouncer)
- Implement query caching

---

## Troubleshooting

### Common Issues

**Container won't start:**
```bash
docker compose logs backend
```

**Database connection issues:**
```bash
docker compose exec backend python -c "from backend.db.database import engine; print('OK')"
```

**Memory issues:**
```bash
docker stats
```

### Getting Help

- Check logs: `docker compose logs -f`
- Review Kubernetes events: `kubectl get events -n aidocindexer`
- Open GitHub issue with details
