# Deployment Guide

## Overview

This guide covers deploying the EHR Timeline Triage system in different environments.

**⚠️ CRITICAL**: This system is a research prototype only. Do not deploy for clinical use.

## Deployment Options

### 1. Local Development (Recommended for Testing)

#### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- Git

#### Setup Steps

**Backend:**
```bash
# Clone repository
git clone <repository-url>
cd ehr-timeline-triage

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Generate synthetic data
python -m ehrtriage.scripts.generate_synthetic

# Train models (optional, for demonstration)
python -m ehrtriage.scripts.train_all

# Start API server
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
# Navigate to web directory
cd web

# Install dependencies
npm install

# Start development server
npm run dev
```

Access:
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

---

### 2. Docker Deployment (Recommended for Production-like Testing)

#### Prerequisites
- Docker
- Docker Compose

#### Setup Steps

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Access:
- Frontend: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

#### Data and Models

To use pre-trained models in Docker:

```bash
# Generate data and train models locally first
python -m ehrtriage.scripts.generate_synthetic
python -m ehrtriage.scripts.train_all

# Then start Docker with volumes mounted
docker-compose up
```

---

### 3. Cloud Deployment (Research Environments Only)

#### AWS Deployment

**Using EC2:**

1. Launch EC2 instance (t3.medium or larger recommended)
2. Install Docker and Docker Compose
3. Clone repository
4. Run `docker-compose up -d`
5. Configure security groups:
   - Port 8000 for API
   - Port 3000 for Web
   - Use HTTPS in production

**Using ECS:**

1. Create ECS cluster
2. Build and push Docker images to ECR
3. Create task definitions for API and Web services
4. Deploy services with Application Load Balancer

**Environment Variables:**
```bash
# .env file
PYTHONUNBUFFERED=1
NODE_ENV=production
NEXT_PUBLIC_API_URL=https://api.your-domain.com
```

#### Azure Deployment

**Using Azure Container Instances:**

```bash
# Build images
docker-compose build

# Tag images
docker tag ehr-triage-api:latest yourregistry.azurecr.io/ehr-triage-api
docker tag ehr-triage-web:latest yourregistry.azurecr.io/ehr-triage-web

# Push to Azure Container Registry
docker push yourregistry.azurecr.io/ehr-triage-api
docker push yourregistry.azurecr.io/ehr-triage-web

# Deploy container group
az container create \
  --resource-group <resource-group> \
  --name ehr-triage \
  --image yourregistry.azurecr.io/ehr-triage-api \
  --dns-name-label ehr-triage-api \
  --ports 8000
```

#### Google Cloud Deployment

**Using Cloud Run:**

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/ehr-triage-api

# Deploy to Cloud Run
gcloud run deploy ehr-triage-api \
  --image gcr.io/PROJECT_ID/ehr-triage-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## Production Considerations

### ⚠️ Critical Disclaimer

**This system is NOT production-ready for clinical use.**

If deploying for research purposes, ensure:

### Security

1. **Authentication & Authorization:**
   ```python
   # Add to api/app.py
   from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
   
   security = HTTPBearer()
   
   @app.middleware("http")
   async def verify_token(request: Request, call_next):
       # Implement token verification
       pass
   ```

2. **HTTPS Only:**
   - Use reverse proxy (nginx, Caddy)
   - Obtain SSL certificates (Let's Encrypt)
   - Redirect HTTP to HTTPS

3. **CORS Configuration:**
   ```python
   # api/app.py - restrict origins in production
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://your-domain.com"],  # Not ["*"]
       allow_credentials=True,
       allow_methods=["GET", "POST"],
       allow_headers=["*"],
   )
   ```

4. **Rate Limiting:**
   ```python
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
   ```

### Data Privacy

1. **No PHI**: Never use real patient data
2. **Encryption**: Encrypt data at rest and in transit
3. **Access Logs**: Monitor all data access
4. **Compliance**: Follow institutional policies

### Monitoring

1. **Application Monitoring:**
   ```python
   # Add to requirements.txt
   # prometheus-fastapi-instrumentator
   
   from prometheus_fastapi_instrumentator import Instrumentator
   
   Instrumentator().instrument(app).expose(app)
   ```

2. **Logging:**
   ```python
   import logging
   
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   ```

3. **Health Checks:**
   ```python
   @app.get("/health")
   async def health_check():
       return {"status": "healthy", "timestamp": datetime.now().isoformat()}
   ```

### Performance

1. **GPU Support** (for sequence models):
   ```dockerfile
   # Use CUDA base image
   FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
   ```

2. **Caching:**
   - Model caching (already implemented)
   - Feature normalization parameters
   - Configuration files

3. **Async Processing:**
   - Batch predictions
   - Background tasks for training

### Scalability

1. **Load Balancing:**
   - Use nginx or cloud load balancers
   - Multiple API instances

2. **Database** (if adding persistence):
   - PostgreSQL for metadata
   - Redis for caching

3. **Model Serving:**
   - Consider TorchServe or TensorFlow Serving
   - Model versioning

---

## Environment Variables

Create a `.env` file (never commit this):

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000

# Model Configuration
MODEL_CACHE_SIZE=100
PREDICTION_TIMEOUT=30

# Security (if implementing)
SECRET_KEY=<generate-secure-random-key>
API_KEY=<generate-api-key>

# Monitoring (if implementing)
SENTRY_DSN=<your-sentry-dsn>
```

---

## Backup and Disaster Recovery

1. **Code**: Use Git (already doing this)
2. **Models**: Backup to S3/Azure Blob/GCS
3. **Data**: Backup synthetic data generation scripts
4. **Configuration**: Version control all configs

---

## Updating Deployment

### Backend Updates

```bash
# Pull latest code
git pull

# Install new dependencies
pip install -r requirements.txt

# Restart service
# Docker: docker-compose restart api
# Systemd: sudo systemctl restart ehr-triage-api
```

### Frontend Updates

```bash
cd web
git pull
npm install
npm run build

# Docker: docker-compose restart web
# Systemd: sudo systemctl restart ehr-triage-web
```

### Model Updates

```bash
# Retrain models
python -m ehrtriage.scripts.train_all

# Restart API to load new models
docker-compose restart api
```

---

## Troubleshooting

### API Won't Start

```bash
# Check logs
docker-compose logs api

# Common issues:
# 1. Port 8000 already in use
lsof -i :8000  # Find process
kill -9 <PID>  # Kill process

# 2. Missing dependencies
pip install -r requirements.txt

# 3. Missing models
python -m ehrtriage.scripts.generate_synthetic
python -m ehrtriage.scripts.train_all
```

### Frontend Can't Connect to API

```bash
# Check API is running
curl http://localhost:8000/health

# Check CORS settings in api/app.py
# Ensure NEXT_PUBLIC_API_URL is set correctly

# Check browser console for errors
```

### Docker Issues

```bash
# Clean rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up

# Check disk space
docker system df
docker system prune  # Clean unused images/containers
```

---

## Decommissioning

When shutting down the system:

```bash
# Stop services
docker-compose down

# Remove data (optional)
rm -rf data/ models/

# Remove Docker images (optional)
docker rmi ehr-triage-api ehr-triage-web
```

---

## Support and Maintenance

### Regular Tasks

- [ ] Monitor error logs daily
- [ ] Check prediction latency weekly
- [ ] Update dependencies monthly
- [ ] Backup models monthly
- [ ] Review security quarterly

### Getting Help

- Check documentation in `docs/`
- Review GitHub issues
- Contact: [your-email@institution.edu]

---

## Legal and Compliance

### Disclaimers

**Every deployment must include:**

1. **UI Disclaimer**: Visible on every page
2. **API Disclaimer**: In API documentation
3. **Terms of Use**: Require acceptance
4. **Audit Log**: Track all predictions

### Institutional Requirements

- [ ] IRB approval (if using any patient data)
- [ ] IT security review
- [ ] HIPAA compliance check (US)
- [ ] GDPR compliance (EU)
- [ ] Data use agreement
- [ ] Risk assessment

### Reporting

If this system is used in research:

- [ ] Document all deployments
- [ ] Record all predictions made
- [ ] Report any incidents
- [ ] Publish results with limitations

---

## Conclusion

This deployment guide provides options for running the EHR Timeline Triage system.

**Remember**: This is a research prototype. Do not use for clinical decision-making.

For questions or issues, please refer to the documentation or contact the maintainers.
