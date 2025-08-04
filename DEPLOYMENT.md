# Deployment Guide - Breast Cancer Ultrasound Classification

This guide covers various deployment options for the breast cancer classification system, from local development to production cloud deployments.

## 🎯 Deployment Options Overview

| Platform | Complexity | Cost | Scalability | Best For |
|----------|------------|------|-------------|----------|
| **Local** | Low | Free | Limited | Development, Testing |
| **Streamlit Cloud** | Low | Free | Medium | Demos, Prototypes |
| **Heroku** | Medium | $7+/month | Medium | Small Production |
| **AWS** | High | $10+/month | High | Enterprise |
| **Docker** | Medium | Variable | High | Any Environment |

## 🖥️ Local Deployment

### Prerequisites
- Python 3.8+
- 4GB+ RAM
- Internet connection (initial setup)

### Quick Start
```bash
git clone https://github.com/yourusername/breast-cancer-ultrasound.git
cd breast-cancer-ultrasound
pip install -r requirements.txt
streamlit run webapp/app.py
```

Access at: `http://localhost:8501`

### Production-Like Local Setup
```bash
# Use production settings
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true

streamlit run webapp/app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true
```

## ☁️ Streamlit Cloud (Recommended for Demos)

### Setup Steps

1. **Fork the Repository**
   - Go to [GitHub](https://github.com/yourusername/breast-cancer-ultrasound)
   - Click "Fork" to create your copy

2. **Deploy to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub account
   - Select your forked repository
   - Set main file path: `webapp/app.py`
   - Click "Deploy"

3. **Configuration**
   - Python version: 3.9
   - Requirements file: `webapp/requirements.txt`
   - Advanced settings (if needed):
     ```toml
     [server]
     headless = true
     port = 8501
     
     [browser]
     gatherUsageStats = false
     ```

### Limitations
- 1GB RAM limit
- CPU-only inference
- Shared resources
- Public access only

## 🐳 Docker Deployment

### Local Docker

1. **Build Image**
```bash
cd webapp
docker build -t breast-cancer-classifier .
```

2. **Run Container**
```bash
docker run -p 8501:8501 breast-cancer-classifier
```

3. **With Volume Mounting**
```bash
docker run -p 8501:8501 \
  -v $(pwd)/../Dataset_BUSI_with_GT:/app/Dataset_BUSI_with_GT \
  -v $(pwd)/../fixed_best_model.pth:/app/fixed_best_model.pth \
  breast-cancer-classifier
```

### Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  web:
    build: ./webapp
    ports:
      - "8501:8501"
    volumes:
      - ./Dataset_BUSI_with_GT:/app/Dataset_BUSI_with_GT
      - ./fixed_best_model.pth:/app/fixed_best_model.pth
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_PORT=8501
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - web
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

### Multi-Stage Dockerfile

```dockerfile
# Build stage
FROM python:3.9-slim as builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:8501/healthz || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🌐 Heroku Deployment

### Prerequisites
- Heroku account
- Heroku CLI installed

### Setup Files

1. **Create `Procfile`**
```
web: streamlit run webapp/app.py --server.port=$PORT --server.address=0.0.0.0
```

2. **Create `runtime.txt`**
```
python-3.9.16
```

3. **Update `requirements.txt`**
```
streamlit==1.28.0
torch==2.0.1
torchvision==0.15.2
numpy==1.24.3
Pillow==9.5.0
opencv-python-headless==4.8.0.74
scikit-image==0.20.0
matplotlib==3.7.1
pandas==2.0.3
```

### Deployment Steps

```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Set buildpack
heroku buildpacks:set heroku/python

# Configure environment
heroku config:set STREAMLIT_SERVER_HEADLESS=true
heroku config:set STREAMLIT_SERVER_PORT=$PORT
heroku config:set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# Open app
heroku open
```

### Optimization for Heroku
- Use CPU-only PyTorch: `torch-cpu`
- Optimize Docker image size
- Use `.slugignore` to exclude unnecessary files:

```
*.pth
Dataset_BUSI_with_GT/
gradcam_outputs/
__pycache__/
*.pyc
.git/
```

## ☁️ AWS Deployment

### Option 1: AWS EC2

1. **Launch EC2 Instance**
```bash
# Create security group
aws ec2 create-security-group \
  --group-name breast-cancer-sg \
  --description "Security group for breast cancer app"

# Add inbound rules
aws ec2 authorize-security-group-ingress \
  --group-name breast-cancer-sg \
  --protocol tcp \
  --port 8501 \
  --cidr 0.0.0.0/0

# Launch instance
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --count 1 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-groups breast-cancer-sg
```

2. **Setup on EC2**
```bash
# Connect to instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Install dependencies
sudo yum update -y
sudo yum install python3 pip git -y

# Clone and setup
git clone https://github.com/yourusername/breast-cancer-ultrasound.git
cd breast-cancer-ultrasound
pip3 install -r requirements.txt

# Run with PM2 for process management
sudo npm install -g pm2
pm2 start "streamlit run webapp/app.py --server.port 8501 --server.address 0.0.0.0" --name breast-cancer-app
pm2 startup
pm2 save
```

### Option 2: AWS ECS (Fargate)

1. **Create Task Definition**
```json
{
  "family": "breast-cancer-classifier",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "breast-cancer-app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/breast-cancer-classifier:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/breast-cancer-classifier",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

2. **Deploy with CDK (TypeScript)**
```typescript
import * as ecs from '@aws-cdk/aws-ecs';
import * as ec2 from '@aws-cdk/aws-ec2';

const vpc = new ec2.Vpc(this, 'BreastCancerVPC');
const cluster = new ecs.Cluster(this, 'BreastCancerCluster', { vpc });

const taskDefinition = new ecs.FargateTaskDefinition(this, 'TaskDef', {
  memoryLimitMiB: 2048,
  cpu: 1024,
});

taskDefinition.addContainer('breast-cancer-app', {
  image: ecs.ContainerImage.fromRegistry('your-image'),
  portMappings: [{ containerPort: 8501 }],
});

new ecs.FargateService(this, 'Service', {
  cluster,
  taskDefinition,
  publicLoadBalancer: true,
});
```

### Option 3: AWS Lambda (Serverless)

For API-only deployment without Streamlit:

1. **Create `lambda_function.py`**
```python
import json
import base64
import io
from PIL import Image
from breast_cancer_classifier import BreastCancerClassifier

classifier = BreastCancerClassifier()

def lambda_handler(event, context):
    try:
        # Decode base64 image
        image_data = base64.b64decode(event['body']['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Make prediction
        result = classifier.predict_from_image(image)
        
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

2. **Deploy with SAM**
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  BreastCancerFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: src/
      Handler: lambda_function.lambda_handler
      Runtime: python3.9
      MemorySize: 1024
      Timeout: 30
      Events:
        Api:
          Type: Api
          Properties:
            Path: /predict
            Method: post
```

## 🌍 Google Cloud Platform

### Cloud Run Deployment

1. **Build and Push to Container Registry**
```bash
# Build Docker image
docker build -t gcr.io/your-project-id/breast-cancer-classifier .

# Push to GCR
docker push gcr.io/your-project-id/breast-cancer-classifier
```

2. **Deploy to Cloud Run**
```bash
gcloud run deploy breast-cancer-classifier \
  --image gcr.io/your-project-id/breast-cancer-classifier \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

### App Engine Deployment

1. **Create `app.yaml`**
```yaml
runtime: python39
service: breast-cancer-classifier

automatic_scaling:
  min_instances: 0
  max_instances: 10
  target_cpu_utilization: 0.6

resources:
  cpu: 2
  memory_gb: 4

env_variables:
  STREAMLIT_SERVER_HEADLESS: "true"
  STREAMLIT_SERVER_PORT: "8080"
```

2. **Deploy**
```bash
gcloud app deploy
```

## 🔧 Production Considerations

### Security

1. **HTTPS Setup**
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

2. **Authentication**
```python
# Add to webapp/app.py
import streamlit_authenticator as stauth

# Load user credentials
with open('config/credentials.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Show main app
    main_app()
elif authentication_status is False:
    st.error('Username/password is incorrect')
```

### Monitoring

1. **Health Checks**
```python
# Add to webapp/app.py
@st.cache_data
def health_check():
    try:
        # Test model loading
        model = load_model()
        return {"status": "healthy", "model": "loaded"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Add endpoint
if st.query_params.get("health"):
    st.json(health_check())
    st.stop()
```

2. **Logging**
```python
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
```

### Performance Optimization

1. **Caching**
```python
@st.cache_resource
def load_model():
    return BreastCancerClassifier()

@st.cache_data
def preprocess_image(image_bytes):
    return process_image(image_bytes)
```

2. **Database for Results**
```python
import sqlite3

def save_prediction(image_name, prediction, confidence):
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (image_name, prediction, confidence, timestamp)
        VALUES (?, ?, ?, datetime('now'))
    ''', (image_name, prediction, confidence))
    conn.commit()
    conn.close()
```

### Scaling

1. **Load Balancing**
```yaml
# docker-compose.yml for multiple replicas
version: '3.8'
services:
  web:
    build: ./webapp
    deploy:
      replicas: 3
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - web
```

2. **Auto-scaling (Kubernetes)**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: breast-cancer-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: breast-cancer-classifier
  template:
    metadata:
      labels:
        app: breast-cancer-classifier
    spec:
      containers:
      - name: app
        image: your-registry/breast-cancer-classifier:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: breast-cancer-classifier-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: breast-cancer-classifier
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 📊 Cost Optimization

### Platform Costs (Monthly Estimates)

| Platform | Basic | Production | Enterprise |
|----------|-------|------------|------------|
| **Streamlit Cloud** | Free | N/A | N/A |
| **Heroku** | $7 | $25 | $50+ |
| **AWS EC2** | $10 | $50 | $200+ |
| **AWS Fargate** | $15 | $60 | $250+ |
| **GCP Cloud Run** | $5 | $30 | $150+ |

### Optimization Tips

1. **Use spot instances** for non-critical workloads
2. **Auto-scaling** to reduce idle costs
3. **CPU-only inference** for cost savings
4. **CDN** for static assets
5. **Reserved instances** for predictable workloads

## 🔍 Troubleshooting

### Common Issues

1. **Memory Issues**
```bash
# Increase container memory
docker run -m 4g breast-cancer-classifier
```

2. **Port Conflicts**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

3. **Model Loading Errors**
```python
# Check model path
import os
print(f"Model exists: {os.path.exists('fixed_best_model.pth')}")
```

4. **Permission Issues**
```bash
# Fix permissions
chmod +x webapp/app.py
chown -R app:app /app
```

This deployment guide covers various scenarios from development to enterprise production. Choose the option that best fits your requirements, budget, and technical constraints.
