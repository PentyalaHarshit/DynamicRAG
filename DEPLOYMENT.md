# DynamicRAG & OmniKnowledge Deployment Guide

This guide covers deployment options for **Docker** (Local & Production Server) and **Amazon Web Services (AWS)** (App Runner, ECS Fargate, and EC2).

---

## 1. Quick Local / Server Deployment with Docker Compose

The easiest way to run both the FastAPI Backend and Vite/React Frontend together:

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/) installed.

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/PentyalaHarshit/DynamicRAG.git
   cd DynamicRAG
   ```

2. **Configure Environment Variables (Optional)**:
   Create a `.env` file or export your API keys:
   ```bash
   GROQ_API_KEY=your_groq_key_here
   GOOGLE_API_KEY=your_google_key_here
   GOOGLE_CSE_ID=your_cse_id_here
   ```

3. **Build & Start Containers**:
   ```bash
   docker compose up --build -d
   ```

4. **Access the Applications**:
   - **Web UI (Frontend)**: `http://localhost:80` (or `http://localhost:3000`)
   - **FastAPI Documentation**: `http://localhost:8000/docs`
   - **Health Check**: `http://localhost:8000/health`

5. **Stop Containers**:
   ```bash
   docker compose down
   ```

---

## 2. AWS App Runner Deployment (Serverless & Auto-Scaling)

AWS App Runner is the simplest zero-maintenance way to deploy containerized web apps on AWS.

### Method A: Deploy directly from GitHub
1. Open the [AWS App Runner Console](https://console.aws.amazon.com/apprunner/).
2. Click **Create an App Runner service**.
3. Choose **Source code repository** and connect your GitHub repo (`DynamicRAG`).
4. Set configuration file to use the repository's `aws/apprunner.yaml`.
5. Under **Environment variables**, supply:
   - `GROQ_API_KEY`
   - `GOOGLE_API_KEY`
   - `GOOGLE_CSE_ID`
6. Click **Create & Deploy**. AWS will provision an auto-scaling HTTPS endpoint.

### Method B: Deploy from Amazon ECR Container
1. Build & push the Docker image to Amazon ECR:
   ```bash
   # 1. Log in to ECR
   aws ecr get-login-password --region <AWS_REGION> | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com

   # 2. Build backend container
   docker build -t dynamic-rag-backend .

   # 3. Tag and push
   docker tag dynamic-rag-backend:latest <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/dynamic-rag-backend:latest
   docker push <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/dynamic-rag-backend:latest
   ```
2. In App Runner, select **Container registry** -> **Amazon ECR** -> Select `dynamic-rag-backend:latest`.

---

## 3. AWS ECS Fargate Deployment (Enterprise Microservices)

For production enterprise workloads requiring independent container scaling and Virtual Private Cloud (VPC) isolation:

1. **Push both Frontend and Backend to ECR**:
   ```bash
   # Backend
   docker build -t <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/dynamic-rag-backend:latest .
   docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/dynamic-rag-backend:latest

   # Frontend
   docker build -t <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/dynamic-rag-frontend:latest ./frontend
   docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/dynamic-rag-frontend:latest
   ```

2. **Register Task Definition**:
   Update `<AWS_ACCOUNT_ID>` and `<AWS_REGION>` in [`aws/ecs-task-definition.json`](aws/ecs-task-definition.json) and register:
   ```bash
   aws ecs register-task-definition --cli-input-json file://aws/ecs-task-definition.json
   ```

3. **Create Service on ECS Cluster**:
   ```bash
   aws ecs create-service \
       --cluster dynamic-rag-cluster \
       --service-name dynamic-rag-service \
       --task-definition dynamic-rag-task \
       --desired-count 2 \
       --launch-type FARGATE \
       --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxxx],securityGroups=[sg-xxxxxx],assignPublicIp=ENABLED}"
   ```

---

## 4. AWS EC2 Deployment (Single Ubuntu/Amazon Linux Instance)

1. Launch an `t3.large` or `t3.xlarge` instance (Ubuntu 22.04 LTS).
2. SSH into your EC2 instance:
   ```bash
   ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
   ```
3. Install Docker:
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io docker-compose-v2
   sudo usermod -aG docker ubuntu
   ```
4. Clone and launch:
   ```bash
   git clone https://github.com/PentyalaHarshit/DynamicRAG.git
   cd DynamicRAG
   docker compose up --build -d
   ```
5. Ensure your EC2 Security Group allows inbound traffic on:
   - Port `80` (HTTP)
   - Port `443` (HTTPS)
   - Port `8000` (FastAPI backend if direct access is needed)

---

## Architecture Summary

```
                  INTERNET / USERS
                         │
                         ▼
                ┌──────────────────┐
                │   AWS ALB / Nginx │ (Port 80 / 443)
                └────────┬─────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
   ┌──────────────┐              ┌──────────────┐
   │ React / Vite │              │ FastAPI / RAG │ (Port 8000)
   │  Frontend    │              │  Backend     │
   └──────────────┘              └──────┬───────┘
                                        │
                         ┌──────────────┼──────────────┐
                         ▼              ▼              ▼
                     ChromaDB     Web Search API   LLM / Groq
```
