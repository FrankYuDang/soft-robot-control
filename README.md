# Soft Robot Control AI Service 🤖

![CI/CD Status](https://github.com/FrankYuDang/soft-robot-control/actions/workflows/deploy.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![AWS](https://img.shields.io/badge/Deployed%20on-AWS-orange)

This project implements a production-grade AI microservice for controlling soft robotic manipulators. It utilizes a hybrid **Attention-LSTM** architecture to predict end-effector coordinates based on cable length inputs.

## 🏗 Architecture

[Local Dev (Mac M1)] --> [GitHub Actions (CI/CD)] --> [Docker Hub] --> [AWS EC2 (Production)]

- **Model:** Attention-LSTM (Optimized with ONNX)
- **API:** FastAPI (Asynchronous)
- **Containerization:** Docker (Multi-arch build support)
- **Deployment:** Automated AWS EC2 deployment via SSH tunneling

## 🚀 Key Features

- **High Performance:** <15ms latency using ONNX Runtime.
- **Resource Optimized:** Custom Docker image optimized for CPU-only environments (90% size reduction).
- **Automated:** Full CI/CD pipeline handles testing, building, and deployment on every push.
- **Robust:** Includes comprehensive logging and health checks.

## 🛠 Quick Start (Run Locally)

```bash
# 1. Pull the image
docker pull frankdang024/soft-robot-api:latest

# 2. Run the container
docker run -p 8000:8000 frankdang024/soft-robot-api:latest


```
🛠 Project Structure
```
├── src/            # Source code (Models, API logic)
├── scripts/        # Training and utility scripts
├── tests/          # Unit tests
├── data/           # Data & Model weights (Ignored by Git)
└── Dockerfile      # Container definition
```

