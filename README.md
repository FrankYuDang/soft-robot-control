# Soft Robot Control AI Service 🤖

This project implements a production-grade AI microservice for controlling soft robotic manipulators. It utilizes a hybrid **Attention-LSTM** architecture to predict end-effector coordinates based on cable length inputs.

## 🏗 Architecture

- **Model:** PyTorch-based Attention-LSTM (Hidden Dim: 256, Heads: 4)
- **Serving:** FastAPI (Asynchronous REST API)
- **Infrastructure:** Docker (Containerized Environment)
- **Testing:** Pytest

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have **Docker Desktop** installed and running.

### 2. Build the Image
```bash
docker build -t soft-robot-api:v1 .
```
### 3. Run the Service
```bash
# Mount local data volume for model persistence
docker run -p 8000:8000 -v $(pwd)/data:/app/data soft-robot-api:v1
```
### 4. API Usage

Visit the interactive Swagger documentation at:

`http://127.0.0.1:8000/docs`

Sample Request (POST /predict):
```json
{
  "sequence": [
    [0.5, 0.5, 0.5],
    ... (10 time steps) ...
  ]
}
```
🛠 Project Structure
```
├── src/            # Source code (Models, API logic)
├── scripts/        # Training and utility scripts
├── tests/          # Unit tests
├── data/           # Data & Model weights (Ignored by Git)
└── Dockerfile      # Container definition
```

