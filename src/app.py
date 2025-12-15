import time
import logging
from fastapi import Request
import torch
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# 导入你的模型定义
# 注意：Docker 里的工作目录是 /app，所以 src 是顶级包
from src.models.lstm_model import AttnLSTM

# 1. 定义请求数据的格式 (Schema)
# 这就像是 API 的“安检门”，不符合格式的数据会被直接挡回去
class CableInput(BaseModel):
    # 假设输入是一个序列，包含 10 个时间步的数据，每个时间步有 3 根线的长度
    # 例如: [[100, 100, 100], [101, 100, 99], ...]
    sequence: List[List[float]] 

# 1. 配置日志 (Logging Configuration)
# 在工业界，我们通常输出 JSON 格式的日志，方便 ELK (Elasticsearch) 分析
# 这里为了简单，我们先用标准格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("soft-robot-api")

app = FastAPI(title="Soft Robot Control API", version="1.2")

# 全局变量存放模型
model = None
DEVICE = "cpu" # 推理通常用 CPU 就够了，除非并发量极大

# 2. 插入中间件 (Middleware) - 这是核心修改
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    这个函数会拦截每一个请求，记录它进入和离开的时间。
    """
    start_time = time.time()
    
    # 处理请求
    response = await call_next(request)
    
    # 计算耗时 (毫秒)
    process_time = (time.time() - start_time) * 1000
    
    # 3. 打印日志
    # 真正的 CTO 会关注：这个请求花了多久？状态码是多少？
    logger.info(f"Path: {request.url.path} | Method: {request.method} | Status: {response.status_code} | Latency: {process_time:.2f}ms")
    
    # 把耗时也加到 Response Header 里，方便客户端查看
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# 2. 启动事件：API 启动时执行一次
@app.on_event("startup")
def load_model():
    global model
    print("🤖 Loading Soft Robot Brain...")
    
    try:
        # 初始化模型架构 (参数必须和你训练时的一致！)
        # 如果你训练时用了 hidden_dim=32, 这里也得是 32
        model = AttnLSTM(
            input_dim=3, 
            hidden_dim=256, 
            output_dim=3, 
            num_heads=4
        )
        
        # 加载权重
        # 注意路径：在 Docker 里，我们挂载的目录是 /app/data
        model_path = "/app/data/trained_model.pth"
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval() # 切换到评估模式 (关闭 Dropout 等)
            print(f"✅ Model loaded successfully from {model_path}")
        else:
            print(f"⚠️ Warning: Model file not found at {model_path}. API will run but predictions will fail.")
            
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

@app.get("/")
def health_check():
    return {"status": "active", "model_loaded": model is not None}

# 3. 预测接口
@app.post("/predict")
def predict_coordinates(input_data: CableInput):
    """
    接收拉线长度序列，返回预测的末端坐标 (x, y, z)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # A. 数据预处理
        # ⚠️ CRITICAL TODO: 这里其实需要加上归一化 (Scaler) 逻辑
        # 你的模型是用归一化数据(0-1)训练的，如果传入真实长度(100mm)，预测会不准。
        # 为了演示流程，我们先假设传入的数据已经是归一化过的。
        
        # 将 list 转为 tensor: [1, seq_len, input_dim]
        input_tensor = torch.tensor(input_data.sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # B. 模型推理
        with torch.no_grad():
            output_tensor = model(input_tensor) # output: [1, 3]
            
        # C. 结果后处理
        # 同样，这里应该反归一化 (Inverse Transform) 才能得到毫米值
        prediction = output_tensor.cpu().numpy().tolist()[0]
        
        return {
            "predicted_coordinates": {
                "x": prediction[0],
                "y": prediction[1],
                "z": prediction[2]
            },
            "raw_output": prediction
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")