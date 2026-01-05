import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List
import os
import datetime

# --- 新增: 数据库相关导入 ---
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# 1. 定义模型结构 (保持不变)
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context

class SoftRobotModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=2):
        super(SoftRobotModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        out = self.fc(context)
        return out

# 2. 初始化 FastAPI
app = FastAPI(title="Soft Robot Control API (With DB)", version="2.0")

# --- 新增: 数据库配置 ---
# 从环境变量获取数据库地址 (我们在 docker-compose.yml 里配过这个)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db") 

# 创建数据库引擎
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 定义数据表结构
class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    cable_1_tension = Column(Float)
    cable_2_tension = Column(Float)
    cable_3_tension = Column(Float)
    predicted_x = Column(Float)
    predicted_y = Column(Float)
    temperature = Column(Float)

# 自动创建表 (如果不存在)
Base.metadata.create_all(bind=engine)

# 依赖项: 获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# ------------------------

# 3. 加载模型 (保持不变)
DEVICE = torch.device("cpu")
model = SoftRobotModel()
model_path = "data/trained_model.pth"

try:
    if os.path.exists(model_path):
        print("🤖 Loading Soft Robot Brain...")
        # 加上 weights_only=False 以抑制警告 (在你完全控制模型来源时是安全的)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE)) #, weights_only=False)) 
        model.eval()
        print(f"✅ Model loaded successfully from {model_path}")
    else:
        print(f"⚠️ Warning: Model not found at {model_path}. Using random weights.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 4. 定义请求体
class CableInput(BaseModel):
    c1: float
    c2: float
    c3: float
    temperature: float

@app.get("/")
def health_check():
    return {"status": "active", "version": "2.0", "db": "connected"}

# 5. 预测接口 (修改版：加入数据库存储)
@app.post("/predict")
def predict_coordinates(data: CableInput, db: Session = Depends(get_db)):
    try:
        # A. 数据预处理
        input_data = np.array([[data.c1, data.c2, data.c3]], dtype=np.float32)
        # 增加时间步维度 (batch, seq_len, features) -> (1, 1, 3)
        input_tensor = torch.tensor(input_data).unsqueeze(1).to(DEVICE)

        # B. 模型推理
        with torch.no_grad():
            prediction = model(input_tensor)
            coords = prediction.cpu().numpy()[0]

        result_x = float(coords[0])
        result_y = float(coords[1])

        # --- 新增: C. 存入数据库 ---
        db_record = PredictionRecord(
            cable_1_tension=data.c1,
            cable_2_tension=data.c2,
            cable_3_tension=data.c3,
            predicted_x=result_x,
            predicted_y=result_y,
            temperature = data.temperature
        )
        db.add(db_record)
        db.commit() # 提交事务
        db.refresh(db_record) # 刷新以获取生成的 ID
        # ------------------------

        return {
            "prediction": {"x": result_x, "y": result_y},
            "db_record_id": db_record.id,  # 返回数据库里的 ID，证明存进去了
            "status": "logged"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))