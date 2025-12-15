import time
import numpy as np
import torch
import onnxruntime as ort
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.lstm_model import AttnLSTM

def benchmark():
    # 1. 准备数据
    # Batch Size = 1 (模拟实时单次请求), Seq Len = 10, Input Dim = 3
    dummy_input = torch.randn(1, 10, 3, dtype=torch.float32)
    numpy_input = dummy_input.numpy() # ONNX 需要 numpy 格式

    print("🔥 Warming up models...")
    
# --- Load PyTorch ---
    # 必须严格匹配训练时的参数！
    pt_model = AttnLSTM(
        input_dim=3, 
        hidden_dim=256, 
        num_layers=2,   # 你的模型只有 2 层 LSTM
        output_dim=3,   # 输出 x, y, z 共 3 个值
        num_heads=4
    )
    pt_model.load_state_dict(torch.load("./data/trained_model.pth", map_location="cpu"))
    pt_model.eval()
    
    # --- Load ONNX ---
    # 创建推理会话 (Session)
    ort_session = ort.InferenceSession("./data/soft_robot_model.onnx")
    
    # 预热 (Warmup) - 让 CPU 缓存加载好
    for _ in range(10):
        pt_model(dummy_input)
        ort_session.run(None, {"input": numpy_input})

    print("🚀 Starting Benchmark (1000 iterations)...")

    # --- Test PyTorch ---
    start = time.time()
    for _ in range(1000):
        with torch.no_grad():
            pt_model(dummy_input)
    pt_time = (time.time() - start) * 1000 / 1000 # 平均耗时 (ms)

    # --- Test ONNX ---
    start = time.time()
    for _ in range(1000):
        # run(output_names, input_feed)
        ort_session.run(None, {"input": numpy_input})
    onnx_time = (time.time() - start) * 1000 / 1000 # 平均耗时 (ms)

    # --- Report ---
    print("\n" + "="*30)
    print(f"🐢 PyTorch Latency: {pt_time:.4f} ms")
    print(f"⚡ ONNX Latency:    {onnx_time:.4f} ms")
    print(f"🚀 Speedup:         {pt_time / onnx_time:.2f}x")
    print("="*30)

if __name__ == "__main__":
    benchmark()