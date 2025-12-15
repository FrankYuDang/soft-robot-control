import torch
import torch.nn as nn
import sys
import os
import onnx

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.lstm_model import AttnLSTM

def export_model():
    # 1. 配置参数 (必须与训练时一致)
    params = {
        "input_dim": 3,
        "hidden_dim": 256,
        "output_dim": 3,
        "num_heads": 4,
        "model_path": "./data/trained_model.pth",
        "onnx_path": "./data/soft_robot_model.onnx"
    }
    
    # 2. 加载模型
    print(f"Loading model from {params['model_path']}...")
    model = AttnLSTM(
        input_dim=params["input_dim"], 
        hidden_dim=params["hidden_dim"],
        output_dim=params["output_dim"],
        num_heads=params["num_heads"]
    )
    
    # 加载权重
    try:
        model.load_state_dict(torch.load(params['model_path'], map_location="cpu"))
        model.eval() # 这一步至关重要！必须切换到 eval 模式
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
        return

    # 3. 创建一个虚拟输入 (Dummy Input)
    # ONNX 需要追踪数据在网络里的流向，所以我们要喂给它一个假数据
    # Shape: [Batch_Size, Seq_Len, Input_Dim] -> [1, 10, 3]
    dummy_input = torch.randn(1, 10, 3, requires_grad=True)

    # 4. 导出 ONNX
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,                      # 运行的模型
        dummy_input,                # 模型输入
        params['onnx_path'],        # 保存路径
        export_params=True,         # 是否在模型文件中存储权重
        opset_version=13,           # ONNX 版本
        do_constant_folding=True,   # 优化常量折叠
        input_names=['input'],      # 输入节点的名称
        output_names=['output'],    # 输出节点的名称
        dynamic_axes={              # 允许动态维度 (Batch Size 可变)
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Model successfully exported to {params['onnx_path']}")
    
    # 5. 验证 ONNX 模型结构是否正确
    onnx_model = onnx.load(params['onnx_path'])
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model validity check passed.")

if __name__ == "__main__":
    export_model()