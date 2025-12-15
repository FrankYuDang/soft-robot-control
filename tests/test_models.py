import torch
import pytest
import sys
import os

# 把项目根目录加入路径，否则找不到 src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.lstm_model import AttnLSTM, AttentionLayer

# --- 测试 Attention Layer ---
def test_attention_layer_shape():
    """
    测试 Attention 层的输入输出维度是否符合预期
    """
    batch_size = 8
    seq_len = 10
    hidden_dim = 32
    num_heads = 4

    # 模拟 LSTM 的输出: [batch, seq_len, hidden]
    dummy_input = torch.randn(batch_size, seq_len, hidden_dim)
    
    layer = AttentionLayer(hidden_dim, num_heads)
    output = layer(dummy_input)

    # 预期输出: Attention 把序列压缩了，只剩 [batch, hidden]
    assert output.shape == (batch_size, hidden_dim), \
        f"Expected shape {(batch_size, hidden_dim)}, but got {output.shape}"

# --- 测试完整的 AttnLSTM 模型 ---
def test_attnlstm_forward_pass():
    """
    测试整个模型能否跑通一次 Forward Pass，且输出维度正确
    """
    batch_size = 16
    seq_len = 10
    input_dim = 3
    hidden_dim = 64
    output_dim = 3  # 预测 x, y, z

    model = AttnLSTM(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        output_dim=output_dim
    )

    # 模拟输入数据 [batch, seq_len, input_dim]
    dummy_x = torch.randn(batch_size, seq_len, input_dim)
    
    # 运行前向传播
    y_pred = model(dummy_x)

    # 1. 检查是否有报错 (如果上面崩了，测试就失败了)
    # 2. 检查输出维度
    assert y_pred.shape == (batch_size, output_dim), \
        f"Expected output shape {(batch_size, output_dim)}, got {y_pred.shape}"

def test_attnlstm_device_movement():
    """
    测试模型能否正确移动到 GPU (如果可用) 或 CPU
    """
    model = AttnLSTM(input_dim=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 检查参数是否在正确的 device 上
    param = next(model.parameters())
    assert param.device.type == device.type