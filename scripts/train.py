import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.lstm_model import AttnLSTM
from src.data_loader.dataset import KinematicsDataset, load_and_process_data

def main():
    # --- Configuration (Must match app.py!) ---
    params = {
        "seq_length": 10,
        "batch_size": 64,
        "input_dim": 3,
        "hidden_dim": 256,   # <--- 这里的 256 必须和 app.py 里的一致！
        "output_dim": 3,
        "num_heads": 4,      # Attention heads
        "epochs": 50,        # 稍微多练几轮
        "lr": 0.001,
        "data_path": "./data/Processed_Data3w.xlsx",
        "save_path": "./data/trained_model.pth" # <--- 直接保存到 data 目录
    }

    # 1. Load Data
    print(f"Loading data from {params['data_path']}...")
    if not os.path.exists(params['data_path']):
        print("Error: Data file not found! Please check path.")
        return

    X, y, _, _ = load_and_process_data(params["data_path"], params["seq_length"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    
    train_loader = DataLoader(KinematicsDataset(X_train, y_train), batch_size=params["batch_size"])
    
    # 2. Init Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device} with hidden_dim={params['hidden_dim']}...")
    
    model = AttnLSTM(
        input_dim=params["input_dim"], 
        hidden_dim=params["hidden_dim"],
        output_dim=params["output_dim"],
        num_heads=params["num_heads"]
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.MSELoss()

    # 3. Train Loop
    model.train()
    for epoch in range(params["epochs"]):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{params['epochs']}: Avg Loss = {total_loss/len(train_loader):.6f}")

    print("Training finished.")

    # 4. Save Model
    torch.save(model.state_dict(), params['save_path'])
    print(f"✅ Model saved to {params['save_path']}")

if __name__ == "__main__":
    main()