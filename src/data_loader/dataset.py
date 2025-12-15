import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

class KinematicsDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]

def load_and_process_data(file_path: str, seq_length: int = 10):
    """
    Loads excel data, normalizes it, and creates sequences.
    """
    if not file_path.endswith('.xlsx'):
        raise ValueError("File must be an Excel file")

    data = pd.read_excel(file_path)
    # Assuming first 3 cols are input, next 3 are targets
    raw_inputs = data.iloc[:, :3].values
    raw_targets = data.iloc[:, 3:].values

    # Normalize
    input_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    norm_inputs = input_scaler.fit_transform(raw_inputs)
    norm_targets = target_scaler.fit_transform(raw_targets)

    # Create Sequences (Vectorized approach is better, but stick to list for readability now)
    X, y = [], []
    for i in range(len(norm_inputs) - seq_length):
        X.append(norm_inputs[i:i+seq_length])
        y.append(norm_targets[i+seq_length])
        
    return np.array(X), np.array(y), input_scaler, target_scaler