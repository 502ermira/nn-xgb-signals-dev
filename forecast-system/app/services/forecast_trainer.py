import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from typing import Tuple, List
import joblib
from config import FORECAST_MODEL_DIR
from config import FORECAST_MODEL_DIR, DEFAULT_WINDOW_SIZE, DEFAULT_FORECAST_SIZE

class NBEATSBlock(nn.Module):
    """Basic N-BEATS block"""
    def __init__(self, input_size: int, theta_size: int, hidden_size: int, num_hidden_layers: int):
        super().__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.hidden_size = hidden_size
        
        self.input_fc = nn.Linear(input_size, hidden_size)
        
        layers = []
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        self.fc_stack = nn.Sequential(*layers)
        
        self.theta_fc = nn.Linear(hidden_size, theta_size)
        
        self.backcast_fc = nn.Linear(theta_size, input_size)
        self.forecast_fc = nn.Linear(theta_size, input_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_fc(x)
        h = self.fc_stack(h)
        theta = self.theta_fc(h)
        backcast = self.backcast_fc(theta)
        forecast = self.forecast_fc(theta)
        return backcast, forecast

class NBEATS(nn.Module):
    """N-BEATS model"""
    def __init__(self, 
                 input_size: int = 50,
                 forecast_size: int = 10,
                 theta_size: int = 16,
                 hidden_size: int = 128,
                 num_hidden_layers: int = 4,
                 num_blocks: int = 3):
        super().__init__()
        self.input_size = input_size
        self.forecast_size = forecast_size
        
        self.blocks = nn.ModuleList([
            NBEATSBlock(input_size, theta_size, hidden_size, num_hidden_layers)
            for _ in range(num_blocks)
        ])
        
        self.forecast_proj = nn.Linear(input_size, forecast_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forecast = torch.zeros(x.size(0), self.forecast_size, device=x.device)
        
        for block in self.blocks:
            backcast, block_forecast = block(x)
            projected_forecast = self.forecast_proj(block_forecast)
            forecast = forecast + projected_forecast
            x = x - backcast
            
        return forecast

class ForecastTrainer:
    def __init__(self, model_dir: str = FORECAST_MODEL_DIR):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.scalers = {}
        
    def create_sequences(self, data: pd.DataFrame, window_size: int, forecast_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets for training"""
        sequences = []
        targets = []
        
        for i in range(len(data) - window_size - forecast_size):
            seq = data.iloc[i:i+window_size]["close"].values
            target = data.iloc[i+window_size:i+window_size+forecast_size]["close"].values
            sequences.append(seq)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)
    
    def train_model(self, 
                   pair: str, 
                   interval: str, 
                   data: pd.DataFrame,
                   window_size: int = 50,
                   forecast_size: int = 10,
                   epochs: int = 100,
                   batch_size: int = 32,
                   learning_rate: float = 0.001) -> None:
        """Train N-BEATS model for a specific pair and timeframe"""
        X, y = self.create_sequences(data, window_size, forecast_size)
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
        y_scaled = scaler.transform(y.reshape(-1, 1)).reshape(y.shape)
        
        model_name = f"{pair.lower().replace('/', '')}_{interval}"
        scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.save")
        joblib.dump(scaler, scaler_path)
        self.scalers[model_name] = scaler
        
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_scaled)
        
        model = NBEATS(
            input_size=window_size,
            forecast_size=forecast_size,
            theta_size=16,
            hidden_size=128,
            num_hidden_layers=4,
            num_blocks=3
        )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in tqdm(range(epochs), desc=f"Training {pair} {interval}"):
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        model_path = os.path.join(self.model_dir, f"{model_name}_nbeats.pt")
        torch.save(model.state_dict(), model_path)
        
    def load_model(self, pair: str, interval: str) -> Tuple[nn.Module, MinMaxScaler]:
        model_name = f"{pair.lower().replace('/', '')}_{interval}"
        model_path = os.path.join(self.model_dir, f"{model_name}_nbeats.pt")
        scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.save")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Model or scaler not found for {pair} {interval}")
        
        model = NBEATS(
            input_size=DEFAULT_WINDOW_SIZE,
            forecast_size=DEFAULT_FORECAST_SIZE
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        scaler = joblib.load(scaler_path)
        
        return model, scaler