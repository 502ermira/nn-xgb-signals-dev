import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List
from datetime import datetime, timedelta
from ..services.forecast_trainer import NBEATS, ForecastTrainer
from sklearn.preprocessing import MinMaxScaler
from config import FORECAST_MODEL_DIR

class ForecastPredictor:
    def __init__(self):
        self.trainer = ForecastTrainer()
        
    def _generate_timestamps(self, 
                           last_timestamp: datetime, 
                           interval: str, 
                           forecast_size: int) -> List[datetime]:
        """Generate future timestamps based on interval"""
        if interval.endswith('min'):
            minutes = int(interval[:-3])
            delta = timedelta(minutes=minutes)
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            delta = timedelta(hours=hours)
        else:
            delta = timedelta(hours=1)
            
        return [last_timestamp + i * delta for i in range(1, forecast_size + 1)]
    
    def _recursive_forecast(self,
                          model: NBEATS,
                          scaler: MinMaxScaler,
                          initial_window: np.ndarray,
                          window_size: int,
                          forecast_size: int) -> np.ndarray:
        """Make recursive multi-step predictions"""
        predictions = []
        current_window = initial_window.copy()
        
        for _ in range(forecast_size):
            with torch.no_grad():
                input_tensor = torch.FloatTensor(current_window[-window_size:]).unsqueeze(0)
                pred = model(input_tensor).numpy().flatten()[0]  # Get first prediction
                predictions.append(pred)
                current_window = np.append(current_window, pred)
                
        return np.array(predictions)
        
    def predict_future_prices(self, 
                            pair: str, 
                            interval: str, 
                            data: pd.DataFrame,
                            window_size: int = 50,
                            forecast_size: int = 10) -> Dict[str, Any]:
        """Predict future prices using N-BEATS model"""
        try:
            if len(data) < window_size:
                raise ValueError(f"Need at least {window_size} historical data points")
                
            model, scaler = self.trainer.load_model(pair, interval)
            
            recent_data = data["close"].values[-window_size:]
            recent_data_scaled = scaler.transform(recent_data.reshape(-1, 1)).flatten()
            
            forecast_scaled = self._recursive_forecast(
                model, scaler, recent_data_scaled, window_size, forecast_size
            )
            forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
                
            last_timestamp = data.index[-1]
            forecast_timestamps = self._generate_timestamps(last_timestamp, interval, forecast_size)
            
            return {
                "success": True,
                "pair": pair,
                "interval": interval,
                "forecast": forecast.tolist(),
                "forecast_timestamps": [ts.isoformat() for ts in forecast_timestamps],
                "historical_timestamps": [ts.isoformat() for ts in data.index[-window_size:]],
                "historical_prices": data["close"].values[-window_size:].tolist(),
                "last_historical_timestamp": last_timestamp.isoformat(),
                "last_historical_price": float(data["close"].values[-1]),
                "window_size": window_size,
                "forecast_size": forecast_size
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "pair": pair,
                "interval": interval
            }