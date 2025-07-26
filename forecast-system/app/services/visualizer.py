import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import Dict, Any
import io
import base64

class ForecastVisualizer:
    @staticmethod
    def plot_forecast(historical_data: pd.DataFrame, 
                     forecast_data: Dict[str, Any]) -> str:
        """Generate visualization of forecast vs historical data"""
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(historical_data.index, 
                historical_data["close"], 
                label='Historical Prices',
                color='blue')
        
        forecast_timestamps = [datetime.fromisoformat(ts) for ts in forecast_data["forecast_timestamps"]]
        forecast_prices = forecast_data["forecast"]
        
        plt.plot(forecast_timestamps, 
                forecast_prices, 
                label='Forecast',
                color='red', 
                linestyle='--', 
                marker='o')
        
        last_point = historical_data.index[-1], historical_data["close"].values[-1]
        plt.scatter([last_point[0]], [last_point[1]], 
                   color='green', 
                   s=100, 
                   label='Current Price')
        
        plt.title(f"{forecast_data['pair']} {forecast_data['interval']} Price Forecast")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_base64