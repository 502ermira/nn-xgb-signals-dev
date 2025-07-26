import pandas as pd
import requests
from typing import Optional
from datetime import datetime, timedelta
import time

class ForecastDataFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com/time_series"
        
    def fetch_ohlcv(self, pair: str, interval: str, output_size: int = 5000) -> pd.DataFrame:
        """Fetch OHLCV data for forecasting"""
        url = (f"{self.base_url}?symbol={pair}&interval={interval}"
               f"&outputsize={output_size}&apikey={self.api_key}&format=JSON")
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if "values" not in data:
                raise ValueError(f"API Error: {data.get('message', 'Unknown error')}")
                
            df = pd.DataFrame(data["values"]).rename(columns={"datetime": "time"})
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time")
            
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df.dropna()
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data: {str(e)}")
    
    def fetch_recent_for_forecast(self, pair: str, interval: str, window_size: int) -> pd.DataFrame:
        """Fetch recent data specifically for forecasting"""
        df = self.fetch_ohlcv(pair, interval, window_size)
        return df[["time", "close"]].set_index("time")