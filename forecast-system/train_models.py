import os
from app.services.data_fetcher import ForecastDataFetcher
from app.services.forecast_trainer import ForecastTrainer
from app.services.forecast_trainer import ForecastTrainer
from config import TWELVE_DATA_API_KEY, FOREX_PAIRS, INTERVALS

def train_all_models():
    fetcher = ForecastDataFetcher(TWELVE_DATA_API_KEY)
    trainer = ForecastTrainer()
    
    for pair in FOREX_PAIRS:
        for interval in INTERVALS:
            print(f"Training {pair} {interval}")
            data = fetcher.fetch_ohlcv(pair, interval)
            trainer.train_model(pair, interval, data)
            
if __name__ == "__main__":
    train_all_models()