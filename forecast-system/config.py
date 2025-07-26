import os
from dotenv import load_dotenv

load_dotenv()

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

FORECAST_MODEL_DIR = os.path.join(os.path.dirname(__file__), "app", "ml", "models", "forecast")
os.makedirs(FORECAST_MODEL_DIR, exist_ok=True)

# Training Parameters
DEFAULT_WINDOW_SIZE = 50  # Input sequence length
DEFAULT_FORECAST_SIZE = 10  # Number of steps to predict
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001

FOREX_PAIRS = [
    "EUR/USD"
]

INTERVALS = ["15min", "30min"]