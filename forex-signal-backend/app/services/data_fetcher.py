import requests
import pandas as pd
from app.core.config import TWELVE_DATA_API_KEY

def fetch_ohlcv(pair="EUR/USD", interval="15min", outputsize=5000):
    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol={pair}&interval={interval}&outputsize={outputsize}&"
        f"apikey={TWELVE_DATA_API_KEY}&format=JSON"
    )

    print(f"[DEBUG] Fetching: {url}")
    response = requests.get(url)
    data = response.json()
    print("[DEBUG] API Response Sample:", {k: data[k] for k in list(data)[:2]})

    if "values" not in data:
        print("[DEBUG] Full API Response:", data)
        raise Exception(f"API Error: {data}")
    
    if len(data["values"]) < 50:
        raise Exception(f"Insufficient data points: {len(data['values'])}")

    df = pd.DataFrame(data["values"]).rename(columns={"datetime": "time"})
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    # Verify we have all required columns
    required_cols = ["open", "high", "low", "close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise Exception(f"Missing columns: {missing}")

    # Convert to numeric
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with missing values
    df = df.dropna()
    
    return df
def fetch_currency_pairs():
    url = f"https://api.twelvedata.com/forex_pairs?apikey={TWELVE_DATA_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "data" not in data:
        raise Exception(f"API Error: {data}")

    return [item["symbol"] for item in data["data"]]