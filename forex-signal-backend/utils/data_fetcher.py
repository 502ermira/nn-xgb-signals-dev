import os
from dotenv import load_dotenv
import requests
import pandas as pd

load_dotenv()
API_KEY = os.getenv("TWELVE_DATA_API_KEY")

if not API_KEY:
    raise Exception("Missing Twelve Data API key")

def fetch_ohlcv(pair="EUR/USD", interval="15min", outputsize=50):
    symbol = pair
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={API_KEY}&format=JSON"

    print(f"[DEBUG] Fetching: {url}")

    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        raise Exception(f"API Error: {data}")

    df = pd.DataFrame(data["values"])
    df = df.rename(columns={"datetime": "time"})
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    numeric_cols = ["open", "high", "low", "close"]
    if "volume" in df.columns:
        numeric_cols.append("volume")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    return df

def fetch_currency_pairs():
    url = f"https://api.twelvedata.com/forex_pairs?apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "data" not in data:
        raise Exception(f"API Error: {data}")

    pairs = [item['symbol'] for item in data['data']]
    return pairs
