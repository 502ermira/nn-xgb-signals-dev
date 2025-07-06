import os
import sys
import pandas as pd
import time
from datetime import datetime
import pandas_ta as ta
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.data_fetcher import fetch_ohlcv

# --- CONFIG ---
SYMBOLS = [
    "EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD", 'XAU/USD',
    
    "EUR/GBP", "EUR/JPY", "EUR/AUD", "EUR/CAD", "EUR/CHF",
    
    "GBP/JPY", "GBP/CAD", "GBP/AUD",
    
    "AUD/CAD", "AUD/JPY", "NZD/JPY", "CAD/JPY",
    
    "USD/THB",
    
    "CHF/JPY"
]

TIMEFRAMES = ["15min","30min", "1h"] 
HISTORY_SIZE = 5000
OUTPUT_SIZE = 10000 
OUTPUT_FILE = "app/ml/data/training_data.csv"
SEQUENCE_LENGTH = 100

def add_indicators(df):
    try:
        df["rsi"] = ta.rsi(df["close"], length=14)
        
        macd = ta.macd(df["close"])
        if macd is not None:
            macd.columns = ["MACD", "MACD_Hist", "MACD_Signal"]
            df = df.join(macd)
        else:
            raise ValueError("MACD calculation returned None")
            
        df = df.join(ta.bbands(df["close"], length=20))
        stoch = ta.stoch(df["high"], df["low"], df["close"])
        df = df.join(stoch)
        df["ema20"] = ta.ema(df["close"], length=20)
        df["ema50"] = ta.ema(df["close"], length=50)
        adx = ta.adx(df["high"], df["low"], df["close"])
        df["adx"] = adx["ADX_14"]
        df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        print("Current DataFrame columns:", df.columns.tolist())
        raise
        
    return df

def label_signal(row):
    buy_conditions = [
        row["rsi"] < 25,
        row["ema20"] > row["ema50"],
        row["MACD"] > row["MACD_Signal"],
        row["close"] < row["BBL_20_2.0"]
    ]
    
    sell_conditions = [
        row["rsi"] > 75,
        row["ema20"] < row["ema50"],
        row["MACD"] < row["MACD_Signal"],
        row["close"] > row["BBU_20_2.0"]
    ]
    
    if sum(buy_conditions) >= 2:
        return 0  # BUY
    elif sum(sell_conditions) >= 2:
        return 2  # SELL
    return 1  # HOLD

def generate_data():
    batch_size = 2
    delay_seconds = 60  # Wait 1 minute between batches
    
    for i in range(0, len(SYMBOLS), batch_size):
        batch = SYMBOLS[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}: {batch}")
        
        for symbol in batch:
            for timeframe in TIMEFRAMES:
                pair_name = symbol.lower().replace("/", "")
                output_file = f"app/ml/data/{pair_name}_{timeframe}.csv"
                
                try:
                    print(f"Fetching {symbol} {timeframe}...")
                    df = fetch_ohlcv(symbol, timeframe, HISTORY_SIZE)
                    df = add_indicators(df)
                    df.dropna(inplace=True)
                    
                    if len(df) < SEQUENCE_LENGTH:
                        print(f"⚠️ Insufficient data for {symbol} {timeframe} ({len(df)} rows)")
                        continue
                        
                    df["label"] = df.apply(label_signal, axis=1)
                    df.to_csv(output_file, index=False)
                    print(f"✅ Saved {len(df)} rows to {output_file}")
                    
                except Exception as e:
                    print(f"❌ Failed {symbol} {timeframe}: {str(e)}")
        
        # Delay between batches (only if more batches remain)
        if i + batch_size < len(SYMBOLS):
            print(f"\n⏳ Waiting {delay_seconds} seconds to avoid rate limits...")
            time.sleep(delay_seconds)
    
    print("\nData generation completed")
    return True

if __name__ == "__main__":
    generate_data()