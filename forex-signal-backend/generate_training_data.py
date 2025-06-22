import os
import sys
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.data_fetcher import fetch_ohlcv

# --- CONFIG ---
SYMBOLS = ["EUR/USD","USD/JPY"]
TIMEFRAMES = ["15min", "1h"] 
HISTORY_SIZE = 5000
OUTPUT_SIZE = 10000 
OUTPUT_FILE = "app/ml/data/training_data.csv"
SEQUENCE_LENGTH = 100

def add_indicators(df):
    # Calculate each indicator separately with error handling
    try:
        # Basic indicators
        df["rsi"] = ta.rsi(df["close"], length=14)
        
        # MACD with explicit column naming
        macd = ta.macd(df["close"])
        if macd is not None:
            macd.columns = ["MACD", "MACD_Hist", "MACD_Signal"]
            df = df.join(macd)
        else:
            raise ValueError("MACD calculation returned None")
            
        # Other indicators
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
    # Make conditions stricter to reduce HOLD cases
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
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            pair_name = symbol.lower().replace("/", "")
            output_file = f"app/ml/data/{pair_name}_{timeframe}.csv"
            
            try:
                df = fetch_ohlcv(symbol, timeframe, HISTORY_SIZE)
                df = add_indicators(df)
                df.dropna(inplace=True)
                
                if len(df) < SEQUENCE_LENGTH:
                    print(f"Warning: Not enough data ({len(df)} rows) for {symbol} {timeframe}")
                    continue
                    
                df["label"] = df.apply(label_signal, axis=1)
                df.to_csv(output_file, index=False)
                print(f"Success: Saved {len(df)} rows to {output_file}")
                
            except Exception as e:
                print(f"Failed {symbol} {timeframe}: {str(e)}")
    
    print("\nData generation completed")
    return True

if __name__ == "__main__":
    generate_data()