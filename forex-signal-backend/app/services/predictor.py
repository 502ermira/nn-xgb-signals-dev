import pandas as pd
import pandas_ta as ta
from app.ml.data_preparation import prepare_cnn_lstm_input
from app.ml.models import load_hybrid_model, hybrid_predict
from app.models.prediction import Prediction
from app.db.database import SessionLocal
import xgboost as xgb
import tensorflow as tf
import numpy as np
import os
from sqlalchemy.orm import Session
from datetime import datetime 
from joblib import load as joblib_load

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def log_prediction(
    symbol: str, 
    timeframe: str, 
    timestamp: str,
    signal: str, 
    probs: dict,
    db: Session
):
    timestamp_dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    
    new_entry = Prediction(
        timestamp=timestamp_dt,
        symbol=symbol,
        timeframe=timeframe,
        signal=signal,
        cnn_lstm_probs=probs.get("cnn_lstm_probs", []),
        xgb_probs=probs.get("xgb_probs", []),
        hybrid_probs=probs.get("hybrid_probs", [])
    )
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)
    return new_entry

def make_prediction(df, symbol: str
                    , timeframe: str):
    df = df.copy()

    # Indicators
    df["rsi"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"])
    macd.columns = ["MACD", "MACD_Hist", "MACD_Signal"]
    df = df.join(macd)
    df = df.join(ta.bbands(df["close"], length=20))
    df = df.join(ta.stoch(df["high"], df["low"], df["close"]))
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["adx"] = ta.adx(df["high"], df["low"], df["close"])["ADX_14"]
    df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    latest = df.iloc[-1]
    signal_reasons = []
    signal = "HOLD"
    # RSI logic
    if latest["rsi"] < 30:
        signal_reasons.append("RSI indicates oversold")
    elif latest["rsi"] > 70:
        signal_reasons.append("RSI indicates overbought")

    # MACD logic
    if latest["MACD"] > latest["MACD_Signal"]:
        signal_reasons.append("MACD bullish crossover")
    elif latest["MACD"] < latest["MACD_Signal"]:
        signal_reasons.append("MACD bearish crossover")

    # Bollinger Band logic
    if latest["close"] < latest["BBL_20_2.0"]:
        signal_reasons.append("Price below lower Bollinger Band (oversold)")
    elif latest["close"] > latest["BBU_20_2.0"]:
        signal_reasons.append("Price above upper Bollinger Band (overbought)")

    # Stochastic logic
    if latest["STOCHk_14_3_3"] < 20:
        signal_reasons.append("Stochastic indicates oversold")
    elif latest["STOCHk_14_3_3"] > 80:
        signal_reasons.append("Stochastic indicates overbought")

    # EMA crossover logic
    if latest["ema20"] > latest["ema50"]:
        signal_reasons.append("EMA20 above EMA50 (bullish trend)")
    elif latest["ema20"] < latest["ema50"]:
        signal_reasons.append("EMA20 below EMA50 (bearish trend)")

    # ADX trend strength
    if latest["adx"] > 25:
        signal_reasons.append("Strong trend detected (ADX > 25)")

    # CCI logic
    if latest["cci"] > 100:
        signal_reasons.append("CCI indicates strong buying pressure")
    elif latest["cci"] < -100:
        signal_reasons.append("CCI indicates strong selling pressure")

    buy_signals = [
        "RSI indicates oversold",
        "MACD bullish crossover",
        "Price below lower Bollinger Band (oversold)",
        "Stochastic indicates oversold",
        "EMA20 above EMA50 (bullish trend)",
        "CCI indicates strong buying pressure"
    ]

    sell_signals = [
        "RSI indicates overbought",
        "MACD bearish crossover",
        "Price above upper Bollinger Band (overbought)",
        "Stochastic indicates overbought",
        "EMA20 below EMA50 (bearish trend)",
        "CCI indicates strong selling pressure"
    ]

    if any(r in signal_reasons for r in buy_signals) and not any(r in signal_reasons for r in sell_signals):
        signal = "BUY"
    elif any(r in signal_reasons for r in sell_signals) and not any(r in signal_reasons for r in buy_signals):
        signal = "SELL"

    feature_cols = ["close", "rsi", "MACD", "MACD_Signal", "BBU_20_2.0", "BBL_20_2.0",
                    "STOCHk_14_3_3", "STOCHd_14_3_3", "ema20", "ema50", "adx", "cci", "atr"]
    
    print(f"[DEBUG] DataFrame shape before preparation: {df.shape}")
    print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")
    print(f"[DEBUG] Checking for missing columns: {[col for col in feature_cols if col not in df.columns]}")
    
    pair_name = symbol.lower().replace("/", "")
    scaler_path = os.path.join("app", "ml", "models", f"{pair_name}_{timeframe}", "scaler.save")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    scaler = joblib_load(scaler_path)
    
    X_input, _ = prepare_cnn_lstm_input(df, feature_cols, scaler=scaler)
    print(f"[DEBUG] X_input shape: {X_input.shape if X_input is not None else 'None'}")

    if len(X_input) > 0:
        X_input = X_input[-1:]
        print(f"[DEBUG] Using only last sequence: {X_input.shape}")
    else:
        raise ValueError("No valid sequences available for prediction")

    cnn_lstm_model, xgb_model = load_hybrid_model(symbol, timeframe)

    hybrid_probs_array, _ = hybrid_predict(cnn_lstm_model, xgb_model, X_input)
    hybrid_probs = hybrid_probs_array[0]

    classes = ["BUY", "HOLD", "SELL"]
    hybrid_signal = classes[np.argmax(hybrid_probs)]

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    db = SessionLocal()
    try:
        log_prediction(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=now,
            signal=hybrid_signal,
            probs={
                "cnn_lstm_probs": [],
                "xgb_probs": [], 
                "hybrid_probs": hybrid_probs.tolist(),
            },
            db=db
        )
    finally:
        db.close()

    return {
        "signal": hybrid_signal,
        "confidence": float(np.max(hybrid_probs)),
        "probabilities": {
            "BUY": float(hybrid_probs[0]),
            "HOLD": float(hybrid_probs[1]),
            "SELL": float(hybrid_probs[2])
        },
        "indicators": {
            "rsi": round(latest["rsi"], 2),
            "macd": {
                "value": round(latest["MACD"], 4),
                "signal": round(latest["MACD_Signal"], 4)
            },
        "close": round(latest["close"], 4),
        "bollinger_upper": round(latest["BBU_20_2.0"], 4),
        "bollinger_lower": round(latest["BBL_20_2.0"], 4),
        "stochastic_k": round(latest["STOCHk_14_3_3"], 2),
        "stochastic_d": round(latest["STOCHd_14_3_3"], 2),
        "ema20": round(latest["ema20"], 4),
        "ema50": round(latest["ema50"], 4),
        "adx": round(latest["adx"], 2),
        "cci": round(latest["cci"], 2),
        "atr": round(latest["atr"], 4),
        },
        "reason": signal_reasons,
    }