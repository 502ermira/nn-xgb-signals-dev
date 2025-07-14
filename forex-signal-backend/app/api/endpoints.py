from fastapi import APIRouter, Query
import os
import pandas as pd
from fastapi.responses import JSONResponse
from app.services.data_fetcher import fetch_ohlcv, fetch_currency_pairs
from app.services.predictor import make_prediction
from app.db.database import SessionLocal
from app.models.prediction import Prediction
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api")

@router.get("/signal")
def get_signal(pair: str = Query("EUR/USD"), tf: str = Query("15min")):
    try:
        df = fetch_ohlcv(pair, tf)
        prediction = make_prediction(df, symbol=pair, timeframe=tf)
        return {
            "pair": pair,
            "timeframe": tf,
            "prediction": prediction,
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/pairs")
def get_supported_pairs():
    try:
        selected_pairs = [
            "AUD/CAD", "AUD/JPY", "AUD/USD",
            "CAD/JPY",
            "CHF/JPY",
            "EUR/AUD", "EUR/CAD", "EUR/CHF", "EUR/GBP", "EUR/JPY", "EUR/USD",
            "GBP/AUD", "GBP/CAD", "GBP/JPY", "GBP/USD",
            "NZD/JPY", "NZD/USD",
            "USD/CAD", "USD/CHF", "USD/JPY", "USD/THB", "USD/INR",
            "XAU/USD",
        ]
        
        available_pairs = fetch_currency_pairs()
        
        normalized_selected = [p.replace("/", "") for p in selected_pairs]
        normalized_available = [p.replace("/", "") for p in available_pairs]
        
        valid_pairs = [
            selected_pairs[i] 
            for i, p in enumerate(normalized_selected) 
            if p in normalized_available
        ]
        
        return {"pairs": valid_pairs}
    except Exception as e:
        return {"error": str(e)}
    
@router.get("/history")
def get_prediction_history(limit: int = 50):
    db = SessionLocal()
    try:
        records = (
            db.query(Prediction)
            .order_by(Prediction.timestamp.desc())
            .limit(limit)
            .all()
        )
        
        return [
            {
                "timestamp": record.timestamp,
                "symbol": record.symbol,
                "timeframe": record.timeframe,
                "signal": record.signal,
                "cnn_lstm_probs": record.cnn_lstm_probs,
                "xgb_probs": record.xgb_probs,
                "hybrid_probs": record.hybrid_probs
            }
            for record in records
        ]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        db.close()