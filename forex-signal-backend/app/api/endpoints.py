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
        pairs = fetch_currency_pairs()
        return {"pairs": pairs}
    except Exception as e:
        return {"error": str(e)}
    
@router.get("/history")
def get_prediction_history(limit: int = 50):
    db = SessionLocal()
    try:
        records = (
            db.query(Prediction)  # Changed from PredictionLog to Prediction
            .order_by(Prediction.timestamp.desc())
            .limit(limit)
            .all()
        )
        
        # Properly serialize SQLAlchemy objects
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