from fastapi import APIRouter, Query
from app.services.data_fetcher import fetch_ohlcv, fetch_currency_pairs
from app.services.predictor import make_prediction

router = APIRouter(prefix="/api")

@router.get("/signal")
def get_signal(pair: str = Query("EUR/USD"), tf: str = Query("15min")):
    try:
        df = fetch_ohlcv(pair, tf)
        prediction = make_prediction(df)
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