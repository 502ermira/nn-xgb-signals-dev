from fastapi import FastAPI, Query
from utils.data_fetcher import fetch_ohlcv
from ml.predictor import make_prediction
from fastapi.middleware.cors import CORSMiddleware
from utils.data_fetcher import fetch_ohlcv, fetch_currency_pairs

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/signal")
def get_signal(pair: str = Query("EUR/USD"), tf: str = Query("15min")):
    try:
        df = fetch_ohlcv(pair, tf)
        prediction = make_prediction(df)

        return {
            "pair": pair,
            "timeframe": tf,
            "prediction": prediction
        }
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/api/pairs")
def get_supported_pairs():
    try:
        pairs = fetch_currency_pairs()
        return {"pairs": pairs}
    except Exception as e:
        return {"error": str(e)}