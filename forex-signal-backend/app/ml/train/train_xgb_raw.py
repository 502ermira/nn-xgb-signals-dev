import os
import sys
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent.parent
sys.path.append(str(root_dir))

def train_raw_xgb(pair: str, timeframe: str):
    pair_name = pair.lower().replace("/", "")
    data_file = os.path.join(root_dir, "app", "ml", "data", f"{pair_name}_{timeframe}.csv")
    model_dir = os.path.join(root_dir, "app", "ml", "models", f"{pair_name}_{timeframe}")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "xgb_raw_model.json")
    scaler_path = os.path.join(model_dir, "xgb_raw_scaler.save")

    df = pd.read_csv(data_file)
    df.dropna(inplace=True)

    feature_cols = [
        "close", "rsi", "MACD", "MACD_Signal", "BBU_20_2.0", 
        "BBL_20_2.0", "STOCHk_14_3_3", "STOCHd_14_3_3", 
        "ema20", "ema50", "adx", "cci", "atr"
    ]
    
    # Prepare features and labels
    X = df[feature_cols]
    y = df["label"]
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    # Train model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "eta": 0.1,
        "max_depth": 4,
        "seed": 42
    }

    model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, "eval")])
    
    # Save artifacts
    model.save_model(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, os.path.join(model_dir, "label_encoder.save"))
    print(f"✅ Raw XGBoost model saved to {model_path}")

if __name__ == "__main__":
    train_raw_xgb(sys.argv[1], sys.argv[2])