import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import tensorflow as tf 

# Get the absolute path to the root directory
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent.parent
sys.path.append(str(root_dir))

try:
    from app.ml.data_preparation import prepare_cnn_lstm_input
except ImportError:
    from data_preparation import prepare_cnn_lstm_input

def extract_cnn_features(df, feature_cols, pair, timeframe):
    pair_name = pair.lower().replace("/", "")
    model_path = os.path.join(root_dir, "app", "ml", "models", f"{pair_name}_{timeframe}", "cnn_lstm_model.h5")
    
    X_seq, _ = prepare_cnn_lstm_input(df, feature_cols)
    cnn_model = load_model(model_path)
    
    # Get features from second-to-last layer
    feature_extractor = tf.keras.Model(
        inputs=cnn_model.inputs,
        outputs=cnn_model.layers[-2].output
    )
    return feature_extractor.predict(X_seq)

def train(pair: str, timeframe: str):
    pair_name = pair.lower().replace("/", "")
    data_file = os.path.join(root_dir, "app", "ml", "data", f"{pair_name}_{timeframe}.csv")
    model_dir = os.path.join(root_dir, "app", "ml", "models", f"{pair_name}_{timeframe}")
    xgb_path = os.path.join(model_dir, "xgb_model.json")
    
    df = pd.read_csv(data_file)

    # Updated feature columns to match your CSV
    feature_cols = [
        "close", 
        "rsi", 
        "MACD",          # Matches your CSV
        "MACD_Signal",   # Matches your CSV
        "BBU_20_2.0",    
        "BBL_20_2.0",    
        "STOCHk_14_3_3", 
        "STOCHd_14_3_3", 
        "ema20", 
        "ema50", 
        "adx", 
        "cci", 
        "atr"
    ]

    # Verify columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])

    X = extract_cnn_features(df, feature_cols, pair, timeframe)
    y = y[-len(X):]  # Align y with X length

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, "eval")], verbose_eval=10)

    os.makedirs(model_dir, exist_ok=True)
    model.save_model(xgb_path)
    print(f"✅ XGBoost model saved to {xgb_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python train_xgb.py <pair> <timeframe>")
        print("Example: python train_xgb.py EUR/USD 15min")
        sys.exit(1)
    train(sys.argv[1], sys.argv[2])