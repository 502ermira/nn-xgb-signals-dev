import numpy as np
import xgboost as xgb
from keras.models import load_model
import os
import tensorflow as tf

# Path to saved models
CNN_LSTM_PATH = "ml/models/cnn_lstm_model.h5"
XGB_PATH = "ml/models/xgb_model.json"

def load_hybrid_model(pair: str, timeframe: str):
    """Load models for specific pair/timeframe"""
    pair_name = pair.lower().replace("/", "")
    model_dir = os.path.join("app", "ml", "models", f"{pair_name}_{timeframe}")
    
    cnn_path = os.path.join(model_dir, "cnn_lstm_model.h5")
    xgb_path = os.path.join(model_dir, "xgb_model.json")
    
    print(f"[DEBUG] Looking for CNN model at: {cnn_path}")
    print(f"[DEBUG] Looking for XGB model at: {xgb_path}")
    
    if not os.path.exists(cnn_path):
        raise FileNotFoundError(f"CNN-LSTM model not found at {cnn_path}")
    if not os.path.exists(xgb_path):
        raise FileNotFoundError(f"XGBoost model not found at {xgb_path}")
    
    return load_model(cnn_path), xgb.Booster(model_file=xgb_path)

def hybrid_predict(cnn_model, xgb_model, X_input):
    # Get features from CNN's second-to-last layer
    feature_extractor = tf.keras.Model(
        inputs=cnn_model.inputs,
        outputs=cnn_model.layers[-2].output
    )
    features = feature_extractor.predict(X_input)
    
    # XGBoost prediction
    dmatrix = xgb.DMatrix(features)
    probs = xgb_model.predict(dmatrix)
    return probs, int(np.argmax(probs))
