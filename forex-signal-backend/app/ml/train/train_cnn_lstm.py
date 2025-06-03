import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import tensorflow as tf

# Get the absolute path to the root directory
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent.parent
sys.path.append(str(root_dir))

# Now import your local module
try:
    from app.ml.data_preparation import prepare_cnn_lstm_input
except ImportError:
    from data_preparation import prepare_cnn_lstm_input

def create_cnn_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train(pair: str, timeframe: str):
    pair_name = pair.lower().replace("/", "")
    data_file = os.path.join(root_dir, "app", "ml", "data", f"{pair_name}_{timeframe}.csv")
    model_dir = os.path.join(root_dir, "app", "ml", "models", f"{pair_name}_{timeframe}")
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "cnn_lstm_model.h5")
    
    df = pd.read_csv(data_file)

    # Updated feature columns to exactly match CSV columns
    feature_cols = [
        "close", 
        "rsi", 
        "MACD", 
        "MACD_Signal",
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
    
    # Verify all required columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    # One-hot encode target
    y = pd.get_dummies(df["label"])

    # Prepare input
    X, scaler = prepare_cnn_lstm_input(df, feature_cols)
    y = y.iloc[-len(X):].values  # align y with X

    model = create_cnn_lstm_model(input_shape=X.shape[1:], num_classes=3)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    model.save(model_path)
    print(f"✅ CNN-LSTM model saved to {model_path}")

if __name__ == "__main__":
    import sys
    train(sys.argv[1], sys.argv[2])