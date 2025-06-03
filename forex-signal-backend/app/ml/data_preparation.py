import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
from pathlib import Path

current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

def prepare_cnn_lstm_input(df, feature_cols, sequence_length=50):
    df = df.copy()
    df.dropna(inplace=True)
    
    print(f"[DEBUG] Input data shape after dropna: {df.shape}")
    
    if len(df) < sequence_length:
        raise ValueError(f"Insufficient data points: {len(df)}. Need at least {sequence_length}")
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    
    X = []
    for i in range(sequence_length, len(scaled_features)):
        X.append(scaled_features[i-sequence_length:i])
    
    X = np.array(X)
    print(f"[DEBUG] Prepared X shape: {X.shape}")
    
    return X, scaler
