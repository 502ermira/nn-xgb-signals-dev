import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
from pathlib import Path

current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

SEQUENCE_LENGTH = 100 

def prepare_cnn_lstm_input(df, feature_cols, sequence_length=SEQUENCE_LENGTH, scaler=None):
    df = df.copy()
    df.dropna(inplace=True)
    
    print(f"[DEBUG] Input data shape after dropna: {df.shape}")
    
    if len(df) < sequence_length:
        raise ValueError(f"Insufficient data points: {len(df)}. Need at least {sequence_length}")
    
    if scaler:
        scaled_features = scaler.transform(df[feature_cols])
    else:
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df[feature_cols])
    
    X = []
    for i in range(sequence_length, len(scaled_features) + 1):
        X.append(scaled_features[i-sequence_length:i])
    
    X = np.array(X)
    print(f"[DEBUG] Prepared X shape: {X.shape}")
    
    return X, scaler
