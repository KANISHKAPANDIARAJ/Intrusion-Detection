import os
import torch
import numpy as np
import pandas as pd
import joblib
import sys
from model import CNN_BiLSTM_IDS


DEVICE = torch.device("cpu")
sys.path.append(os.path.dirname(__file__))

# === Config ===
BASE_DIR = os.path.dirname(__file__)
SEQ_LEN = 20
CAT_COLS = [1, 2, 3]  # protocol_type, service, flag
LABEL_ENCODERS = {c: os.path.join(BASE_DIR, f"enc_col_{c}.joblib") for c in CAT_COLS}
SCALER_FILE = os.path.join(BASE_DIR, "scaler.pkl")
MODEL_FILE = os.path.join(BASE_DIR, "best_ids_model.pth")

# === Load preprocessing objects ===
encoders = {c: joblib.load(f) for c, f in LABEL_ENCODERS.items()}
scaler = joblib.load(SCALER_FILE)

# === Helper: preprocess a single batch of network data ===
def preprocess(df):
    for c in CAT_COLS:
        df[c] = encoders[c].transform(df[c].astype(str))
    for col in df.columns:
        if col not in CAT_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(0, inplace=True)
    X = scaler.transform(df.values)
    sequences = []
    for i in range(0, len(X) - SEQ_LEN + 1, SEQ_LEN):
        sequences.append(X[i:i+SEQ_LEN])
    return np.array(sequences)

# === Example usage ===
if __name__ == "__main__":
    # Load new network traffic data (CSV format)
    df_new = pd.read_csv("data/new_traffic.csv", header=None)
    
    # Preprocess to get input_dim
    X_seq = preprocess(df_new)
    if len(X_seq) == 0:
        print("❌ Not enough data to form sequences.")
        exit()
    
    input_dim = X_seq.shape[2]
    seq_len = X_seq.shape[1]
    num_classes = 2

    # === Load trained model ===
    model = CNN_BiLSTM_IDS(input_dim=input_dim, seq_len=seq_len, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()

    # === Prediction function ===
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        out = model(X_tensor)
    preds = torch.argmax(out, axis=1).cpu().numpy()
    
    for i, p in enumerate(preds):
        label = "Normal" if p == 0 else "Attack"
        print(f"Sequence {i+1}: {label}")
