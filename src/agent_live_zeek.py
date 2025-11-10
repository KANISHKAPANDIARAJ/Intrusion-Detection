# src/agent_live_zeek.py
import os
import json
import joblib
import numpy as np
import torch
import time
from pathlib import Path
from model_attn import CNN_BiLSTM_Attn_IDS  # replace if your model class is in model.py

# --- CONFIG ---
BASE = Path(__file__).resolve().parent
SEQ_FILE = BASE / "X_zeek_seq.npy"
KEYS_FILE = BASE / "flow_keys.npy"
MODEL_FILE = BASE / "best_ids_model.pth"
OUT_ALERTS = BASE / "alerts.json"
THRESH = 0.5               # probability threshold for attack
BATCH_SIZE = 64            # inference batch size
DEVICE = torch.device("cpu")

# --- safety checks
if not SEQ_FILE.exists():
    raise FileNotFoundError(f"{SEQ_FILE} not found. Run zeek_to_features.py first.")
if not KEYS_FILE.exists():
    raise FileNotFoundError(f"{KEYS_FILE} not found. Run zeek_to_features.py first.")
if not MODEL_FILE.exists():
    raise FileNotFoundError(f"{MODEL_FILE} not found. Train or copy best_ids_model.pth into {BASE}")

# --- load sequences and flow keys
X_seqs = np.load(SEQ_FILE, allow_pickle=False)
flow_keys = joblib.load(KEYS_FILE)   # saved as Python list of lists
print(f"Loaded sequences: {X_seqs.shape}, flow_keys: {len(flow_keys)}")

# --- instantiate model (make sure input_dim & seq_len match)
seq_len = X_seqs.shape[1]
input_dim = X_seqs.shape[2]
print("Model input dim:", input_dim, "seq_len:", seq_len)

# model_attn returns logits shape (batch,1) — adjust if different
model = CNN_BiLSTM_Attn_IDS(input_dim=input_dim, seq_len=seq_len, num_classes=1).to(DEVICE)
state = torch.load(MODEL_FILE, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# --- inference helper
def infer_and_alert(X_arr, keys_arr, threshold=THRESH, batch_size=BATCH_SIZE):
    alerts = []
    n = len(X_arr)
    i = 0
    while i < n:
        batch = X_arr[i:i+batch_size]
        with torch.no_grad():
            xt = torch.tensor(batch, dtype=torch.float32).to(DEVICE)
            logits = model(xt)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        for j, p in enumerate(probs):
            seq_idx = i + j
            is_attack = int(p > threshold)
            if is_attack:
                alert = {
                    "sequence_index": int(seq_idx),
                    "probability": float(p),
                    "flows": keys_arr[seq_idx]  # list of 5-tuples for flows in this sequence
                }
                alerts.append(alert)
        i += batch_size
    return alerts

# --- run inference
start = time.time()
alerts = infer_and_alert(X_seqs, flow_keys)
dur = time.time() - start
print(f"Inference done in {dur:.2f}s. Total sequences: {len(X_seqs)}. Alerts: {len(alerts)}")

# --- save alerts.json
with open(OUT_ALERTS, "w", encoding="utf-8") as f:
    json.dump({"generated_at": time.time(), "alerts": alerts}, f, indent=2)

print(f"Saved {len(alerts)} alerts to {OUT_ALERTS}")

# --- print summary for quick inspection
for a in alerts[:20]:
    print(f"ALERT seq#{a['sequence_index']} prob={a['probability']:.3f} flows={a['flows']}")
