import torch
import numpy as np
from model import CNN_LSTM_IDS
from sklearn.metrics import classification_report
import os
DEVICE = torch.device("cpu")

# === Load Data ===
X_test = np.load("X_test_seq.npy")
y_test = np.load("y_test_seq.npy")

input_dim = X_test.shape[2]
seq_len = X_test.shape[1]
num_classes = len(set(y_test))

# === Load Model ===
model = CNN_LSTM_IDS(input_dim=input_dim, seq_len=seq_len, num_classes=num_classes).to(DEVICE)
model_path = os.path.join(os.path.dirname(__file__), "best_ids_model.pth")
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

X_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

with torch.no_grad():
    out = model(X_tensor)
preds = torch.argmax(out, axis=1).cpu().numpy()

print("📊 Evaluation Report:")
print(classification_report(y_test, preds, zero_division=0))
