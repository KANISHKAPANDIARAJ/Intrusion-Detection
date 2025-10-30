import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model_attn import CNN_BiLSTM_Attn_IDS  # ✅ your actual model file

# -------------------------
# Load test data
# -------------------------
X_test = np.load("X_test_seq.npy")
y_test = np.load("y_test_seq.npy")

# Convert to tensors
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# -------------------------
# Load model
# -------------------------
DEVICE = torch.device("cpu")
INPUT_DIM = X_test.shape[2]
SEQ_LEN = X_test.shape[1]
NUM_CLASSES = 2

model = CNN_BiLSTM_Attn_IDS(input_dim=INPUT_DIM, seq_len=SEQ_LEN, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("best_ids_model.pth", map_location=DEVICE))
model.eval()

# -------------------------
# Prediction
# -------------------------
with torch.no_grad():
    logits = model(X_test.to(DEVICE))
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs > 0.5).astype(int)

# -------------------------
# Metrics
# -------------------------
y_true = y_test.cpu().numpy().astype(int)
print("\n=== Final Test Report ===")
print(classification_report(y_true, preds, digits=4))

cm = confusion_matrix(y_true, preds)
print("\nConfusion Matrix:\n", cm)

# -------------------------
# Plot Confusion Matrix
# -------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - CNN + BiLSTM + Attention Model')
plt.show()
