# src/train.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from model_attn import CNN_BiLSTM_Attn_IDS

# --- Config ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

NUM_CLASSES = 1   # output = 1 neuron (binary)
DEVICE = torch.device("cpu")   # use CUDA if available: torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 64
EPOCHS = 6
LR = 5e-4
PATIENCE = 6   # early stopping patience on val F1
MODEL_SAVE = "best_ids_model.pth"

# --- Load data ---
X = np.load("X_train_seq.npy")
y = np.load("y_train_seq.npy")
X_test_full = np.load("X_test_seq.npy")
y_test_full = np.load("y_test_seq.npy")

print("Loaded shapes:", X.shape, y.shape, X_test_full.shape, y_test_full.shape)

# --- Split train/val ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print("Train/Val shapes:", X_train.shape, X_val.shape)

# --- Oversample training data ---
n_seq, seq_len, feat = X_train.shape
X_flat = X_train.reshape(n_seq, -1)
ros = RandomOverSampler(random_state=SEED)
X_res, y_res = ros.fit_resample(X_flat, y_train)
X_res = X_res.reshape(-1, seq_len, feat)
X_train, y_train = X_res, y_res
print("After oversample:", X_train.shape, np.bincount(y_train))

# --- DataLoaders ---
train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                         torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                       torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
test_ds = TensorDataset(torch.tensor(X_test_full, dtype=torch.float32),
                        torch.tensor(y_test_full, dtype=torch.float32).unsqueeze(1))

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

# --- Model ---
input_dim = X_train.shape[2]
seq_len = X_train.shape[1]
model = CNN_BiLSTM_Attn_IDS(input_dim=input_dim, seq_len=seq_len, num_classes=NUM_CLASSES).to(DEVICE)
print("Model created. Input dim:", input_dim, "Seq len:", seq_len)

# --- Define Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return loss.mean()

# --- Compute pos_weight (to balance minority class) ---
pos = np.sum(y_train == 1)
neg = np.sum(y_train == 0)
pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(DEVICE)
print(f"pos_weight: {pos_weight.item():.4f}")

# Choose loss function (FocalLoss or weighted BCE)
criterion = FocalLoss(alpha=1.0, gamma=2.0)
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# --- Optimizer & scheduler ---
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=3)

# --- Training ---
best_f1 = -1.0
patience_cnt = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * Xb.size(0)

    train_loss /= len(train_loader.dataset)

    # --- Validation ---
    model.eval()
    val_losses = 0.0
    preds_list, trues_list = [], []

    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = model(Xb)
            loss = criterion(logits, yb)
            val_losses += loss.item() * Xb.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.4).astype(int)   # lowered threshold for better recall
            preds_list.extend(preds.tolist())
            trues_list.extend(yb.cpu().numpy().astype(int).tolist())

    val_loss = val_losses / len(val_loader.dataset)
    val_f1 = f1_score(trues_list, preds_list, average='binary', zero_division=0)
    print(f"Epoch {epoch}/{EPOCHS}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Val F1: {val_f1:.4f}")

    scheduler.step(val_loss)

    if val_f1 > best_f1 + 1e-5:
        best_f1 = val_f1
        patience_cnt = 0
        torch.save(model.state_dict(), MODEL_SAVE)
        print(f"✅ Saved best model (val F1={best_f1:.4f})")
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print("⏱ Early stopping triggered.")
            break

# --- Final test evaluation ---
model.load_state_dict(torch.load(MODEL_SAVE, map_location=DEVICE))
model.eval()
preds_list, trues_list = [], []

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(DEVICE)
        logits = model(Xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.4).astype(int)  # same threshold used during val
        preds_list.extend(preds.tolist())
        trues_list.extend(yb.cpu().numpy().astype(int).tolist())

print("\n=== Test Classification Report ===")
print(classification_report(trues_list, preds_list, digits=4))
print("Confusion matrix:\n", confusion_matrix(trues_list, preds_list))
