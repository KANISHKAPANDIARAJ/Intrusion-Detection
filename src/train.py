import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import RandomOverSampler

from model_attn import CNN_BiLSTM_Attn_IDS

# ===============================
# 🔧 CONFIG
# ===============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

NUM_CLASSES = 2   # 0: Normal, 1:Attack
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 10
LR = 5e-4
PATIENCE = 5

MODEL_SAVE_PATH = "best_ids_model.pth"

# ===============================
# 📦 LOAD DATA
# ===============================
X = np.load("X_train_seq.npy")
y = np.load("y_train_seq.npy").astype(int)

X_test = np.load("X_test_seq.npy")
y_test = np.load("y_test_seq.npy").astype(int)

print("X:", X.shape, "y:", y.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)

# ===============================
# ✂ TRAIN / VAL SPLIT
# ===============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# ===============================
# ⚖ OVERSAMPLING (MULTICLASS SAFE)
# ===============================
n_seq, seq_len, feat_dim = X_train.shape
X_flat = X_train.reshape(n_seq, -1)

ros = RandomOverSampler(random_state=SEED)
X_res, y_res = ros.fit_resample(X_flat, y_train)

X_train = X_res.reshape(-1, seq_len, feat_dim)
y_train = y_res

print("After oversampling:", np.bincount(y_train))

# ===============================
# 🔄 DATALOADERS
# ===============================
def make_loader(X, y, shuffle=False):
    return DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)   # 🔥 LONG, NOT FLOAT
        ),
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val, y_val)
test_loader  = make_loader(X_test, y_test)

# ===============================
# 🧠 MODEL
# ===============================
model = CNN_BiLSTM_Attn_IDS(
    input_dim=feat_dim,
    seq_len=seq_len,
    num_classes=NUM_CLASSES
).to(DEVICE)

print("Model initialized on", DEVICE)

# ===============================
# 🎯 LOSS + OPTIMIZER
# ===============================
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.3, patience=2
)

# ===============================
# 🚀 TRAINING LOOP
# ===============================
best_f1 = 0.0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits.squeeze(), yb)


        loss.backward()
        optimizer.step()

        train_loss += loss.item() * Xb.size(0)

    train_loss /= len(train_loader.dataset)

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0.0
    preds_all, trues_all = [], []

    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = model(Xb)
            loss = criterion(logits.squeeze(), yb)
            val_loss += loss.item() * Xb.size(0)

            preds = torch.argmax(logits, dim=1)
            preds_all.extend(preds.cpu().numpy())
            trues_all.extend(yb.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_f1 = f1_score(trues_all, preds_all, average="macro")

    print(f"Epoch {epoch}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val F1: {val_f1:.4f}")

    scheduler.step(val_loss)

    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("🔥 Best model saved")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹ Early stopping")
            break

# ===============================
# 🧪 FINAL TEST
# ===============================
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

preds_all, trues_all = [], []

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(DEVICE)
        logits = model(Xb)
        probs = torch.sigmoid(logits.squeeze())
        preds = (probs > 0.5).long()

        preds_all.extend(preds.cpu().numpy())
        trues_all.extend(yb.numpy())

print("\n=== TEST RESULTS ===")
print(classification_report(trues_all, preds_all, digits=4))
print("Confusion Matrix:\n", confusion_matrix(trues_all, preds_all))
