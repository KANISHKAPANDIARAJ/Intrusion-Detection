# src/ensemble.py
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from model import CNN_BiLSTM_IDS

DEVICE = torch.device("cpu")
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Paths (adjust if you moved files)
MODEL_PTH = os.path.join(os.path.dirname(__file__), "best_ids_model.pth")
X_TRAIN_NPY = os.path.join(os.path.dirname(__file__), "..", "X_train_seq.npy")
Y_TRAIN_NPY = os.path.join(os.path.dirname(__file__), "..", "y_train_seq.npy")
X_TEST_NPY  = os.path.join(os.path.dirname(__file__), "..", "X_test_seq.npy")
Y_TEST_NPY  = os.path.join(os.path.dirname(__file__), "..", "y_test_seq.npy")

# Load data
X_all = np.load(X_TRAIN_NPY)   # (n_samples, seq_len, feat)
y_all = np.load(Y_TRAIN_NPY)
X_test = np.load(X_TEST_NPY)
y_test = np.load(Y_TEST_NPY)

print("Loaded shapes -- train(all):", X_all.shape, y_all.shape, "test:", X_test.shape, y_test.shape)

# We'll split the original training set into train/val for the ensemble training
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_all)
print("Ensemble split -- train:", X_train.shape, "val:", X_val.shape)

# Load CNN-BiLSTM model and weights
# instantiate model with input_dim and seq_len
input_dim = X_all.shape[2]
seq_len = X_all.shape[1]
net = CNN_BiLSTM_IDS(input_dim=input_dim, seq_len=seq_len).to(DEVICE)

# Load weights
state_path = MODEL_PTH
if not os.path.exists(state_path):
    raise FileNotFoundError(f"Model weights not found at {state_path}")
net.load_state_dict(torch.load(state_path, map_location=DEVICE))
net.eval()
print("Loaded CNN-BiLSTM weights.")

# Helper: compute "embedding" for sequences using the trained net
# We will extract the penultimate representation (the vector before final fc)
def compute_embeddings(model, X_np, batch=128):
    """
    Return numpy array of embeddings of shape (n_samples, embed_dim).
    We'll replicate forward pass up to 'last' and then fc1+bn_fc+relu to get dense embedding.
    """
    embeddings = []
    with torch.no_grad():
        n = X_np.shape[0]
        for start in range(0, n, batch):
            end = min(n, start + batch)
            xb = torch.tensor(X_np[start:end], dtype=torch.float32).to(DEVICE)  # (b, seq_len, feat)
            # replicate internals: conv -> pool -> lstm -> last -> fc1 + bn_fc + relu
            x = xb.permute(0, 2, 1)  # (b, feat, seq)
            # conv1
            x = model.act(model.bn1(model.conv1(x)))
            x = model.pool(x)
            x = model.act(model.bn2(model.conv2(x)))
            x = model.act(model.bn3(model.conv3(x)))
            x = model.dropout_cnn(x)
            # prepare for LSTM
            x = x.permute(0, 2, 1)  # (b, new_seq_len, channels)
            lstm_out, _ = model.lstm(x)
            last = lstm_out[:, -1, :]          # (b, 2*hidden)
            last = model.dropout_lstm(last)
            # fc1 -> bn_fc -> relu
            with torch.no_grad():
                h = model.act(model.bn_fc(model.fc1(last)))  # (b, 128)
            embeddings.append(h.cpu().numpy())
    embeddings = np.vstack(embeddings)
    return embeddings  # shape (n_samples, embed_dim=128)

# Handcrafted features from sequences: per-feature stats over time (mean, std, min, max)
def handcrafted_stats(X_np):
    # X_np: (n, seq_len, feat)
    means = X_np.mean(axis=1)   # (n, feat)
    stds  = X_np.std(axis=1)
    mins  = X_np.min(axis=1)
    maxs  = X_np.max(axis=1)
    # concatenate -> (n, feat*4)
    return np.concatenate([means, stds, mins, maxs], axis=1)

# Compute embeddings & stats for train/val/test
print("Computing embeddings and features for train ...")
emb_train = compute_embeddings(net, X_train)
feat_train = handcrafted_stats(X_train)
X_train_ens = np.hstack([emb_train, feat_train])

print("Computing embeddings and features for val ...")
emb_val = compute_embeddings(net, X_val)
feat_val = handcrafted_stats(X_val)
X_val_ens = np.hstack([emb_val, feat_val])

print("Computing embeddings and features for test ...")
emb_test = compute_embeddings(net, X_test)
feat_test = handcrafted_stats(X_test)
X_test_ens = np.hstack([emb_test, feat_test])

print("Shapes for ensemble inputs:", X_train_ens.shape, X_val_ens.shape, X_test_ens.shape)

# Train a RandomForest on these combined features
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train_ens, y_train)

# Validate
val_preds = rf.predict(X_val_ens)
print("\n=== Validation report (Ensemble) ===")
print(classification_report(y_val, val_preds, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_val, val_preds))

# Test
test_preds = rf.predict(X_test_ens)
print("\n=== Test report (Ensemble) ===")
print(classification_report(y_test, test_preds, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, test_preds))

# Save RF model
rf_path = os.path.join(os.path.dirname(__file__), "rf_on_embeddings.joblib")
joblib.dump(rf, rf_path)
print(f"\nSaved RandomForest ensemble to: {rf_path}")
