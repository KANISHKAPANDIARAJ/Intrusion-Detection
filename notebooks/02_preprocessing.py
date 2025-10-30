import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# === Paths ===
TRAIN_FILE = "data/KDDTrain+.txt"
TEST_FILE  = "data/KDDTest+.txt"
SEQ_LEN = 20

# === Load dataset ===
train_df = pd.read_csv(TRAIN_FILE, header=None)
test_df  = pd.read_csv(TEST_FILE, header=None)

# === Identify columns ===
label_col = train_df.shape[1] - 2       # last column is label
cat_cols = [1, 2, 3]                    # protocol_type, service, flag

# === Encode categorical features ===
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[c], test_df[c]], axis=0)
    le.fit(combined.astype(str))
    train_df[c] = le.transform(train_df[c].astype(str))
    test_df[c]  = test_df[c].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else 0)  # unknown -> 0
    encoders[c] = le
    joblib.dump(le, f"enc_col_{c}.joblib")

# === Encode labels for IDS ===
train_labels = train_df[label_col].astype(str)
test_labels  = test_df[label_col].astype(str)

# Detect normal class
normal_class_candidates = [lbl for lbl in train_labels.unique() if "normal" in lbl.lower()]
if not normal_class_candidates:
    raise ValueError("❌ 'normal' class not found in dataset labels!")
normal_class = normal_class_candidates[0]

# Binary encoding: 0 = normal, 1 = attack
y_train = np.where(train_labels == normal_class, 0, 1)
y_test  = np.where(test_labels == normal_class, 0, 1)  # unseen labels automatically treated as attacks

# === Ensure numeric features ===
for col in train_df.columns:
    if col not in cat_cols + [label_col]:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        test_df[col]  = pd.to_numeric(test_df[col], errors='coerce')

train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

# === Extract features ===
X_train = train_df.drop(columns=[label_col]).values
X_test  = test_df.drop(columns=[label_col]).values

# === Scale features ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# === Convert to sequences for LSTM ===
def create_sequences(X, y, seq_len=SEQ_LEN):
    sequences, labels = [], []
    for i in range(0, len(X) - seq_len + 1, seq_len):  # non-overlapping
        sequences.append(X[i:i+seq_len])
        labels.append(np.bincount(y[i:i+seq_len]).argmax())
    return np.array(sequences), np.array(labels)

X_train_seq, y_train_seq = create_sequences(X_train, y_train)
X_test_seq, y_test_seq   = create_sequences(X_test, y_test)

# === Save sequences ===
np.save("X_train_seq.npy", X_train_seq)
np.save("y_train_seq.npy", y_train_seq)
np.save("X_test_seq.npy", X_test_seq)
np.save("y_test_seq.npy", y_test_seq)

print("✅ Preprocessing done. Sequences saved.")
