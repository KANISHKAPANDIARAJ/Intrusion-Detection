import torch
import torch.nn as nn

class CNN_BiLSTM_Attn_IDS(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes):
        super().__init__()

        # --- CNN ---
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.dropout_cnn = nn.Dropout(0.3)

        # --- BiLSTM ---
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.dropout_lstm = nn.Dropout(0.4)

        # --- Attention ---
        self.attn = nn.Linear(256, 1)

        # --- Classifier ---
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)  # 🔥 MULTICLASS

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout_cnn(x)

        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout_lstm(lstm_out)

        # Attention
        attn_weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)

        x = torch.relu(self.fc1(context))
        return self.fc2(x)   # logits
