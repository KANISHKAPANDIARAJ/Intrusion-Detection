# src/model.py
import torch
import torch.nn as nn

class CNN_BiLSTM_IDS(nn.Module):
    """
    CNN feature extractor -> Bidirectional LSTM -> Dense -> single-logit output for binary IDS.
    Returns raw logits (use BCEWithLogitsLoss).
    """
    def __init__(self, input_dim, seq_len):
        super(CNN_BiLSTM_IDS, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

        # Convolutional extractor (Conv1d over time with channels=input_dim)
        # We set conv to treat input_dim as channels, kernel over seq length.
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.act = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool1d(kernel_size=2)  # reduces time dimension
        self.dropout_cnn = nn.Dropout(0.3)

        # LSTM input size must equal number of channels after conv (128)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.dropout_lstm = nn.Dropout(0.4)

        # Fully connected head
        self.fc1 = nn.Linear(128 * 2, 128)  # bidirectional -> 2*hidden
        self.bn_fc = nn.BatchNorm1d(128)
        self.fc_out = nn.Linear(128, 1)  # single logit

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)            # -> (batch, input_dim, seq_len)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool(x)                  # (batch, 64, seq_len/2)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.dropout_cnn(x)           # (batch, 128, new_seq_len)

        x = x.permute(0, 2, 1)            # -> (batch, new_seq_len, channels=128)
        # LSTM expects (batch, seq, features)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]         # take last timestep (batch, 2*hidden)
        last = self.dropout_lstm(last)

        h = self.act(self.bn_fc(self.fc1(last)))
        out = self.fc_out(h)              # raw logits
        return out.squeeze(1)             # (batch,) logits
