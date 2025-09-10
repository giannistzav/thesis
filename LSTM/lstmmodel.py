import numpy as np
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, total_words, embedding_dim=64, hidden_dim=100,num_layers=2,dropout_rate=0.5):
        super().__init__()
        self.embedding = nn.Embedding(total_words, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, total_words)  # *2 για bidirectional
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x