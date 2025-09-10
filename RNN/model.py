import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, n_layers=1):
    super(RNN, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers

    self.encoder = nn.Embedding(input_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=0.2 ,batch_first=True,
                      bidirectional=False)
    self.decoder = nn.Linear(hidden_size, output_size)

  def forward(self, input, hidden):
    embedded = self.encoder(input)  # (batch, seq_len, embed_dim)
    output, hidden = self.gru(embedded, hidden)
    output = self.decoder(output[:, -1, :]) # to teleutaio vima tou seq

    return output, hidden

  def init_hidden(self, batch_size):
    return torch.zeros(self.n_layers, batch_size, self.hidden_size)
