import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, n_vocab, embedding_dim, hidden_dim, dropout=0.2):
        super(Net, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(n_vocab, embedding_dim)  # input size, hidden1 size        
        # print(self.embeddings.weight)

        nn.init.xavier_uniform_(self.embeddings.weight, gain=math.sqrt(2.0))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout)  # hidden1 size, hidden2 size
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                # nn.init.xavier_normal(param)
                nn.init.xavier_uniform_(param, gain=math.sqrt(2.0))

        self.hidden2out = nn.Linear(hidden_dim, n_vocab)  # hidden2 size, output size

    def forward(self, seq_in):
        embeddings = self.embeddings(seq_in.t()) # LSTM takes 3D inputs (timesteps, batch, features)
        embeddings = F.relu(embeddings)          #                    = (seq_length, batch_size, embedding_dim)

        lstm_out, _ = self.lstm(embeddings)      # Each timestep outputs 1 hidden_state
        lstm_out = F.relu(lstm_out)              # Combined in lstm_out = (seq_length, batch_size, hidden_dim) 

        ht = lstm_out[-1]                        # ht = last hidden state = (batch_size, hidden_dim)
                                                 # Use the last hidden state to predict the following character        

        out = self.hidden2out(ht)                # Fully-connected layer, predict (batch_size, n_vocab)

        return out
