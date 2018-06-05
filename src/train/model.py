import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, rnn_model='LSTM', n_layers=1, dropout=0.2):
        super(Net, self).__init__()
        self.input_size = input_dim
        self.embedding_size = embedding_dim
        self.hidden_size = hidden_dim
        self.output_size = output_dim
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)  # input size, hidden1 size        
        # print(self.encoder.weight)
        # nn.init.xavier_uniform_(self.encoder.weight, gain=5/3)
        
        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                                dropout=dropout, batch_first=True, bidirectional=False)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                                dropout=dropout, batch_first=True, bidirectional=False)
        else:
            raise LookupError('Currently only support LSTM and GRU, or trying self-defined RNN-cell')
        
        # for name, param in self.lstm.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #     elif 'weight' in name:
        #         # nn.init.xavier_normal(param)
        #         nn.init.xavier_uniform_(param, gain=5/3)

        self.decoder = nn.Linear(self.hidden_size, self.output_size)  # hidden2 size, output size


    def forward(self, input):
        input = self.encoder(input.t()) # LSTM takes 3D inputs (timesteps, batch, features)
        # encoder = F.relu(encoder)          #                    = (seq_length, batch_size, embedding_dim)
        # pass hidden param help speed up training
        rnn_out, _ = self.rnn(input)      # Each timestep outputs 1 hidden_state
        # rnn_out = F.relu(rnn_out)              # Combined in lstm_out = (seq_length, batch_size, hidden_dim) 
        
        ht = rnn_out[-1]                        # ht = last hidden state = (batch_size, hidden_dim)
                                                 # Use the last hidden state to predict the following character        

        out = self.decoder(ht)                # Fully-connected layer, predict (batch_size, input_dim)

        return out
    
