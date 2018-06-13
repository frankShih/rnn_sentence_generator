import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, rnn_model='LSTM', n_layers=1, dropout=0.2):
        super(Net, self).__init__()
        self.input_size = input_dim
        self.embedding_size = embedding_dim
        self.hidden_size = hidden_dim
        self.output_size = output_dim
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size).to(device)  # input size, hidden1 size
        # print(self.encoder.weight)
        # nn.init.xavier_uniform_(self.encoder.weight, gain=5/3)

        if rnn_model == 'LSTM':
            # set batch_first=true,  input/output tensors are provided as (batch, seq, feature)
            self.rnn = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                                dropout=dropout, batch_first=True, bidirectional=False).to(device)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                                dropout=dropout, batch_first=True, bidirectional=False).to(device)
        else:
            raise LookupError('Currently only support LSTM and GRU, or trying self-defined RNN-cell')

        # for name, param in self.lstm.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #     elif 'weight' in name:
        #         # nn.init.xavier_normal(param)
        #         nn.init.xavier_uniform_(param, gain=5/3)

        self.decoder = nn.Linear(self.hidden_size, self.output_size).to(device)  # hidden2 size, output size


    def forward(self, input):
        # print('forwarding~~~')
        input = self.encoder(input.t()) # LSTM takes 3D inputs (timesteps, batch, features)
                                        #   = (seq_length, batch_size, embedding_dim)
        # print(input.shape)
        # pass hidden param help speed up training
        rnn_out, _ = self.rnn(input)      # Each timestep outputs 1 hidden_state
                                          # Combined in lstm_out = (seq_length, batch_size, hidden_dim)
        # print(rnn_out.shape)
        # ht = rnn_out[-1]                # ht = last hidden state = (batch_size, hidden_dim)
                                          # Use the last hidden state to predict the following character
        # print(ht.shape)
        out = self.decoder(rnn_out[-1, :, :]).to(device)   # Fully-connected layer, predict (batch_size, input_dim)
        # print(out.shape)
        return out



# Bidirectional recurrent neural network (many-to-many)
class BiRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, n_layers=1, dropout=0.2):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.encoder = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=dropout).to(device)
        self.fc = nn.Linear(hidden_size*2, output_size)  # 2 for bidirection

    def forward(self, input):
        # Set initial states
        input = self.encoder(input.t()) # LSTM takes 3D inputs (timesteps, batch, features)
        # print(input.shape)
        h0 = torch.zeros(self.num_layers*2, input.size(0), self.hidden_size).to(device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, input.size(0), self.hidden_size).to(device)
        # print(h0.shape)
        # Forward propagate LSTM
        out, _ = self.lstm(input, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[-1, :, :])
        return out
