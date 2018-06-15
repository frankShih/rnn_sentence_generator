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
        self.hidden_size = hidden_dim
        self.output_size = output_dim
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, embedding_dim).to(device)  # input size, hidden1 size
        # print(self.encoder.weight)
        # nn.init.xavier_uniform_(self.encoder.weight, gain=5/3)

        # set batch_first=true,  input/output tensors are provided as (batch, seq, feature)
        if rnn_model == 'LSTM':
            # set batch_first=true,  input/output tensors are provided as (batch, seq, feature)
            self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_size, num_layers=self.n_layers,
                               dropout=dropout, batch_first=True, bidirectional=False).to(device)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=self.hidden_size, num_layers=self.n_layers,
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
        input = self.encoder(input.t())  # inputs (timesteps, batch, features) = (seq_length, batch_size, embedding_dim)
        # print(input.shape)
        # pass hidden param help speed up training
        rnn_out, _ = self.rnn(input)  # Each timestep outputs 1 hidden_state
        # Combined in lstm_out = (seq_length, batch_size, hidden_dim)
        # print(rnn_out.shape)
        # ht = rnn_out[-1]                # ht = last hidden state = (batch_size, hidden_dim)
        # Use the last hidden state to predict the following character
        # print(ht.shape)
        out = self.decoder(rnn_out[-1, :, :])  # Fully-connected layer, predict (batch_size, input_dim)
        # print(out.shape)
        return out


# Bidirectional recurrent neural network (many-to-many)
class BiRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, rnn_model='LSTM', n_layers=1, dropout=0.2):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.encoder = nn.Embedding(input_size, embedding_dim)

        if rnn_model == 'LSTM':
            # set batch_first=true,  input/output tensors are provided as (batch, seq, feature)
            self.rnn = nn.LSTM(embedding_dim, self.hidden_size, num_layers=self.num_layers, batch_first=True,
                               bidirectional=True, dropout=dropout).to(device)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU(embedding_dim, self.hidden_size, num_layers=self.num_layers, batch_first=True,
                              bidirectional=True, dropout=dropout).to(device)
        else:
            raise LookupError('Currently only support LSTM and GRU, or trying self-defined RNN-cell')

        self.fc = nn.Linear(hidden_size * 2, output_size).to(device)  # 2 for bidirection

    def forward(self, input):
        input = self.encoder(input.t()).to(device)  # LSTM takes 3D inputs (timesteps, batch, features)
        # print(input.shape)
        h0 = torch.zeros(self.num_layers * 2, input.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, input.size(0), self.hidden_size).to(device)
        # print(h0.shape)

        # Forward propagate GRU
        out, _ = self.rnn(input, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[-1, :, :])
        return out


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # if not hidden:
        #     print(input_seqs.size())
        #     hidden = nn.Parameter(torch.zeros(1, input_seqs.size()[1], self.hidden_size), requires_grad=True)
        # Note: we run this all at once (over multiple batches of multiple sequences)
        print("forwarding ~~~")
        # print(input_seqs.size())
        embedded = self.embedding(input_seqs)
        print(embedded.size(), input_lengths)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        print(packed, hidden)
        # for name, param in self.named_parameters():
        #     # if param.requires_grad:
        #     print(name, param.data.size())
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden


USE_CUDA = torch.cuda.is_available()


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len))  # B x S

        if USE_CUDA:  attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        # softmax on timestamp dim
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.mm(encoder_output.t())
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            # print(hidden.size(), energy.size())
            energy = hidden.mm(energy.t())
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.mm(energy.t())
        else:
            print("undefined scoring strategy!!!")
            return
        return energy


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        # TODO: FIX BATCHING

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1)  # S=1 x B x N
        word_embedded = self.dropout(word_embedded)

        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N (batch matmul)
        context = context.transpose(0, 1)  # 1 x B x N

        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        output, hidden = self.gru(rnn_input, last_hidden)

        # Final output layer
        output = output.squeeze(0)  # B x N
        output = F.log_softmax(self.out(torch.cat((output, context), 1)))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        if attn_model != 'none':  # Choose attention model
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size)  # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights
