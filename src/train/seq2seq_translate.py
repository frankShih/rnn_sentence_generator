# -*- coding: utf-8 -*

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os
import jieba

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = False

SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3
MAX_LENGTH = 200

class Lang:
    def __init__(self):
        self.val2ind = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3}
        self.val_counter = {}
        self.ind2val = {0: '<SOS>', 1: '<EOS>', 2: '<PAD>', 3: '<UNK>'}
        self.num_val = 4 # Count default tokens

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.val2ind:
            self.val2ind[word] = self.num_val
            self.val_counter[word] = 1
            self.ind2val[self.num_val] = word
            self.num_val += 1
        else:
            self.val_counter[word] += 1


def read_langs(path, mode):
    lang = Lang()
    rm = re.compile(r"\s+", re.MULTILINE)

    # Read text
    raw_text = ""
    if os.path.isdir(path):
        print("loading from path...")
        for filename in os.listdir(path):
            print(path+filename)
            if os.path.isdir(os.path.join(path, filename)): continue
            with open(os.path.join(path, filename), encoding='UTF-8', mode='r') as f:
                temp = rm.sub("", f.read())
                temp = re.sub(r"[『“]", r"「", temp)
                temp = re.sub(r"[』”]", r"」", temp)
                raw_text += temp
    elif os.path.isfile(path):
        print("loading from file...")
        with open(path, encoding='UTF-8', mode='r') as f:
            temp = rm.sub("", f.read())
            temp = re.sub(r"[『“]", r"「", temp)
            temp = re.sub(r"[』”]", r"」", temp)
            raw_text += temp
    else:
        print("Invalid file path. Exiting..." )
        os._exit(1)

    # for i in cut_sentence(raw_text):    print(i)
    if mode == 'char':
        word_list = list(raw_text)
    elif mode == 'word':
        word_list = [w for w in jieba.cut(raw_text, cut_all=False)]
    else:
        print('Non-supported mode for training. Exiting...')
        os._exit(1)

    # Map char to int / int to char
    for word in word_list:
        if word not in lang.val2ind:
            lang.val2ind[word] = lang.num_val
            lang.val_counter[word] = 1
            lang.ind2val[lang.num_val] = word
            lang.num_val += 1
        else:
            lang.val_counter[word] += 1
    # print(self.val_counter)
    # Prepare training data, every <seq_length> sequence, predict 1 char after it
    pairs = []
    sentences = cut_sentence(raw_text)
    for ind in range(len(sentences)-1):
        pairs.append([sentences[ind], sentences[ind+1]])
    return lang, pairs


def cut_sentence(words):
    # words = (words).decode('utf8')
    start = 0
    i = 0
    sents = []
    closure_flag = False
    punt_list = '.!?:;~。！？：；～』”」'
    closure_list = "「“『』”」"
    for word in words:
        if word in closure_list:    closure_flag = not (closure_flag)
        if word in punt_list and token not in punt_list and not (closure_flag):
            # check if next word is punctuation or not
            sents.append(words[start:i + 1])
            start = i + 1
            i += 1
        else:
            i += 1
            token = list(words[start:i + 2]).pop()
            # get next word
    if start < len(words):
        sents.append(words[start:])
    return sents

def cut_sentence(words):
    # words = (words).decode('utf8')
    start = 0
    i = 0
    sents = []
    closure_flag = False
    punt_list = '.!?:;~。！？：；～』”」'
    closure_list = "「“『』”」"
    for word in words:
        if word in closure_list:    closure_flag = not (closure_flag)
        if word in punt_list and token not in punt_list and not (closure_flag):
            # check if next word is punctuation or not
            sents.append(words[start:i + 1])
            start = i + 1
            i += 1
        else:
            i += 1
            token = list(words[start:i + 2]).pop()
            # get next word
    if start < len(words):
        sents.append(words[start:])
    return sents



def prepare_data(path, mode='word'):
    output_lang, pairs = read_langs(path, mode)
    print("Read {} sentence pairs, total {} words.".format(len(pairs), output_lang.num_val))

    # pairs = self.filter_pairs(pairs)
    # print("Filtered to %d pairs" % len(pairs))
    # for p in pairs:
    #     print(p)
    return output_lang, pairs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.01, batch_size=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        #         print(input)
        #         embedded = self.embedding(input).view(1, 1, -1)
        #         print(embedded.shape)
        #         output, hidden = self.gru(embedded, hidden)
        #         print(output.shape)

        embedded = self.embedding(input_seqs)
        #         print(embedded.size(), input_lengths)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        output, hidden = self.gru(packed, hidden)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # Sum bidirectional outputs
        #         print(output.shape, hidden.shape)
        return output, hidden

    def initHidden(self):
        return torch.zeros(2, self.batch_size, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

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
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)).to(device)  # B x S

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

    def forward(self, input_seq, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        #         print("decoder forwarding~~~")
        #         print(input_seq.shape, last_hidden.shape, encoder_outputs.shape)
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        #         print(batch_size, embedded.shape)
        embedded = self.embedding_dropout(embedded)
        #         print(batch_size, embedded.shape)
        embedded = embedded.view(1, batch_size, self.hidden_size)  # S=1 x B x N
        # print(last_context.shape, last_context.unsqueeze(0).shape, embedded.shape)
        # embedded = torch.cat((embedded, last_context.unsqueeze(0)), 2)
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
        return output, context, hidden, attn_weights


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, mode='word'):
    if mode == 'char':
        temp = [lang.val2ind[letter] for letter in sentence]
        temp.append(EOS_token)
        return temp
    elif mode == 'word':
        temp = [lang.val2ind[word] for word in jieba.cut(sentence, cut_all=False)]
        temp.append(EOS_token)
        return temp
    else:
        print('Non-supported mode for preprocessing! Exiting...')
        os._exit(1)


def tensorFromSentence(lang, sentence, mode='word'):
    indexes = indexes_from_sentence(lang, sentence, mode)
    # indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, mode='word'):
    input_tensor = tensorFromSentence(output_lang, pair[0], mode)
    target_tensor = tensorFromSentence(output_lang, pair[1], mode)
    return (input_tensor, target_tensor)


# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def random_batch(lang, pairs, batch_size=5, mode='word'):
    input_seqs = []
    target_seqs = []

    for _ in range(batch_size):         # Choose random pairs
        pair = random.choice(pairs)
        input_seqs.append(indexes_from_sentence(lang, pair[0], mode))
        target_seqs.append(indexes_from_sentence(lang, pair[1], mode))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1).to(device)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1).to(device)

    # print(input_var, input_lengths, target_var, target_lengths)
    return input_var, input_lengths, target_var, target_lengths


import torch
from torch.nn import functional
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand).to(device)
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    length = Variable(torch.LongTensor(length)).to(device)

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes), softmax on timestamp dim
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0
    batch_size = len(input_lengths)

    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, encoder_hidden)
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size)).to(device)
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size)).to(device)
#     print(encoder_hidden.shape, decoder_hidden.shape)
    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size)).to(device)

    teacher_forcing_ratio = 0.5
    use_teacher_forcing = True #if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for t in range(max(target_lengths)):
            decoder_output, decoder_context, decoder_hidden, decoder_attn \
                = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)

            all_decoder_outputs[t] = decoder_output
            decoder_input = target_batches[t] # Next input is current target

        # loss += self.criterion(decoder_output, target_variable[di])
        loss = masked_cross_entropy(    # calculate loss for whole seq. at once
            all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
            target_batches.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths)
    else:
        # Without teacher forcing: use its own predictions as the next input
        for t in range(max(target_lengths)):
#             print(decoder_hidden.shape)
            decoder_output, decoder_context, decoder_hidden, decoder_attn \
                = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)

            all_decoder_outputs[t] = decoder_output
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().to(device)  # detach from history as input
            if decoder_input.item() == EOS_token:   break

        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
            target_batches.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths)

    loss.backward()     # backpropagation
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1e-0/2)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1e-0/2)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data.item()


import pickle

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, out_e, out_d, print_every=1000, learning_rate=0.005, batch_size=10):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        input_batches, input_lengths, target_batches, target_lengths \
            = random_batch(output_lang, pairs, batch_size=batch_size, mode='word')

        loss = train(input_batches, input_lengths, target_batches, target_lengths,
                     encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

            save_pickle((output_lang, pairs), args.output_c)
            encoder.eval()
            torch.save(encoder, out_e)
            decoder.eval()
            torch.save(decoder, out_d)

            plot_loss_avg = plot_loss_total / print_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    save_pickle((output_lang, pairs), args.output_c)
    encoder.eval()
    torch.save(encoder, out_e)
    decoder.eval()
    torch.save(decoder, out_d)


def beamSearchInfer(sample, top_k, decoder):
    alpha = 0.5
    # for current sample, search k possible situations
    samples = []
    decoder_input = Variable(torch.LongTensor([[sample[0][-1]]])).to(device)
    sequence, pre_scores, fin_scores, avg_scores, decoder_context, decoder_hidden, decoder_attention, encoder_outputs = sample
    decoder_output, decoder_context, decoder_hidden, decoder_attention \
        = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)

    # choose topk
    topk = decoder_output.data.topk(top_k)
    for k in range(top_k):
        topk_prob = topk[0][0][k]
        topk_index = int(topk[1][0][k])
        pre_scores += topk_prob
        fin_scores = pre_scores - (k - 1 ) * alpha
        samples.append([sequence+[topk_index], pre_scores, fin_scores, avg_scores, decoder_context, decoder_hidden, decoder_attention, encoder_outputs])
    return samples


import pandas as pd

def evaluate(encoder, decoder, sentence, temperature=0.1, max_length=MAX_LENGTH, use_beam_search=False):
    with torch.no_grad():
        input_tensor = tensorFromSentence(output_lang, sentence)
        input_length = input_tensor.size()[0]

        encoder_hidden = torch.zeros(2, 1, encoder.hidden_size, device=device)#encoder.initHidden(), 2 for bi-direction
        encoder_outputs, encoder_hidden = encoder(input_tensor, [input_length], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        decoder_context = Variable(torch.zeros(1, decoder.hidden_size)).to(device)

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        if use_beam_search:
            top_k = 2
            decoder_output, decoder_context, decoder_hidden, decoder_attn \
                = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            topk = decoder_output.data.topk(top_k)
            samples = [[]] * top_k
            dead_k = 0
            final_samples = []
            for index in range(top_k): # init topK samples for first round
                topk_prob = topk[0][0][index]
                topk_index = int(topk[1][0][index])
                samples[index] = [[topk_index], topk_prob, 0, 0, decoder_context, decoder_hidden, decoder_attn, encoder_outputs]

            for _ in range(max_length):
                tmp = []
                for index in range(len(samples)):
                    tmp.extend(beamSearchInfer(samples[index], top_k, decoder))
                # samples = []

                # select topk
                df = pd.DataFrame(tmp)
                df.columns = ['sequence', 'pre_socres', 'fin_scores', "avg_scores", "decoder_context", "decoder_hidden", "decoder_attention", "encoder_outputs"]
                sequence_len = df.sequence.apply(lambda x:len(x))
                df['avg_scores'] = df['fin_scores'] / sequence_len  # instead of greedy search, choose the one with highest AVG_SCORE
                df = df.sort_values('avg_scores', ascending=False).reset_index().drop(['index'], axis=1)
                df = df[:(top_k - dead_k)]
                for index in range(len(df)):
                    group = df.ix[index]    # deprecated, replace it with loc/iloc
                    if group.tolist()[0][-1] == EOS_token:
                        final_samples.append(group.tolist())
                        df = df.drop([index], axis=0)
                        dead_k += 1
                        print("drop {}, {}".format(group.tolist()[0], dead_k))
                samples = df.values.tolist()
                if len(samples) == 0:   break

            if len(final_samples) < top_k:  # unable to get enough samples that predict EOS
                final_samples.extend(samples[:(top_k - dead_k)])

            final_samples = sorted(final_samples,key=lambda x:(x[3]), reverse=True)

            return [output_lang.ind2val[ind] for ind in final_samples[0][0]], final_samples[0][3]


        else:
            for di in range(max_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attn \
                    = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                decoder_attentions[di,:decoder_attn.size(2)] += decoder_attn.squeeze(0).squeeze(0).cpu().data
                # topv, topi = decoder_output.data.topk(1)  #return k largest element, along with index

                pred = decoder_output.data.view(-1).div(temperature).exp()
                topi = torch.multinomial(pred, 1)[0]
                decoder_input = Variable(torch.LongTensor([topi])).to(device)#topi.squeeze().detach()

                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(output_lang.ind2val[topi.item()])

            return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10, use_beam_search=False):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], use_beam_search=use_beam_search)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


import argparse
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train seq2seq model')
    parser.add_argument('--input-c', type=str, default="corpus.bin", metavar='F',
                        help='corpus bin (default: corpus.bin)')
    parser.add_argument('--input-e', type=str, default="encoder.bin", metavar='F',
                        help='model bin (default: encoder.bin)')
    parser.add_argument('--input-d', type=str, default="decoder.bin", metavar='F',
                        help='model bin (default: decoder.bin)')
    parser.add_argument('--action', type=str, default="train", metavar='F',
                        help='train / predict')
    parser.add_argument('--beam-search', dest='use_beam_search', action='store_true',
                        help='use beam search straregy for prediction')
    parser.add_argument('corpus', type=str, metavar='F',
                        help='training corpus file')
    parser.add_argument('--seq-length', type=int, default=50, metavar='N',
                        help='input sequence length (default: 50)')
    parser.add_argument('--batch-size', type=int, default=5, metavar='N',
                        help='training batch size (default: 1)')
    parser.add_argument('--hidden-dim', type=int, default=16, metavar='N',
                        help='hidden state dimension (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--dropout', type=float, default=0.001, metavar='DR',
                        help='dropout rate (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='number of batches to wait before logging status (default: 100)')
    parser.add_argument('--mode', type=str, default='char', metavar='N',
                        help='char/word level to train model (default: char)')
    parser.add_argument('--output-e', type=str, default='encoder.bin', metavar='F',
                        help='output model file')
    parser.add_argument('--output-d', type=str, default='decoder.bin', metavar='F',
                        help='output model file')
    parser.add_argument('--output-c', type=str, default='corpus.bin', metavar='F',
                        help='output corpus related file (mappings & vocab)')
    parser.set_defaults(use_beam_search=False)
    args = parser.parse_args()

    # Load mappings & vocabularies
    print("####################################################")
    print("# loading... " + os.path.abspath(args.input_c))
    print("# loading... " + os.path.abspath(args.input_e))
    print("# loading... " + os.path.abspath(args.input_d))
    print("####################################################")
    print()

    if args.action == 'train':
        if os.path.exists(args.input_c) and os.path.exists(args.input_e) and os.path.exists(args.input_d):
            comfirm = input("Train with existing model and corpus? [Y/n]")
            if comfirm == "y" or comfirm == "Y":
                output_lang, pairs = load_pickle(args.input_c)
                encoder = torch.load(args.input_e)
                attn_decoder = torch.load(args.input_d)
            else:
                print("Re-train the model ...")
                output_lang, pairs = prepare_data(args.corpus, args.mode)
                encoder = EncoderRNN(output_lang.num_val, args.hidden_dim, 1, batch_size=args.batch_size).to(device)
                attn_decoder = LuongAttnDecoderRNN('general', args.hidden_dim, output_lang.num_val, 1).to(device)

        else:
            print("Train a new model ...")
            output_lang, pairs = prepare_data(args.corpus, args.mode)
            print(len(output_lang.val2ind))
            encoder = EncoderRNN(output_lang.num_val, args.hidden_dim, 1, batch_size=args.batch_size).to(device)
            attn_decoder = LuongAttnDecoderRNN('general', args.hidden_dim, output_lang.num_val, 1).to(device)

        print(random.choice(pairs))
        trainIters(encoder, attn_decoder, args.epochs, args.output_e, args.output_d, print_every=args.log_interval, learning_rate=args.lr, batch_size=args.batch_size)

    else:
        output_lang, pairs = load_pickle(args.input_c)
        encoder = torch.load(args.input_e)
        attn_decoder = torch.load(args.input_d)
        evaluateRandomly(encoder, attn_decoder, n=5, use_beam_search=args.use_beam_search)



