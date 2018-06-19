import torch
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy
import masked_cross_entropy
import time
import math
import re
import os
import jieba
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import model
import torch.nn as nn
import sconce

from dataUtil import *

MIN_LENGTH = 3
MAX_LENGTH = 25

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Lang:
    def __init__(self):
        # self.trimmed = False
        self.word2index = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3}
        self.word2count = {}
        self.index2word = {0: '<SOS>', 1: '<EOS>', 2: '<PAD>', 3: '<UNK>'}
        self.n_words = 4 # Count default tokens

    # def index_words(self, sentence):
    #     for word in sentence.split(' '):
    #         self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, pairs, min_count=3):
        # if self.trimmed:
        #     print("already trimmed, exiting")
        #     return
        # self.trimmed = True

        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words)+4, len(self.word2index), (len(keep_words)+4) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {'SOS': 0, 'EOS': 1, 'PAD': 2, 'UNK': 3}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS', 2: 'PAD', 3: 'UNK'}
        self.n_words = 4 # Count default tokens

        for word in keep_words: self.index_word(word)

        keep_pairs = []
        for pair in pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep = True

            for word in input_sentence.split(' '):
                if word not in self.word2index:
                    keep = False
                    break

            for word in output_sentence.split(' '):
                if word not in self.word2index:
                    keep = False
                    break

            # Remove if pair doesn't match input and output conditions
            if keep:    keep_pairs.append(pair)

        print("Trimmed from %d pairs to %d, %.4f of total" % (len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
        return keep_pairs


    # Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    def normalize_string(self, s):
        s = unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([,.!?])", r" \1 ", s)
        s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    def read_langs(self, path, mode):
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

        # for i in cut_sentence_new(raw_text):    print(i)
        if mode == 'char':
            word_list = list(raw_text)
        elif mode == 'word':
            word_list = [w for w in jieba.cut(raw_text, cut_all=False)]
        else:
            print('Non-supported mode for training. Exiting...')
            os._exit(1)


        # Map char to int / int to char
        for word in word_list:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1
        # print(self.word2count)
        # Prepare training data, every <seq_length> sequence, predict 1 char after it
        pairs = []
        sentences = cut_sentence_new(raw_text)
        for ind in range(len(sentences)-1):
            pairs.append([sentences[ind], sentences[ind+1]])
        return self, pairs


    def filter_pairs(self, pairs):
        MIN_LENGTH = 3
        MAX_LENGTH = 25

        filtered_pairs = []
        for pair in pairs:
            if len(pair[0]) >= MIN_LENGTH and len(pair[0]) <= MAX_LENGTH \
                and len(pair[1]) >= MIN_LENGTH and len(pair[1]) <= MAX_LENGTH:
                    filtered_pairs.append(pair)
        return filtered_pairs


    def prepare_data(self, path, mode='word'):
        output_lang, pairs = self.read_langs(path, mode)
        print("Read %d sentence pairs" % len(pairs))

        # pairs = self.filter_pairs(pairs)
        # print("Filtered to %d pairs" % len(pairs))
        # for p in pairs:
        #     print(p)
        return output_lang, pairs

    # Return a list of indexes, one for each word in the sentence, plus EOS
    def indexes_from_sentence(self, sentence, mode='word'):
        if mode == 'char':
            return [self.word2index[word] for word in sentence] + [EOS_token]
        elif mode == 'word':
            return [self.word2index[word] for word in jieba.cut(sentence, cut_all=False)] + [EOS_token]
        else:
            print('Non-supported mode for preprocessing! Exiting...')
            os._exit(1)


    # Pad a with the PAD symbol
    def pad_seq(self, seq, max_length):
        seq += [PAD_token for i in range(max_length - len(seq))]
        return seq

    def random_batch(self, pairs, batch_size=5, mode='word'):
        input_seqs = []
        target_seqs = []

        # Choose random pairs
        for _ in range(batch_size):
            pair = random.choice(pairs)
            input_seqs.append(self.indexes_from_sentence(pair[0], mode))
            target_seqs.append(self.indexes_from_sentence(pair[1], mode))

        # Zip into pairs, sort by length (descending), unzip
        seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
        input_seqs, target_seqs = zip(*seq_pairs)

        # For input and target sequences, get array of lengths and pad with 0s to max length
        input_lengths = [len(s) for s in input_seqs]
        input_padded = [self.pad_seq(s, max(input_lengths)) for s in input_seqs]
        target_lengths = [len(s) for s in target_seqs]
        target_padded = [self.pad_seq(s, max(target_lengths)) for s in target_seqs]

        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1).to(device)
        target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1).to(device)

        # print(input_var, input_lengths, target_var, target_lengths)
        return input_var, input_lengths, target_var, target_lengths

def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size)).to(device)
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size)).to(device)

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t] # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy.masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
        target_lengths
    )
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data.item(), ec, dc

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def evaluate(lang, input_seq, max_length=MAX_LENGTH):
    input_lengths = [len(input_seq)]
    input_seqs = [lang.indexes_from_sentence(input_seq, mode='word')]
    input_batches = Variable(torch.LongTensor(input_seqs), requires_grad=False).transpose(0, 1).to(device)

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder

    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([SOS_token]), volatile=True).to(device) # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder

    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(lang.index2word[ni])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni])).to(device)

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]

def evaluate_randomly(lang):
    [input_sentence, target_sentence] = random.choice(pairs)
    evaluate_and_show_attention(lang, input_sentence, target_sentence)

def evaluate_and_show_attention(lang, input_sentence, target_sentence=None):
    output_words, attentions = evaluate(lang, input_sentence)
    output_sentence = ' '.join(output_words)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    print('<', output_sentence)

    # show_attention(input_sentence, output_words, attentions)

    # Show input, target, output text in visdom
    # win = 'evaluted (%s)' % hostname
    # text = '<p>&gt; %s</p><p>= %s</p><p>&lt; %s</p>' % (input_sentence, target_sentence, output_sentence)
    # vis.text(text, win=win, opts={'title': win})


if __name__ == '__main__':
    lang = Lang()
    lang, pairs = lang.prepare_data(path="C:\\Users\\han_shih.ASUS\\Documents\\story\\testing\\001.txt", mode='word')
    # lang.trim(pairs, min_count=2)

    small_batch_size = 3
    input_batches, input_lengths, target_batches, target_lengths = lang.random_batch(pairs, batch_size=small_batch_size, mode='word')
    print('input_batches', input_batches.size()) # (max_len x batch_size)
    print('target_batches', target_batches.size()) # (max_len x batch_size)
    small_hidden_size = 8
    small_n_layers = 2

    encoder_test = model.EncoderRNN(lang.n_words, small_hidden_size, small_n_layers).to(device)
    decoder_test = model.LuongAttnDecoderRNN('general', small_hidden_size, lang.n_words, small_n_layers).to(device)

    encoder_outputs, encoder_hidden = encoder_test(input_batches, input_lengths, None)
    print('encoder_outputs', encoder_outputs.size()) # max_len x batch_size x hidden_size
    print('encoder_hidden', encoder_hidden.size()) # n_layers * 2 x batch_size x hidden_size

    max_target_length = max(target_lengths)

    # Prepare decoder input and outputs
    decoder_input = Variable(torch.LongTensor([SOS_token] * small_batch_size)).to(device)
    decoder_hidden = encoder_hidden[:decoder_test.n_layers].to(device) # Use last (forward) hidden state from encoder
    all_decoder_outputs = Variable(torch.zeros(max_target_length, small_batch_size, decoder_test.output_size)).to(device)


    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder_test(
            decoder_input, decoder_hidden, encoder_outputs
        )
        all_decoder_outputs[t] = decoder_output # Store this step's outputs
        decoder_input = target_batches[t] # Next input is current target

    # Test masked cross entropy loss
    loss = masked_cross_entropy.masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),
        target_batches.transpose(0, 1).contiguous(),
        target_lengths
    )
    print('loss', loss.data.item())


    # Configure models
    attn_model = 'dot'
    hidden_size = 10
    n_layers = 2
    dropout = 0.1
    # batch_size = 100
    batch_size = 1

    # Configure training/optimization
    clip = 50.0
    teacher_forcing_ratio = 0.5
    learning_rate = 0.001
    decoder_learning_ratio = 5.0
    n_epochs = 500
    epoch = 0
    plot_every = 20
    print_every = 1
    evaluate_every = 5

    # Initialize models
    encoder = model.EncoderRNN(lang.n_words, hidden_size, n_layers, dropout=dropout).to(device)
    decoder = model.LuongAttnDecoderRNN(attn_model, hidden_size, lang.n_words, n_layers, dropout=dropout).to(device)

    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    criterion = nn.CrossEntropyLoss()

    # job = sconce.Job('seq2seq-GENERATE', {
    #     'attn_model': attn_model,
    #     'n_layers': n_layers,
    #     'dropout': dropout,
    #     'hidden_size': hidden_size,
    #     'learning_rate': learning_rate,
    #     'clip': clip,
    #     'teacher_forcing_ratio': teacher_forcing_ratio,
    #     'decoder_learning_ratio': decoder_learning_ratio,
    # })
    # job.plot_every = plot_every
    # job.log_every = print_every

    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    print_loss_total = 0 # Reset every print_every
    plot_loss_total = 0 # Reset every plot_every


    # Begin!
    ecs = []
    dcs = []
    eca = 0
    dca = 0


    while epoch < n_epochs:
        epoch += 1

        # Get training data for this cycle
        input_batches, input_lengths, target_batches, target_lengths = lang.random_batch(pairs, batch_size=batch_size, mode='word')

        # Run the train function
        loss, ec, dc = train(
            input_batches, input_lengths, target_batches, target_lengths,
            encoder, decoder,
            encoder_optimizer, decoder_optimizer, criterion
        )

        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss
        eca += ec
        dca += dc

        # job.record(epoch, loss)

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (epoch=%d %d%%) avg_loss= %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            print(print_summary)

        if epoch % evaluate_every == 0:
            evaluate_randomly(lang)

        # if epoch % plot_every == 0:
        #     plot_loss_avg = plot_loss_total / plot_every
        #     plot_losses.append(plot_loss_avg)
        #     plot_loss_total = 0

        #     # TODO: Running average helper
        #     ecs.append(eca / plot_every)
        #     dcs.append(dca / plot_every)
        #     ecs_win = 'encoder grad (%s)' % hostname
        #     dcs_win = 'decoder grad (%s)' % hostname
        #     vis.line(np.array(ecs), win=ecs_win, opts={'title': ecs_win})
        #     vis.line(np.array(dcs), win=dcs_win, opts={'title': dcs_win})
        #     eca = 0
        #     dca = 0

