import numpy as np
import torch
import jieba
import os
import re
import random
from torch.autograd import Variable


def parse_corpus(path, mode, batch_size, seq_length=50):
    '''
        Parse raw corpus text into input-output pairs
          input: sequence of words,
          output: 1 word after input sequence
    '''

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

    # for i in cut_sentence_new(raw_text):
    #     print(i)
    if mode == 'char':
        word_list = list(raw_text)
    elif mode == 'word':
        word_list = [w for w in jieba.cut(raw_text, cut_all=False)]
    else:
        print('Non-supported mode for data preprocessing. Exiting...')
        os._exit(1)

    # Get unique characters
    words = list(set(word_list))

    # Map char to int / int to char
    # word_to_int = {'SOS': 0, 'EOS': 1, 'PAD': 2, 'UNK': 3}
    # int_to_word = {0: 'SOS', 1: 'EOS', 2: 'PAD', 3: 'UNK'}
    word_to_int = dict((c, i) for i, c in enumerate(words))
    int_to_word = dict((i, c) for i, c in enumerate(words))

    # Prepare training data, every <seq_length> sequence, predict 1 char after it
    dataX = [] # N x seq_length
    dataY = [] # N x 1
    for i in range(0, len(word_list) - seq_length):
        seq_in = word_list[i:i + seq_length]
        seq_out = word_list[i + seq_length]
        dataX.append([word_to_int[w] for w in seq_in])
        dataY.append(word_to_int[seq_out])
    print(len(word_list), len(words))
    # print(len(dataX), len(dataX[0]))
    # print(len(dataY), dataY[0])

    # For simplicity, discard trailing data not fitting into batch_size
    n_patterns = len(dataY)
    n_patterns = n_patterns - n_patterns % batch_size
    X = dataX[:n_patterns]
    Y = dataY[:n_patterns]

    # Parse X
    X = np.array(X)
    _, seq_length = X.shape
    X = X.reshape(-1, batch_size, seq_length)
    X = torch.LongTensor(X)

    # Parse Y
    Y = np.array(Y)
    Y = Y.reshape(-1, batch_size)
    Y = torch.LongTensor(Y)

    return (list(zip(X, Y)), dataX, dataY, word_to_int, int_to_word, words)

def format_data(dataX, dataY, n_classes, batch_size=64):
    '''Parse into minibatches, return Tensors'''

    # For simplicity, discard trailing data not fitting into batch_size
    n_patterns = len(dataY)
    n_patterns = n_patterns - n_patterns % batch_size
    X = dataX[:n_patterns]
    Y = dataY[:n_patterns]

    # Parse X
    X = np.array(X)
    _, seq_length = X.shape
    X = X.reshape(-1, batch_size, seq_length)

    X = torch.LongTensor(X)

    # Parse Y
    Y = np.array(Y)
    Y = Y.reshape(-1, batch_size)

    Y = torch.LongTensor(Y)

    return list(zip(X, Y))

def is_chinese(uchar):
    return uchar >= u'/u4e00' and uchar<=u'/u9fa5'

def is_number(uchar):
    return uchar >= u'/u0030' and uchar<=u'/u0039'

def is_alphabet(uchar):
    return (uchar >= u'/u0041' and uchar<=u'/u005a') or (uchar >= u'/u0061' and uchar<=u'/u007a')

def cut_sentence_new(words):
    # words = (words).decode('utf8')
    start = 0
    i = 0
    sents = []
    closure_flag = False
    punt_list = '.!?:;~。！？：；～』”」'
    closure_list = "「“『』”」"
    for word in words:
        if word in closure_list:    closure_flag = not(closure_flag)
        if word in punt_list and token not in punt_list and not(closure_flag):
            #check if next word is puncuation or not
            sents.append(words[start:i+1])
            start = i+1
            i += 1
        else:
            i += 1
            token = list(words[start:i+2]).pop()
            # get next word
    if start < len(words):
        sents.append(words[start:])
    return sents


PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
USE_CUDA = torch.cuda.is_available()
class Lang:
    def __init__(self):
        # self.trimmed = False
        self.word2index = {'SOS': 0, 'EOS': 1, 'PAD': 2, 'UNK': 3}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS', 2: 'PAD', 3: 'UNK'}
        self.n_words = 4 # Count default tokens

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

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

    '''
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
    '''
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


    def prepare_data(self, path, mode='char'):
        output_lang, pairs = self.read_langs(path, mode)
        print("Read %d sentence pairs" % len(pairs))

        # pairs = self.filter_pairs(pairs)
        # print("Filtered to %d pairs" % len(pairs))
        # for p in pairs:
        #     print(p)
        return output_lang, pairs

    # Return a list of indexes, one for each word in the sentence, plus EOS
    def indexes_from_sentence(self, sentence, mode='char'):
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

    def random_batch(self, pairs, batch_size=5, mode='char'):
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
        input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
        target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

        if USE_CUDA:
            input_var = input_var.cuda()
            target_var = target_var.cuda()
        print(input_var, input_lengths, target_var, target_lengths)
        return input_var, input_lengths, target_var, target_lengths



if __name__ == '__main__':
    lang = Lang()
    lang, pairs = lang.prepare_data(path="C:\\Users\\han_shih.ASUS\\Documents\\story\\testing\\001.txt", mode='word')
    # lang.trim(pairs, min_count=2)
    input_batches, input_lengths, target_batches, target_lengths = lang.random_batch(pairs, mode='word')
    print('input_batches', input_batches.size()) # (max_len x batch_size)
    print('target_batches', target_batches.size()) # (max_len x batch_size)