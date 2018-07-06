import numpy as np
import torch
import jieba
import os
import re
import random
from torch.autograd import Variable
import pickle



def load_data(path, seq_length, batch_size, mode):
    dataX, dataY, target_to_int, int_to_target, targets = parse_corpus(path, mode, seq_length=seq_length)
    data = format_data(dataX, dataY, batch_size=batch_size)
    return data, dataX, dataY, target_to_int, int_to_target, targets


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def parse_corpus(path, mode, seq_length=50):
    """
        Parse raw corpus text into input-output pairs
          input: sequence of words,
          output: 1 word after input sequence
    """

    rm = re.compile(r"\s+", re.MULTILINE)

    # Read text
    raw_text = ""
    if os.path.isdir(path):
        print("loading from path...")
        for filename in os.listdir(path):
            print(path + filename)
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
        print("Invalid file path. Exiting...")
        os._exit(1)

    word_list = []
    if mode == 'char':
        for sent in cut_sentence_new(raw_text):
            temp = ['<SOS>'] + list(sent) + ['<EOS>']
            word_list.extend(temp)
    elif mode == 'word':
        for sent in cut_sentence_new(raw_text):
            temp = ['<SOS>'] + [w for w in jieba.cut(sent, cut_all=False)] + ['<EOS>']
            word_list.extend(temp)
    else:
        print('Non-supported mode for data pre-processing. Exiting...')
        os._exit(1)

    # Get unique characters
    words = ['<SOS>', '<EOS>', '<PAD>', '<UNK>']
    words.extend(word_list)
    words = list(set(words))
    print(len(word_list), len(words))
    # print(words)
    word_to_int = dict((c, i) for i, c in enumerate(words))
    int_to_word = dict((i, c) for i, c in enumerate(words))

    # Prepare training data, every <seq_length> sequence, predict 1 char after it
    dataX = []  # N x seq_length
    dataY = []  # N x 1
    for i in range(0, len(word_list) - seq_length):
        seq_in = word_list[i:i + seq_length]
        seq_out = word_list[i + seq_length]
        dataX.append([word_to_int[w] for w in seq_in])
        dataY.append(word_to_int[seq_out])

    return dataX, dataY, word_to_int, int_to_word, words


def format_data(dataX, dataY, batch_size=64):
    """Parse into mini-batches, return Tensors"""
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
    return uchar >= u'/u4e00' and uchar <= u'/u9fa5'


def is_number(uchar):
    return uchar >= u'/u0030' and uchar <= u'/u0039'


def is_alphabet(uchar):
    return (uchar >= u'/u0041' and uchar <= u'/u005a') or (uchar >= u'/u0061' and uchar <= u'/u007a')


def cut_sentence_new(words):
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

