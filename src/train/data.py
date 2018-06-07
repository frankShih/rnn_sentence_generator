import numpy as np
import torch
import jieba
import os


def parse_corpus(path, mode, seq_length=50):
    '''
        Parse raw corpus text into input-output pairs
          input: sequence of words, 
          output: 1 word after input sequence
    '''

    # Read text            
    if os.path.isdir(path):
        print("loading from path...")
        raw_text = ""
        for filename in os.listdir(path):
            print(path+filename)
            with open(os.path.join(path, filename), encoding='UTF-8', mode='r') as f:
                raw_text += f.read().strip(' \t\n\r　')
    elif os.path.isfile(path):
        print("loading from file...")
        with open(path, encoding='UTF-8', mode='r') as f:
            raw_text = f.read().strip(' \t\n\r　')
    else:  
        print("Invalid file path. Exiting..." )
        os._exit(1)

    if mode == 'char':
        word_list = list(raw_text)
    elif mode == 'word':
        word_list = [w for w in jieba.cut(raw_text, cut_all=False)]
    else:
        print('Non-supported mode for training. Exiting...')
        os._exit(1)

    # Get unique characters
    words = list(set(word_list))

    # Map char to int / int to char
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
    return (dataX, dataY, word_to_int, int_to_word, words)



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
    if uchar >= u'/u4e00' and uchar<=u'/u9fa5':
            return True
    else:
            return False


def is_number(uchar):
    if uchar >= u'/u0030' and uchar<=u'/u0039':
            return True
    else:
            return False


def is_alphabet(uchar):
    if (uchar >= u'/u0041' and uchar<=u'/u005a') or (uchar >= u'/u0061' and uchar<=u'/u007a'):
            return True
    else:
            return False
