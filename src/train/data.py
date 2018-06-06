import numpy as np
import torch
import jieba
import os

def parse_corpus(path, seq_length=50):
    '''
        Parse raw corpus text into input-output pairs
          input: sequence of characters, 
          output: 1 character after input sequence
    '''

    if os.path.isdir(path):
        print("loading from path...")
        raw_text = ""
        for filename in os.listdir(path):
            print(path+filename)
            with open(os.path.join(path, filename), encoding='UTF-8', mode='r') as f:
                raw_text += f.read().replace('\n', '')
    elif os.path.isfile(path):
        print("loading from file...")
        with open(path, encoding='UTF-8', mode='r') as f:
            raw_text = f.read().replace('\n', '')
    else:  
        print("Invalid file path. Exiting..." )
        os._exit(1) 
    # Read text
    
    # Get unique characters
    chars = sorted(list(set(raw_text)))

    # Map char to int / int to char
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    
    # Prepare training data, for every <seq_length> chars, predict 1 char after the sequence
    n_chars = len(raw_text)
    dataX = [] # N x seq_length
    dataY = [] # N x 1
    for i in range(0, n_chars - seq_length):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    print(len(chars), n_chars)
    return (dataX, dataY, char_to_int, int_to_char, chars)


def parse_corpus_word(path, seq_length=50):
    '''
        Parse raw corpus text into input-output pairs
          input: sequence of words, 
          output: 1 word after input sequence
    '''

    # Read text            
    word_list = []
    if os.path.isdir(path):
        print("loading from path...")
        raw_text = ""
        for filename in os.listdir(path):
            print(path+filename)
            with open(os.path.join(path, filename), encoding='UTF-8', mode='r') as f:
                raw_text += f.read().replace('\n', '')
    elif os.path.isfile(path):
        print("loading from file...")
        with open(path, encoding='UTF-8', mode='r') as f:
            raw_text = f.read().replace('\n', '')
    else:  
        print("Invalid file path. Exiting..." )
        os._exit(1)
    
    word_list = [w for w in jieba.cut(raw_text, cut_all=False)]
    # Get unique characters
    words = list(set(word_list))


    # Map char to int / int to char
    word_to_int = dict((c, i) for i, c in enumerate(words))
    int_to_word = dict((i, c) for i, c in enumerate(words))
    
    # Prepare training data, for every <seq_length> chars, predict 1 char after the sequence
    n_words = len(word_list)
    dataX = [] # N x seq_length
    dataY = [] # N x 1
    for i in range(0, n_words - seq_length):
        seq_in = word_list[i:i + seq_length]
        seq_out = word_list[i + seq_length]
        dataX.append([word_to_int[w] for w in seq_in])
        dataY.append(word_to_int[seq_out])
    print(len(word_list), len(words))
    # print(dataX[0], dataY[0])
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


if __name__ == '__main__':
    parse_corpus_word('C:/Users/han_shih.ASUS/Documents/projects/resurrecting-the-dead-chinese/corpus/glin_utf8.txt', seq_length=50)